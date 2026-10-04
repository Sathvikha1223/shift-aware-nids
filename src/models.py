from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightgbm import LGBMClassifier
from sklearn.neural_network import MLPClassifier


class LightGBMModel:
    def __init__(self, seed: int = 42, **kwargs):
        self.seed = seed
        self.feature_names_ = None
        self.params = {
            "n_estimators": kwargs.get("n_estimators", 300),
            "learning_rate": kwargs.get("learning_rate", 0.05),
            "num_leaves": kwargs.get("num_leaves", 31),
            "subsample": kwargs.get("subsample", 0.8),
            "colsample_bytree": kwargs.get("colsample_bytree", 0.8),
            "random_state": seed,
            "n_jobs": -1,
            "verbose": -1,
        }
        self.model = None

    def fit(self, X, y, **kwargs):
        self.model = LGBMClassifier(**self.params)
        safe_kwargs = dict(kwargs)
        self.feature_names_ = safe_kwargs.pop("feature_names", None)
        safe_kwargs.pop("domains", None)
        safe_kwargs.pop("X_target", None)
        X_fit = self._as_frame(X)
        self.model.fit(X_fit, y, **safe_kwargs)
        return self

    def _as_frame(self, X):
        if isinstance(X, pd.DataFrame):
            return X
        arr = np.asarray(X, dtype=np.float32)
        if self.feature_names_ is not None:
            return pd.DataFrame(arr, columns=self.feature_names_)
        return arr

    def predict_proba(self, X):
        if self.model is None:
            raise ValueError("Model must be fitted before predict_proba().")
        return self.model.predict_proba(self._as_frame(X))


class MLPModel:
    def __init__(self, seed: int = 42, hidden_layer_sizes=(128, 64), max_iter=300):
        self.seed = seed
        self.hidden_layer_sizes = hidden_layer_sizes
        self.max_iter = max_iter
        self.model = None

    def fit(self, X, y, **kwargs):
        self.model = MLPClassifier(
            hidden_layer_sizes=self.hidden_layer_sizes,
            max_iter=self.max_iter,
            early_stopping=True,
            random_state=self.seed,
        )
        self.model.fit(X, y)
        return self

    def predict_proba(self, X):
        if self.model is None:
            raise ValueError("Model must be fitted before predict_proba().")
        return self.model.predict_proba(X)


class CORALModel:
    """Transductive CORAL mapping of unlabeled target features into source space."""

    def __init__(self, seed: int = 42, hidden_layer_sizes=(128, 64), max_iter=300):
        self.seed = seed
        self.base = MLPModel(seed=seed, hidden_layer_sizes=hidden_layer_sizes, max_iter=max_iter)
        self.source_mean_ = None
        self.source_cov_ = None
        self.target_mean_ = None
        self.target_cov_ = None

    def _cov(self, X):
        X = np.asarray(X, dtype=np.float32)
        mu = X.mean(axis=0)
        centered = X - mu
        cov = centered.T @ centered / max(1, len(X) - 1)
        return mu, cov + 1e-3 * np.eye(X.shape[1])

    def fit(self, X, y, domains=None, X_target=None, **kwargs):
        X = np.asarray(X, dtype=np.float32)
        self.source_mean_, self.source_cov_ = self._cov(X)
        if X_target is not None:
            self.target_mean_, self.target_cov_ = self._cov(np.asarray(X_target, dtype=np.float32))
        self.base.fit(X, y)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float32)
        if self.source_cov_ is None:
            return X
        target_cov = self.target_cov_ if self.target_cov_ is not None else self.source_cov_
        evals_s, evecs_s = np.linalg.eigh(self.source_cov_)
        sqrt_s = evecs_s @ np.diag(np.clip(evals_s, 1e-3, None) ** 0.5) @ evecs_s.T
        evals_t, evecs_t = np.linalg.eigh(target_cov)
        invsqrt_t = evecs_t @ np.diag(np.clip(evals_t, 1e-3, None) ** -0.5) @ evecs_t.T
        A = sqrt_s @ invsqrt_t
        mu = self.target_mean_ if self.target_mean_ is not None else self.source_mean_
        return (X - mu) @ A.T + self.source_mean_

    def predict_proba(self, X):
        return self.base.predict_proba(self.transform(X))

    def predict_source_proba(self, X):
        return self.base.predict_proba(X)


class GroupDROModel:
    """A source-domain reweighting approximation of GroupDRO using a standard MLP base."""

    def __init__(self, seed: int = 42, hidden_layer_sizes=(128, 64), max_iter=300):
        self.seed = seed
        self.base = MLPModel(seed=seed, hidden_layer_sizes=hidden_layer_sizes, max_iter=max_iter)
        self.sample_weight_ = None

    def fit(self, X, y, domains=None, **kwargs):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)
        if domains is None:
            self.sample_weight_ = np.ones(len(y), dtype=np.float32)
            train_X, train_y = X, y
        else:
            domains = np.asarray(domains)
            counts = np.bincount(domains.astype(int))
            domain_weight = {k: 1.0 / max(1, counts[k]) for k in np.unique(domains)}
            weights = np.array([domain_weight.get(int(d), 1.0) for d in domains], dtype=np.float32)
            weights /= weights.mean()
            self.sample_weight_ = weights
            repeats = np.clip(np.round(weights * 10.0), 1, 20).astype(int)
            idx = np.repeat(np.arange(len(X)), repeats)
            train_X = X[idx]
            train_y = y[idx]
        self.base.model = MLPClassifier(
            hidden_layer_sizes=(128, 64),
            max_iter=self.base.max_iter,
            early_stopping=True,
            random_state=self.seed,
        )
        self.base.model.fit(train_X, train_y)
        return self

    def predict_proba(self, X):
        return self.base.model.predict_proba(X)


class _GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, features, coefficient):
        ctx.coefficient = coefficient
        return features.view_as(features)

    @staticmethod
    def backward(ctx, gradient):
        return -ctx.coefficient * gradient, None


class GradientReversalLayer(nn.Module):
    def __init__(self, coefficient: float = 1.0):
        super().__init__()
        if coefficient < 0:
            raise ValueError("Gradient reversal coefficient must be non-negative.")
        self.coefficient = float(coefficient)

    def forward(self, features):
        return _GradientReversalFunction.apply(features, self.coefficient)


class DANNModel:
    """Domain-Adversarial Neural Network trained with gradient reversal."""

    class _FeatureExtractor(nn.Module):
        def __init__(self, in_dim, hid=64):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, hid),
                nn.ReLU(),
                nn.Linear(hid, hid),
                nn.ReLU(),
            )

        def forward(self, x):
            return self.net(x)

    class _LabelHead(nn.Module):
        def __init__(self, in_dim):
            super().__init__()
            self.net = nn.Linear(in_dim, 1)

        def forward(self, x):
            return self.net(x)

    class _DomainHead(nn.Module):
        def __init__(self, in_dim, n_domains):
            super().__init__()
            self.net = nn.Linear(in_dim, n_domains)

        def forward(self, x):
            return self.net(x)

    def __init__(
        self,
        seed: int = 42,
        hidden_dim: int = 64,
        max_epochs: int = 25,
        lr: float = 1e-3,
        grl_lambda: float = 1.0,
    ):
        self.seed = seed
        self.hidden_dim = hidden_dim
        self.max_epochs = max_epochs
        self.lr = lr
        self.grl_lambda = float(grl_lambda)
        self.model = None
        self.n_domains_ = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_extractor_ = None
        self.label_classifier_ = None
        self.domain_classifier_ = None
        self.gradient_reversal_ = GradientReversalLayer(self.grl_lambda)
        torch.manual_seed(seed)
        np.random.seed(seed)

    def fit(self, X, y, domains=None, X_target=None, **kwargs):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        domains = np.asarray(domains if domains is not None else np.zeros(len(X), dtype=int))
        if len(X) != len(y) or len(X) != len(domains):
            raise ValueError("X, y, and domains must contain the same number of rows.")
        _, domains = np.unique(domains, return_inverse=True)
        self.n_domains_ = int(np.unique(domains).size)
        feat = self._FeatureExtractor(X.shape[1], self.hidden_dim).to(self.device)
        label_head = self._LabelHead(self.hidden_dim).to(self.device)
        domain_head = self._DomainHead(self.hidden_dim, self.n_domains_).to(self.device)
        reversal = self.gradient_reversal_.to(self.device)
        opt = torch.optim.Adam(
            list(feat.parameters()) + list(label_head.parameters()) + list(domain_head.parameters()),
            lr=self.lr,
        )

        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        y_t = torch.tensor(y, dtype=torch.float32, device=self.device).view(-1, 1)
        d_t = torch.tensor(domains, dtype=torch.long, device=self.device)

        for _ in range(self.max_epochs):
            feat_out = feat(X_t)
            label_logits = label_head(feat_out)
            domain_logits = domain_head(reversal(feat_out))
            clf_loss = F.binary_cross_entropy_with_logits(label_logits, y_t)
            domain_loss = F.cross_entropy(domain_logits, d_t)
            loss = clf_loss + domain_loss
            opt.zero_grad()
            loss.backward()
            opt.step()

        self.feature_extractor_ = feat
        self.label_classifier_ = label_head
        self.domain_classifier_ = domain_head
        self.model = (feat, label_head, domain_head)
        return self

    def predict_proba(self, X):
        if self.model is None:
            raise ValueError("DANNModel is not fitted.")
        feat, label_head, _ = self.model
        X = torch.tensor(np.asarray(X, dtype=np.float32), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            pred = torch.sigmoid(label_head(feat(X))).cpu().numpy()[:, 0]
        return np.column_stack([1.0 - pred, pred])


class TreeMLPEnsembleModel:
    """Average of a tree classifier and a small MLP."""

    def __init__(self, seed: int = 42, tree_kwargs=None, mlp_kwargs=None):
        self.seed = seed
        self.tree = LightGBMModel(seed=seed, **(tree_kwargs or {}))
        self.mlp = MLPModel(seed=seed, **(mlp_kwargs or {}))

    def fit(self, X, y, **kwargs):
        self.tree.fit(X, y, **{k: v for k, v in kwargs.items() if k not in {"domains", "X_target"}})
        self.mlp.fit(X, y)
        return self

    def predict_proba(self, X):
        p_tree = self.tree.predict_proba(X)
        p_mlp = self.mlp.predict_proba(X)
        return 0.5 * (p_tree + p_mlp)


MODEL_MAP = {
    "lightgbm": LightGBMModel,
    "mlp": MLPModel,
    "coral": CORALModel,
    "groupdro": GroupDROModel,
    "dann": DANNModel,
    "tree_mlp_ensemble": TreeMLPEnsembleModel,
}


def get_model(model_name: str, seed: int = 42, **kwargs):
    key = str(model_name).lower().strip()
    if key not in MODEL_MAP:
        raise ValueError(f"Unknown model: {model_name}. Available: {sorted(MODEL_MAP)}")
    return MODEL_MAP[key](seed=seed, **kwargs)

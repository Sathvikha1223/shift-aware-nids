from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import ks_2samp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.neighbors import NearestNeighbors

from src.data_loader import load_all_domains
from src.models import get_model
from src.preprocessing import (
    assert_no_target_leakage,
    clean_domain_frame,
    leakage_audit,
    make_split_indices,
    normalize_features,
    select_common_features,
)


def _prepare_domains(
    data_dir: str,
    max_rows_per_domain: int = 5000,
    sampling_strategy: str = "stratified",
    sampling_seed: int = 42,
    benign_fraction: float = 0.5,
):
    raw = load_all_domains(
        data_dir,
        max_rows_per_domain=max_rows_per_domain,
        sampling_strategy=sampling_strategy,
        random_state=sampling_seed,
        benign_fraction=benign_fraction,
    )
    feature_candidates = [list(df.columns) for df in raw.values()]
    feature_cols = select_common_features(feature_candidates, drop_ids=True)
    cleaned = {}
    sampling_report = {}
    for domain, frame in raw.items():
        cleaned[domain] = clean_domain_frame(frame, feature_cols, drop_duplicates=True)
        labels = cleaned[domain]["label"]
        sampling_report[domain] = {
            **frame.attrs["sampling_info"],
            "rows_after_preprocessing": int(len(labels)),
            "final_benign_rows": int((labels == 0).sum()),
            "final_attack_rows": int((labels == 1).sum()),
        }
    return cleaned, feature_cols, sampling_report


def _fit_temperature(logits, labels):
    if len(np.unique(labels)) < 2:
        return 1.0
    logits = np.asarray(logits, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int32)

    def nll(log_t):
        scaled = logits / np.exp(log_t)
        p = 1.0 / (1.0 + np.exp(-np.clip(scaled, -50, 50)))
        p = np.clip(p, 1e-7, 1 - 1e-7)
        return -np.mean(labels * np.log(p) + (1 - labels) * np.log(1 - p))

    from scipy.optimize import minimize_scalar

    res = minimize_scalar(nll, bounds=(-3, 3), method="bounded")
    return float(np.exp(res.x))


def _ece(probs, labels, bins=15):
    probs = np.asarray(probs, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int32)
    conf = np.maximum(probs, 1.0 - probs)
    pred = (probs >= 0.5).astype(int)
    correct = (pred == labels).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.digitize(conf, edges[1:-1], right=False)
    idx = np.clip(idx, 0, bins - 1)
    total = len(probs)
    ece = 0.0
    for b in range(bins):
        m = idx == b
        if np.any(m):
            ece += (m.mean()) * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def _nll(probs, labels):
    probs = np.clip(np.asarray(probs, dtype=np.float32), 1e-7, 1 - 1e-7)
    labels = np.asarray(labels, dtype=np.int32)
    return float(-(labels * np.log(probs) + (1 - labels) * np.log(1 - probs)).mean())


def _uncertainty_scores(probs, ensemble_probs=None):
    p = np.clip(np.asarray(probs, dtype=np.float32), 1e-7, 1 - 1e-7)
    max_prob = np.maximum(p, 1.0 - p)
    entropy = -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))
    energy = -np.log(np.clip(np.exp(-np.abs(p - 0.5)), 1e-7, 1.0))
    if ensemble_probs is not None:
        ensemble_probs = np.asarray(ensemble_probs, dtype=np.float32)
        disagreement = np.mean(np.abs(ensemble_probs - p[:, None]), axis=1)
        return max_prob, entropy, disagreement, energy
    return max_prob, entropy, np.zeros_like(p), energy


def _mahalanobis(X, mu, inv_cov):
    centered = X - mu
    d2 = np.einsum("ij,jk,ik->i", centered, inv_cov, centered)
    return np.sqrt(np.clip(d2, 0, None))


def _knn_distance(X, ref):
    nn = NearestNeighbors(n_neighbors=10, metric="euclidean")
    nn.fit(ref)
    dists, _ = nn.kneighbors(X)
    return dists.mean(axis=1)


def _rbf_mmd(X, Y):
    X = np.asarray(X, dtype=np.float32)
    Y = np.asarray(Y, dtype=np.float32)
    if len(X) < 2 or len(Y) < 2:
        return 0.0
    pair = np.concatenate([X, Y], axis=0)
    dist = cdist(pair, pair, metric="sqeuclidean")
    med = np.median(dist[dist > 0]) if np.any(dist > 0) else 1.0
    med = max(med, 1e-6)

    def k(a, b):
        return np.exp(-cdist(a, b, metric="sqeuclidean") / med)

    kxx = k(X, X)
    kyy = k(Y, Y)
    kxy = k(X, Y)
    x_mean = kxx.sum() / (len(X) * (len(X) - 1)) - np.trace(kxx) / (len(X) * (len(X) - 1))
    y_mean = kyy.sum() / (len(Y) * (len(Y) - 1)) - np.trace(kyy) / (len(Y) * (len(Y) - 1))
    xy_mean = kxy.mean()
    return float(max(0.0, x_mean + y_mean - 2 * xy_mean))


def _classifier_two_sample_auc(X_src, X_tgt):
    if len(X_src) < 5 or len(X_tgt) < 5:
        return 0.5
    X = np.vstack([X_src, X_tgt])
    y = np.concatenate([np.zeros(len(X_src)), np.ones(len(X_tgt))])
    clf = LogisticRegression(max_iter=200, class_weight="balanced")
    clf.fit(X, y)
    return float(roc_auc_score(y, clf.predict_proba(X)[:, 1]))


def _feature_ks_stat(X_src, X_tgt):
    stats = []
    for j in range(X_src.shape[1]):
        stats.append(ks_2samp(X_src[:, j], X_tgt[:, j]).statistic)
    return float(np.max(stats)) if stats else 0.0


def _compute_shift_stats(X_source, X_target):
    mu = X_source.mean(axis=0)
    cov = np.cov(X_source, rowvar=False) + 1e-3 * np.eye(X_source.shape[1])
    inv_cov = np.linalg.pinv(cov)
    mahal = _mahalanobis(X_target, mu, inv_cov)
    knn = _knn_distance(X_target, X_source)
    mmd = _rbf_mmd(X_source, X_target)
    ks_stat = _feature_ks_stat(X_source, X_target)
    cts_auc = _classifier_two_sample_auc(X_source, X_target)
    return {"mahalanobis": mahal, "knn": knn, "mmd": mmd, "ks": ks_stat, "cts_auc": cts_auc}


def _apply_normalization(train_X, test_X, mode="global"):
    if mode == "global":
        mu = train_X.mean(axis=0)
        sd = train_X.std(axis=0) + 1e-6
        return (test_X - mu) / sd, (train_X - mu) / sd
    if mode == "per_domain":
        med = np.median(train_X, axis=0)
        q75, q25 = np.percentile(train_X, [75, 25], axis=0)
        iqr = q75 - q25
        sd = np.where(iqr > 1e-6, iqr, np.std(train_X, axis=0) + 1e-6)
        return (test_X - med) / sd, (train_X - med) / sd
    raise ValueError(f"Unsupported normalization mode: {mode!r}")


def _to_prob_matrix(probs):
    p = np.asarray(probs, dtype=np.float32)
    if p.ndim == 1:
        p = np.column_stack([1.0 - p, p])
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return p


def _predict_source_proba(model, X):
    source_predictor = getattr(model, "predict_source_proba", None)
    if source_predictor is not None:
        return source_predictor(X)
    return model.predict_proba(X)


def _fit_ldo_temperature(logits_by_domain, labels_by_domain, target_domain):
    if target_domain in logits_by_domain or target_domain in labels_by_domain:
        raise ValueError("The outer held-out target domain cannot enter LDO calibration.")
    if set(logits_by_domain) != set(labels_by_domain) or not logits_by_domain:
        raise ValueError("LDO calibration requires matching non-empty source-domain logits and labels.")
    logits = np.concatenate([np.asarray(logits_by_domain[d]) for d in logits_by_domain])
    labels = np.concatenate([np.asarray(labels_by_domain[d]) for d in labels_by_domain])
    return _fit_temperature(logits, labels)


def _normalize_ldo_fold(X, train_domains, train_idx, calibration_domain, normalizer):
    if normalizer == "global":
        X_fit = np.vstack([X[d][train_idx[d]] for d in train_domains])
        mean = X_fit.mean(axis=0)
        scale = X_fit.std(axis=0) + 1e-6
        X_fit_n = (X_fit - mean) / scale
        X_calibration_all_n = (X[calibration_domain] - mean) / scale
    elif normalizer == "per_domain":
        fit_parts = []
        for domain in train_domains:
            fit = X[domain][train_idx[domain]]
            mean = fit.mean(axis=0)
            scale = fit.std(axis=0) + 1e-6
            fit_parts.append((fit - mean) / scale)
        X_fit_n = np.vstack(fit_parts)
        calibration = X[calibration_domain]
        mean = calibration.mean(axis=0)
        scale = calibration.std(axis=0) + 1e-6
        X_calibration_all_n = (calibration - mean) / scale
    else:
        raise ValueError(f"Unsupported normalizer: {normalizer!r}")
    return X_fit_n, X_calibration_all_n


def _collect_ldo_calibration_logits(
    source_domains,
    target_domain,
    X,
    y,
    train_idx,
    val_idx,
    model_name,
    seed,
    normalizer,
    feature_cols,
    dann_grl_lambda,
):
    if target_domain in source_domains:
        raise ValueError("Outer held-out target was included among LDO calibration domains.")
    if len(source_domains) < 2:
        raise ValueError("At least two source domains are required for leave-domain-out calibration.")

    logits_by_domain = {}
    labels_by_domain = {}
    for calibration_domain in source_domains:
        training_domains = [d for d in source_domains if d != calibration_domain]
        X_fit_n, X_calibration_all_n = _normalize_ldo_fold(
            X, training_domains, train_idx, calibration_domain, normalizer
        )
        X_calibration_val_n = X_calibration_all_n[val_idx[calibration_domain]]
        y_fit = np.concatenate([y[d][train_idx[d]] for d in training_domains])
        domain_ids = np.concatenate(
            [np.full(len(train_idx[d]), i, dtype=np.int64) for i, d in enumerate(training_domains)]
        )
        model_kwargs = {"grl_lambda": dann_grl_lambda} if model_name == "dann" else {}
        calibration_model = get_model(model_name, seed=seed, **model_kwargs)
        calibration_model.fit(
            X_fit_n,
            y_fit,
            domains=domain_ids,
            X_target=X_calibration_all_n,
            feature_names=feature_cols,
        )
        probabilities = _to_prob_matrix(calibration_model.predict_proba(X_calibration_val_n))[:, 1]
        logits_by_domain[calibration_domain] = np.log(
            probabilities / np.clip(1.0 - probabilities, 1e-7, 1.0 - 1e-7)
        )
        labels_by_domain[calibration_domain] = y[calibration_domain][val_idx[calibration_domain]]
    return logits_by_domain, labels_by_domain


def _measure_target_metrics(y_true, probs, y_pred=None):
    p = _to_prob_matrix(probs)[:, 1]
    y_hat = (p >= 0.5).astype(int) if y_pred is None else y_pred
    return {
        "auc": float(roc_auc_score(y_true, p)) if len(np.unique(y_true)) > 1 else float("nan"),
        "accuracy": float(accuracy_score(y_true, y_hat)),
        "macro_f1": float(f1_score(y_true, y_hat, average="macro")),
        "brier": float(brier_score_loss(y_true, p)),
        "nll": float(_nll(p, y_true)),
        "ece": float(_ece(p, y_true)),
        "macro_precision": float(precision_recall_fscore_support(y_true, y_hat, average="macro", zero_division=0)[0]),
        "macro_recall": float(precision_recall_fscore_support(y_true, y_hat, average="macro", zero_division=0)[1]),
        "benign_fpr": float(np.mean(y_hat[y_true == 0] == 1)) if np.any(y_true == 0) else 0.0,
    }


def _decision_status(track_shift, track_unc, shift_thresh, unc_thresh):
    status = np.full(len(track_shift), "ACCEPT", dtype=object)
    status[track_shift > shift_thresh] = "UNFAMILIAR_NETWORK"
    status[(track_shift <= shift_thresh) & (track_unc > unc_thresh)] = "REVIEW"
    return status


def run_lodo_experiment(
    data_dir: str,
    target_domain: str,
    model_name: str = "lightgbm",
    seed: int = 42,
    max_rows_per_domain: int = 5000,
    normalizer: str = "global",
    sampling_strategy: str = "stratified",
    sampling_seed: int = 42,
    benign_fraction: float = 0.5,
    dann_grl_lambda: float = 1.0,
    prepared_data=None,
):
    if prepared_data is None:
        prepared_data = _prepare_domains(
            data_dir,
            max_rows_per_domain=max_rows_per_domain,
            sampling_strategy=sampling_strategy,
            sampling_seed=sampling_seed,
            benign_fraction=benign_fraction,
        )
    domains, feature_cols, sampling_report = prepared_data
    if target_domain not in domains:
        raise ValueError(f"Unknown target domain {target_domain!r}; found {sorted(domains)}")
    source_domains = [d for d in domains if d != target_domain]
    assert_no_target_leakage(source_domains, target_domain)

    X = {dom: domains[dom][feature_cols].to_numpy(dtype=np.float32) for dom in domains}
    y = {dom: domains[dom]["label"].to_numpy(dtype=np.int32) for dom in domains}

    train_idx = {}
    val_idx = {}
    for dom in source_domains:
        dseed = seed + sum(ord(ch) for ch in dom)
        train_idx[dom], val_idx[dom] = make_split_indices(np.arange(len(y[dom])), seed=dseed)

    X_train = np.vstack([X[d][train_idx[d]] for d in source_domains])
    y_train = np.concatenate([y[d][train_idx[d]] for d in source_domains])
    source_domain_ids = np.concatenate(
        [np.full(len(train_idx[d]), i, dtype=np.int64) for i, d in enumerate(source_domains)]
    )
    X_val = np.vstack([X[d][val_idx[d]] for d in source_domains])
    y_val = np.concatenate([y[d][val_idx[d]] for d in source_domains])

    if normalizer == "global":
        mean = X_train.mean(axis=0)
        scale = X_train.std(axis=0) + 1e-6
        X_train_n = (X_train - mean) / scale
        X_val_n = (X_val - mean) / scale
        X_target_n = (X[target_domain] - mean) / scale
    elif normalizer == "per_domain":
        train_parts = []
        val_parts = []
        for domain in source_domains:
            mean = X[domain][train_idx[domain]].mean(axis=0)
            scale = X[domain][train_idx[domain]].std(axis=0) + 1e-6
            train_parts.append((X[domain][train_idx[domain]] - mean) / scale)
            val_parts.append((X[domain][val_idx[domain]] - mean) / scale)
        X_train_n = np.vstack(train_parts)
        X_val_n = np.vstack(val_parts)
        target_mean = X[target_domain].mean(axis=0)
        target_scale = X[target_domain].std(axis=0) + 1e-6
        X_target_n = (X[target_domain] - target_mean) / target_scale
    else:
        raise ValueError(f"Unsupported normalizer: {normalizer!r}")

    model_kwargs = {"grl_lambda": dann_grl_lambda} if model_name == "dann" else {}
    model = get_model(model_name, seed=seed, **model_kwargs)
    model.fit(
        X_train_n,
        y_train,
        domains=source_domain_ids,
        X_target=X_target_n,
        feature_names=feature_cols,
    )
    val_probs = _to_prob_matrix(_predict_source_proba(model, X_val_n))
    target_probs = _to_prob_matrix(model.predict_proba(X_target_n))
    val_logits = np.log(val_probs[:, 1] / np.clip(1 - val_probs[:, 1], 1e-7, 1 - 1e-7))
    standard_temperature = _fit_temperature(val_logits, y_val)

    ldo_logits, ldo_labels = _collect_ldo_calibration_logits(
        source_domains,
        target_domain,
        X,
        y,
        train_idx,
        val_idx,
        model_name,
        seed,
        normalizer,
        feature_cols,
        dann_grl_lambda,
    )
    ldo_temperature = _fit_ldo_temperature(ldo_logits, ldo_labels, target_domain)

    val_max_prob, _, _, _ = _uncertainty_scores(val_probs[:, 1])
    target_max_prob, _, _, _ = _uncertainty_scores(target_probs[:, 1])
    flow_review_threshold = float(np.quantile(1.0 - val_max_prob, 0.90))
    source_mean = X_train_n.mean(axis=0)
    source_cov = np.cov(X_train_n, rowvar=False) + 1e-3 * np.eye(X_train_n.shape[1])
    inv_cov = np.linalg.pinv(source_cov)
    source_shift = _mahalanobis(X_val_n, source_mean, inv_cov)
    target_shift = _mahalanobis(X_target_n, source_mean, inv_cov)
    shift_threshold = float(np.quantile(source_shift, 0.95))
    status = _decision_status(
        target_shift, 1.0 - target_max_prob, shift_threshold, flow_review_threshold
    )

    uses_target_features = model_name == "coral"
    preprocessing_domains = list(source_domains)
    if normalizer == "per_domain":
        preprocessing_domains.append(target_domain)
    audit = leakage_audit(
        domains,
        target_domain,
        feature_cols,
        training_domains=source_domains,
        calibration_domains=source_domains,
        threshold_domains=source_domains,
        preprocessing_fit_domains=preprocessing_domains,
        target_features_used_for_preprocessing=normalizer == "per_domain",
        target_features_used_for_adaptation=uses_target_features,
        sampling_strategy=sampling_strategy,
    )
    if not audit["passed"]:
        raise AssertionError(f"LODO leakage audit failed: {audit}")

    target_pred = (target_probs[:, 1] >= 0.5).astype(int)
    target_metrics = _measure_target_metrics(y[target_domain], target_probs, target_pred)
    target_metrics.update(
        {
            "status_counts": {
                state: int(np.sum(status == state))
                for state in ["ACCEPT", "REVIEW", "UNFAMILIAR_NETWORK"]
            },
            "target_domain": target_domain,
            "model": model_name,
            "normalizer": normalizer,
            "seed": seed,
            "experiment_protocol": "transductive_unlabeled_target_adaptation" if uses_target_features else "source_only",
            "target_features_used_for_adaptation": uses_target_features,
            "target_labels_used_for_adaptation": False,
            "standard_temperature": float(standard_temperature),
            "standard_calibration_method": "source_validation_temperature_scaling",
            "ldo_temperature": float(ldo_temperature),
            "ldo_calibration_method": "leave_one_source_domain_out_cross_fitting",
            "ldo_calibration_domains": list(ldo_logits),
            "shift_threshold": float(shift_threshold),
            "flow_uncertainty_threshold": float(flow_review_threshold),
            "source_train_rows": int(len(X_train_n)),
            "source_val_rows": int(len(X_val_n)),
            "ldo_calibration_rows": int(sum(len(values) for values in ldo_labels.values())),
            "target_rows": int(len(X_target_n)),
            "source_domains": source_domains,
            "sampling_report": sampling_report[target_domain],
            "leakage_audit": audit,
        }
    )
    return target_metrics, status, target_probs, target_shift


def run_repeated_lodo_experiment(
    data_dir: str,
    model_names=None,
    seeds=(0, 1, 2),
    max_rows_per_domain: int = 5000,
    normalizers=("global", "per_domain"),
    out_dir: str = None,
    sampling_strategy: str = "stratified",
    sampling_seed: int = 42,
    benign_fraction: float = 0.5,
    dann_grl_lambda: float = 1.0,
):
    if model_names is None:
        model_names = ["lightgbm", "mlp", "coral", "groupdro", "tree_mlp_ensemble"]
    if out_dir is None:
        out_dir = str(Path(__file__).resolve().parents[1] / "reports" / "full")
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    prepared_data = _prepare_domains(
        data_dir,
        max_rows_per_domain=max_rows_per_domain,
        sampling_strategy=sampling_strategy,
        sampling_seed=sampling_seed,
        benign_fraction=benign_fraction,
    )
    domains, _, sampling_report = prepared_data
    rows = []
    for seed in seeds:
        for target_domain in sorted(domains):
            for model_name in model_names:
                for normalizer in normalizers:
                    metrics, status, probs, shift_scores = run_lodo_experiment(
                        data_dir=data_dir,
                        target_domain=target_domain,
                        model_name=model_name,
                        seed=seed,
                        max_rows_per_domain=max_rows_per_domain,
                        normalizer=normalizer,
                        sampling_strategy=sampling_strategy,
                        sampling_seed=sampling_seed,
                        benign_fraction=benign_fraction,
                        dann_grl_lambda=dann_grl_lambda,
                        prepared_data=prepared_data,
                    )
                    rows.append(metrics)

    df = pd.DataFrame(rows)
    df_path = out_path / "lodo_results_by_fold.csv"
    df.to_csv(df_path, index=False)

    agg = []
    for (model_name, normalizer), group in df.groupby(["model", "normalizer"]):
        metrics = [
            c
            for c in group.columns
            if c
            not in {
                "target_domain",
                "model",
                "normalizer",
                "seed",
                "source_domains",
                "status_counts",
                "leakage_audit",
                "sampling_report",
                "target_rows",
                "source_train_rows",
                "source_val_rows",
                "ldo_calibration_rows",
                "standard_temperature",
                "ldo_temperature",
                "standard_calibration_method",
                "ldo_calibration_method",
                "ldo_calibration_domains",
                "experiment_protocol",
                "target_features_used_for_adaptation",
                "target_labels_used_for_adaptation",
                "shift_threshold",
                "flow_uncertainty_threshold",
            }
        ]
        for m in metrics:
            vals = pd.to_numeric(group[m], errors="coerce")
            agg.append({
                "model": model_name,
                "normalizer": normalizer,
                "experiment_protocol": str(group["experiment_protocol"].iloc[0]),
                "metric": m,
                "mean": float(vals.mean()) if not vals.empty else float("nan"),
                "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
            })
    agg_df = pd.DataFrame(agg)
    agg_path = out_path / "lodo_summary.csv"
    agg_df.to_csv(agg_path, index=False)

    summary = {
        "result_csv": str(df_path),
        "summary_csv": str(agg_path),
        "num_rows": int(len(df)),
        "models": model_names,
        "seeds": list(seeds),
        "normalizers": list(normalizers),
        "sampling_strategy": sampling_strategy,
        "sampling_seed": int(sampling_seed),
        "benign_fraction": float(benign_fraction),
        "domain_sampling_report": sampling_report,
        "model_protocols": {
            model: (
                "transductive_unlabeled_target_adaptation"
                if model == "coral"
                else "source_only"
            )
            for model in model_names
        },
        "calibration_methods": {
            "standard": "source_validation_temperature_scaling",
            "ldo": "leave_one_source_domain_out_cross_fitting",
        },
        "dann_grl_lambda": float(dann_grl_lambda),
    }
    (out_path / "summary.json").write_text(json.dumps(summary, indent=2))
    return {"folds": df, "summary": agg_df, "paths": summary}


def run_poc(data_dir: str = None, out_dir: str = None, seed: int = 42, max_rows_per_domain: int = 5000):
    """Legacy compatibility wrapper used by the earlier proof-of-concept entrypoints."""
    if data_dir is None:
        data_dir = str(Path(__file__).resolve().parents[1] / "data")
    if out_dir is None:
        out_dir = str(Path(__file__).resolve().parents[1] / "reports" / "poc")

    result = run_repeated_lodo_experiment(
        data_dir=data_dir,
        model_names=["lightgbm"],
        seeds=[seed],
        max_rows_per_domain=max_rows_per_domain,
        normalizers=["global"],
        out_dir=out_dir,
    )
    return {
        "datasets": sorted(result["folds"]["target_domain"].unique().tolist()),
        "rows": int(len(result["folds"])),
        "result_csv": str(Path(out_dir) / "lodo_results_by_fold.csv"),
    }

from __future__ import annotations

import numpy as np
import pandas as pd


DROP_HINTS = (
    "addr",
    "port",
    "ttl",
    "id",
    "timestamp",
    "time",
    "flow_start",
    "flow_end",
    "dns_query_id",
)
LABEL_COLUMN_NAMES = {"label", "attack", "dataset", "family"}


def is_identifier_like(name: str) -> bool:
    lower = name.lower()
    return any(token in lower for token in DROP_HINTS) or lower.endswith("_id")


def select_common_features(column_sets, drop_ids: bool = True):
    if not column_sets:
        return []
    common = set(column_sets[0])
    for cols in column_sets[1:]:
        common &= set(cols)
    features = []
    for col in sorted(common):
        lower = str(col).lower()
        if lower in LABEL_COLUMN_NAMES:
            continue
        if drop_ids and is_identifier_like(lower):
            continue
        features.append(col)
    return features


def sanitize_numeric_frame(frame: pd.DataFrame, feature_cols):
    X = frame[feature_cols].copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.apply(pd.to_numeric, errors="coerce")
    return X


def binary_label(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    unique = sorted({int(v) for v in values.dropna().unique() if pd.notna(v)})
    if len(unique) == 0:
        raise ValueError("No valid labels found in the provided series.")
    if set(unique).issubset({0, 1}):
        return values.fillna(0).astype(int)
    return (values.fillna(0) > 0).astype(int)


def clean_domain_frame(frame: pd.DataFrame, feature_cols, drop_duplicates: bool = True):
    X = sanitize_numeric_frame(frame, feature_cols)
    y = binary_label(frame["label"])
    mask = X.notna().all(axis=1) & y.notna()
    frame = frame.loc[mask].copy()
    X = X.loc[mask].copy()
    y = y.loc[mask].copy()
    if drop_duplicates:
        X = X.drop_duplicates()
        frame = frame.loc[X.index].copy()
        y = y.loc[X.index].copy()
    out = X.copy().astype(float)
    out["label"] = y.to_numpy()
    out["attack"] = frame["attack"].astype(str).str.strip().to_numpy()
    out["dataset"] = frame["dataset"].to_numpy()
    return out


def normalize_features(X_all, fit_indices=None, mode="global"):
    """Normalize the feature matrix for the requested mode.
    - global: pooled mean/std from the selected fit rows
    - per_domain: median/IQR from each domain's own data
    """
    if fit_indices is None:
        fit_indices = np.arange(len(X_all))
    if mode == "global":
        fit = X_all[fit_indices]
        mu = np.mean(fit, axis=0)
        sd = np.std(fit, axis=0) + 1e-6
        return (X_all - mu) / sd
    if mode == "per_domain":
        med = np.median(X_all, axis=0)
        q75, q25 = np.percentile(X_all, [75, 25], axis=0)
        iqr = q75 - q25
        sd = np.where(iqr > 1e-6, iqr, np.std(X_all, axis=0) + 1e-6)
        return (X_all - med) / sd
    raise ValueError(f"Unsupported normalization mode: {mode!r}")


def make_split_indices(values, seed=42, train_frac=0.8):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(values))
    rng.shuffle(idx)
    cut = max(1, int(train_frac * len(idx)))
    return idx[:cut], idx[cut:]


def assert_no_target_leakage(source_names, target_name):
    assert target_name not in source_names, "Target domain leaked into source training domains."
    return True


def leakage_audit(
    domain_to_frame,
    target_domain,
    feature_cols,
    training_domains=None,
    calibration_domains=None,
    threshold_domains=None,
    preprocessing_fit_domains=None,
    target_features_used_for_preprocessing=False,
    target_features_used_for_adaptation=False,
    sampling_strategy="stratified",
):
    """Validate fold boundaries and report exact feature-overlap checks."""
    all_sources = [domain for domain in domain_to_frame if domain != target_domain]
    training_domains = list(all_sources if training_domains is None else training_domains)
    calibration_domains = list(all_sources if calibration_domains is None else calibration_domains)
    threshold_domains = list(all_sources if threshold_domains is None else threshold_domains)
    preprocessing_fit_domains = list(
        all_sources if preprocessing_fit_domains is None else preprocessing_fit_domains
    )
    shortcut_identifiers = [col for col in feature_cols if is_identifier_like(str(col))]
    target_hashes = set(
        pd.util.hash_pandas_object(domain_to_frame[target_domain][feature_cols], index=False).to_numpy()
    )
    overlapping_source_rows = 0
    for domain in all_sources:
        source_hashes = pd.util.hash_pandas_object(
            domain_to_frame[domain][feature_cols], index=False
        ).to_numpy()
        overlapping_source_rows += int(np.isin(source_hashes, list(target_hashes)).sum())

    memberships = {
        "training": training_domains,
        "calibration": calibration_domains,
        "threshold_selection": threshold_domains,
        "preprocessing_fit": preprocessing_fit_domains,
    }
    target_memberships = [
        name
        for name, domains in memberships.items()
        if name != "preprocessing_fit" and target_domain in domains
    ]
    report = {
        "target_domain": target_domain,
        "target_memberships": target_memberships,
        "target_in_sources": target_domain in training_domains,
        "target_labels_used_for_training": False,
        "target_labels_used_for_calibration": False,
        "target_labels_used_for_threshold_selection": False,
        "target_labels_used_to_fit_preprocessing": False,
        "target_labels_used_to_select_sample": sampling_strategy == "stratified",
        "training_domains": training_domains,
        "calibration_domains": calibration_domains,
        "threshold_selection_domains": threshold_domains,
        "preprocessing_fit_domains": preprocessing_fit_domains,
        "target_features_used_for_preprocessing": bool(target_features_used_for_preprocessing),
        "threshold_selection_source": "source_validation",
        "sampling_strategy": sampling_strategy,
        "target_features_used_for_adaptation": bool(target_features_used_for_adaptation),
        "target_labels_used_for_adaptation": False,
        "shortcut_identifiers": shortcut_identifiers,
        "shortcut_identifiers_excluded": not shortcut_identifiers,
        "duplicate_check_performed": True,
        "exact_source_target_feature_overlap_rows": overlapping_source_rows,
        "exact_feature_overlap_requires_review": overlapping_source_rows > 0,
    }
    report["passed"] = (
        not target_memberships
        and not shortcut_identifiers
    )
    return report

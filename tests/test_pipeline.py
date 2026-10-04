import os
import warnings

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
import torch

from src.data_loader import discover_dataset_files, detect_label_columns, load_dataset_sample
from src.experiment import _fit_ldo_temperature
from src.models import CORALModel, DANNModel, GradientReversalLayer, LightGBMModel
from src.preprocessing import leakage_audit, select_common_features

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(ROOT, "data")


def test_detect_dataset_files():
    files = discover_dataset_files(DATA_DIR)
    assert len(files) >= 4, files
    names = {os.path.basename(p) for p in files}
    assert any("NF-ToN-IoT" in x for x in names)
    assert any("NF-BoT-IoT" in x for x in names)
    assert any("NF-UNSW-NB15" in x for x in names)
    assert any("NF-CICIDS2018" in x for x in names)


def test_label_detection():
    cols = ["FLOW_START_MILLISECONDS", "FLOW_END_MILLISECONDS", "Attack", "Label", "IPV4_SRC_ADDR"]
    label_col, attack_col = detect_label_columns(cols)
    assert label_col == "Label"
    assert attack_col == "Attack"


def test_feature_selection_ignores_leakage_columns():
    cols = [
        "IPV4_SRC_ADDR",
        "L4_SRC_PORT",
        "FLOW_START_MILLISECONDS",
        "IN_BYTES",
        "OUT_BYTES",
        "Attack",
        "Label",
    ]
    features = select_common_features([cols], drop_ids=True)
    assert "IN_BYTES" in features
    assert "IPV4_SRC_ADDR" not in features
    assert "Attack" not in features
    assert "Label" not in features


def test_lightgbm_model_preserves_feature_names_without_warnings():
    X = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]], dtype=np.float32)
    y = np.array([0, 0, 1, 1], dtype=np.int32)

    model = LightGBMModel(seed=0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(X, y, feature_names=["f0", "f1"])
        model.predict_proba(X)

    messages = [str(item.message) for item in caught]
    assert not any("valid feature names" in msg for msg in messages)


def test_dataset_sampling_covers_classes_across_full_file(tmp_path):
    rows = 40
    source = tmp_path / "ordered.csv"
    frame = np.column_stack(
        [
            np.arange(rows),
            np.r_[np.zeros(20, dtype=int), np.ones(20, dtype=int)],
            np.r_[np.repeat("Benign", 20), np.repeat("Attack", 20)],
        ]
    )
    np.savetxt(source, frame, delimiter=",", header="feature,Label,Attack", comments="", fmt="%s")

    sample = load_dataset_sample(str(source), max_rows=10, chunksize=7, random_state=13)

    assert len(sample) == 10
    assert set(sample["label"].unique()) == {0, 1}
    assert not sample["feature"].duplicated().any()
    assert sample.attrs["sampling_info"]["sampling_strategy"] == "stratified"
    assert sample.attrs["sampling_info"]["population_rows"] == rows
    assert sample.attrs["sampling_info"]["sampled_benign_rows"] == 5
    assert sample.attrs["sampling_info"]["sampled_attack_rows"] == 5
    repeated = load_dataset_sample(str(source), max_rows=10, chunksize=7, random_state=13)
    assert sample["feature"].tolist() == repeated["feature"].tolist()


def test_parquet_sampling_uses_batches_and_preserves_both_classes(tmp_path):
    rows = 30
    frame = pd.DataFrame(
        {
            "feature": np.arange(rows),
            "Label": np.r_[np.zeros(15, dtype=int), np.ones(15, dtype=int)],
            "Attack": ["Benign"] * 15 + ["Scan"] * 15,
        }
    )
    source = tmp_path / "ordered.parquet"
    pq.write_table(pa.Table.from_pandas(frame), source, row_group_size=6)

    sample = load_dataset_sample(str(source), max_rows=8, chunksize=5, random_state=4)

    assert len(sample) == 8
    assert set(sample["label"].unique()) == {0, 1}
    assert sample.attrs["sampling_info"]["population_rows"] == rows


def test_leakage_audit_checks_fold_membership_shortcuts_and_duplicates():
    target = pd.DataFrame({"feature": [1.0, 2.0]})
    source = pd.DataFrame({"feature": [3.0, 4.0]})
    domains = {"source": source, "target": target}
    report = leakage_audit(
        domains,
        "target",
        ["feature"],
        training_domains=["source"],
        calibration_domains=["source"],
        threshold_domains=["source"],
        preprocessing_fit_domains=["source"],
    )
    assert report["passed"]
    assert report["duplicate_check_performed"]
    assert report["target_labels_used_for_training"] is False
    assert report["target_labels_used_for_calibration"] is False
    assert report["target_labels_used_for_threshold_selection"] is False

    duplicate_domains = {
        "source": pd.DataFrame({"feature": [1.0, 4.0]}),
        "target": target,
    }
    duplicate_report = leakage_audit(duplicate_domains, "target", ["feature"])
    assert duplicate_report["exact_source_target_feature_overlap_rows"] == 1
    assert duplicate_report["exact_feature_overlap_requires_review"]
    assert duplicate_report["passed"]


def test_gradient_reversal_layer_reverses_and_scales_gradient():
    features = torch.tensor([1.0, -2.0], requires_grad=True)
    strength = 0.35
    GradientReversalLayer(strength)(features).sum().backward()

    assert torch.allclose(features.grad, torch.full_like(features, -strength))


def test_dann_tiny_sanity_training():
    rng = np.random.default_rng(5)
    features = rng.normal(size=(16, 4)).astype(np.float32)
    labels = np.array([0, 1] * 8, dtype=np.int64)
    domains = np.repeat([0, 1], 8)
    model = DANNModel(seed=3, hidden_dim=8, max_epochs=2, grl_lambda=0.4)

    model.fit(features, labels, domains=domains)
    probabilities = model.predict_proba(features)

    assert model.grl_lambda == 0.4
    assert probabilities.shape == (16, 2)
    assert np.isfinite(probabilities).all()


def test_coral_maps_unlabeled_target_features_to_source_statistics():
    rng = np.random.default_rng(11)
    source = rng.normal(size=(4000, 3)) @ np.array(
        [[2.0, 0.8, 0.0], [0.0, 0.7, 0.4], [0.3, 0.0, 1.4]]
    ) + np.array([3.0, -2.0, 1.0])
    target = rng.normal(size=(4000, 3)) @ np.array(
        [[0.5, -0.6, 0.2], [1.2, 1.8, 0.0], [0.0, 0.3, 0.6]]
    ) + np.array([-4.0, 5.0, 2.0])
    model = CORALModel(seed=1, max_iter=2)
    model.source_mean_, model.source_cov_ = model._cov(source)
    model.target_mean_, model.target_cov_ = model._cov(target)

    aligned = model.transform(target)

    assert np.allclose(aligned.mean(axis=0), source.mean(axis=0), atol=0.1)
    assert np.allclose(np.cov(aligned, rowvar=False), np.cov(source, rowvar=False), atol=0.15)


def test_ldo_temperature_rejects_outer_target_domain():
    source_logits = {
        "source-a": np.array([-2.0, -0.5, 0.5, 2.0]),
        "source-b": np.array([-1.5, -0.2, 0.2, 1.5]),
    }
    source_labels = {
        "source-a": np.array([0, 0, 1, 1]),
        "source-b": np.array([0, 0, 1, 1]),
    }

    temperature = _fit_ldo_temperature(source_logits, source_labels, "outer-target")

    assert np.isfinite(temperature) and temperature > 0
    with np.testing.assert_raises_regex(ValueError, "outer held-out target"):
        _fit_ldo_temperature(
            {**source_logits, "outer-target": np.array([0.0, 100.0])},
            {**source_labels, "outer-target": np.array([0, 1])},
            "outer-target",
        )

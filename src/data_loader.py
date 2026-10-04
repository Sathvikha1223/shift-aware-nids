from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


DATASET_HINTS = {
    "unsw": "NF-UNSW-NB15-v2",
    "ton": "NF-ToN-IoT-v2",
    "bot": "NF-BoT-IoT-v2",
    "cic": "NF-CSE-CIC-IDS2018-v2",
    "2018": "NF-CSE-CIC-IDS2018-v2",
}


def discover_dataset_files(data_dir: str | os.PathLike[str]) -> List[str]:
    """Return CSV dataset files, skipping feature metadata CSVs."""
    base = Path(data_dir)
    files = []
    for pattern in ("**/*.csv", "**/*.parquet", "**/*.pq"):
        files.extend(glob.glob(str(base / pattern), recursive=True))
    ordered = sorted(files)
    cleaned = []
    for path in ordered:
        name = os.path.basename(path).lower()
        if "netflow_v3_features" in name or "features.csv" in name:
            continue
        if Path(path).suffix.lower() in {".csv", ".parquet", ".pq"} and os.path.isfile(path):
            cleaned.append(path)
    return cleaned


def infer_dataset_name(path: str | os.PathLike[str]) -> str:
    name = os.path.basename(str(path)).lower()
    for key, value in DATASET_HINTS.items():
        if key in name:
            return value
    return os.path.splitext(os.path.basename(path))[0]


def to_lower_keys(columns: Iterable[str]) -> dict[str, str]:
    return {str(c).strip().lower(): str(c).strip() for c in columns}


def detect_label_columns(columns: Sequence[str]) -> Tuple[Optional[str], Optional[str]]:
    """Return (label_col, attack_col) using common NetFlow naming conventions."""
    lookup = to_lower_keys(columns)
    label_names = ["label", "class", "attack_label", "label_name"]
    attack_names = ["attack", "attack_label", "category", "attack_category"]

    label_col = next((lookup[name] for name in label_names if name in lookup), None)
    attack_col = next((lookup[name] for name in attack_names if name in lookup), None)
    if label_col is None and "label" in lookup:
        label_col = lookup["label"]
    if attack_col is None and "attack" in lookup:
        attack_col = lookup["attack"]
    return label_col, attack_col


def stream_csv_chunks(path: str, chunksize: int = 200_000):
    return pd.read_csv(path, chunksize=chunksize)


def stream_dataset_chunks(path: str, chunksize: int = 200_000) -> Iterator[pd.DataFrame]:
    """Yield bounded CSV chunks or Parquet record batches as DataFrames."""
    suffix = Path(path).suffix.lower()
    if suffix == ".csv":
        yield from stream_csv_chunks(path, chunksize=chunksize)
        return
    if suffix in {".parquet", ".pq"}:
        parquet_file = pq.ParquetFile(path)
        for batch in parquet_file.iter_batches(batch_size=chunksize):
            yield batch.to_pandas()
        return
    raise ValueError(f"Unsupported dataset format {suffix!r} for {path!r}")


def _binary_sampling_strata(chunk: pd.DataFrame, label_col: str, attack_col: str) -> np.ndarray:
    labels = pd.to_numeric(chunk[label_col], errors="coerce")
    attack = chunk[attack_col].astype(str).str.strip().str.lower()
    benign = labels.eq(0) | attack.eq("benign")
    return np.where(benign.to_numpy(), 0, 1)


def _update_reservoir(rows, priorities, new_rows, new_priorities, capacity):
    if len(new_rows) == 0:
        return rows, priorities
    if rows is None:
        candidate_rows = new_rows.reset_index(drop=True)
        candidate_priorities = new_priorities
    else:
        candidate_rows = pd.concat([rows, new_rows], ignore_index=True)
        candidate_priorities = np.concatenate([priorities, new_priorities])
    if len(candidate_rows) > capacity:
        keep = np.argpartition(candidate_priorities, capacity - 1)[:capacity]
        candidate_rows = candidate_rows.iloc[keep].reset_index(drop=True)
        candidate_priorities = candidate_priorities[keep]
    return candidate_rows, candidate_priorities


def load_dataset_sample(
    path: str,
    max_rows: int = 5000,
    chunksize: int = 200_000,
    sampling_strategy: str = "stratified",
    random_state: int = 42,
    benign_fraction: float = 0.5,
):
    """Sample a whole CSV/Parquet file with bounded, deterministic memory use."""
    if max_rows is None or max_rows < 1:
        raise ValueError("max_rows must be a positive integer; uncapped full-file loading is disabled.")
    if sampling_strategy not in {"uniform", "stratified"}:
        raise ValueError("sampling_strategy must be 'uniform' or 'stratified'.")
    if not 0.0 <= benign_fraction <= 1.0:
        raise ValueError("benign_fraction must be between 0 and 1.")

    suffix = Path(path).suffix.lower()
    if suffix == ".csv":
        columns = list(pd.read_csv(path, nrows=0).columns)
    elif suffix in {".parquet", ".pq"}:
        columns = pq.ParquetFile(path).schema_arrow.names
    else:
        raise ValueError(f"Unsupported dataset format {suffix!r} for {path!r}")
    label_col, attack_col = detect_label_columns(columns)
    if label_col is None or attack_col is None:
        raise ValueError(
            f"Could not detect label/attack columns in {path}. Columns were: {columns}"
        )

    rng = np.random.default_rng(random_state)
    buckets = (0, 1) if sampling_strategy == "stratified" else (2,)
    reservoirs = {bucket: None for bucket in buckets}
    priorities = {bucket: np.empty(0, dtype=np.float64) for bucket in buckets}
    population_counts = {0: 0, 1: 0}
    population_rows = 0

    for chunk in stream_dataset_chunks(path, chunksize=chunksize):
        if label_col not in chunk or attack_col not in chunk:
            raise ValueError(f"Dataset columns changed while reading {path!r}")
        strata = _binary_sampling_strata(chunk, label_col, attack_col)
        population_rows += len(chunk)
        population_counts[0] += int(np.sum(strata == 0))
        population_counts[1] += int(np.sum(strata == 1))
        keys = np.zeros(len(chunk), dtype=np.int8) if sampling_strategy == "uniform" else strata
        for bucket in buckets:
            positions = np.flatnonzero(keys == bucket)
            if positions.size == 0:
                continue
            bucket_rows = chunk.iloc[positions]
            bucket_priorities = rng.random(len(bucket_rows))
            reservoirs[bucket], priorities[bucket] = _update_reservoir(
                reservoirs[bucket], priorities[bucket], bucket_rows, bucket_priorities, max_rows
            )

    if population_rows == 0:
        raise ValueError(f"No rows loaded from {path!r}")

    if sampling_strategy == "stratified":
        take = {
            0: min(population_counts[0], int(round(max_rows * benign_fraction))),
            1: min(population_counts[1], max_rows - int(round(max_rows * benign_fraction))),
        }
        remaining = min(max_rows, population_rows) - sum(take.values())
        for bucket in (1, 0):
            available = population_counts[bucket] - take[bucket]
            extra = min(remaining, available)
            take[bucket] += extra
            remaining -= extra
        sampled_parts = []
        for bucket in (0, 1):
            if take[bucket]:
                chosen = np.argpartition(priorities[bucket], take[bucket] - 1)[:take[bucket]]
                sampled_parts.append(reservoirs[bucket].iloc[chosen])
        df = pd.concat(sampled_parts, ignore_index=True)
    else:
        chosen = np.argpartition(priorities[2], min(max_rows, population_rows) - 1)[:min(max_rows, population_rows)]
        df = reservoirs[2].iloc[chosen].reset_index(drop=True)

    if len(df) > 1:
        df = df.iloc[rng.permutation(len(df))].reset_index(drop=True)
    df = df.rename(columns={label_col: "label", attack_col: "attack"})
    df["dataset"] = infer_dataset_name(path)
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df["attack"] = df["attack"].astype(str).str.strip()
    df["is_benign"] = df["attack"].str.lower().eq("benign")
    sampled_strata = _binary_sampling_strata(df, "label", "attack")
    df.attrs["sampling_info"] = {
        "sampling_strategy": sampling_strategy,
        "sampling_seed": int(random_state),
        "requested_rows": int(max_rows),
        "population_rows": int(population_rows),
        "population_benign_rows": int(population_counts[0]),
        "population_attack_rows": int(population_counts[1]),
        "sampled_rows": int(len(df)),
        "sampled_benign_rows": int(np.sum(sampled_strata == 0)),
        "sampled_attack_rows": int(np.sum(sampled_strata == 1)),
        "both_classes_available": bool(population_counts[0] and population_counts[1]),
        "both_classes_sampled": bool(np.any(sampled_strata == 0) and np.any(sampled_strata == 1)),
    }
    return df


def load_all_domains(
    data_dir: str,
    max_rows_per_domain: int = 5000,
    sampling_strategy: str = "stratified",
    random_state: int = 42,
    benign_fraction: float = 0.5,
) -> dict[str, pd.DataFrame]:
    datasets = {}
    for index, path in enumerate(discover_dataset_files(data_dir)):
        domain = infer_dataset_name(path)
        df = load_dataset_sample(
            path,
            max_rows=max_rows_per_domain,
            sampling_strategy=sampling_strategy,
            random_state=random_state + index,
            benign_fraction=benign_fraction,
        )
        datasets[domain] = df
    return datasets

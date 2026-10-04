from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experiment import run_repeated_lodo_experiment


def main():
    parser = argparse.ArgumentParser(description="Run the repeated strict LODO NIDS experiment with multiple models and source-only threshold selection.")
    parser.add_argument("--data-dir", default=str(Path(__file__).resolve().parents[1] / "data"), type=str)
    parser.add_argument("--out-dir", default=str(Path(__file__).resolve().parents[1] / "reports" / "full"), type=str)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--models", nargs="+", default=["lightgbm", "mlp", "coral", "groupdro", "dann", "tree_mlp_ensemble"], type=str)
    parser.add_argument("--max-rows-per-domain", type=int, default=2000)
    parser.add_argument("--normalizers", nargs="+", default=["global", "per_domain"], type=str)
    parser.add_argument("--sampling-strategy", choices=["stratified", "uniform"], default="stratified")
    parser.add_argument("--sampling-seed", type=int, default=42)
    parser.add_argument("--benign-fraction", type=float, default=0.5)
    parser.add_argument("--dann-grl-lambda", type=float, default=1.0)
    args = parser.parse_args()

    result = run_repeated_lodo_experiment(
        data_dir=args.data_dir,
        model_names=args.models,
        seeds=tuple(args.seeds),
        max_rows_per_domain=args.max_rows_per_domain,
        normalizers=tuple(args.normalizers),
        out_dir=args.out_dir,
        sampling_strategy=args.sampling_strategy,
        sampling_seed=args.sampling_seed,
        benign_fraction=args.benign_fraction,
        dann_grl_lambda=args.dann_grl_lambda,
    )
    print(result["paths"])


if __name__ == "__main__":
    main()

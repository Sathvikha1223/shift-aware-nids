from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experiment import run_poc


def main():
    parser = argparse.ArgumentParser(description="Run the small laptop-friendly proof-of-concept for the Shift-Aware NIDS pipeline.")
    parser.add_argument("--data-dir", default=str(Path(__file__).resolve().parents[1] / "data"), type=str)
    parser.add_argument("--out-dir", default=str(Path(__file__).resolve().parents[1] / "reports" / "poc"), type=str)
    parser.add_argument("--max-rows-per-domain", default=20_000, type=int)
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()

    result = run_poc(
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        seed=args.seed,
        max_rows_per_domain=args.max_rows_per_domain,
    )
    print(result)


if __name__ == "__main__":
    main()

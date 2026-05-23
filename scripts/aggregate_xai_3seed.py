"""Aggregate multi-seed XAI faithfulness with bootstrap CIs over seed/sample pairs."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.bootstrap import bootstrap_ci


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate XAI faithfulness JSONs across seeds.")
    parser.add_argument("--xai-dir", type=Path, default=ROOT / "data" / "paper_assets" / "xai_full")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 17, 2026])
    parser.add_argument("--suffix", type=str, default="_reflectance64")
    parser.add_argument("--output-stem", type=str, default="faithfulness_n200_3seed_reflectance64")
    return parser.parse_args()


def _input_path(xai_dir: Path, seed: int, suffix: str) -> Path:
    return xai_dir / f"faithfulness_n200_seed{seed}{suffix}.json"


def main() -> None:
    args = parse_args()
    rows = []
    for seed in args.seeds:
        path = _input_path(args.xai_dir, seed, args.suffix)
        if not path.exists():
            raise FileNotFoundError(f"Missing XAI faithfulness file: {path}")
        data = json.loads(path.read_text(encoding="utf-8"))
        for row in data["per_sample"]:
            rows.append({"seed": seed, **row})

    df = pd.DataFrame(rows)
    args.xai_dir.mkdir(parents=True, exist_ok=True)
    detail_csv = args.xai_dir / f"{args.output_stem}.csv"
    df.to_csv(detail_csv, index=False)

    summary = []
    for method, sub in df.groupby("method"):
        out = {"method": method, "n_observations": len(sub), "n_seeds": sub["seed"].nunique()}
        for col in ("deletion_auc", "insertion_auc", "faithfulness_gap"):
            vals = sub[col].to_numpy(dtype=np.float64)
            point, low, high = bootstrap_ci(vals, n_bootstrap=2000)
            out[col] = round(point, 3)
            out[f"{col}_ci_low"] = round(low, 3)
            out[f"{col}_ci_high"] = round(high, 3)
        summary.append(out)

    summary_df = pd.DataFrame(summary).sort_values("method")
    summary_csv = args.xai_dir / f"{args.output_stem}_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"Wrote {detail_csv}")
    print(f"Wrote {summary_csv}")
    print(summary_df.to_markdown(index=False))


if __name__ == "__main__":
    main()

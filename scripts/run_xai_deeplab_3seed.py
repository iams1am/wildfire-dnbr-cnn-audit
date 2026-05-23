"""Run XAI faithfulness on the DeepLabv3+ headline winner across 3 seeds.

The existing main XAI table reports faithfulness for the Siamese model only;
this driver runs the same audit on DeepLabv3+, the cross-region winner, by
invoking scripts/run_xai_faithfulness.py
once per seed with the appropriate DeepLabv3+ checkpoint, then aggregates a
3-seed summary CSV with the same shape as
data/paper_assets/xai_full/faithfulness_n200_3seed_reflectance64_summary.csv.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEEDS = [42, 17, 2026]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DeepLabv3+ XAI faithfulness across 3 seeds.")
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--num-samples", type=int, default=200)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--patch-index",
        type=Path,
        default=PROJECT_ROOT / "data" / "patches" / "australia_full" / "patch_index.csv",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "paper_assets" / "xai_deeplab",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    per_seed_csvs = []
    for seed in args.seeds:
        ckpt = PROJECT_ROOT / "data" / "runs" / f"deeplab_qaon_seed{seed}_reflectance64" / "best_model.pt"
        if not ckpt.exists():
            print(f"  SKIP seed {seed}: missing {ckpt}")
            continue
        out_json = args.out_dir / f"faithfulness_n{args.num_samples}_seed{seed}.json"
        out_csv = args.out_dir / f"faithfulness_n{args.num_samples}_seed{seed}.csv"
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "run_xai_faithfulness.py"),
            "--model-name", "deeplab",
            "--checkpoint", str(ckpt),
            "--patch-index", str(args.patch_index),
            "--output-json", str(out_json),
            "--output-csv", str(out_csv),
            "--num-samples", str(args.num_samples),
            "--seed", str(seed),
            "--device", args.device,
        ]
        print(f"\n=== DeepLabv3+ XAI seed={seed} ===")
        subprocess.run(cmd, check=True)
        per_seed_csvs.append((seed, out_csv))

    # Aggregate 3-seed: per-method insertion / deletion / faithfulness gap with bootstrap-percentile CI95.
    if not per_seed_csvs:
        print("No seeds produced a result.")
        return

    rows = []
    for seed, csv in per_seed_csvs:
        df = pd.read_csv(csv)
        df["seed"] = seed
        rows.append(df)
    pooled = pd.concat(rows, ignore_index=True)
    pooled.to_csv(args.out_dir / f"faithfulness_n{args.num_samples}_3seed_long.csv", index=False)

    rng = np.random.default_rng(0)
    summary_rows = []
    for method, sub in pooled.groupby("method"):
        for col in ("insertion_auc", "deletion_auc", "faithfulness_gap"):
            x = sub[col].to_numpy()
            mean_val = float(np.nanmean(x))
            boot = rng.choice(x, size=(2000, len(x)), replace=True).mean(axis=1)
            lo, hi = np.nanpercentile(boot, [2.5, 97.5])
            summary_rows.append({
                "method": method,
                "metric": col,
                "mean": round(mean_val, 4),
                "ci_low": round(float(lo), 4),
                "ci_high": round(float(hi), 4),
                "n": int(len(x)),
            })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(args.out_dir / f"faithfulness_n{args.num_samples}_3seed_summary.csv", index=False)
    print("\n=== DeepLabv3+ 3-seed XAI faithfulness summary ===")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

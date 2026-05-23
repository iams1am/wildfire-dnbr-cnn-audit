"""Mediterranean third-region evaluation across all 3 seeds {42, 17, 2026}.

eval_mediterranean_third_region.py only ran seed 42 (the only QA-off checkpoint
available when that diagnostic was added). All six architectures now have
seed-17 and seed-2026 QA-off reflectance64 checkpoints.

The label-circular caveat still applies: Mediterranean labels come from the same
dNBR rule at tau=0.10 used in training, so this measures cross-region
re-derivation, not independent perimeter agreement.
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
MODELS = ["baseline", "siamese", "siamese_fcn_conc", "siamese_fcn_diff", "deeplab", "change_transformer"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mediterranean evaluation across 3 seeds.")
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=PROJECT_ROOT
        / "data"
        / "evaluation"
        / "mediterranean_3seed_reflectance64"
        / "mediterranean_summary_3seed.csv",
    )
    return parser.parse_args()


def run_one_seed(seed: int, device: str, batch_size: int) -> Path:
    out_dir = PROJECT_ROOT / "data" / "evaluation" / f"mediterranean_external_test_seed{seed}_reflectance64"
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "eval_mediterranean_third_region.py"),
        "--seed", str(seed),
        "--device", device,
        "--batch-size", str(batch_size),
        "--output-dir", str(out_dir),
    ]
    print(f"\n=== Mediterranean evaluation, seed={seed} ===")
    subprocess.run(cmd, check=True)
    return out_dir / "mediterranean_summary.csv"


def main() -> None:
    args = parse_args()
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)

    per_seed_frames = []
    for seed in args.seeds:
        csv = run_one_seed(seed, args.device, args.batch_size)
        if not csv.exists():
            print(f"  WARNING: {csv} not produced for seed {seed}")
            continue
        df = pd.read_csv(csv)
        df["seed"] = seed
        per_seed_frames.append(df)

    if not per_seed_frames:
        print("No per-seed results produced.")
        return

    long = pd.concat(per_seed_frames, ignore_index=True)
    long.to_csv(args.out_csv.with_name(args.out_csv.stem + "_long.csv"), index=False)

    # Per-model: seed-mean and seed-std of pixel-pooled IoU/F1/precision/recall.
    agg = long.groupby("model").agg(
        iou_mean=("iou", "mean"),
        iou_std=("iou", "std"),
        f1_mean=("f1", "mean"),
        f1_std=("f1", "std"),
        precision_mean=("precision", "mean"),
        precision_std=("precision", "std"),
        recall_mean=("recall", "mean"),
        recall_std=("recall", "std"),
        n_seeds=("seed", "nunique"),
    ).round(4).reset_index().sort_values("iou_mean", ascending=False)

    agg.to_csv(args.out_csv, index=False)
    print(f"\n=== 3-seed Mediterranean summary ===")
    print(agg.to_string(index=False))
    print(f"\nWrote {args.out_csv}")
    print(f"Wrote {args.out_csv.with_name(args.out_csv.stem + '_long.csv')}")


if __name__ == "__main__":
    main()

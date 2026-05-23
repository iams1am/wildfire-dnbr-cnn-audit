from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.bootstrap import bootstrap_ci
from src.evaluation.inference import evaluate_checkpoint
from src.models.factory import list_models


METRICS = ["iou", "f1", "precision", "recall", "area_mae_sq_km", "area_rmse_sq_km"]
NON_NEG = {"area_mae_sq_km", "area_rmse_sq_km"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate CaBuAr patches for all seeds and aggregate CIs.")
    parser.add_argument("--patch-index", type=Path, default=PROJECT_ROOT / "data" / "patches_cabuar_full" / "patch_index.csv")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data" / "evaluation" / "cabuar_full_reflectance64")
    parser.add_argument("--checkpoint-root", type=Path, default=PROJECT_ROOT / "data" / "runs")
    parser.add_argument("--models", nargs="+", choices=list_models(), default=list_models())
    parser.add_argument("--qa-state", choices=["on", "off"], default="off")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 17, 2026])
    parser.add_argument("--run-suffix", type=str, default="_reflectance64")
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--pixel-resolution-m", type=float, default=30.0)
    parser.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=False,
                        help="CaBuAr extraction already stores reflectance-like [0,1] values; keep this off by default.")
    return parser.parse_args()


def _checkpoint_path(root: Path, model: str, qa_state: str, seed: int, suffix: str) -> Path:
    return root / f"{model}_qa{qa_state}_seed{seed}{suffix}" / "best_model.pt"


def _aggregate(summary_rows: list[dict[str, object]]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    df = pd.DataFrame(summary_rows)
    for model, group in df.groupby("model"):
        row: dict[str, object] = {"model": model, "n_seeds": int(group["seed"].nunique())}
        for metric in METRICS:
            vals = group[metric].to_numpy(dtype=np.float64)
            floor = 0.0 if metric in NON_NEG else None
            point, low, high = bootstrap_ci(vals, lower_floor=floor)
            row[metric] = point
            row[f"{metric}_ci_low"] = low
            row[f"{metric}_ci_high"] = high
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    if not args.patch_index.exists():
        raise FileNotFoundError(f"CaBuAr patch index not found: {args.patch_index}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, object]] = []
    for seed in args.seeds:
        seed_dir = args.output_dir / f"seed{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        for model in args.models:
            checkpoint = _checkpoint_path(args.checkpoint_root, model, args.qa_state, seed, args.run_suffix)
            if not checkpoint.exists():
                print(f"missing checkpoint, skipping: {checkpoint}")
                continue
            patch_csv = seed_dir / f"{model}_patch_metrics.csv"
            summary = evaluate_checkpoint(
                model_name=model,
                checkpoint_path=checkpoint,
                patch_index_csv=args.patch_index,
                output_csv=patch_csv,
                base_channels=args.base_channels,
                batch_size=args.batch_size,
                device=args.device,
                threshold=args.threshold,
                pixel_resolution_m=args.pixel_resolution_m,
                normalize=args.normalize,
            )
            summary_json = seed_dir / f"{model}_summary.json"
            summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            summary_rows.append({"model": model, "seed": seed, **summary})
            print(f"{model} seed={seed}: IoU={summary['iou']:.4f}")

    if not summary_rows:
        raise ValueError("No CaBuAr evaluations were completed.")

    per_seed = pd.DataFrame(summary_rows)
    per_seed_csv = args.output_dir / "cabuar_per_seed.csv"
    per_seed.to_csv(per_seed_csv, index=False)
    aggregate = _aggregate(summary_rows)
    aggregate_csv = args.output_dir / "cabuar_summary_ci.csv"
    aggregate.to_csv(aggregate_csv, index=False)
    aggregate_md = args.output_dir / "cabuar_summary_ci.md"
    aggregate_md.write_text(aggregate.to_markdown(index=False), encoding="utf-8")

    print(f"Wrote {per_seed_csv}")
    print(f"Wrote {aggregate_csv}")
    print(f"Wrote {aggregate_md}")


if __name__ == "__main__":
    main()

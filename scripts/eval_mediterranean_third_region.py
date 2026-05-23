"""Evaluate the 6 reflectance64 seed-42 QA-off checkpoints on the Mediterranean
third-region patch set, then aggregate into a single comparison table.

This produces the third-region external-validity result.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.inference import evaluate_checkpoint

MODELS = ["baseline", "siamese", "siamese_fcn_conc", "siamese_fcn_diff", "deeplab", "change_transformer"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate reflectance64 models on the Mediterranean third region.")
    parser.add_argument("--patch-index", type=Path, default=PROJECT_ROOT / "data" / "patches_noqa" / "mediterranean_full" / "patch_index.csv")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data" / "evaluation" / "mediterranean_external_test_seed42_reflectance64")
    parser.add_argument("--ckpt-suffix", type=str, default="_reflectance64")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for model_name in MODELS:
        ckpt = PROJECT_ROOT / "data" / "runs" / f"{model_name}_qaoff_seed{args.seed}{args.ckpt_suffix}" / "best_model.pt"
        if not ckpt.exists():
            print(f"  SKIP {model_name}: missing {ckpt}")
            continue
        print(f"Evaluating {model_name}...")
        out_csv = args.output_dir / f"{model_name}_patch_metrics.csv"
        try:
            summary = evaluate_checkpoint(
                model_name=model_name,
                checkpoint_path=ckpt,
                patch_index_csv=args.patch_index,
                output_csv=out_csv,
                base_channels=args.base_channels,
                batch_size=args.batch_size,
                device=args.device,
                normalize=True,
            )
            (args.output_dir / f"{model_name}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
            rows.append({
                "model": model_name,
                "iou": round(summary["iou"], 4),
                "f1": round(summary["f1"], 4),
                "precision": round(summary["precision"], 4),
                "recall": round(summary["recall"], 4),
                "iou_patch_mean": round(summary["iou_patch_mean"], 4),
                "n_patches": int(summary["patch_count"]),
            })
            print(f"  IoU={summary['iou']:.4f}  F1={summary['f1']:.4f}  P={summary['precision']:.4f}  R={summary['recall']:.4f}")
        except Exception as e:
            print(f"  FAILED {model_name}: {type(e).__name__}: {e}")

    if not rows:
        print("No models evaluated.")
        return
    df = pd.DataFrame(rows).sort_values("iou", ascending=False)
    csv_path = args.output_dir / "mediterranean_summary.csv"
    md_path = args.output_dir / "mediterranean_summary.md"
    df.to_csv(csv_path, index=False)
    md_path.write_text(df.to_markdown(index=False), encoding="utf-8")
    print()
    print(df.to_markdown(index=False))
    print(f"\nWrote {csv_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()

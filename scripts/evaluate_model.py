from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.inference import evaluate_checkpoint
from src.models.factory import list_models


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained segmentation model on patch index.")
    parser.add_argument("--model-name", choices=list_models(), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--patch-index", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--pixel-resolution-m", type=float, default=30.0)
    parser.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    patch_metrics_csv = args.output_dir / f"{args.model_name}_patch_metrics.csv"
    summary_json = args.output_dir / f"{args.model_name}_summary.json"

    summary = evaluate_checkpoint(
        model_name=args.model_name,
        checkpoint_path=args.checkpoint,
        patch_index_csv=args.patch_index,
        output_csv=patch_metrics_csv,
        base_channels=args.base_channels,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        threshold=args.threshold,
        pixel_resolution_m=args.pixel_resolution_m,
        normalize=args.normalize,
    )
    summary.update(
        {
            "model_name": args.model_name,
            "checkpoint": str(args.checkpoint),
            "patch_index": str(args.patch_index),
            "base_channels": args.base_channels,
            "num_workers": args.num_workers,
            "threshold": args.threshold,
            "normalize": args.normalize,
        }
    )
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote patch metrics: {patch_metrics_csv}")
    print(f"Wrote summary: {summary_json}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

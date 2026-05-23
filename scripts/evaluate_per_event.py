from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.per_event import evaluate_per_event
from src.models.factory import list_models


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint by event/sensor group.")
    parser.add_argument("--model-name", choices=list_models(), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--patch-index", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--pixel-resolution-m", type=float, default=30.0)
    parser.add_argument("--group-columns", nargs="+", default=["event_id", "sensor"])
    parser.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    grouped = evaluate_per_event(
        model_name=args.model_name,
        checkpoint_path=args.checkpoint,
        patch_index_csv=args.patch_index,
        output_csv=args.output_csv,
        base_channels=args.base_channels,
        batch_size=args.batch_size,
        device=args.device,
        pixel_resolution_m=args.pixel_resolution_m,
        group_columns=tuple(args.group_columns),
        threshold=args.threshold,
        normalize=args.normalize,
    )
    print(f"Wrote per-event metrics: {args.output_csv}")
    print(grouped.to_string(index=False))


if __name__ == "__main__":
    main()

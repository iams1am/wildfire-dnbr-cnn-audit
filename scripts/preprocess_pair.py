from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocess import harmonize_pair


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Harmonize one pre/post wildfire image pair.")
    parser.add_argument("--pre", type=Path, required=True, help="Path to pre-event image.")
    parser.add_argument("--post", type=Path, required=True, help="Path to post-event image.")
    parser.add_argument(
        "--label",
        type=Path,
        default=None,
        help="Optional path to burned-area binary label raster.",
    )
    parser.add_argument(
        "--pre-clear-mask",
        type=Path,
        default=None,
        help="Optional path to a pre-event binary clear mask (1=clear, 0=masked).",
    )
    parser.add_argument(
        "--post-clear-mask",
        type=Path,
        default=None,
        help="Optional path to a post-event binary clear mask (1=clear, 0=masked).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for harmonized outputs.",
    )
    parser.add_argument("--target-crs", default="EPSG:6933")
    parser.add_argument("--target-resolution-m", type=float, default=30.0)
    parser.add_argument(
        "--spectral-resampling",
        choices=["nearest", "bilinear", "cubic", "average"],
        default="bilinear",
    )
    parser.add_argument(
        "--mask-resampling",
        choices=["nearest", "bilinear", "cubic", "average"],
        default="nearest",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = harmonize_pair(
        pre_image_path=args.pre,
        post_image_path=args.post,
        label_mask_path=args.label,
        pre_clear_mask_path=args.pre_clear_mask,
        post_clear_mask_path=args.post_clear_mask,
        output_dir=args.output_dir,
        target_crs=args.target_crs,
        target_resolution_m=args.target_resolution_m,
        spectral_resampling=args.spectral_resampling,
        mask_resampling=args.mask_resampling,
    )
    for key, value in outputs.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()

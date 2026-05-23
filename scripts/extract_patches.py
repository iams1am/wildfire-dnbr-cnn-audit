from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.patching import extract_patches_from_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract training patches from harmonized manifests.")
    parser.add_argument("--manifest", type=Path, required=True, help="Path to *_harmonized.csv manifest.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "patches",
        help="Directory where patch files and patch_index.csv will be saved.",
    )
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--overlap-fraction", type=float, default=0.5)
    parser.add_argument("--min-valid-fraction", type=float, default=0.8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    patch_index = extract_patches_from_manifest(
        manifest_path=args.manifest,
        output_root=args.output_root,
        patch_size=args.patch_size,
        overlap_fraction=args.overlap_fraction,
        min_valid_fraction=args.min_valid_fraction,
    )
    print(f"Wrote patch index: {patch_index}")


if __name__ == "__main__":
    main()

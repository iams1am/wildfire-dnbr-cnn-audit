from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.stac_downloader import create_dnbr_label


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regenerate dNBR label rasters from existing pre/post stacks using corrected reflectance scaling."
    )
    parser.add_argument("--manifests", nargs="+", type=Path, required=True)
    parser.add_argument("--dnbr-threshold", type=float, default=0.1)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _path_from_row(row: pd.Series, column: str) -> Path:
    value = str(row.get(column, "")).strip()
    if not value or value.lower() == "nan":
        raise ValueError(f"Missing {column}")
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> None:
    args = parse_args()
    total = 0
    seen: set[Path] = set()
    for manifest_path in args.manifests:
        manifest = pd.read_csv(manifest_path)
        required = {"pair_id", "sensor", "pre_image_path", "post_image_path", "label_mask_path"}
        missing = required.difference(manifest.columns)
        if missing:
            raise ValueError(f"{manifest_path} missing columns {sorted(missing)}")

        for _, row in manifest.iterrows():
            pair_id = str(row["pair_id"]).strip()
            sensor = str(row["sensor"]).strip().lower()
            pre_path = _path_from_row(row, "pre_image_path")
            post_path = _path_from_row(row, "post_image_path")
            label_path = _path_from_row(row, "label_mask_path")
            label_key = label_path.resolve()
            if label_key in seen:
                continue
            seen.add(label_key)
            if not pre_path.exists() or not post_path.exists():
                print(f"Skipping {pair_id}: missing pre/post stack")
                continue
            print(f"Regenerating {label_path} from {pair_id} ({sensor})")
            if not args.dry_run:
                create_dnbr_label(
                    pre_stack_path=pre_path,
                    post_stack_path=post_path,
                    output_path=label_path,
                    sensor=sensor,
                    dnbr_threshold=args.dnbr_threshold,
                )
            total += 1
    print(f"{'Would regenerate' if args.dry_run else 'Regenerated'} {total} label rasters.")


if __name__ == "__main__":
    main()

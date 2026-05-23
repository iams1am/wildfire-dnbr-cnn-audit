from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset_config import load_dataset_config
from src.data.manifest_builder import default_qa_mode
from src.data.preprocess import harmonize_pair
from src.data.qa_masks import SUPPORTED_QA_MODES, build_clear_mask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch preprocess image pairs from a manifest CSV."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "datasets.yaml")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Reprocess rows even if output paths already exist.",
    )
    return parser.parse_args()


def _safe_str(value: object) -> str:
    return "" if pd.isna(value) else str(value)


def _require_path(path_value: str, field_name: str, pair_id: str) -> Path:
    path = Path(path_value)
    if not path.exists():
        raise FileNotFoundError(f"{field_name} not found for pair '{pair_id}': {path}")
    return path


def main() -> None:
    args = parse_args()
    cfg = load_dataset_config(args.config)

    manifest_df = pd.read_csv(args.manifest)
    required_columns = {"pair_id", "pre_image_path", "post_image_path", "label_mask_path", "cloud_fraction"}
    missing = required_columns.difference(set(manifest_df.columns))
    if missing:
        raise ValueError(f"Manifest missing columns: {sorted(missing)}")

    for optional_column in ("pre_qa_path", "post_qa_path", "qa_mode", "sensor"):
        if optional_column not in manifest_df.columns:
            manifest_df[optional_column] = ""

    max_cloud = cfg.pairing_rules.max_cloud_fraction
    output_rows: list[dict[str, object]] = []

    for _, row in manifest_df.iterrows():
        cloud_fraction = float(row["cloud_fraction"]) if not pd.isna(row["cloud_fraction"]) else 0.0
        if cloud_fraction > max_cloud:
            continue

        pair_id = _safe_str(row["pair_id"]).strip()
        if not pair_id:
            continue

        pre_image = _require_path(_safe_str(row["pre_image_path"]).strip(), "pre_image_path", pair_id)
        post_image = _require_path(_safe_str(row["post_image_path"]).strip(), "post_image_path", pair_id)
        label_mask_raw = _safe_str(row["label_mask_path"]).strip()
        label_mask = _require_path(label_mask_raw, "label_mask_path", pair_id) if label_mask_raw else None
        sensor = _safe_str(row["sensor"]).strip()
        qa_mode = _safe_str(row["qa_mode"]).strip().lower() or default_qa_mode(sensor)
        pre_qa_raw = _safe_str(row["pre_qa_path"]).strip()
        post_qa_raw = _safe_str(row["post_qa_path"]).strip()

        pair_output_dir = args.output_root / pair_id
        pre_out = pair_output_dir / "pre_harmonized.tif"
        post_out = pair_output_dir / "post_harmonized.tif"
        pre_clear_mask_harmonized = pair_output_dir / "pre_clear_mask_harmonized.tif"
        post_clear_mask_harmonized = pair_output_dir / "post_clear_mask_harmonized.tif"
        pre_clear_mask_raw = pair_output_dir / "pre_clear_mask_raw.tif"
        post_clear_mask_raw = pair_output_dir / "post_clear_mask_raw.tif"

        pre_clear_mask_path: Path | None = None
        post_clear_mask_path: Path | None = None

        if pre_qa_raw or post_qa_raw:
            if not pre_qa_raw or not post_qa_raw:
                raise ValueError(
                    f"Both pre_qa_path and post_qa_path are required when QA masking is used (pair '{pair_id}')."
                )
            if not qa_mode:
                raise ValueError(f"qa_mode is required when QA paths are provided (pair '{pair_id}').")
            if qa_mode not in SUPPORTED_QA_MODES:
                raise ValueError(f"Unsupported qa_mode '{qa_mode}' for pair '{pair_id}'.")

            pre_qa_path = _require_path(pre_qa_raw, "pre_qa_path", pair_id)
            post_qa_path = _require_path(post_qa_raw, "post_qa_path", pair_id)
            if args.overwrite or not pre_clear_mask_raw.exists():
                build_clear_mask(pre_qa_path, pre_clear_mask_raw, qa_mode)
            if args.overwrite or not post_clear_mask_raw.exists():
                build_clear_mask(post_qa_path, post_clear_mask_raw, qa_mode)
            pre_clear_mask_path = pre_clear_mask_raw
            post_clear_mask_path = post_clear_mask_raw

        if not args.overwrite and pre_out.exists() and post_out.exists():
            outputs = {"pre": pre_out, "post": post_out}
            label_out = pair_output_dir / "label_harmonized.tif"
            if label_out.exists():
                outputs["label"] = label_out
            if pre_clear_mask_harmonized.exists():
                outputs["pre_clear_mask"] = pre_clear_mask_harmonized
            if post_clear_mask_harmonized.exists():
                outputs["post_clear_mask"] = post_clear_mask_harmonized
        else:
            outputs = harmonize_pair(
                pre_image_path=pre_image,
                post_image_path=post_image,
                label_mask_path=label_mask,
                pre_clear_mask_path=pre_clear_mask_path,
                post_clear_mask_path=post_clear_mask_path,
                output_dir=pair_output_dir,
                target_crs=cfg.harmonization.target_crs,
                target_resolution_m=cfg.harmonization.target_resolution_m,
                spectral_resampling=cfg.harmonization.spectral_resampling,
                mask_resampling=cfg.harmonization.mask_resampling,
            )

        row_dict = row.to_dict()
        row_dict["pre_image_harmonized"] = str(outputs["pre"])
        row_dict["post_image_harmonized"] = str(outputs["post"])
        row_dict["label_mask_harmonized"] = str(outputs["label"]) if "label" in outputs else ""
        row_dict["pre_clear_mask_harmonized"] = str(outputs["pre_clear_mask"]) if "pre_clear_mask" in outputs else ""
        row_dict["post_clear_mask_harmonized"] = str(outputs["post_clear_mask"]) if "post_clear_mask" in outputs else ""
        output_rows.append(row_dict)

    output_columns = list(manifest_df.columns) + [
        "pre_image_harmonized",
        "post_image_harmonized",
        "label_mask_harmonized",
        "pre_clear_mask_harmonized",
        "post_clear_mask_harmonized",
    ]
    output_df = pd.DataFrame(output_rows, columns=output_columns)
    output_path = args.manifest.with_name(f"{args.manifest.stem}_harmonized.csv")
    output_df.to_csv(output_path, index=False)
    print(f"Wrote harmonized manifest: {output_path}")


if __name__ == "__main__":
    main()

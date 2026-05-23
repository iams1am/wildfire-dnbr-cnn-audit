from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window


def _start_positions(length: int, patch_size: int, stride: int) -> list[int]:
    if patch_size > length:
        print(f"patch_size ({patch_size}) is larger than dimension ({length}). Will pad right/bottom edges.")

    return list(range(0, length, stride))


def extract_pair_patches(
    *,
    pair_id: str,
    pre_image_path: Path,
    post_image_path: Path,
    output_dir: Path,
    label_mask_path: Path | None = None,
    patch_size: int = 256,
    overlap_fraction: float = 0.5,
    min_valid_fraction: float = 0.8,
    event_id: str = "",
    sensor: str = "",
    region: str = "",
    split: str = "",
) -> pd.DataFrame:
    if not 0.0 <= overlap_fraction < 1.0:
        raise ValueError("overlap_fraction must be in [0, 1).")
    if not 0.0 <= min_valid_fraction <= 1.0:
        raise ValueError("min_valid_fraction must be in [0, 1].")

    stride = max(1, int(round(patch_size * (1.0 - overlap_fraction))))
    output_dir.mkdir(parents=True, exist_ok=True)

    with rasterio.open(pre_image_path) as pre_src, rasterio.open(post_image_path) as post_src:
        if (pre_src.width, pre_src.height) != (post_src.width, post_src.height):
            raise ValueError(f"Shape mismatch for pair '{pair_id}' between pre and post images.")
        if pre_src.crs != post_src.crs or pre_src.transform != post_src.transform:
            raise ValueError(f"Grid mismatch for pair '{pair_id}' between pre and post images.")
        label_src = rasterio.open(label_mask_path) if label_mask_path is not None else None
        try:
            if label_src is not None:
                if (pre_src.width, pre_src.height) != (label_src.width, label_src.height):
                    raise ValueError(f"Shape mismatch for pair '{pair_id}' between image and label.")
                if pre_src.crs != label_src.crs or pre_src.transform != label_src.transform:
                    raise ValueError(f"Grid mismatch for pair '{pair_id}' between image and label.")

            row_starts = _start_positions(pre_src.height, patch_size, stride)
            col_starts = _start_positions(pre_src.width, patch_size, stride)
            records: list[dict[str, object]] = []
            patch_index = 0

            for row_start in row_starts:
                for col_start in col_starts:
                    window = Window(col_start, row_start, patch_size, patch_size)
                    pre_patch = pre_src.read(window=window, boundless=True, fill_value=0)
                    post_patch = post_src.read(window=window, boundless=True, fill_value=0)

                    valid = np.any(pre_patch != 0, axis=0) & np.any(post_patch != 0, axis=0)
                    valid_fraction = float(valid.mean())
                    if valid_fraction < min_valid_fraction:
                        continue

                    if label_src is not None:
                        label_patch = label_src.read(1, window=window, boundless=True, fill_value=0).astype(np.uint8)
                        burned_fraction = float((label_patch > 0).mean())
                    else:
                        label_patch = np.zeros((patch_size, patch_size), dtype=np.uint8)
                        burned_fraction = float("nan")

                    patch_name = f"{pair_id}_r{row_start}_c{col_start}.npz"
                    patch_path = output_dir / patch_name
                    np.savez_compressed(
                        patch_path,
                        pre=pre_patch.astype(np.float32),
                        post=post_patch.astype(np.float32),
                        label=label_patch,
                    )
                    records.append(
                        {
                            "pair_id": pair_id,
                            "event_id": event_id,
                            "sensor": sensor,
                            "region": region,
                            "split": split,
                            "patch_path": str(patch_path),
                            "row_start": row_start,
                            "col_start": col_start,
                            "patch_size": patch_size,
                            "valid_fraction": valid_fraction,
                            "burned_fraction": burned_fraction,
                        }
                    )
                    patch_index += 1

            return pd.DataFrame(records)
        finally:
            if label_src is not None:
                label_src.close()


def extract_patches_from_manifest(
    manifest_path: Path,
    output_root: Path,
    *,
    patch_size: int = 256,
    overlap_fraction: float = 0.5,
    min_valid_fraction: float = 0.8,
) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(manifest_path)
    required_columns = {"pair_id", "pre_image_harmonized", "post_image_harmonized"}
    missing = required_columns.difference(set(manifest.columns))
    if missing:
        raise ValueError(f"Harmonized manifest missing columns: {sorted(missing)}")

    all_records: list[pd.DataFrame] = []
    for _, row in manifest.iterrows():
        pair_id = str(row["pair_id"]).strip()
        if not pair_id:
            continue

        pre = Path(str(row["pre_image_harmonized"]).strip())
        post = Path(str(row["post_image_harmonized"]).strip())
        if not pre.exists() or not post.exists():
            continue

        label_raw = str(row.get("label_mask_harmonized", "")).strip()
        label_path = Path(label_raw) if label_raw else None
        pair_df = extract_pair_patches(
            pair_id=pair_id,
            pre_image_path=pre,
            post_image_path=post,
            label_mask_path=label_path if label_path is None or label_path.exists() else None,
            output_dir=output_root / pair_id,
            patch_size=patch_size,
            overlap_fraction=overlap_fraction,
            min_valid_fraction=min_valid_fraction,
            event_id=str(row.get("event_id", "")).strip(),
            sensor=str(row.get("sensor", "")).strip(),
            region=str(row.get("region", "")).strip(),
            split=str(row.get("split", "")).strip(),
        )
        if not pair_df.empty:
            all_records.append(pair_df)

    if all_records:
        patch_index_df = pd.concat(all_records, ignore_index=True)
    else:
        patch_index_df = pd.DataFrame(
            columns=[
                "pair_id",
                "event_id",
                "sensor",
                "region",
                "split",
                "patch_path",
                "row_start",
                "col_start",
                "patch_size",
                "valid_fraction",
                "burned_fraction",
            ]
        )

    patch_index_path = output_root / "patch_index.csv"
    patch_index_df.to_csv(patch_index_path, index=False)
    return patch_index_path

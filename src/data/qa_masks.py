from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio


SUPPORTED_QA_MODES = {"landsat_qa_pixel", "sentinel2_qa60", "sentinel2_scl"}


def landsat_clear_mask_from_qa_pixel(qa_pixel: np.ndarray) -> np.ndarray:
    if qa_pixel.ndim != 2:
        raise ValueError("Landsat QA_PIXEL must be a 2D array.")
    invalid_bits = [0, 1, 2, 3, 4, 5]
    invalid = np.zeros_like(qa_pixel, dtype=bool)
    for bit in invalid_bits:
        invalid |= ((qa_pixel >> bit) & 1).astype(bool)
    return ~invalid


def sentinel2_clear_mask_from_qa60(qa60: np.ndarray) -> np.ndarray:
    if qa60.ndim != 2:
        raise ValueError("Sentinel-2 QA60 must be a 2D array.")
    cloud = (qa60 & (1 << 10)) != 0
    cirrus = (qa60 & (1 << 11)) != 0
    return ~(cloud | cirrus)


def sentinel2_clear_mask_from_scl(scl: np.ndarray) -> np.ndarray:
    if scl.ndim != 2:
        raise ValueError("Sentinel-2 SCL must be a 2D array.")
    invalid_classes = {0, 1, 3, 7, 8, 9, 10, 11}
    clear = np.ones_like(scl, dtype=bool)
    for cls in invalid_classes:
        clear &= scl != cls
    return clear


def build_clear_mask(qa_path: str | Path, output_path: str | Path, qa_mode: str) -> Path:
    qa_path = Path(qa_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    mode = qa_mode.strip().lower()
    if mode not in SUPPORTED_QA_MODES:
        raise ValueError(f"Unsupported qa_mode '{qa_mode}'. Supported: {sorted(SUPPORTED_QA_MODES)}")

    with rasterio.open(qa_path) as src:
        qa = src.read(1)
        if mode == "landsat_qa_pixel":
            clear = landsat_clear_mask_from_qa_pixel(qa)
        elif mode == "sentinel2_qa60":
            clear = sentinel2_clear_mask_from_qa60(qa)
        else:
            clear = sentinel2_clear_mask_from_scl(qa)

        profile = src.profile.copy()
        profile.update({"count": 1, "dtype": "uint8", "compress": "lzw", "nodata": 0})
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(clear.astype("uint8"), 1)
    return output_path


def apply_clear_mask(
    image_path: str | Path,
    clear_mask_path: str | Path,
    output_path: str | Path,
) -> Path:
    image_path = Path(image_path)
    clear_mask_path = Path(clear_mask_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(image_path) as src, rasterio.open(clear_mask_path) as mask_src:
        if src.width != mask_src.width or src.height != mask_src.height:
            raise ValueError("Image and clear-mask dimensions do not match.")
        if src.crs != mask_src.crs:
            raise ValueError("Image and clear-mask CRS do not match.")
        if src.transform != mask_src.transform:
            raise ValueError("Image and clear-mask grids are not aligned.")

        clear = mask_src.read(1) > 0
        data = src.read()
        fill_value = src.nodata if src.nodata is not None else 0

        masked_data = data.copy()
        masked_data[:, ~clear] = fill_value

        profile = src.profile.copy()
        profile.update({"compress": "lzw"})
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(masked_data)

    return output_path

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import rasterio
import torch
from rasterio.windows import Window
from torch import nn

from src.data.sensor_normalization import normalize_patch


def start_positions(length: int, patch_size: int, stride: int) -> list[int]:
    if length <= 0:
        raise ValueError("length must be positive.")
    if patch_size <= 0 or stride <= 0:
        raise ValueError("patch_size and stride must be positive.")
    return list(range(0, length, stride))


def blending_weight(patch_size: int, *, min_weight: float = 0.05) -> np.ndarray:
    """2D Hann-style weight map with a positive floor for image borders."""
    if patch_size <= 1:
        return np.ones((patch_size, patch_size), dtype=np.float32)
    one_d = np.hanning(patch_size).astype(np.float32)
    weight = np.outer(one_d, one_d)
    weight = weight / max(float(weight.max()), 1e-6)
    return np.maximum(weight, min_weight).astype(np.float32)


def _read_boundless(src: rasterio.io.DatasetReader, row: int, col: int, patch_size: int) -> np.ndarray:
    window = Window(col, row, patch_size, patch_size)
    return src.read(window=window, boundless=True, fill_value=0).astype(np.float32)


def _read_mask_boundless(src: rasterio.io.DatasetReader, row: int, col: int, patch_size: int) -> np.ndarray:
    window = Window(col, row, patch_size, patch_size)
    return src.read(1, window=window, boundless=True, fill_value=0)


def _metrics(pred: np.ndarray, target: np.ndarray, valid: np.ndarray) -> dict[str, float]:
    pred_f = (pred > 0).astype(np.float32)
    target_f = (target > 0).astype(np.float32)
    valid_f = (valid > 0).astype(np.float32)
    pred_f *= valid_f
    target_f *= valid_f

    tp = float((pred_f * target_f).sum())
    fp = float((pred_f * (1.0 - target_f)).sum())
    fn = float(((1.0 - pred_f) * target_f).sum())
    union = tp + fp + fn
    denom_f1 = (2.0 * tp) + fp + fn
    precision_denom = tp + fp
    recall_denom = tp + fn
    return {
        "iou": tp / union if union > 0.0 else float("nan"),
        "f1": (2.0 * tp) / denom_f1 if denom_f1 > 0.0 else float("nan"),
        "precision": tp / precision_denom if precision_denom > 0.0 else float("nan"),
        "recall": tp / recall_denom if recall_denom > 0.0 else float("nan"),
        "tp_total": tp,
        "fp_total": fp,
        "fn_total": fn,
        "valid_pixels": float(valid_f.sum()),
    }


def predict_scene_weighted(
    *,
    model: nn.Module,
    pre_image_path: Path,
    post_image_path: Path,
    sensor: str,
    patch_size: int = 256,
    stride: int = 128,
    device: str = "cuda",
    threshold: float = 0.5,
    label_mask_path: Path | None = None,
    pre_clear_mask_path: Path | None = None,
    post_clear_mask_path: Path | None = None,
    normalize: bool = True,
    output_prob_path: Path | None = None,
    output_mask_path: Path | None = None,
) -> dict[str, float]:
    """Run overlapped scene inference and optionally evaluate against a scene label."""
    device_obj = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    model = model.to(device_obj)
    model.eval()

    with rasterio.open(pre_image_path) as pre_src, rasterio.open(post_image_path) as post_src:
        if (pre_src.width, pre_src.height) != (post_src.width, post_src.height):
            raise ValueError("Pre and post images have different dimensions.")
        if pre_src.crs != post_src.crs or pre_src.transform != post_src.transform:
            raise ValueError("Pre and post images are not on the same grid.")

        height, width = pre_src.height, pre_src.width
        weighted_sum = np.zeros((height, width), dtype=np.float32)
        weight_sum = np.zeros((height, width), dtype=np.float32)
        valid_scene = np.zeros((height, width), dtype=bool)
        weight = blending_weight(patch_size)

        row_starts = start_positions(height, patch_size, stride)
        col_starts = start_positions(width, patch_size, stride)
        window_count = 0

        with torch.no_grad():
            for row in row_starts:
                for col in col_starts:
                    pre_patch = _read_boundless(pre_src, row, col, patch_size)
                    post_patch = _read_boundless(post_src, row, col, patch_size)
                    valid_patch = np.any(pre_patch != 0, axis=0) & np.any(post_patch != 0, axis=0)
                    if not valid_patch.any():
                        continue
                    if normalize:
                        pre_patch = normalize_patch(pre_patch, sensor)
                        post_patch = normalize_patch(post_patch, sensor)

                    pre_t = torch.from_numpy(pre_patch).unsqueeze(0).to(device_obj)
                    post_t = torch.from_numpy(post_patch).unsqueeze(0).to(device_obj)
                    probs = torch.sigmoid(model(pre_t, post_t))[0, 0].detach().cpu().numpy().astype(np.float32)

                    r1 = min(row + patch_size, height)
                    c1 = min(col + patch_size, width)
                    rr = r1 - row
                    cc = c1 - col
                    weighted_sum[row:r1, col:c1] += probs[:rr, :cc] * weight[:rr, :cc]
                    weight_sum[row:r1, col:c1] += weight[:rr, :cc]
                    valid_scene[row:r1, col:c1] |= valid_patch[:rr, :cc]
                    window_count += 1

        probabilities = np.divide(
            weighted_sum,
            weight_sum,
            out=np.zeros_like(weighted_sum, dtype=np.float32),
            where=weight_sum > 0,
        )
        pred_mask = ((probabilities >= threshold) & valid_scene).astype(np.uint8)

        if pre_clear_mask_path is not None and post_clear_mask_path is not None:
            with rasterio.open(pre_clear_mask_path) as pre_clear, rasterio.open(post_clear_mask_path) as post_clear:
                clear = (pre_clear.read(1) > 0) & (post_clear.read(1) > 0)
                valid_scene &= clear
                pred_mask = ((probabilities >= threshold) & valid_scene).astype(np.uint8)

        profile = pre_src.profile.copy()
        profile.update(count=1, compress="deflate")
        if output_prob_path is not None:
            output_prob_path.parent.mkdir(parents=True, exist_ok=True)
            prob_profile = profile.copy()
            prob_profile.update(dtype="float32", nodata=None)
            with rasterio.open(output_prob_path, "w", **prob_profile) as dst:
                dst.write(probabilities.astype(np.float32), 1)
        if output_mask_path is not None:
            output_mask_path.parent.mkdir(parents=True, exist_ok=True)
            mask_profile = profile.copy()
            mask_profile.update(dtype="uint8", nodata=0)
            with rasterio.open(output_mask_path, "w", **mask_profile) as dst:
                dst.write(pred_mask.astype(np.uint8), 1)

        result = {
            "width": float(width),
            "height": float(height),
            "window_count": float(window_count),
            "valid_fraction": float(valid_scene.mean()),
        }

        if label_mask_path is not None:
            with rasterio.open(label_mask_path) as label_src:
                if (label_src.width, label_src.height) != (width, height):
                    raise ValueError("Label image has different dimensions from the input images.")
                label = (label_src.read(1) > 0).astype(np.uint8)
            scene_metrics = _metrics(pred_mask, label, valid_scene)
            pixel_area_sq_km = abs(pre_src.transform.a * pre_src.transform.e) / 1_000_000.0
            true_area = float(((label > 0) & valid_scene).sum()) * pixel_area_sq_km
            pred_area = float(pred_mask.sum()) * pixel_area_sq_km
            scene_metrics.update(
                {
                    "true_area_sq_km": true_area,
                    "pred_area_sq_km": pred_area,
                    "abs_error_sq_km": abs(pred_area - true_area),
                    "area_rmse_sq_km": math.sqrt((pred_area - true_area) ** 2),
                }
            )
            result.update(scene_metrics)

        return result

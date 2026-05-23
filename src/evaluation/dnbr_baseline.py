from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.training.datasets import PatchPairDataset


NIR_CHANNEL_INDEX = 3
SWIR_CHANNEL_INDEX = 4


def _compute_dnbr_mask(pre: np.ndarray, post: np.ndarray, threshold: float) -> np.ndarray:
    pre_nir = pre[NIR_CHANNEL_INDEX].astype(np.float32)
    pre_swir = pre[SWIR_CHANNEL_INDEX].astype(np.float32)
    post_nir = post[NIR_CHANNEL_INDEX].astype(np.float32)
    post_swir = post[SWIR_CHANNEL_INDEX].astype(np.float32)

    pre_denom = pre_nir + pre_swir
    post_denom = post_nir + post_swir
    valid = (pre_denom != 0.0) & (post_denom != 0.0)
    pre_nbr = np.zeros_like(pre_nir, dtype=np.float32)
    post_nbr = np.zeros_like(post_nir, dtype=np.float32)
    pre_nbr[valid] = (pre_nir[valid] - pre_swir[valid]) / (pre_denom[valid] + 1e-6)
    post_nbr[valid] = (post_nir[valid] - post_swir[valid]) / (post_denom[valid] + 1e-6)
    dnbr = pre_nbr - post_nbr
    return ((dnbr >= threshold) & valid).astype(np.uint8)


def _metrics_for_pair(pred: np.ndarray, target: np.ndarray, smooth: float = 1e-6) -> tuple[float, float, float, float, float, float, float]:
    pred_f = pred.astype(np.float32)
    target_f = (target >= 0.5).astype(np.float32)
    tp = float((pred_f * target_f).sum())
    fp = float((pred_f * (1.0 - target_f)).sum())
    fn = float(((1.0 - pred_f) * target_f).sum())
    union = tp + fp + fn
    iou = (tp + smooth) / (union + smooth)
    f1 = (2.0 * tp + smooth) / (2.0 * tp + fp + fn + smooth)
    precision = (tp + smooth) / (tp + fp + smooth)
    recall = (tp + smooth) / (tp + fn + smooth)
    return tp, fp, fn, iou, f1, precision, recall


def evaluate_dnbr_baseline(
    *,
    patch_index_csv: Path,
    output_csv: Path,
    threshold: float = 0.1,
    pixel_resolution_m: float = 30.0,
    normalize: bool = True,
) -> dict[str, float]:
    """Non-ML dNBR thresholding baseline evaluated on the patch set."""
    dataset = PatchPairDataset(patch_index_csv, normalize=normalize)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    pixel_area_sq_km = (pixel_resolution_m * pixel_resolution_m) / 1_000_000.0

    tp_total = fp_total = fn_total = 0.0
    ious, f1s, precisions, recalls, abs_errors = [], [], [], [], []
    rows: list[dict[str, float]] = []

    with torch.no_grad():
        for batch in loader:
            pre = batch["pre"][0].numpy()
            post = batch["post"][0].numpy()
            label = batch["label"][0, 0].numpy() if batch["label"].ndim == 4 else batch["label"][0].numpy()

            pred = _compute_dnbr_mask(pre, post, threshold)
            tp, fp, fn, iou, f1, precision, recall = _metrics_for_pair(pred, label)
            tp_total += tp
            fp_total += fp
            fn_total += fn
            ious.append(iou)
            f1s.append(f1)
            precisions.append(precision)
            recalls.append(recall)

            true_px = float((label >= 0.5).sum())
            pred_px = float(pred.sum())
            abs_errors.append(abs(pred_px - true_px) * pixel_area_sq_km)
            rows.append(
                {
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "true_area_sq_km": true_px * pixel_area_sq_km,
                    "pred_area_sq_km": pred_px * pixel_area_sq_km,
                    "abs_error_sq_km": abs_errors[-1],
                    "iou": iou,
                    "f1": f1,
                    "precision": precision,
                    "recall": recall,
                }
            )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_csv, index=False)

    union_total = tp_total + fp_total + fn_total
    pixel_iou = tp_total / union_total if union_total > 0 else float("nan")
    pixel_f1 = (2 * tp_total) / (2 * tp_total + fp_total + fn_total) if (2 * tp_total + fp_total + fn_total) > 0 else float("nan")
    pixel_precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else float("nan")
    pixel_recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else float("nan")

    mae = float(np.mean(abs_errors))
    rmse = float(math.sqrt(np.mean(np.square(abs_errors))))
    return {
        "model": "dnbr_threshold",
        "threshold": threshold,
        "iou": pixel_iou,
        "f1": pixel_f1,
        "precision": pixel_precision,
        "recall": pixel_recall,
        "iou_patch_mean": float(np.mean(ious)),
        "f1_patch_mean": float(np.mean(f1s)),
        "precision_patch_mean": float(np.mean(precisions)),
        "recall_patch_mean": float(np.mean(recalls)),
        "area_mae_sq_km": mae,
        "area_rmse_sq_km": rmse,
        "patch_count": float(len(rows)),
        "tp_total": tp_total,
        "fp_total": fp_total,
        "fn_total": fn_total,
    }


def sweep_dnbr_thresholds(
    *,
    patch_index_csv: Path,
    thresholds: list[float],
    output_csv: Path,
    pixel_resolution_m: float = 30.0,
    normalize: bool = True,
) -> pd.DataFrame:
    """Sensitivity sweep: metrics for a set of dNBR thresholds."""
    summaries: list[dict[str, float]] = []
    for t in thresholds:
        tmp_csv = output_csv.with_name(f"{output_csv.stem}_t{t:.2f}.csv")
        summary = evaluate_dnbr_baseline(
            patch_index_csv=patch_index_csv,
            output_csv=tmp_csv,
            threshold=t,
            pixel_resolution_m=pixel_resolution_m,
            normalize=normalize,
        )
        summaries.append(summary)
    df = pd.DataFrame(summaries)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    return df

"""Patch-pooled and pixel-pooled segmentation evaluation.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.models.factory import build_model
from src.training.datasets import PatchPairDataset


def _per_patch_counts(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-patch TP/FP/FN counts."""
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()
    targets = (targets >= 0.5).float()
    tp = (preds * targets).sum(dim=(1, 2, 3))
    fp = (preds * (1 - targets)).sum(dim=(1, 2, 3))
    fn = ((1 - preds) * targets).sum(dim=(1, 2, 3))
    return tp, fp, fn


def _per_patch_metrics(tp: np.ndarray, fp: np.ndarray, fn: np.ndarray, smooth: float = 1e-6) -> dict[str, np.ndarray]:
    """Per-patch IoU/F1/precision/recall with a tiny smoothing constant."""
    union = tp + fp + fn
    iou = (tp + smooth) / (union + smooth)
    f1 = (2 * tp + smooth) / (2 * tp + fp + fn + smooth)
    precision = (tp + smooth) / (tp + fp + smooth)
    recall = (tp + smooth) / (tp + fn + smooth)
    return {"iou": iou, "f1": f1, "precision": precision, "recall": recall}


def _pixel_pooled(tp_total: float, fp_total: float, fn_total: float, smooth: float = 0.0) -> dict[str, float]:
    """Pixel-pooled (micro-aggregated) metrics over all patches."""
    union = tp_total + fp_total + fn_total
    iou = tp_total / (union + smooth) if union + smooth > 0 else float("nan")
    f1 = (2 * tp_total) / (2 * tp_total + fp_total + fn_total + smooth) if (2 * tp_total + fp_total + fn_total + smooth) > 0 else float("nan")
    precision = tp_total / (tp_total + fp_total + smooth) if (tp_total + fp_total + smooth) > 0 else float("nan")
    recall = tp_total / (tp_total + fn_total + smooth) if (tp_total + fn_total + smooth) > 0 else float("nan")
    return {"iou": iou, "f1": f1, "precision": precision, "recall": recall}


def evaluate_checkpoint(
    *,
    model_name: str,
    checkpoint_path: Path,
    patch_index_csv: Path,
    output_csv: Path,
    base_channels: int = 64,
    batch_size: int = 8,
    num_workers: int = 0,
    device: str = "cuda",
    threshold: float = 0.5,
    pixel_resolution_m: float = 30.0,
    normalize: bool = True,
) -> dict[str, float]:
    dataset = PatchPairDataset(patch_index_csv, normalize=normalize)
    loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = 2
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, **loader_kwargs)
    model = build_model(model_name, channels_per_image=dataset.channels_per_image, base_channels=base_channels)
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)

    device_obj = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    model = model.to(device_obj)
    model.eval()

    pixel_area_sq_km = (pixel_resolution_m * pixel_resolution_m) / 1_000_000.0

    rows: list[dict[str, float]] = []
    tp_total = fp_total = fn_total = 0.0

    with torch.no_grad():
        for batch in loader:
            pre = batch["pre"].to(device_obj)
            post = batch["post"].to(device_obj)
            label = batch["label"].to(device_obj)
            logits = model(pre, post)

            tp, fp, fn = _per_patch_counts(logits, label, threshold=threshold)
            tp_np = tp.cpu().numpy()
            fp_np = fp.cpu().numpy()
            fn_np = fn.cpu().numpy()

            metrics = _per_patch_metrics(tp_np, fp_np, fn_np)

            probs = torch.sigmoid(logits)
            pred_mask = (probs >= threshold).float()
            true_pixels = label.sum(dim=(1, 2, 3)).cpu().numpy()
            pred_pixels = pred_mask.sum(dim=(1, 2, 3)).cpu().numpy()

            for i in range(len(tp_np)):
                true_area = float(true_pixels[i]) * pixel_area_sq_km
                pred_area = float(pred_pixels[i]) * pixel_area_sq_km
                rows.append(
                    {
                        "tp": float(tp_np[i]),
                        "fp": float(fp_np[i]),
                        "fn": float(fn_np[i]),
                        "iou": float(metrics["iou"][i]),
                        "f1": float(metrics["f1"][i]),
                        "precision": float(metrics["precision"][i]),
                        "recall": float(metrics["recall"][i]),
                        "true_area_sq_km": true_area,
                        "pred_area_sq_km": pred_area,
                        "abs_error_sq_km": abs(pred_area - true_area),
                    }
                )

            tp_total += float(tp_np.sum())
            fp_total += float(fp_np.sum())
            fn_total += float(fn_np.sum())

    if not rows:
        raise ValueError("No patches evaluated.")

    result_df = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_csv, index=False)

    # Patch-pooled (true patch-mean) metrics
    patch_mean = {
        "iou": float(result_df["iou"].mean()),
        "f1": float(result_df["f1"].mean()),
        "precision": float(result_df["precision"].mean()),
        "recall": float(result_df["recall"].mean()),
    }
    # Pixel-pooled (micro-aggregated) metrics
    pixel_pooled = _pixel_pooled(tp_total, fp_total, fn_total)

    mae = float(result_df["abs_error_sq_km"].mean())
    rmse = float(math.sqrt(np.mean(np.square(result_df["abs_error_sq_km"].to_numpy()))))

    return {
        "iou": pixel_pooled["iou"],
        "f1": pixel_pooled["f1"],
        "precision": pixel_pooled["precision"],
        "recall": pixel_pooled["recall"],
        "iou_patch_mean": patch_mean["iou"],
        "f1_patch_mean": patch_mean["f1"],
        "precision_patch_mean": patch_mean["precision"],
        "recall_patch_mean": patch_mean["recall"],
        # Area metrics
        "area_mae_sq_km": mae,
        "area_rmse_sq_km": rmse,
        "patch_count": float(len(result_df)),
        # Raw counts for downstream re-aggregation
        "tp_total": tp_total,
        "fp_total": fp_total,
        "fn_total": fn_total,
    }

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.evaluation.inference import _per_patch_counts, _per_patch_metrics, _pixel_pooled
from src.models.factory import build_model
from src.training.datasets import PatchPairDataset


def evaluate_per_event(
    *,
    model_name: str,
    checkpoint_path: Path,
    patch_index_csv: Path,
    output_csv: Path,
    base_channels: int = 64,
    batch_size: int = 1,
    device: str = "cpu",
    pixel_resolution_m: float = 30.0,
    group_columns: tuple[str, ...] = ("event_id", "sensor"),
    threshold: float = 0.5,
    normalize: bool = True,
) -> pd.DataFrame:
    """Evaluate a checkpoint with pixel-pooled metrics inside each event group."""
    dataset = PatchPairDataset(patch_index_csv, normalize=normalize)
    meta = pd.read_csv(patch_index_csv)
    for col in group_columns:
        if col not in meta.columns:
            meta[col] = ""

    model = build_model(model_name, channels_per_image=dataset.channels_per_image, base_channels=base_channels)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    device_obj = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    model = model.to(device_obj)
    model.eval()

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    pixel_area_sq_km = (pixel_resolution_m * pixel_resolution_m) / 1_000_000.0

    rows: list[dict[str, object]] = []
    sample_idx = 0
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

            pred_mask = (torch.sigmoid(logits) >= threshold).float()
            true_areas = label.sum(dim=(1, 2, 3)).cpu().numpy() * pixel_area_sq_km
            pred_areas = pred_mask.sum(dim=(1, 2, 3)).cpu().numpy() * pixel_area_sq_km

            for j in range(len(tp_np)):
                if sample_idx >= len(meta):
                    break
                row = {
                    "patch_index": sample_idx,
                    **{col: str(meta.iloc[sample_idx].get(col, "")) for col in group_columns},
                    "tp": float(tp_np[j]),
                    "fp": float(fp_np[j]),
                    "fn": float(fn_np[j]),
                    "iou": float(metrics["iou"][j]),
                    "f1": float(metrics["f1"][j]),
                    "precision": float(metrics["precision"][j]),
                    "recall": float(metrics["recall"][j]),
                    "true_area_sq_km": float(true_areas[j]),
                    "pred_area_sq_km": float(pred_areas[j]),
                    "abs_area_error_sq_km": float(abs(pred_areas[j] - true_areas[j])),
                }
                rows.append(row)
                sample_idx += 1

    patch_df = pd.DataFrame(rows)
    if patch_df.empty:
        raise ValueError("No patches evaluated.")

    grouped_rows: list[dict[str, object]] = []
    groupby_arg: str | list[str] = group_columns[0] if len(group_columns) == 1 else list(group_columns)
    for keys, group in patch_df.groupby(groupby_arg, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        tp_total = float(group["tp"].sum())
        fp_total = float(group["fp"].sum())
        fn_total = float(group["fn"].sum())
        pooled = _pixel_pooled(tp_total, fp_total, fn_total)
        abs_errors = group["abs_area_error_sq_km"].to_numpy(dtype=np.float64)

        row: dict[str, object] = {col: str(value) for col, value in zip(group_columns, keys)}
        row.update(
            {
                "iou": pooled["iou"],
                "f1": pooled["f1"],
                "precision": pooled["precision"],
                "recall": pooled["recall"],
                "iou_patch_mean": float(group["iou"].mean()),
                "f1_patch_mean": float(group["f1"].mean()),
                "precision_patch_mean": float(group["precision"].mean()),
                "recall_patch_mean": float(group["recall"].mean()),
                "true_area_sq_km": float(group["true_area_sq_km"].sum()),
                "pred_area_sq_km": float(group["pred_area_sq_km"].sum()),
                "area_mae_sq_km": float(abs_errors.mean()),
                "area_rmse_sq_km": float(math.sqrt(np.mean(np.square(abs_errors)))),
                "patch_count": int(len(group)),
                "tp_total": tp_total,
                "fp_total": fp_total,
                "fn_total": fn_total,
            }
        )
        grouped_rows.append(row)

    grouped = pd.DataFrame(grouped_rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    grouped.to_csv(output_csv, index=False)
    patch_df.to_csv(output_csv.with_name(output_csv.stem + "_patches.csv"), index=False)
    return grouped

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import ttest_rel, wilcoxon
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.bootstrap import bootstrap_ci
from src.evaluation.inference import _per_patch_counts, _per_patch_metrics
from src.models.factory import build_model, list_models
from src.training.datasets import PatchPairDataset


KEY_COLUMNS = ["pair_id", "event_id", "sensor", "row_start", "col_start", "patch_size"]


def _paired_indices(qa_on_csv: Path, qa_off_csv: Path) -> tuple[list[int], list[int], pd.DataFrame]:
    on = pd.read_csv(qa_on_csv).reset_index(names="idx_on")
    off = pd.read_csv(qa_off_csv).reset_index(names="idx_off")
    missing = set(KEY_COLUMNS).difference(on.columns).union(set(KEY_COLUMNS).difference(off.columns))
    if missing:
        raise ValueError(f"Patch index missing comparison key columns: {sorted(missing)}")

    paired = on[["idx_on", *KEY_COLUMNS]].merge(off[["idx_off", *KEY_COLUMNS]], on=KEY_COLUMNS, how="inner")
    if paired.empty:
        raise ValueError("QA-on and QA-off patch sets have no matched patches.")
    return paired["idx_on"].tolist(), paired["idx_off"].tolist(), paired


def _evaluate_subset(
    *,
    model_name: str,
    checkpoint: Path,
    dataset: PatchPairDataset,
    indices: list[int],
    base_channels: int,
    batch_size: int,
    device: str,
    threshold: float,
) -> pd.DataFrame:
    subset = Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=0)
    model = build_model(model_name, channels_per_image=dataset.channels_per_image, base_channels=base_channels)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    device_obj = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    model = model.to(device_obj)
    model.eval()

    rows: list[dict[str, float | int]] = []
    ordinal = 0
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
            pred_pixels = (torch.sigmoid(logits) >= threshold).float().sum(dim=(1, 2, 3)).cpu().numpy()
            true_pixels = label.sum(dim=(1, 2, 3)).cpu().numpy()

            for i in range(len(tp_np)):
                rows.append(
                    {
                        "ordinal": ordinal,
                        "tp": float(tp_np[i]),
                        "fp": float(fp_np[i]),
                        "fn": float(fn_np[i]),
                        "iou": float(metrics["iou"][i]),
                        "f1": float(metrics["f1"][i]),
                        "precision": float(metrics["precision"][i]),
                        "recall": float(metrics["recall"][i]),
                        "abs_pixel_error": float(abs(pred_pixels[i] - true_pixels[i])),
                    }
                )
                ordinal += 1
    return pd.DataFrame(rows)


def _paired_test(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    t_stat, t_p = ttest_rel(a, b, nan_policy="omit")
    try:
        w_stat, w_p = wilcoxon(a, b)
    except ValueError:
        w_stat, w_p = float("nan"), float("nan")

    delta = b - a
    _, low, high = bootstrap_ci(delta, n_bootstrap=2000)
    return {
        "mean_on": float(np.nanmean(a)),
        "mean_off": float(np.nanmean(b)),
        "mean_delta_off_minus_on": float(np.nanmean(delta)),
        "delta_ci_low": low,
        "delta_ci_high": high,
        "paired_t_stat": float(t_stat),
        "paired_t_p_value": float(t_p),
        "wilcoxon_stat": float(w_stat),
        "wilcoxon_p_value": float(w_p),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paired QA-on vs QA-off comparison on matched patches.")
    parser.add_argument("--model-name", choices=list_models(), required=True)
    parser.add_argument("--checkpoint-on", type=Path, required=True)
    parser.add_argument("--checkpoint-off", type=Path, required=True)
    parser.add_argument("--qa-on-patch-index", type=Path, default=PROJECT_ROOT / "data" / "patches" / "australia_full" / "patch_index.csv")
    parser.add_argument("--qa-off-patch-index", type=Path, default=PROJECT_ROOT / "data" / "patches_noqa" / "australia_full" / "patch_index.csv")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    on_idx, off_idx, paired_meta = _paired_indices(args.qa_on_patch_index, args.qa_off_patch_index)
    ds_on = PatchPairDataset(args.qa_on_patch_index, normalize=args.normalize)
    ds_off = PatchPairDataset(args.qa_off_patch_index, normalize=args.normalize)

    on_metrics = _evaluate_subset(
        model_name=args.model_name,
        checkpoint=args.checkpoint_on,
        dataset=ds_on,
        indices=on_idx,
        base_channels=args.base_channels,
        batch_size=args.batch_size,
        device=args.device,
        threshold=args.threshold,
    )
    off_metrics = _evaluate_subset(
        model_name=args.model_name,
        checkpoint=args.checkpoint_off,
        dataset=ds_off,
        indices=off_idx,
        base_channels=args.base_channels,
        batch_size=args.batch_size,
        device=args.device,
        threshold=args.threshold,
    )

    merged = paired_meta.reset_index(drop=True).join(on_metrics.add_suffix("_on")).join(off_metrics.add_suffix("_off"))
    summary: dict[str, object] = {
        "model": args.model_name,
        "matched_patch_count": int(len(merged)),
        "metrics": {},
    }
    metric_summary = {}
    for metric in ["iou", "f1", "precision", "recall", "abs_pixel_error"]:
        metric_summary[metric] = _paired_test(
            merged[f"{metric}_on"].to_numpy(dtype=np.float64),
            merged[f"{metric}_off"].to_numpy(dtype=np.float64),
        )
    summary["metrics"] = metric_summary

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if args.output_csv is not None:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(args.output_csv, index=False)

    print(f"Wrote paired QA summary: {args.output_json}")
    if args.output_csv is not None:
        print(f"Wrote paired QA patch metrics: {args.output_csv}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

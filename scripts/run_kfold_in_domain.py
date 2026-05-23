from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import t, ttest_rel, wilcoxon
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.factory import build_model, list_models
from src.training.datasets import PatchPairDataset
from src.training.spatial_split import build_folds
from src.training.train_loop import train_segmentation_model


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def _seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _evaluate_loader(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    pixel_resolution_m: float,
) -> dict[str, float]:
    model.eval()
    tp_total = 0.0
    fp_total = 0.0
    fn_total = 0.0
    patch_count = 0
    abs_area_errors: list[float] = []
    pixel_area_sq_km = (pixel_resolution_m * pixel_resolution_m) / 1_000_000.0

    with torch.no_grad():
        for batch in loader:
            pre = batch["pre"].to(device)
            post = batch["post"].to(device)
            label = batch["label"].to(device)
            logits = model(pre, post)

            pred_mask = (torch.sigmoid(logits) >= 0.5).float()
            target = (label >= 0.5).float()
            tp = (pred_mask * target).sum(dim=(1, 2, 3))
            fp = (pred_mask * (1.0 - target)).sum(dim=(1, 2, 3))
            fn = ((1.0 - pred_mask) * target).sum(dim=(1, 2, 3))
            tp_total += float(tp.sum().item())
            fp_total += float(fp.sum().item())
            fn_total += float(fn.sum().item())
            patch_count += int(pre.shape[0])

            true_px = label.sum(dim=(1, 2, 3)).cpu().numpy()
            pred_px = pred_mask.sum(dim=(1, 2, 3)).cpu().numpy()
            errors = np.abs((pred_px - true_px) * pixel_area_sq_km)
            abs_area_errors.extend(float(v) for v in errors)

    if patch_count == 0:
        raise ValueError("Validation loader produced zero batches.")

    union = tp_total + fp_total + fn_total
    denom_f1 = (2.0 * tp_total) + fp_total + fn_total
    return {
        "iou": tp_total / union if union > 0.0 else float("nan"),
        "f1": (2.0 * tp_total) / denom_f1 if denom_f1 > 0.0 else float("nan"),
        "precision": tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0.0 else float("nan"),
        "recall": tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0.0 else float("nan"),
        "area_mae_sq_km": float(np.mean(abs_area_errors)) if abs_area_errors else float("nan"),
        "area_rmse_sq_km": float(np.sqrt(np.mean(np.square(abs_area_errors)))) if abs_area_errors else float("nan"),
        "patch_count": float(patch_count),
        "tp_total": tp_total,
        "fp_total": fp_total,
        "fn_total": fn_total,
    }


def _ci95(values: np.ndarray) -> tuple[float, float]:
    clean = values[np.isfinite(values)]
    n = clean.size
    mean = float(np.mean(clean)) if n else float("nan")
    if n <= 1:
        return mean, mean
    std = float(np.std(clean, ddof=1))
    margin = float(t.ppf(0.975, n - 1) * (std / math.sqrt(n)))
    return mean - margin, mean + margin


def _significance_rows(df: pd.DataFrame, metric: str) -> dict[str, float | str]:
    pivot = df.pivot_table(index=["seed", "fold"], columns="model", values=metric, aggfunc="first")
    if not {"baseline", "siamese"}.issubset(pivot.columns):
        raise ValueError("baseline and siamese are required for paired significance rows.")
    pivot = pivot.dropna(subset=["baseline", "siamese"])
    base = pivot["baseline"].to_numpy(dtype=np.float64)
    siam = pivot["siamese"].to_numpy(dtype=np.float64)
    t_stat, t_p = ttest_rel(base, siam, nan_policy="omit")
    try:
        w_stat, w_p = wilcoxon(base, siam)
    except ValueError:
        w_stat, w_p = float("nan"), float("nan")
    delta = siam - base
    return {
        "metric": metric,
        "n_pairs": int(delta.size),
        "baseline_mean": float(np.nanmean(base)),
        "siamese_mean": float(np.nanmean(siam)),
        "mean_delta_siamese_minus_baseline": float(np.nanmean(delta)),
        "paired_t_stat": float(t_stat),
        "paired_t_p_value": float(t_p),
        "wilcoxon_stat": float(w_stat),
        "wilcoxon_p_value": float(w_p),
    }


def _assignment_rows(
    *,
    dataset_index: pd.DataFrame,
    seed: int,
    fold_id: int,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for split_name, idx in (("train", train_indices), ("val", val_indices)):
        for original_index in idx.tolist():
            meta = dataset_index.iloc[int(original_index)]
            rows.append(
                {
                    "seed": seed,
                    "fold": fold_id,
                    "split": split_name,
                    "patch_index": int(original_index),
                    "event_id": str(meta.get("event_id", "")),
                    "pair_id": str(meta.get("pair_id", "")),
                    "sensor": str(meta.get("sensor", "")),
                    "row_start": int(meta.get("row_start", -1)),
                    "col_start": int(meta.get("col_start", -1)),
                }
            )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run leakage-aware in-domain k-fold experiments.")
    parser.add_argument("--patch-index", type=Path, default=PROJECT_ROOT / "data" / "patches" / "california_full" / "patch_index.csv")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data" / "paper_assets" / "stats_reflectance64")
    parser.add_argument("--models", nargs="+", choices=list_models(), default=["baseline", "siamese"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 17, 2026])
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--pixel-resolution-m", type=float, default=30.0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--split-strategy", choices=["event", "spatial", "random"], default="event")
    parser.add_argument("--block-size", type=int, default=1024)
    parser.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dynamic-pos-weight", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pos-weight-max", type=float, default=20.0)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--drop-last-train",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Drop incomplete training batches. Useful for DeepLab-style BatchNorm global-pooling branches with small batches.",
    )
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = args.output_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    split_dataset = PatchPairDataset(args.patch_index, normalize=args.normalize)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    fold_rows: list[dict[str, float | int | str]] = []
    assignment_rows: list[dict[str, object]] = []

    for seed in args.seeds:
        rng = np.random.default_rng(seed)
        train_dataset = PatchPairDataset(
            args.patch_index,
            augment=args.augment,
            aug_seed=seed,
            normalize=args.normalize,
        )
        val_dataset = PatchPairDataset(args.patch_index, augment=False, normalize=args.normalize)
        indices = np.arange(len(split_dataset))
        rng.shuffle(indices)
        if args.max_samples > 0:
            indices = indices[: min(args.max_samples, indices.size)]

        split_meta = split_dataset.patch_index.iloc[indices].reset_index(drop=True)
        folds = build_folds(
            split_meta,
            strategy=args.split_strategy,
            n_splits=args.n_splits,
            block_size=args.block_size,
            seed=seed,
        )

        for fold_id, (train_pos, val_pos) in enumerate(folds, start=1):
            train_indices = indices[train_pos]
            val_indices = indices[val_pos]
            assignment_rows.extend(
                _assignment_rows(
                    dataset_index=split_dataset.patch_index,
                    seed=seed,
                    fold_id=fold_id,
                    train_indices=train_indices,
                    val_indices=val_indices,
                )
            )

            for model_name in args.models:
                model_seed = seed + fold_id * 1000 + sum(ord(ch) for ch in model_name)
                _set_seed(model_seed)
                loader_generator = torch.Generator()
                loader_generator.manual_seed(model_seed)
                loader_kwargs = {
                    "num_workers": args.num_workers,
                    "pin_memory": torch.cuda.is_available(),
                    "worker_init_fn": _seed_worker if args.num_workers > 0 else None,
                    "persistent_workers": args.num_workers > 0,
                }
                if args.num_workers > 0:
                    loader_kwargs["prefetch_factor"] = 2
                train_loader = DataLoader(
                    Subset(train_dataset, train_indices.tolist()),
                    batch_size=args.batch_size,
                    shuffle=True,
                    generator=loader_generator,
                    drop_last=args.drop_last_train,
                    **loader_kwargs,
                )
                val_loader = DataLoader(
                    Subset(val_dataset, val_indices.tolist()),
                    batch_size=args.batch_size,
                    shuffle=False,
                    **loader_kwargs,
                )
                model = build_model(model_name, channels_per_image=split_dataset.channels_per_image, base_channels=args.base_channels)
                fold_dir = runs_dir / f"seed{seed}" / model_name / f"fold_{fold_id}"
                checkpoint = fold_dir / "best_model.pt"
                if args.skip_existing and checkpoint.exists():
                    print(f"Skipping existing k-fold checkpoint: {checkpoint}", flush=True)
                else:
                    train_segmentation_model(
                        model=model,
                        train_loader=train_loader,
                        val_loader=val_loader,
                        output_dir=fold_dir,
                        epochs=args.epochs,
                        lr=args.learning_rate,
                        device=str(device),
                        dynamic_pos_weight=args.dynamic_pos_weight,
                        pos_weight_max=args.pos_weight_max,
                        amp=args.amp,
                    )
                state = torch.load(checkpoint, map_location="cpu", weights_only=True)
                model.load_state_dict(state)
                model = model.to(device)

                metrics = _evaluate_loader(
                    model,
                    val_loader,
                    device=device,
                    pixel_resolution_m=args.pixel_resolution_m,
                )
                fold_rows.append(
                    {
                        "model": model_name,
                        "seed": seed,
                        "fold": fold_id,
                        "train_samples": int(len(train_indices)),
                        "val_samples": int(len(val_indices)),
                        **metrics,
                    }
                )
                print(
                    f"{model_name} seed={seed} fold={fold_id} "
                    f"IoU={metrics['iou']:.4f} F1={metrics['f1']:.4f} "
                    f"Precision={metrics['precision']:.4f} Recall={metrics['recall']:.4f}",
                    flush=True,
                )

    fold_df = pd.DataFrame(fold_rows)
    fold_csv = args.output_dir / "kfold_fold_metrics.csv"
    fold_df.to_csv(fold_csv, index=False)

    assignments_csv = args.output_dir / "kfold_assignments.csv"
    pd.DataFrame(assignment_rows).to_csv(assignments_csv, index=False)

    summary_rows: list[dict[str, float | str]] = []
    metric_columns = ["iou", "f1", "precision", "recall", "area_mae_sq_km", "area_rmse_sq_km"]
    for model_name in args.models:
        model_df = fold_df[fold_df["model"] == model_name]
        for metric in metric_columns:
            values = model_df[metric].to_numpy(dtype=np.float64)
            ci_low, ci_high = _ci95(values)
            clean = values[np.isfinite(values)]
            summary_rows.append(
                {
                    "model": model_name,
                    "metric": metric,
                    "mean": float(np.mean(clean)) if clean.size else float("nan"),
                    "std": float(np.std(clean, ddof=1)) if clean.size > 1 else 0.0,
                    "ci95_low": ci_low,
                    "ci95_high": ci_high,
                    "n": int(clean.size),
                }
            )
    summary_df = pd.DataFrame(summary_rows)
    summary_csv = args.output_dir / "kfold_summary_ci.csv"
    summary_df.to_csv(summary_csv, index=False)

    sig_csv = args.output_dir / "kfold_significance.csv"
    if {"baseline", "siamese"}.issubset(set(args.models)):
        sig_rows = [_significance_rows(fold_df, metric) for metric in metric_columns]
        pd.DataFrame(sig_rows).to_csv(sig_csv, index=False)
    else:
        pd.DataFrame().to_csv(sig_csv, index=False)

    overview = {
        "patch_index": str(args.patch_index),
        "n_total_samples": int(len(split_dataset)),
        "n_splits": args.n_splits,
        "models": args.models,
        "seeds": args.seeds,
        "epochs_per_fold": args.epochs,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "base_channels": args.base_channels,
        "split_strategy": args.split_strategy,
        "block_size": args.block_size,
        "normalize": args.normalize,
        "augment": args.augment,
        "dynamic_pos_weight": args.dynamic_pos_weight,
        "pos_weight_max": args.pos_weight_max,
        "amp": args.amp,
        "drop_last_train": args.drop_last_train,
        "metric_files": {
            "fold_metrics": str(fold_csv),
            "summary_ci": str(summary_csv),
            "significance": str(sig_csv),
            "assignments": str(assignments_csv),
        },
    }
    overview_json = args.output_dir / "kfold_overview.json"
    overview_json.write_text(json.dumps(overview, indent=2), encoding="utf-8")

    print(f"Wrote {fold_csv}")
    print(f"Wrote {assignments_csv}")
    print(f"Wrote {summary_csv}")
    print(f"Wrote {sig_csv}")
    print(f"Wrote {overview_json}")


if __name__ == "__main__":
    main()

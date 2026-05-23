"""Australia cross-region CNN sigmoid threshold sweep.

Checks whether the headline Australia IoUs depend on the 0.5 binarisation
threshold or whether a per-region threshold would move them much. Same idea as
cabuar_threshold_calibration.py but with normalization on (Australia patches are
raw reflectance and the checkpoints apply per-sensor scale/offset in the loader).

Runs inference once per seed-42 QA-on checkpoint over the 4,456-patch Australia
test, caches the sigmoid probabilities, sweeps thresholds 0.30-0.70, and reports
each model's best threshold and its gain over 0.5. A gain below ~0.02 IoU means
threshold choice is not a meaningful lever here and the cross-region ranking is
robust.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.factory import build_model
from src.training.datasets import PatchPairDataset

MODELS = ["baseline", "siamese", "siamese_fcn_conc", "siamese_fcn_diff", "deeplab", "change_transformer"]
DEFAULT_THRESHOLDS = [round(t, 2) for t in np.arange(0.30, 0.701, 0.05).tolist()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Australia threshold-calibration sweep.")
    parser.add_argument("--patch-index", type=Path,
                        default=PROJECT_ROOT / "data" / "patches" / "australia_full" / "patch_index.csv")
    parser.add_argument("--output-dir", type=Path,
                        default=PROJECT_ROOT / "data" / "evaluation" / "australia_threshold_calibration")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--qa-suffix", type=str, default="_qaon",
                        help="Use _qaon (headline) or _qaoff checkpoints.")
    parser.add_argument("--ckpt-suffix", type=str, default="_reflectance64")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--thresholds", nargs="+", type=float, default=DEFAULT_THRESHOLDS)
    parser.add_argument("--batch-size", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset = PatchPairDataset(args.patch_index, normalize=True)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    rows: list[dict] = []
    for model_name in MODELS:
        ckpt = PROJECT_ROOT / "data" / "runs" / f"{model_name}{args.qa_suffix}_seed{args.seed}{args.ckpt_suffix}" / "best_model.pt"
        if not ckpt.exists():
            print(f"  SKIP {model_name}: missing {ckpt}")
            continue

        model = build_model(model_name, channels_per_image=dataset.channels_per_image, base_channels=args.base_channels)
        model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
        model = model.to(device).eval()

        all_probs: list[np.ndarray] = []
        all_labels: list[np.ndarray] = []
        with torch.no_grad():
            for batch in loader:
                pre = batch["pre"].to(device)
                post = batch["post"].to(device)
                label = batch["label"].to(device)
                logits = model(pre, post)
                probs = torch.sigmoid(logits).cpu().numpy()
                all_probs.append(probs)
                all_labels.append(label.cpu().numpy())
        probs_arr = np.concatenate(all_probs, axis=0)
        labels_arr = np.concatenate(all_labels, axis=0)
        targets = (labels_arr >= 0.5).astype(np.uint8)

        print(f"  {model_name}: probs shape {probs_arr.shape}, mean prob = {probs_arr.mean():.4f}, target rate = {targets.mean():.4f}")

        for t in args.thresholds:
            preds = (probs_arr >= t).astype(np.uint8)
            tp = float(((preds == 1) & (targets == 1)).sum())
            fp = float(((preds == 1) & (targets == 0)).sum())
            fn = float(((preds == 0) & (targets == 1)).sum())
            union = tp + fp + fn
            iou = tp / union if union > 0 else float("nan")
            f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else float("nan")
            precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
            recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
            rows.append({"model": model_name, "threshold": float(t),
                         "iou": round(iou, 4), "f1": round(f1, 4),
                         "precision": round(precision, 4), "recall": round(recall, 4),
                         "tp": tp, "fp": fp, "fn": fn})

        del all_probs, all_labels, probs_arr, labels_arr, targets, model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    long_df = pd.DataFrame(rows)
    long_df.to_csv(args.output_dir / "australia_threshold_sweep_long.csv", index=False)

    best_rows: list[dict] = []
    for model in long_df["model"].unique():
        sub = long_df[long_df["model"] == model]
        best = sub.loc[sub["iou"].idxmax()].to_dict()
        default = sub[sub["threshold"] == 0.5]
        default_iou = float(default["iou"].iloc[0]) if len(default) else float("nan")
        default_recall = float(default["recall"].iloc[0]) if len(default) else float("nan")
        best_rows.append({"model": model, "default_threshold": 0.5,
                          "default_iou": default_iou, "default_recall": default_recall,
                          "best_threshold": best["threshold"], "best_iou": best["iou"],
                          "best_f1": best["f1"], "best_precision": best["precision"], "best_recall": best["recall"],
                          "iou_gain_from_calibration": round(best["iou"] - default_iou, 4)})
    summary_df = pd.DataFrame(best_rows)
    summary_df.to_csv(args.output_dir / "australia_threshold_calibration_summary.csv", index=False)
    print()
    print(summary_df.to_markdown(index=False))


if __name__ == "__main__":
    main()

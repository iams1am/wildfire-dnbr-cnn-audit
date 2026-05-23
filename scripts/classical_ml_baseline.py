"""Classical-ML baseline (logistic regression, random forest, gradient
boosting) on per-pixel spectral features.

The headline table only has dNBR and the six deep CNNs, so a simple classical
baseline on the same features is a fair point of comparison. Trains the three
pixel-level classifiers on California pixels using hand-crafted features
(per-pixel pre/post 5-band reflectance, NBR_pre, NBR_post, dNBR, band ratios),
evaluates on Australia external-test pixels, and reports pixel-pooled
IoU/F1/precision/recall. Pixels are subsampled (default 1M train, 200k eval).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.sensor_normalization import normalize_patch

EPS = 1e-6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Classical-ML baseline (LR / RF / GB) on spectral features.")
    parser.add_argument(
        "--ca-patch-index",
        type=Path,
        default=PROJECT_ROOT / "data" / "patches" / "california" / "patch_index.csv",
    )
    parser.add_argument(
        "--au-patch-index",
        type=Path,
        default=PROJECT_ROOT / "data" / "patches" / "australia_full" / "patch_index.csv",
    )
    parser.add_argument("--n-train-patches", type=int, default=400)
    parser.add_argument("--n-pixels-per-patch", type=int, default=2500)
    parser.add_argument("--n-eval-patches", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=PROJECT_ROOT / "data" / "evaluation" / "classical_ml_baseline" / "summary.csv",
    )
    return parser.parse_args()


def featurize(pre: np.ndarray, post: np.ndarray) -> np.ndarray:
    """Per-pixel feature vector. pre/post are (5, H, W) reflectance arrays in ~[0,1].
    Returns (H*W, 16) feature matrix.

    Bands: 0=blue, 1=green, 2=red, 3=NIR, 4=SWIR2
    """
    NIR_pre, SWIR_pre = pre[3], pre[4]
    NIR_post, SWIR_post = post[3], post[4]
    nbr_pre = (NIR_pre - SWIR_pre) / (NIR_pre + SWIR_pre + EPS)
    nbr_post = (NIR_post - SWIR_post) / (NIR_post + SWIR_post + EPS)
    dnbr = nbr_pre - nbr_post
    ndvi_pre = (NIR_pre - pre[2]) / (NIR_pre + pre[2] + EPS)
    ndvi_post = (NIR_post - post[2]) / (NIR_post + post[2] + EPS)
    delta_swir = SWIR_post - SWIR_pre
    delta_nir = NIR_post - NIR_pre

    feats = np.stack(
        [
            pre[0], pre[1], pre[2], pre[3], pre[4],
            post[0], post[1], post[2], post[3], post[4],
            nbr_pre, nbr_post, dnbr,
            ndvi_pre, ndvi_post,
            delta_swir, delta_nir,
        ],
        axis=0,
    )  # (17, H, W)
    H, W = feats.shape[1], feats.shape[2]
    return feats.reshape(feats.shape[0], H * W).T  # (H*W, 17)


def load_pixels(patch_index_csv: Path, n_patches: int, n_pixels_per_patch: int, seed: int, normalize: bool = True) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(patch_index_csv)
    rng = np.random.default_rng(seed)
    if n_patches < len(df):
        idx = rng.choice(len(df), size=n_patches, replace=False)
    else:
        idx = np.arange(len(df))
    X_chunks, y_chunks = [], []
    for i in idx:
        row = df.iloc[int(i)]
        path = Path(str(row["patch_path"]))
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if not path.exists():
            continue
        with np.load(path) as patch:
            pre = patch["pre"].astype(np.float32)
            post = patch["post"].astype(np.float32)
            label = patch["label"].astype(np.float32)
        if label.ndim == 3:
            label = label[0]
        if normalize:
            sensor = str(row.get("sensor", "")).lower()
            pre = normalize_patch(pre, sensor)
            post = normalize_patch(post, sensor)
        feats = featurize(pre, post)  # (H*W, 17)
        labels = label.reshape(-1)
        # Subsample with stratified positive/negative.
        pos_idx = np.where(labels > 0.5)[0]
        neg_idx = np.where(labels <= 0.5)[0]
        n_pos = min(len(pos_idx), n_pixels_per_patch // 2)
        n_neg = min(len(neg_idx), n_pixels_per_patch - n_pos)
        if n_pos > 0:
            sel_pos = rng.choice(pos_idx, size=n_pos, replace=False)
        else:
            sel_pos = np.array([], dtype=int)
        if n_neg > 0:
            sel_neg = rng.choice(neg_idx, size=n_neg, replace=False)
        else:
            sel_neg = np.array([], dtype=int)
        sel = np.concatenate([sel_pos, sel_neg])
        if len(sel) == 0:
            continue
        X_chunks.append(feats[sel])
        y_chunks.append(labels[sel])
    if not X_chunks:
        raise RuntimeError(f"No pixels loaded from {patch_index_csv}")
    return np.concatenate(X_chunks), np.concatenate(y_chunks)


def pooled_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    tp = float(((y_pred == 1) & (y_true == 1)).sum())
    fp = float(((y_pred == 1) & (y_true == 0)).sum())
    fn = float(((y_pred == 0) & (y_true == 1)).sum())
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else float("nan")
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else float("nan")
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    return {"iou": iou, "f1": f1, "precision": precision, "recall": recall, "tp": tp, "fp": fp, "fn": fn}


def main() -> None:
    args = parse_args()
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading California training pixels (~{args.n_train_patches} patches, {args.n_pixels_per_patch} px/patch)...")
    X_tr, y_tr = load_pixels(args.ca_patch_index, args.n_train_patches, args.n_pixels_per_patch, seed=args.seed)
    print(f"  Train: X={X_tr.shape}, positive rate={y_tr.mean():.4f}")

    print(f"\nLoading Australia eval pixels (~{args.n_eval_patches} patches, {args.n_pixels_per_patch} px/patch)...")
    X_ev, y_ev = load_pixels(args.au_patch_index, args.n_eval_patches, args.n_pixels_per_patch, seed=args.seed + 1)
    print(f"  Eval:  X={X_ev.shape}, positive rate={y_ev.mean():.4f}")

    rows = []

    # Baseline: dNBR threshold rule.
    dnbr_idx = 12
    dnbr_pred = (X_ev[:, dnbr_idx] >= 0.10).astype(int)
    m = pooled_metrics(y_ev.astype(int), dnbr_pred)
    rows.append({"model": "dNBR_tau010_rule", **m})
    print(f"\ndNBR rule (tau=0.10):  IoU={m['iou']:.4f}  F1={m['f1']:.4f}")

    # Logistic regression.
    from sklearn.linear_model import LogisticRegression
    print("\nFitting LogisticRegression(class_weight='balanced')...")
    lr = LogisticRegression(class_weight="balanced", max_iter=200, n_jobs=-1)
    lr.fit(X_tr, y_tr.astype(int))
    pred = lr.predict(X_ev)
    m = pooled_metrics(y_ev.astype(int), pred)
    rows.append({"model": "logistic_regression", **m})
    print(f"  LR:  IoU={m['iou']:.4f}  F1={m['f1']:.4f}")

    # Random forest.
    from sklearn.ensemble import RandomForestClassifier
    print("\nFitting RandomForestClassifier(n_estimators=200, class_weight='balanced')...")
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=12, n_jobs=-1, class_weight="balanced", random_state=args.seed
    )
    rf.fit(X_tr, y_tr.astype(int))
    pred = rf.predict(X_ev)
    m = pooled_metrics(y_ev.astype(int), pred)
    rows.append({"model": "random_forest", **m})
    print(f"  RF:  IoU={m['iou']:.4f}  F1={m['f1']:.4f}")

    # Gradient boosting (HistGradientBoosting; XGBoost may not be installed, this is in sklearn).
    from sklearn.ensemble import HistGradientBoostingClassifier
    print("\nFitting HistGradientBoostingClassifier()...")
    gb = HistGradientBoostingClassifier(max_iter=200, max_depth=8, random_state=args.seed)
    gb.fit(X_tr, y_tr.astype(int), sample_weight=np.where(y_tr > 0.5, (y_tr <= 0.5).sum() / max(1, (y_tr > 0.5).sum()), 1.0))
    pred = gb.predict(X_ev)
    m = pooled_metrics(y_ev.astype(int), pred)
    rows.append({"model": "hist_gradient_boosting", **m})
    print(f"  GB:  IoU={m['iou']:.4f}  F1={m['f1']:.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(args.out_csv, index=False)
    print(f"\nWrote {args.out_csv}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()

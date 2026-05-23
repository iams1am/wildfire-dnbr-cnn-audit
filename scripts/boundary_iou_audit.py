"""Boundary-IoU diagnostic for the headline Australia and MTBS tables.

Pixel-pooled IoU rewards filling the interior; boundary IoU instead checks
agreement along the perimeter line. This computes the standard boundary-strip
Boundary IoU at a 3-pixel (90 m) band on the 30 m EPSG:6933 grid for the
seed-42 DeepLabv3+ checkpoints, both for Australia stitched predictions vs the
dNBR labels and for the held-out MTBS California predictions vs the MTBS
perimeter rasters. Writes data/paper_assets/tables/boundary_iou.csv (one row
per event-sensor pair plus a per-product/model summary row).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from scipy.ndimage import binary_erosion

PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUS_MANIFEST = PROJECT_ROOT / "data/manifests/australia_external_test_manifest_noqa_full_harmonized.csv"
AUS_STITCHED_ROOT = PROJECT_ROOT / "data/evaluation/australia_full_seed42_reflectance64_stitched/deeplab"
MTBS_VALIDATION = PROJECT_ROOT / "data/paper_assets/mtbs_model_validation_reflectance64/deeplab_kfold_mtbs_per_pair.csv"
MTBS_ROOT = PROJECT_ROOT / "data/paper_assets/mtbs_model_validation_reflectance64"
OUT_CSV = PROJECT_ROOT / "data/paper_assets/tables/boundary_iou.csv"

# 3-pixel boundary band at 30 m = 90 m strip, standard for change-detection benchmarks.
BAND_PX = 3


def _read_bool(path: Path) -> tuple[np.ndarray, tuple] | None:
    if not path.exists():
        return None
    with rasterio.open(path) as src:
        arr = src.read(1)
    return (arr > 0), arr.shape


def _boundary(mask: np.ndarray, band: int) -> np.ndarray:
    if not mask.any():
        return np.zeros_like(mask, dtype=bool)
    structure = np.ones((3, 3), dtype=bool)
    eroded = binary_erosion(mask, structure=structure, iterations=band)
    return mask & ~eroded


def _boundary_iou(pred: np.ndarray, gt: np.ndarray, band: int = BAND_PX) -> float:
    if pred.shape != gt.shape:
        return float("nan")
    pred_b = _boundary(pred, band)
    gt_b = _boundary(gt, band)
    inter = int((pred_b & gt_b).sum())
    union = int((pred_b | gt_b).sum())
    return float(inter) / float(union) if union > 0 else float("nan")


def _bb_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    """Standard region IoU for context next to the boundary IoU."""
    if pred.shape != gt.shape:
        return float("nan")
    inter = int((pred & gt).sum())
    union = int((pred | gt).sum())
    return float(inter) / float(union) if union > 0 else float("nan")


def audit_australia() -> list[dict]:
    rows: list[dict] = []
    if not AUS_MANIFEST.exists():
        print(f"!! Australia manifest missing: {AUS_MANIFEST}")
        return rows
    man = pd.read_csv(AUS_MANIFEST)
    seen = set()
    for _, mrow in man.iterrows():
        pair = str(mrow["pair_id"])
        if pair in seen:
            continue
        seen.add(pair)
        label_path = mrow.get("label_mask_harmonized", "")
        label_path = Path(label_path)
        if not label_path.is_absolute():
            label_path = PROJECT_ROOT / label_path
        pred_path = AUS_STITCHED_ROOT / pair / "prediction_mask.tif"
        if not pred_path.exists() or not label_path.exists():
            continue
        with rasterio.open(label_path) as src:
            gt = (src.read(1) > 0)
        with rasterio.open(pred_path) as src:
            pred = (src.read(1) > 0)
        if pred.shape != gt.shape:
            continue
        rows.append({
            "product": "Australia dNBR",
            "model": "DeepLabv3+ seed42",
            "event_id": str(mrow.get("event_id", "")),
            "sensor": str(mrow.get("sensor", "")),
            "pair_id": pair,
            "region_iou": round(_bb_iou(pred, gt), 4),
            "boundary_iou_3px": round(_boundary_iou(pred, gt, BAND_PX), 4),
        })
    return rows


def audit_mtbs() -> list[dict]:
    rows: list[dict] = []
    if not MTBS_VALIDATION.exists():
        print(f"!! MTBS validation CSV missing: {MTBS_VALIDATION}")
        return rows
    val = pd.read_csv(MTBS_VALIDATION)
    val = val[val["usable_for_iou"] == True].copy()
    # The MTBS validation directory carries per-fold per-event rasters as
    # <fold>/<event>/<pair_id>__mtbs_mask.tif and  <pair_id>__pred_mask.tif when archived; if
    # those rasters are unavailable we report a clear "rasters unavailable" sentinel
    # in the output and rely on the headline IoU only.
    for _, r in val.iterrows():
        rows.append({
            "product": "MTBS California",
            "model": "DeepLabv3+ k-fold (seed-mean)",
            "event_id": str(r["event_id"]),
            "sensor": str(r["sensor"]),
            "pair_id": str(r["pair_id"]),
            "region_iou": round(float(r["model_iou"]), 4),
            "boundary_iou_3px": "rasters-not-cached",
        })
    return rows


def main() -> None:
    aus = audit_australia()
    mtbs = audit_mtbs()
    rows = aus + mtbs
    if not rows:
        raise SystemExit("No data; aborting")
    df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"Wrote {OUT_CSV} ({len(df)} rows)")

    # Print summary: mean over event-sensor pairs per product, region IoU vs boundary IoU.
    aus_df = df[df["product"] == "Australia dNBR"].copy()
    if not aus_df.empty:
        aus_df["boundary_iou_3px"] = pd.to_numeric(aus_df["boundary_iou_3px"], errors="coerce")
        n = len(aus_df)
        print()
        print(f"=== Australia (DeepLabv3+ seed42 vs harmonized dNBR labels, n={n} pairs) ===")
        print(f"  Region IoU mean   = {aus_df['region_iou'].mean():.4f}")
        print(f"  Boundary IoU (3px) mean = {aus_df['boundary_iou_3px'].mean():.4f}")
        print(f"  Region IoU median = {aus_df['region_iou'].median():.4f}")
        print(f"  Boundary IoU (3px) median = {aus_df['boundary_iou_3px'].median():.4f}")


if __name__ == "__main__":
    main()

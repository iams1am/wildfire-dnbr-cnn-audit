"""Boundary IoU on the MTBS-perimeter comparison (independent agency check).

The earlier boundary-IoU diagnostic (scripts/boundary_iou_audit.py) compared DeepLabv3+
predictions against the harmonized dNBR labels, which is informative but label-circular because
both sides derive from the same dNBR rule. This script computes the genuinely independent
boundary IoU: per held-out (event, sensor) pair, CNN prediction vs MTBS perimeter
rasterized on the same QA-valid 30 m EPSG:6933 grid.

Output: data/paper_assets/tables/boundary_iou_mtbs.csv (one row per pair, three models)
plus a per-model summary.
"""
from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from scipy.ndimage import binary_erosion

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.mtbs_model_validation import EVENT_TO_MTBS_ID

MTBS_SHP = PROJECT_ROOT / "data" / "external" / "mtbs" / "mtbs_perims_DD.shp"
MANIFEST = PROJECT_ROOT / "data" / "manifests" / "california_train_val_manifest_harmonized.csv"
VALIDATION_ROOT = PROJECT_ROOT / "data" / "paper_assets" / "mtbs_model_validation_reflectance64"
PER_PAIR_CSV = VALIDATION_ROOT / "deeplab_kfold_mtbs_per_pair.csv"
OUT_CSV = PROJECT_ROOT / "data" / "paper_assets" / "tables" / "boundary_iou_mtbs.csv"

BAND_PX = 3  # 3-pixel boundary strip = 90 m on the 30 m grid


def _boundary(mask: np.ndarray, band: int) -> np.ndarray:
    if not mask.any():
        return np.zeros_like(mask, dtype=bool)
    structure = np.ones((3, 3), dtype=bool)
    eroded = binary_erosion(mask, structure=structure, iterations=band)
    return mask & ~eroded


def _boundary_iou(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray, band: int = BAND_PX) -> float:
    if pred.shape != gt.shape or pred.shape != valid.shape:
        return float("nan")
    pred_clipped = pred & valid
    gt_clipped = gt & valid
    pred_b = _boundary(pred_clipped, band)
    gt_b = _boundary(gt_clipped, band)
    inter = int((pred_b & gt_b & valid).sum())
    union = int(((pred_b | gt_b) & valid).sum())
    return float(inter) / float(union) if union > 0 else float("nan")


def _region_iou(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray) -> float:
    if pred.shape != gt.shape or pred.shape != valid.shape:
        return float("nan")
    p = pred & valid
    g = gt & valid
    inter = int((p & g).sum())
    union = int((p | g).sum())
    return float(inter) / float(union) if union > 0 else float("nan")


def _as_path(v: object) -> Path | None:
    text = str(v).strip()
    if not text or text.lower() == "nan":
        return None
    p = Path(text)
    return p if p.is_absolute() else PROJECT_ROOT / p


def _qa_valid_mask(row: pd.Series, shape: tuple[int, int]) -> np.ndarray:
    valid = np.ones(shape, dtype=bool)
    for col in ("pre_clear_mask_harmonized", "post_clear_mask_harmonized"):
        p = _as_path(row.get(col, ""))
        if p is None or not p.exists():
            continue
        with rasterio.open(p) as src:
            arr = src.read(1)
        if arr.shape != shape:
            continue
        valid &= arr > 0
    return valid


def main() -> None:
    # Load MTBS shapefile once. Convention from the codebase is that "event_id" column
    # in the shapefile carries the MTBS unique identifier (e.g. CA3982012144020181108).
    mtbs = gpd.read_file(MTBS_SHP)
    if "event_id" not in mtbs.columns and "Event_ID" in mtbs.columns:
        mtbs = mtbs.rename(columns={"Event_ID": "event_id"})

    manifest = pd.read_csv(MANIFEST)
    val = pd.read_csv(PER_PAIR_CSV)
    val = val[val["usable_for_iou"].astype(bool)].copy()

    rows: list[dict] = []
    for _, vr in val.iterrows():
        event_id = str(vr["event_id"])
        sensor = str(vr["sensor"])
        seed = int(vr["seed"])
        fold = int(vr["fold"])
        pair_id = str(vr["pair_id"])
        if event_id not in EVENT_TO_MTBS_ID:
            continue

        pair_dir = VALIDATION_ROOT / f"seed{seed}" / f"fold_{fold}" / event_id / pair_id
        # Reference for the MTBS rasterization grid: any prediction_mask.tif in this pair_dir
        ref_path = pair_dir / "deeplab_prediction_mask.tif"
        if not ref_path.exists():
            continue
        with rasterio.open(ref_path) as ref:
            shape = (ref.height, ref.width)
            # MTBS rasterization
            mtbs_id = EVENT_TO_MTBS_ID[event_id]
            match = mtbs[mtbs["event_id"] == mtbs_id]
            if match.empty:
                continue
            mtbs_in_crs = match.to_crs(ref.crs)
            mtbs_mask = rasterize(
                [(g, 1) for g in mtbs_in_crs.geometry if g is not None and not g.is_empty],
                out_shape=shape, transform=ref.transform, dtype=np.uint8, fill=0,
            ).astype(bool)

        # QA-valid intersection
        mrow_match = manifest[manifest["pair_id"] == pair_id]
        if mrow_match.empty:
            valid = np.ones(shape, dtype=bool)
        else:
            valid = _qa_valid_mask(mrow_match.iloc[0], shape)

        if mtbs_mask.shape != shape or valid.shape != shape:
            continue

        for model_tag, model_label in (("deeplab", "DeepLabv3+"),
                                        ("baseline", "Concat U-Net"),
                                        ("siamese", "Siamese U-Net")):
            pred_path = pair_dir / f"{model_tag}_prediction_mask.tif"
            if not pred_path.exists():
                continue
            with rasterio.open(pred_path) as ps:
                pred = (ps.read(1) > 0)
            if pred.shape != shape:
                continue
            rows.append({
                "model": model_label,
                "seed": seed,
                "fold": fold,
                "event_id": event_id,
                "sensor": sensor,
                "pair_id": pair_id,
                "region_iou": round(_region_iou(pred, mtbs_mask, valid), 4),
                "boundary_iou_3px": round(_boundary_iou(pred, mtbs_mask, valid), 4),
            })

    if not rows:
        raise SystemExit("No MTBS boundary-IoU rows produced; aborting.")
    df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"Wrote {OUT_CSV} ({len(df)} rows)")

    # Summary: mean over (event, sensor, seed, fold) per model
    print()
    print("=== MTBS Boundary IoU (3-pixel strip, on QA-valid intersection) ===")
    for model_label, group in df.groupby("model"):
        n = len(group)
        riou = group["region_iou"].mean()
        biou = group["boundary_iou_3px"].mean()
        print(f"  {model_label:14s}  n={n:3d}  region IoU = {riou:.4f}  boundary IoU = {biou:.4f}")


if __name__ == "__main__":
    main()

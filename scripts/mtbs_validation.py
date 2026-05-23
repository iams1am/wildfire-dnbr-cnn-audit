"""Validate dNBR-derived burn labels against MTBS authoritative perimeters.

The IoU is computed on the QA-valid intersection only: pixels where both the
pre and post QA masks say the pixel is informative, so cloud/shadow/no-data do
not create mechanical false negatives. Two coverage statistics are also
reported: bbox_coverage = in-bbox MTBS area / total MTBS perimeter area, and
valid_coverage = in-valid-pixels MTBS area / total MTBS perimeter area. A low
valid_coverage means the chosen STAC scene only sees a small share of the fire,
so any IoU on that subset is agreement on what was actually observable, not on
the whole fire.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.warp import reproject, Resampling

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.qa_masks import (
    landsat_clear_mask_from_qa_pixel,
    sentinel2_clear_mask_from_qa60,
    sentinel2_clear_mask_from_scl,
)


EVENT_TO_MTBS_ID: dict[str, str] = {
    "camp_fire_2018":      "CA3982012144020181108",
    "creek_fire_2020":     "CA3720111927220200905",
    "dixie_fire_2021":     "CA3987612137920210714",
    "carr_fire_2018":      "CA4065012263020180723",
    "thomas_fire_2017":    "CA3442911910020171205",
    "kincade_fire_2019":   "CA3879612276720191023",
    "august_complex_2020": "CA3966012280920200817",
    "lnu_complex_2020":    "CA3850412233720200817",
}


def _binary_metrics(pred: np.ndarray, target: np.ndarray, valid: np.ndarray) -> dict[str, float]:
    """Pixel-pooled binary metrics on a sub-region defined by `valid`. No smoothing."""
    pred = (pred > 0)[valid]
    target = (target > 0)[valid]
    tp = float(np.logical_and(pred, target).sum())
    fp = float(np.logical_and(pred, ~target).sum())
    fn = float(np.logical_and(~pred, target).sum())
    union = tp + fp + fn
    iou = tp / union if union > 0 else float("nan")
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else float("nan")
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    return {"iou": iou, "f1": f1, "precision": precision, "recall": recall, "tp": tp, "fp": fp, "fn": fn}


def _decode_qa(qa_arr: np.ndarray, qa_mode: str) -> np.ndarray:
    qa_mode = qa_mode.strip().lower()
    if qa_mode == "landsat_qa_pixel":
        return landsat_clear_mask_from_qa_pixel(qa_arr)
    if qa_mode == "sentinel2_qa60":
        return sentinel2_clear_mask_from_qa60(qa_arr)
    if qa_mode == "sentinel2_scl":
        return sentinel2_clear_mask_from_scl(qa_arr)
    raise ValueError(f"Unknown qa_mode {qa_mode!r}")


def _load_qa_aligned_to(label_src, qa_path: str, qa_mode: str) -> np.ndarray:
    """Load a QA file and reproject it onto the label raster's grid."""
    with rasterio.open(qa_path) as qa_src:
        qa = qa_src.read(1)
        clear_native = _decode_qa(qa, qa_mode).astype(np.uint8)
        clear_aligned = np.zeros((label_src.height, label_src.width), dtype=np.uint8)
        reproject(
            source=clear_native,
            destination=clear_aligned,
            src_transform=qa_src.transform,
            src_crs=qa_src.crs,
            dst_transform=label_src.transform,
            dst_crs=label_src.crs,
            resampling=Resampling.nearest,
        )
        return clear_aligned > 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MTBS validation with QA-valid clipping.")
    parser.add_argument("--mtbs-shapefile", type=Path, default=PROJECT_ROOT / "data" / "external" / "mtbs" / "mtbs_perims_DD.shp")
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "data" / "manifests" / "california_train_val_manifest.csv")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data" / "paper_assets" / "mtbs_validation")
    parser.add_argument("--pixel-resolution-m", type=float, default=30.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading MTBS perimeters from {args.mtbs_shapefile} ...")
    gdf = gpd.read_file(args.mtbs_shapefile)
    manifest = pd.read_csv(args.manifest)
    pixel_area_sq_km = (args.pixel_resolution_m * args.pixel_resolution_m) / 1_000_000.0

    rows: list[dict[str, object]] = []
    for _, row in manifest.iterrows():
        event_id = str(row.get("event_id", "")).strip()
        sensor = str(row.get("sensor", "")).strip()
        label_path = Path(str(row.get("label_mask_path", "")).strip())
        pre_qa_path = str(row.get("pre_qa_path", "")).strip()
        post_qa_path = str(row.get("post_qa_path", "")).strip()
        qa_mode = str(row.get("qa_mode", "")).strip()

        if not event_id or not label_path.exists():
            continue
        if event_id not in EVENT_TO_MTBS_ID:
            print(f"  no MTBS mapping for {event_id}, skipping")
            continue

        mtbs_event_id = EVENT_TO_MTBS_ID[event_id]
        match = gdf[gdf["event_id"] == mtbs_event_id]
        if match.empty:
            print(f"  MTBS record {mtbs_event_id} not found, skipping")
            continue
        mtbs_acres = float(match.iloc[0]["burnbndac"])
        mtbs_total_sq_km = mtbs_acres * 0.00404686

        with rasterio.open(label_path) as src:
            mtbs_in_crs = match.to_crs(src.crs)
            mtbs_geom_proj = mtbs_in_crs.geometry.iloc[0]
            mtbs_bbox_mask = rasterize(
                [(mtbs_geom_proj, 1)],
                out_shape=(src.height, src.width),
                transform=src.transform,
                dtype=np.uint8,
                fill=0,
            )
            dnbr_mask = src.read(1)

            # Build QA-valid mask = (pre QA valid) AND (post QA valid)
            valid = np.ones_like(mtbs_bbox_mask, dtype=bool)
            if pre_qa_path and Path(pre_qa_path).exists() and qa_mode:
                try:
                    valid &= _load_qa_aligned_to(src, pre_qa_path, qa_mode)
                except Exception as e:
                    print(f"  pre QA load failed for {event_id} {sensor}: {e}")
            if post_qa_path and Path(post_qa_path).exists() and qa_mode:
                try:
                    valid &= _load_qa_aligned_to(src, post_qa_path, qa_mode)
                except Exception as e:
                    print(f"  post QA load failed for {event_id} {sensor}: {e}")


        # Areas are computed in km^2 using `pixel_area_sq_km` (30 m * 30 m / 1e6).
        mtbs_in_bbox_sq_km = float((mtbs_bbox_mask > 0).sum()) * pixel_area_sq_km
        mtbs_in_valid_sq_km = float(((mtbs_bbox_mask > 0) & valid).sum()) * pixel_area_sq_km
        dnbr_total_sq_km = float((dnbr_mask > 0).sum()) * pixel_area_sq_km
        dnbr_in_valid_sq_km = float(((dnbr_mask > 0) & valid).sum()) * pixel_area_sq_km

        bbox_coverage = mtbs_in_bbox_sq_km / mtbs_total_sq_km if mtbs_total_sq_km > 0 else float("nan")
        valid_coverage = mtbs_in_valid_sq_km / mtbs_total_sq_km if mtbs_total_sq_km > 0 else float("nan")

        # IoU on QA-valid intersection only
        m = _binary_metrics(dnbr_mask, mtbs_bbox_mask, valid)

        # Old-style "all-bbox" IoU for comparison
        m_bbox_only = _binary_metrics(dnbr_mask, mtbs_bbox_mask, np.ones_like(valid))

        rows.append({
            "event_id": event_id,
            "sensor": sensor,
            "mtbs_event_id": mtbs_event_id,
            "mtbs_total_sq_km": round(mtbs_total_sq_km, 2),
            "mtbs_in_bbox_sq_km": round(mtbs_in_bbox_sq_km, 2),
            "mtbs_in_valid_sq_km": round(mtbs_in_valid_sq_km, 2),
            "dnbr_in_valid_sq_km": round(dnbr_in_valid_sq_km, 2),
            "bbox_coverage": round(bbox_coverage, 3),
            "valid_coverage": round(valid_coverage, 3),
            "iou_clipped": round(m["iou"], 4) if not np.isnan(m["iou"]) else float("nan"),
            "f1_clipped": round(m["f1"], 4) if not np.isnan(m["f1"]) else float("nan"),
            "precision_clipped": round(m["precision"], 4) if not np.isnan(m["precision"]) else float("nan"),
            "recall_clipped": round(m["recall"], 4) if not np.isnan(m["recall"]) else float("nan"),
            "iou_old_bbox_only": round(m_bbox_only["iou"], 4) if not np.isnan(m_bbox_only["iou"]) else float("nan"),
        })

    df = pd.DataFrame(rows)
    csv_path = args.output_dir / "mtbs_vs_dnbr_per_pair.csv"
    df.to_csv(csv_path, index=False)

    # Per-event summary (mean across sensor pairs)
    summary = (
        df.groupby("event_id")
        .agg(
            mtbs_total_sq_km=("mtbs_total_sq_km", "first"),
            valid_coverage_mean=("valid_coverage", "mean"),
            iou_clipped_mean=("iou_clipped", "mean"),
            f1_clipped_mean=("f1_clipped", "mean"),
            precision_clipped_mean=("precision_clipped", "mean"),
            recall_clipped_mean=("recall_clipped", "mean"),
            iou_old_mean=("iou_old_bbox_only", "mean"),
            n_pairs=("sensor", "count"),
        )
        .reset_index()
        .sort_values("iou_clipped_mean", ascending=False)
    )
    summary_csv = args.output_dir / "mtbs_vs_dnbr_summary.csv"
    summary.to_csv(summary_csv, index=False)

    overall = {
        "n_pairs_evaluated": int(len(df)),
        "events_evaluated": sorted(df["event_id"].unique().tolist()),
        "mean_iou_clipped": float(df["iou_clipped"].mean(skipna=True)),
        "median_iou_clipped": float(df["iou_clipped"].median(skipna=True)),
        "mean_iou_old_bbox_only": float(df["iou_old_bbox_only"].mean(skipna=True)),
        "mean_valid_coverage": float(df["valid_coverage"].mean(skipna=True)),
        "n_pairs_with_valid_coverage_at_least_50pct": int((df["valid_coverage"] >= 0.5).sum()),
        "n_events_with_mean_valid_coverage_at_least_50pct": int(
            (summary["valid_coverage_mean"] >= 0.5).sum()
        ),
    }
    (args.output_dir / "mtbs_overall.json").write_text(json.dumps(overall, indent=2), encoding="utf-8")

    print()
    print("Per-pair table (clipped to QA-valid intersection):")
    print(df.to_string(index=False))
    print()
    print("Per-event summary:")
    print(summary.to_string(index=False))
    print()
    print(json.dumps(overall, indent=2))


if __name__ == "__main__":
    main()

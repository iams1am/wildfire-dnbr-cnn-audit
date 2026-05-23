"""Leave-one-event-out target-domain threshold calibration on Copernicus EMS.

If the EMS CNN-vs-dNBR result is just a threshold artefact, calibrating the
sigmoid threshold on the other EMS events and testing on the held-out one
should recover a gain; if it does not, the failure is not calibration.

Per (arch, seed): open each event's cached stitched probability raster,
rasterize the EMS AOI and burn polygons on the same 30 m EPSG:6933 grid,
restrict to EMS-AOI inside the scene footprint, pick the IoU-maximising tau on
the other two events, and score the held-out event at that tau against both the
default tau=0.5 and the dNBR rule. Output goes to
data/paper_assets/tables/copernicus_ems_threshold_calibration.csv.
"""
from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = PROJECT_ROOT / "data" / "manifests" / "mediterranean_external_test_manifest_harmonized_noqa.csv"
EMS_CASES = PROJECT_ROOT / "data" / "external" / "copernicus_ems" / "copernicus_ems_cases.json"
ARCHS = ["deeplab", "baseline", "siamese"]
SEEDS = [42, 17, 2026]
THRESHOLDS = [round(t, 2) for t in np.arange(0.10, 0.91, 0.05).tolist()]


def _as_path(value: object) -> Path | None:
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    path = Path(text)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _rasterize(gdf: gpd.GeoDataFrame, ref_path: Path) -> np.ndarray:
    with rasterio.open(ref_path) as ref:
        gdf_ref = gdf.to_crs(ref.crs)
        shapes = [(g, 1) for g in gdf_ref.geometry if g is not None and not g.is_empty]
        if not shapes:
            return np.zeros((ref.height, ref.width), dtype=np.uint8)
        return rasterize(shapes, out_shape=(ref.height, ref.width),
                         transform=ref.transform, fill=0, dtype=np.uint8)


def _read_vector(path: Path, layer: str | None) -> gpd.GeoDataFrame:
    if layer:
        return gpd.read_file(path, layer=layer)
    return gpd.read_file(path)


def _valid_mask(row: pd.Series, shape: tuple[int, int]) -> np.ndarray:
    valid = np.ones(shape, dtype=bool)
    for col in ("pre_clear_mask_harmonized", "post_clear_mask_harmonized"):
        p = _as_path(row.get(col, ""))
        if p is None or not p.exists():
            continue
        with rasterio.open(p) as src:
            mask = src.read(1)
        if mask.shape != shape:
            continue
        valid &= mask > 0
    return valid


def _iou(pred: np.ndarray, target: np.ndarray, valid: np.ndarray) -> float:
    pred_b = (pred > 0) & valid
    target_b = (target > 0) & valid
    tp = float((pred_b & target_b).sum())
    fp = float((pred_b & ~target_b).sum())
    fn = float((~pred_b & target_b).sum())
    union = tp + fp + fn
    return tp / union if union > 0 else float("nan")


def main() -> None:
    manifest = pd.read_csv(MANIFEST)
    cases = json.loads(EMS_CASES.read_text(encoding="utf-8"))

    # Build per-(event, sensor) tuples: (event_id, sensor, prob_arr, dnbr_arr, target_arr, valid_arr).
    # Cache the burn-mask and valid-mask once per (event, sensor) so we can reuse across archs.
    base_records = []
    for case in cases:
        aoi_gdf = _read_vector(_as_path(case["aoi_vector"]), case.get("aoi_layer") or None)
        burn_gdf = _read_vector(_as_path(case["burn_vector"]), case.get("burn_layer") or None)
        event_id = case["event_id"]
        for _, mrow in manifest[manifest["event_id"].astype(str) == event_id].iterrows():
            sensor = str(mrow["sensor"])
            pair_id = str(mrow["pair_id"])
            label_path = _as_path(mrow.get("label_mask_harmonized", ""))
            if label_path is None or not label_path.exists():
                continue
            with rasterio.open(label_path) as ref:
                dnbr_label = ref.read(1).astype(np.uint8)
                shape = dnbr_label.shape
            aoi_raster = _rasterize(aoi_gdf, label_path).astype(bool)
            burn_raster = _rasterize(burn_gdf, label_path).astype(bool) & aoi_raster
            valid = aoi_raster & _valid_mask(mrow, shape)
            if int(valid.sum()) == 0:
                continue
            base_records.append({
                "event_id": event_id, "sensor": sensor, "pair_id": pair_id,
                "dnbr_label": dnbr_label, "burn": burn_raster.astype(np.uint8),
                "valid": valid,
            })

    pairs = [(r["event_id"], r["sensor"]) for r in base_records]
    print(f"EMS valid pairs: {pairs}")

    rows: list[dict] = []
    for arch in ARCHS:
        for seed in SEEDS:
            stitched_dir = (
                PROJECT_ROOT / "data" / "evaluation"
                / f"copernicus_ems_{arch}_stitched_seed{seed}_reflectance64"
            )
            # Load probability + compute IoU(τ) for each (event, sensor, τ).
            prob_iou: dict[tuple[str, str], dict[float, float]] = {}
            dnbr_iou: dict[tuple[str, str], float] = {}
            for rec in base_records:
                prob_path = stitched_dir / rec["pair_id"] / "probability.tif"
                if not prob_path.exists():
                    continue
                with rasterio.open(prob_path) as ps:
                    prob = ps.read(1).astype(np.float32)
                if prob.shape != rec["valid"].shape:
                    continue
                # Map probabilities to a small set of thresholds.
                ious: dict[float, float] = {}
                for t in THRESHOLDS:
                    pred = (prob >= t).astype(np.uint8)
                    ious[t] = _iou(pred, rec["burn"], rec["valid"])
                key = (rec["event_id"], rec["sensor"])
                prob_iou[key] = ious
                dnbr_iou[key] = _iou(rec["dnbr_label"], rec["burn"], rec["valid"])
            if not prob_iou:
                continue

            # Leave-one-event-out target-domain threshold calibration.
            event_ids = sorted({k[0] for k in prob_iou.keys()})
            for test_event in event_ids:
                calib_pairs = [k for k in prob_iou.keys() if k[0] != test_event]
                # For each threshold, compute mean calibration-set IoU and pick the best.
                if not calib_pairs:
                    continue
                best_t = max(
                    THRESHOLDS,
                    key=lambda t: np.nanmean([prob_iou[k].get(t, float("nan")) for k in calib_pairs]),
                )
                # Evaluate the held-out test event at the calibrated threshold + default threshold.
                test_keys = [k for k in prob_iou.keys() if k[0] == test_event]
                for test_key in test_keys:
                    default_iou = prob_iou[test_key].get(0.5, float("nan"))
                    calibrated_iou = prob_iou[test_key].get(best_t, float("nan"))
                    dnbr = dnbr_iou.get(test_key, float("nan"))
                    rows.append({
                        "arch": arch,
                        "seed": seed,
                        "test_event": test_event,
                        "test_sensor": test_key[1],
                        "calib_events": "+".join(sorted({k[0] for k in calib_pairs})),
                        "best_threshold": best_t,
                        "iou_default_05": round(default_iou, 4) if not np.isnan(default_iou) else None,
                        "iou_calibrated": round(calibrated_iou, 4) if not np.isnan(calibrated_iou) else None,
                        "iou_dnbr": round(dnbr, 4) if not np.isnan(dnbr) else None,
                        "delta_calibrated_vs_dnbr": round(calibrated_iou - dnbr, 4) if not np.isnan(calibrated_iou) and not np.isnan(dnbr) else None,
                        "delta_calibrated_vs_default": round(calibrated_iou - default_iou, 4) if not np.isnan(calibrated_iou) and not np.isnan(default_iou) else None,
                    })

    out_csv = PROJECT_ROOT / "data" / "paper_assets" / "tables" / "copernicus_ems_threshold_calibration.csv"
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print()
    print("=== Per-arch 3-seed mean delta (calibrated - dNBR) on held-out events ===")
    if not df.empty:
        summary = df.groupby("arch").agg(
            n_records=("delta_calibrated_vs_dnbr", "size"),
            mean_delta_calibrated_vs_dnbr=("delta_calibrated_vs_dnbr", "mean"),
            mean_delta_calibrated_vs_default=("delta_calibrated_vs_default", "mean"),
        ).round(4)
        print(summary.to_string())
        out_summary = PROJECT_ROOT / "data" / "paper_assets" / "tables" / "copernicus_ems_threshold_calibration_summary.csv"
        summary.to_csv(out_summary)
        print(f"\nWrote {out_csv} and {out_summary}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio

from copernicus_ems_validation import PROJECT_ROOT, _as_path, _counts, _rasterize_layer, _valid_scene_mask


NSW_FIRE_HISTORY_LAYER = (
    "https://portal.spatial.nsw.gov.au/server/rest/services/Hosted/NSWFireHistory/FeatureServer/0/query"
)

EVENTS = {
    "blue_mountains_2019": {
        "bbox": [150.00, -34.00, 150.70, -33.40],
        "start": "2019-10-01",
        "end": "2020-02-29",
    },
    "nsw_black_summer_2020": {
        "bbox": [149.20, -37.70, 150.70, -35.40],
        "start": "2019-10-01",
        "end": "2020-03-31",
    },
}


def _query_url(event_id: str) -> str:
    event = EVENTS[event_id]
    bbox = event["bbox"]
    where = (
        "fire_type='Bushfire' "
        f"AND ignition_date >= DATE '{event['start']}' "
        f"AND ignition_date <= DATE '{event['end']}'"
    )
    params = {
        "f": "geojson",
        "where": where,
        "geometry": f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}",
        "geometryType": "esriGeometryEnvelope",
        "inSR": "4326",
        "outSR": "4326",
        "spatialRel": "esriSpatialRelIntersects",
        "outFields": "objectid,fire_id,fire_name,ignition_date,extinguish_date,fire_type,area_ha,agency",
        "returnGeometry": "true",
        "resultRecordCount": "2000",
        "orderByFields": "area_ha DESC",
    }
    return NSW_FIRE_HISTORY_LAYER + "?" + urlencode(params)


def _load_or_fetch(event_id: str, cache_dir: Path, refresh: bool) -> gpd.GeoDataFrame:
    cache_path = cache_dir / f"{event_id}_nsw_fire_history.geojson"
    if cache_path.exists() and not refresh:
        return gpd.read_file(cache_path)
    cache_dir.mkdir(parents=True, exist_ok=True)
    with urlopen(_query_url(event_id), timeout=60) as response:
        payload = response.read()
    data = json.loads(payload.decode("utf-8"))
    if "error" in data:
        raise RuntimeError(f"NSW Fire History query failed for {event_id}: {data['error']}")
    cache_path.write_bytes(payload)
    gdf = gpd.read_file(cache_path)
    if gdf.empty:
        raise ValueError(f"NSW Fire History returned no features for {event_id}")
    gdf = gdf[gdf.geometry.notnull() & ~gdf.geometry.is_empty].copy()
    if gdf.empty:
        raise ValueError(f"NSW Fire History returned no valid geometries for {event_id}")
    gdf.to_file(cache_path, driver="GeoJSON")
    return gdf


def _prediction_path(root: Path, pair_id: str) -> Path:
    return root / pair_id / "prediction_mask.tif"


def validate(args: argparse.Namespace) -> pd.DataFrame:
    manifest = pd.read_csv(args.manifest)
    rows: list[dict[str, object]] = []

    for event_id in args.events:
        fire_history = _load_or_fetch(event_id, args.cache_dir, args.refresh)
        feature_count = int(len(fire_history))
        area_ha_sum = float(pd.to_numeric(fire_history.get("area_ha"), errors="coerce").fillna(0).sum())
        subset = manifest[manifest["event_id"].astype(str) == event_id].copy()
        if subset.empty:
            raise ValueError(f"No manifest rows for {event_id} in {args.manifest}")

        for _, row in subset.iterrows():
            pair_id = str(row["pair_id"])
            sensor = str(row["sensor"])
            label_path = _as_path(row.get("label_mask_harmonized", ""))
            if label_path is None or not label_path.exists():
                continue
            with rasterio.open(label_path) as ref:
                label = ref.read(1)
                agency_mask = _rasterize_layer(fire_history, ref, all_touched=args.all_touched).astype(bool)
                valid = _valid_scene_mask(row, label.shape)
                pixel_area_sq_km = abs(float(ref.transform.a) * float(ref.transform.e)) / 1_000_000.0

            valid_agency_pixels = int(np.count_nonzero(agency_mask & valid))
            if valid_agency_pixels == 0:
                continue
            dnbr_metrics = _counts(label, agency_mask, valid)
            rows.append(
                {
                    "source": "NSW Fire History",
                    "event_id": event_id,
                    "pair_id": pair_id,
                    "sensor": sensor,
                    "comparison": "dnbr_vs_nsw_fire_history",
                    "feature_count": feature_count,
                    "source_area_ha_sum": round(area_ha_sum, 3),
                    "valid_pixels": int(np.count_nonzero(valid)),
                    "agency_burned_pixels": valid_agency_pixels,
                    "agency_burned_area_sq_km": round(valid_agency_pixels * pixel_area_sq_km, 4),
                    "pred_burned_pixels": int(np.count_nonzero((label > 0) & valid)),
                    **{k: round(v, 6) if np.isfinite(v) else "" for k, v in dnbr_metrics.items()},
                }
            )

            pred_path = _prediction_path(args.prediction_root, pair_id)
            if pred_path.exists():
                with rasterio.open(pred_path) as src:
                    pred = src.read(1)
                    if pred.shape != agency_mask.shape:
                        raise ValueError(f"Prediction shape mismatch for {pred_path}: {pred.shape} vs {agency_mask.shape}")
                pred_metrics = _counts(pred, agency_mask, valid)
                rows.append(
                    {
                        "source": "NSW Fire History",
                        "event_id": event_id,
                        "pair_id": pair_id,
                        "sensor": sensor,
                        "comparison": "deeplab_vs_nsw_fire_history",
                        "feature_count": feature_count,
                        "source_area_ha_sum": round(area_ha_sum, 3),
                        "valid_pixels": int(np.count_nonzero(valid)),
                        "agency_burned_pixels": valid_agency_pixels,
                        "agency_burned_area_sq_km": round(valid_agency_pixels * pixel_area_sq_km, 4),
                        "pred_burned_pixels": int(np.count_nonzero((pred > 0) & valid)),
                        **{k: round(v, 6) if np.isfinite(v) else "" for k, v in pred_metrics.items()},
                    }
                )

    if not rows:
        raise ValueError("No NSW Fire History validation rows were produced.")
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    wide = df.pivot_table(
        index=["event_id", "pair_id", "sensor"],
        columns="comparison",
        values="iou",
        aggfunc="first",
    ).reset_index()
    if {"dnbr_vs_nsw_fire_history", "deeplab_vs_nsw_fire_history"}.issubset(wide.columns):
        wide["deeplab_minus_dnbr_iou"] = (
            wide["deeplab_vs_nsw_fire_history"] - wide["dnbr_vs_nsw_fire_history"]
        )
    return wide.round(6)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Australia masks against NSW Fire History polygons.")
    parser.add_argument("--events", nargs="+", default=list(EVENTS.keys()), choices=list(EVENTS.keys()))
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "data" / "manifests" / "australia_external_test_manifest_harmonized.csv",
    )
    parser.add_argument(
        "--prediction-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "evaluation" / "australia_full_seed42_reflectance64_stitched" / "deeplab",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "external" / "nsw_fire_history",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=PROJECT_ROOT / "data" / "paper_assets" / "tables" / "nsw_fire_history_validation.csv",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=PROJECT_ROOT / "data" / "paper_assets" / "tables" / "nsw_fire_history_validation_summary.csv",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=PROJECT_ROOT / "data" / "paper_assets" / "tables" / "nsw_fire_history_validation.json",
    )
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--all-touched", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = validate(args)
    summary = summarize(df)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    summary.to_csv(args.summary_csv, index=False)
    args.output_json.write_text(json.dumps(df.to_dict(orient="records"), indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"Wrote {args.output_csv}")
    print(f"Wrote {args.summary_csv}")
    print(f"Wrote {args.output_json}")


if __name__ == "__main__":
    main()

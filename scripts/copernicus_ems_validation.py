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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _as_path(value: object) -> Path | None:
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    path = Path(text)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _rasterize_layer(gdf: gpd.GeoDataFrame, ref: rasterio.DatasetReader, *, all_touched: bool) -> np.ndarray:
    if gdf.empty:
        return np.zeros((ref.height, ref.width), dtype=np.uint8)
    gdf_ref = gdf.to_crs(ref.crs)
    shapes = [(geom, 1) for geom in gdf_ref.geometry if geom is not None and not geom.is_empty]
    if not shapes:
        return np.zeros((ref.height, ref.width), dtype=np.uint8)
    return rasterize(
        shapes,
        out_shape=(ref.height, ref.width),
        transform=ref.transform,
        fill=0,
        dtype=np.uint8,
        all_touched=all_touched,
    )


def _counts(pred: np.ndarray, target: np.ndarray, valid: np.ndarray) -> dict[str, float]:
    pred_b = (pred > 0) & valid
    target_b = (target > 0) & valid
    tp = float(np.count_nonzero(pred_b & target_b))
    fp = float(np.count_nonzero(pred_b & ~target_b))
    fn = float(np.count_nonzero(~pred_b & target_b))
    union = tp + fp + fn
    f1_den = 2.0 * tp + fp + fn
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "iou": tp / union if union > 0 else float("nan"),
        "f1": (2.0 * tp) / f1_den if f1_den > 0 else float("nan"),
        "precision": tp / (tp + fp) if tp + fp > 0 else float("nan"),
        "recall": tp / (tp + fn) if tp + fn > 0 else float("nan"),
    }


def _valid_scene_mask(row: pd.Series, shape: tuple[int, int]) -> np.ndarray:
    valid = np.ones(shape, dtype=bool)
    for column in ("pre_clear_mask_harmonized", "post_clear_mask_harmonized"):
        mask_path = _as_path(row.get(column, ""))
        if mask_path is None or not mask_path.exists():
            continue
        with rasterio.open(mask_path) as src:
            mask = src.read(1)
        if mask.shape != shape:
            raise ValueError(f"{column} shape mismatch for {mask_path}: {mask.shape} vs {shape}")
        valid &= mask > 0
    return valid


def _read_vector(path: Path, layer: str | None = None) -> gpd.GeoDataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    if layer:
        return gpd.read_file(path, layer=layer)
    return gpd.read_file(path)


def _case_path(value: object, fallback: Path | None = None) -> Path | None:
    if value is None:
        return fallback
    path = _as_path(value)
    return path if path is not None else fallback


def _build_single_case(args: argparse.Namespace) -> dict[str, object]:
    aoi_vector = args.aoi_vector if args.aoi_vector is not None else args.gdb
    burn_vector = args.burn_vector if args.burn_vector is not None else args.gdb
    return {
        "activation": args.activation,
        "event_id": args.event_id,
        "source": args.source,
        "comparison_tag": args.comparison_tag,
        "aoi_vector": str(aoi_vector),
        "aoi_layer": args.aoi_layer if args.aoi_vector is None else "",
        "burn_vector": str(burn_vector),
        "burn_layer": args.burn_layer if args.burn_vector is None else "",
        "prediction_root": str(args.prediction_root) if args.prediction_root is not None else "",
    }


def _load_cases(args: argparse.Namespace) -> list[dict[str, object]]:
    if args.case_config is None:
        return [_build_single_case(args)]
    config_path = _as_path(args.case_config)
    if config_path is None or not config_path.exists():
        raise FileNotFoundError(args.case_config)
    cases = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(cases, list) or not cases:
        raise ValueError(f"Expected a non-empty JSON list in {config_path}")
    return [dict(case) for case in cases]


def _run_case(
    case: dict[str, object],
    manifest: pd.DataFrame,
    *,
    all_touched: bool,
) -> list[dict[str, object]]:
    event_id = str(case["event_id"])
    source = str(case.get("source", case.get("activation", "Copernicus EMS")))
    tag = str(case.get("comparison_tag", str(case.get("activation", "ems")).lower()))
    aoi_vector = _case_path(case.get("aoi_vector"))
    burn_vector = _case_path(case.get("burn_vector"), fallback=aoi_vector)
    if aoi_vector is None or burn_vector is None:
        raise ValueError(f"Case {event_id} is missing aoi_vector or burn_vector.")

    aoi = _read_vector(aoi_vector, str(case["aoi_layer"]) if case.get("aoi_layer") else None)
    burn = _read_vector(burn_vector, str(case["burn_layer"]) if case.get("burn_layer") else None)
    prediction_root = _case_path(case.get("prediction_root"))

    subset = manifest[manifest["event_id"].astype(str) == event_id].copy()
    if subset.empty:
        raise ValueError(f"No rows for event_id={event_id} in manifest")

    rows: list[dict[str, object]] = []
    for _, row in subset.iterrows():
        pair_id = str(row["pair_id"])
        sensor = str(row["sensor"])
        label_path = _as_path(row.get("label_mask_harmonized", ""))
        if label_path is None:
            continue
        with rasterio.open(label_path) as ref:
            label = ref.read(1)
            aoi_mask = _rasterize_layer(aoi, ref, all_touched=all_touched).astype(bool)
            burn_mask = _rasterize_layer(burn, ref, all_touched=all_touched).astype(bool)
            burn_mask &= aoi_mask
            valid = aoi_mask & _valid_scene_mask(row, label.shape)
            pixel_area_sq_km = abs(float(ref.transform.a) * float(ref.transform.e)) / 1_000_000.0

        metrics = _counts(label, burn_mask, valid)
        aoi_pixels = int(np.count_nonzero(aoi_mask))
        valid_pixels = int(np.count_nonzero(valid))
        ems_burned_pixels = int(np.count_nonzero(burn_mask & valid))
        base = {
            "source": source,
            "event_id": event_id,
            "pair_id": pair_id,
            "sensor": sensor,
            "comparison": f"dnbr_vs_{tag}",
            "aoi_pixels": aoi_pixels,
            "valid_pixels": valid_pixels,
            "ems_burned_pixels": ems_burned_pixels,
            "pred_burned_pixels": int(np.count_nonzero((label > 0) & valid)),
            "aoi_area_sq_km": round(float(aoi_pixels) * pixel_area_sq_km, 4),
            "valid_area_sq_km": round(float(valid_pixels) * pixel_area_sq_km, 4),
            "ems_burned_area_sq_km": round(float(ems_burned_pixels) * pixel_area_sq_km, 4),
            **{k: round(v, 6) if np.isfinite(v) else "" for k, v in metrics.items()},
        }
        rows.append(base)

        if prediction_root is not None:
            pred_path = prediction_root / pair_id / "prediction_mask.tif"
            if pred_path.exists():
                with rasterio.open(pred_path) as src:
                    pred = src.read(1)
                    if pred.shape != burn_mask.shape:
                        raise ValueError(f"Prediction shape mismatch for {pred_path}: {pred.shape} vs {burn_mask.shape}")
                pred_metrics = _counts(pred, burn_mask, valid)
                rows.append(
                    {
                        "source": source,
                        "event_id": event_id,
                        "pair_id": pair_id,
                        "sensor": sensor,
                        "comparison": f"deeplab_vs_{tag}",
                        "aoi_pixels": aoi_pixels,
                        "valid_pixels": valid_pixels,
                        "ems_burned_pixels": ems_burned_pixels,
                        "pred_burned_pixels": int(np.count_nonzero((pred > 0) & valid)),
                        "aoi_area_sq_km": round(float(aoi_pixels) * pixel_area_sq_km, 4),
                        "valid_area_sq_km": round(float(valid_pixels) * pixel_area_sq_km, 4),
                        "ems_burned_area_sq_km": round(float(ems_burned_pixels) * pixel_area_sq_km, 4),
                        **{k: round(v, 6) if np.isfinite(v) else "" for k, v in pred_metrics.items()},
                    }
                )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate dNBR/CNN masks against Copernicus EMS wildfire delineations."
    )
    parser.add_argument("--case-config", type=Path, default=None, help="JSON list of Copernicus EMS validation cases.")
    parser.add_argument("--activation", type=str, default="EMSN159")
    parser.add_argument("--source", type=str, default="Copernicus EMSN159 P07 WildfireDEL")
    parser.add_argument("--comparison-tag", type=str, default="emsn159")
    parser.add_argument(
        "--gdb",
        type=Path,
        default=PROJECT_ROOT
        / "data"
        / "external"
        / "copernicus_ems"
        / "EMSN159"
        / "EMSN159_STD_UTM35N_v02.gdb",
    )
    parser.add_argument("--aoi-vector", type=Path, default=None, help="Direct AOI vector path; overrides --gdb.")
    parser.add_argument("--burn-vector", type=Path, default=None, help="Direct wildfire/damage vector path; overrides --gdb.")
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "data" / "manifests" / "mediterranean_external_test_manifest_harmonized_noqa.csv")
    parser.add_argument("--event-id", type=str, default="rhodes_fire_2023")
    parser.add_argument("--aoi-layer", type=str, default="P00_aoi")
    parser.add_argument("--burn-layer", type=str, default="P07WFRmm_WildfireDEL")
    parser.add_argument("--prediction-root", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, default=PROJECT_ROOT / "data" / "paper_assets" / "tables" / "copernicus_ems_validation.csv")
    parser.add_argument("--output-json", type=Path, default=PROJECT_ROOT / "data" / "paper_assets" / "tables" / "copernicus_ems_validation.json")
    parser.add_argument("--all-touched", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = pd.read_csv(args.manifest)
    rows: list[dict[str, object]] = []
    for case in _load_cases(args):
        rows.extend(_run_case(case, manifest, all_touched=args.all_touched))

    if not rows:
        raise ValueError("No validation rows were produced.")
    out = pd.DataFrame(rows)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv, index=False)
    args.output_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(out.to_string(index=False))
    print(f"Wrote {args.output_csv}")
    print(f"Wrote {args.output_json}")


if __name__ == "__main__":
    main()

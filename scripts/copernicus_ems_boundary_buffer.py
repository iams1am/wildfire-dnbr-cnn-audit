from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from scipy.ndimage import binary_dilation, binary_erosion

from copernicus_ems_validation import (
    PROJECT_ROOT,
    _as_path,
    _case_path,
    _counts,
    _load_cases,
    _rasterize_layer,
    _read_vector,
    _valid_scene_mask,
)


SEEDS = (42, 17, 2026)
ARCHITECTURES = {
    "concat_unet": "copernicus_ems_baseline_stitched_seed{seed}_reflectance64",
    "siamese_unet": "copernicus_ems_siamese_stitched_seed{seed}_reflectance64",
    "deeplabv3_plus": "copernicus_ems_deeplab_stitched_seed{seed}_reflectance64",
}


def _boundary_band(target: np.ndarray, valid: np.ndarray, buffer_px: int) -> np.ndarray:
    target_b = (target > 0) & valid
    if not np.any(target_b):
        return np.zeros_like(valid, dtype=bool)
    structure = np.ones((3, 3), dtype=bool)
    edge = binary_dilation(target_b, structure=structure) ^ binary_erosion(target_b, structure=structure)
    if buffer_px > 0:
        edge = binary_dilation(edge, structure=structure, iterations=buffer_px)
    return edge & valid


def _round_metrics(metrics: dict[str, float]) -> dict[str, float | str]:
    return {k: round(v, 6) if np.isfinite(v) else "" for k, v in metrics.items()}


def _prediction_path(arch: str, seed: int, pair_id: str) -> Path:
    root_name = ARCHITECTURES[arch].format(seed=seed)
    return PROJECT_ROOT / "data" / "evaluation" / root_name / pair_id / "prediction_mask.tif"


def run_boundary_buffer(args: argparse.Namespace) -> pd.DataFrame:
    manifest = pd.read_csv(args.manifest)
    cases = _load_cases(args)
    rows: list[dict[str, object]] = []

    for case in cases:
        event_id = str(case["event_id"])
        activation = str(case.get("activation", "EMS"))
        tag = str(case.get("comparison_tag", activation.lower()))
        aoi_vector = _case_path(case.get("aoi_vector"))
        burn_vector = _case_path(case.get("burn_vector"), fallback=aoi_vector)
        if aoi_vector is None or burn_vector is None:
            raise ValueError(f"Case {event_id} is missing aoi_vector or burn_vector.")

        aoi = _read_vector(aoi_vector, str(case["aoi_layer"]) if case.get("aoi_layer") else None)
        burn = _read_vector(burn_vector, str(case["burn_layer"]) if case.get("burn_layer") else None)
        subset = manifest[manifest["event_id"].astype(str) == event_id].copy()
        if subset.empty:
            raise ValueError(f"No rows for event_id={event_id} in {args.manifest}")

        for _, row in subset.iterrows():
            pair_id = str(row["pair_id"])
            sensor = str(row["sensor"])
            label_path = _as_path(row.get("label_mask_harmonized", ""))
            if label_path is None or not label_path.exists():
                continue

            with rasterio.open(label_path) as ref:
                label = ref.read(1)
                aoi_mask = _rasterize_layer(aoi, ref, all_touched=args.all_touched).astype(bool)
                burn_mask = _rasterize_layer(burn, ref, all_touched=args.all_touched).astype(bool) & aoi_mask
                valid = aoi_mask & _valid_scene_mask(row, label.shape)

            if not np.any(burn_mask & valid):
                continue

            for buffer_px in args.buffers:
                band = _boundary_band(burn_mask, valid, buffer_px)
                if not np.any(band):
                    continue
                dnbr_metrics = _counts(label, burn_mask, band)
                if not np.isfinite(dnbr_metrics["iou"]):
                    continue

                for arch in ARCHITECTURES:
                    for seed in SEEDS:
                        pred_path = _prediction_path(arch, seed, pair_id)
                        if not pred_path.exists():
                            continue
                        with rasterio.open(pred_path) as src:
                            pred = src.read(1)
                            if pred.shape != burn_mask.shape:
                                raise ValueError(
                                    f"Prediction shape mismatch for {pred_path}: {pred.shape} vs {burn_mask.shape}"
                                )
                        pred_metrics = _counts(pred, burn_mask, band)
                        rows.append(
                            {
                                "activation": activation,
                                "event_id": event_id,
                                "pair_id": pair_id,
                                "sensor": sensor,
                                "comparison_tag": tag,
                                "arch": arch,
                                "seed": seed,
                                "buffer_px": buffer_px,
                                "band_pixels": int(np.count_nonzero(band)),
                                "target_pixels_in_band": int(np.count_nonzero((burn_mask > 0) & band)),
                                "dnbr_iou": round(float(dnbr_metrics["iou"]), 6),
                                "cnn_iou": round(float(pred_metrics["iou"]), 6)
                                if np.isfinite(pred_metrics["iou"])
                                else "",
                                "delta_iou": round(float(pred_metrics["iou"] - dnbr_metrics["iou"]), 6)
                                if np.isfinite(pred_metrics["iou"])
                                else "",
                                **{f"dnbr_{k}": v for k, v in _round_metrics(dnbr_metrics).items()},
                                **{f"cnn_{k}": v for k, v in _round_metrics(pred_metrics).items()},
                            }
                        )

    if not rows:
        raise ValueError("No EMS boundary-buffer rows were produced.")
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    numeric = df.copy()
    for column in ("cnn_iou", "delta_iou", "cnn_precision", "cnn_recall", "dnbr_precision", "dnbr_recall"):
        numeric[column] = pd.to_numeric(numeric[column], errors="coerce")
    grouped = (
        numeric.groupby(["arch", "sensor", "buffer_px"], as_index=False)
        .agg(
            n_pair_seed=("delta_iou", "count"),
            mean_dnbr_iou=("dnbr_iou", "mean"),
            mean_cnn_iou=("cnn_iou", "mean"),
            mean_delta_iou=("delta_iou", "mean"),
            std_delta_iou=("delta_iou", "std"),
            mean_dnbr_precision=("dnbr_precision", "mean"),
            mean_cnn_precision=("cnn_precision", "mean"),
            mean_dnbr_recall=("dnbr_recall", "mean"),
            mean_cnn_recall=("cnn_recall", "mean"),
        )
        .sort_values(["arch", "sensor", "buffer_px"])
    )
    for column in grouped.columns:
        if column.startswith("mean_") or column.startswith("std_"):
            grouped[column] = grouped[column].round(6)
    return grouped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute EMS boundary-buffer IoU for dNBR and CNN predictions."
    )
    parser.add_argument(
        "--case-config",
        type=Path,
        default=PROJECT_ROOT / "data" / "external" / "copernicus_ems" / "copernicus_ems_cases.json",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "data" / "manifests" / "copernicus_ems_harmonized_noqa.csv",
    )
    parser.add_argument("--buffers", type=int, nargs="+", default=[0, 2, 5, 10])
    parser.add_argument("--all-touched", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=PROJECT_ROOT / "data" / "paper_assets" / "tables" / "copernicus_ems_boundary_buffer.csv",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=PROJECT_ROOT / "data" / "paper_assets" / "tables" / "copernicus_ems_boundary_buffer_summary.csv",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=PROJECT_ROOT / "data" / "paper_assets" / "tables" / "copernicus_ems_boundary_buffer_summary.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = run_boundary_buffer(args)
    summary = summarize(df)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    summary.to_csv(args.summary_csv, index=False)
    args.summary_json.write_text(json.dumps(summary.to_dict(orient="records"), indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"Wrote {args.output_csv}")
    print(f"Wrote {args.summary_csv}")
    print(f"Wrote {args.summary_json}")


if __name__ == "__main__":
    main()

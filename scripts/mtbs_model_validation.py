"""Compare stitched CNN predictions and dNBR labels directly against MTBS perimeters."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import torch
from rasterio.features import rasterize

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.sliding_window import predict_scene_weighted
from src.models.factory import build_model, list_models


EVENT_TO_MTBS_ID: dict[str, str] = {
    "camp_fire_2018": "CA3982012144020181108",
    "creek_fire_2020": "CA3720111927220200905",
    "dixie_fire_2021": "CA3987612137920210714",
    "carr_fire_2018": "CA4065012263020180723",
    "thomas_fire_2017": "CA3442911910020171205",
    "kincade_fire_2019": "CA3879612276720191023",
    "august_complex_2020": "CA3966012280920200817",
    "lnu_complex_2020": "CA3850412233720200817",
}


def _as_path(value: object) -> Path | None:
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    path = Path(text)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _binary_metrics(pred: np.ndarray, target: np.ndarray, valid: np.ndarray) -> dict[str, float]:
    pred_b = (pred > 0)[valid]
    target_b = (target > 0)[valid]
    tp = float(np.logical_and(pred_b, target_b).sum())
    fp = float(np.logical_and(pred_b, ~target_b).sum())
    fn = float(np.logical_and(~pred_b, target_b).sum())
    union = tp + fp + fn
    denom_f1 = (2.0 * tp) + fp + fn
    return {
        "iou": tp / union if union > 0 else float("nan"),
        "f1": (2.0 * tp) / denom_f1 if denom_f1 > 0 else float("nan"),
        "precision": tp / (tp + fp) if (tp + fp) > 0 else float("nan"),
        "recall": tp / (tp + fn) if (tp + fn) > 0 else float("nan"),
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def _load_or_empty(path: Path | None, shape: tuple[int, int]) -> np.ndarray:
    if path is None or not path.exists():
        return np.ones(shape, dtype=bool)
    with rasterio.open(path) as src:
        return src.read(1) > 0


def _mtbs_mask_for_row(
    *,
    row: pd.Series,
    mtbs_gdf: gpd.GeoDataFrame,
    reference_src: rasterio.io.DatasetReader,
) -> tuple[np.ndarray, float, str]:
    event_id = str(row.get("event_id", "")).strip()
    if event_id not in EVENT_TO_MTBS_ID:
        raise KeyError(f"No MTBS mapping for event_id={event_id}")
    mtbs_event_id = EVENT_TO_MTBS_ID[event_id]
    match = mtbs_gdf[mtbs_gdf["event_id"] == mtbs_event_id]
    if match.empty:
        raise KeyError(f"MTBS record not found: {mtbs_event_id}")
    mtbs_in_crs = match.to_crs(reference_src.crs)
    mask = rasterize(
        [(mtbs_in_crs.geometry.iloc[0], 1)],
        out_shape=(reference_src.height, reference_src.width),
        transform=reference_src.transform,
        dtype=np.uint8,
        fill=0,
    )
    total_sq_km = float(match.iloc[0]["burnbndac"]) * 0.00404686
    return mask, total_sq_km, mtbs_event_id


def _load_model(
    *,
    model_name: str,
    checkpoint: Path,
    channels_per_image: int,
    base_channels: int,
) -> torch.nn.Module:
    model = build_model(model_name, channels_per_image=channels_per_image, base_channels=base_channels)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate dNBR labels and stitched CNN predictions against MTBS on identical QA-valid pixels."
    )
    parser.add_argument("--model-name", choices=list_models(), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "data" / "manifests" / "california_train_val_manifest_harmonized.csv")
    parser.add_argument("--mtbs-shapefile", type=Path, default=PROJECT_ROOT / "data" / "external" / "mtbs" / "mtbs_perims_DD.shp")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data" / "paper_assets" / "mtbs_model_validation")
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--write-rasters", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-pairs", type=int, default=0, help="Debug limit; 0 means all manifest rows.")
    parser.add_argument("--event-id", type=str, default="", help="Optional event filter for targeted/debug runs.")
    parser.add_argument("--sensor", type=str, default="", help="Optional sensor filter for targeted/debug runs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(args.manifest)
    if args.event_id:
        manifest = manifest[manifest["event_id"].astype(str) == args.event_id]
    if args.sensor:
        manifest = manifest[manifest["sensor"].astype(str).str.lower() == args.sensor.strip().lower()]
    if args.max_pairs > 0:
        manifest = manifest.head(args.max_pairs)
    if manifest.empty:
        raise ValueError(f"Manifest is empty: {args.manifest}")

    mtbs_gdf = gpd.read_file(args.mtbs_shapefile)
    first_pre = _as_path(manifest.iloc[0]["pre_image_harmonized"])
    if first_pre is None:
        raise ValueError("Manifest missing pre_image_harmonized path.")
    with rasterio.open(first_pre) as src:
        channels_per_image = int(src.count)
    model = _load_model(
        model_name=args.model_name,
        checkpoint=args.checkpoint,
        channels_per_image=channels_per_image,
        base_channels=args.base_channels,
    )

    rows: list[dict[str, object]] = []
    for _, row in manifest.iterrows():
        pair_id = str(row.get("pair_id", "")).strip()
        event_id = str(row.get("event_id", "")).strip()
        sensor = str(row.get("sensor", "")).strip().lower()
        pre_path = _as_path(row.get("pre_image_harmonized", ""))
        post_path = _as_path(row.get("post_image_harmonized", ""))
        label_path = _as_path(row.get("label_mask_harmonized", ""))
        pre_clear = _as_path(row.get("pre_clear_mask_harmonized", ""))
        post_clear = _as_path(row.get("post_clear_mask_harmonized", ""))
        if pre_path is None or post_path is None or label_path is None:
            print(f"Skipping {pair_id}: missing harmonized paths")
            continue
        if event_id not in EVENT_TO_MTBS_ID:
            print(f"Skipping {pair_id}: no MTBS mapping")
            continue

        pair_dir = args.output_dir / pair_id
        pred_path = pair_dir / f"{args.model_name}_prediction_mask.tif"
        prob_path = pair_dir / f"{args.model_name}_probability.tif" if args.write_rasters else None
        pred_write_path = pred_path if args.write_rasters or not pred_path.exists() else pred_path

        if not pred_path.exists():
            predict_scene_weighted(
                model=model,
                pre_image_path=pre_path,
                post_image_path=post_path,
                sensor=sensor,
                patch_size=args.patch_size,
                stride=args.stride,
                device=args.device,
                threshold=args.threshold,
                label_mask_path=None,
                pre_clear_mask_path=pre_clear,
                post_clear_mask_path=post_clear,
                normalize=args.normalize,
                output_prob_path=prob_path,
                output_mask_path=pred_write_path,
            )

        with rasterio.open(label_path) as label_src:
            dnbr_mask = label_src.read(1)
            mtbs_mask, mtbs_total_sq_km, mtbs_event_id = _mtbs_mask_for_row(
                row=row,
                mtbs_gdf=mtbs_gdf,
                reference_src=label_src,
            )
            pre_clear_arr = _load_or_empty(pre_clear, dnbr_mask.shape)
            post_clear_arr = _load_or_empty(post_clear, dnbr_mask.shape)
            valid = pre_clear_arr & post_clear_arr
            pixel_area_sq_km = abs(label_src.transform.a * label_src.transform.e) / 1_000_000.0

        with rasterio.open(pred_path) as pred_src:
            pred_mask = pred_src.read(1)
            if pred_mask.shape != dnbr_mask.shape:
                raise ValueError(f"Prediction/label shape mismatch for {pair_id}: {pred_mask.shape} vs {dnbr_mask.shape}")

        mtbs_in_valid_sq_km = float(((mtbs_mask > 0) & valid).sum()) * pixel_area_sq_km
        dnbr_in_valid_sq_km = float(((dnbr_mask > 0) & valid).sum()) * pixel_area_sq_km
        model_in_valid_sq_km = float(((pred_mask > 0) & valid).sum()) * pixel_area_sq_km
        valid_coverage = mtbs_in_valid_sq_km / mtbs_total_sq_km if mtbs_total_sq_km > 0 else float("nan")
        usable_for_iou = mtbs_in_valid_sq_km > 0.0
        if usable_for_iou:
            dnbr_metrics = _binary_metrics(dnbr_mask, mtbs_mask, valid)
            model_metrics = _binary_metrics(pred_mask, mtbs_mask, valid)
        else:
            dnbr_metrics = {k: float("nan") for k in ("iou", "f1", "precision", "recall", "tp", "fp", "fn")}
            model_metrics = {k: float("nan") for k in ("iou", "f1", "precision", "recall", "tp", "fp", "fn")}

        record = {
            "pair_id": pair_id,
            "event_id": event_id,
            "sensor": sensor,
            "mtbs_event_id": mtbs_event_id,
            "mtbs_total_sq_km": mtbs_total_sq_km,
            "mtbs_in_valid_sq_km": mtbs_in_valid_sq_km,
            "valid_coverage": valid_coverage,
            "dnbr_area_sq_km": dnbr_in_valid_sq_km,
            "model_area_sq_km": model_in_valid_sq_km,
            "usable_for_iou": usable_for_iou,
            "dnbr_iou": dnbr_metrics["iou"],
            "dnbr_f1": dnbr_metrics["f1"],
            "dnbr_precision": dnbr_metrics["precision"],
            "dnbr_recall": dnbr_metrics["recall"],
            "model_iou": model_metrics["iou"],
            "model_f1": model_metrics["f1"],
            "model_precision": model_metrics["precision"],
            "model_recall": model_metrics["recall"],
            "model_minus_dnbr_iou": model_metrics["iou"] - dnbr_metrics["iou"]
            if np.isfinite(model_metrics["iou"]) and np.isfinite(dnbr_metrics["iou"])
            else float("nan"),
            "dnbr_tp": dnbr_metrics["tp"],
            "dnbr_fp": dnbr_metrics["fp"],
            "dnbr_fn": dnbr_metrics["fn"],
            "model_tp": model_metrics["tp"],
            "model_fp": model_metrics["fp"],
            "model_fn": model_metrics["fn"],
        }
        rows.append(record)
        print(
            f"{pair_id}: model IoU={record['model_iou']:.4f}, "
            f"dNBR IoU={record['dnbr_iou']:.4f}, coverage={record['valid_coverage']:.3f}"
        )

    if not rows:
        raise ValueError("No MTBS model-validation rows were produced.")

    df = pd.DataFrame(rows)
    per_pair_csv = args.output_dir / f"{args.model_name}_mtbs_per_pair.csv"
    df.to_csv(per_pair_csv, index=False)

    event_summary = (
        df.groupby("event_id")
        .agg(
            valid_coverage_mean=("valid_coverage", "mean"),
            dnbr_iou_mean=("dnbr_iou", "mean"),
            model_iou_mean=("model_iou", "mean"),
            model_minus_dnbr_iou_mean=("model_minus_dnbr_iou", "mean"),
            n_pairs=("pair_id", "count"),
        )
        .reset_index()
        .sort_values("model_iou_mean", ascending=False)
    )
    event_csv = args.output_dir / f"{args.model_name}_mtbs_per_event.csv"
    event_summary.to_csv(event_csv, index=False)

    usable = df[df["usable_for_iou"].astype(bool)]
    comparable = usable[np.isfinite(usable["model_minus_dnbr_iou"])]
    overall = {
        "model_name": args.model_name,
        "checkpoint": str(args.checkpoint),
        "manifest": str(args.manifest),
        "n_pairs": int(len(df)),
        "n_events": int(df["event_id"].nunique()),
        "n_pairs_usable_for_iou": int(len(usable)),
        "n_events_usable_for_iou": int(usable["event_id"].nunique()) if not usable.empty else 0,
        "mean_valid_coverage": float(df["valid_coverage"].mean(skipna=True)),
        "mean_dnbr_iou": float(usable["dnbr_iou"].mean(skipna=True)) if not usable.empty else float("nan"),
        "mean_model_iou": float(usable["model_iou"].mean(skipna=True)) if not usable.empty else float("nan"),
        "mean_model_minus_dnbr_iou": float(comparable["model_minus_dnbr_iou"].mean(skipna=True)),
        "median_dnbr_iou": float(usable["dnbr_iou"].median(skipna=True)) if not usable.empty else float("nan"),
        "median_model_iou": float(usable["model_iou"].median(skipna=True)) if not usable.empty else float("nan"),
        "pairs_model_beats_dnbr": int((comparable["model_minus_dnbr_iou"] > 0).sum()),
        "pairs_model_loses_to_dnbr": int((comparable["model_minus_dnbr_iou"] < 0).sum()),
    }
    overall_json = args.output_dir / f"{args.model_name}_mtbs_overall.json"
    overall_json.write_text(json.dumps(overall, indent=2), encoding="utf-8")

    print(f"Wrote per-pair MTBS comparison: {per_pair_csv}")
    print(f"Wrote per-event MTBS comparison: {event_csv}")
    print(f"Wrote MTBS overall summary: {overall_json}")
    print(json.dumps(overall, indent=2))


if __name__ == "__main__":
    main()

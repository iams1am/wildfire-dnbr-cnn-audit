from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.sliding_window import predict_scene_weighted
from src.models.factory import build_model, list_models


def _as_path(value: object) -> Path | None:
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    path = Path(text)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _aggregate(rows: list[dict[str, object]]) -> dict[str, float]:
    tp = float(sum(float(r.get("tp_total", 0.0)) for r in rows))
    fp = float(sum(float(r.get("fp_total", 0.0)) for r in rows))
    fn = float(sum(float(r.get("fn_total", 0.0)) for r in rows))
    union = tp + fp + fn
    denom_f1 = (2.0 * tp) + fp + fn
    errors = np.array([float(r["abs_error_sq_km"]) for r in rows if "abs_error_sq_km" in r], dtype=np.float64)
    return {
        "iou": tp / union if union > 0.0 else float("nan"),
        "f1": (2.0 * tp) / denom_f1 if denom_f1 > 0.0 else float("nan"),
        "precision": tp / (tp + fp) if (tp + fp) > 0.0 else float("nan"),
        "recall": tp / (tp + fn) if (tp + fn) > 0.0 else float("nan"),
        "area_mae_sq_km": float(errors.mean()) if errors.size else float("nan"),
        "area_rmse_sq_km": float(np.sqrt(np.mean(np.square(errors)))) if errors.size else float("nan"),
        "scene_count": float(len(rows)),
        "tp_total": tp,
        "fp_total": fp,
        "fn_total": fn,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a model with overlapped stitched scene inference.")
    parser.add_argument("--model-name", choices=list_models(), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--write-rasters", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = pd.read_csv(args.manifest)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if manifest.empty:
        raise ValueError(f"Manifest is empty: {args.manifest}")
    first_pre = _as_path(manifest.iloc[0]["pre_image_harmonized"])
    if first_pre is None:
        raise ValueError("Manifest missing pre_image_harmonized path.")

    import rasterio

    with rasterio.open(first_pre) as src:
        channels_per_image = int(src.count)

    model = build_model(args.model_name, channels_per_image=channels_per_image, base_channels=args.base_channels)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state)

    rows: list[dict[str, object]] = []
    for _, row in manifest.iterrows():
        pair_id = str(row.get("pair_id", "")).strip()
        sensor = str(row.get("sensor", "")).strip().lower()
        pre_path = _as_path(row.get("pre_image_harmonized", ""))
        post_path = _as_path(row.get("post_image_harmonized", ""))
        label_path = _as_path(row.get("label_mask_harmonized", ""))
        pre_clear = _as_path(row.get("pre_clear_mask_harmonized", ""))
        post_clear = _as_path(row.get("post_clear_mask_harmonized", ""))
        if pre_path is None or post_path is None:
            continue

        pair_dir = args.output_dir / pair_id
        prob_path = pair_dir / "probability.tif" if args.write_rasters else None
        mask_path = pair_dir / "prediction_mask.tif" if args.write_rasters else None
        metrics = predict_scene_weighted(
            model=model,
            pre_image_path=pre_path,
            post_image_path=post_path,
            sensor=sensor,
            patch_size=args.patch_size,
            stride=args.stride,
            device=args.device,
            threshold=args.threshold,
            label_mask_path=label_path,
            pre_clear_mask_path=pre_clear,
            post_clear_mask_path=post_clear,
            normalize=args.normalize,
            output_prob_path=prob_path,
            output_mask_path=mask_path,
        )
        record = {
            "pair_id": pair_id,
            "event_id": str(row.get("event_id", "")),
            "sensor": sensor,
            "region": str(row.get("region", "")),
            **metrics,
        }
        rows.append(record)
        print(f"{pair_id}: IoU={record.get('iou', float('nan')):.4f} valid={record.get('valid_fraction', 0.0):.3f}")

    detail_df = pd.DataFrame(rows)
    detail_csv = args.output_dir / f"{args.model_name}_scene_metrics.csv"
    detail_df.to_csv(detail_csv, index=False)
    summary = _aggregate(rows)
    summary_json = args.output_dir / f"{args.model_name}_scene_summary.json"
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote scene metrics: {detail_csv}")
    print(f"Wrote scene summary: {summary_json}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

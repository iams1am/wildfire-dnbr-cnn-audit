from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

METRICS = [
    "iou",
    "f1",
    "precision",
    "recall",
    "iou_patch_mean",
    "f1_patch_mean",
    "precision_patch_mean",
    "recall_patch_mean",
    "true_area_sq_km",
    "pred_area_sq_km",
    "area_mae_sq_km",
    "area_rmse_sq_km",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate per-event/sensor evaluation CSVs across seeds.")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 17, 2026])
    parser.add_argument(
        "--input-template",
        type=str,
        default=str(PROJECT_ROOT / "data" / "paper_assets" / "tables" / "per_event_results_seed{seed}_reflectance64.csv"),
        help="Template containing {seed}.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=PROJECT_ROOT / "data" / "paper_assets" / "tables" / "per_event_results_3seed_reflectance64.csv",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=PROJECT_ROOT / "data" / "paper_assets" / "tables" / "per_event_results_3seed_reflectance64.md",
    )
    parser.add_argument("--group-columns", nargs="+", default=["event_id", "sensor"])
    return parser.parse_args()


def _load_seed_csv(template: str, seed: int) -> pd.DataFrame:
    path = Path(template.format(seed=seed))
    if not path.exists():
        raise FileNotFoundError(f"Missing per-event CSV for seed {seed}: {path}")
    df = pd.read_csv(path)
    df.insert(0, "seed", seed)
    return df


def _format_md(df: pd.DataFrame) -> str:
    cols = [
        "event_id",
        "sensor",
        "iou_mean",
        "iou_std",
        "recall_mean",
        "area_mae_sq_km_mean",
        "patch_count",
    ]
    cols = [c for c in cols if c in df.columns]
    view = df[cols].copy()
    for col in view.columns:
        if col.endswith("_mean") or col.endswith("_std"):
            view[col] = view[col].apply(lambda v: f"{v:.3f}" if isinstance(v, (int, float, np.floating)) and np.isfinite(v) else "")
    return view.to_markdown(index=False)


def main() -> None:
    args = parse_args()
    frames = [_load_seed_csv(args.input_template, seed) for seed in args.seeds]
    raw = pd.concat(frames, ignore_index=True)

    missing_groups = [col for col in args.group_columns if col not in raw.columns]
    if missing_groups:
        raise ValueError(f"Missing group columns in per-event CSVs: {missing_groups}")

    agg_parts: list[pd.DataFrame] = []
    grouped = raw.groupby(args.group_columns, dropna=False)
    for keys, group in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        row: dict[str, object] = {col: value for col, value in zip(args.group_columns, keys)}
        row["n_seeds"] = int(group["seed"].nunique())
        if "patch_count" in group.columns:
            row["patch_count"] = int(round(float(group["patch_count"].median())))
            row["patches"] = row["patch_count"]
        for metric in METRICS:
            if metric not in group.columns:
                continue
            values = group[metric].to_numpy(dtype=np.float64)
            row[f"{metric}_mean"] = float(np.nanmean(values))
            row[f"{metric}_std"] = float(np.nanstd(values, ddof=1)) if np.isfinite(values).sum() > 1 else 0.0
        agg_parts.append(pd.DataFrame([row]))

    if not agg_parts:
        raise ValueError("No per-event rows were aggregated.")

    out = pd.concat(agg_parts, ignore_index=True)
    sort_cols = [c for c in ["iou_mean", "event_id", "sensor"] if c in out.columns]
    if "iou_mean" in sort_cols:
        out = out.sort_values("iou_mean", ascending=False)
    elif sort_cols:
        out = out.sort_values(sort_cols)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv, index=False)
    args.output_md.write_text(_format_md(out), encoding="utf-8")
    print(f"Wrote {args.output_csv}")
    print(f"Wrote {args.output_md}")
    print(_format_md(out))


if __name__ == "__main__":
    main()

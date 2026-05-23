"""Run CNN-vs-MTBS validation on held-out events from leakage-aware k-fold runs."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.factory import list_models


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate each k-fold checkpoint against MTBS on only its held-out event(s)."
    )
    parser.add_argument("--model-name", choices=list_models(), default="siamese")
    parser.add_argument("--kfold-dir", type=Path, default=PROJECT_ROOT / "data" / "paper_assets" / "stats_reflectance64")
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "data" / "manifests" / "california_train_val_manifest_harmonized.csv")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data" / "paper_assets" / "mtbs_model_validation_reflectance64")
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--write-rasters", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _run(cmd: list[str], *, dry_run: bool) -> None:
    print(" ".join(cmd))
    if not dry_run:
        subprocess.run(cmd, check=True)


def _summarize(rows: pd.DataFrame) -> dict[str, float | int | str]:
    usable = rows[rows["usable_for_iou"].astype(bool)]
    comparable = usable[np.isfinite(usable["model_minus_dnbr_iou"])]
    return {
        "n_pairs": int(len(rows)),
        "n_events": int(rows["event_id"].nunique()) if not rows.empty else 0,
        "n_pairs_usable_for_iou": int(len(usable)),
        "n_events_usable_for_iou": int(usable["event_id"].nunique()) if not usable.empty else 0,
        "mean_valid_coverage": float(rows["valid_coverage"].mean(skipna=True)) if not rows.empty else float("nan"),
        "mean_dnbr_iou": float(usable["dnbr_iou"].mean(skipna=True)) if not usable.empty else float("nan"),
        "mean_model_iou": float(usable["model_iou"].mean(skipna=True)) if not usable.empty else float("nan"),
        "mean_model_minus_dnbr_iou": float(comparable["model_minus_dnbr_iou"].mean(skipna=True)) if not comparable.empty else float("nan"),
        "median_dnbr_iou": float(usable["dnbr_iou"].median(skipna=True)) if not usable.empty else float("nan"),
        "median_model_iou": float(usable["model_iou"].median(skipna=True)) if not usable.empty else float("nan"),
        "pairs_model_beats_dnbr": int((comparable["model_minus_dnbr_iou"] > 0).sum()) if not comparable.empty else 0,
        "pairs_model_loses_to_dnbr": int((comparable["model_minus_dnbr_iou"] < 0).sum()) if not comparable.empty else 0,
    }


def main() -> None:
    args = parse_args()
    assignments_csv = args.kfold_dir / "kfold_assignments.csv"
    if not assignments_csv.exists():
        raise FileNotFoundError(
            f"Missing k-fold assignments: {assignments_csv}. "
            "Run scripts/run_kfold_in_domain.py first with the corrected protocol."
        )
    assignments = pd.read_csv(assignments_csv)
    val = assignments[assignments["split"] == "val"].copy()
    if val.empty:
        raise ValueError(f"No validation assignments found in {assignments_csv}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[pd.DataFrame] = []
    fold_keys = val[["seed", "fold"]].drop_duplicates().sort_values(["seed", "fold"])
    for _, key in fold_keys.iterrows():
        seed = int(key["seed"])
        fold = int(key["fold"])
        events = sorted(str(e) for e in val[(val["seed"] == seed) & (val["fold"] == fold)]["event_id"].dropna().unique())
        checkpoint = args.kfold_dir / "runs" / f"seed{seed}" / args.model_name / f"fold_{fold}" / "best_model.pt"
        if not checkpoint.exists():
            print(f"missing checkpoint, skipping: {checkpoint}")
            continue
        for event_id in events:
            fold_out = args.output_dir / f"seed{seed}" / f"fold_{fold}" / event_id
            cmd = [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "mtbs_model_validation.py"),
                "--model-name",
                args.model_name,
                "--checkpoint",
                str(checkpoint),
                "--manifest",
                str(args.manifest),
                "--output-dir",
                str(fold_out),
                "--base-channels",
                str(args.base_channels),
                "--patch-size",
                str(args.patch_size),
                "--stride",
                str(args.stride),
                "--threshold",
                str(args.threshold),
                "--device",
                args.device,
                "--event-id",
                event_id,
            ]
            cmd.append("--normalize" if args.normalize else "--no-normalize")
            cmd.append("--write-rasters" if args.write_rasters else "--no-write-rasters")
            _run(cmd, dry_run=args.dry_run)
            per_pair = fold_out / f"{args.model_name}_mtbs_per_pair.csv"
            if per_pair.exists():
                df = pd.read_csv(per_pair)
                df.insert(0, "fold", fold)
                df.insert(0, "seed", seed)
                all_rows.append(df)

    if args.dry_run:
        return
    if not all_rows:
        raise ValueError("No per-pair MTBS validation outputs were produced.")

    combined = pd.concat(all_rows, ignore_index=True)
    combined_csv = args.output_dir / f"{args.model_name}_kfold_mtbs_per_pair.csv"
    combined.to_csv(combined_csv, index=False)

    event_summary = (
        combined.groupby("event_id")
        .agg(
            n_pairs=("pair_id", "count"),
            n_seeds=("seed", "nunique"),
            valid_coverage_mean=("valid_coverage", "mean"),
            dnbr_iou_mean=("dnbr_iou", "mean"),
            model_iou_mean=("model_iou", "mean"),
            model_minus_dnbr_iou_mean=("model_minus_dnbr_iou", "mean"),
        )
        .reset_index()
        .sort_values("model_iou_mean", ascending=False)
    )
    event_csv = args.output_dir / f"{args.model_name}_kfold_mtbs_per_event.csv"
    event_summary.to_csv(event_csv, index=False)

    summary = _summarize(combined)
    summary.update({"model_name": args.model_name, "kfold_dir": str(args.kfold_dir)})
    summary_json = args.output_dir / f"{args.model_name}_kfold_mtbs_overall.json"
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote {combined_csv}")
    print(f"Wrote {event_csv}")
    print(f"Wrote {summary_json}")
    print(pd.Series(summary).to_string())


if __name__ == "__main__":
    main()

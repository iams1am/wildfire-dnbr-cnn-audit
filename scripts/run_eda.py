from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.eda_drift import drift_report, scene_statistics_from_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run EDA and drift analysis from harmonized manifests.")
    parser.add_argument(
        "--manifests",
        type=Path,
        nargs="+",
        required=True,
        help="One or more harmonized manifest CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "eda",
        help="Output directory for EDA artifacts.",
    )
    parser.add_argument(
        "--group-column",
        type=str,
        default="region",
        help="Column used to compare drift across groups.",
    )
    parser.add_argument(
        "--reference-group",
        type=str,
        default="California, USA",
        help="Reference group for PSI/JSD comparisons.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest_frames = [pd.read_csv(path) for path in args.manifests]
    merged_manifest = pd.concat(manifest_frames, ignore_index=True) if manifest_frames else pd.DataFrame()
    stats_df = scene_statistics_from_manifest(merged_manifest)
    scene_stats_path = args.output_dir / "scene_stats.csv"
    stats_df.to_csv(scene_stats_path, index=False)

    features = [
        "pre_mean",
        "pre_std",
        "pre_p10",
        "pre_p90",
        "post_mean",
        "post_std",
        "post_p10",
        "post_p90",
        "delta_mean",
        "delta_std",
        "valid_fraction",
        "burned_fraction",
    ]
    drift_df = drift_report(
        stats_df,
        group_column=args.group_column,
        reference_group=args.reference_group,
        features=features,
    )
    drift_path = args.output_dir / "drift_metrics.csv"
    drift_df.to_csv(drift_path, index=False)

    summary_path = args.output_dir / "eda_summary.txt"
    if stats_df.empty:
        summary_text = "No valid harmonized rows found in manifests.\n"
    else:
        counts = stats_df.groupby(args.group_column)["pair_id"].count().sort_values(ascending=False)
        summary_lines = [
            f"Total valid pairs: {len(stats_df)}",
            f"Reference group: {args.reference_group}",
            "",
            "Pairs per group:",
        ]
        summary_lines.extend([f"- {group}: {count}" for group, count in counts.items()])
        summary_text = "\n".join(summary_lines) + "\n"
    summary_path.write_text(summary_text, encoding="utf-8")

    print(f"Wrote scene stats: {scene_stats_path}")
    print(f"Wrote drift metrics: {drift_path}")
    print(f"Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()

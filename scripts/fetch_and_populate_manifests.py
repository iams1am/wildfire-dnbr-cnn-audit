from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset_config import load_dataset_config
from src.data.event_config import load_event_config
from src.data.manifest_builder import MANIFEST_COLUMNS
from src.data.stac_downloader import SENSOR_SPECS, build_real_pair_record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch real STAC wildfire imagery, build pre/post pairs, and populate manifest CSV files."
    )
    parser.add_argument(
        "--dataset-config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "datasets.yaml",
        help="Path to dataset configuration YAML.",
    )
    parser.add_argument(
        "--events-config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "wildfire_events.yaml",
        help="Path to wildfire events YAML.",
    )
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "manifests",
        help="Directory where manifests will be written.",
    )
    parser.add_argument(
        "--download-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw",
        help="Root directory for downloaded/clipped real imagery.",
    )
    parser.add_argument(
        "--dnbr-threshold",
        type=float,
        default=0.1,
        help="dNBR threshold used to derive real burned-area masks from paired imagery.",
    )
    parser.add_argument(
        "--max-events-per-dataset",
        type=int,
        default=None,
        help="Optional cap on number of events fetched per dataset for staged runs.",
    )
    parser.add_argument(
        "--sensors",
        nargs="+",
        default=None,
        help="Optional sensor subset (e.g., --sensors sentinel2 landsat8).",
    )
    parser.add_argument(
        "--skip-missing-pairs",
        action="store_true",
        help="Skip sensor/event pairs that have no usable STAC scenes instead of exiting.",
    )
    return parser.parse_args()


def _validate_sensor(sensor: str) -> None:
    if sensor not in SENSOR_SPECS:
        raise ValueError(
            f"Unsupported sensor '{sensor}' in dataset config. Supported sensors: {sorted(SENSOR_SPECS)}"
        )


def main() -> None:
    args = parse_args()
    dataset_cfg = load_dataset_config(args.dataset_config)
    events_cfg = load_event_config(args.events_config)
    args.manifest_dir.mkdir(parents=True, exist_ok=True)
    args.download_root.mkdir(parents=True, exist_ok=True)

    for dataset_name, dataset in dataset_cfg.datasets.items():
        if dataset_name not in events_cfg:
            raise ValueError(
                f"Dataset '{dataset_name}' missing from event config '{args.events_config}'."
            )

        rows: list[dict[str, str]] = []
        events = events_cfg[dataset_name]
        if args.max_events_per_dataset is not None:
            if args.max_events_per_dataset <= 0:
                raise ValueError("--max-events-per-dataset must be positive when provided.")
            events = events[: args.max_events_per_dataset]

        selected_sensors = dataset.sensors
        if args.sensors is not None:
            selected = [sensor for sensor in dataset.sensors if sensor in args.sensors]
            if not selected:
                raise ValueError(
                    f"None of requested sensors {args.sensors} are in dataset '{dataset_name}' sensors {dataset.sensors}."
                )
            selected_sensors = selected

        for event in events:
            for sensor in selected_sensors:
                _validate_sensor(sensor)
                print(
                    f"Fetching dataset={dataset_name} event={event.event_id} sensor={sensor}..."
                )
                try:
                    row = build_real_pair_record(
                        event=event,
                        dataset_name=dataset_name,
                        dataset_region=dataset.region,
                        dataset_split=dataset.role,
                        sensor=sensor,
                        output_root=args.download_root,
                        pre_window_days=dataset_cfg.pairing_rules.pre_event_window_days,
                        post_window_days=dataset_cfg.pairing_rules.post_event_window_days,
                        max_cloud_fraction=dataset_cfg.pairing_rules.max_cloud_fraction,
                        min_overlap_fraction=dataset_cfg.pairing_rules.min_overlap_fraction,
                        dnbr_threshold=args.dnbr_threshold,
                    )
                    rows.append(row)
                except RuntimeError:
                    if not args.skip_missing_pairs:
                        raise
                    print(
                        f"Skipping dataset={dataset_name} event={event.event_id} sensor={sensor} "
                        "because no usable pair was found."
                    )

        manifest_df = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
        manifest_path = args.manifest_dir / f"{dataset_name}_manifest.csv"
        manifest_df.to_csv(manifest_path, index=False)
        print(f"Wrote manifest with {len(manifest_df)} real rows: {manifest_path}")


if __name__ == "__main__":
    main()

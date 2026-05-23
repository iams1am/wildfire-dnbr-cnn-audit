"""Focused fetch script for the Mediterranean Europe third-region extension.

Calls `build_real_pair_record` directly for each (event, sensor) in the
Mediterranean dataset only, so it does not touch the existing California or
Australia manifests on disk. Writes a fresh manifest at
`data/manifests/mediterranean_external_test_manifest.csv`.
"""
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
from src.data.stac_downloader import build_real_pair_record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mediterranean Europe-only STAC fetch.")
    parser.add_argument("--dataset-name", type=str, default="mediterranean_external_test")
    parser.add_argument("--dataset-config", type=Path, default=PROJECT_ROOT / "configs" / "datasets.yaml")
    parser.add_argument("--events-config", type=Path, default=PROJECT_ROOT / "configs" / "wildfire_events.yaml")
    parser.add_argument("--manifest-dir", type=Path, default=PROJECT_ROOT / "data" / "manifests")
    parser.add_argument("--download-root", type=Path, default=PROJECT_ROOT / "data" / "raw")
    parser.add_argument("--dnbr-threshold", type=float, default=0.1)
    parser.add_argument("--sensors", nargs="+", default=None)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_cfg = load_dataset_config(args.dataset_config)
    events_cfg = load_event_config(args.events_config)

    if args.dataset_name not in dataset_cfg.datasets:
        raise ValueError(f"Dataset '{args.dataset_name}' not in {args.dataset_config}")
    if args.dataset_name not in events_cfg:
        raise ValueError(f"Dataset '{args.dataset_name}' not in {args.events_config}")

    dataset = dataset_cfg.datasets[args.dataset_name]
    events = events_cfg[args.dataset_name]
    selected_sensors = args.sensors or dataset.sensors

    args.manifest_dir.mkdir(parents=True, exist_ok=True)
    args.download_root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.manifest_dir / f"{args.dataset_name}_manifest.csv"

    rows: list[dict[str, str]] = []
    if args.skip_existing and manifest_path.exists():
        existing = pd.read_csv(manifest_path)
        rows.extend(existing.to_dict(orient="records"))
        existing_keys = {(r.get("event_id"), r.get("sensor")) for r in rows}
    else:
        existing_keys = set()

    for event in events:
        for sensor in selected_sensors:
            if (event.event_id, sensor) in existing_keys:
                print(f"  SKIP existing: {event.event_id} / {sensor}")
                continue
            print(f"Fetching {args.dataset_name} {event.event_id} {sensor}...")
            try:
                row = build_real_pair_record(
                    event=event,
                    dataset_name=args.dataset_name,
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
            except Exception as e:
                print(f"  FAILED {event.event_id}/{sensor}: {type(e).__name__}: {e}")

    if not rows:
        print("No rows fetched.")
        return
    df = pd.DataFrame(rows)
    columns = [c for c in MANIFEST_COLUMNS if c in df.columns] + [c for c in df.columns if c not in MANIFEST_COLUMNS]
    df = df[columns]
    df.to_csv(manifest_path, index=False)
    print(f"\nWrote manifest with {len(df)} rows: {manifest_path}")


if __name__ == "__main__":
    main()

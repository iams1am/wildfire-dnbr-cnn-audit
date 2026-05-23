from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.dataset_config import ProjectDatasetConfig


MANIFEST_COLUMNS = [
    "pair_id",
    "event_id",
    "region",
    "split",
    "sensor",
    "pre_image_path",
    "post_image_path",
    "pre_qa_path",
    "post_qa_path",
    "qa_mode",
    "label_mask_path",
    "pre_date",
    "post_date",
    "cloud_fraction",
    "notes",
]


def default_qa_mode(sensor: str) -> str:
    sensor_key = sensor.strip().lower()
    if sensor_key.startswith("landsat"):
        return "landsat_qa_pixel"
    if sensor_key.startswith("sentinel2"):
        return "sentinel2_scl"
    return ""


def create_manifest_template(output_csv: Path, region: str, split: str, sensors: list[str]) -> None:
    rows = [
        {
            "pair_id": "",
            "event_id": "",
            "region": region,
            "split": split,
            "sensor": sensor,
            "pre_image_path": "",
            "post_image_path": "",
            "pre_qa_path": "",
            "post_qa_path": "",
            "qa_mode": default_qa_mode(sensor),
            "label_mask_path": "",
            "pre_date": "",
            "post_date": "",
            "cloud_fraction": "",
            "notes": "",
        }
        for sensor in sensors
    ]
    template_df = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    template_df.to_csv(output_csv, index=False)


def bootstrap_manifests(config: ProjectDatasetConfig, output_dir: Path) -> list[Path]:
    created_paths: list[Path] = []
    for dataset_name, dataset in config.datasets.items():
        output_csv = output_dir / f"{dataset_name}_manifest.csv"
        create_manifest_template(
            output_csv=output_csv,
            region=dataset.region,
            split=dataset.role,
            sensors=dataset.sensors,
        )
        created_paths.append(output_csv)
    return created_paths

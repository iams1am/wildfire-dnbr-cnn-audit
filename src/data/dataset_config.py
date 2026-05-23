from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class PairingRules:
    pre_event_window_days: int
    post_event_window_days: int
    max_cloud_fraction: float
    min_overlap_fraction: float


@dataclass(frozen=True)
class HarmonizationConfig:
    target_crs: str
    target_resolution_m: int
    spectral_resampling: str
    mask_resampling: str
    per_sensor_reflectance_normalization: bool


@dataclass(frozen=True)
class RegionDatasetConfig:
    name: str
    role: str
    region: str
    sensors: list[str]
    imagery_sources: list[str]
    label_sources: list[str]
    notes: str


@dataclass(frozen=True)
class ProjectDatasetConfig:
    project_title: str
    target_task: str
    output_units: list[str]
    pairing_rules: PairingRules
    harmonization: HarmonizationConfig
    datasets: dict[str, RegionDatasetConfig]


def _as_dict(raw: dict[str, Any], key: str) -> dict[str, Any]:
    value = raw.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Missing or invalid '{key}' section in dataset config.")
    return value


def load_dataset_config(path: str | Path) -> ProjectDatasetConfig:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        raw: dict[str, Any] = yaml.safe_load(handle)

    project = _as_dict(raw, "project")
    pairing = _as_dict(raw, "pairing_rules")
    harmonization = _as_dict(raw, "harmonization")
    datasets_raw = _as_dict(raw, "datasets")

    pairing_rules = PairingRules(
        pre_event_window_days=int(pairing["pre_event_window_days"]),
        post_event_window_days=int(pairing["post_event_window_days"]),
        max_cloud_fraction=float(pairing["max_cloud_fraction"]),
        min_overlap_fraction=float(pairing["min_overlap_fraction"]),
    )

    harmonization_cfg = HarmonizationConfig(
        target_crs=str(harmonization["target_crs"]),
        target_resolution_m=int(harmonization["target_resolution_m"]),
        spectral_resampling=str(harmonization["resampling"]["spectral"]),
        mask_resampling=str(harmonization["resampling"]["mask"]),
        per_sensor_reflectance_normalization=bool(harmonization.get("per_sensor_reflectance_normalization", True)),
    )

    datasets: dict[str, RegionDatasetConfig] = {}
    for name, item in datasets_raw.items():
        if not isinstance(item, dict):
            raise ValueError(f"Dataset '{name}' must be a mapping.")

        # fallback to label_validation if label_sources is missing
        label_sources = item.get("label_sources", item.get("label_validation", []))

        datasets[name] = RegionDatasetConfig(
            name=name,
            role=str(item["role"]),
            region=str(item["region"]),
            sensors=[str(sensor) for sensor in item.get("sensors", [])],
            imagery_sources=[str(source) for source in item.get("imagery_sources", [])],
            label_sources=[str(source) for source in label_sources],
            notes=str(item.get("notes", "")),
        )

    return ProjectDatasetConfig(
        project_title=str(project["title"]),
        target_task=str(project["target_task"]),
        output_units=[str(unit) for unit in project["output_units"]],
        pairing_rules=pairing_rules,
        harmonization=harmonization_cfg,
        datasets=datasets,
    )

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class WildfireEvent:
    event_id: str
    bbox: tuple[float, float, float, float]
    start_date: date
    end_date: date


def _as_list(raw: dict[str, Any], key: str) -> list[dict[str, Any]]:
    value = raw.get(key)
    if not isinstance(value, list):
        raise ValueError(f"Missing or invalid '{key}' events section.")
    output: list[dict[str, Any]] = []
    for row in value:
        if not isinstance(row, dict):
            raise ValueError(f"Event entry for '{key}' must be a mapping.")
        output.append(row)
    return output


def _parse_date(value: str, field_name: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"Invalid {field_name} value '{value}', expected YYYY-MM-DD.") from exc


def load_event_config(path: str | Path) -> dict[str, list[WildfireEvent]]:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Event config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        raw: dict[str, Any] = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError("Event config root must be a mapping.")

    dataset_events: dict[str, list[WildfireEvent]] = {}
    for dataset_name in raw.keys():
        entries = _as_list(raw, dataset_name)
        parsed: list[WildfireEvent] = []
        for entry in entries:
            bbox_raw = entry.get("bbox")
            if (
                not isinstance(bbox_raw, list)
                or len(bbox_raw) != 4
                or not all(isinstance(v, (int, float)) for v in bbox_raw)
            ):
                raise ValueError(f"Event '{entry.get('event_id', 'unknown')}' has invalid bbox.")
            parsed.append(
                WildfireEvent(
                    event_id=str(entry["event_id"]),
                    bbox=(float(bbox_raw[0]), float(bbox_raw[1]), float(bbox_raw[2]), float(bbox_raw[3])),
                    start_date=_parse_date(str(entry["start_date"]), "start_date"),
                    end_date=_parse_date(str(entry["end_date"]), "end_date"),
                )
            )
        dataset_events[dataset_name] = parsed
    return dataset_events

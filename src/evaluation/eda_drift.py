from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from scipy.stats import entropy


SCENE_STAT_COLUMNS = [
    "pair_id",
    "region",
    "split",
    "sensor",
    "pre_mean",
    "pre_std",
    "pre_p10",
    "pre_p90",
    "post_mean",
    "post_std",
    "post_p10",
    "post_p90",
    "valid_fraction",
    "burned_fraction",
    "delta_mean",
    "delta_std",
]


def _read_image_stats(image_path: Path) -> dict[str, float]:
    with rasterio.open(image_path) as src:
        data = src.read().astype(np.float32)
        valid = np.any(data != 0, axis=0)
        valid_fraction = float(valid.mean())
        if not np.any(valid):
            return {
                "mean": float("nan"),
                "std": float("nan"),
                "p10": float("nan"),
                "p90": float("nan"),
                "valid_fraction": valid_fraction,
            }

        valid_pixels = data[:, valid]
        return {
            "mean": float(valid_pixels.mean()),
            "std": float(valid_pixels.std()),
            "p10": float(np.percentile(valid_pixels, 10)),
            "p90": float(np.percentile(valid_pixels, 90)),
            "valid_fraction": valid_fraction,
        }


def _read_burned_fraction(label_path: Path | None) -> float:
    if label_path is None or not label_path.exists():
        return float("nan")
    with rasterio.open(label_path) as src:
        label = src.read(1)
    return float((label > 0).mean())


def scene_statistics_from_manifest(manifest_df: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for _, row in manifest_df.iterrows():
        pair_id = str(row.get("pair_id", "")).strip()
        if not pair_id:
            continue

        pre_path = Path(str(row.get("pre_image_harmonized", "")).strip())
        post_path = Path(str(row.get("post_image_harmonized", "")).strip())
        if not pre_path.exists() or not post_path.exists():
            continue

        pre_stats = _read_image_stats(pre_path)
        post_stats = _read_image_stats(post_path)
        label_raw = str(row.get("label_mask_harmonized", "")).strip()
        label_path = Path(label_raw) if label_raw else None
        burned_fraction = _read_burned_fraction(label_path)

        records.append(
            {
                "pair_id": pair_id,
                "region": str(row.get("region", "")),
                "split": str(row.get("split", "")),
                "sensor": str(row.get("sensor", "")),
                "pre_mean": pre_stats["mean"],
                "pre_std": pre_stats["std"],
                "pre_p10": pre_stats["p10"],
                "pre_p90": pre_stats["p90"],
                "post_mean": post_stats["mean"],
                "post_std": post_stats["std"],
                "post_p10": post_stats["p10"],
                "post_p90": post_stats["p90"],
                "valid_fraction": min(pre_stats["valid_fraction"], post_stats["valid_fraction"]),
                "burned_fraction": burned_fraction,
                "delta_mean": post_stats["mean"] - pre_stats["mean"],
                "delta_std": post_stats["std"] - pre_stats["std"],
            }
        )

    return pd.DataFrame(records, columns=SCENE_STAT_COLUMNS)


def _prepare_hist(reference: np.ndarray, target: np.ndarray, bins: int = 10) -> tuple[np.ndarray, np.ndarray]:
    if reference.size == 0 or target.size == 0:
        raise ValueError("Cannot compute histogram-based drift with empty arrays.")
    combined = np.concatenate([reference, target])
    min_value = float(np.nanmin(combined))
    max_value = float(np.nanmax(combined))
    if min_value == max_value:
        max_value = min_value + 1e-6
    edges = np.linspace(min_value, max_value, bins + 1)
    ref_hist, _ = np.histogram(reference, bins=edges)
    tgt_hist, _ = np.histogram(target, bins=edges)
    eps = 1e-8
    ref_prob = (ref_hist + eps) / (ref_hist.sum() + eps * bins)
    tgt_prob = (tgt_hist + eps) / (tgt_hist.sum() + eps * bins)
    return ref_prob, tgt_prob


def population_stability_index(reference: np.ndarray, target: np.ndarray, bins: int = 10) -> float:
    ref_prob, tgt_prob = _prepare_hist(reference, target, bins=bins)
    return float(np.sum((ref_prob - tgt_prob) * np.log(ref_prob / tgt_prob)))


def jensen_shannon_divergence(reference: np.ndarray, target: np.ndarray, bins: int = 10) -> float:
    ref_prob, tgt_prob = _prepare_hist(reference, target, bins=bins)
    mid = 0.5 * (ref_prob + tgt_prob)
    return float(0.5 * entropy(ref_prob, mid) + 0.5 * entropy(tgt_prob, mid))


def drift_report(
    stats_df: pd.DataFrame,
    *,
    group_column: str,
    reference_group: str,
    features: list[str],
) -> pd.DataFrame:
    if stats_df.empty:
        return pd.DataFrame(
            columns=["reference_group", "target_group", "feature", "psi", "jsd", "reference_count", "target_count"]
        )

    groups = sorted(str(value) for value in stats_df[group_column].dropna().unique())
    if reference_group not in groups:
        raise ValueError(f"Reference group '{reference_group}' not found in {group_column} values: {groups}")

    ref_df = stats_df[stats_df[group_column] == reference_group]
    records: list[dict[str, object]] = []

    for target_group in groups:
        if target_group == reference_group:
            continue
        target_df = stats_df[stats_df[group_column] == target_group]

        for feature in features:
            ref_values = ref_df[feature].dropna().to_numpy(dtype=np.float64)
            target_values = target_df[feature].dropna().to_numpy(dtype=np.float64)
            if ref_values.size < 2 or target_values.size < 2:
                continue

            records.append(
                {
                    "reference_group": reference_group,
                    "target_group": target_group,
                    "feature": feature,
                    "psi": population_stability_index(ref_values, target_values),
                    "jsd": jensen_shannon_divergence(ref_values, target_values),
                    "reference_count": int(ref_values.size),
                    "target_count": int(target_values.size),
                }
            )

    return pd.DataFrame(records)

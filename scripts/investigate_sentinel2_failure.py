from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.sensor_normalization import normalize_patch


# Spectral band order in our stack: [blue, green, red, nir, swir22]
BAND_NAMES = ["blue", "green", "red", "nir", "swir22"]
NIR_IDX = 3
SWIR_IDX = 4


def patch_band_stats(patch_path: Path, sensor: str) -> dict[str, float]:
    """Per-patch summary statistics needed for the S2-vs-L8 investigation."""
    arr = np.load(patch_path)
    pre = normalize_patch(arr["pre"].astype(np.float32), sensor)
    post = normalize_patch(arr["post"].astype(np.float32), sensor)
    label = arr["label"].astype(np.uint8)

    out: dict[str, float] = {}
    for stage, x in (("pre", pre), ("post", post)):
        for bi, bn in enumerate(BAND_NAMES):
            band = x[bi]
            out[f"{stage}_{bn}_mean"] = float(np.mean(band))
            out[f"{stage}_{bn}_std"] = float(np.std(band))
            out[f"{stage}_{bn}_p98"] = float(np.percentile(band, 98))
            out[f"{stage}_{bn}_zero_frac"] = float(np.mean(band == 0))
        nbr_denom = x[NIR_IDX] + x[SWIR_IDX]
        valid = nbr_denom != 0.0
        nbr = np.zeros_like(x[NIR_IDX])
        nbr[valid] = (x[NIR_IDX][valid] - x[SWIR_IDX][valid]) / (nbr_denom[valid] + 1e-6)
        out[f"{stage}_nbr_mean"] = float(np.mean(nbr))
        out[f"{stage}_nbr_std"] = float(np.std(nbr))

    # dNBR
    pre_denom = pre[NIR_IDX] + pre[SWIR_IDX]
    post_denom = post[NIR_IDX] + post[SWIR_IDX]
    valid = (pre_denom != 0.0) & (post_denom != 0.0)
    pre_nbr = np.zeros_like(pre[NIR_IDX])
    post_nbr = np.zeros_like(post[NIR_IDX])
    pre_nbr[valid] = (pre[NIR_IDX][valid] - pre[SWIR_IDX][valid]) / (pre_denom[valid] + 1e-6)
    post_nbr[valid] = (post[NIR_IDX][valid] - post[SWIR_IDX][valid]) / (post_denom[valid] + 1e-6)
    dnbr = pre_nbr - post_nbr
    out["dnbr_mean"] = float(np.mean(dnbr))
    out["dnbr_std"] = float(np.std(dnbr))
    out["dnbr_p10"] = float(np.percentile(dnbr, 10))
    out["dnbr_p50"] = float(np.percentile(dnbr, 50))
    out["dnbr_p90"] = float(np.percentile(dnbr, 90))
    out["dnbr_above_threshold_frac"] = float(np.mean(dnbr >= 0.1))

    out["label_burned_frac"] = float(np.mean(label > 0))
    out["valid_pixel_frac"] = float(np.mean(valid))
    return out


def collect_stats(patch_index_csv: Path, max_patches_per_group: int = 200, seed: int = 42) -> pd.DataFrame:
    df = pd.read_csv(patch_index_csv)
    if not {"event_id", "sensor", "patch_path"}.issubset(df.columns):
        raise ValueError("patch_index missing event_id/sensor/patch_path columns")
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for (event, sensor), grp in df.groupby(["event_id", "sensor"]):
        n = min(max_patches_per_group, len(grp))
        sample = grp.sample(n=n, random_state=int(rng.integers(0, 2**31 - 1)))
        for _, r in sample.iterrows():
            stats = patch_band_stats(Path(str(r["patch_path"])), str(sensor))
            rows.append({"event_id": event, "sensor": sensor, **stats})
    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in df.columns if c not in {"event_id", "sensor"}]
    return df.groupby(["event_id", "sensor"])[cols].mean().reset_index()


def make_plots(df: pd.DataFrame, summary: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # Plot 1: dNBR distribution by sensor
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for sensor, sub in df.groupby("sensor"):
        ax.hist(sub["dnbr_mean"], bins=40, alpha=0.55, label=f"{sensor} (n={len(sub)})", density=True)
    ax.axvline(0.1, ls="--", color="black", label=r"$\tau=0.10$ (label threshold)")
    ax.set_xlabel("Patch-mean dNBR")
    ax.set_ylabel("Density")
    ax.set_title("dNBR distribution per sensor")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "dnbr_distribution_by_sensor.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Plot 2: SWIR2 mean per (event, sensor)
    fig, ax = plt.subplots(figsize=(10, 5))
    pivoted = summary.pivot(index="event_id", columns="sensor", values="post_swir22_mean")
    pivoted.plot(kind="bar", ax=ax)
    ax.set_ylabel("Mean post-event SWIR2 (raw DN, before per-sensor scaling)")
    ax.set_title("SWIR2 raw DN per (event, sensor): Sentinel-2 and Landsat-8 use different scaling")
    ax.legend(title="Sensor")
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(output_dir / "swir2_per_event_sensor.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Plot 3: dNBR-above-threshold fraction (i.e., what fraction of pixels become "burned" labels)
    fig, ax = plt.subplots(figsize=(10, 5))
    pivoted = summary.pivot(index="event_id", columns="sensor", values="dnbr_above_threshold_frac")
    pivoted.plot(kind="bar", ax=ax)
    ax.set_ylabel(r"Fraction of pixels with $\mathrm{dNBR} \geq 0.10$")
    ax.set_title("dNBR-derived burn-label density per (event, sensor)")
    ax.legend(title="Sensor")
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(output_dir / "label_density_per_event_sensor.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Plot 4: per-band reflectance ratio S2/L8 across all events (where both sensors present)
    pairs = []
    for event, grp in summary.groupby("event_id"):
        sensors = set(grp["sensor"])
        if {"sentinel2", "landsat8"}.issubset(sensors):
            s2 = grp[grp["sensor"] == "sentinel2"].iloc[0]
            l8 = grp[grp["sensor"] == "landsat8"].iloc[0]
            for bn in BAND_NAMES:
                col = f"post_{bn}_mean"
                if l8[col] != 0:
                    pairs.append({"event_id": event, "band": bn, "s2_over_l8": s2[col] / l8[col]})
    if pairs:
        pdf = pd.DataFrame(pairs)
        fig, ax = plt.subplots(figsize=(10, 5))
        pdf.boxplot(column="s2_over_l8", by="band", ax=ax)
        ax.axhline(1.0, ls="--", color="grey")
        ax.set_ylabel("Sentinel-2 / Landsat-8 raw-DN ratio")
        ax.set_title("Per-band S2/L8 raw-DN ratio across events (boxplot of event-level means).\n"
                     "Values < 1 reflect different per-sensor scaling, not a physical reflectance difference.")
        plt.suptitle("")
        fig.tight_layout()
        fig.savefig(output_dir / "s2_over_l8_band_ratio.png", dpi=200, bbox_inches="tight")
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sentinel-2 vs Landsat-8 failure investigation.")
    parser.add_argument("--patch-index", type=Path, default=PROJECT_ROOT / "data" / "patches" / "australia_full" / "patch_index.csv")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data" / "paper_assets" / "s2_investigation")
    parser.add_argument("--max-patches-per-group", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = collect_stats(args.patch_index, max_patches_per_group=args.max_patches_per_group, seed=args.seed)
    summary = summarise(df)
    df.to_csv(args.output_dir / "patch_band_stats.csv", index=False)
    summary.to_csv(args.output_dir / "summary_per_event_sensor.csv", index=False)
    make_plots(df, summary, args.output_dir)

    # Cross-check headline finding with a one-line interpretation
    sentinel_label_density = df[df["sensor"] == "sentinel2"]["dnbr_above_threshold_frac"].mean()
    landsat_label_density = df[df["sensor"] == "landsat8"]["dnbr_above_threshold_frac"].mean()
    sentinel_swir = df[df["sensor"] == "sentinel2"]["post_swir22_mean"].mean()
    landsat_swir = df[df["sensor"] == "landsat8"]["post_swir22_mean"].mean()
    summary_json = {
        "sentinel2_label_density": sentinel_label_density,
        "landsat8_label_density": landsat_label_density,
        "label_density_ratio_s2_over_l8": sentinel_label_density / max(landsat_label_density, 1e-9),
        "sentinel2_post_swir22_mean": sentinel_swir,
        "landsat8_post_swir22_mean": landsat_swir,
        "swir22_ratio_s2_over_l8": sentinel_swir / max(landsat_swir, 1e-9),
        "n_sentinel2_patches": int((df["sensor"] == "sentinel2").sum()),
        "n_landsat8_patches": int((df["sensor"] == "landsat8").sum()),
    }
    (args.output_dir / "headline_findings.json").write_text(json.dumps(summary_json, indent=2), encoding="utf-8")
    print(json.dumps(summary_json, indent=2))


if __name__ == "__main__":
    main()

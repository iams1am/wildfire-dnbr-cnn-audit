"""Aggregate the AU->CA reverse-transfer result across the 3 seeds.

Reads the per-pair CSVs from
data/paper_assets/au2ca_mtbs_validation{,_seed17,_seed2026}/deeplab_mtbs_per_pair.csv,
averages model_iou per (event, sensor) over the seeds, takes the per-pair delta
against dNBR, counts seed-averaged wins/losses, and runs an event-cluster
block-bootstrap (resampling whole events) for a 95% CI on the mean delta.
Writes data/paper_assets/tables/au2ca_3seed_summary.{json,csv}.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEEDS = {
    42: PROJECT_ROOT / "data/paper_assets/au2ca_mtbs_validation/deeplab_mtbs_per_pair.csv",
    17: PROJECT_ROOT / "data/paper_assets/au2ca_mtbs_validation_seed17/deeplab_mtbs_per_pair.csv",
    2026: PROJECT_ROOT / "data/paper_assets/au2ca_mtbs_validation_seed2026/deeplab_mtbs_per_pair.csv",
}
OUT_CSV = PROJECT_ROOT / "data/paper_assets/tables/au2ca_3seed_summary.csv"
OUT_JSON = PROJECT_ROOT / "data/paper_assets/tables/au2ca_3seed_summary.json"
N_BOOTSTRAP = 2000
RNG_SEED = 42


def main() -> None:
    frames = []
    for seed, path in SEEDS.items():
        if not path.exists():
            raise SystemExit(f"Missing seed {seed} CSV: {path}")
        df = pd.read_csv(path)
        df["seed"] = seed
        frames.append(df)
    pooled = pd.concat(frames, ignore_index=True)
    usable = pooled[pooled["usable_for_iou"].astype(bool)].copy()

    # Per (event, sensor) mean across seeds
    grp = usable.groupby(["event_id", "sensor"]).agg(
        dnbr_iou=("dnbr_iou", "mean"),
        model_iou=("model_iou", "mean"),
        delta=("model_minus_dnbr_iou", "mean"),
        n_seeds=("model_iou", "size"),
    ).reset_index()
    grp = grp[grp["n_seeds"] > 0].copy()

    # Win/loss on seed-averaged delta
    wins = int((grp["delta"] > 0).sum())
    losses = int((grp["delta"] < 0).sum())

    mean_delta = float(grp["delta"].mean())
    median_delta = float(grp["delta"].median())
    mean_dnbr_iou = float(grp["dnbr_iou"].mean())
    mean_model_iou = float(grp["model_iou"].mean())

    # Event-cluster block bootstrap on event_id
    events = grp["event_id"].unique().tolist()
    by_event = {e: grp[grp["event_id"] == e]["delta"].to_numpy() for e in events}
    rng = np.random.default_rng(RNG_SEED)
    boot_means = np.empty(N_BOOTSTRAP)
    for b in range(N_BOOTSTRAP):
        sampled = rng.choice(events, size=len(events), replace=True)
        vals = np.concatenate([by_event[e] for e in sampled])
        boot_means[b] = vals.mean()
    ci_low, ci_high = float(np.quantile(boot_means, 0.025)), float(np.quantile(boot_means, 0.975))

    summary = {
        "direction": "AU -> CA (reverse transfer, 3 seeds)",
        "n_pair_observations": int(len(usable)),
        "n_unique_pairs": int(len(grp)),
        "n_unique_events": int(len(events)),
        "seeds": list(SEEDS.keys()),
        "mean_dnbr_iou": round(mean_dnbr_iou, 4),
        "mean_au_trained_model_iou": round(mean_model_iou, 4),
        "mean_delta": round(mean_delta, 4),
        "median_delta": round(median_delta, 4),
        "wins_seed_averaged": wins,
        "losses_seed_averaged": losses,
        "event_cluster_bootstrap_ci95": [round(ci_low, 4), round(ci_high, 4)],
        "bootstrap_replicates": N_BOOTSTRAP,
        "headline_ca_to_mtbs_delta": 0.029,
        "headline_ca_to_mtbs_ci95": [0.015, 0.042],
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(summary, indent=2))
    grp.to_csv(OUT_CSV, index=False)
    print(json.dumps(summary, indent=2))
    print(f"\nWrote {OUT_JSON} and {OUT_CSV}")


if __name__ == "__main__":
    main()

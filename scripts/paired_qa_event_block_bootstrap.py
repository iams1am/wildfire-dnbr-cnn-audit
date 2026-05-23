"""Event-level paired tests and a block-bootstrap CI for the paired-QA ablation.

The per-patch paired t-test in paired_qa_comparison.py gives absurd p-values
(~1e-125) because adjacent 256-px patches overlap 50% (stride 128), so the
effective sample size is nowhere near the patch count. This recomputes the
QA-off effect two more honest ways: an 8-event paired t-test on per-event mean
delta IoU, and an event-cluster block bootstrap (resample whole events 2,000
times) for a percentile CI. Both treat the event as the unit of independence.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Event-level + block-bootstrap re-analysis of paired QA ablation.")
    parser.add_argument(
        "--in-csv",
        type=Path,
        nargs="+",
        default=[
            ROOT / "data" / "paper_assets" / "tables" / "paired_qa_comparison_baseline_3seed_reflectance64.csv",
            ROOT / "data" / "paper_assets" / "tables" / "paired_qa_comparison_siamese_3seed_reflectance64.csv",
        ],
    )
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--metric", type=str, default="iou", choices=["iou", "f1", "precision", "recall"])
    parser.add_argument(
        "--out-json",
        type=Path,
        default=ROOT / "data" / "paper_assets" / "tables" / "paired_qa_event_block_bootstrap.json",
    )
    return parser.parse_args()


def event_block_bootstrap(per_patch_delta: pd.Series, event_ids: pd.Series, n_bootstrap: int, rng: np.random.Generator) -> tuple[float, float, float]:
    """Bootstrap-percentile 95% CI for the mean of per_patch_delta where the
    independence unit is `event_id`. We resample events with replacement,
    pool all per-patch deltas from the resampled events, and take the mean."""
    df = pd.DataFrame({"delta": per_patch_delta.values, "event": event_ids.values})
    events = df["event"].unique()
    point = float(df["delta"].mean())
    boot_means = np.empty(n_bootstrap, dtype=np.float64)
    for b in range(n_bootstrap):
        sampled_events = rng.choice(events, size=len(events), replace=True)
        # pool deltas from the resampled events
        sampled = pd.concat([df[df["event"] == e] for e in sampled_events], axis=0, ignore_index=True)
        boot_means[b] = float(sampled["delta"].mean())
    low, high = np.percentile(boot_means, [2.5, 97.5])
    return point, float(low), float(high)


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    out: dict[str, dict] = {}
    for csv_path in args.in_csv:
        if "baseline" in csv_path.name:
            model = "baseline"
        elif "siamese" in csv_path.name:
            model = "siamese"
        else:
            model = csv_path.stem
        df = pd.read_csv(csv_path)
        print(f"\n=== {model}  ({csv_path.name}, n_obs = {len(df)}, n_events = {df['event_id'].nunique()}) ===")

        m = args.metric
        col_on = f"{m}_on"
        col_off = f"{m}_off"
        if col_on not in df.columns or col_off not in df.columns:
            raise KeyError(f"Missing {col_on}/{col_off} in {csv_path}")

        df["delta"] = df[col_off] - df[col_on]

        # (1) Event-level paired test
        per_event = df.groupby("event_id").agg(
            n_obs=("delta", "size"),
            mean_on=(col_on, "mean"),
            mean_off=(col_off, "mean"),
            mean_delta=("delta", "mean"),
        ).reset_index()
        per_event["mean_delta"] = per_event["mean_off"] - per_event["mean_on"]

        on_means = per_event["mean_on"].to_numpy(dtype=np.float64)
        off_means = per_event["mean_off"].to_numpy(dtype=np.float64)
        if len(on_means) >= 2:
            t_stat, t_p = ttest_rel(off_means, on_means, nan_policy="omit")
            try:
                w_stat, w_p = wilcoxon(off_means, on_means)
                w_p = float(w_p)
            except ValueError:
                w_p = float("nan")
        else:
            t_p = float("nan")
            w_p = float("nan")

        # (2) Event-cluster block bootstrap on the per-patch deltas
        boot_point, boot_low, boot_high = event_block_bootstrap(
            df["delta"], df["event_id"], n_bootstrap=args.n_bootstrap, rng=rng
        )

        # (3) Per-patch t-test for comparison (the inflated number)
        per_patch_t_stat, per_patch_t_p = ttest_rel(
            df[col_off].to_numpy(dtype=np.float64),
            df[col_on].to_numpy(dtype=np.float64),
            nan_policy="omit",
        )

        out[model] = {
            "n_per_patch_observations": int(len(df)),
            "n_events": int(df["event_id"].nunique()),
            "metric": m,
            "per_patch_mean_on": float(df[col_on].mean()),
            "per_patch_mean_off": float(df[col_off].mean()),
            "per_patch_mean_delta": float(df["delta"].mean()),
            "per_patch_paired_t_p_value": float(per_patch_t_p),
            "event_level_mean_on": float(np.mean(on_means)),
            "event_level_mean_off": float(np.mean(off_means)),
            "event_level_mean_delta": float(np.mean(off_means - on_means)),
            "event_level_paired_t_p_value": float(t_p),
            "event_level_wilcoxon_p_value": float(w_p),
            "block_bootstrap_mean_delta": boot_point,
            "block_bootstrap_ci95_low": boot_low,
            "block_bootstrap_ci95_high": boot_high,
            "block_bootstrap_n_bootstrap": int(args.n_bootstrap),
            "per_event_table": per_event.round(4).to_dict(orient="records"),
        }

        print(f"  per-patch:    delta={df['delta'].mean():+.4f}  paired-t p={per_patch_t_p:.2e}  (inflated; effective n << {len(df)})")
        print(f"  event-level:  delta={(off_means - on_means).mean():+.4f}  paired-t p={t_p:.4f}  Wilcoxon p={w_p:.4f}  (n_events={len(on_means)})")
        print(f"  block-bootstrap CI95 over events: [{boot_low:+.4f}, {boot_high:+.4f}]  point={boot_point:+.4f}")

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nWrote {args.out_json}")


if __name__ == "__main__":
    main()

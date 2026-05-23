"""Event-cluster bootstrap CI for the Mediterranean cross-region IoU per
architecture, matching the Australia event-cluster table.

The Mediterranean external test has only 4 events, so the resulting CI is
inevitably wide; it is reported explicitly so the Mediterranean ranking is
not over-interpreted under n=4.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PATCH_INDEX = ROOT / "data" / "patches_noqa" / "mediterranean_full" / "patch_index.csv"
EVAL_ROOT = ROOT / "data" / "evaluation"
OUT_CSV = ROOT / "data" / "paper_assets" / "tables" / "mediterranean_event_block_bootstrap.csv"

ARCHS = ["baseline", "deeplab", "siamese", "siamese_fcn_conc", "siamese_fcn_diff", "change_transformer"]
SEEDS = [42, 17, 2026]
N_BOOT = 2000


def load_pooled(arch: str, patch_index: pd.DataFrame) -> pd.DataFrame:
    n = len(patch_index)
    tp = np.zeros(n, dtype=np.float64)
    fp = np.zeros(n, dtype=np.float64)
    fn = np.zeros(n, dtype=np.float64)
    iou_seedmean = np.zeros(n, dtype=np.float64)
    n_loaded = 0
    for seed in SEEDS:
        csv = EVAL_ROOT / f"mediterranean_external_test_seed{seed}_reflectance64" / f"{arch}_patch_metrics.csv"
        if not csv.exists():
            print(f"  WARN: missing {csv}")
            continue
        df = pd.read_csv(csv)
        assert len(df) == n, f"{csv} has {len(df)} rows, expected {n}"
        tp += df["tp"].values
        fp += df["fp"].values
        fn += df["fn"].values
        iou_seedmean += df["iou"].values
        n_loaded += 1
    if n_loaded == 0:
        return pd.DataFrame()
    iou_seedmean /= n_loaded
    return pd.DataFrame({
        "event_id": patch_index["event_id"].values,
        "sensor": patch_index["sensor"].values,
        "tp": tp, "fp": fp, "fn": fn,
        "iou_patch_seedmean": iou_seedmean,
    })


def main() -> None:
    patch_index = pd.read_csv(PATCH_INDEX)
    events = sorted(patch_index["event_id"].unique().tolist())
    n_events = len(events)
    rng = np.random.default_rng(42)

    rows = []
    for arch in ARCHS:
        pooled = load_pooled(arch, patch_index)
        if pooled.empty:
            continue
        per_event = pooled.groupby("event_id").agg(
            tp=("tp", "sum"), fp=("fp", "sum"), fn=("fn", "sum"),
        ).reindex(events)
        denom = per_event["tp"].sum() + per_event["fp"].sum() + per_event["fn"].sum()
        iou_point = float(per_event["tp"].sum() / denom) if denom > 0 else float("nan")
        iou_boot = np.empty(N_BOOT, dtype=np.float64)
        for b in range(N_BOOT):
            idx = rng.integers(0, n_events, size=n_events)
            tp_b = per_event["tp"].values[idx].sum()
            fp_b = per_event["fp"].values[idx].sum()
            fn_b = per_event["fn"].values[idx].sum()
            denom_b = tp_b + fp_b + fn_b
            iou_boot[b] = (tp_b / denom_b) if denom_b > 0 else np.nan
        lo, hi = np.nanpercentile(iou_boot, [2.5, 97.5])
        rows.append({
            "arch": arch,
            "n_events": int(n_events),
            "iou_pooled": round(iou_point, 4),
            "iou_ci_low": round(float(lo), 4),
            "iou_ci_high": round(float(hi), 4),
            "n_bootstrap": N_BOOT,
        })
        print(f"  {arch:22s} IoU={iou_point:.4f} CI=[{lo:.4f},{hi:.4f}] over {n_events} events")

    df = pd.DataFrame(rows).sort_values("iou_pooled", ascending=False)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {OUT_CSV}")


if __name__ == "__main__":
    main()

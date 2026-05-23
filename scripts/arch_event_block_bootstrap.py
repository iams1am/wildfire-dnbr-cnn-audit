"""Event-cluster block-bootstrap 95% CIs for the cross-region architecture
comparison (Table 4 in the paper).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

ARCH_NAMES = [
    "baseline",
    "deeplab",
    "siamese",
    "siamese_fcn_conc",
    "siamese_fcn_diff",
    "change_transformer",
]

SEEDS = [42, 17, 2026]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Event-cluster block bootstrap CIs for Table 4 architectures."
    )
    parser.add_argument(
        "--patch-index",
        type=Path,
        default=ROOT / "data" / "patches" / "australia_full" / "patch_index.csv",
    )
    parser.add_argument(
        "--eval-root",
        type=Path,
        default=ROOT / "data" / "evaluation",
        help="Root containing australia_full_seed<S>_reflectance64/<arch>_patch_metrics.csv.",
    )
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=ROOT / "data" / "paper_assets" / "tables" / "arch_event_block_bootstrap.csv",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=ROOT / "data" / "paper_assets" / "tables" / "arch_event_block_bootstrap.json",
    )
    return parser.parse_args()


def load_arch_pooled_patches(eval_root: Path, patch_index: pd.DataFrame, arch: str) -> pd.DataFrame:
    """Load and seed-pool per-patch tp/fp/fn for one architecture.

    Returns a DataFrame keyed by patch index (matching patch_index.csv row order),
    with tp/fp/fn summed over the 3 seeds. This makes the event-cluster bootstrap
    operate on pixel-pooled-over-seeds counts, matching the headline aggregation.
    """
    n_patches = len(patch_index)
    tp = np.zeros(n_patches, dtype=np.float64)
    fp = np.zeros(n_patches, dtype=np.float64)
    fn = np.zeros(n_patches, dtype=np.float64)
    iou_patch = np.zeros((len(SEEDS), n_patches), dtype=np.float64)
    n_loaded = 0
    for seed in SEEDS:
        csv = eval_root / f"australia_full_seed{seed}_reflectance64" / f"{arch}_patch_metrics.csv"
        if not csv.exists():
            raise FileNotFoundError(csv)
        df = pd.read_csv(csv)
        assert len(df) == n_patches, f"{csv} has {len(df)} rows, expected {n_patches}"
        tp += df["tp"].values
        fp += df["fp"].values
        fn += df["fn"].values
        iou_patch[n_loaded] = df["iou"].values
        n_loaded += 1
    return pd.DataFrame(
        {
            "event_id": patch_index["event_id"].values,
            "sensor": patch_index["sensor"].values,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "iou_patch_seedmean": iou_patch.mean(axis=0),
        }
    )


def pixel_pooled_iou(tp: np.ndarray, fp: np.ndarray, fn: np.ndarray) -> float:
    denom = tp.sum() + fp.sum() + fn.sum()
    if denom == 0:
        return float("nan")
    return float(tp.sum() / denom)


def pixel_pooled_f1(tp: np.ndarray, fp: np.ndarray, fn: np.ndarray) -> float:
    denom = 2 * tp.sum() + fp.sum() + fn.sum()
    if denom == 0:
        return float("nan")
    return float(2 * tp.sum() / denom)


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    patch_index = pd.read_csv(args.patch_index)
    events = sorted(patch_index["event_id"].unique().tolist())
    n_events = len(events)

    results = []
    json_dump = {"n_bootstrap": args.n_bootstrap, "events": events, "by_model": {}}

    for arch in ARCH_NAMES:
        pooled = load_arch_pooled_patches(args.eval_root, patch_index, arch)
        # Precompute per-event tp/fp/fn so the inner loop is cheap.
        per_event = pooled.groupby("event_id").agg(
            tp=("tp", "sum"),
            fp=("fp", "sum"),
            fn=("fn", "sum"),
            iou_patch_seedmean_mean=("iou_patch_seedmean", "mean"),
            n_patches=("tp", "size"),
        )
        per_event = per_event.reindex(events)

        # Headline point estimates (over all events, pixel-pooled).
        iou_point = pixel_pooled_iou(per_event["tp"].values, per_event["fp"].values, per_event["fn"].values)
        f1_point = pixel_pooled_f1(per_event["tp"].values, per_event["fp"].values, per_event["fn"].values)
        iou_patch_point = float(pooled["iou_patch_seedmean"].mean())

        # Bootstrap: resample whole events with replacement.
        iou_boot = np.empty(args.n_bootstrap, dtype=np.float64)
        f1_boot = np.empty(args.n_bootstrap, dtype=np.float64)
        iou_patch_boot = np.empty(args.n_bootstrap, dtype=np.float64)
        # Build event -> rows mapping once for the patch-mean resample.
        rows_by_event = {ev: pooled.index[pooled["event_id"] == ev].to_numpy() for ev in events}

        for b in range(args.n_bootstrap):
            idx = rng.integers(0, n_events, size=n_events)
            samp_events = [events[i] for i in idx]
            tp_b = per_event["tp"].values[idx].sum()
            fp_b = per_event["fp"].values[idx].sum()
            fn_b = per_event["fn"].values[idx].sum()
            denom_iou = tp_b + fp_b + fn_b
            denom_f1 = 2 * tp_b + fp_b + fn_b
            iou_boot[b] = (tp_b / denom_iou) if denom_iou > 0 else np.nan
            f1_boot[b] = (2 * tp_b / denom_f1) if denom_f1 > 0 else np.nan
            # patch-mean IoU on the same event-cluster resample.
            sampled_rows = np.concatenate([rows_by_event[ev] for ev in samp_events])
            iou_patch_boot[b] = pooled["iou_patch_seedmean"].values[sampled_rows].mean()

        ci_lo_iou, ci_hi_iou = np.nanpercentile(iou_boot, [2.5, 97.5])
        ci_lo_f1, ci_hi_f1 = np.nanpercentile(f1_boot, [2.5, 97.5])
        ci_lo_pat, ci_hi_pat = np.nanpercentile(iou_patch_boot, [2.5, 97.5])

        results.append(
            {
                "model": arch,
                "n_patches_per_seed": int(len(pooled) // len(SEEDS) * len(SEEDS) / len(SEEDS)),
                "iou_pooled": iou_point,
                "iou_ci_low": float(ci_lo_iou),
                "iou_ci_high": float(ci_hi_iou),
                "f1_pooled": f1_point,
                "f1_ci_low": float(ci_lo_f1),
                "f1_ci_high": float(ci_hi_f1),
                "iou_patch_mean": iou_patch_point,
                "iou_patch_ci_low": float(ci_lo_pat),
                "iou_patch_ci_high": float(ci_hi_pat),
            }
        )
        json_dump["by_model"][arch] = results[-1]
        print(
            f"{arch:22s}  IoU = {iou_point:.4f}  [event-cluster CI95 {ci_lo_iou:.4f}, {ci_hi_iou:.4f}]  "
            f"F1 = {f1_point:.4f}  IoU_patch = {iou_patch_point:.4f}"
        )

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(args.out_csv, index=False)
    args.out_json.write_text(json.dumps(json_dump, indent=2))
    print(f"\nWrote {args.out_csv}")
    print(f"Wrote {args.out_json}")


if __name__ == "__main__":
    main()

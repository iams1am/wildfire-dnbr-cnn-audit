"""Percentile bootstrap CI on the EMS valid-pair
mean delta (DeepLab IoU minus dNBR IoU) across the 5 valid Copernicus EMS
event-sensor comparisons.

This gives a CI on the headline "-0.010" mean delta that honestly reports
the small-n uncertainty without needing a second seed run.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CSV = PROJECT_ROOT / "data" / "paper_assets" / "tables" / "copernicus_ems_validation.csv"
OUT = PROJECT_ROOT / "data" / "paper_assets" / "tables" / "copernicus_ems_bootstrap_ci.json"


def main() -> None:
    df = pd.read_csv(CSV)
    df = df.dropna(subset=["iou"]).copy()
    # Build paired (dNBR vs CNN) deltas per (event_id, sensor).
    pairs: list[tuple[str, str, float, float, float]] = []
    for (event, sensor), grp in df.groupby(["event_id", "sensor"]):
        dnbr_row = grp[grp["comparison"].str.startswith("dnbr_vs_")]
        cnn_row = grp[grp["comparison"].str.startswith("deeplab_vs_")]
        if dnbr_row.empty or cnn_row.empty:
            continue
        dnbr_iou = float(dnbr_row["iou"].iloc[0])
        cnn_iou = float(cnn_row["iou"].iloc[0])
        delta = cnn_iou - dnbr_iou
        pairs.append((event, sensor, dnbr_iou, cnn_iou, delta))

    df_pairs = pd.DataFrame(pairs, columns=["event_id", "sensor", "dnbr_iou", "deeplab_iou", "delta"])
    print("Paired EMS comparisons:")
    print(df_pairs.to_string(index=False))

    deltas = df_pairs["delta"].to_numpy()
    n = deltas.size
    rng = np.random.default_rng(42)
    boot = rng.choice(deltas, size=(20000, n), replace=True).mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    boot_se = float(boot.std(ddof=1))
    mean = float(deltas.mean())
    # Sign test: how many of the 20000 bootstrap means are negative?
    p_neg = float((boot < 0).mean())
    # Wilcoxon signed-rank, two-sided
    try:
        from scipy.stats import wilcoxon
        wstat = wilcoxon(deltas, alternative="two-sided") if n >= 2 else None
        wilcoxon_p = float(wstat.pvalue) if wstat is not None else None
    except Exception:
        wilcoxon_p = None

    summary = {
        "n_valid_pairs": int(n),
        "mean_delta": round(mean, 4),
        "bootstrap_se": round(boot_se, 4),
        "bootstrap_ci95_lo": round(float(lo), 4),
        "bootstrap_ci95_hi": round(float(hi), 4),
        "bootstrap_p_lt0": round(p_neg, 4),
        "wilcoxon_p_two_sided": round(wilcoxon_p, 4) if wilcoxon_p is not None else None,
        "pairs": df_pairs.round(4).to_dict(orient="records"),
        "bootstrap_n_resamples": 20000,
    }
    print("\n=== Bootstrap summary ===")
    for k, v in summary.items():
        if k not in ("pairs",):
            print(f"  {k:24s}: {v}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()

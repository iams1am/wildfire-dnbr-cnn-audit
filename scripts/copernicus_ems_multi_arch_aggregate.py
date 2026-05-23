"""Re-aggregate the per-architecture EMS validation long CSV into a per-event
matrix with one column per architecture. The original aggregation in
copernicus_ems_multi_arch.py mis-filtered rows (both dnbr_vs_<tag> and
deeplab_vs_<tag> contained the arch name in `comparison`, so `.iloc[0]`
returned the dNBR row). This script repairs the matrix.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LONG = ROOT / "data" / "paper_assets" / "tables" / "copernicus_ems_multi_arch_long.csv"
ORIGINAL_EMS = ROOT / "data" / "paper_assets" / "tables" / "copernicus_ems_validation.csv"
OUT = ROOT / "data" / "paper_assets" / "tables" / "copernicus_ems_multi_arch.csv"
ARCHS = ["baseline", "siamese"]


def main() -> None:
    df_long = pd.read_csv(LONG).dropna(subset=["iou"])
    df_orig = pd.read_csv(ORIGINAL_EMS).dropna(subset=["iou"])

    dnbr_rows = (
        df_orig[df_orig["comparison"].str.startswith("dnbr_vs_")]
        .drop_duplicates(subset=["event_id", "sensor"])
        [["event_id", "sensor", "source", "iou"]]
        .rename(columns={"iou": "dnbr_iou"})
    )

    deeplab_rows = (
        df_orig[df_orig["comparison"].str.startswith("deeplab_vs_")]
        [["event_id", "sensor", "iou"]]
        .rename(columns={"iou": "deeplab_iou"})
    )


    cnn_per_arch: dict[str, pd.DataFrame] = {}
    for arch in ARCHS:
        sub = df_long[(df_long["arch"] == arch) & df_long["comparison"].str.startswith("deeplab_vs_")]
        col = f"{arch}_iou"
        cnn_per_arch[arch] = sub[["event_id", "sensor", "iou"]].rename(columns={"iou": col})

    summary = dnbr_rows.copy()
    summary = summary.merge(deeplab_rows, on=["event_id", "sensor"], how="left")
    summary["deeplab_delta"] = (summary["deeplab_iou"] - summary["dnbr_iou"]).round(4)
    for arch in ARCHS:
        summary = summary.merge(cnn_per_arch[arch], on=["event_id", "sensor"], how="left")
        summary[f"{arch}_delta"] = (summary[f"{arch}_iou"] - summary["dnbr_iou"]).round(4)

    for col in summary.columns:
        if summary[col].dtype.kind == "f":
            summary[col] = summary[col].round(4)

    mean_row = {"event_id": "VALID-PAIR MEAN", "sensor": "--", "source": "--",
                "dnbr_iou": round(float(summary["dnbr_iou"].mean()), 4),
                "deeplab_iou": round(float(summary["deeplab_iou"].mean()), 4)}
    mean_row["deeplab_delta"] = round(mean_row["deeplab_iou"] - mean_row["dnbr_iou"], 4)
    for arch in ARCHS:
        mean_row[f"{arch}_iou"] = round(float(summary[f"{arch}_iou"].mean()), 4)
        mean_row[f"{arch}_delta"] = round(mean_row[f"{arch}_iou"] - mean_row["dnbr_iou"], 4)
    summary = pd.concat([summary, pd.DataFrame([mean_row])], ignore_index=True)

    col_order = (
        ["event_id", "sensor", "source", "dnbr_iou"]
        + [c for arch in ["baseline", "siamese", "deeplab"] for c in (f"{arch}_iou", f"{arch}_delta")]
    )
    col_order = [c for c in col_order if c in summary.columns]
    summary = summary[col_order]

    summary.to_csv(OUT, index=False)
    md_path = OUT.with_suffix(".md")
    md_path.write_text(summary.to_markdown(index=False), encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"\nWrote {OUT}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()

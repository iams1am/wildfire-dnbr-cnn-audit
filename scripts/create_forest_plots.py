"""Create two forest-plot figures:

  (a) Australia cross-region IoU with event-cluster 95% CIs for the 6
      architectures. Source: arch_event_block_bootstrap.csv.

  (b) MCD64A1 independent dNBR-recall by event. Source:
      mcd64a1_independent_validation.csv.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = PROJECT_ROOT / "data" / "paper_assets" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

ARCH_PRETTY = {
    "baseline": "Concat U-Net",
    "deeplab": "DeepLabv3+-Light",
    "siamese": "Siamese U-Net",
    "siamese_fcn_conc": "Siamese-FCN-Conc",
    "siamese_fcn_diff": "Siamese-FCN-Diff",
    "change_transformer": "BIT transformer",
}


def forest_architectures() -> Path:
    csv = PROJECT_ROOT / "data" / "paper_assets" / "tables" / "arch_event_block_bootstrap.csv"
    df = pd.read_csv(csv)
    df["model_pretty"] = df["model"].map(ARCH_PRETTY)
    df = df.sort_values("iou_pooled", ascending=True)

    fig, ax = plt.subplots(figsize=(8, 3.6))
    y = np.arange(len(df))
    iou = df["iou_pooled"].values
    lo = df["iou_ci_low"].values
    hi = df["iou_ci_high"].values
    err = np.stack([iou - lo, hi - iou], axis=0)

    # Use DeepLabv3+ as the visual highlight color, others muted.
    colors = ["#444"] * len(df)
    for i, m in enumerate(df["model"].values):
        if m == "deeplab":
            colors[i] = "#1f77b4"
        elif m == "baseline":
            colors[i] = "#2ca02c"

    ax.errorbar(iou, y, xerr=err, fmt="o", capsize=4, color="black", ecolor="gray", lw=1.0)
    for i, c in enumerate(colors):
        ax.plot(iou[i], y[i], "o", color=c, markersize=7)
    ax.set_yticks(y)
    ax.set_yticklabels(df["model_pretty"].values)
    ax.set_xlabel(r"Cross-region pixel-pooled IoU (Australia, QA-on) with event-cluster 95\% CI")
    ax.axvline(0.93, ls="--", color="black", lw=0.8, alpha=0.4)
    ax.text(0.93, len(df) - 0.3, "dNBR (label-circular)", rotation=90, va="top", ha="right", fontsize=8, alpha=0.7)
    ax.grid(axis="x", linestyle=":", alpha=0.4)
    ax.set_xlim(0.0, 1.0)
    fig.tight_layout()
    out = FIG_DIR / "forest_architectures.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")
    return out


def forest_mcd64a1() -> Path:
    csv = PROJECT_ROOT / "data" / "paper_assets" / "tables" / "mcd64a1_independent_validation.csv"
    df = pd.read_csv(csv)
    df = df[df["comparison"] == "dnbr_vs_mcd64a1_30m_recall"].copy()
    df = df.dropna(subset=["recall"])
    df["label"] = df["region"].str.title() + ": " + df["event_id"].str.replace("_", " ", regex=False)
    df = df.sort_values(["region", "recall"])

    fig, ax = plt.subplots(figsize=(8, 0.4 * len(df) + 1.5))
    y = np.arange(len(df))
    rec = df["recall"].values
    colors = ["#1f77b4" if r == "mediterranean" else "#d62728" for r in df["region"]]
    ax.barh(y, rec, color=colors, alpha=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels(df["label"].values)
    ax.set_xlabel("dNBR recall against MCD64A1 (30 m)")
    ax.set_xlim(0.0, 1.0)
    for yi, ri in zip(y, rec):
        ax.text(min(ri + 0.02, 0.95), yi, f"{ri:.2f}", va="center", fontsize=9)
    ax.grid(axis="x", linestyle=":", alpha=0.4)

    from matplotlib.patches import Patch
    legend = [
        Patch(color="#1f77b4", label="Mediterranean"),
        Patch(color="#d62728", label="Australia"),
    ]
    ax.legend(handles=legend, loc="lower right", fontsize=9)
    fig.tight_layout()
    out = FIG_DIR / "forest_mcd64a1.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")
    return out


def main() -> None:
    forest_architectures()
    forest_mcd64a1()


if __name__ == "__main__":
    main()

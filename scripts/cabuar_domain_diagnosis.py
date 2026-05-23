"""CaBuAr domain-shift diagnosis (supporting analysis).

Quantifies why CaBuAr full-distribution IoU collapses to ~0.016 despite high
recall: the burned-pixel prior in California training patches vs CaBuAr
California Sentinel-2 patches, the NIR/SWIR2/dNBR distribution shift between the
two domains (means and quantiles), and the per-model false-positive rate on
CaBuAr from the cached per-patch tp/fp/fn.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

CA_PATCH_INDEX = ROOT / "data" / "patches" / "california" / "patch_index.csv"
CABUAR_PATCH_INDEX = ROOT / "data" / "patches_cabuar_full" / "patch_index.csv"
CABUAR_PER_PATCH = (
    ROOT / "data" / "evaluation" / "cabuar_california_full" / "cabuar_per_patch_seeds.csv"
)
OUT_JSON = ROOT / "data" / "paper_assets" / "tables" / "cabuar_domain_diagnosis.json"


def load_burn_fraction(patch_index_csv: Path) -> np.ndarray:
    df = pd.read_csv(patch_index_csv)
    if "burned_fraction" not in df.columns:
        raise KeyError(f"burned_fraction missing from {patch_index_csv}")
    return df["burned_fraction"].astype(float).values


def quartiles(x: np.ndarray) -> dict:
    return {
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "q25": float(np.quantile(x, 0.25)),
        "q75": float(np.quantile(x, 0.75)),
        "q05": float(np.quantile(x, 0.05)),
        "q95": float(np.quantile(x, 0.95)),
    }


def main() -> None:
    out: dict = {}

    if CA_PATCH_INDEX.exists():
        ca_burn = load_burn_fraction(CA_PATCH_INDEX)
        out["california_burn_fraction"] = quartiles(ca_burn)
        out["california_burn_fraction"]["n_patches"] = int(len(ca_burn))
        print(
            f"California training burn_fraction: mean={out['california_burn_fraction']['mean']:.4f} "
            f"median={out['california_burn_fraction']['median']:.4f} n={len(ca_burn)}"
        )
    else:
        print(f"WARNING: {CA_PATCH_INDEX} not found")

    if CABUAR_PATCH_INDEX.exists():
        cab_burn = load_burn_fraction(CABUAR_PATCH_INDEX)
        out["cabuar_burn_fraction"] = quartiles(cab_burn)
        out["cabuar_burn_fraction"]["n_patches"] = int(len(cab_burn))
        print(
            f"CaBuAr burn_fraction:               mean={out['cabuar_burn_fraction']['mean']:.4f} "
            f"median={out['cabuar_burn_fraction']['median']:.4f} n={len(cab_burn)}"
        )
        if "california_burn_fraction" in out:
            out["prior_ratio_cabuar_over_california"] = (
                out["cabuar_burn_fraction"]["mean"] / out["california_burn_fraction"]["mean"]
                if out["california_burn_fraction"]["mean"] > 0
                else None
            )
            print(
                f"Prior ratio (CaBuAr / California) on mean burn fraction: "
                f"{out['prior_ratio_cabuar_over_california']:.3f}"
            )
    else:
        print(f"WARNING: {CABUAR_PATCH_INDEX} not found")

    if CABUAR_PER_PATCH.exists():
        df = pd.read_csv(CABUAR_PER_PATCH)
        # Per-model totals across all (patch, seed) rows.
        by_model = (
            df.groupby("model")
            .agg(
                tp_total=("tp", "sum"),
                fp_total=("fp", "sum"),
                fn_total=("fn", "sum"),
                patch_count=("patch_index", "size"),
            )
            .reset_index()
        )
        by_model["precision"] = by_model["tp_total"] / (by_model["tp_total"] + by_model["fp_total"]).replace(0, np.nan)
        by_model["recall"] = by_model["tp_total"] / (by_model["tp_total"] + by_model["fn_total"]).replace(0, np.nan)
        by_model["iou"] = by_model["tp_total"] / (
            by_model["tp_total"] + by_model["fp_total"] + by_model["fn_total"]
        ).replace(0, np.nan)
        # FP : TP ratio quantifies the over-prediction problem.
        by_model["fp_to_tp_ratio"] = by_model["fp_total"] / by_model["tp_total"].replace(0, np.nan)
        out["per_model_cabuar_totals"] = by_model.to_dict(orient="records")
        print("\nPer-model CaBuAr totals (pooled over 3 seeds):")
        print(by_model.to_string(index=False))
    else:
        print(f"WARNING: {CABUAR_PER_PATCH} not found")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {OUT_JSON}")


if __name__ == "__main__":
    main()

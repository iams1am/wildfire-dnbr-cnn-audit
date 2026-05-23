"""Per-product perimeter-area calibration plot.

Predicted burned area (km^2) vs agency-reference burned area (km^2),
log-log, one panel per independent product (MTBS, EMS, NSW), with the
y=x identity line and the dNBR-rule reference.

This shows whether the CNN systematically over- or under-estimates burn extent
on the perimeter-reference products, which IoU does not directly expose.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MTBS_DEEP = PROJECT_ROOT / "data/paper_assets/mtbs_model_validation_reflectance64/deeplab_kfold_mtbs_per_pair.csv"
MTBS_BASE = PROJECT_ROOT / "data/paper_assets/mtbs_model_validation_reflectance64/baseline_kfold_mtbs_per_pair.csv"
EMS_CSV = PROJECT_ROOT / "data/paper_assets/tables/copernicus_ems_validation.csv"
NSW_CSV = PROJECT_ROOT / "data/paper_assets/tables/nsw_fire_history_validation.csv"
OUT_DIR = PROJECT_ROOT / "data/paper_assets/figures"
PIXEL_SQ_KM = 30 * 30 / 1e6  # 30 m EPSG:6933 cell area


def _scatter_panel(ax, agency_km2, dnbr_km2, cnn_km2, title: str) -> None:
    agency = np.asarray(agency_km2, dtype=float)
    dnbr = np.asarray(dnbr_km2, dtype=float)
    cnn = np.asarray(cnn_km2, dtype=float)
    mask = (agency > 0) & (dnbr > 0) & np.isfinite(agency) & np.isfinite(dnbr) & np.isfinite(cnn) & (cnn > 0)
    agency, dnbr, cnn = agency[mask], dnbr[mask], cnn[mask]
    if agency.size == 0:
        ax.text(0.5, 0.5, "no valid pairs", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        return
    lo = min(agency.min(), dnbr.min(), cnn.min()) * 0.5
    hi = max(agency.max(), dnbr.max(), cnn.max()) * 2.0
    grid = np.array([lo, hi])
    ax.plot(grid, grid, color="black", linestyle="--", linewidth=1, label="y = x (perfect)")
    ax.scatter(agency, dnbr, marker="o", s=42, facecolors="none", edgecolors="#1f77b4",
               linewidths=1.4, label="dNBR rule")
    ax.scatter(agency, cnn, marker="^", s=42, color="#d62728", alpha=0.85, label="DeepLabv3+")
    # Median bias factors
    dnbr_log_bias = float(np.median(np.log10(dnbr / agency)))
    cnn_log_bias = float(np.median(np.log10(cnn / agency)))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Agency burned area (km$^2$)")
    ax.set_ylabel("Predicted burned area (km$^2$)")
    ax.set_title(
        f"{title}\nmedian log10 bias: dNBR {dnbr_log_bias:+.2f}, DeepLab {cnn_log_bias:+.2f} "
        f"(n={agency.size})",
        fontsize=10,
    )
    ax.grid(True, which="both", linestyle=":", alpha=0.4)
    ax.legend(loc="upper left", fontsize=8)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.3))

    # ---- MTBS panel -----------------------------------------------------
    mtbs_d = pd.read_csv(MTBS_DEEP)
    mtbs_b = pd.read_csv(MTBS_BASE)
    # Use the QA-valid in-AOI MTBS area as the agency reference.
    mtbs_d = mtbs_d[mtbs_d["usable_for_iou"] == True].copy()
    mtbs_b = mtbs_b[mtbs_b["usable_for_iou"] == True].copy()
    # Average over seeds per (event, sensor)
    grp_d = mtbs_d.groupby(["event_id", "sensor"]).agg(
        agency=("mtbs_in_valid_sq_km", "mean"),
        dnbr=("dnbr_area_sq_km", "mean"),
        cnn=("model_area_sq_km", "mean"),
    ).reset_index()
    _scatter_panel(axes[0], grp_d["agency"], grp_d["dnbr"], grp_d["cnn"],
                   "MTBS California (in-domain perimeter)")

    # ---- EMS panel ------------------------------------------------------
    ems = pd.read_csv(EMS_CSV)
    # dNBR row has comparison column == "dnbr_vs_emsn159" etc; pred row has model-tag in comparison
    # Use dNBR rows: pred_burned_pixels × PIXEL area gives dNBR area; agency = ems_burned_area_sq_km
    # And for the CNN, we use the per-event predicted area aggregated from the bundled multi_arch CSV.
    ems_dnbr = ems[ems["comparison"].str.contains("dnbr_vs", na=False)].copy()
    ems_dnbr["pred_km2"] = ems_dnbr["pred_burned_pixels"] * PIXEL_SQ_KM
    # CNN per-event area: read multi_arch_3seed_long to get model-predicted areas
    ems_long_path = PROJECT_ROOT / "data/paper_assets/tables/copernicus_ems_multi_arch_3seed_long.csv"
    if ems_long_path.exists():
        ems_long = pd.read_csv(ems_long_path)
        # Pick rows where the model under test is DeepLab AND the row reports the CNN prediction
        # (comparison column begins with "deeplab_vs_" rather than "dnbr_vs_").
        deep_rows = ems_long[
            (ems_long["arch"].str.lower() == "deeplab")
            & ems_long["comparison"].str.startswith("deeplab_vs_", na=False)
        ]
        if not deep_rows.empty:
            deep_grp = deep_rows.groupby(["event_id", "sensor"])["pred_burned_pixels"].mean().reset_index()
            deep_grp["cnn_km2"] = deep_grp["pred_burned_pixels"] * PIXEL_SQ_KM
            merged = ems_dnbr.merge(deep_grp[["event_id", "sensor", "cnn_km2"]], on=["event_id", "sensor"], how="left")
        else:
            merged = ems_dnbr.copy()
            merged["cnn_km2"] = merged["pred_km2"]
    else:
        merged = ems_dnbr.copy()
        merged["cnn_km2"] = merged["pred_km2"]
    _scatter_panel(axes[1], merged["ems_burned_area_sq_km"], merged["pred_km2"], merged["cnn_km2"],
                   "Copernicus EMS (out-of-domain Mediterranean)")

    # ---- NSW panel ------------------------------------------------------
    nsw = pd.read_csv(NSW_CSV)
    nsw_dnbr = nsw[nsw["comparison"] == "dnbr_vs_nsw_fire_history"].copy()
    nsw_cnn = nsw[nsw["comparison"] == "deeplab_vs_nsw_fire_history"].copy()
    nsw_dnbr["dnbr_km2"] = nsw_dnbr["pred_burned_pixels"] * PIXEL_SQ_KM
    nsw_cnn["cnn_km2"] = nsw_cnn["pred_burned_pixels"] * PIXEL_SQ_KM
    merged = nsw_dnbr.merge(nsw_cnn[["event_id", "sensor", "cnn_km2"]],
                             on=["event_id", "sensor"], how="left")
    _scatter_panel(axes[2], merged["agency_burned_area_sq_km"], merged["dnbr_km2"], merged["cnn_km2"],
                   "NSW Fire History (out-of-domain Australia)")

    plt.suptitle(
        "Perimeter-area calibration: predicted vs. agency-reference burned area per event-sensor pair",
        fontsize=12, y=1.02,
    )
    plt.tight_layout()
    out_png = OUT_DIR / "perimeter_area_calibration.png"
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()

"""A clean event-level map for Rhodes 2023 (Landsat-8) showing
dNBR label + DeepLabv3+ prediction + official Copernicus EMSN159 burn perimeter
overlay, all on the harmonized 30 m EPSG:6933 grid.

Rhodes L8 is the EMS event where DeepLabv3+ wins +0.019 IoU over dNBR, so this
figure visually substantiates the favourable side of the otherwise-mixed EMS
Table~\\ref{tab:copernicus-ems}.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from rasterio.warp import reproject, Resampling

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVENT = "rhodes_fire_2023_landsat8"
PROC = PROJECT_ROOT / "data" / "processed" / EVENT
PRED = PROJECT_ROOT / "data" / "evaluation" / "copernicus_ems_deeplab_stitched_seed42_reflectance64" / EVENT / "prediction_mask.tif"
EMS_GDB = PROJECT_ROOT / "data" / "external" / "copernicus_ems" / "EMSN159" / "EMSN159_STD_UTM35N_v02.gdb"
EMS_BURN_LAYER = "P07WFRmm_WildfireDEL"
EMS_AOI_LAYER = "P00_aoi"
OUT = PROJECT_ROOT / "data" / "paper_assets" / "figures" / "event_map_rhodes_ems.png"


def _load_post_rgb() -> tuple[np.ndarray, rasterio.Affine, str, tuple]:
    with rasterio.open(PROC / "post_harmonized.tif") as src:
        arr = src.read().astype(np.float32) * 2.75e-5 - 0.2
        crs = src.crs.to_string()
        transform = src.transform
        bounds = src.bounds
    swir = np.clip(arr[4], 0, 1)
    nir = np.clip(arr[3], 0, 1)
    red = np.clip(arr[2], 0, 1)
    rgb = np.stack([swir, nir, red], axis=-1)
    lo, hi = np.percentile(rgb, (2, 98))
    rgb = np.clip((rgb - lo) / (hi - lo + 1e-9), 0, 1)
    return rgb, transform, crs, bounds


def _align(src_path: Path, label_path: Path) -> np.ndarray:
    with rasterio.open(label_path) as dst_ds:
        dst_shape = dst_ds.shape
        dst_t = dst_ds.transform
        dst_crs = dst_ds.crs
    with rasterio.open(src_path) as src_ds:
        src_arr = src_ds.read(1).astype(np.float32)
        out = np.zeros(dst_shape, dtype=np.float32)
        reproject(source=src_arr, destination=out,
                  src_transform=src_ds.transform, src_crs=src_ds.crs,
                  dst_transform=dst_t, dst_crs=dst_crs,
                  resampling=Resampling.nearest)
    return out


def _rasterize_vector(gdb: Path, layer: str, ref_path: Path) -> np.ndarray:
    gdf = gpd.read_file(gdb, layer=layer)
    with rasterio.open(ref_path) as ref:
        gdf_ref = gdf.to_crs(ref.crs)
        shapes = [(g, 1) for g in gdf_ref.geometry if g is not None and not g.is_empty]
        if not shapes:
            return np.zeros((ref.height, ref.width), dtype=np.uint8)
        return rasterize(shapes, out_shape=(ref.height, ref.width),
                         transform=ref.transform, fill=0, dtype=np.uint8, all_touched=False)


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    label_path = PROC / "label_harmonized.tif"
    with rasterio.open(label_path) as ls:
        label = ls.read(1).astype(np.uint8)
        bounds = ls.bounds
    H, W = label.shape

    rgb, _, crs, _ = _load_post_rgb()
    pred = (_align(PRED, label_path) > 0.5).astype(np.uint8)
    ems_burn = _rasterize_vector(EMS_GDB, EMS_BURN_LAYER, label_path)
    ems_aoi = _rasterize_vector(EMS_GDB, EMS_AOI_LAYER, label_path)

    extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]
    cmn = dict(extent=extent, origin="upper")

    fig, axes = plt.subplots(1, 4, figsize=(15, 4.5))
    axes[0].imshow(rgb, **cmn)
    axes[0].set_title("(a) Post-event L8 SWIR2-NIR-Red")
    axes[1].imshow(label, cmap="Reds", **cmn)
    axes[1].set_title(f"(b) dNBR label (\\(\\tau{{=}}0.10\\)) IoU=0.824")
    axes[2].imshow(pred, cmap="Blues", **cmn)
    axes[2].set_title("(c) DeepLabv3+ prediction IoU=0.844")

    # Panel (d): Three-way overlay on RGB.
    overlay = np.zeros((H, W, 4), dtype=np.float32)
    overlay[..., 0] = label.astype(np.float32) * 0.85       # dNBR -> red
    overlay[..., 2] = pred.astype(np.float32) * 0.85        # DeepLab -> blue
    overlay[..., 1] = ems_burn.astype(np.float32) * 0.85    # EMS -> green
    overlay[..., 3] = np.maximum.reduce([label, pred, ems_burn]) * 0.55
    axes[3].imshow(rgb, **cmn)
    axes[3].imshow(overlay, **cmn)
    # AOI outline on top.
    axes[3].contour(ems_aoi, levels=[0.5], colors=["white"], linewidths=1.3, extent=extent, origin="upper")
    axes[3].set_title("(d) Overlay: dNBR (red), DLv3+ (blue), EMS (green)")

    for ax in axes:
        ax.set_xlabel(f"Easting (m) [{crs}]", fontsize=8)
        ax.set_ylabel("Northing (m)", fontsize=8)
        ax.ticklabel_format(style="sci", axis="both", scilimits=(0, 0))
        ax.tick_params(labelsize=7)

    # Scale bar 5 km on panel d.
    scale_m = 5000.0
    x0 = bounds.left + (bounds.right - bounds.left) * 0.04
    y0 = bounds.bottom + (bounds.top - bounds.bottom) * 0.06
    axes[3].plot([x0, x0 + scale_m], [y0, y0], color="white", lw=3, solid_capstyle="butt")
    axes[3].text(x0 + scale_m / 2, y0 + (bounds.top - bounds.bottom) * 0.025,
                 "5 km", ha="center", va="bottom", color="white",
                 fontsize=9, bbox=dict(facecolor="black", alpha=0.5, edgecolor="none", pad=2))

    fig.suptitle("Rhodes 2023 (Landsat-8): dNBR, DeepLabv3+, Copernicus EMSN159 wildfire delineation on the 30 m EPSG:6933 grid (DeepLabv3+ \\(\\Delta{{=}}+0.019\\) IoU vs.\\ dNBR)", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUT, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()

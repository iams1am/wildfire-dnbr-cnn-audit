"""One clean event-level map figure showing pre/post imagery +
dNBR label + CNN prediction + independent MCD64A1 perimeter overlay with
scale and coordinates.

Event: Blue Mountains 2019 (Landsat-8). Chosen because the MCD64A1 independent
validation shows 79.5% dNBR-recall on this event (the headline non-Pilbara
Australia case), so the figure visually substantiates the table.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import rasterio
from rasterio.warp import reproject, Resampling

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVENT = "blue_mountains_2019_landsat8"
PROC = PROJECT_ROOT / "data" / "processed_full" / EVENT
PRED = PROJECT_ROOT / "data" / "evaluation" / "australia_full_seed42_reflectance64_stitched" / "deeplab" / EVENT / "prediction_mask.tif"
MCD = PROJECT_ROOT / "data" / "external" / "mcd64a1_tiles" / "mcd64a1_burndate_2019-12_150.3_-34.0_150.95_-33.5.tif"
OUT = PROJECT_ROOT / "data" / "paper_assets" / "figures" / "event_map_blue_mountains.png"


def _load_band_stack(path: Path) -> np.ndarray:
    """Read first 5 bands as (5, H, W) reflectance after sensor scaling.
    Landsat C2L2 scaling: DN * 2.75e-5 - 0.2."""
    with rasterio.open(path) as src:
        arr = src.read().astype(np.float32)
        if arr.shape[0] < 3:
            return arr
        scaled = arr * 2.75e-5 - 0.2
        return np.clip(scaled, 0.0, 1.0)


def _rgb_composite(stack: np.ndarray) -> np.ndarray:
    """Return (H, W, 3) RGB using bands 2,1,0 (red, green, blue) -- our band order
    is blue=0, green=1, red=2, NIR=3, SWIR2=4 -- with mild stretch."""
    r = stack[2]
    g = stack[1]
    b = stack[0]
    rgb = np.stack([r, g, b], axis=-1)
    lo, hi = np.percentile(rgb, (2, 98))
    return np.clip((rgb - lo) / (hi - lo + 1e-9), 0, 1)


def _swir2_rgb(stack: np.ndarray) -> np.ndarray:
    """False-colour SWIR-NIR-red composite that highlights burn scars."""
    swir = stack[4]
    nir = stack[3]
    r = stack[2]
    rgb = np.stack([swir, nir, r], axis=-1)
    lo, hi = np.percentile(rgb, (2, 98))
    return np.clip((rgb - lo) / (hi - lo + 1e-9), 0, 1)


def _align_to(label_src_path: Path, src_path: Path) -> np.ndarray:
    """Reproject src_path onto the grid of label_src_path. Returns a 2D array."""
    with rasterio.open(label_src_path) as dst_ds:
        dst_shape = dst_ds.shape
        dst_transform = dst_ds.transform
        dst_crs = dst_ds.crs
    with rasterio.open(src_path) as src_ds:
        src_arr = src_ds.read(1).astype(np.float32)
        out = np.zeros(dst_shape, dtype=np.float32)
        reproject(
            source=src_arr,
            destination=out,
            src_transform=src_ds.transform,
            src_crs=src_ds.crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.nearest,
        )
    return out


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pre_stack = _load_band_stack(PROC / "pre_harmonized.tif")
    post_stack = _load_band_stack(PROC / "post_harmonized.tif")
    with rasterio.open(PROC / "label_harmonized.tif") as ls:
        label = ls.read(1).astype(np.uint8)
        H, W = label.shape
        bounds = ls.bounds
        crs = ls.crs.to_string()
    pred = _align_to(PROC / "label_harmonized.tif", PRED) > 0.5
    mcd = _align_to(PROC / "label_harmonized.tif", MCD) > 0
    pre_rgb = _rgb_composite(pre_stack)
    post_swir_rgb = _swir2_rgb(post_stack)

    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))

    extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]
    common_kwargs = dict(extent=extent, origin="upper")

    axes[0, 0].imshow(pre_rgb, **common_kwargs)
    axes[0, 0].set_title("(a) Pre-event Landsat-8 RGB (true-color)")
    axes[0, 1].imshow(post_swir_rgb, **common_kwargs)
    axes[0, 1].set_title("(b) Post-event SWIR2-NIR-Red false-color")
    axes[0, 2].imshow(label, cmap="Reds", **common_kwargs)
    axes[0, 2].set_title(f"(c) dNBR label ($\\tau=0.10$): {int(label.sum())/H/W*100:.1f}% burned")

    axes[1, 0].imshow(pred, cmap="Blues", **common_kwargs)
    axes[1, 0].set_title(f"(d) DeepLabv3+ prediction: {int(pred.sum())/H/W*100:.1f}% burned")
    axes[1, 1].imshow(mcd, cmap="Greens", **common_kwargs)
    axes[1, 1].set_title("(e) Independent MCD64A1 burn ($500\\,$m, reprojected)")

    # Panel f: overlay all three.
    overlay = np.zeros((H, W, 4), dtype=np.float32)  # RGBA
    # dNBR (red)
    overlay[..., 0] = label.astype(np.float32) * 0.85
    # DeepLab (blue)
    overlay[..., 2] = pred.astype(np.float32) * 0.85
    # MCD64A1 (green)
    overlay[..., 1] = mcd.astype(np.float32) * 0.85
    # alpha
    overlay[..., 3] = np.maximum.reduce([label, pred.astype(np.uint8), mcd.astype(np.uint8)]) * 0.6
    axes[1, 2].imshow(post_swir_rgb, **common_kwargs)
    axes[1, 2].imshow(overlay, **common_kwargs)
    axes[1, 2].set_title("(f) Overlay: dNBR (red), CNN (blue), MCD64A1 (green)")

    for ax in axes.flat:
        ax.set_xlabel(f"Easting (m) [{crs}]")
        ax.set_ylabel("Northing (m)")
        ax.ticklabel_format(style="sci", axis="both", scilimits=(0, 0))
        ax.tick_params(labelsize=8)

    # Scale bar: 5 km on the lower-left of panel f.
    scale_m = 5000.0
    x0 = bounds.left + (bounds.right - bounds.left) * 0.04
    y0 = bounds.bottom + (bounds.top - bounds.bottom) * 0.06
    axes[1, 2].plot([x0, x0 + scale_m], [y0, y0], color="white", lw=3, solid_capstyle="butt")
    axes[1, 2].text(x0 + scale_m / 2, y0 + (bounds.top - bounds.bottom) * 0.02,
                    "5 km", ha="center", va="bottom", color="white",
                    fontsize=10, bbox=dict(facecolor="black", alpha=0.5, edgecolor="none", pad=2))

    fig.suptitle(f"Blue Mountains 2019 (Landsat-8): three burn delineations on the same {H}x{W} EPSG:6933 grid",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()

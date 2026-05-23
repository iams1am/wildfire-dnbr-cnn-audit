from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))


def _read_stack(path: Path) -> tuple[np.ndarray, rasterio.Affine, object]:
    with rasterio.open(path) as src:
        return src.read().astype(np.float32), src.transform, src.crs


def _stretch(rgb: np.ndarray) -> np.ndarray:
    out = np.zeros_like(rgb, dtype=np.float32)
    for idx in range(rgb.shape[-1]):
        channel = rgb[..., idx]
        finite = np.isfinite(channel)
        positive = finite & (channel > 0)
        sample = channel[positive] if np.any(positive) else channel[finite]
        if sample.size == 0:
            continue
        lo, hi = np.percentile(sample, [2, 98])
        if hi <= lo:
            hi = lo + 1.0
        out[..., idx] = np.clip((channel - lo) / (hi - lo), 0, 1)
    return out


def _rgb(stack: np.ndarray, bands: tuple[int, int, int]) -> np.ndarray:
    rgb = np.stack([stack[b] for b in bands], axis=-1)
    return _stretch(rgb)


def _read_mask(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        return src.read(1) > 0


def _reproject_mask(path: Path, shape: tuple[int, int], transform: rasterio.Affine, crs: object) -> np.ndarray:
    destination = np.zeros(shape, dtype=np.uint8)
    with rasterio.open(path) as src:
        source = (src.read(1) > 0).astype(np.uint8)
        reproject(
            source=source,
            destination=destination,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs=crs,
            resampling=Resampling.nearest,
        )
    return destination > 0


def _fetch_dea_mask(shape: tuple[int, int], transform: rasterio.Affine, crs: object) -> tuple[np.ndarray | None, dict[str, object]]:
    try:
        from dea_independent_validation import EVENTS, fetch_dea_burnt_mask

        event = EVENTS["pilbara_fires_2023"]
        dea_mask, tile_count, tile_records = fetch_dea_burnt_mask(
            event["bbox"],
            event["post_months"],
            shape,
            transform,
            crs,
        )
        if dea_mask is None:
            return None, {"status": "unavailable", "tile_count": int(tile_count), "tile_records": tile_records}
        return dea_mask > 0, {"status": "ok", "tile_count": int(tile_count), "tile_records": tile_records}
    except Exception as exc:  # pragma: no cover - network/source availability is external.
        return None, {"status": "unavailable", "error": str(exc)}


def _show_mask(ax: plt.Axes, mask: np.ndarray, title: str, color: str) -> None:
    ax.imshow(mask, cmap="gray", interpolation="nearest")
    if np.any(mask):
        ax.contour(mask.astype(float), levels=[0.5], colors=color, linewidths=0.6)
    ax.set_title(title, fontsize=8)
    ax.axis("off")


def _safe_contour(ax: plt.Axes, mask: np.ndarray, *, color: str, label: str, linewidth: float = 0.8) -> None:
    if np.any(mask):
        ax.contour(mask.astype(float), levels=[0.5], colors=color, linewidths=linewidth)
        ax.plot([], [], color=color, linewidth=linewidth, label=label)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create the Pilbara Sentinel-2 independent-product failure map.")
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed_full" / "pilbara_fires_2023_sentinel2",
    )
    parser.add_argument(
        "--prediction",
        type=Path,
        default=PROJECT_ROOT
        / "data"
        / "evaluation"
        / "australia_full_seed42_reflectance64_stitched"
        / "deeplab"
        / "pilbara_fires_2023_sentinel2"
        / "prediction_mask.tif",
    )
    parser.add_argument(
        "--mcd64a1",
        type=Path,
        default=PROJECT_ROOT
        / "data"
        / "external"
        / "mcd64a1_tiles"
        / "mcd64a1_burndate_2023-02_117.5_-22.5_120.0_-21.0.tif",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "paper_assets" / "figures" / "event_map_pilbara_failure.png",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=PROJECT_ROOT / "data" / "paper_assets" / "tables" / "pilbara_failure_map_summary.json",
    )
    parser.add_argument("--skip-dea", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pre, _, _ = _read_stack(args.processed_dir / "pre_harmonized_masked.tif")
    post, transform, crs = _read_stack(args.processed_dir / "post_harmonized_masked.tif")
    shape = post.shape[1:]
    label = _read_mask(args.processed_dir / "label_harmonized.tif")
    pred = _read_mask(args.prediction)
    mcd = _reproject_mask(args.mcd64a1, shape, transform, crs)
    dea, dea_status = (None, {"status": "skipped"})
    if not args.skip_dea:
        dea, dea_status = _fetch_dea_mask(shape, transform, crs)

    pre_rgb = _rgb(pre, (2, 1, 0))
    post_false = _rgb(post, (4, 3, 2))

    fig, axes = plt.subplots(2, 3, figsize=(11, 7), constrained_layout=True)
    axes = axes.ravel()
    axes[0].imshow(pre_rgb)
    axes[0].set_title("(a) Pre-fire true color", fontsize=8)
    axes[0].axis("off")
    axes[1].imshow(post_false)
    axes[1].set_title("(b) Post-fire SWIR/NIR/red", fontsize=8)
    axes[1].axis("off")
    _show_mask(axes[2], label, "(c) dNBR label", "red")
    _show_mask(axes[3], pred, "(d) DeepLabv3+ prediction", "blue")
    _show_mask(axes[4], mcd, "(e) MCD64A1 burn mask", "gold")
    axes[5].imshow(post_false)
    _safe_contour(axes[5], label, color="red", label="dNBR")
    _safe_contour(axes[5], pred, color="dodgerblue", label="DeepLabv3+")
    _safe_contour(axes[5], mcd, color="gold", label="MCD64A1")
    if dea is not None:
        _safe_contour(axes[5], dea, color="lime", label="DEA 10 m")
    axes[5].set_title("(f) Independent-product overlay", fontsize=8)
    axes[5].axis("off")
    axes[5].legend(loc="lower right", fontsize=7, frameon=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=220)
    plt.close(fig)

    summary = {
        "event_id": "pilbara_fires_2023_sentinel2",
        "dnbr_pixels": int(np.count_nonzero(label)),
        "deeplab_pixels": int(np.count_nonzero(pred)),
        "mcd64a1_pixels": int(np.count_nonzero(mcd)),
        "dea_status": dea_status,
        "dea_pixels": int(np.count_nonzero(dea)) if dea is not None else None,
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Wrote {args.output}")
    print(f"Wrote {args.summary_json}")


if __name__ == "__main__":
    main()

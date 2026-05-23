"""Per-(event, sensor, arch) bootstrap CI on the CNN-minus-dNBR IoU delta
against the Copernicus EMS official wildfire delineation polygons.

For each pair it loads the dNBR label, the stitched CNN prediction, and the
EMS AOI/burn rasters (all on the 30 m EPSG:6933 grid from
copernicus_ems_validation.py), restricts to the AOI inside the scene footprint,
tiles the valid pixels into ~2 km (64x64) spatial blocks, and resamples whole
blocks with replacement 2,000 times to get a percentile CI on the IoU delta.
Output goes to data/paper_assets/tables/copernicus_ems_per_event_ci.csv.
"""
from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = PROJECT_ROOT / "data" / "manifests" / "mediterranean_external_test_manifest_harmonized_noqa.csv"
EMS_CASES = PROJECT_ROOT / "data" / "external" / "copernicus_ems" / "copernicus_ems_cases.json"
ARCHS = ["deeplab", "baseline", "siamese"]
SEED = 42


def _as_path(value: object) -> Path | None:
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    path = Path(text)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _rasterize_layer(gdf: gpd.GeoDataFrame, ref_path: Path) -> np.ndarray:
    with rasterio.open(ref_path) as ref:
        gdf_ref = gdf.to_crs(ref.crs)
        shapes = [(g, 1) for g in gdf_ref.geometry if g is not None and not g.is_empty]
        if not shapes:
            return np.zeros((ref.height, ref.width), dtype=np.uint8)
        return rasterize(shapes, out_shape=(ref.height, ref.width),
                         transform=ref.transform, fill=0, dtype=np.uint8, all_touched=False)


def _read_vector(path: Path, layer: str | None) -> gpd.GeoDataFrame:
    if layer:
        return gpd.read_file(path, layer=layer)
    return gpd.read_file(path)


def _valid_mask(row: pd.Series, shape: tuple[int, int]) -> np.ndarray:
    valid = np.ones(shape, dtype=bool)
    for col in ("pre_clear_mask_harmonized", "post_clear_mask_harmonized"):
        p = _as_path(row.get(col, ""))
        if p is None or not p.exists():
            continue
        with rasterio.open(p) as src:
            mask = src.read(1)
        if mask.shape != shape:
            raise ValueError(f"{col} shape mismatch")
        valid &= mask > 0
    return valid


def _block_indices(valid: np.ndarray, block: int) -> tuple[np.ndarray, list[np.ndarray]]:
    """Return (block_ids array of shape valid.shape, list of flat pixel indices per block).
    Blocks that contain no valid pixels are dropped."""
    H, W = valid.shape
    block_row = np.arange(H) // block
    block_col = np.arange(W) // block
    n_block_cols = (W + block - 1) // block
    bid = block_row[:, None] * n_block_cols + block_col[None, :]
    flat = bid.ravel()
    valid_flat = valid.ravel()
    used_pixels: dict[int, list[int]] = {}
    for i, b in enumerate(flat):
        if valid_flat[i]:
            used_pixels.setdefault(int(b), []).append(i)
    block_pixel_idx = [np.array(v, dtype=np.int64) for v in used_pixels.values()]
    return bid, block_pixel_idx


def _iou_on_indices(pred: np.ndarray, target: np.ndarray, pixel_idx: np.ndarray) -> float:
    p = pred.ravel()[pixel_idx]
    t = target.ravel()[pixel_idx]
    tp = float(((p > 0) & (t > 0)).sum())
    fp = float(((p > 0) & (t == 0)).sum())
    fn = float(((p == 0) & (t > 0)).sum())
    union = tp + fp + fn
    return tp / union if union > 0 else float("nan")


def bootstrap_ci(
    pred_cnn: np.ndarray,
    pred_dnbr: np.ndarray,
    target: np.ndarray,
    valid: np.ndarray,
    *,
    block: int = 64,
    n_boot: int = 2000,
    rng_seed: int = 42,
) -> dict:
    rng = np.random.default_rng(rng_seed)
    _, block_indices = _block_indices(valid, block)
    n_blocks = len(block_indices)
    if n_blocks < 2:
        return {"delta_mean": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan"),
                "n_blocks": n_blocks}
    # Pre-flatten valid pixels and per-block index arrays.
    full_idx = np.concatenate(block_indices)
    iou_dnbr_full = _iou_on_indices(pred_dnbr, target, full_idx)
    iou_cnn_full = _iou_on_indices(pred_cnn, target, full_idx)
    delta_mean = iou_cnn_full - iou_dnbr_full

    deltas = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        choose = rng.integers(0, n_blocks, size=n_blocks)
        sel = np.concatenate([block_indices[i] for i in choose])
        iou_d = _iou_on_indices(pred_dnbr, target, sel)
        iou_c = _iou_on_indices(pred_cnn, target, sel)
        deltas[b] = iou_c - iou_d
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return {
        "delta_mean": round(float(delta_mean), 4),
        "ci_lo": round(float(lo), 4),
        "ci_hi": round(float(hi), 4),
        "n_blocks": int(n_blocks),
        "iou_dnbr": round(float(iou_dnbr_full), 4),
        "iou_cnn": round(float(iou_cnn_full), 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archs", nargs="+", default=ARCHS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--block-px", type=int, default=64,
                        help="Block side length in pixels (64 px = ~2 km at 30 m).")
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--out-csv", type=Path,
                        default=PROJECT_ROOT / "data" / "paper_assets" / "tables" / "copernicus_ems_per_event_ci.csv")
    args = parser.parse_args()

    manifest = pd.read_csv(MANIFEST)
    cases = json.loads(EMS_CASES.read_text(encoding="utf-8"))
    rows = []
    for case in cases:
        aoi_vec_path = _as_path(case["aoi_vector"])
        burn_vec_path = _as_path(case["burn_vector"])
        aoi_layer = case.get("aoi_layer") or None
        burn_layer = case.get("burn_layer") or None
        aoi_gdf = _read_vector(aoi_vec_path, aoi_layer)
        burn_gdf = _read_vector(burn_vec_path, burn_layer)
        event_id = case["event_id"]
        sub = manifest[manifest["event_id"].astype(str) == event_id]
        for _, mrow in sub.iterrows():
            sensor = str(mrow["sensor"])
            pair_id = str(mrow["pair_id"])
            label_path = _as_path(mrow.get("label_mask_harmonized", ""))
            if label_path is None or not label_path.exists():
                continue
            with rasterio.open(label_path) as ref:
                dnbr_label = ref.read(1).astype(np.uint8)
                shape = dnbr_label.shape
            aoi_raster = _rasterize_layer(aoi_gdf, label_path).astype(bool)
            burn_raster = _rasterize_layer(burn_gdf, label_path).astype(bool)
            burn_raster &= aoi_raster
            valid_scene = _valid_mask(mrow, shape)
            valid = aoi_raster & valid_scene
            if int(valid.sum()) == 0:
                continue

            target = burn_raster.astype(np.uint8)
            for arch in args.archs:
                pred_path = (
                    PROJECT_ROOT / "data" / "evaluation"
                    / f"copernicus_ems_{arch}_stitched_seed{args.seed}_reflectance64"
                    / pair_id / "prediction_mask.tif"
                )
                if not pred_path.exists():
                    print(f"  SKIP {arch} {pair_id}: missing {pred_path}")
                    continue
                with rasterio.open(pred_path) as ps:
                    pred = (ps.read(1) > 0).astype(np.uint8)
                if pred.shape != shape:
                    print(f"  SKIP {arch} {pair_id}: shape mismatch")
                    continue
                ci = bootstrap_ci(
                    pred_cnn=pred, pred_dnbr=dnbr_label, target=target, valid=valid,
                    block=args.block_px, n_boot=args.n_boot, rng_seed=42,
                )
                rec = {
                    "event_id": event_id,
                    "sensor": sensor,
                    "arch": arch,
                    "pair_id": pair_id,
                    **ci,
                }
                rows.append(rec)
                print(f"  {arch} {pair_id}: delta={ci['delta_mean']:+.4f} CI=[{ci['ci_lo']:+.4f},{ci['ci_hi']:+.4f}] n_blocks={ci['n_blocks']}")

    df = pd.DataFrame(rows)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"\nWrote {args.out_csv}")


if __name__ == "__main__":
    main()

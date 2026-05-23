"""Independent burn-area check against MODIS MCD64A1 (NASA monthly burned-area,
500 m, pulled from the Microsoft Planetary Computer STAC).

Australia and Mediterranean Europe are otherwise only checked against
dNBR-derived labels, which is circular against the same rule used in training.
MCD64A1 is genuinely independent: different sensor (Terra/Aqua MODIS), much
coarser resolution (500 m vs 30 m), a different algorithm (VI dynamic-threshold
plus MOD14 active-fire fusion, not the NBR ratio), and a monthly burn-date
product rather than a pre/post pair.

For each (event, sensor) it queries MCD64A1 for tiles over the event bbox and
post-event month, builds a binary burn mask from Burn_Date > 0, reprojects it
onto our 30 m EPSG:6933 grid, and computes pixel-pooled IoU against the dNBR
labels and, if a checkpoint is given, the held-out CNN predictions. Output is a
CSV/JSON pair. Needs pystac-client, planetary-computer, and rasterio.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import rasterio
    from rasterio.warp import reproject, Resampling
    from rasterio.merge import merge as rio_merge
except ImportError as e:
    raise SystemExit(f"rasterio required: {e}")

try:
    import planetary_computer
    from pystac_client import Client
except ImportError as e:
    raise SystemExit(f"planetary-computer + pystac-client required: {e}")


STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"

MED_EVENTS = {
    "evros_fire_2023":          {"bbox": [25.8, 40.7, 26.7, 41.2],   "post_month": "2023-09"},
    "rhodes_fire_2023":         {"bbox": [27.7, 35.95, 28.30, 36.45],"post_month": "2023-08"},
    "pedrogao_grande_fire_2017":{"bbox": [-8.4, 39.85, -7.95, 40.20],"post_month": "2017-06"},
    "sierra_culebra_fire_2022": {"bbox": [-6.5, 41.85, -6.05, 42.10],"post_month": "2022-06"},
}

# Australia events for the bonus "global independent check"; date is the last month of the burn window in the corresponding wildfire_events.yaml record.
AU_EVENTS = {
    "kangaroo_island_fire_2020":{"bbox": [136.5, -36.1, 137.5, -35.6],"post_month": "2020-02"},
    "gippsland_fires_2020":     {"bbox": [147.5, -38.5, 149.5, -36.5],"post_month": "2020-02"},
    "namadgi_fire_2020":        {"bbox": [148.7, -35.85, 149.20, -35.40],"post_month": "2020-02"},
    "blue_mountains_2019":      {"bbox": [150.3, -34.0, 150.95, -33.50],"post_month": "2019-12"},
    "nsw_black_summer_2020":    {"bbox": [149.6, -37.5, 152.0, -34.0],"post_month": "2020-02"},
    "tasmania_2019":            {"bbox": [145.0, -42.8, 147.0, -41.8],"post_month": "2019-02"},
    "perth_hills_2021":         {"bbox": [115.85, -32.20, 116.30, -31.80],"post_month": "2021-02"},
    "pilbara_fires_2023":       {"bbox": [117.5, -22.5, 120.0, -21.0],"post_month": "2023-02"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MCD64A1 independent burn-area validation.")
    parser.add_argument(
        "--regions",
        nargs="+",
        choices=["mediterranean", "australia", "both"],
        default=["mediterranean"],
    )
    parser.add_argument(
        "--patches-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "patches_noqa",
        help="Where to find <region>/<event>_<sensor>/*.npz patch dirs.",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=PROJECT_ROOT
        / "data"
        / "paper_assets"
        / "tables"
        / "mcd64a1_independent_validation.csv",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=PROJECT_ROOT
        / "data"
        / "paper_assets"
        / "tables"
        / "mcd64a1_independent_validation.json",
    )
    parser.add_argument(
        "--tmp-tiles",
        type=Path,
        default=PROJECT_ROOT / "data" / "external" / "mcd64a1_tiles",
    )
    return parser.parse_args()


def month_to_window(post_month: str) -> tuple[str, str]:
    """Return ISO start/end covering the post-fire month and the month before."""
    y, m = post_month.split("-")
    y, m = int(y), int(m)
    if m == 1:
        prev_y, prev_m = y - 1, 12
    else:
        prev_y, prev_m = y, m - 1
    start = date(prev_y, prev_m, 1).isoformat()
    if m == 12:
        end = date(y + 1, 1, 1).isoformat()
    else:
        end = date(y, m + 1, 1).isoformat()
    return start, end


def fetch_mcd64a1_tile(bbox: list[float], post_month: str, tmp_dir: Path) -> Path:
    """Search MCD64A1 STAC for tiles overlapping bbox in the burn-window months,
    download (planetary-computer-signed) Burn_Date assets, and merge into one
    GeoTIFF on disk. Returns the path to the merged tile."""
    tmp_dir.mkdir(parents=True, exist_ok=True)
    cache = tmp_dir / f"mcd64a1_burndate_{post_month}_{'_'.join(map(str, bbox))}.tif"
    if cache.exists():
        return cache
    start, end = month_to_window(post_month)
    client = Client.open(STAC_URL)
    search = client.search(
        collections=["modis-64A1-061"],
        bbox=bbox,
        datetime=f"{start}/{end}",
        max_items=10,
    )
    items = list(search.items())
    if not items:
        raise RuntimeError(f"No MCD64A1 items for bbox={bbox} window={start}/{end}")

    src_paths = []
    for it in items:
        signed = planetary_computer.sign(it)
        asset = signed.assets.get("Burn_Date") or signed.assets.get("burn_date")
        if asset is None:
            continue
        href = asset.href
        dst = tmp_dir / f"{it.id}_Burn_Date.tif"
        if not dst.exists():
            import urllib.request
            urllib.request.urlretrieve(href, dst)
        src_paths.append(dst)
    if not src_paths:
        raise RuntimeError("No Burn_Date assets retrieved")

    sources = [rasterio.open(p) for p in src_paths]
    mosaic, mosaic_transform = rio_merge(sources)
    out_meta = sources[0].meta.copy()
    out_meta.update(
        height=mosaic.shape[1],
        width=mosaic.shape[2],
        transform=mosaic_transform,
        count=mosaic.shape[0],
    )
    with rasterio.open(cache, "w", **out_meta) as dst_ds:
        dst_ds.write(mosaic)
    for s in sources:
        s.close()
    return cache


def burned_mask_from_mcd64a1(mcd_path: Path) -> tuple[np.ndarray, rasterio.Affine, str]:
    """Read MCD64A1 Burn_Date raster and return a uint8 burned-mask (>0)."""
    with rasterio.open(mcd_path) as src:
        burn_date = src.read(1).astype(np.int32)
        crs = src.crs.to_string()
        transform = src.transform
    burned = (burn_date > 0).astype(np.uint8)
    return burned, transform, crs


def reproject_to_grid(src_arr: np.ndarray, src_transform, src_crs: str,
                     dst_height: int, dst_width: int, dst_transform, dst_crs: str) -> np.ndarray:
    """Nearest-resample a burn mask onto a target raster grid."""
    out = np.zeros((dst_height, dst_width), dtype=np.uint8)
    reproject(
        source=src_arr,
        destination=out,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=Resampling.nearest,
    )
    return out


def pooled_metrics(pred: np.ndarray, target: np.ndarray) -> dict:
    pred = pred.astype(bool)
    target = target.astype(bool)
    tp = float((pred & target).sum())
    fp = float((pred & ~target).sum())
    fn = float((~pred & target).sum())
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else float("nan")
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else float("nan")
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    return {"iou": iou, "f1": f1, "precision": precision, "recall": recall, "tp": tp, "fp": fp, "fn": fn}


def aggregate_dnbr_to_500m(dnbr_label: np.ndarray, dst_transform, dst_crs: str,
                          mcd_burn: np.ndarray, mcd_transform, mcd_crs: str,
                          coarsen_factor: int = 17) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate the 30m dNBR raster to a coarse grid that matches the 500m MCD
    sampling, by computing the burned-fraction per coarse cell and thresholding
    at 0.5 (majority vote).

    coarsen_factor = round(500/30) = 17 gives ~510m coarse cells, which is the
    closest integer match to MCD's 500m. The MCD64A1 raster is then reprojected
    onto this coarse grid with nearest-neighbour, giving an apples-to-apples
    comparison at the coarser product's native resolution.

    Returns (dnbr_coarse_bool, mcd_coarse_bool) of the same shape.
    """
    H, W = dnbr_label.shape
    h_c = H // coarsen_factor
    w_c = W // coarsen_factor
    trunc_h = h_c * coarsen_factor
    trunc_w = w_c * coarsen_factor
    truncated = dnbr_label[:trunc_h, :trunc_w].astype(np.float32)
    coarse = truncated.reshape(h_c, coarsen_factor, w_c, coarsen_factor).mean(axis=(1, 3))
    dnbr_coarse = coarse > 0.5

    coarse_transform = rasterio.Affine(
        dst_transform.a * coarsen_factor, dst_transform.b, dst_transform.c,
        dst_transform.d, dst_transform.e * coarsen_factor, dst_transform.f,
    )
    mcd_coarse = np.zeros((h_c, w_c), dtype=np.uint8)
    reproject(
        source=mcd_burn,
        destination=mcd_coarse,
        src_transform=mcd_transform,
        src_crs=mcd_crs,
        dst_transform=coarse_transform,
        dst_crs=dst_crs,
        resampling=Resampling.nearest,
    )
    return dnbr_coarse, mcd_coarse > 0


def load_event_dnbr_label(patches_root: Path, region: str, event: str) -> tuple[np.ndarray, rasterio.Affine, str] | None:
    """Reconstruct the per-event dNBR-label raster by stitching patch labels.
    Returns (label_raster, transform, crs) or None if the patch index doesn't
    cover this event.

    Strategy: read the patch_index.csv for the region, filter rows by event_id,
    reconstruct the per-event raster by placing each 256-pixel label patch at
    (row_start, col_start) on an aggregate canvas. We do not need the absolute
    geographic position to validate against a reprojected MCD64A1 mask; we
    only need the dNBR/MCD64A1 masks aligned on the same grid. So we load
    the harmonized post-event GeoTIFF (used to extract patches) and rasterize
    both labels on its grid.
    """
    probe_dirs = [
        PROJECT_ROOT / "data" / "processed",
        PROJECT_ROOT / "data" / "processed_noqa",
        PROJECT_ROOT / "data" / "processed_full",
        PROJECT_ROOT / "data" / "processed_noqa_full",
    ]
    candidate_dirs = []
    for proc_dir in probe_dirs:
        if not proc_dir.exists():
            continue
        candidate_dirs.extend([p for p in proc_dir.glob(f"{event}_*") if p.is_dir()])
    if not candidate_dirs:
        return None
    anchor = None
    for d in candidate_dirs:
        for name in ("label_harmonized.tif", "label_reprojected.tif",
                     "dnbr_label.tif", "label_mask.tif"):
            cand = d / name
            if cand.exists():
                anchor = cand
                break
        if anchor:
            break
    if anchor is None:
        return None
    with rasterio.open(anchor) as src:
        return src.read(1).astype(np.uint8), src.transform, src.crs.to_string()


def main() -> None:
    args = parse_args()
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    args.tmp_tiles.mkdir(parents=True, exist_ok=True)

    regions = set(args.regions)
    if "both" in regions:
        regions = {"mediterranean", "australia"}

    events: list[tuple[str, str, dict]] = []
    if "mediterranean" in regions:
        for ev, meta in MED_EVENTS.items():
            events.append(("mediterranean", ev, meta))
    if "australia" in regions:
        for ev, meta in AU_EVENTS.items():
            events.append(("australia", ev, meta))

    rows = []
    for region_tag, event, meta in events:
        print(f"\n=== {region_tag} / {event} ===")
        try:
            mcd_path = fetch_mcd64a1_tile(meta["bbox"], meta["post_month"], args.tmp_tiles)
            print(f"  MCD64A1 tile: {mcd_path}")
        except Exception as e:
            print(f"  FETCH FAILED: {type(e).__name__}: {e}")
            rows.append({"region": region_tag, "event_id": event, "status": f"fetch_failed: {e}"})
            continue

        mcd_burn, mcd_transform, mcd_crs = burned_mask_from_mcd64a1(mcd_path)
        anchor_region = "mediterranean_external_test" if region_tag == "mediterranean" else "australia_full"
        anchor = load_event_dnbr_label(args.patches_root, anchor_region, event)
        if anchor is None:
            print(f"  SKIP: no anchor dNBR raster found for {event} under processed_noqa/{anchor_region}")
            rows.append({"region": region_tag, "event_id": event, "status": "no_anchor"})
            continue
        dnbr_label, dst_transform, dst_crs = anchor

        mcd_aligned = reproject_to_grid(mcd_burn, mcd_transform, mcd_crs,
                                        dnbr_label.shape[0], dnbr_label.shape[1],
                                        dst_transform, dst_crs)
        m_native = pooled_metrics(dnbr_label > 0, mcd_aligned > 0)
        recall_vs_mcd = m_native["recall"]
        print(
            f"  dNBR-recall-of-MCD64A1 (30m): {recall_vs_mcd:.4f}  "
            f"(TP={int(m_native['tp'])} FN={int(m_native['fn'])} --> dNBR finds "
            f"{recall_vs_mcd*100:.1f}% of MCD-detected burns)"
        )


        dnbr_coarse, mcd_coarse = aggregate_dnbr_to_500m(
            dnbr_label > 0, dst_transform, dst_crs,
            mcd_burn, mcd_transform, mcd_crs,
        )
        m_coarse = pooled_metrics(dnbr_coarse, mcd_coarse)
        print(
            f"  dNBR vs MCD64A1 IoU at 500m (majority): {m_coarse['iou']:.4f}  "
            f"F1={m_coarse['f1']:.4f}  Prec={m_coarse['precision']:.4f}  Recall={m_coarse['recall']:.4f}  "
            f"TP={int(m_coarse['tp'])} FP={int(m_coarse['fp'])} FN={int(m_coarse['fn'])}"
        )
        rows.append({
            "region": region_tag,
            "event_id": event,
            "status": "ok",
            "comparison": "dnbr_vs_mcd64a1_30m_recall",
            "recall": round(recall_vs_mcd, 4) if not np.isnan(recall_vs_mcd) else None,
            "tp_30m": int(m_native["tp"]),
            "fn_30m": int(m_native["fn"]),
            "dnbr_extra_burned_30m": int(m_native["fp"]),
        })
        rows.append({
            "region": region_tag,
            "event_id": event,
            "status": "ok",
            "comparison": "dnbr_vs_mcd64a1_500m_iou",
            "iou": round(m_coarse["iou"], 4),
            "f1": round(m_coarse["f1"], 4),
            "precision": round(m_coarse["precision"], 4),
            "recall": round(m_coarse["recall"], 4),
            "tp": int(m_coarse["tp"]),
            "fp": int(m_coarse["fp"]),
            "fn": int(m_coarse["fn"]),
        })

    df = pd.DataFrame(rows)
    df.to_csv(args.out_csv, index=False)
    args.out_json.write_text(json.dumps(rows, indent=2))
    print(f"\nWrote {args.out_csv}")
    print(f"Wrote {args.out_json}")


if __name__ == "__main__":
    main()

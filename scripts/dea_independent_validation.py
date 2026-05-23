"""Second independent burn-product check for Australia, using the Digital Earth
Australia Sentinel-2 Burnt Area Collection 3 product (ga_s2_ba_provisional_3)
from the DEA Explorer STAC.

This complements the MCD64A1 check: DEA is Sentinel-2 at native 10 m (so it
registers the small fires MCD64A1's 500 m grid misses), produced operationally
by Geoscience Australia from a delta-NBR/BSI/NDVI fusion that is independent of
our dNBR rule. DEA coverage only starts 2021-07-01, so just two of our eight
Australia events fall in window (Perth Hills 2021, Pilbara 2023); for those we
report dNBR-recall against the DEA mask.

It queries DEA for items over each event bbox and post-event month, reads the
delta-NBR asset over /vsicurl/ without a full download, thresholds at the
product's documented delta_nbr >= 0.27, reprojects to our 30 m EPSG:6933 grid,
and computes recall = TP / (TP + FN_dea). Output is a CSV summary plus a
per-tile table.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import reproject, Resampling

import urllib.request
import xml.etree.ElementTree as ET

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEA_S3_HTTPS = "https://dea-public-data.s3.ap-southeast-2.amazonaws.com"
COLLECTION_PREFIX = "derivative/ga_s2_ba_provisional_3/1-6-0"
# MGRS tile grid squares covering Pilbara 2023 fires (118.80--120.20E, -22.40 to -21.20S).
# Sentinel-2 MGRS UTM zone 50 (Australia west) -- relevant 100km grid squares
# are KQ*, KR*, KP*. We probe all candidates and skip those without coverage.
PILBARA_TILES = [f"50K{c1}{c2}" for c1 in "QRP" for c2 in "ABC"]

# Events in DEA's coverage window (2021-07-01 to 2023-08-29).
EVENTS = {
    # Perth Hills 2021 (Feb 2021) is BEFORE the DEA collection start (2021-07-01),
    # so it cannot be validated against DEA. Kept here only for documentation.
    # "perth_hills_2021": {"bbox": [116.00, -31.95, 116.35, -31.60], "post_months": ["2021-02"]},
    "pilbara_fires_2023": {"bbox": [118.80, -22.40, 120.20, -21.20],
                           "post_months": ["2023-08", "2023-09", "2023-10"]},
}

DEA_NBR_THRESHOLD = 0.27  # operational burnt threshold for delta_nbr (ga_s2_ba_provisional_3)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DEA ga_s2_ba_provisional_3 independent validation.")
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=PROJECT_ROOT / "data" / "paper_assets" / "tables" / "dea_independent_validation.csv",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=PROJECT_ROOT / "data" / "paper_assets" / "tables" / "dea_independent_validation.json",
    )
    return parser.parse_args()


def month_window(post_month: str) -> tuple[str, str]:
    y, m = map(int, post_month.split("-"))
    start = date(y, m, 1) - timedelta(days=15)
    end = (date(y + (m // 12), (m % 12) + 1, 1) if m < 12 else date(y + 1, 1, 1)) + timedelta(days=15)
    return start.isoformat(), end.isoformat()


def s3_to_https(s3_uri: str) -> str:
    # s3://dea-public-data/... -> https://dea-public-data.s3.ap-southeast-2.amazonaws.com/...
    assert s3_uri.startswith("s3://"), s3_uri
    rest = s3_uri[len("s3://"):]
    bucket, key = rest.split("/", 1)
    return f"/vsicurl/https://{bucket}.s3.ap-southeast-2.amazonaws.com/{key}"


def list_dea_tile_deltas(tile: str, year: int, month: int) -> list[str]:
    """List S3 keys for one MGRS tile + year+month, returning the delta-nbr
    asset URLs (`/vsicurl/https://...`). Uses S3 ListObjectsV2 directly so
    DEA STAC's WAF is not on the critical path."""
    prefix = f"{COLLECTION_PREFIX}/{tile[:2]}/{tile[2:]}/{year}/{month:02d}/"
    list_url = f"{DEA_S3_HTTPS}/?list-type=2&prefix={prefix}&max-keys=500"
    try:
        req = urllib.request.Request(list_url, headers={"User-Agent": "wildfire-stac-benchmark/1.0"})
        with urllib.request.urlopen(req, timeout=60) as r:
            body = r.read()
    except Exception as e:
        print(f"  WARN failed to list {tile} {year}-{month:02d}: {e}")
        return []
    root = ET.fromstring(body)
    ns = "{http://s3.amazonaws.com/doc/2006-03-01/}"
    delta_nbr_keys = []
    for c in root.findall(f"{ns}Contents"):
        key = c.find(f"{ns}Key").text
        if key.endswith("_delta-nbr.tif"):
            delta_nbr_keys.append(f"/vsicurl/{DEA_S3_HTTPS}/{key}")
    return delta_nbr_keys


def load_event_anchor(event: str):
    for proc_root_name in ("processed_full", "processed_noqa_full", "processed", "processed_noqa"):
        proc_root = PROJECT_ROOT / "data" / proc_root_name
        if not proc_root.exists():
            continue
        candidates = list(proc_root.glob(f"{event}_*"))
        if not candidates:
            continue
        for d in candidates:
            cand = d / "label_harmonized.tif"
            if cand.exists():
                with rasterio.open(cand) as src:
                    return src.read(1).astype(np.uint8), src.transform, src.crs.to_string()
    return None, None, None


def fetch_dea_burnt_mask(bbox_wgs84, post_months, target_shape, target_transform, target_crs, tiles=None):
    """List DEA Burnt-Area Sentinel-2 tiles directly from S3 across the given
    months, open each delta_nbr via /vsicurl/, read only the windowed region
    overlapping the event bbox, threshold, and union into the target grid.

    Using windowed reads instead of full-tile reads cuts a 482 MB float32
    download per tile to a few tens of MB (the bbox overlap is small)."""
    import rasterio.windows
    from rasterio.warp import transform_bounds
    if tiles is None:
        tiles = PILBARA_TILES
    candidate_urls = []
    for post_month in post_months:
        y, m = map(int, post_month.split("-"))
        for tile in tiles:
            urls = list_dea_tile_deltas(tile, y, m)
            candidate_urls.extend(urls)
    if not candidate_urls:
        return None, 0, []
    print(f"  Found {len(candidate_urls)} DEA delta-nbr tile-acquisitions across {post_months}")

    burnt_target = np.zeros(target_shape, dtype=np.uint8)
    tile_records = []
    for url in candidate_urls:
        try:
            with rasterio.open(url) as src:
                src_t = src.transform
                src_crs = src.crs
                src_bounds_wgs84 = transform_bounds(src_crs, "EPSG:4326", *src.bounds)
                overlap = (max(bbox_wgs84[0], src_bounds_wgs84[0]),
                           max(bbox_wgs84[1], src_bounds_wgs84[1]),
                           min(bbox_wgs84[2], src_bounds_wgs84[2]),
                           min(bbox_wgs84[3], src_bounds_wgs84[3]))
                if overlap[0] >= overlap[2] or overlap[1] >= overlap[3]:
                    continue
                src_win_bounds = transform_bounds("EPSG:4326", src_crs, *overlap)
                win = rasterio.windows.from_bounds(*src_win_bounds, transform=src.transform)
                arr = src.read(1, window=win)
                win_transform = rasterio.windows.transform(win, src.transform)
        except Exception as e:
            print(f"  WARN failed to open {url}: {type(e).__name__}: {e}")
            continue
        burnt_native = ((arr >= DEA_NBR_THRESHOLD) & np.isfinite(arr)).astype(np.uint8)
        if burnt_native.sum() == 0:
            tile_records.append({"url": url, "native_burnt_pixels": 0})
            continue
        tile_records.append({"url": url, "native_burnt_pixels": int(burnt_native.sum())})
        tmp = np.zeros(target_shape, dtype=np.uint8)
        reproject(
            source=burnt_native, destination=tmp,
            src_transform=win_transform, src_crs=src_crs,
            dst_transform=target_transform, dst_crs=target_crs,
            resampling=Resampling.nearest,
        )
        burnt_target |= tmp
        print(f"  + {url.split('/')[-1]}: native burnt={int(burnt_native.sum())}; target cumulative={int(burnt_target.sum())}")

    return burnt_target, int(burnt_target.sum()), tile_records


def pooled_recall(dnbr_label: np.ndarray, dea_burnt: np.ndarray) -> dict:
    pred = dnbr_label > 0
    target = dea_burnt > 0
    tp = float((pred & target).sum())
    fn = float((~pred & target).sum())
    fp = float((pred & ~target).sum())
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    return {"tp": int(tp), "fp": int(fp), "fn": int(fn), "recall": recall}


def main() -> None:
    args = parse_args()
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for event, meta in EVENTS.items():
        print(f"\n=== {event} ===")
        label, transform, crs = load_event_anchor(event)
        if label is None:
            print(f"  SKIP {event}: no anchor raster found under data/processed/")
            rows.append({"event_id": event, "status": "no_anchor"})
            continue
        dea_mask, dea_count, tile_records = fetch_dea_burnt_mask(
            meta["bbox"], meta["post_months"], label.shape, transform, crs
        )
        if dea_mask is None or dea_count == 0:
            print(f"  DEA returned zero burnt pixels (or no tiles) in event bbox.")
            rows.append({"event_id": event, "status": "dea_zero", "dea_pixels_in_event_bbox": dea_count})
            continue
        m = pooled_recall(label, dea_mask)
        print(f"  dNBR-recall-of-DEA at 30 m: {m['recall']:.4f} (TP={m['tp']} FN={m['fn']}; DEA-burnt {dea_count} px in event bbox).")
        rows.append({
            "event_id": event,
            "status": "ok",
            "recall": round(m["recall"], 4),
            "tp": m["tp"],
            "fn": m["fn"],
            "dnbr_only_pixels": m["fp"],
            "dea_pixels_in_event_bbox": dea_count,
            "n_dea_tiles": len(tile_records),
        })

    pd.DataFrame(rows).to_csv(args.out_csv, index=False)
    args.out_json.write_text(json.dumps(rows, indent=2))
    print(f"\nWrote {args.out_csv}")


if __name__ == "__main__":
    main()

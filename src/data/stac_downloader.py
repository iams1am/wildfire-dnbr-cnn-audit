from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote

import numpy as np
import rasterio
import requests
from rasterio.errors import RasterioIOError
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.warp import reproject, transform_bounds
from rasterio.windows import Window, from_bounds, transform as window_transform
from shapely.geometry import box

from src.data.event_config import WildfireEvent


PLANETARY_COMPUTER_SEARCH_API = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
PLANETARY_COMPUTER_SIGN_API = "https://planetarycomputer.microsoft.com/api/sas/v1/sign?href="


@dataclass(frozen=True)
class SensorSpec:
    sensor_name: str
    collection: str
    platform: str | None
    spectral_assets: tuple[str, ...]
    spectral_alias: dict[str, str]
    qa_asset: str
    qa_mode: str


SENSOR_SPECS: dict[str, SensorSpec] = {
    "sentinel2": SensorSpec(
        sensor_name="sentinel2",
        collection="sentinel-2-l2a",
        platform=None,
        spectral_assets=("blue", "green", "red", "nir", "swir22"),
        spectral_alias={
            "blue": "B02",
            "green": "B03",
            "red": "B04",
            "nir": "B08",
            "swir22": "B12",
        },
        qa_asset="SCL",
        qa_mode="sentinel2_scl",
    ),
    "landsat8": SensorSpec(
        sensor_name="landsat8",
        collection="landsat-c2-l2",
        platform="landsat-8",
        spectral_assets=("blue", "green", "red", "nir", "swir22"),
        spectral_alias={"nir": "nir08"},
        qa_asset="qa_pixel",
        qa_mode="landsat_qa_pixel",
    ),
    "landsat9": SensorSpec(
        sensor_name="landsat9",
        collection="landsat-c2-l2",
        platform="landsat-9",
        spectral_assets=("blue", "green", "red", "nir", "swir22"),
        spectral_alias={"nir": "nir08"},
        qa_asset="qa_pixel",
        qa_mode="landsat_qa_pixel",
    ),
}


def _iso_datetime(day: date, *, end_of_day: bool) -> str:
    if end_of_day:
        dt = datetime(day.year, day.month, day.day, 23, 59, 59, tzinfo=timezone.utc)
    else:
        dt = datetime(day.year, day.month, day.day, 0, 0, 0, tzinfo=timezone.utc)
    return dt.isoformat().replace("+00:00", "Z")


def _datetime_range(start_day: date, end_day: date) -> str:
    return f"{_iso_datetime(start_day, end_of_day=False)}/{_iso_datetime(end_day, end_of_day=True)}"


def _item_datetime(item: dict[str, Any]) -> datetime:
    raw = str(item["properties"]["datetime"])
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    return datetime.fromisoformat(raw)


def _cloud_fraction(item: dict[str, Any]) -> float:
    cloud_cover = item["properties"].get("eo:cloud_cover")
    if cloud_cover is None:
        return 1.0
    return float(cloud_cover) / 100.0


def _overlap_fraction(item: dict[str, Any], event_bbox: tuple[float, float, float, float]) -> float:
    item_bbox = item.get("bbox")
    if not isinstance(item_bbox, list) or len(item_bbox) != 4:
        return 0.0
    event_geom = box(*event_bbox)
    item_geom = box(float(item_bbox[0]), float(item_bbox[1]), float(item_bbox[2]), float(item_bbox[3]))
    if event_geom.area <= 0:
        return 0.0
    return float(event_geom.intersection(item_geom).area / event_geom.area)


def _asset_available(item: dict[str, Any], spec: SensorSpec) -> bool:
    assets = item.get("assets", {})
    if not isinstance(assets, dict):
        return False
    required = [spec.spectral_alias.get(name, name) for name in spec.spectral_assets] + [spec.qa_asset]
    return all(key in assets for key in required)


def search_items(
    *,
    spec: SensorSpec,
    bbox: tuple[float, float, float, float],
    start_day: date,
    end_day: date,
    max_items: int = 100,
    max_retries: int = 4,
) -> list[dict[str, Any]]:
    payload: dict[str, Any] = {
        "collections": [spec.collection],
        "bbox": list(bbox),
        "datetime": _datetime_range(start_day, end_day),
        "limit": max_items,
    }
    import time as _time
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            response = requests.post(PLANETARY_COMPUTER_SEARCH_API, json=payload, timeout=120)
            if response.status_code in {429, 502, 503, 504}:
                # Transient - back off (exponential) and retry
                _time.sleep(2 ** attempt)
                continue
            response.raise_for_status()
            feature_collection = response.json()
            features = feature_collection.get("features")
            if not isinstance(features, list):
                raise RuntimeError("STAC search response missing features list.")
            return [item for item in features if isinstance(item, dict)]
        except requests.exceptions.RequestException as exc:
            last_exc = exc
            _time.sleep(2 ** attempt)
            continue
    if last_exc is not None:
        raise last_exc
    raise RuntimeError(f"STAC search returned a transient failure after {max_retries} retries.")


def _select_best_item(
    *,
    items: list[dict[str, Any]],
    spec: SensorSpec,
    event_bbox: tuple[float, float, float, float],
    target_day: date,
    max_cloud_fraction: float,
    min_overlap_fraction: float,
) -> dict[str, Any]:
    base_filtered: list[dict[str, Any]] = []
    for item in items:
        if spec.platform is not None and str(item["properties"].get("platform", "")).lower() != spec.platform:
            continue
        if not _asset_available(item, spec):
            continue
        base_filtered.append(item)
    if not base_filtered:
        raise RuntimeError(f"No STAC items available for sensor={spec.sensor_name} after filtering.")

    filtered = [item for item in base_filtered if _overlap_fraction(item, event_bbox) >= min_overlap_fraction]
    if not filtered:
        filtered = [item for item in base_filtered if _overlap_fraction(item, event_bbox) > 0.05]
    if not filtered:
        raise RuntimeError(
            f"No STAC items overlap event bbox for sensor={spec.sensor_name}. "
            "Consider adjusting event bbox."
        )

    strict_cloud = [item for item in filtered if _cloud_fraction(item) <= max_cloud_fraction]
    candidates = strict_cloud if strict_cloud else filtered

    def score(item: dict[str, Any]) -> tuple[float, float, int]:
        cloud = _cloud_fraction(item)
        overlap = _overlap_fraction(item, event_bbox)
        day_delta = abs((_item_datetime(item).date() - target_day).days)
        return (cloud, -overlap, day_delta)

    return sorted(candidates, key=score)[0]


def select_pre_post_items(
    *,
    event: WildfireEvent,
    spec: SensorSpec,
    pre_window_days: int,
    post_window_days: int,
    max_cloud_fraction: float,
    min_overlap_fraction: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    pre_start = event.start_date - timedelta(days=pre_window_days)
    pre_end = event.start_date - timedelta(days=1)
    post_start = event.end_date + timedelta(days=1)
    post_end = event.end_date + timedelta(days=post_window_days)

    pre_items = search_items(spec=spec, bbox=event.bbox, start_day=pre_start, end_day=pre_end)
    post_items = search_items(spec=spec, bbox=event.bbox, start_day=post_start, end_day=post_end)

    pre_item = _select_best_item(
        items=pre_items,
        spec=spec,
        event_bbox=event.bbox,
        target_day=event.start_date - timedelta(days=1),
        max_cloud_fraction=max_cloud_fraction,
        min_overlap_fraction=min_overlap_fraction,
    )
    post_item = _select_best_item(
        items=post_items,
        spec=spec,
        event_bbox=event.bbox,
        target_day=event.end_date + timedelta(days=1),
        max_cloud_fraction=max_cloud_fraction,
        min_overlap_fraction=min_overlap_fraction,
    )
    return pre_item, post_item


def _intersect_window(src: rasterio.io.DatasetReader, win: Window) -> Window:
    full = Window(0, 0, src.width, src.height)
    intersected = win.intersection(full)
    if intersected.width <= 0 or intersected.height <= 0:
        raise RuntimeError("Computed clip window does not intersect source raster.")
    return intersected.round_offsets().round_lengths()


def clip_asset_to_bbox(
    *,
    href: str,
    bbox_wgs84: tuple[float, float, float, float],
    output_path: Path,
    max_side_px: int = 2048,
    resampling: Resampling = Resampling.bilinear,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for attempt in range(1, 4):
        try:
            with rasterio.open(href) as src:
                bounds_src = transform_bounds("EPSG:4326", src.crs, *bbox_wgs84, densify_pts=21)
                win = from_bounds(*bounds_src, transform=src.transform)
                win = _intersect_window(src, win)
                out_width = int(win.width)
                out_height = int(win.height)
                if max_side_px > 0 and (out_width > max_side_px or out_height > max_side_px):
                    scale = max(out_width / max_side_px, out_height / max_side_px)
                    out_width = max(1, int(round(out_width / scale)))
                    out_height = max(1, int(round(out_height / scale)))
                    data = src.read(
                        1,
                        window=win,
                        out_shape=(out_height, out_width),
                        resampling=resampling,
                    )
                else:
                    data = src.read(1, window=win)

                out_transform: Affine = window_transform(win, src.transform) * Affine.scale(
                    float(win.width) / float(out_width),
                    float(win.height) / float(out_height),
                )

                profile = src.profile.copy()
                profile.update(
                    {
                        "height": out_height,
                        "width": out_width,
                        "transform": out_transform,
                        "count": 1,
                        "compress": "lzw",
                    }
                )
                with rasterio.open(output_path, "w", **profile) as dst:
                    dst.write(data, 1)
            return output_path
        except RasterioIOError as exc:
            last_error = exc
            if attempt == 3:
                break
    raise RuntimeError(f"Failed to clip remote asset after retries: {href}") from last_error


def _sign_href(href: str) -> str:
    sign_url = PLANETARY_COMPUTER_SIGN_API + quote(href, safe="")
    response = requests.get(sign_url, timeout=90)
    response.raise_for_status()
    payload = response.json()
    signed = payload.get("href")
    if not isinstance(signed, str) or not signed:
        raise RuntimeError("Planetary Computer sign API returned invalid response.")
    return signed


def _read_aligned_band(
    path: Path,
    *,
    ref_transform: Affine,
    ref_crs: Any,
    ref_height: int,
    ref_width: int,
    ref_dtype: str,
) -> np.ndarray:
    with rasterio.open(path) as src:
        if (
            src.crs == ref_crs
            and src.transform == ref_transform
            and src.height == ref_height
            and src.width == ref_width
        ):
            return src.read(1)
        destination = np.zeros((ref_height, ref_width), dtype=np.dtype(ref_dtype))
        reproject(
            source=rasterio.band(src, 1),
            destination=destination,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=ref_transform,
            dst_crs=ref_crs,
            resampling=Resampling.bilinear,
        )
        return destination


def build_multiband_stack(
    *,
    band_paths: list[Path],
    band_names: list[str],
    output_path: Path,
) -> Path:
    if len(band_paths) != len(band_names):
        raise ValueError("band_paths and band_names must have the same length.")
    if not band_paths:
        raise ValueError("At least one band path is required to build a stack.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(band_paths[0]) as ref:
        profile = ref.profile.copy()
        profile.update({"count": len(band_paths), "compress": "lzw"})
        ref_transform = ref.transform
        ref_crs = ref.crs
        ref_height = ref.height
        ref_width = ref.width
        ref_dtype = ref.dtypes[0]

        with rasterio.open(output_path, "w", **profile) as dst:
            for idx, band_path in enumerate(band_paths, start=1):
                band = _read_aligned_band(
                    band_path,
                    ref_transform=ref_transform,
                    ref_crs=ref_crs,
                    ref_height=ref_height,
                    ref_width=ref_width,
                    ref_dtype=ref_dtype,
                )
                dst.write(band, idx)
                dst.set_band_description(idx, band_names[idx - 1])
    return output_path


def _align_band_to_reference(
    src_dataset: rasterio.io.DatasetReader,
    band_index: int,
    *,
    ref_transform: Affine,
    ref_crs: Any,
    ref_height: int,
    ref_width: int,
) -> np.ndarray:
    if (
        src_dataset.crs == ref_crs
        and src_dataset.transform == ref_transform
        and src_dataset.height == ref_height
        and src_dataset.width == ref_width
    ):
        return src_dataset.read(band_index).astype(np.float32)

    destination = np.zeros((ref_height, ref_width), dtype=np.float32)
    reproject(
        source=rasterio.band(src_dataset, band_index),
        destination=destination,
        src_transform=src_dataset.transform,
        src_crs=src_dataset.crs,
        dst_transform=ref_transform,
        dst_crs=ref_crs,
        resampling=Resampling.bilinear,
    )
    return destination


def create_dnbr_label(
    *,
    pre_stack_path: Path,
    post_stack_path: Path,
    output_path: Path,
    sensor: str,
    dnbr_threshold: float = 0.1,
) -> Path:
    from src.data.sensor_normalization import normalize_patch

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(pre_stack_path) as pre_src, rasterio.open(post_stack_path) as post_src:
        pre_nir = normalize_patch(pre_src.read(4), sensor)
        pre_swir = normalize_patch(pre_src.read(5), sensor)
        post_nir_raw = _align_band_to_reference(
            post_src,
            4,
            ref_transform=pre_src.transform,
            ref_crs=pre_src.crs,
            ref_height=pre_src.height,
            ref_width=pre_src.width,
        )
        post_swir_raw = _align_band_to_reference(
            post_src,
            5,
            ref_transform=pre_src.transform,
            ref_crs=pre_src.crs,
            ref_height=pre_src.height,
            ref_width=pre_src.width,
        )
        post_nir = normalize_patch(post_nir_raw, sensor)
        post_swir = normalize_patch(post_swir_raw, sensor)

        pre_denom = pre_nir + pre_swir
        post_denom = post_nir + post_swir
        valid = (pre_denom != 0.0) & (post_denom != 0.0)
        pre_nbr = np.zeros_like(pre_nir, dtype=np.float32)
        post_nbr = np.zeros_like(post_nir, dtype=np.float32)
        pre_nbr[valid] = (pre_nir[valid] - pre_swir[valid]) / (pre_denom[valid] + 1e-6)
        post_nbr[valid] = (post_nir[valid] - post_swir[valid]) / (post_denom[valid] + 1e-6)
        dnbr = pre_nbr - post_nbr
        label = np.zeros_like(dnbr, dtype=np.uint8)
        label[(dnbr >= dnbr_threshold) & valid] = 1

        profile = pre_src.profile.copy()
        profile.update({"count": 1, "dtype": "uint8", "compress": "lzw", "nodata": 0})
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(label, 1)
    return output_path


def _clip_event_assets(
    *,
    item: dict[str, Any],
    event_bbox: tuple[float, float, float, float],
    output_dir: Path,
    spec: SensorSpec,
    stage: str,
) -> tuple[Path, Path]:
    assets = item["assets"]
    stage_dir = output_dir / stage
    stage_dir.mkdir(parents=True, exist_ok=True)

    clipped_band_paths: list[Path] = []
    for canonical in spec.spectral_assets:
        asset_name = spec.spectral_alias.get(canonical, canonical)
        href = _sign_href(str(assets[asset_name]["href"]))
        band_path = stage_dir / f"{canonical}.tif"
        clipped_band_paths.append(
            clip_asset_to_bbox(
                href=href,
                bbox_wgs84=event_bbox,
                output_path=band_path,
                max_side_px=2048,
                resampling=Resampling.bilinear,
            )
        )

    stack_path = stage_dir / "image_stack.tif"
    build_multiband_stack(
        band_paths=clipped_band_paths,
        band_names=list(spec.spectral_assets),
        output_path=stack_path,
    )

    qa_href = _sign_href(str(assets[spec.qa_asset]["href"]))
    qa_path = stage_dir / "qa.tif"
    clip_asset_to_bbox(
        href=qa_href,
        bbox_wgs84=event_bbox,
        output_path=qa_path,
        max_side_px=2048,
        resampling=Resampling.nearest,
    )
    return stack_path, qa_path


def build_real_pair_record(
    *,
    event: WildfireEvent,
    dataset_name: str,
    dataset_region: str,
    dataset_split: str,
    sensor: str,
    output_root: Path,
    pre_window_days: int,
    post_window_days: int,
    max_cloud_fraction: float,
    min_overlap_fraction: float,
    dnbr_threshold: float = 0.1,
) -> dict[str, str]:
    if sensor not in SENSOR_SPECS:
        raise ValueError(f"Unsupported sensor '{sensor}'. Supported sensors: {sorted(SENSOR_SPECS)}")

    spec = SENSOR_SPECS[sensor]
    pre_item, post_item = select_pre_post_items(
        event=event,
        spec=spec,
        pre_window_days=pre_window_days,
        post_window_days=post_window_days,
        max_cloud_fraction=max_cloud_fraction,
        min_overlap_fraction=min_overlap_fraction,
    )

    pair_id = f"{event.event_id}_{sensor}"
    pair_root = output_root / dataset_name / pair_id
    pre_stack, pre_qa = _clip_event_assets(
        item=pre_item,
        event_bbox=event.bbox,
        output_dir=pair_root,
        spec=spec,
        stage="pre",
    )
    post_stack, post_qa = _clip_event_assets(
        item=post_item,
        event_bbox=event.bbox,
        output_dir=pair_root,
        spec=spec,
        stage="post",
    )

    label_path = create_dnbr_label(
        pre_stack_path=pre_stack,
        post_stack_path=post_stack,
        output_path=pair_root / "label_dnbr.tif",
        sensor=sensor,
        dnbr_threshold=dnbr_threshold,
    )

    pre_dt = _item_datetime(pre_item).date().isoformat()
    post_dt = _item_datetime(post_item).date().isoformat()
    cloud = max(_cloud_fraction(pre_item), _cloud_fraction(post_item))
    pre_item_id = str(pre_item["id"])
    post_item_id = str(post_item["id"])

    return {
        "pair_id": pair_id,
        "event_id": event.event_id,
        "region": dataset_region,
        "split": dataset_split,
        "sensor": sensor,
        "pre_image_path": str(pre_stack),
        "post_image_path": str(post_stack),
        "pre_qa_path": str(pre_qa),
        "post_qa_path": str(post_qa),
        "qa_mode": spec.qa_mode,
        "label_mask_path": str(label_path),
        "pre_date": pre_dt,
        "post_date": post_dt,
        "cloud_fraction": f"{cloud:.4f}",
        "notes": (
            f"real_stac_pair pre_item={pre_item_id} post_item={post_item_id} "
            f"label=dNBR(threshold={dnbr_threshold})"
        ),
    }

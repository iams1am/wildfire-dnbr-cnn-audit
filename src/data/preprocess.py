from __future__ import annotations

from pathlib import Path
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject

from src.data.qa_masks import apply_clear_mask


def _to_resampling(method: str) -> Resampling:
    mapping = {
        "nearest": Resampling.nearest,
        "bilinear": Resampling.bilinear,
        "cubic": Resampling.cubic,
        "average": Resampling.average,
    }
    try:
        return mapping[method]
    except KeyError as exc:
        raise ValueError(f"Unsupported resampling method: {method}") from exc


def reproject_multiband(
    input_path: str | Path,
    output_path: str | Path,
    *,
    target_crs: str,
    target_resolution_m: float,
    resampling: str,
) -> Path:
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(input_path) as src:
        dst_transform, dst_width, dst_height = calculate_default_transform(
            src.crs,
            target_crs,
            src.width,
            src.height,
            *src.bounds,
            resolution=target_resolution_m,
        )

        profile = src.profile.copy()
        profile.update(
            {
                "crs": target_crs,
                "transform": dst_transform,
                "width": dst_width,
                "height": dst_height,
                "compress": "lzw",
            }
        )

        with rasterio.open(output_path, "w", **profile) as dst:
            for band_index in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, band_index),
                    destination=rasterio.band(dst, band_index),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=dst_transform,
                    dst_crs=target_crs,
                    resampling=_to_resampling(resampling),
                )
    return output_path


def align_to_reference_grid(
    input_path: str | Path,
    reference_path: str | Path,
    output_path: str | Path,
    *,
    resampling: str,
) -> Path:
    input_path = Path(input_path)
    reference_path = Path(reference_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(reference_path) as ref, rasterio.open(input_path) as src:
        profile = src.profile.copy()
        profile.update(
            {
                "crs": ref.crs,
                "transform": ref.transform,
                "width": ref.width,
                "height": ref.height,
                "compress": "lzw",
            }
        )

        with rasterio.open(output_path, "w", **profile) as dst:
            for band_index in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, band_index),
                    destination=rasterio.band(dst, band_index),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=ref.transform,
                    dst_crs=ref.crs,
                    resampling=_to_resampling(resampling),
                )
    return output_path


def harmonize_pair(
    pre_image_path: str | Path,
    post_image_path: str | Path,
    output_dir: str | Path,
    *,
    label_mask_path: str | Path | None = None,
    pre_clear_mask_path: str | Path | None = None,
    post_clear_mask_path: str | Path | None = None,
    target_crs: str = "EPSG:6933",
    target_resolution_m: float = 30.0,
    spectral_resampling: str = "bilinear",
    mask_resampling: str = "nearest",
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pre_out = output_dir / "pre_harmonized.tif"
    post_out = output_dir / "post_harmonized.tif"

    pre_target = reproject_multiband(
        pre_image_path,
        pre_out,
        target_crs=target_crs,
        target_resolution_m=target_resolution_m,
        resampling=spectral_resampling,
    )

    post_target = reproject_multiband(
        post_image_path,
        output_dir / "post_reprojected.tif",
        target_crs=target_crs,
        target_resolution_m=target_resolution_m,
        resampling=spectral_resampling,
    )

    align_to_reference_grid(
        post_target,
        pre_target,
        post_out,
        resampling=spectral_resampling,
    )

    outputs: dict[str, Path] = {"pre": pre_target, "post": post_out}

    if pre_clear_mask_path is not None:
        pre_mask_reprojected = reproject_multiband(
            pre_clear_mask_path,
            output_dir / "pre_clear_mask_reprojected.tif",
            target_crs=target_crs,
            target_resolution_m=target_resolution_m,
            resampling=mask_resampling,
        )
        pre_mask_out = output_dir / "pre_clear_mask_harmonized.tif"
        align_to_reference_grid(
            pre_mask_reprojected,
            pre_target,
            pre_mask_out,
            resampling=mask_resampling,
        )
        pre_masked_out = output_dir / "pre_harmonized_masked.tif"
        apply_clear_mask(pre_target, pre_mask_out, pre_masked_out)
        outputs["pre"] = pre_masked_out
        outputs["pre_clear_mask"] = pre_mask_out

    if post_clear_mask_path is not None:
        post_mask_reprojected = reproject_multiband(
            post_clear_mask_path,
            output_dir / "post_clear_mask_reprojected.tif",
            target_crs=target_crs,
            target_resolution_m=target_resolution_m,
            resampling=mask_resampling,
        )
        post_mask_out = output_dir / "post_clear_mask_harmonized.tif"
        align_to_reference_grid(
            post_mask_reprojected,
            pre_target,
            post_mask_out,
            resampling=mask_resampling,
        )
        post_masked_out = output_dir / "post_harmonized_masked.tif"
        apply_clear_mask(post_out, post_mask_out, post_masked_out)
        outputs["post"] = post_masked_out
        outputs["post_clear_mask"] = post_mask_out

    if label_mask_path is not None:
        label_reprojected = reproject_multiband(
            label_mask_path,
            output_dir / "label_reprojected.tif",
            target_crs=target_crs,
            target_resolution_m=target_resolution_m,
            resampling=mask_resampling,
        )
        label_out = output_dir / "label_harmonized.tif"
        align_to_reference_grid(
            label_reprojected,
            pre_target,
            label_out,
            resampling=mask_resampling,
        )
        outputs["label"] = label_out

    return outputs

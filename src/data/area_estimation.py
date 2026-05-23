from __future__ import annotations

import numpy as np
from affine import Affine

SQ_METERS_PER_SQ_MILE = 2_589_988.110336


def estimate_area_sq_m(mask: np.ndarray, transform: Affine) -> float:
    if mask.ndim != 2:
        raise ValueError("Mask must be a 2D array.")
    pixel_area_sq_m = abs(transform.a * transform.e - transform.b * transform.d)
    burned_pixels = int(np.count_nonzero(mask > 0))
    return burned_pixels * pixel_area_sq_m


def estimate_area_sq_km(mask: np.ndarray, transform: Affine) -> float:
    return estimate_area_sq_m(mask, transform) / 1_000_000.0


def estimate_area_sq_miles(mask: np.ndarray, transform: Affine) -> float:
    return estimate_area_sq_m(mask, transform) / SQ_METERS_PER_SQ_MILE

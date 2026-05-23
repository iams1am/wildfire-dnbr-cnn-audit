from __future__ import annotations

import numpy as np


# Per-sensor reflectance scaling so that band values are roughly in [0, 1]  regardless of source.
# References:
#   Sentinel-2 L2A: surface reflectance encoded as DN, scale_factor = 1/10000.
#   Landsat Collection-2 Level-2 surface reflectance: scale = 2.75e-5, offset = -0.2.
SENSOR_SCALE: dict[str, dict[str, float]] = {
    "sentinel2": {"scale": 1.0 / 10000.0, "offset": 0.0},
    "landsat8": {"scale": 2.75e-5, "offset": -0.2},
    "landsat9": {"scale": 2.75e-5, "offset": -0.2},
}


def normalize_patch(arr: np.ndarray, sensor: str) -> np.ndarray:

    if sensor not in SENSOR_SCALE:
        return arr.astype(np.float32)
    s = SENSOR_SCALE[sensor]
    out = arr.astype(np.float32) * float(s["scale"]) + float(s["offset"])
    return np.clip(out, 0.0, 1.0).astype(np.float32)

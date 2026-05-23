import numpy as np

from src.data.sensor_normalization import normalize_patch


def test_landsat_collection2_scale_offset_and_clip():
    raw = np.array([0, 10000, 50000], dtype=np.uint16)

    out = normalize_patch(raw, "landsat8")

    np.testing.assert_allclose(out, np.array([0.0, 0.075, 1.0], dtype=np.float32))


def test_sentinel2_scale():
    raw = np.array([0, 2500, 10000], dtype=np.uint16)

    out = normalize_patch(raw, "sentinel2")

    np.testing.assert_allclose(out, np.array([0.0, 0.25, 1.0], dtype=np.float32))

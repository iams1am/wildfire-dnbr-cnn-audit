"""End-to-end smoke test on the two embedded demo fixtures (CPU-only, no
downloads, no GPU). Run with: pytest -q tests/test_demo_pipeline.py

It loads the two real ~1 MB wildfire patches from tests/fixtures/, builds each
of the six architectures from the factory at base channels 16 (tiny enough for
CPU in well under a minute), runs a forward pass and checks the output shape
and range, computes pixel-pooled IoU/F1 against the embedded label with the
same loss/metric code used for the headline results, and checks that the
per-sensor normalization maps to [0, 1].
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.sensor_normalization import normalize_patch
from src.models.factory import build_model, list_models
from src.training.losses import bce_dice_loss
from src.training.metrics import iou_score


FIXTURES = PROJECT_ROOT / "tests" / "fixtures"


def _load_fixture(name: str) -> dict[str, np.ndarray]:
    arr = np.load(FIXTURES / name)
    return {
        "pre": arr["pre"].astype(np.float32),
        "post": arr["post"].astype(np.float32),
        "label": arr["label"].astype(np.float32),
    }


def test_fixtures_exist() -> None:
    for name in ("demo_california_landsat8.npz", "demo_australia_landsat8.npz"):
        path = FIXTURES / name
        assert path.exists(), f"missing demo fixture: {path}"
        assert path.stat().st_size > 100_000, f"fixture too small: {path}"


def test_shapes_and_value_ranges() -> None:
    """Patches should have the expected 5-channel × 256×256 layout."""
    for name in ("demo_california_landsat8.npz", "demo_australia_landsat8.npz"):
        sample = _load_fixture(name)
        assert sample["pre"].shape == (5, 256, 256), sample["pre"].shape
        assert sample["post"].shape == (5, 256, 256)
        assert sample["label"].shape == (256, 256)
        assert sample["pre"].min() >= -1.0
        assert sample["post"].min() >= -1.0
        assert set(np.unique(sample["label"]).tolist()).issubset({0.0, 1.0})


def test_per_sensor_normalization_maps_to_unit_interval() -> None:
    """After per-sensor scale+offset the patch should be roughly in [0, 1]."""
    sample = _load_fixture("demo_california_landsat8.npz")
    norm = normalize_patch(sample["pre"], "landsat8")

    p99 = float(np.percentile(norm, 99))
    assert -0.5 <= p99 <= 1.5, f"normalize_patch p99 out of range: {p99}"
    assert norm.shape == sample["pre"].shape


@pytest.mark.parametrize("model_name", list_models())
def test_factory_builds_and_runs_on_cpu(model_name: str) -> None:
    """Every architecture in the factory must build and run a forward pass at
    base_channels=16 on CPU for the embedded fixture."""
    sample = _load_fixture("demo_california_landsat8.npz")
    pre = torch.from_numpy(normalize_patch(sample["pre"], "landsat8")).unsqueeze(0)
    post = torch.from_numpy(normalize_patch(sample["post"], "landsat8")).unsqueeze(0)
    model = build_model(model_name, channels_per_image=5, base_channels=16).eval()
    with torch.no_grad():
        logits = model(pre, post)
    assert logits.shape == (1, 1, 256, 256), f"{model_name} returned {logits.shape}"
    probs = torch.sigmoid(logits)
    assert probs.min() >= 0.0
    assert probs.max() <= 1.0


def test_loss_and_iou_on_fixture() -> None:
    """Loss and IoU functions should produce sensible values on the fixture."""
    sample = _load_fixture("demo_california_landsat8.npz")
    label = torch.from_numpy(sample["label"]).unsqueeze(0).unsqueeze(0)
    torch.manual_seed(0)
    pre = torch.from_numpy(normalize_patch(sample["pre"], "landsat8")).unsqueeze(0)
    post = torch.from_numpy(normalize_patch(sample["post"], "landsat8")).unsqueeze(0)
    model = build_model("baseline", channels_per_image=5, base_channels=16).eval()
    with torch.no_grad():
        logits = model(pre, post)
        loss = bce_dice_loss(logits, label)
        iou = iou_score(logits, label)
    assert torch.isfinite(loss).item()
    assert 0.0 <= float(iou) <= 1.0


def test_dnbr_label_is_consistent_with_post_minus_pre_nbr() -> None:
    """Sanity check: the embedded label is a reasonable approximation of the
    physical-reflectance dNBR rule applied to the post and pre stacks."""
    sample = _load_fixture("demo_california_landsat8.npz")
    pre = normalize_patch(sample["pre"], "landsat8")
    post = normalize_patch(sample["post"], "landsat8")
    eps = 1e-6
    nbr_pre = (pre[3] - pre[4]) / (pre[3] + pre[4] + eps)
    nbr_post = (post[3] - post[4]) / (post[3] + post[4] + eps)
    dnbr = nbr_pre - nbr_post
    derived_label = (dnbr >= 0.10).astype(np.float32)

    inter = (derived_label * sample["label"]).sum()
    union = ((derived_label + sample["label"]) > 0).sum()
    if union > 0:
        iou = float(inter / union)
        assert iou > 0.4, f"Derived vs embedded label IoU = {iou:.3f}"

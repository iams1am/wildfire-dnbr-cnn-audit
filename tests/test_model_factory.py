import torch
import pytest

from src.models.factory import build_model, list_models


def test_model_registry_contains_submission_architectures():
    assert list_models() == [
        "baseline",
        "change_transformer",
        "deeplab",
        "siamese",
        "siamese_fcn_conc",
        "siamese_fcn_diff",
    ]


def test_baseline_forward_shape_smoke():
    model = build_model("baseline", channels_per_image=5, base_channels=4).eval()
    pre = torch.zeros((1, 5, 32, 32))
    post = torch.zeros_like(pre)

    with torch.no_grad():
        logits = model(pre, post)

    assert logits.shape == (1, 1, 32, 32)


def test_unknown_model_name_is_rejected():
    with pytest.raises(ValueError):
        build_model("not_a_model", channels_per_image=5)

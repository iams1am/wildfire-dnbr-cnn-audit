from __future__ import annotations

from torch import nn

from src.models.baseline_unet import BaselineChangeUNet
from src.models.change_transformer import ChangeTransformer
from src.models.deeplab_change import DeepLabChange
from src.models.siamese_fcn import SiameseFCNConc, SiameseFCNDiff
from src.models.siamese_unet import SiameseChangeUNet


MODEL_REGISTRY: dict[str, type[nn.Module]] = {
    "baseline": BaselineChangeUNet,
    "siamese": SiameseChangeUNet,
    "siamese_fcn_diff": SiameseFCNDiff,
    "siamese_fcn_conc": SiameseFCNConc,
    "deeplab": DeepLabChange,
    "change_transformer": ChangeTransformer,
}


def build_model(model_name: str, *, channels_per_image: int, base_channels: int = 64) -> nn.Module:
    key = model_name.strip().lower()
    if key not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{model_name}'. Available: {sorted(MODEL_REGISTRY)}"
        )
    return MODEL_REGISTRY[key](channels_per_image=channels_per_image, base_channels=base_channels)


def list_models() -> list[str]:
    return sorted(MODEL_REGISTRY)

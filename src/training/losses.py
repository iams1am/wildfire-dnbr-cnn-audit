from __future__ import annotations

import torch
import torch.nn.functional as F


def batch_positive_weight(
    targets: torch.Tensor,
    *,
    min_weight: float = 1.0,
    max_weight: float = 20.0,
    eps: float = 1.0,
) -> torch.Tensor:
    """Compute a stable BCE positive-class weight from the current mask batch."""
    targets = (targets >= 0.5).float()
    positives = targets.sum()
    negatives = targets.numel() - positives
    if float(positives.item()) <= 0.0:
        return torch.as_tensor(min_weight, dtype=targets.dtype, device=targets.device)
    weight = negatives / (positives + eps)
    return weight.clamp(min=min_weight, max=max_weight)


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    probs = probs.view(probs.shape[0], -1)
    targets = targets.view(targets.shape[0], -1)

    intersection = (probs * targets).sum(dim=1)
    denominator = probs.sum(dim=1) + targets.sum(dim=1)
    dice = (2.0 * intersection + smooth) / (denominator + smooth)
    return 1.0 - dice.mean()


def bce_dice_loss(logits: torch.Tensor, targets: torch.Tensor, bce_weight: float = 0.5, pos_weight: torch.Tensor | float | None = None) -> torch.Tensor:
    if pos_weight is not None:
        if not isinstance(pos_weight, torch.Tensor):
            pos_weight = torch.tensor([pos_weight], device=logits.device, dtype=logits.dtype)
        bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)
    else:
        bce = F.binary_cross_entropy_with_logits(logits, targets)
    dice = dice_loss(logits, targets)
    return bce_weight * bce + (1.0 - bce_weight) * dice

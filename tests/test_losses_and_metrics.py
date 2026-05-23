import pytest
import torch

from src.training.losses import batch_positive_weight, bce_dice_loss
from src.training.metrics import f1_score, iou_score


def test_dynamic_positive_weight_handles_imbalance_and_empty_positive_class():
    targets = torch.zeros((1, 1, 4, 4))
    targets[0, 0, 0, 0] = 1.0

    assert batch_positive_weight(targets).item() == pytest.approx(7.5)
    assert batch_positive_weight(torch.zeros_like(targets)).item() == pytest.approx(1.0)


def test_bce_dice_loss_is_finite_with_pos_weight():
    logits = torch.zeros((2, 1, 4, 4))
    targets = torch.zeros_like(logits)
    targets[:, :, :2, :2] = 1.0

    loss = bce_dice_loss(logits, targets, pos_weight=batch_positive_weight(targets))

    assert torch.isfinite(loss)
    assert loss.item() > 0.0


def test_iou_and_f1_scores_match_manual_patch_case():
    logits = torch.tensor([[[[10.0, -10.0], [10.0, -10.0]]]])
    targets = torch.tensor([[[[1.0, 0.0], [0.0, 0.0]]]])

    assert iou_score(logits, targets) == pytest.approx(0.5, abs=1e-5)
    assert f1_score(logits, targets) == pytest.approx(2.0 / 3.0, abs=1e-5)

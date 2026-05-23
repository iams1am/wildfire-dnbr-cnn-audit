from __future__ import annotations

from typing import Callable

import numpy as np
import torch
from torch import nn


def _top_k_mask(saliency: torch.Tensor, fraction: float) -> torch.Tensor:
    batch, _, height, width = saliency.shape
    total = height * width
    k = max(1, int(round(total * fraction)))
    flat = saliency.view(batch, -1)
    _, idx = torch.topk(flat, k=k, dim=1)
    mask = torch.zeros_like(flat)
    mask.scatter_(1, idx, 1.0)
    return mask.view(batch, 1, height, width)


def _mean_baseline(x: torch.Tensor) -> torch.Tensor:
    return x.mean(dim=(2, 3), keepdim=True).expand_as(x)


def insertion_deletion_scores(
    *,
    model: nn.Module,
    pre: torch.Tensor,
    post: torch.Tensor,
    saliency: torch.Tensor,
    fractions: tuple[float, ...] = (0.05, 0.10, 0.20, 0.30, 0.50),
    target_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> dict[str, float]:
    """Insertion-deletion faithfulness for segmentation saliency.
    """
    model.eval()
    with torch.no_grad():
        base_logits = model(pre, post)
        base = base_logits.mean() if target_fn is None else target_fn(base_logits)
        baseline_pre = _mean_baseline(pre)
        baseline_post = _mean_baseline(post)
        baseline_logits = model(baseline_pre, baseline_post)
        baseline = baseline_logits.mean() if target_fn is None else target_fn(baseline_logits)

        deletion_scores: list[float] = []
        insertion_scores: list[float] = []
        for frac in fractions:
            mask = _top_k_mask(saliency, frac)
            keep = 1.0 - mask
            del_pre = (pre * keep) + (baseline_pre * mask)
            del_post = (post * keep) + (baseline_post * mask)
            del_logits = model(del_pre, del_post)
            del_target = del_logits.mean() if target_fn is None else target_fn(del_logits)

            ins_pre = (baseline_pre * keep) + (pre * mask)
            ins_post = (baseline_post * keep) + (post * mask)
            ins_logits = model(ins_pre, ins_post)
            ins_target = ins_logits.mean() if target_fn is None else target_fn(ins_logits)

            deletion_scores.append(float(del_target.item()))
            insertion_scores.append(float(ins_target.item()))

    fractions_np = np.array(fractions, dtype=np.float64)
    width = fractions_np[-1] - fractions_np[0]
    trapezoid = getattr(np, "trapezoid", getattr(np, "trapz", None))
    if trapezoid is None:
        raise RuntimeError("Neither numpy.trapezoid nor numpy.trapz is available.")
    deletion_auc = float(trapezoid(deletion_scores, fractions_np) / max(width, 1e-8))
    insertion_auc = float(trapezoid(insertion_scores, fractions_np) / max(width, 1e-8))

    return {
        "base_target": float(base.item()),
        "baseline_target": float(baseline.item()),
        "deletion_auc": deletion_auc,
        "insertion_auc": insertion_auc,
        "faithfulness_gap": insertion_auc - deletion_auc,
        "fractions": list(fractions),
        "deletion_curve": deletion_scores,
        "insertion_curve": insertion_scores,
    }

from __future__ import annotations

from typing import Callable

import torch
from torch import nn


class SegmentationOcclusion:
    """Occlusion sensitivity for change-detection models (Zeiler & Fergus, 2014).
    """

    def __init__(self, model: nn.Module, window: int = 32, stride: int = 16) -> None:
        if window <= 0 or stride <= 0:
            raise ValueError("window and stride must be positive.")
        self.model = model
        self.window = window
        self.stride = stride

    @torch.no_grad()
    def generate(
        self,
        pre: torch.Tensor,
        post: torch.Tensor,
        target_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> torch.Tensor:
        self.model.eval()
        batch, _, height, width = pre.shape
        scores = torch.zeros((batch, 1, height, width), device=pre.device)
        counts = torch.zeros((batch, 1, height, width), device=pre.device)
        pre_fill = pre.mean(dim=(2, 3), keepdim=True)
        post_fill = post.mean(dim=(2, 3), keepdim=True)

        base_logits = self.model(pre, post)
        base_target = base_logits.mean(dim=(1, 2, 3)) if target_fn is None else target_fn(base_logits).view(batch)

        for row in range(0, max(1, height - self.window + 1), self.stride):
            for col in range(0, max(1, width - self.window + 1), self.stride):
                occ_pre = pre.clone()
                occ_post = post.clone()
                occ_pre[:, :, row : row + self.window, col : col + self.window] = pre_fill
                occ_post[:, :, row : row + self.window, col : col + self.window] = post_fill
                occ_logits = self.model(occ_pre, occ_post)
                occ_target = (
                    occ_logits.mean(dim=(1, 2, 3)) if target_fn is None else target_fn(occ_logits).view(batch)
                )
                drop = (base_target - occ_target).view(batch, 1, 1, 1)
                scores[:, :, row : row + self.window, col : col + self.window] += drop
                counts[:, :, row : row + self.window, col : col + self.window] += 1.0

        saliency = scores / counts.clamp_min(1.0)
        smin = saliency.amin(dim=(2, 3), keepdim=True)
        smax = saliency.amax(dim=(2, 3), keepdim=True)
        return ((saliency - smin) / (smax - smin + 1e-8)).detach()

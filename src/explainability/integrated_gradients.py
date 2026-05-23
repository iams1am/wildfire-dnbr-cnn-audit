from __future__ import annotations

from typing import Callable

import torch
from torch import nn


class SegmentationIntegratedGradients:
    """Integrated Gradients for change-detection models (Sundararajan et al., 2017).

    Attributes importance to each input pixel by integrating the gradient of a
    scalar target function along a straight-line path from a per-channel spatial
    mean baseline to the observed pair. Returned saliency is the absolute sum
    over channels of pre and post contributions, normalised to [0, 1].
    """

    def __init__(self, model: nn.Module, steps: int = 32) -> None:
        if steps < 2:
            raise ValueError("Integrated Gradients needs at least 2 integration steps.")
        self.model = model
        self.steps = steps

    @torch.no_grad()
    def _baseline_like(self, x: torch.Tensor, use_mean: bool = True) -> torch.Tensor:
        if use_mean:
            mean_val = x.mean(dim=(2, 3), keepdim=True)
            return mean_val.expand_as(x)
        return torch.zeros_like(x)

    def generate(
        self,
        pre: torch.Tensor,
        post: torch.Tensor,
        target_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> torch.Tensor:
        self.model.eval()
        pre_base = self._baseline_like(pre, use_mean=True)
        post_base = self._baseline_like(post, use_mean=True)

        integrated_pre = torch.zeros_like(pre)
        integrated_post = torch.zeros_like(post)

        alphas = torch.linspace(0.0, 1.0, steps=self.steps, device=pre.device)
        for alpha in alphas:
            interp_pre = (pre_base + alpha * (pre - pre_base)).detach().requires_grad_(True)
            interp_post = (post_base + alpha * (post - post_base)).detach().requires_grad_(True)
            logits = self.model(interp_pre, interp_post)
            target = logits.mean() if target_fn is None else target_fn(logits)
            self.model.zero_grad(set_to_none=True)
            grads = torch.autograd.grad(target, (interp_pre, interp_post), retain_graph=False)
            integrated_pre = integrated_pre + grads[0].detach()
            integrated_post = integrated_post + grads[1].detach()

        integrated_pre = (pre - pre_base) * integrated_pre / self.steps
        integrated_post = (post - post_base) * integrated_post / self.steps

        saliency = integrated_pre.abs().sum(dim=1, keepdim=True) + integrated_post.abs().sum(dim=1, keepdim=True)
        smin = saliency.amin(dim=(2, 3), keepdim=True)
        smax = saliency.amax(dim=(2, 3), keepdim=True)
        return ((saliency - smin) / (smax - smin + 1e-8)).detach()

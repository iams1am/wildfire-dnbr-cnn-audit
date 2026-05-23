from __future__ import annotations

from typing import Callable

import torch
import torch.nn.functional as F
from torch import nn


class SegmentationGradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self._hooks: list[torch.utils.hooks.RemovableHandle] = []
        self._register_hooks()

    def _register_hooks(self) -> None:
        def forward_hook(_: nn.Module, __: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            self.activations = output

        def backward_hook(_: nn.Module, grad_input: tuple[torch.Tensor, ...], grad_output: tuple[torch.Tensor, ...]) -> None:
            _ = grad_input
            self.gradients = grad_output[0]

        self._hooks.append(self.target_layer.register_forward_hook(forward_hook))
        self._hooks.append(self.target_layer.register_full_backward_hook(backward_hook))

    def remove_hooks(self) -> None:
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def generate(
        self,
        pre: torch.Tensor,
        post: torch.Tensor,
        target_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> torch.Tensor:
        self.model.eval()
        logits = self.model(pre, post)
        if target_fn is None:
            target = logits.mean()
        else:
            target = target_fn(logits)

        self.model.zero_grad(set_to_none=True)
        target.backward(retain_graph=True)

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations/gradients.")

        gradients = self.gradients
        activations = self.activations

        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=pre.shape[-2:], mode="bilinear", align_corners=False)

        cam_min = cam.amin(dim=(2, 3), keepdim=True)
        cam_max = cam.amax(dim=(2, 3), keepdim=True)
        normalized_cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        return normalized_cam.detach()

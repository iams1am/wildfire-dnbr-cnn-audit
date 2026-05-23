from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from src.models.blocks import ConvBlock


class _ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling, reduced-size variant suitable for CPU budgets."""

    def __init__(self, in_channels: int, out_channels: int, rates: tuple[int, ...] = (1, 6, 12, 18)) -> None:
        super().__init__()
        branches = []
        for rate in rates:
            padding = rate if rate > 1 else 0
            kernel = 3 if rate > 1 else 1
            branches.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=kernel, padding=padding, dilation=rate, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                )
            )
        self.branches = nn.ModuleList(branches)
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * (len(rates) + 1), out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        features = [branch(x) for branch in self.branches]
        pooled = F.interpolate(self.global_pool(x), size=(h, w), mode="bilinear", align_corners=False)
        features.append(pooled)
        return self.project(torch.cat(features, dim=1))


class DeepLabChange(nn.Module):
    """Lightweight DeepLabv3+-style change-detection model on concatenated [pre; post]."""

    def __init__(self, channels_per_image: int, out_channels: int = 1, base_channels: int = 64) -> None:
        super().__init__()
        in_channels = channels_per_image * 2
        c1, c2, c3 = base_channels, base_channels * 2, base_channels * 4
        aspp_channels = base_channels * 8

        self.pool = nn.MaxPool2d(2)
        self.enc1 = ConvBlock(in_channels, c1)
        self.enc2 = ConvBlock(c1, c2)
        self.enc3 = ConvBlock(c2, c3)
        self.aspp = _ASPP(c3, aspp_channels)

        self.low_level_project = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(aspp_channels + c1, aspp_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(aspp_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(aspp_channels, aspp_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(aspp_channels),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Conv2d(aspp_channels, out_channels, kernel_size=1)

    def forward(self, pre: torch.Tensor, post: torch.Tensor) -> torch.Tensor:
        x = torch.cat([pre, post], dim=1)
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        aspp_out = self.aspp(e3)

        low = self.low_level_project(e1)
        upsampled = F.interpolate(aspp_out, size=low.shape[-2:], mode="bilinear", align_corners=False)
        fused = torch.cat([upsampled, low], dim=1)
        decoded = self.decoder(fused)
        logits = self.head(decoded)
        return F.interpolate(logits, size=pre.shape[-2:], mode="bilinear", align_corners=False)

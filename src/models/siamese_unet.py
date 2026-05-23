from __future__ import annotations

import torch
from torch import nn

from src.models.blocks import ConvBlock, UpBlock


class SiameseChangeUNet(nn.Module):
    def __init__(self, channels_per_image: int, out_channels: int = 1, base_channels: int = 64) -> None:
        super().__init__()
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8
        bottleneck_channels = base_channels * 16

        self.pool = nn.MaxPool2d(2)

        # Shared encoder.
        self.enc1 = ConvBlock(channels_per_image, c1)
        self.enc2 = ConvBlock(c1, c2)
        self.enc3 = ConvBlock(c2, c3)
        self.enc4 = ConvBlock(c3, c4)
        self.bottleneck = ConvBlock(c4, bottleneck_channels)

        fused_bottleneck_channels = bottleneck_channels * 3
        self.up4 = UpBlock(fused_bottleneck_channels, c4 * 3, c4)
        self.up3 = UpBlock(c4, c3 * 3, c3)
        self.up2 = UpBlock(c3, c2 * 3, c2)
        self.up1 = UpBlock(c2, c1 * 3, c1)
        self.head = nn.Conv2d(c1, out_channels, kernel_size=1)

    def _encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        return e1, e2, e3, e4, b

    @staticmethod
    def _fuse(pre: torch.Tensor, post: torch.Tensor) -> torch.Tensor:
        return torch.cat([pre, post, torch.abs(post - pre)], dim=1)

    def forward(self, pre: torch.Tensor, post: torch.Tensor) -> torch.Tensor:
        pre_e1, pre_e2, pre_e3, pre_e4, pre_b = self._encode(pre)
        post_e1, post_e2, post_e3, post_e4, post_b = self._encode(post)

        fused_b = self._fuse(pre_b, post_b)
        fused_e4 = self._fuse(pre_e4, post_e4)
        fused_e3 = self._fuse(pre_e3, post_e3)
        fused_e2 = self._fuse(pre_e2, post_e2)
        fused_e1 = self._fuse(pre_e1, post_e1)

        d4 = self.up4(fused_b, fused_e4)
        d3 = self.up3(d4, fused_e3)
        d2 = self.up2(d3, fused_e2)
        d1 = self.up1(d2, fused_e1)
        return self.head(d1)

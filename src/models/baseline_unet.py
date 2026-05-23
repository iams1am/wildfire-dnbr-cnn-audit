from __future__ import annotations

import torch
from torch import nn

from src.models.blocks import ConvBlock, UpBlock


class BaselineChangeUNet(nn.Module):
    def __init__(self, channels_per_image: int, out_channels: int = 1, base_channels: int = 64) -> None:
        super().__init__()
        in_channels = channels_per_image * 2

        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8
        bottleneck_channels = base_channels * 16

        self.enc1 = ConvBlock(in_channels, c1)
        self.enc2 = ConvBlock(c1, c2)
        self.enc3 = ConvBlock(c2, c3)
        self.enc4 = ConvBlock(c3, c4)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(c4, bottleneck_channels)

        self.up4 = UpBlock(bottleneck_channels, c4, c4)
        self.up3 = UpBlock(c4, c3, c3)
        self.up2 = UpBlock(c3, c2, c2)
        self.up1 = UpBlock(c2, c1, c1)
        self.head = nn.Conv2d(c1, out_channels, kernel_size=1)

    def forward(self, pre: torch.Tensor, post: torch.Tensor) -> torch.Tensor:
        x = torch.cat([pre, post], dim=1)
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        d4 = self.up4(b, e4)
        d3 = self.up3(d4, e3)
        d2 = self.up2(d3, e2)
        d1 = self.up1(d2, e1)
        return self.head(d1)

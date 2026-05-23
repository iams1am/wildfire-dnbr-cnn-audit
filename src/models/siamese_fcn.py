from __future__ import annotations

import torch
from torch import nn

from src.models.blocks import ConvBlock, UpBlock


class SiameseFCNDiff(nn.Module):
    """Daudt et al. 2018 Fully Convolutional Siamese-Diff variant for change detection.
    Reference: Daudt, R. C., Le Saux, B., Boulch, A. (2018). Fully Convolutional
    Siamese Networks for Change Detection. ICIP.
    """

    def __init__(self, channels_per_image: int, out_channels: int = 1, base_channels: int = 64) -> None:
        super().__init__()
        c1, c2, c3, c4 = base_channels, base_channels * 2, base_channels * 4, base_channels * 8
        bottleneck = base_channels * 16

        self.pool = nn.MaxPool2d(2)
        self.enc1 = ConvBlock(channels_per_image, c1)
        self.enc2 = ConvBlock(c1, c2)
        self.enc3 = ConvBlock(c2, c3)
        self.enc4 = ConvBlock(c3, c4)
        self.bottleneck = ConvBlock(c4, bottleneck)

        self.up4 = UpBlock(bottleneck, c4, c4)
        self.up3 = UpBlock(c4, c3, c3)
        self.up2 = UpBlock(c3, c2, c2)
        self.up1 = UpBlock(c2, c1, c1)
        self.head = nn.Conv2d(c1, out_channels, kernel_size=1)

    def _encode(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        return e1, e2, e3, e4, b

    def forward(self, pre: torch.Tensor, post: torch.Tensor) -> torch.Tensor:
        pre_e1, pre_e2, pre_e3, pre_e4, pre_b = self._encode(pre)
        post_e1, post_e2, post_e3, post_e4, post_b = self._encode(post)

        diff_b = torch.abs(post_b - pre_b)
        diff_e4 = torch.abs(post_e4 - pre_e4)
        diff_e3 = torch.abs(post_e3 - pre_e3)
        diff_e2 = torch.abs(post_e2 - pre_e2)
        diff_e1 = torch.abs(post_e1 - pre_e1)

        d4 = self.up4(diff_b, diff_e4)
        d3 = self.up3(d4, diff_e3)
        d2 = self.up2(d3, diff_e2)
        d1 = self.up1(d2, diff_e1)
        return self.head(d1)


class SiameseFCNConc(nn.Module):
    """Daudt et al. 2018 Fully Convolutional Siamese-Conc variant.

    Shared encoder; decoder consumes concatenated [pre; post] skips.
    """

    def __init__(self, channels_per_image: int, out_channels: int = 1, base_channels: int = 64) -> None:
        super().__init__()
        c1, c2, c3, c4 = base_channels, base_channels * 2, base_channels * 4, base_channels * 8
        bottleneck = base_channels * 16

        self.pool = nn.MaxPool2d(2)
        self.enc1 = ConvBlock(channels_per_image, c1)
        self.enc2 = ConvBlock(c1, c2)
        self.enc3 = ConvBlock(c2, c3)
        self.enc4 = ConvBlock(c3, c4)
        self.bottleneck = ConvBlock(c4, bottleneck)

        self.up4 = UpBlock(bottleneck * 2, c4 * 2, c4)
        self.up3 = UpBlock(c4, c3 * 2, c3)
        self.up2 = UpBlock(c3, c2 * 2, c2)
        self.up1 = UpBlock(c2, c1 * 2, c1)
        self.head = nn.Conv2d(c1, out_channels, kernel_size=1)

    def _encode(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        return e1, e2, e3, e4, b

    def forward(self, pre: torch.Tensor, post: torch.Tensor) -> torch.Tensor:
        pre_e1, pre_e2, pre_e3, pre_e4, pre_b = self._encode(pre)
        post_e1, post_e2, post_e3, post_e4, post_b = self._encode(post)

        d4 = self.up4(torch.cat([pre_b, post_b], dim=1), torch.cat([pre_e4, post_e4], dim=1))
        d3 = self.up3(d4, torch.cat([pre_e3, post_e3], dim=1))
        d2 = self.up2(d3, torch.cat([pre_e2, post_e2], dim=1))
        d1 = self.up1(d2, torch.cat([pre_e1, post_e1], dim=1))
        return self.head(d1)

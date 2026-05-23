from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from src.models.blocks import ConvBlock, UpBlock


class _LightTransformerBlock(nn.Module):
    """A small Pre-LN Transformer encoder block: MultiheadAttn + MLP."""

    def __init__(self, dim: int, num_heads: int = 4, mlp_ratio: float = 2.0, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = self.norm1(x)
        attn_out, _ = self.attn(n, n, n, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class ChangeTransformer(nn.Module):
    """A bitemporal-image-transformer-style change-detection model.

    Loosely follows BIT (Chen et al. 2022) but keeps the same configurable
    width as the CNN baselines. A shared CNN encoder produces bitemporal tokens
    at the bottleneck resolution, two transformer encoder blocks apply
    self-attention over the (pre, post) tokens, and a CNN decoder maps back to
    1-channel logits.
    """

    def __init__(
        self,
        channels_per_image: int,
        out_channels: int = 1,
        base_channels: int = 64,
        num_transformer_blocks: int = 2,
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8
        bottleneck = base_channels * 16

        self.pool = nn.MaxPool2d(2)
        self.enc1 = ConvBlock(channels_per_image, c1)
        self.enc2 = ConvBlock(c1, c2)
        self.enc3 = ConvBlock(c2, c3)
        self.enc4 = ConvBlock(c3, c4)
        self.bottleneck = ConvBlock(c4, bottleneck)

        # Each token = bitemporal vector at spatial position. We project
        # concat([pre, post]) of the bottleneck features into a token of width `bottleneck`.
        self.token_proj = nn.Linear(bottleneck * 2, bottleneck)
        self.transformer = nn.ModuleList(
            [_LightTransformerBlock(bottleneck, num_heads=num_heads) for _ in range(num_transformer_blocks)]
        )

        # Decoder consumes (concat([pre, post]) at each level; bottleneck input is the transformer output.
        self.up4 = UpBlock(bottleneck, c4 * 2, c4)
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

        b, c, h, w = pre_b.shape
        # Tokens at the bottleneck: stack pre and post along the channel dim per spatial position.
        tokens = torch.cat([pre_b, post_b], dim=1).flatten(2).transpose(1, 2)  # (B, H*W, 2C)
        tokens = self.token_proj(tokens)  # (B, H*W, C)
        for blk in self.transformer:
            tokens = blk(tokens)
        bottleneck = tokens.transpose(1, 2).reshape(b, c, h, w)

        d4 = self.up4(bottleneck, torch.cat([pre_e4, post_e4], dim=1))
        d3 = self.up3(d4, torch.cat([pre_e3, post_e3], dim=1))
        d2 = self.up2(d3, torch.cat([pre_e2, post_e2], dim=1))
        d1 = self.up1(d2, torch.cat([pre_e1, post_e1], dim=1))
        return self.head(d1)

"""Render the Siamese U-Net architecture diagram.

Layout (left to right):
  Pre encoder column   ->   Per-level fusion column   ->   Decoder column   ->   Output
  Post encoder column  ->   (shared with above)

The shared-weight encoder is shown by a gray bracket spanning both encoder columns.
Fusion -> decoder arrows go to the matching level (no crossing).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def _box(ax, x, y, w, h, text, color):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.06",
        linewidth=1.0,
        edgecolor="#1f2937",
        facecolor=color,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8.5)


def _arrow(ax, x0, y0, x1, y1, color="#374151", style="-|>", lw=1.0):
    ax.add_patch(
        FancyArrowPatch(
            (x0, y0), (x1, y1),
            arrowstyle=style,
            mutation_scale=10,
            linewidth=lw,
            color=color,
            shrinkA=2, shrinkB=2,
        )
    )


def draw_architecture(output_png: Path, output_svg: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 7.0))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 7.2)
    ax.axis("off")

    # Colors
    enc_color = "#C6DBEF"
    fuse_color = "#FCBBA1"
    dec_color = "#C7E9C0"
    out_color = "#FCA3B7"

    # Column x positions
    x_pre   = 0.6
    x_post  = 3.4
    x_fuse  = 6.4
    x_dec   = 10.2
    x_out   = 12.4

    # Box dimensions
    enc_w, enc_h = 1.8, 0.55
    fuse_w, fuse_h = 2.6, 0.55
    dec_w, dec_h = 1.4, 0.55

    # Vertical positions for the 5 levels (top -> bottom = level 1 -> level 5/bottleneck)
    levels = [
        ("e1 (C)",   5.4),
        ("e2 (2C)",  4.5),
        ("e3 (4C)",  3.6),
        ("e4 (8C)",  2.7),
        ("b (16C)",  1.8),
    ]

    # Pre encoder column
    for label, y in levels:
        _box(ax, x_pre, y, enc_w, enc_h, f"Pre {label}", enc_color)

    # Post encoder column
    for label, y in levels:
        _box(ax, x_post, y, enc_w, enc_h, f"Post {label}", enc_color)

    # Shared-weight bracket (above both encoder columns)
    bracket_y = 6.4
    ax.plot([x_pre, x_post + enc_w], [bracket_y, bracket_y], color="#9ca3af", lw=1.5)
    ax.plot([x_pre, x_pre], [bracket_y - 0.12, bracket_y], color="#9ca3af", lw=1.5)
    ax.plot([x_post + enc_w, x_post + enc_w], [bracket_y - 0.12, bracket_y], color="#9ca3af", lw=1.5)
    ax.text((x_pre + x_post + enc_w) / 2, bracket_y + 0.22,
            "Shared-weight encoder",
            ha="center", va="center", fontsize=10.5, fontweight="bold", color="#374151")

    # Fusion column (per-level [pre; post; |Δ|])
    for label, y in levels:
        _box(ax, x_fuse, y, fuse_w, fuse_h,
             f"[pre; post; |Δ|]  {label}", fuse_color)

    # Encoder -> Fusion arrows. Pre arrow enters fusion box top-third, post arrow enters bottom-third.
    for _, y in levels:
        y_mid = y + enc_h / 2
        # Pre encoder -> fusion (slight downward angle to enter at top of fusion box)
        _arrow(ax, x_pre + enc_w, y_mid, x_fuse, y + enc_h * 0.7)
        # Post encoder -> fusion (slight upward angle to enter at bottom of fusion box)
        _arrow(ax, x_post + enc_w, y_mid, x_fuse, y + enc_h * 0.3)

    # Decoder column, labeled top-down so up1 is at the top decoder feature map
    decoder_labels = [
        ("up1 (C)",   5.4),
        ("up2 (2C)",  4.5),
        ("up3 (4C)",  3.6),
        ("up4 (8C)",  2.7),
        ("bottleneck",1.8),
    ]
    for label, y in decoder_labels:
        _box(ax, x_dec, y, dec_w, dec_h, label, dec_color)

    # Fusion -> Decoder arrows (level-i goes to level-i; parallel, no crossing)
    for _, y in levels:
        y_mid = y + fuse_h / 2
        _arrow(ax, x_fuse + fuse_w, y_mid, x_dec, y_mid)

    # Decoder bottom-up flow (information flows from bottleneck up)
    for i in range(len(decoder_labels) - 1):
        y_lower = decoder_labels[i + 1][1] + dec_h
        y_upper = decoder_labels[i][1]
        x_mid = x_dec + dec_w / 2
        _arrow(ax, x_mid, y_lower, x_mid, y_upper, color="#6b7280", style="->", lw=0.9)

    # Head -> Output (1×1 conv to logits)
    head_y = decoder_labels[0][1] + dec_h / 2
    _arrow(ax, x_dec + dec_w, head_y, x_out, head_y, color="#7f1d1d", lw=1.4)
    _box(ax, x_out, decoder_labels[0][1], 0.85, dec_h, "logits\n1ch", out_color)

    # Input/output labels - placed below their columns to avoid overlap with the shared-weight bracket
    ax.text(x_pre + enc_w / 2, 1.45, "Pre image input",
            ha="center", fontsize=9.5, fontweight="bold", color="#1f2937")
    ax.text(x_post + enc_w / 2, 1.45, "Post image input",
            ha="center", fontsize=9.5, fontweight="bold", color="#1f2937")
    ax.text(x_out + 0.45, decoder_labels[0][1] + dec_h + 0.3,
            "Burned mask\n(output)", ha="center", fontsize=8.5, fontweight="bold", color="#7f1d1d")

    # Equation reference at the bottom
    ax.text(6.5, 0.6,
            r"Per-level fusion:   $z_i = [\,e_i^{\mathrm{pre}};\ e_i^{\mathrm{post}};\ |e_i^{\mathrm{post}} - e_i^{\mathrm{pre}}|\,]$",
            ha="center", fontsize=10.5, color="#1f2937")

    # Legend
    legend_entries = [
        mpatches.Patch(facecolor=enc_color, edgecolor="#1f2937", label="Shared encoder levels"),
        mpatches.Patch(facecolor=fuse_color, edgecolor="#1f2937", label="Per-level [pre; post; |Δ|] fusion"),
        mpatches.Patch(facecolor=dec_color, edgecolor="#1f2937", label="U-Net decoder"),
        mpatches.Patch(facecolor=out_color, edgecolor="#1f2937", label="1x1 conv head"),
    ]
    ax.legend(handles=legend_entries, loc="lower left", fontsize=8.5, frameon=False, bbox_to_anchor=(0.0, 0.02))

    output_png.parent.mkdir(parents=True, exist_ok=True)
    output_svg.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_png, bbox_inches="tight", dpi=220)
    plt.savefig(output_svg, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render Siamese architecture figure (clean version).")
    parser.add_argument("--output-png", type=Path, default=Path("data/paper_assets/figures/siamese_architecture.png"))
    parser.add_argument("--output-svg", type=Path, default=Path("data/paper_assets/figures/siamese_architecture.svg"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    draw_architecture(args.output_png, args.output_svg)
    print(f"Wrote {args.output_png}")
    print(f"Wrote {args.output_svg}")


if __name__ == "__main__":
    main()

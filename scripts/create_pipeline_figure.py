from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


def _node(ax: plt.Axes, xy: tuple[float, float], text: str, color: str) -> None:
    width, height = 0.17, 0.12
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.02",
        linewidth=1.5,
        edgecolor="#1f2937",
        facecolor=color,
    )
    ax.add_patch(patch)
    ax.text(x + width / 2.0, y + height / 2.0, text, ha="center", va="center", fontsize=10)


def _arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float]) -> None:
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops=dict(arrowstyle="->", linewidth=1.8, color="#111827"),
    )


def main() -> None:
    output_dir = Path("data/paper_assets/figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 4.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    nodes = [
        ((0.03, 0.42), "STAC Fetch\n(S2/L8 Real Scenes)", "#dbeafe"),
        ((0.22, 0.42), "Pair Selection\n(Pre/Post by Event)", "#e0e7ff"),
        ((0.41, 0.42), "Harmonize + QA\n(CRS, Res, Masks)", "#dcfce7"),
        ((0.60, 0.42), "Patch Extraction\n(Train/Test Tiles)", "#fef3c7"),
        ((0.79, 0.42), "Train + External Test\n(CA→AU)", "#fee2e2"),
    ]
    for xy, text, color in nodes:
        _node(ax, xy, text, color)

    _arrow(ax, (0.20, 0.48), (0.22, 0.48))
    _arrow(ax, (0.39, 0.48), (0.41, 0.48))
    _arrow(ax, (0.58, 0.48), (0.60, 0.48))
    _arrow(ax, (0.77, 0.48), (0.79, 0.48))

    ax.text(0.5, 0.88, "Wildfire Change-Detection Method Pipeline", ha="center", va="center", fontsize=14, fontweight="bold")
    ax.text(
        0.5,
        0.17,
        "Training domain: California | External generalization domain: Australia | Outputs: segmentation + area metrics + XAI",
        ha="center",
        va="center",
        fontsize=10,
    )

    png_path = output_dir / "method_pipeline.png"
    svg_path = output_dir / "method_pipeline.svg"
    fig.tight_layout()
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote {png_path}")
    print(f"Wrote {svg_path}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.explainability.gradcam import SegmentationGradCAM
from src.data.sensor_normalization import normalize_patch
from src.models.siamese_unet import SiameseChangeUNet


def _normalize_rgb(rgb: np.ndarray) -> np.ndarray:
    flat = rgb.reshape(-1, 3)
    p2 = np.percentile(flat, 2, axis=0)
    p98 = np.percentile(flat, 98, axis=0)
    denom = np.maximum(p98 - p2, 1e-6)
    norm = (rgb - p2) / denom
    return np.clip(norm, 0.0, 1.0)


def _to_rgb(post_patch: np.ndarray) -> np.ndarray:
    # Band order in stack: [blue, green, red, nir, swir22].
    rgb = np.stack([post_patch[2], post_patch[1], post_patch[0]], axis=-1)
    return _normalize_rgb(rgb)


def _load_model(checkpoint: Path, channels_per_image: int, base_channels: int, device: torch.device) -> SiameseChangeUNet:
    model = SiameseChangeUNet(channels_per_image=channels_per_image, base_channels=base_channels)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Grad-CAM overlays on representative Australia patches.")
    parser.add_argument("--checkpoint", type=Path, default=Path("data/runs/siamese_real/best_model.pt"))
    parser.add_argument("--patch-index", type=Path, default=Path("data/patches/australia/patch_index.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/paper_assets/xai"))
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    patch_df = pd.read_csv(args.patch_index)
    if patch_df.empty:
        raise ValueError(f"Patch index is empty: {args.patch_index}")

    ranked = patch_df.sort_values(by="burned_fraction", ascending=False).head(args.num_samples)
    if ranked.empty:
        raise ValueError("No ranked patches available for XAI export.")

    first_patch = np.load(Path(str(ranked.iloc[0]["patch_path"])))
    channels = int(first_patch["pre"].shape[0])
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model = _load_model(args.checkpoint, channels, args.base_channels, device)
    target_layer = model.bottleneck.block[3]
    gradcam = SegmentationGradCAM(model=model, target_layer=target_layer)

    for idx, row in ranked.reset_index(drop=True).iterrows():
        patch_path = Path(str(row["patch_path"]))
        sample = np.load(patch_path)
        pre = sample["pre"].astype(np.float32)
        post = sample["post"].astype(np.float32)
        label = sample["label"].astype(np.float32)
        sensor = str(row.get("sensor", "")).strip().lower()
        if args.normalize:
            pre = normalize_patch(pre, sensor)
            post = normalize_patch(post, sensor)

        pre_t = torch.from_numpy(pre).unsqueeze(0).to(device)
        post_t = torch.from_numpy(post).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(pre_t, post_t)
            pred = (torch.sigmoid(logits) >= 0.5).float().cpu().numpy()[0, 0]

        cam = gradcam.generate(pre_t, post_t)[0, 0].cpu().numpy()
        rgb = _to_rgb(post)

        fig, axes = plt.subplots(1, 4, figsize=(14, 3.8))
        axes[0].imshow(rgb)
        axes[0].set_title("Post RGB")
        axes[1].imshow(label, cmap="gray")
        axes[1].set_title("Label")
        axes[2].imshow(pred, cmap="gray")
        axes[2].set_title("Prediction")
        axes[3].imshow(rgb)
        axes[3].imshow(cam, cmap="jet", alpha=0.45)
        axes[3].set_title("Grad-CAM Overlay")
        for ax in axes:
            ax.axis("off")
        fig.suptitle(f"XAI Sample {idx+1}: {patch_path.name}", fontsize=10)
        fig.tight_layout()

        out_path = args.output_dir / f"xai_sample_{idx+1:02d}.png"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote {out_path}")

    gradcam.remove_hooks()


if __name__ == "__main__":
    main()

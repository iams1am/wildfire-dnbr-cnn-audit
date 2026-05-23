"""Compare predictions on a Pilbara S2 patch: pre-normalization model vs augnorm model.

Picks the patch with the largest IoU improvement after normalization.
"""
from __future__ import annotations

import sys
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.factory import build_model
from src.training.datasets import PatchPairDataset


def load_model(ckpt: Path, channels: int, base_channels: int) -> torch.nn.Module:
    m = build_model("siamese", channels_per_image=channels, base_channels=base_channels)
    m.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
    return m.cuda().eval()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create before/after reflectance-normalization comparison figure.")
    parser.add_argument("--raw-checkpoint", type=Path, default=ROOT / "data" / "runs" / "siamese_qaoff_seed42" / "best_model.pt")
    parser.add_argument("--normalized-checkpoint", type=Path, default=ROOT / "data" / "runs" / "siamese_qaoff_seed42_reflectance64" / "best_model.pt")
    parser.add_argument("--raw-base-channels", type=int, default=16)
    parser.add_argument("--normalized-base-channels", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    patch_index = ROOT / "data" / "patches_noqa" / "australia_full" / "patch_index.csv"
    df = pd.read_csv(patch_index)
    pilbara_s2 = df[(df["event_id"] == "pilbara_fires_2023") & (df["sensor"] == "sentinel2")].reset_index(drop=True)
    print(f"Pilbara S2 patches: {len(pilbara_s2)}")
    pilbara_s2 = pilbara_s2[pilbara_s2["burned_fraction"] > 0.05].reset_index(drop=True)
    print(f"  with burn > 5%: {len(pilbara_s2)}")
    if pilbara_s2.empty:
        print("No suitable patch found")
        return
    pilbara_csv = ROOT / "data" / "patches_noqa" / "australia_full" / "pilbara_s2_subset.csv"
    pilbara_s2.to_csv(pilbara_csv, index=False)

    # Two datasets: raw (no normalize) and normalized
    ds_raw = PatchPairDataset(pilbara_csv, normalize=False)
    ds_norm = PatchPairDataset(pilbara_csv, normalize=True)

    model_raw = load_model(args.raw_checkpoint, ds_raw.channels_per_image, args.raw_base_channels)
    model_norm = load_model(args.normalized_checkpoint, ds_norm.channels_per_image, args.normalized_base_channels)

    # Find the patch where augnorm improves most over raw
    best_delta, best_idx = -1.0, 0
    with torch.no_grad():
        for i in range(len(ds_raw)):
            raw_s = ds_raw[i]
            nrm_s = ds_norm[i]
            label = (raw_s["label"] >= 0.5).float().numpy()[0]
            pre_r, post_r = raw_s["pre"][None].cuda(), raw_s["post"][None].cuda()
            pre_n, post_n = nrm_s["pre"][None].cuda(), nrm_s["post"][None].cuda()
            pred_r = (torch.sigmoid(model_raw(pre_r, post_r)) >= 0.5).float().cpu().numpy()[0, 0]
            pred_n = (torch.sigmoid(model_norm(pre_n, post_n)) >= 0.5).float().cpu().numpy()[0, 0]
            iou_r = ((pred_r * label).sum() + 1.0) / ((pred_r + label - pred_r * label).sum() + 1.0)
            iou_n = ((pred_n * label).sum() + 1.0) / ((pred_n + label - pred_n * label).sum() + 1.0)
            d = iou_n - iou_r
            if d > best_delta:
                best_delta, best_idx = d, i
    print(f"Best improvement on patch {best_idx}: IoU delta = {best_delta:.3f}")

    raw_s = ds_raw[best_idx]
    nrm_s = ds_norm[best_idx]
    label = (raw_s["label"] >= 0.5).float().numpy()[0]
    with torch.no_grad():
        pred_raw = (torch.sigmoid(model_raw(raw_s["pre"][None].cuda(), raw_s["post"][None].cuda())) >= 0.5).float().cpu().numpy()[0, 0]
        pred_norm = (torch.sigmoid(model_norm(nrm_s["pre"][None].cuda(), nrm_s["post"][None].cuda())) >= 0.5).float().cpu().numpy()[0, 0]

    rgb_post = nrm_s["post"][[2, 1, 0]].numpy().transpose(1, 2, 0)
    rgb_post = np.clip(rgb_post * 4.0, 0, 1)

    iou_raw_v = ((pred_raw * label).sum() + 1.0) / ((pred_raw + label - pred_raw * label).sum() + 1.0)
    iou_norm_v = ((pred_norm * label).sum() + 1.0) / ((pred_norm + label - pred_norm * label).sum() + 1.0)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.4), dpi=150)
    axes[0].imshow(rgb_post)
    axes[0].set_title("Pilbara 2023 post-fire (Sentinel-2 RGB)", fontsize=11)
    axes[0].axis("off")
    axes[1].imshow(label, cmap="Reds", vmin=0, vmax=1)
    axes[1].set_title("Ground truth (dNBR)", fontsize=11)
    axes[1].axis("off")
    axes[2].imshow(pred_raw, cmap="Reds", vmin=0, vmax=1)
    axes[2].set_title(f"Without normalization\nIoU = {iou_raw_v:.2f}", fontsize=11)
    axes[2].axis("off")
    axes[3].imshow(pred_norm, cmap="Reds", vmin=0, vmax=1)
    axes[3].set_title(f"With per-sensor normalization\nIoU = {iou_norm_v:.2f}", fontsize=11)
    axes[3].axis("off")
    plt.suptitle("Effect of per-sensor reflectance normalization on a single Pilbara Sentinel-2 patch",
                 fontsize=12.5, fontweight="bold", y=1.02)

    out_png = ROOT / "data" / "paper_assets" / "figures" / "normalization_effect.png"
    out_svg = ROOT / "data" / "paper_assets" / "figures" / "normalization_effect.svg"
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    fig.savefig(out_svg, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_png}")
    print(f"Wrote {out_svg}")


if __name__ == "__main__":
    main()

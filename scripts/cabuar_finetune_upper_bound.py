"""CaBuAr target-domain fine-tune upper-bound experiment.

Tests whether the CaBuAr collapse is a real representation failure or just a
class-prior/optimizer-basin issue that a few epochs of target-domain
fine-tuning could undo. Per (architecture, seed) it loads the California-trained
checkpoint, fine-tunes for K epochs on a small stratified CaBuAr subset
(default 200 patches, rest held out), evaluates pixel-pooled metrics on the
held-out set, and writes one CSV row per (arch, seed, epoch).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.factory import build_model
from src.training.datasets import PatchPairDataset
from src.training.train_loop import train_segmentation_model

ARCH_NAMES = [
    "baseline",
    "deeplab",
    "siamese",
    "siamese_fcn_conc",
    "siamese_fcn_diff",
    "change_transformer",
]
SEEDS = [42, 17, 2026]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CaBuAr target-domain fine-tune upper-bound.")
    parser.add_argument(
        "--cabuar-patch-index",
        type=Path,
        default=PROJECT_ROOT / "data" / "patches_cabuar_full" / "patch_index.csv",
    )
    parser.add_argument(
        "--ckpt-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "runs",
        help="Where to find <arch>_qaon_seed<S>_reflectance64/best_model.pt",
    )
    parser.add_argument(
        "--ft-subset-size",
        type=int,
        default=200,
        help="Number of CaBuAr patches used for fine-tuning (rest is held out for eval).",
    )
    parser.add_argument("--epochs", type=int, default=3, help="Fine-tune epochs.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Fine-tune LR (smaller than cold-start).")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--archs", nargs="+", default=ARCH_NAMES)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=PROJECT_ROOT
        / "data"
        / "evaluation"
        / "cabuar_finetune_upper_bound"
        / "cabuar_finetune_results.csv",
    )
    return parser.parse_args()


def make_subset_indices(patch_index_csv: Path, subset_size: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Stratified split that keeps ~1/3 of burned-positive patches in the eval
    set so that recall/IoU are well-defined post-FT.

    CaBuAr is extremely class-imbalanced (~3% of patches have any burn), so a
    naive "half positives in FT" split exhausts the positive pool and leaves
    eval with no positives. We instead cap the number of positives put into FT
    at one-third of the available positives, then fill the rest of the FT
    subset from negatives.
    """
    df = pd.read_csv(patch_index_csv)
    rng = np.random.default_rng(seed)
    burned_mask = df["burned_fraction"].astype(float).values > 0.0
    pos_idx = np.where(burned_mask)[0]
    neg_idx = np.where(~burned_mask)[0]
    n_pos = min(len(pos_idx) // 3, subset_size // 2)
    n_pos = max(n_pos, 1) if len(pos_idx) > 0 else 0
    n_neg = max(0, subset_size - n_pos)
    n_neg = min(n_neg, len(neg_idx))
    ft_pos = rng.choice(pos_idx, size=n_pos, replace=False) if n_pos else np.array([], dtype=int)
    ft_neg = rng.choice(neg_idx, size=n_neg, replace=False) if n_neg else np.array([], dtype=int)
    ft_idx = np.sort(np.concatenate([ft_pos, ft_neg]))
    eval_idx = np.array([i for i in range(len(df)) if i not in set(ft_idx.tolist())])
    n_pos_in_eval = int(burned_mask[eval_idx].sum())
    print(f"  Split: FT n_pos={n_pos}, n_neg={n_neg}; eval n_pos={n_pos_in_eval}, n_neg={len(eval_idx)-n_pos_in_eval}")
    return ft_idx, eval_idx


def preload_patches(dataset: PatchPairDataset, indices: np.ndarray) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One-shot disk read: stack all (pre, post, label) tensors for the given indices.
    Returns three CPU tensors that can be indexed in-memory during the training/eval loops,
    avoiding the per-patch __getitem__ disk hit that dominated the previous IO-bound run."""
    pres, posts, labels = [], [], []
    for j in indices:
        s = dataset[int(j)]
        pres.append(s["pre"])
        posts.append(s["post"])
        labels.append(s["label"])
    return torch.stack(pres), torch.stack(posts), torch.stack(labels)


def evaluate_pixel_pooled_preloaded(model: torch.nn.Module, pre: torch.Tensor, post: torch.Tensor, label: torch.Tensor, device: torch.device, batch_size: int = 8) -> dict:
    model.eval()
    tp = fp = fn = 0
    with torch.no_grad():
        for i in range(0, pre.shape[0], batch_size):
            p = pre[i : i + batch_size].to(device, non_blocking=True)
            q = post[i : i + batch_size].to(device, non_blocking=True)
            y = label[i : i + batch_size].to(device, non_blocking=True)
            logits = model(p, q)
            pred = (torch.sigmoid(logits) > 0.5).float()
            tp += float(((pred == 1) & (y == 1)).sum().item())
            fp += float(((pred == 1) & (y == 0)).sum().item())
            fn += float(((pred == 0) & (y == 1)).sum().item())
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else float("nan")
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else float("nan")
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    return {"iou": iou, "f1": f1, "precision": precision, "recall": recall, "tp": tp, "fp": fp, "fn": fn}


def main() -> None:
    args = parse_args()
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    ft_idx, eval_idx = make_subset_indices(args.cabuar_patch_index, args.ft_subset_size, seed=42)
    print(
        f"CaBuAr split: {len(ft_idx)} FT patches, {len(eval_idx)} held-out (seed=42 split)"
    )

    # One-shot preload all CaBuAr patches into CPU tensors so the eval and FT loops
    # are RAM-bound, not disk-bound. The previous run spent >>15 min on per-patch
    # disk reads alone.
    print("Preloading CaBuAr patches into memory...")
    ds_full = PatchPairDataset(args.cabuar_patch_index, normalize=False)
    channels = ds_full.channels_per_image
    pre_eval, post_eval, label_eval = preload_patches(ds_full, eval_idx)
    pre_ft, post_ft, label_ft = preload_patches(ds_full, ft_idx)
    print(
        f"  eval tensors: pre={tuple(pre_eval.shape)}, label={tuple(label_eval.shape)}; "
        f"ft tensors: pre={tuple(pre_ft.shape)}"
    )

    rows = []
    for arch in args.archs:
        for seed in args.seeds:
            ckpt = args.ckpt_root / f"{arch}_qaon_seed{seed}_reflectance64" / "best_model.pt"
            if not ckpt.exists():
                print(f"  SKIP {arch} seed={seed}: checkpoint not found at {ckpt}")
                continue

            # Build model and load California checkpoint.
            model = build_model(arch, channels_per_image=channels, base_channels=64).to(device)
            state = torch.load(ckpt, map_location=device, weights_only=False)
            model.load_state_dict(state["model_state_dict"] if "model_state_dict" in state else state)

            # Cold-start (epoch 0) eval before any FT.
            cold = evaluate_pixel_pooled_preloaded(model, pre_eval, post_eval, label_eval, device, args.batch_size)
            rows.append({"arch": arch, "seed": seed, "epoch": 0, **cold})
            print(
                f"  {arch} seed={seed} epoch=0 (cold)  IoU={cold['iou']:.4f}  recall={cold['recall']:.3f}  precision={cold['precision']:.4f}"
            )

            # Fine-tune K epochs; evaluate after each.
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
            n_ft = pre_ft.shape[0]
            for epoch in range(1, args.epochs + 1):
                model.train()
                rng = np.random.default_rng(1000 * seed + epoch)
                shuffled = rng.permutation(n_ft)
                for i in range(0, n_ft, args.batch_size):
                    sel = shuffled[i : i + args.batch_size]
                    pre = pre_ft[sel].to(device, non_blocking=True)
                    post = post_ft[sel].to(device, non_blocking=True)
                    label = label_ft[sel].to(device, non_blocking=True)
                    logits = model(pre, post)
                    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, label)
                    prob = torch.sigmoid(logits)
                    dice_num = 2 * (prob * label).sum() + 1e-6
                    dice_den = prob.sum() + label.sum() + 1e-6
                    dice = 1 - dice_num / dice_den
                    loss = 0.5 * bce + 0.5 * dice
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                ft_eval = evaluate_pixel_pooled_preloaded(model, pre_eval, post_eval, label_eval, device, args.batch_size)
                rows.append({"arch": arch, "seed": seed, "epoch": epoch, **ft_eval})
                print(
                    f"  {arch} seed={seed} epoch={epoch}      IoU={ft_eval['iou']:.4f}  recall={ft_eval['recall']:.3f}  precision={ft_eval['precision']:.4f}"
                )

            pd.DataFrame(rows).to_csv(args.out_csv, index=False)

    pd.DataFrame(rows).to_csv(args.out_csv, index=False)
    print(f"\nWrote {args.out_csv}")


if __name__ == "__main__":
    main()

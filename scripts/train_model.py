from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.factory import build_model, list_models
from src.training.datasets import PatchPairDataset, create_train_val_loaders
from src.training.train_loop import train_segmentation_model


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a change-segmentation model.")
    parser.add_argument("--model-name", choices=list_models(), required=True)
    parser.add_argument("--patch-index", type=Path, required=True, help="Path to patch_index.csv.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--val-split", choices=["spatial", "event", "random"], default="spatial")
    parser.add_argument("--block-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dynamic-pos-weight", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pos-weight-max", type=float, default=20.0)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _set_seed(args.seed)
    dataset = PatchPairDataset(args.patch_index, normalize=args.normalize)
    train_loader, val_loader = create_train_val_loaders(
        args.patch_index,
        batch_size=args.batch_size,
        val_fraction=args.val_fraction,
        seed=args.seed,
        augment_train=args.augment,
        normalize=args.normalize,
        split_strategy=args.val_split,
        block_size=args.block_size,
        num_workers=args.num_workers,
    )
    model = build_model(
        args.model_name,
        channels_per_image=dataset.channels_per_image,
        base_channels=args.base_channels,
    )

    history_path = train_segmentation_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        output_dir=args.output_dir,
        epochs=args.epochs,
        lr=args.learning_rate,
        device=args.device,
        dynamic_pos_weight=args.dynamic_pos_weight,
        pos_weight_max=args.pos_weight_max,
        amp=args.amp,
    )

    run_config = {
        "model_name": args.model_name,
        "patch_index": str(args.patch_index),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "val_fraction": args.val_fraction,
        "val_split": args.val_split,
        "block_size": args.block_size,
        "seed": args.seed,
        "num_workers": args.num_workers,
        "base_channels": args.base_channels,
        "normalize": args.normalize,
        "augment": args.augment,
        "dynamic_pos_weight": args.dynamic_pos_weight,
        "pos_weight_max": args.pos_weight_max,
        "amp": args.amp,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    print(f"Wrote training history: {history_path}", flush=True)
    print(f"Wrote best checkpoint: {args.output_dir / 'best_model.pt'}", flush=True)
    print(f"Wrote run config: {args.output_dir / 'run_config.json'}", flush=True)


if __name__ == "__main__":
    main()

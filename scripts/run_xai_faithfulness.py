from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.xai_metrics import insertion_deletion_scores
from src.explainability.gradcam import SegmentationGradCAM
from src.explainability.integrated_gradients import SegmentationIntegratedGradients
from src.explainability.occlusion import SegmentationOcclusion
from src.models.factory import build_model, list_models
from src.training.datasets import PatchPairDataset


def _target_layer(model: nn.Module) -> nn.Module:
    if hasattr(model, "bottleneck"):
        return getattr(model, "bottleneck")
    if hasattr(model, "aspp"):
        return getattr(model, "aspp")
    conv_layers = [module for module in model.modules() if isinstance(module, nn.Conv2d)]
    if not conv_layers:
        raise ValueError("Could not locate a convolutional target layer for Grad-CAM.")
    return conv_layers[-1]


def _sample_indices(patch_index: pd.DataFrame, *, n: int, seed: int, min_burned_fraction: float) -> list[int]:
    candidates = patch_index.index.to_numpy()
    if "burned_fraction" in patch_index.columns:
        burned = patch_index["burned_fraction"].fillna(0.0).to_numpy(dtype=np.float64)
        selected = patch_index.index[burned >= min_burned_fraction].to_numpy()
        if selected.size > 0:
            candidates = selected
    rng = np.random.default_rng(seed)
    size = min(n, candidates.size)
    return rng.choice(candidates, size=size, replace=False).astype(int).tolist()


def _curve_record(
    *,
    method: str,
    sample_index: int,
    scores: dict[str, object],
) -> dict[str, object]:
    return {
        "sample_index": int(sample_index),
        "method": method,
        "base_target": float(scores["base_target"]),
        "baseline_target": float(scores["baseline_target"]),
        "deletion_auc": float(scores["deletion_auc"]),
        "insertion_auc": float(scores["insertion_auc"]),
        "faithfulness_gap": float(scores["faithfulness_gap"]),
        "fractions": scores["fractions"],
        "deletion_curve": scores["deletion_curve"],
        "insertion_curve": scores["insertion_curve"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run XAI faithfulness with physical mean-reflectance perturbation baselines.")
    parser.add_argument("--model-name", choices=list_models(), default="siamese")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--patch-index", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--num-samples", type=int, default=200)
    parser.add_argument("--min-burned-fraction", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--ig-steps", type=int, default=32)
    parser.add_argument("--occlusion-window", type=int, default=32)
    parser.add_argument("--occlusion-stride", type=int, default=16)
    parser.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = PatchPairDataset(args.patch_index, normalize=args.normalize)
    meta = pd.read_csv(args.patch_index)
    indices = _sample_indices(meta, n=args.num_samples, seed=args.seed, min_burned_fraction=args.min_burned_fraction)
    if not indices:
        raise ValueError("No samples available for XAI faithfulness.")

    model = build_model(args.model_name, channels_per_image=dataset.channels_per_image, base_channels=args.base_channels)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    device_obj = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model = model.to(device_obj)
    model.eval()

    gradcam = SegmentationGradCAM(model=model, target_layer=_target_layer(model))
    integrated_gradients = SegmentationIntegratedGradients(model=model, steps=args.ig_steps)
    occlusion = SegmentationOcclusion(model=model, window=args.occlusion_window, stride=args.occlusion_stride)

    rows: list[dict[str, object]] = []
    try:
        for sample_index in indices:
            sample = dataset[sample_index]
            pre = sample["pre"].unsqueeze(0).to(device_obj)
            post = sample["post"].unsqueeze(0).to(device_obj)

            method_maps = {
                "gradcam": gradcam.generate(pre, post),
                "integrated_gradients": integrated_gradients.generate(pre, post),
                "occlusion": occlusion.generate(pre, post),
            }
            for method, saliency in method_maps.items():
                scores = insertion_deletion_scores(model=model, pre=pre, post=post, saliency=saliency)
                rows.append(_curve_record(method=method, sample_index=sample_index, scores=scores))
            print(f"XAI sample {sample_index}: finished {len(method_maps)} methods")
    finally:
        gradcam.remove_hooks()

    out = {
        "model_name": args.model_name,
        "checkpoint": str(args.checkpoint),
        "patch_index": str(args.patch_index),
        "num_samples": len(indices),
        "seed": args.seed,
        "baseline": "per-channel spatial mean reflectance",
        "per_sample": rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(out, indent=2), encoding="utf-8")
    if args.output_csv is not None:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(args.output_csv, index=False)

    print(f"Wrote XAI faithfulness JSON: {args.output_json}")
    if args.output_csv is not None:
        print(f"Wrote XAI faithfulness CSV: {args.output_csv}")


if __name__ == "__main__":
    main()

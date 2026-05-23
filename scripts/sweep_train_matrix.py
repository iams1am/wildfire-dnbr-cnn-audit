from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.factory import list_models


def _patch_index_for_qa(qa: str) -> Path:
    if qa == "on":
        return PROJECT_ROOT / "data" / "patches" / "california_full" / "patch_index.csv"
    if qa == "off":
        return PROJECT_ROOT / "data" / "patches_noqa" / "california_full" / "patch_index.csv"
    raise ValueError(f"Unsupported QA state: {qa}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the full model x QA x seed matrix.")
    parser.add_argument("--models", nargs="+", choices=list_models(), default=list_models())
    parser.add_argument("--qa-states", nargs="+", choices=["on", "off"], default=["on", "off"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 17, 2026])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--val-split", choices=["spatial", "event", "random"], default="spatial")
    parser.add_argument("--block-size", type=int, default=1024)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--run-suffix", type=str, default="_reflectance64")
    parser.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dynamic-pos-weight", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pos-weight-max", type=float, default=20.0)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--amp-exclude", nargs="*", default=["change_transformer"])
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for qa in args.qa_states:
        patch_index = _patch_index_for_qa(qa)
        if not patch_index.exists():
            raise FileNotFoundError(f"Patch index not found for QA {qa}: {patch_index}")
        for seed in args.seeds:
            for model in args.models:
                output_dir = PROJECT_ROOT / "data" / "runs" / f"{model}_qa{qa}_seed{seed}{args.run_suffix}"
                if args.skip_existing and (output_dir / "best_model.pt").exists() and (output_dir / "history.csv").exists():
                    print(f"Skipping existing completed run: {output_dir}", flush=True)
                    continue
                use_amp = bool(args.amp and model not in set(args.amp_exclude))
                cmd = [
                    sys.executable,
                    str(PROJECT_ROOT / "scripts" / "train_model.py"),
                    "--model-name",
                    model,
                    "--patch-index",
                    str(patch_index),
                    "--output-dir",
                    str(output_dir),
                    "--epochs",
                    str(args.epochs),
                    "--batch-size",
                    str(args.batch_size),
                    "--learning-rate",
                    str(args.learning_rate),
                    "--val-fraction",
                    str(args.val_fraction),
                    "--val-split",
                    args.val_split,
                    "--block-size",
                    str(args.block_size),
                    "--seed",
                    str(seed),
                    "--base-channels",
                    str(args.base_channels),
                    "--device",
                    args.device,
                    "--num-workers",
                    str(args.num_workers),
                    "--pos-weight-max",
                    str(args.pos_weight_max),
                ]
                cmd.append("--normalize" if args.normalize else "--no-normalize")
                cmd.append("--augment" if args.augment else "--no-augment")
                cmd.append("--dynamic-pos-weight" if args.dynamic_pos_weight else "--no-dynamic-pos-weight")
                cmd.append("--amp" if use_amp else "--no-amp")
                print(" ".join(cmd), flush=True)
                if not args.dry_run:
                    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()

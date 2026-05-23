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
        return PROJECT_ROOT / "data" / "patches" / "australia_full" / "patch_index.csv"
    if qa == "off":
        return PROJECT_ROOT / "data" / "patches_noqa" / "australia_full" / "patch_index.csv"
    raise ValueError(f"Unsupported QA state: {qa}")


def _manifest_for_qa(qa: str) -> Path:
    if qa == "on":
        return PROJECT_ROOT / "data" / "manifests" / "australia_external_test_manifest_harmonized.csv"
    if qa == "off":
        return PROJECT_ROOT / "data" / "manifests" / "australia_external_test_manifest_noqa_full_harmonized.csv"
    raise ValueError(f"Unsupported QA state: {qa}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained California models on the Australia target set.")
    parser.add_argument("--models", nargs="+", choices=list_models(), default=list_models())
    parser.add_argument("--qa-states", nargs="+", choices=["on", "off"], default=["on", "off"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 17, 2026])
    parser.add_argument("--run-suffix", type=str, default="_reflectance64")
    parser.add_argument("--eval-suffix", type=str, default="_reflectance64")
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--scene-stitched", action="store_true")
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for qa in args.qa_states:
        patch_index = _patch_index_for_qa(qa)
        manifest = _manifest_for_qa(qa)
        if not patch_index.exists():
            raise FileNotFoundError(f"Patch index not found for QA {qa}: {patch_index}")
        if args.scene_stitched and not manifest.exists():
            raise FileNotFoundError(f"Manifest not found for QA {qa}: {manifest}")

        for seed in args.seeds:
            eval_name = f"australia_{'noqa_' if qa == 'off' else ''}full_seed{seed}{args.eval_suffix}"
            if args.scene_stitched:
                eval_name += "_stitched"
            output_dir = PROJECT_ROOT / "data" / "evaluation" / eval_name

            for model in args.models:
                checkpoint = PROJECT_ROOT / "data" / "runs" / f"{model}_qa{qa}_seed{seed}{args.run_suffix}" / "best_model.pt"
                if not checkpoint.exists():
                    raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

                if args.scene_stitched:
                    expected_output = output_dir / model / f"{model}_scene_summary.json"
                    cmd = [
                        sys.executable,
                        str(PROJECT_ROOT / "scripts" / "evaluate_scene_stitched.py"),
                        "--model-name",
                        model,
                        "--checkpoint",
                        str(checkpoint),
                        "--manifest",
                        str(manifest),
                        "--output-dir",
                        str(output_dir / model),
                        "--base-channels",
                        str(args.base_channels),
                        "--patch-size",
                        str(args.patch_size),
                        "--stride",
                        str(args.stride),
                        "--threshold",
                        str(args.threshold),
                        "--device",
                        args.device,
                    ]
                else:
                    expected_output = output_dir / f"{model}_summary.json"
                    cmd = [
                        sys.executable,
                        str(PROJECT_ROOT / "scripts" / "evaluate_model.py"),
                        "--model-name",
                        model,
                        "--checkpoint",
                        str(checkpoint),
                        "--patch-index",
                        str(patch_index),
                        "--output-dir",
                        str(output_dir),
                        "--base-channels",
                        str(args.base_channels),
                        "--batch-size",
                        str(args.batch_size),
                        "--num-workers",
                        str(args.num_workers),
                        "--threshold",
                        str(args.threshold),
                        "--device",
                        args.device,
                    ]
                if args.skip_existing and expected_output.exists():
                    print(f"Skipping existing evaluation: {expected_output}")
                    continue
                cmd.append("--normalize" if args.normalize else "--no-normalize")
                print(" ".join(cmd))
                if not args.dry_run:
                    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()

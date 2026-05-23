from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.dnbr_baseline import sweep_dnbr_thresholds
from src.models.factory import list_models


def _run(cmd: list[str], *, dry_run: bool) -> None:
    print(" ".join(cmd), flush=True)
    if not dry_run:
        subprocess.run(cmd, check=True)


def _cross_region_eval_cmd(args: argparse.Namespace, seed_args: list[str], *, scene_stitched: bool) -> list[str]:
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "sweep_cross_region_eval.py"),
        "--models",
        *args.models,
        "--seeds",
        *seed_args,
        "--run-suffix",
        args.run_suffix,
        "--eval-suffix",
        args.eval_suffix,
        "--base-channels",
        str(args.base_channels),
        "--batch-size",
        str(args.batch_size),
        "--num-workers",
        str(args.num_workers),
        "--device",
        args.device,
    ]
    if scene_stitched:
        cmd.append("--scene-stitched")
    return cmd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the corrected post-fix experiment pipeline.")
    parser.add_argument("--models", nargs="+", choices=list_models(), default=list_models())
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 17, 2026])
    parser.add_argument("--run-suffix", type=str, default="_reflectance64")
    parser.add_argument("--eval-suffix", type=str, default="_reflectance64")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--amp-exclude", nargs="*", default=["change_transformer"])
    parser.add_argument("--skip-existing-train", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--skip-dnbr", action="store_true")
    parser.add_argument("--skip-tables", action="store_true")
    parser.add_argument("--run-cabuar", action="store_true")
    parser.add_argument("--run-stitched-scenes", action="store_true")
    parser.add_argument("--run-per-event", action="store_true")
    parser.add_argument("--run-paired-qa", action="store_true")
    parser.add_argument("--run-latency", action="store_true")
    parser.add_argument("--run-xai", action="store_true")
    parser.add_argument("--run-kfold", action="store_true")
    parser.add_argument("--run-mtbs-model", action="store_true")
    parser.add_argument("--kfold-epochs", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_args = [str(seed) for seed in args.seeds]

    if not args.skip_train:
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "sweep_train_matrix.py"),
            "--models",
            *args.models,
            "--seeds",
            *seed_args,
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--base-channels",
            str(args.base_channels),
            "--device",
            args.device,
            "--num-workers",
            str(args.num_workers),
            "--run-suffix",
            args.run_suffix,
            "--amp-exclude",
            *args.amp_exclude,
            "--amp" if args.amp else "--no-amp",
        ]
        if args.skip_existing_train:
            cmd.append("--skip-existing")
        _run(cmd, dry_run=args.dry_run)

    if not args.skip_dnbr:
        thresholds = [round(float(v), 3) for v in np.arange(0.02, 0.301, 0.02)]
        for qa, patch_dir in (
            ("on", PROJECT_ROOT / "data" / "patches" / "australia_full" / "patch_index.csv"),
            ("off", PROJECT_ROOT / "data" / "patches_noqa" / "australia_full" / "patch_index.csv"),
        ):
            out_dir = PROJECT_ROOT / "data" / "evaluation" / f"australia_{'noqa_' if qa == 'off' else ''}full_dnbr{args.eval_suffix}"
            out_csv = out_dir / "dnbr_sweep.csv"
            print(f"dNBR sweep QA={qa}: {out_csv}")
            if not args.dry_run:
                sweep_dnbr_thresholds(patch_index_csv=patch_dir, thresholds=thresholds, output_csv=out_csv, normalize=True)

    if not args.skip_eval:
        # Patch-level Australia evaluation is the source for the results tables.
        # Stitched scene inference is an additional artifact for edge-effect checks.
        _run(_cross_region_eval_cmd(args, seed_args, scene_stitched=False), dry_run=args.dry_run)
        if args.run_stitched_scenes:
            _run(_cross_region_eval_cmd(args, seed_args, scene_stitched=True), dry_run=args.dry_run)

    if args.run_per_event:
        table_dir = PROJECT_ROOT / "data" / "paper_assets" / "tables"
        for seed in args.seeds:
            checkpoint = PROJECT_ROOT / "data" / "runs" / f"siamese_qaoff_seed{seed}{args.run_suffix}" / "best_model.pt"
            _run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "scripts" / "evaluate_per_event.py"),
                    "--model-name",
                    "siamese",
                    "--checkpoint",
                    str(checkpoint),
                    "--patch-index",
                    str(PROJECT_ROOT / "data" / "patches_noqa" / "australia_full" / "patch_index.csv"),
                    "--output-csv",
                    str(table_dir / f"per_event_results_seed{seed}{args.eval_suffix}.csv"),
                    "--base-channels",
                    str(args.base_channels),
                    "--batch-size",
                    str(args.batch_size),
                    "--device",
                    args.device,
                    "--normalize",
                ],
                dry_run=args.dry_run,
            )
        _run(
            [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "aggregate_per_event_3seed.py"),
                "--seeds",
                *seed_args,
                "--input-template",
                str(table_dir / f"per_event_results_seed{{seed}}{args.eval_suffix}.csv"),
                "--output-csv",
                str(table_dir / f"per_event_results_3seed{args.eval_suffix}.csv"),
                "--output-md",
                str(table_dir / f"per_event_results_3seed{args.eval_suffix}.md"),
            ],
            dry_run=args.dry_run,
        )

    if args.run_paired_qa:
        table_dir = PROJECT_ROOT / "data" / "paper_assets" / "tables"
        for model in ("baseline", "siamese"):
            for seed in args.seeds:
                _run(
                    [
                        sys.executable,
                        str(PROJECT_ROOT / "scripts" / "paired_qa_comparison.py"),
                        "--model-name",
                        model,
                        "--checkpoint-on",
                        str(PROJECT_ROOT / "data" / "runs" / f"{model}_qaon_seed{seed}{args.run_suffix}" / "best_model.pt"),
                        "--checkpoint-off",
                        str(PROJECT_ROOT / "data" / "runs" / f"{model}_qaoff_seed{seed}{args.run_suffix}" / "best_model.pt"),
                        "--output-json",
                        str(table_dir / f"paired_qa_comparison_{model}_seed{seed}{args.eval_suffix}.json"),
                        "--output-csv",
                        str(table_dir / f"paired_qa_comparison_{model}_seed{seed}{args.eval_suffix}.csv"),
                        "--base-channels",
                        str(args.base_channels),
                        "--batch-size",
                        str(args.batch_size),
                        "--device",
                        args.device,
                        "--normalize",
                    ],
                    dry_run=args.dry_run,
                )
            _run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "scripts" / "aggregate_paired_qa_3seed.py"),
                    "--model-name",
                    model,
                    "--seeds",
                    *seed_args,
                    "--input-template",
                    str(table_dir / f"paired_qa_comparison_{model}_seed{{seed}}{args.eval_suffix}.csv"),
                    "--output-json",
                    str(table_dir / f"paired_qa_comparison_{model}_3seed{args.eval_suffix}.json"),
                    "--output-csv",
                    str(table_dir / f"paired_qa_comparison_{model}_3seed{args.eval_suffix}.csv"),
                ],
                dry_run=args.dry_run,
            )

    if args.run_cabuar:
        _run(
            [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "cabuar_evaluate_with_ci.py"),
                "--models",
                *args.models,
                "--seeds",
                *seed_args,
                "--run-suffix",
                args.run_suffix,
                "--base-channels",
                str(args.base_channels),
                "--batch-size",
                str(args.batch_size),
                "--device",
                args.device,
            ],
            dry_run=args.dry_run,
        )
        _run(
            [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "aggregate_cabuar_table.py"),
            ],
            dry_run=args.dry_run,
        )

    if args.run_latency:
        _run(
            [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "measure_inference_latency.py"),
            ],
            dry_run=args.dry_run,
        )

    if args.run_kfold:
        _run(
            [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "run_kfold_in_domain.py"),
                "--models",
                "baseline",
                "siamese",
                "--seeds",
                *seed_args,
                "--epochs",
                str(args.kfold_epochs),
                "--batch-size",
                str(args.batch_size),
                "--base-channels",
                str(args.base_channels),
                "--device",
                args.device,
                "--num-workers",
                str(args.num_workers),
                "--output-dir",
                str(PROJECT_ROOT / "data" / "paper_assets" / f"stats{args.eval_suffix}"),
                "--amp" if args.amp else "--no-amp",
            ],
            dry_run=args.dry_run,
        )

    if args.run_mtbs_model:
        _run(
            [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "mtbs_kfold_model_validation.py"),
                "--model-name",
                "siamese",
                "--kfold-dir",
                str(PROJECT_ROOT / "data" / "paper_assets" / f"stats{args.eval_suffix}"),
                "--output-dir",
                str(PROJECT_ROOT / "data" / "paper_assets" / f"mtbs_model_validation{args.eval_suffix}"),
                "--base-channels",
                str(args.base_channels),
                "--device",
                args.device,
            ],
            dry_run=args.dry_run,
        )

    if args.run_xai:
        for seed in args.seeds:
            checkpoint = PROJECT_ROOT / "data" / "runs" / f"siamese_qaoff_seed{seed}{args.run_suffix}" / "best_model.pt"
            _run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "scripts" / "run_xai_faithfulness.py"),
                    "--model-name",
                    "siamese",
                    "--checkpoint",
                    str(checkpoint),
                    "--patch-index",
                    str(PROJECT_ROOT / "data" / "patches_noqa" / "australia_full" / "patch_index.csv"),
                    "--output-json",
                    str(PROJECT_ROOT / "data" / "paper_assets" / "xai_full" / f"faithfulness_n200_seed{seed}{args.eval_suffix}.json"),
                    "--output-csv",
                    str(PROJECT_ROOT / "data" / "paper_assets" / "xai_full" / f"faithfulness_n200_seed{seed}{args.eval_suffix}.csv"),
                    "--num-samples",
                    "200",
                    "--seed",
                    str(seed),
                    "--base-channels",
                    str(args.base_channels),
                    "--device",
                    args.device,
                ],
                dry_run=args.dry_run,
            )
        _run(
            [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "aggregate_xai_3seed.py"),
                "--seeds",
                *seed_args,
                "--suffix",
                args.eval_suffix,
                "--output-stem",
                f"faithfulness_n200_3seed{args.eval_suffix}",
            ],
            dry_run=args.dry_run,
        )

    table_script = PROJECT_ROOT / "scripts" / "aggregate_results_tables.py"
    if not args.skip_tables and table_script.exists():
        _run(
            [
                sys.executable,
                str(table_script),
                "--models",
                *args.models,
                "--seeds",
                *seed_args,
                "--qa-on-suffix",
                args.eval_suffix,
                "--qa-off-suffix",
                args.eval_suffix,
                "--dnbr-sweep-csv",
                str(PROJECT_ROOT / "data" / "evaluation" / f"australia_full_dnbr{args.eval_suffix}" / "dnbr_sweep.csv"),
                "--results-name",
                f"results_table{args.eval_suffix}",
                "--ablation-name",
                f"ablation_table{args.eval_suffix}",
            ],
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()

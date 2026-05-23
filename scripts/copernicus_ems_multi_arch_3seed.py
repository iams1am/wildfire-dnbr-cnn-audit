"""Run the multi-architecture Copernicus EMS validation across all three seeds
{42, 17, 2026} for Concat U-Net, Siamese U-Net, and DeepLabv3+, so the EMS
table is not single-seed.

Per (arch, seed) it does stitched scene inference over the Mediterranean noqa
manifest with the QA-off checkpoint, then rasterizes and scores IoU against
each EMS activation. Aggregates to IoU mean +/- seed-std per (arch, event,
sensor) and a valid-pair mean over the 5 valid event-sensor pairs with a
seed-pooled bootstrap CI per arch.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "manifests" / "mediterranean_external_test_manifest_harmonized_noqa.csv"
DEFAULT_CASES = PROJECT_ROOT / "data" / "external" / "copernicus_ems" / "copernicus_ems_cases.json"
ARCHS = ["deeplab", "baseline", "siamese"]
SEEDS = [42, 17, 2026]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-arch x multi-seed EMS validation driver.")
    parser.add_argument("--archs", nargs="+", default=ARCHS)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=PROJECT_ROOT / "data" / "paper_assets" / "tables" / "copernicus_ems_multi_arch_3seed.csv",
    )
    return parser.parse_args()


def stitched_dir(arch: str, seed: int) -> Path:
    return PROJECT_ROOT / "data" / "evaluation" / f"copernicus_ems_{arch}_stitched_seed{seed}_reflectance64"


def checkpoint_path(arch: str, seed: int) -> Path:
    return PROJECT_ROOT / "data" / "runs" / f"{arch}_qaoff_seed{seed}_reflectance64" / "best_model.pt"


def run_stitching(arch: str, seed: int, manifest: Path, device: str) -> Path:
    out_dir = stitched_dir(arch, seed)
    summary_json = out_dir / f"{arch}_scene_summary.json"
    if summary_json.exists():
        print(f"  [{arch} seed={seed}] cached stitched output at {out_dir}")
        return out_dir
    ckpt = checkpoint_path(arch, seed)
    if not ckpt.exists():
        raise SystemExit(f"Missing checkpoint: {ckpt}")
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "evaluate_scene_stitched.py"),
        "--model-name", arch,
        "--checkpoint", str(ckpt),
        "--manifest", str(manifest),
        "--output-dir", str(out_dir),
        "--device", device,
    ]
    print(f"  [{arch} seed={seed}] stitching ...")
    subprocess.run(cmd, check=True)
    return out_dir


def run_ems_validation(arch: str, seed: int, prediction_root: Path, work_dir: Path) -> Path:
    base_cases = json.loads(DEFAULT_CASES.read_text(encoding="utf-8"))
    arch_seed_cases = []
    for case in base_cases:
        case_copy = deepcopy(case)
        case_copy["prediction_root"] = str(prediction_root)
        case_copy["comparison_tag"] = f"{case['comparison_tag']}_{arch}_s{seed}"
        arch_seed_cases.append(case_copy)
    work_dir.mkdir(parents=True, exist_ok=True)
    case_json = work_dir / f"ems_cases_{arch}_seed{seed}.json"
    case_json.write_text(json.dumps(arch_seed_cases, indent=2))
    out_csv = work_dir / f"ems_{arch}_seed{seed}.csv"
    out_json = work_dir / f"ems_{arch}_seed{seed}.json"
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "copernicus_ems_validation.py"),
        "--case-config", str(case_json),
        "--output-csv", str(out_csv),
        "--output-json", str(out_json),
    ]
    print(f"  [{arch} seed={seed}] EMS validation ...")
    subprocess.run(cmd, check=True)
    return out_csv


def main() -> None:
    args = parse_args()
    work_dir = args.out_csv.parent / "copernicus_ems_multi_arch_3seed_work"
    long_rows: list[pd.DataFrame] = []
    for seed in args.seeds:
        for arch in args.archs:
            run_stitching(arch, seed, args.manifest, args.device)
            pred_root = stitched_dir(arch, seed)
            csv = run_ems_validation(arch, seed, pred_root, work_dir)
            df = pd.read_csv(csv)
            df["arch"] = arch
            df["seed"] = seed
            long_rows.append(df)
    long = pd.concat(long_rows, ignore_index=True).dropna(subset=["iou"])
    long.to_csv(args.out_csv.with_name(args.out_csv.stem + "_long.csv"), index=False)


    rows = []
    rng = np.random.default_rng(42)
    for (arch, event, sensor), grp in long.groupby(["arch", "event_id", "sensor"]):
        dnbr = grp[grp["comparison"].str.startswith("dnbr_vs_")]
        cnn = grp[grp["comparison"].str.startswith("deeplab_vs_")]
        if dnbr.empty or cnn.empty:
            continue
        cnn_iou = cnn["iou"].astype(float).to_numpy()
        cnn_mean = float(cnn_iou.mean())
        cnn_std = float(cnn_iou.std(ddof=1)) if cnn_iou.size > 1 else 0.0
        dnbr_iou = float(dnbr["iou"].astype(float).iloc[0])
        delta = cnn_mean - dnbr_iou
        rows.append({
            "arch": arch,
            "event_id": event,
            "sensor": sensor,
            "dnbr_iou": round(dnbr_iou, 4),
            "cnn_iou_mean": round(cnn_mean, 4),
            "cnn_iou_std": round(cnn_std, 4),
            "delta": round(delta, 4),
            "n_seeds": int(cnn_iou.size),
        })
    per_event = pd.DataFrame(rows).sort_values(["arch", "event_id", "sensor"])

    # Valid-pair means per arch with bootstrap CI over the 5 events (paired).
    summary_rows = []
    for arch, grp in per_event.groupby("arch"):
        deltas = grp["delta"].to_numpy()
        mean = float(deltas.mean())
        boot = rng.choice(deltas, size=(20000, deltas.size), replace=True).mean(axis=1)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        summary_rows.append({
            "arch": arch,
            "n_valid_pairs": int(deltas.size),
            "valid_pair_mean_delta": round(mean, 4),
            "ci95_lo": round(float(lo), 4),
            "ci95_hi": round(float(hi), 4),
        })
    summary = pd.DataFrame(summary_rows)
    per_event.to_csv(args.out_csv, index=False)
    summary_path = args.out_csv.with_name("copernicus_ems_multi_arch_3seed_summary.csv")
    summary.to_csv(summary_path, index=False)
    print()
    print("=== Per (arch, event, sensor) 3-seed mean ===")
    print(per_event.to_string(index=False))
    print()
    print("=== Valid-pair-mean summary per arch ===")
    print(summary.to_string(index=False))
    print(f"\nWrote {args.out_csv}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()

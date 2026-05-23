"""Running the Copernicus EMS independent validation for several architectures, not
just the headline one, so the EMS table is symmetric with the Australia table.

For each (arch, seed=42) checkpoint it does stitched scene inference over the
Mediterranean manifest, re-runs copernicus_ems_validation.py against each EMS
activation (Rhodes/Sierra/Evros) pointed at that arch's predictions, and
aggregates dNBR-IoU vs CNN-IoU into copernicus_ems_multi_arch.csv .
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "manifests" / "mediterranean_external_test_manifest_harmonized_noqa.csv"
DEFAULT_CASES = PROJECT_ROOT / "data" / "external" / "copernicus_ems" / "copernicus_ems_cases.json"
ARCHS = ["deeplab", "baseline", "siamese"]
SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-architecture EMS validation driver.")
    parser.add_argument("--archs", nargs="+", default=ARCHS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=PROJECT_ROOT / "data" / "paper_assets" / "tables" / "copernicus_ems_multi_arch.csv",
    )
    parser.add_argument(
        "--skip-stitching",
        action="store_true",
        help="Reuse existing stitched outputs; only re-run EMS validation.",
    )
    return parser.parse_args()


def stitched_dir(arch: str, seed: int) -> Path:
    return PROJECT_ROOT / "data" / "evaluation" / f"copernicus_ems_{arch}_stitched_seed{seed}_reflectance64"


def checkpoint_path(arch: str, seed: int) -> Path:
    return PROJECT_ROOT / "data" / "runs" / f"{arch}_qaoff_seed{seed}_reflectance64" / "best_model.pt"


def run_stitching(arch: str, seed: int, manifest: Path, device: str) -> Path:
    out_dir = stitched_dir(arch, seed)
    if (out_dir / f"{arch}_scene_summary.json").exists():
        print(f"  [{arch} seed={seed}] stitched output already exists at {out_dir}; reusing")
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
    print(f"  [{arch} seed={seed}] stitching to {out_dir} ...")
    subprocess.run(cmd, check=True)
    return out_dir


def run_ems_validation(arch: str, prediction_root: Path, out_dir: Path) -> Path:
    """Run scripts/copernicus_ems_validation.py with a per-arch case config
    that points prediction_root at the new directory."""
    out_dir.mkdir(parents=True, exist_ok=True)
    base_cases = json.loads(DEFAULT_CASES.read_text(encoding="utf-8"))
    arch_cases = []
    for case in base_cases:
        case_copy = deepcopy(case)
        case_copy["prediction_root"] = str(prediction_root)
        case_copy["comparison_tag"] = f"{case['comparison_tag']}_{arch}"
        arch_cases.append(case_copy)
    arch_case_json = out_dir / f"copernicus_ems_cases_{arch}.json"
    arch_case_json.write_text(json.dumps(arch_cases, indent=2))

    out_csv = out_dir / f"copernicus_ems_{arch}.csv"
    out_json = out_dir / f"copernicus_ems_{arch}.json"
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "copernicus_ems_validation.py"),
        "--case-config", str(arch_case_json),
        "--output-csv", str(out_csv),
        "--output-json", str(out_json),
    ]
    print(f"  [{arch}] EMS validation -> {out_csv}")
    subprocess.run(cmd, check=True)
    return out_csv


def main() -> None:
    args = parse_args()
    rows_all: list[pd.DataFrame] = []
    out_dir = args.out_csv.parent
    work_dir = out_dir / "copernicus_ems_multi_arch_work"

    for arch in args.archs:
        if not args.skip_stitching:
            run_stitching(arch, args.seed, args.manifest, args.device)
        else:
            print(f"  [{arch}] skipping stitching (per --skip-stitching)")
        pred_root = stitched_dir(arch, args.seed)
        if not pred_root.exists() and not args.skip_stitching:
            raise SystemExit(f"Stitched output missing: {pred_root}")
        ems_csv = run_ems_validation(arch, pred_root, work_dir)
        df = pd.read_csv(ems_csv)
        df["arch"] = arch
        rows_all.append(df)

    long = pd.concat(rows_all, ignore_index=True)
    long.to_csv(out_dir / "copernicus_ems_multi_arch_long.csv", index=False)


    long = long.dropna(subset=["iou"]).copy()
    summary_rows: list[dict] = []
    valid_pairs = long.loc[
        long["comparison"].astype(str).str.startswith("dnbr_vs_"),
        ["event_id", "sensor", "iou", "source"],
    ].rename(columns={"iou": "dnbr_iou"}).drop_duplicates(subset=["event_id", "sensor"])
    for _, row in valid_pairs.iterrows():
        rec = {
            "event_id": row["event_id"],
            "sensor": row["sensor"],
            "source": row["source"],
            "dnbr_iou": float(row["dnbr_iou"]),
        }
        for arch in args.archs:
            arch_rows = long[
                (long["event_id"] == row["event_id"])
                & (long["sensor"] == row["sensor"])
                & (long["arch"] == arch)
                & long["comparison"].astype(str).str.contains(arch)
            ]
            if arch_rows.empty:
                rec[f"{arch}_iou"] = float("nan")
                rec[f"{arch}_delta"] = float("nan")
            else:
                arch_iou = float(arch_rows["iou"].iloc[0])
                rec[f"{arch}_iou"] = arch_iou
                rec[f"{arch}_delta"] = round(arch_iou - float(row["dnbr_iou"]), 4)
        summary_rows.append(rec)

    summary = pd.DataFrame(summary_rows)
    # Valid-pair mean per architecture.
    means = {"event_id": "VALID-PAIR MEAN", "sensor": "--", "source": "--",
             "dnbr_iou": round(float(summary["dnbr_iou"].mean()), 4)}
    for arch in args.archs:
        means[f"{arch}_iou"] = round(float(summary[f"{arch}_iou"].mean()), 4)
        means[f"{arch}_delta"] = round(float(summary[f"{arch}_delta"].mean()), 4)
    summary_with_mean = pd.concat([summary, pd.DataFrame([means])], ignore_index=True)

    summary_with_mean.to_csv(args.out_csv, index=False)
    md_path = args.out_csv.with_suffix(".md")
    md_path.write_text(summary_with_mean.to_markdown(index=False), encoding="utf-8")
    print(f"\nWrote {args.out_csv}")
    print(f"Wrote {md_path}")
    print(summary_with_mean.to_string(index=False))


if __name__ == "__main__":
    main()

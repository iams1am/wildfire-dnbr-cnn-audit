"""Direction-reversed transfer experiment (Australia-trained -> California test).

After training DeepLabv3+ on Australia patches (`scripts/prepare_au2ca_patch_index.py`
+ `scripts/train_model.py --model-name deeplab --patch-index data/patches/australia_full_train/...`),
this script evaluates the resulting checkpoint against MTBS California perimeters
using the same QA-valid, coverage-clipped protocol as the held-out MTBS table,
then writes a one-row diagnostic comparing AU->CA vs the headline CA->AU/CA-MTBS
DeepLabv3+ result.

Output: data/paper_assets/au2ca_mtbs_validation/{per_pair,per_event}.csv
        and a one-row summary file logging both transfer directions side-by-side.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AU->CA reverse transfer: evaluate AU-trained DeepLab on California MTBS perimeters.")
    p.add_argument("--checkpoint", type=Path, default=PROJECT_ROOT / "data" / "checkpoints" / "au2ca_deeplab_seed42" / "best_model.pt")
    p.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data" / "paper_assets" / "au2ca_mtbs_validation")
    p.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "data" / "manifests" / "california_train_val_manifest_harmonized.csv")
    p.add_argument("--mtbs-shapefile", type=Path, default=PROJECT_ROOT / "data" / "external" / "mtbs" / "mtbs_perims_DD.shp")
    p.add_argument("--base-channels", type=int, default=64)
    p.add_argument("--patch-size", type=int, default=256)
    p.add_argument("--stride", type=int, default=128)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.checkpoint.exists():
        raise SystemExit(
            f"Checkpoint not found: {args.checkpoint}\n"
            "Train AU-DeepLab first:\n"
            "  python scripts/prepare_au2ca_patch_index.py\n"
            "  python scripts/train_model.py --model-name deeplab \\\n"
            "    --patch-index data/patches/australia_full_train/patch_index.csv \\\n"
            "    --output-dir data/checkpoints/au2ca_deeplab_seed42 --epochs 20 --batch-size 4 \\\n"
            "    --learning-rate 1e-3 --seed 42 --base-channels 64 --val-split event \\\n"
            "    --normalize --augment --dynamic-pos-weight --amp"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, str(PROJECT_ROOT / "scripts" / "mtbs_model_validation.py"),
        "--model-name", "deeplab",
        "--checkpoint", str(args.checkpoint),
        "--manifest", str(args.manifest),
        "--mtbs-shapefile", str(args.mtbs_shapefile),
        "--output-dir", str(args.output_dir),
        "--base-channels", str(args.base_channels),
        "--patch-size", str(args.patch_size),
        "--stride", str(args.stride),
        "--threshold", str(args.threshold),
        "--device", args.device,
        "--normalize",
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

    # Aggregate the per-pair outputs into a one-row reverse-transfer summary
    per_pair_csv = args.output_dir / "deeplab_mtbs_per_pair.csv"
    if not per_pair_csv.exists():
        raise SystemExit(f"Missing expected output: {per_pair_csv}")
    df = pd.read_csv(per_pair_csv)
    usable = df[df["usable_for_iou"].astype(bool)].copy()
    delta = usable["model_iou"] - usable["dnbr_iou"]
    summary = {
        "direction": "AU -> CA (reverse transfer)",
        "checkpoint": str(args.checkpoint.relative_to(PROJECT_ROOT)),
        "n_pairs_evaluated": int(len(df)),
        "n_pairs_usable_for_iou": int(len(usable)),
        "n_events_usable": int(usable["event_id"].nunique()),
        "mean_dnbr_iou_california": float(usable["dnbr_iou"].mean()),
        "mean_au_trained_model_iou_california": float(usable["model_iou"].mean()),
        "mean_model_minus_dnbr_iou": float(delta.mean()),
        "median_model_minus_dnbr_iou": float(delta.median()),
        "pairs_model_beats_dnbr": int((delta > 0).sum()),
        "pairs_model_loses_to_dnbr": int((delta < 0).sum()),
        "headline_ca_to_mtbs_delta_iou": 0.029,
        "headline_ca_to_mtbs_ci95": "[+0.015, +0.042]",
        "headline_ca_to_mtbs_wl": "32/4",
        "interpretation": (
            "If AU->CA mean_model_minus_dnbr_iou is substantially less than the "
            "+0.029 CA->MTBS headline (or negative), the audit framework's "
            "direction-dependent asymmetry is empirically supported."
        ),
    }
    out_json = args.output_dir / "reverse_transfer_summary.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"\nWrote {per_pair_csv}")
    print(f"Wrote {out_json}")


if __name__ == "__main__":
    main()

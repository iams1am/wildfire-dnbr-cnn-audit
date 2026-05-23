from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.bootstrap import bootstrap_ci

METRICS = ["iou", "f1", "precision", "recall", "abs_pixel_error"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate paired QA-on/off patch comparisons across seeds.")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 17, 2026])
    parser.add_argument(
        "--input-template",
        type=str,
        default=str(PROJECT_ROOT / "data" / "paper_assets" / "tables" / "paired_qa_comparison_{model}_seed{seed}_reflectance64.csv"),
        help="Template containing {model} and {seed}.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Default: data/paper_assets/tables/paired_qa_comparison_<model>_3seed_reflectance64.json",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Default: data/paper_assets/tables/paired_qa_comparison_<model>_3seed_reflectance64.csv",
    )
    return parser.parse_args()


def _paired_test(on: np.ndarray, off: np.ndarray) -> dict[str, float]:
    mask = np.isfinite(on) & np.isfinite(off)
    on = on[mask]
    off = off[mask]
    delta = off - on
    t_stat, t_p = ttest_rel(on, off, nan_policy="omit")
    try:
        w_stat, w_p = wilcoxon(on, off)
    except ValueError:
        w_stat, w_p = float("nan"), float("nan")
    _, low, high = bootstrap_ci(delta, n_bootstrap=2000)
    return {
        "n": int(delta.size),
        "mean_on": float(np.nanmean(on)),
        "mean_off": float(np.nanmean(off)),
        "mean_delta_off_minus_on": float(np.nanmean(delta)),
        "delta_ci_low": float(low),
        "delta_ci_high": float(high),
        "paired_t_stat": float(t_stat),
        "paired_t_p_value": float(t_p),
        "wilcoxon_stat": float(w_stat),
        "wilcoxon_p_value": float(w_p),
    }


def main() -> None:
    args = parse_args()
    output_json = args.output_json or (
        PROJECT_ROOT
        / "data"
        / "paper_assets"
        / "tables"
        / f"paired_qa_comparison_{args.model_name}_3seed_reflectance64.json"
    )
    output_csv = args.output_csv or (
        PROJECT_ROOT
        / "data"
        / "paper_assets"
        / "tables"
        / f"paired_qa_comparison_{args.model_name}_3seed_reflectance64.csv"
    )

    frames: list[pd.DataFrame] = []
    for seed in args.seeds:
        path = Path(args.input_template.format(model=args.model_name, seed=seed))
        if not path.exists():
            raise FileNotFoundError(f"Missing paired QA CSV for {args.model_name} seed {seed}: {path}")
        df = pd.read_csv(path)
        df.insert(0, "seed", seed)
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)

    summary: dict[str, object] = {
        "model": args.model_name,
        "seeds": args.seeds,
        "matched_patch_count_per_seed": int(combined.groupby("seed").size().min()),
        "pooled_rows": int(len(combined)),
        "metrics": {},
    }
    for metric in METRICS:
        on_col = f"{metric}_on"
        off_col = f"{metric}_off"
        if on_col not in combined.columns or off_col not in combined.columns:
            continue
        summary["metrics"][metric] = _paired_test(
            combined[on_col].to_numpy(dtype=np.float64),
            combined[off_col].to_numpy(dtype=np.float64),
        )

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    combined.to_csv(output_csv, index=False)
    print(f"Wrote {output_json}")
    print(f"Wrote {output_csv}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

"""Aggregate held-out k-fold CNN-vs-MTBS comparisons and coverage sensitivity."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, ttest_1samp, wilcoxon

PROJECT_ROOT = Path(__file__).resolve().parents[1]


MODEL_LABELS = {
    "baseline": "Concat U-Net",
    "deeplab": "DeepLabv3+",
    "siamese": "Siamese U-Net",
}


def _bootstrap_ci(values: np.ndarray, *, n_bootstrap: int = 5000, seed: int = 42) -> tuple[float, float]:
    clean = values[np.isfinite(values)]
    if clean.size == 0:
        return float("nan"), float("nan")
    if clean.size == 1:
        value = float(clean[0])
        return value, value
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, clean.size, size=(n_bootstrap, clean.size))
    means = clean[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _delta_stats(delta: np.ndarray) -> dict[str, float | int]:
    clean = delta[np.isfinite(delta)]
    if clean.size == 0:
        return {
            "n": 0,
            "mean_delta": float("nan"),
            "median_delta": float("nan"),
            "ci95_low_bootstrap": float("nan"),
            "ci95_high_bootstrap": float("nan"),
            "wins": 0,
            "losses": 0,
            "ties": 0,
            "paired_t_stat": float("nan"),
            "paired_t_p_value": float("nan"),
            "wilcoxon_stat": float("nan"),
            "wilcoxon_p_value": float("nan"),
            "binomial_win_p_greater": float("nan"),
        }
    low, high = _bootstrap_ci(clean)
    wins = int((clean > 0).sum())
    losses = int((clean < 0).sum())
    ties = int((clean == 0).sum())
    try:
        t_stat, t_p = ttest_1samp(clean, popmean=0.0, nan_policy="omit")
    except ValueError:
        t_stat, t_p = float("nan"), float("nan")
    try:
        w_stat, w_p = wilcoxon(clean)
    except ValueError:
        w_stat, w_p = float("nan"), float("nan")
    try:
        binom_p = binomtest(wins, wins + losses, p=0.5, alternative="greater").pvalue if wins + losses else float("nan")
    except ValueError:
        binom_p = float("nan")
    return {
        "n": int(clean.size),
        "mean_delta": float(np.mean(clean)),
        "median_delta": float(np.median(clean)),
        "ci95_low_bootstrap": low,
        "ci95_high_bootstrap": high,
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "paired_t_stat": float(t_stat),
        "paired_t_p_value": float(t_p),
        "wilcoxon_stat": float(w_stat),
        "wilcoxon_p_value": float(w_p),
        "binomial_win_p_greater": float(binom_p),
    }


def _load_model_pairs(input_dir: Path, model: str) -> pd.DataFrame:
    path = input_dir / f"{model}_kfold_mtbs_per_pair.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing MTBS per-pair CSV for {model}: {path}")
    df = pd.read_csv(path)
    df["model"] = model
    df["model_label"] = MODEL_LABELS.get(model, model)
    return df


def _overall_rows(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model, df in frames.items():
        usable = df[df["usable_for_iou"].astype(bool)].copy()
        delta = usable["model_minus_dnbr_iou"].to_numpy(dtype=np.float64)
        stats = _delta_stats(delta)
        rows.append(
            {
                "model": model,
                "model_label": MODEL_LABELS.get(model, model),
                "n_pairs": int(len(df)),
                "n_pairs_usable": int(len(usable)),
                "n_events_usable": int(usable["event_id"].nunique()) if not usable.empty else 0,
                "mean_valid_coverage": float(df["valid_coverage"].mean(skipna=True)),
                "mean_dnbr_iou": float(usable["dnbr_iou"].mean(skipna=True)),
                "median_dnbr_iou": float(usable["dnbr_iou"].median(skipna=True)),
                "mean_model_iou": float(usable["model_iou"].mean(skipna=True)),
                "median_model_iou": float(usable["model_iou"].median(skipna=True)),
                **stats,
            }
        )
    return pd.DataFrame(rows)


def _pairwise_rows(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    key_cols = ["seed", "fold", "pair_id"]
    rows: list[dict[str, object]] = []
    models = sorted(frames)
    slim = {}
    for model, df in frames.items():
        usable = df[df["usable_for_iou"].astype(bool)].copy()
        slim[model] = usable[key_cols + ["event_id", "sensor", "valid_coverage", "model_iou"]].rename(
            columns={"model_iou": f"{model}_iou"}
        )
    for i, left_model in enumerate(models):
        for right_model in models[i + 1 :]:
            merged = slim[left_model].merge(slim[right_model], on=key_cols, suffixes=("_left", "_right"))
            delta = merged[f"{left_model}_iou"].to_numpy(dtype=np.float64) - merged[f"{right_model}_iou"].to_numpy(dtype=np.float64)
            rows.append(
                {
                    "comparison": f"{left_model}_iou_minus_{right_model}_iou",
                    "left_model": left_model,
                    "right_model": right_model,
                    **_delta_stats(delta),
                }
            )
            rows.append(
                {
                    "comparison": f"{right_model}_iou_minus_{left_model}_iou",
                    "left_model": right_model,
                    "right_model": left_model,
                    **_delta_stats(-delta),
                }
            )
    return pd.DataFrame(rows)


def _coverage_sensitivity(frames: dict[str, pd.DataFrame], thresholds: list[float]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model, df in frames.items():
        for threshold in thresholds:
            usable = df[df["usable_for_iou"].astype(bool) & (df["valid_coverage"] >= threshold)].copy()
            delta = usable["model_minus_dnbr_iou"].to_numpy(dtype=np.float64)
            stats = _delta_stats(delta)
            rows.append(
                {
                    "model": model,
                    "model_label": MODEL_LABELS.get(model, model),
                    "coverage_threshold": threshold,
                    "n_pairs": int(len(usable)),
                    "n_events": int(usable["event_id"].nunique()) if not usable.empty else 0,
                    "mean_valid_coverage": float(usable["valid_coverage"].mean(skipna=True)) if not usable.empty else float("nan"),
                    "mean_dnbr_iou": float(usable["dnbr_iou"].mean(skipna=True)) if not usable.empty else float("nan"),
                    "mean_model_iou": float(usable["model_iou"].mean(skipna=True)) if not usable.empty else float("nan"),
                    **stats,
                }
            )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "paper_assets" / "mtbs_model_validation_reflectance64",
    )
    parser.add_argument("--models", nargs="+", default=["baseline", "siamese", "deeplab"])
    parser.add_argument("--coverage-thresholds", nargs="+", type=float, default=[0.0, 0.1, 0.3, 0.5, 0.7])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frames = {model: _load_model_pairs(args.input_dir, model) for model in args.models}
    overall = _overall_rows(frames)
    pairwise = _pairwise_rows(frames)
    coverage = _coverage_sensitivity(frames, args.coverage_thresholds)

    overall_csv = args.input_dir / "mtbs_model_overall_comparison.csv"
    pairwise_csv = args.input_dir / "mtbs_model_comparison_stats.csv"
    coverage_csv = args.input_dir / "mtbs_coverage_sensitivity.csv"
    overall_json = args.input_dir / "mtbs_model_overall_comparison.json"
    pairwise_json = args.input_dir / "mtbs_model_comparison_stats.json"

    overall.to_csv(overall_csv, index=False)
    pairwise.to_csv(pairwise_csv, index=False)
    coverage.to_csv(coverage_csv, index=False)
    overall_json.write_text(json.dumps(overall.to_dict(orient="records"), indent=2), encoding="utf-8")
    pairwise_json.write_text(json.dumps(pairwise.to_dict(orient="records"), indent=2), encoding="utf-8")

    print(f"Wrote {overall_csv}")
    print(f"Wrote {pairwise_csv}")
    print(f"Wrote {coverage_csv}")
    print(overall.to_string(index=False))


if __name__ == "__main__":
    main()

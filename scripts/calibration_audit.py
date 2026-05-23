"""Per-pixel calibration / uncertainty audit.

For each Australia event-sensor pair this loads the cached DeepLabv3+ sigmoid
raster and the dNBR label, computes ECE and Brier score over the QA-valid
pixels, fits a single-parameter temperature scaling on a held-out calibration
event, and re-measures ECE after scaling -- the standard Guo et al. (2017)
calibration check, applied pixel-pooled. Writes per-event metrics to
data/paper_assets/tables/calibration_audit.csv and a pre/post reliability
diagram to data/paper_assets/figures/reliability_diagram.png.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from scipy.optimize import minimize_scalar

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MANIFEST = PROJECT_ROOT / "data" / "manifests" / "australia_external_test_manifest_noqa_full_harmonized.csv"
STITCHED_ROOT = PROJECT_ROOT / "data" / "evaluation" / "australia_full_seed42_reflectance64_stitched" / "deeplab"
OUT_CSV = PROJECT_ROOT / "data" / "paper_assets" / "tables" / "calibration_audit.csv"
OUT_PNG = PROJECT_ROOT / "data" / "paper_assets" / "figures" / "reliability_diagram.png"

CALIB_EVENT = "kangaroo_island_fire_2020"
N_BINS = 15
EPS = 1e-7


def _as_path(v: object) -> Path | None:
    text = str(v).strip()
    if not text or text.lower() == "nan":
        return None
    p = Path(text)
    return p if p.is_absolute() else PROJECT_ROOT / p


def _qa_valid(row: pd.Series, shape: tuple[int, int]) -> np.ndarray:
    valid = np.ones(shape, dtype=bool)
    for col in ("pre_clear_mask_harmonized", "post_clear_mask_harmonized"):
        p = _as_path(row.get(col, ""))
        if p is None or not p.exists():
            continue
        with rasterio.open(p) as src:
            arr = src.read(1)
        if arr.shape != shape:
            continue
        valid &= arr > 0
    return valid


def _temperature_scale(prob: np.ndarray, t: float) -> np.ndarray:
    """Re-temper a sigmoid probability via logit/T -> sigmoid."""
    prob = np.clip(prob, EPS, 1.0 - EPS)
    logit = np.log(prob / (1.0 - prob))
    return 1.0 / (1.0 + np.exp(-logit / t))


def _ece(prob: np.ndarray, label: np.ndarray, n_bins: int = N_BINS) -> float:
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = prob.size
    if n == 0:
        return float("nan")
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (prob >= lo) & (prob < hi) if i < n_bins - 1 else (prob >= lo) & (prob <= hi)
        k = int(mask.sum())
        if k == 0:
            continue
        conf = float(prob[mask].mean())
        acc = float(label[mask].mean())
        ece += abs(acc - conf) * k / n
    return float(ece)


def _brier(prob: np.ndarray, label: np.ndarray) -> float:
    return float(np.mean((prob - label) ** 2))


def _nll(prob: np.ndarray, label: np.ndarray) -> float:
    p = np.clip(prob, EPS, 1.0 - EPS)
    return float(-(label * np.log(p) + (1 - label) * np.log(1 - p)).mean())


def _fit_temperature(prob: np.ndarray, label: np.ndarray) -> float:
    """Find scalar T that minimises NLL after temperature scaling."""
    def objective(t: float) -> float:
        if t <= 0:
            return 1e9
        return _nll(_temperature_scale(prob, t), label)
    res = minimize_scalar(objective, bounds=(0.1, 10.0), method="bounded")
    return float(res.x)


def main() -> None:
    if not STITCHED_ROOT.exists():
        raise SystemExit(f"Missing {STITCHED_ROOT}")
    manifest = pd.read_csv(MANIFEST)

    per_event_records: list[dict] = []
    # Pool calibration-event pixels for fitting the temperature
    calib_probs: list[np.ndarray] = []
    calib_labels: list[np.ndarray] = []
    # Pool eval pixels for the overall reliability diagram
    eval_probs_pre: list[np.ndarray] = []
    eval_labels: list[np.ndarray] = []

    pair_records: list[tuple[np.ndarray, np.ndarray, dict]] = []
    for _, mrow in manifest.iterrows():
        pair_id = str(mrow["pair_id"])
        prob_path = STITCHED_ROOT / pair_id / "probability.tif"
        label_path = _as_path(mrow.get("label_mask_harmonized", ""))
        if not prob_path.exists() or label_path is None or not label_path.exists():
            continue
        with rasterio.open(prob_path) as ps:
            prob = ps.read(1).astype(np.float32)
        with rasterio.open(label_path) as ls:
            label = (ls.read(1) > 0).astype(np.float32)
        if prob.shape != label.shape:
            continue
        valid = _qa_valid(mrow, prob.shape)
        if valid.sum() == 0:
            continue
        p = prob[valid].astype(np.float32)
        y = label[valid].astype(np.float32)
        pair_records.append((p, y, {"event_id": str(mrow["event_id"]),
                                     "sensor": str(mrow["sensor"]),
                                     "pair_id": pair_id}))
        if str(mrow["event_id"]) == CALIB_EVENT:
            calib_probs.append(p); calib_labels.append(y)
        else:
            eval_probs_pre.append(p); eval_labels.append(y)


    by_event: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
    for p, y, meta in pair_records:
        by_event.setdefault(meta["event_id"], []).append((p, y))

    loeo_rows: list[dict] = []
    for held_out, pairs_out in by_event.items():
        held_p = np.concatenate([p for p, _ in pairs_out])
        held_y = np.concatenate([y for _, y in pairs_out])
        train_p = np.concatenate([p for ev, ps in by_event.items() if ev != held_out for p, _ in ps])
        train_y = np.concatenate([y for ev, ys in by_event.items() if ev != held_out for _, y in ys])
        T_fit = _fit_temperature(train_p, train_y)
        held_p_post = _temperature_scale(held_p, T_fit)
        loeo_rows.append({
            "held_out_event": held_out,
            "T_fitted": round(T_fit, 4),
            "n_held_pixels": int(held_p.size),
            "ece_pre":  round(_ece(held_p, held_y), 4),
            "ece_post": round(_ece(held_p_post, held_y), 4),
            "brier_pre":  round(_brier(held_p, held_y), 4),
            "brier_post": round(_brier(held_p_post, held_y), 4),
        })
    loeo_df = pd.DataFrame(loeo_rows).sort_values("held_out_event")
    LOEO_CSV = PROJECT_ROOT / "data" / "paper_assets" / "tables" / "calibration_audit_loeo.csv"
    loeo_df.to_csv(LOEO_CSV, index=False)
    print()
    print("=== LOEO calibration audit ===")
    print(loeo_df.to_string(index=False))
    print(f"\nLOEO mean ECE   pre = {loeo_df['ece_pre'].mean():.4f}   post = {loeo_df['ece_post'].mean():.4f}")
    print(f"LOEO mean Brier pre = {loeo_df['brier_pre'].mean():.4f}   post = {loeo_df['brier_post'].mean():.4f}")
    print(f"LOEO median T-fit   = {loeo_df['T_fitted'].median():.4f}")
    print(f"Wrote {LOEO_CSV}")

    if not calib_probs:
        raise SystemExit(f"Calibration event '{CALIB_EVENT}' has no pairs with cached probability+label rasters")

    calib_p = np.concatenate(calib_probs); calib_y = np.concatenate(calib_labels)
    eval_p_pre = np.concatenate(eval_probs_pre); eval_y = np.concatenate(eval_labels)

    print(f"\nCalibration event ({CALIB_EVENT}): {calib_p.size:,} valid pixels")
    print(f"Evaluation events (7 non-calibration): {eval_p_pre.size:,} valid pixels")

    # Fit temperature on calibration event, evaluate on the rest (legacy single-fit reference)
    T = _fit_temperature(calib_p, calib_y)
    eval_p_post = _temperature_scale(eval_p_pre, T)

    pre_ece = _ece(eval_p_pre, eval_y)
    post_ece = _ece(eval_p_post, eval_y)
    pre_brier = _brier(eval_p_pre, eval_y)
    post_brier = _brier(eval_p_post, eval_y)
    pre_nll = _nll(eval_p_pre, eval_y)
    post_nll = _nll(eval_p_post, eval_y)

    print(f"\nFitted temperature T = {T:.4f}")
    print(f"  ECE   pre  = {pre_ece:.4f}   post = {post_ece:.4f}")
    print(f"  Brier pre  = {pre_brier:.4f}   post = {post_brier:.4f}")
    print(f"  NLL   pre  = {pre_nll:.4f}   post = {post_nll:.4f}")

    # Per-pair table
    for p, y, meta in pair_records:
        post_p = _temperature_scale(p, T)
        per_event_records.append({
            **meta,
            "n_pixels": int(p.size),
            "burn_fraction": round(float(y.mean()), 4),
            "ece_pre": round(_ece(p, y), 4),
            "ece_post": round(_ece(post_p, y), 4),
            "brier_pre": round(_brier(p, y), 4),
            "brier_post": round(_brier(post_p, y), 4),
            "temperature_T": round(T, 4),
        })
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(per_event_records).to_csv(OUT_CSV, index=False)
    print(f"\nWrote {OUT_CSV}")

    T_median = float(loeo_df["T_fitted"].median())
    all_p = np.concatenate([p for p, _ in [(p, y) for p, y, _ in pair_records]])
    all_y = np.concatenate([y for _, y, _ in pair_records])
    all_p_post = _temperature_scale(all_p, T_median)
    overall_ece_pre = _ece(all_p, all_y)
    overall_ece_post = _ece(all_p_post, all_y)
    fig, ax = plt.subplots(1, 1, figsize=(7, 6))
    bin_edges = np.linspace(0.0, 1.0, N_BINS + 1)
    for probs, lbl, color in ((all_p, "pre (raw sigmoid)", "#1f77b4"),
                              (all_p_post, f"post (median LOEO T = {T_median:.2f})", "#d62728")):
        bin_acc, bin_conf = [], []
        for i in range(N_BINS):
            lo, hi = bin_edges[i], bin_edges[i + 1]
            mask = (probs >= lo) & (probs < hi) if i < N_BINS - 1 else (probs >= lo) & (probs <= hi)
            if mask.sum() == 0:
                bin_acc.append(np.nan); bin_conf.append(np.nan); continue
            bin_acc.append(float(all_y[mask].mean()))
            bin_conf.append(float(probs[mask].mean()))
        ax.plot(bin_conf, bin_acc, marker="o", label=lbl, color=color, linewidth=1.8)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="perfect")
    ax.set_xlabel("Mean predicted probability (within bin)")
    ax.set_ylabel("Observed burned-pixel fraction (within bin)")
    ax.set_title(
        f"Reliability diagram (DeepLabv3+ seed-42, all 8 Australia events, n={all_p.size:,} pixels)\n"
        f"ECE: {overall_ece_pre:.4f} (raw) vs.\\ {overall_ece_post:.4f} (post; median LOEO T={T_median:.2f}). "
        f"LOEO mean ECE pre={loeo_df['ece_pre'].mean():.4f}, post={loeo_df['ece_post'].mean():.4f}.",
        fontsize=9,
    )
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.legend(loc="upper left")
    plt.tight_layout()
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    print(f"Wrote {OUT_PNG}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import numpy as np


def bootstrap_ci(
    values: np.ndarray,
    *,
    statistic: str = "mean",
    n_bootstrap: int = 2000,
    ci: float = 0.95,
    seed: int = 42,
    lower_floor: float | None = None,
) -> tuple[float, float, float]:
    """Percentile bootstrap confidence interval.

    Unlike the t-interval, this respects physical bounds on non-negative
    quantities (e.g., area MAE) via `lower_floor`.

    Returns (point_estimate, ci_low, ci_high).
    """
    clean = np.asarray(values, dtype=np.float64)
    clean = clean[np.isfinite(clean)]
    if clean.size == 0:
        return float("nan"), float("nan"), float("nan")

    rng = np.random.default_rng(seed)
    n = clean.size
    draws = rng.integers(low=0, high=n, size=(n_bootstrap, n))
    samples = clean[draws]

    if statistic == "mean":
        stats = samples.mean(axis=1)
        point = float(clean.mean())
    elif statistic == "median":
        stats = np.median(samples, axis=1)
        point = float(np.median(clean))
    else:
        raise ValueError(f"Unsupported statistic: {statistic}")

    alpha = (1.0 - ci) / 2.0
    low = float(np.quantile(stats, alpha))
    high = float(np.quantile(stats, 1.0 - alpha))
    if lower_floor is not None:
        low = max(lower_floor, low)
        high = max(lower_floor, high)
        point = max(lower_floor, point)
    return point, low, high

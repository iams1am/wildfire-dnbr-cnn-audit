from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _grid_block_id(row_start: int, col_start: int, block_size: int) -> str:
    return f"{row_start // block_size}_{col_start // block_size}"


def load_patch_index(patch_index_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(patch_index_csv)
    if "pair_id" not in df.columns:
        raise ValueError("patch_index missing 'pair_id'")
    return df


def event_level_folds(
    df: pd.DataFrame,
    n_splits: int,
    seed: int = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Fold assignment where every patch from one event is held out together.
    """
    key_col = "event_id" if "event_id" in df.columns and df["event_id"].astype(str).str.len().sum() > 0 else "pair_id"
    groups = df[key_col].astype(str).to_numpy()
    unique = np.array(sorted(set(groups)))
    if unique.size < n_splits:
        raise ValueError(
            f"Only {unique.size} unique {key_col} groups; cannot produce {n_splits}-fold event-level split."
        )
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    chunks = np.array_split(unique, n_splits)
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    all_idx = np.arange(len(df))
    for held_out in chunks:
        val_mask = np.isin(groups, held_out)
        folds.append((all_idx[~val_mask], all_idx[val_mask]))
    return folds


def spatial_block_folds(
    df: pd.DataFrame,
    n_splits: int,
    block_size: int = 1024,
    seed: int = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Fold assignment where patches are grouped by spatial block inside their pair.

    Prevents adjacent patches of the same scene ending up in both train and val.
    Introduces a dataset leakage boundary by dropping patches that straddle the
    block boundary from both train and validation folds.
    """
    required = {"pair_id", "row_start", "col_start", "patch_size"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"patch_index missing columns {sorted(missing)} for spatial-block split")

    block_keys = np.array(
        [
            f"{pair}_{_grid_block_id(int(r), int(c), block_size)}"
            for pair, r, c in zip(df["pair_id"].astype(str), df["row_start"], df["col_start"])
        ]
    )


    is_buffer = []
    for r, c, p_size in zip(df["row_start"], df["col_start"], df["patch_size"]):
        r_mod = r % block_size
        c_mod = c % block_size
        if (r_mod + p_size > block_size) or (c_mod + p_size > block_size):
            is_buffer.append(True)
        else:
            is_buffer.append(False)
    is_buffer_np = np.array(is_buffer)

    unique = np.array(sorted(set(block_keys)))
    if unique.size < n_splits:
        raise ValueError(
            f"Only {unique.size} spatial blocks; increase data or decrease n_splits."
        )
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    chunks = np.array_split(unique, n_splits)
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    all_idx = np.arange(len(df))
    for held_out in chunks:
        heldout_mask = np.isin(block_keys, held_out)
        val_mask = heldout_mask & (~is_buffer_np)
        train_mask = (~heldout_mask) & (~is_buffer_np)
        if not train_mask.any() or not val_mask.any():
            raise ValueError(
                "Spatial split produced an empty train or validation fold after dropping boundary patches; "
                "try a larger block_size, fewer folds, or a less aggressive patch overlap."
            )
        folds.append((all_idx[train_mask], all_idx[val_mask]))
    return folds


def build_folds(
    df: pd.DataFrame,
    *,
    strategy: str,
    n_splits: int,
    block_size: int = 1024,
    seed: int = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
    key = strategy.strip().lower()
    if key == "event":
        return event_level_folds(df, n_splits=n_splits, seed=seed)
    if key == "spatial":
        return spatial_block_folds(df, n_splits=n_splits, block_size=block_size, seed=seed)
    if key == "random":
        rng = np.random.default_rng(seed)
        idx = np.arange(len(df))
        rng.shuffle(idx)
        chunks = np.array_split(idx, n_splits)
        return [(np.setdiff1d(np.arange(len(df)), val), val) for val in chunks]
    raise ValueError(f"Unknown split strategy '{strategy}'. Use event|spatial|random.")

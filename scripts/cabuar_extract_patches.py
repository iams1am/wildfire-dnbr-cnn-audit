from __future__ import annotations

import argparse
import sys
from pathlib import Path

import hdf5plugin  # noqa: F401  registers Blosc/Zstd filters used by the CaBuAr HDF5 shards
import h5py
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# CaBuAr stores 12-band Sentinel-2 in B01..B12 order (axis -1).
# Our pipeline expects [blue, green, red, nir, swir22] = [B02, B03, B04, B08, B12].
CABUAR_BAND_INDICES = [1, 2, 3, 7, 11]


def _meanpool_3x3(arr: np.ndarray) -> np.ndarray:
    """Mean-pool 10 m raster down to 30 m (factor 3) to match training resolution.

    Accepts (H, W) or (C, H, W). Crops to a multiple of 3 first.
    """
    if arr.ndim == 2:
        h, w = arr.shape
        h3, w3 = (h // 3) * 3, (w // 3) * 3
        a = arr[:h3, :w3].astype(np.float32)
        return a.reshape(h3 // 3, 3, w3 // 3, 3).mean(axis=(1, 3))
    if arr.ndim == 3:
        c, h, w = arr.shape
        h3, w3 = (h // 3) * 3, (w // 3) * 3
        a = arr[:, :h3, :w3].astype(np.float32)
        return a.reshape(c, h3 // 3, 3, w3 // 3, 3).mean(axis=(2, 4))
    raise ValueError(f"Unsupported ndim {arr.ndim}")


def _extract_patches_from_tile(
    pre: np.ndarray,
    post: np.ndarray,
    mask: np.ndarray,
    *,
    patch_size: int,
    stride: int,
    min_burned_frac: float,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, int, int]]:
    """Yield strided patches that meet a minimum burn fraction. Pads edges automatically."""
    c, h, w = pre.shape
    pad_h = (patch_size - h % patch_size) % patch_size
    pad_w = (patch_size - w % patch_size) % patch_size

    if pad_h > 0 or pad_w > 0:
        pre = np.pad(pre, ((0, 0), (0, pad_h), (0, pad_w)), mode="constant", constant_values=0)
        post = np.pad(post, ((0, 0), (0, pad_h), (0, pad_w)), mode="constant", constant_values=0)
        mask = np.pad(mask, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=0)
        h, w = mask.shape

    patches: list[tuple[np.ndarray, np.ndarray, np.ndarray, int, int]] = []
    for r in range(0, h - patch_size + 1, stride):
        for c in range(0, w - patch_size + 1, stride):
            m = mask[r:r + patch_size, c:c + patch_size]
            burn_frac = float((m > 0).mean())
            if burn_frac < min_burned_frac:
                continue
            patches.append((pre[:, r:r + patch_size, c:c + patch_size],
                            post[:, r:r + patch_size, c:c + patch_size],
                            m, r, c))
    return patches


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract CaBuAr Sentinel-2 tiles into our 5-band 256-px patch format.")
    parser.add_argument("--shards", nargs="+", type=Path, required=True, help="Paths to downloaded CaBuAr HDF5 shards.")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "data" / "patches_cabuar")
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument("--min-burned-frac", type=float, default=0.005,
                        help="Per-patch minimum burn fraction. CaBuAr tiles average <0.5% burn.")
    parser.add_argument("--max-patches-per-tile", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-meanpool", action="store_true",
                        help="Disable 3x3 mean pooling. By default we downsample 10m -> 30m to match training resolution.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    rows: list[dict[str, object]] = []

    for shard_path in args.shards:
        print(f"Reading shard {shard_path.name}")
        with h5py.File(shard_path, "r") as f:
            for fold_key in f.keys():
                fold_grp = f[fold_key]
                for uuid in fold_grp:
                    grp = fold_grp[uuid]
                    keys = set(grp.keys())
                    if not {"pre_fire", "post_fire", "mask"}.issubset(keys):
                        continue
                    try:
                        pre_full = grp["pre_fire"][:]
                        post_full = grp["post_fire"][:]
                        mask_full = grp["mask"][:]
                    except OSError as exc:
                        print(f"    skipping {uuid[:8]} due to read error: {exc}")
                        continue

                    # CaBuAr "normalized/complete" stores uint8; rescale to ~[0,1] reflectance.
                    pre = np.transpose(pre_full[..., CABUAR_BAND_INDICES], (2, 0, 1)).astype(np.float32) / 255.0
                    post = np.transpose(post_full[..., CABUAR_BAND_INDICES], (2, 0, 1)).astype(np.float32) / 255.0
                    mask_full = (mask_full > 0).astype(np.float32)

                    if not args.no_meanpool:
                        # 3x3 mean-pool: 10 m -> 30 m to match California training resolution.
                        pre = _meanpool_3x3(pre)
                        post = _meanpool_3x3(post)
                        mask_full = (_meanpool_3x3(mask_full) >= 0.5).astype(np.uint8)
                    else:
                        mask_full = mask_full.astype(np.uint8)

                    patches = _extract_patches_from_tile(
                        pre, post, mask_full,
                        patch_size=args.patch_size,
                        stride=args.stride,
                        min_burned_frac=args.min_burned_frac,
                    )
                    if not patches:
                        continue
                    if len(patches) > args.max_patches_per_tile:
                        idx = rng.choice(len(patches), size=args.max_patches_per_tile, replace=False)
                        patches = [patches[i] for i in idx]

                    out_dir = args.output_root / f"shard{shard_path.stem}_uuid{uuid[:8]}"
                    out_dir.mkdir(parents=True, exist_ok=True)
                    for k, (p_pre, p_post, p_mask, r, c) in enumerate(patches):
                        out_path = out_dir / f"patch_r{r}_c{c}.npz"
                        np.savez_compressed(out_path, pre=p_pre.astype(np.float32),
                                            post=p_post.astype(np.float32), label=p_mask.astype(np.uint8))
                        rows.append({
                            "pair_id": uuid,
                            "event_id": f"cabuar_{uuid[:8]}",
                            "sensor": "sentinel2",
                            "region": "California",
                            "split": "train_val",
                            "patch_path": str(out_path),
                            "row_start": r,
                            "col_start": c,
                            "patch_size": args.patch_size,
                            "valid_fraction": 1.0,
                            "burned_fraction": float((p_mask > 0).mean()),
                        })

    df = pd.DataFrame(rows)
    out_csv = args.output_root / "patch_index.csv"
    df.to_csv(out_csv, index=False)
    n_unique = df["pair_id"].nunique() if not df.empty else 0
    print(f"Wrote {len(df)} patches across {n_unique} unique tiles -> {out_csv}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from src.data.sensor_normalization import normalize_patch
from src.training.augmentations import AugmentationConfig, apply_augmentations
from src.training.spatial_split import build_folds


class PatchPairDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        patch_index_csv: Path,
        *,
        augment: bool = False,
        aug_seed: int = 0,
        aug_cfg: AugmentationConfig | None = None,
        normalize: bool = False,
    ) -> None:
        self.patch_index_csv = patch_index_csv
        self.patch_index = pd.read_csv(patch_index_csv)
        if "patch_path" not in self.patch_index.columns:
            raise ValueError(f"patch_path column missing in: {patch_index_csv}")
        self.patch_paths = [Path(str(path)) for path in self.patch_index["patch_path"].tolist()]
        if not self.patch_paths:
            raise ValueError(f"No patch rows found in patch index: {patch_index_csv}")
        self.sensors = (
            [str(s).strip().lower() for s in self.patch_index["sensor"].tolist()]
            if "sensor" in self.patch_index.columns
            else [""] * len(self.patch_paths)
        )
        self.augment = augment
        self.aug_cfg = aug_cfg if aug_cfg is not None else AugmentationConfig(enabled=augment)
        self.aug_cfg.enabled = augment
        self._aug_seed = aug_seed
        self.normalize = normalize

    def __len__(self) -> int:
        return len(self.patch_paths)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        with np.load(self.patch_paths[index]) as patch:
            pre = patch["pre"].astype(np.float32)
            post = patch["post"].astype(np.float32)
            label = patch["label"].astype(np.float32)
        if label.ndim == 2:
            label = np.expand_dims(label, axis=0)

        if self.normalize:
            sensor = self.sensors[index] if index < len(self.sensors) else ""
            pre = normalize_patch(pre, sensor)
            post = normalize_patch(post, sensor)

        if self.augment and self.aug_cfg.enabled:
            rng = np.random.default_rng(self._aug_seed + index * 9973 + int(torch.randint(0, 2_000_000_000, (1,)).item()))
            pre, post, label = apply_augmentations(pre, post, label, rng=rng, cfg=self.aug_cfg)

        pre = np.nan_to_num(pre, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
        post = np.nan_to_num(post, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
        label = np.nan_to_num(label, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)

        return {
            "pre": torch.from_numpy(pre),
            "post": torch.from_numpy(post),
            "label": torch.from_numpy(label),
        }

    @property
    def channels_per_image(self) -> int:
        with np.load(self.patch_paths[0]) as sample:
            pre = sample["pre"]
            if pre.ndim != 3:
                raise ValueError(f"Expected pre tensor of shape [C,H,W], got {pre.shape}")
            channels = int(pre.shape[0])
        return channels


def _seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def create_train_val_loaders(
    patch_index_csv: Path,
    *,
    batch_size: int,
    val_fraction: float = 0.2,
    seed: int = 42,
    augment_train: bool = False,
    normalize: bool = True,
    split_strategy: str = "spatial",
    block_size: int = 1024,
    num_workers: int = 0,
    pin_memory: bool | None = None,
) -> tuple[DataLoader, DataLoader]:
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1.")

    train_dataset = PatchPairDataset(patch_index_csv, augment=augment_train, aug_seed=seed, normalize=normalize)
    val_dataset = PatchPairDataset(patch_index_csv, augment=False, normalize=normalize)
    num_samples = len(train_dataset)
    if num_samples < 2:
        raise ValueError("Need at least 2 patches to create train/val loaders.")

    strategy = split_strategy.strip().lower()
    if strategy == "random":
        rng = np.random.default_rng(seed)
        indices = np.arange(num_samples)
        rng.shuffle(indices)

        val_size = max(1, int(round(num_samples * val_fraction)))
        val_indices = indices[:val_size]
        train_indices = indices[val_size:]
        if train_indices.size == 0:
            train_indices = val_indices[:1]
            val_indices = indices[1:]
    else:
        n_splits = max(2, int(round(1.0 / val_fraction)))
        folds = build_folds(
            train_dataset.patch_index,
            strategy=strategy,
            n_splits=n_splits,
            block_size=block_size,
            seed=seed,
        )
        train_indices, val_indices = folds[0]

    train_subset = Subset(train_dataset, train_indices.tolist())
    val_subset = Subset(val_dataset, val_indices.tolist())

    loader_generator = torch.Generator()
    loader_generator.manual_seed(seed)
    use_pin_memory = torch.cuda.is_available() if pin_memory is None else pin_memory
    loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": use_pin_memory,
        "worker_init_fn": _seed_worker if num_workers > 0 else None,
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = 2

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        generator=loader_generator,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        **loader_kwargs,
    )
    return train_loader, val_loader

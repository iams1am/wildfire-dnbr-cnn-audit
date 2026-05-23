from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class AugmentationConfig:
    """Augmentation policy for paired pre/post Earth-observation patches.

    All transforms are applied identically to (pre, post, label) so spatial
    correspondence is preserved. Photometric augmentations are applied
    identically to pre and post so that the dNBR change signal is preserved.
    """

    p_hflip: float = 0.5
    p_vflip: float = 0.5
    p_rot90: float = 0.75   # 0/90/180/270 chosen uniformly when triggered
    p_brightness: float = 0.5
    brightness_delta: float = 0.05  # additive in normalized reflectance space
    p_contrast: float = 0.5
    contrast_range: tuple[float, float] = (0.9, 1.1)
    p_gauss_noise: float = 0.3
    gauss_noise_std: float = 0.01

    enabled: bool = True


def apply_augmentations(
    pre: np.ndarray,
    post: np.ndarray,
    label: np.ndarray,
    *,
    rng: np.random.Generator,
    cfg: AugmentationConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply paired augmentations consistently to pre, post, and label.

    Inputs are float32 arrays of shape (C, H, W) for pre/post and (1, H, W) or
    (H, W) for label. The label is left as a binary integer mask.
    """
    if not cfg.enabled:
        return pre, post, label

    if rng.random() < cfg.p_hflip:
        pre = pre[:, :, ::-1].copy()
        post = post[:, :, ::-1].copy()
        if label.ndim == 3:
            label = label[:, :, ::-1].copy()
        else:
            label = label[:, ::-1].copy()

    if rng.random() < cfg.p_vflip:
        pre = pre[:, ::-1, :].copy()
        post = post[:, ::-1, :].copy()
        if label.ndim == 3:
            label = label[:, ::-1, :].copy()
        else:
            label = label[::-1, :].copy()

    if rng.random() < cfg.p_rot90:
        k = int(rng.integers(1, 4))
        pre = np.rot90(pre, k=k, axes=(1, 2)).copy()
        post = np.rot90(post, k=k, axes=(1, 2)).copy()
        if label.ndim == 3:
            label = np.rot90(label, k=k, axes=(1, 2)).copy()
        else:
            label = np.rot90(label, k=k, axes=(0, 1)).copy()

    if rng.random() < cfg.p_brightness:
        delta = float(rng.uniform(-cfg.brightness_delta, cfg.brightness_delta))
        pre = pre + delta
        post = post + delta

    if rng.random() < cfg.p_contrast:
        gamma = float(rng.uniform(*cfg.contrast_range))
        pre = pre * gamma
        post = post * gamma

    if rng.random() < cfg.p_gauss_noise:
        noise_pre = rng.normal(0.0, cfg.gauss_noise_std, size=pre.shape).astype(np.float32)
        noise_post = rng.normal(0.0, cfg.gauss_noise_std, size=post.shape).astype(np.float32)
        pre = pre + noise_pre
        post = post + noise_post

    return pre.astype(np.float32), post.astype(np.float32), label

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.optim import AdamW
from tqdm import tqdm

from src.training.losses import batch_positive_weight, bce_dice_loss


def _batch_counts(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> tuple[float, float, float]:
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()
    targets = (targets >= 0.5).float()
    tp = float((preds * targets).sum().item())
    fp = float((preds * (1.0 - targets)).sum().item())
    fn = float(((1.0 - preds) * targets).sum().item())
    return tp, fp, fn


def _batch_patch_scores(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    smooth: float = 1e-6,
) -> tuple[float, float]:
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()
    targets = (targets >= 0.5).float()
    tp = (preds * targets).sum(dim=(1, 2, 3))
    fp = (preds * (1.0 - targets)).sum(dim=(1, 2, 3))
    fn = ((1.0 - preds) * targets).sum(dim=(1, 2, 3))
    iou = (tp + smooth) / (tp + fp + fn + smooth)
    f1 = (2.0 * tp + smooth) / (2.0 * tp + fp + fn + smooth)
    return float(iou.mean().item()), float(f1.mean().item())


def _pixel_scores(tp: float, fp: float, fn: float) -> tuple[float, float]:
    union = tp + fp + fn
    iou = tp / union if union > 0.0 else float("nan")
    denom = (2.0 * tp) + fp + fn
    f1 = (2.0 * tp) / denom if denom > 0.0 else float("nan")
    return iou, f1


def _run_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    *,
    device: torch.device,
    optimizer: AdamW | None,
    scaler: torch.amp.GradScaler | None = None,
    amp: bool = True,
    dynamic_pos_weight: bool = True,
    pos_weight_max: float = 20.0,
) -> dict[str, float]:
    is_train = optimizer is not None
    if is_train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_patch_iou = 0.0
    total_patch_f1 = 0.0
    total_samples = 0
    tp_total = 0.0
    fp_total = 0.0
    fn_total = 0.0

    for batch in tqdm(loader, leave=False):
        pre = batch["pre"].to(device)
        post = batch["post"].to(device)
        label = batch["label"].to(device)
        batch_size = int(pre.shape[0])
        pos_weight = batch_positive_weight(label, max_weight=pos_weight_max) if dynamic_pos_weight else None

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
                logits = model(pre, post)
            loss = bce_dice_loss(logits.float(), label.float(), pos_weight=pos_weight)
            if scaler is not None and scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
        else:
            with torch.no_grad():
                with torch.amp.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
                    logits = model(pre, post)
                loss = bce_dice_loss(logits.float(), label.float(), pos_weight=pos_weight)

        detached_logits = logits.detach().float()
        detached_label = label.detach()
        patch_iou, patch_f1 = _batch_patch_scores(detached_logits, detached_label)
        tp, fp, fn = _batch_counts(detached_logits, detached_label)

        total_loss += float(loss.item()) * batch_size
        total_patch_iou += patch_iou * batch_size
        total_patch_f1 += patch_f1 * batch_size
        total_samples += batch_size
        tp_total += tp
        fp_total += fp
        fn_total += fn

    if total_samples == 0:
        raise ValueError("Empty data loader received in training loop.")

    pixel_iou, pixel_f1 = _pixel_scores(tp_total, fp_total, fn_total)

    return {
        "loss": total_loss / total_samples,
        "iou": pixel_iou,
        "f1": pixel_f1,
        "iou_patch_mean": total_patch_iou / total_samples,
        "f1_patch_mean": total_patch_f1 / total_samples,
        "tp_total": tp_total,
        "fp_total": fp_total,
        "fn_total": fn_total,
    }


def train_segmentation_model(
    *,
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    output_dir: Path,
    epochs: int = 20,
    lr: float = 1e-3,
    device: str = "cuda",
    dynamic_pos_weight: bool = True,
    pos_weight_max: float = 20.0,
    amp: bool = True,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    device_obj = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    model = model.to(device_obj)
    optimizer = AdamW(model.parameters(), lr=lr)
    amp_enabled = bool(amp and device_obj.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    best_val_iou = -1.0
    history: list[dict[str, float | int]] = []

    for epoch in range(1, epochs + 1):
        train_metrics = _run_epoch(
            model,
            train_loader,
            device=device_obj,
            optimizer=optimizer,
            scaler=scaler,
            amp=amp_enabled,
            dynamic_pos_weight=dynamic_pos_weight,
            pos_weight_max=pos_weight_max,
        )
        val_metrics = _run_epoch(
            model,
            val_loader,
            device=device_obj,
            optimizer=None,
            scaler=None,
            amp=amp_enabled,
            dynamic_pos_weight=dynamic_pos_weight,
            pos_weight_max=pos_weight_max,
        )

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_iou": train_metrics["iou"],
            "train_f1": train_metrics["f1"],
            "train_iou_patch_mean": train_metrics["iou_patch_mean"],
            "train_f1_patch_mean": train_metrics["f1_patch_mean"],
            "val_loss": val_metrics["loss"],
            "val_iou": val_metrics["iou"],
            "val_f1": val_metrics["f1"],
            "val_iou_patch_mean": val_metrics["iou_patch_mean"],
            "val_f1_patch_mean": val_metrics["f1_patch_mean"],
            "val_tp_total": val_metrics["tp_total"],
            "val_fp_total": val_metrics["fp_total"],
            "val_fn_total": val_metrics["fn_total"],
        }
        history.append(row)
        print(
            "Epoch {epoch}/{epochs} | "
            "train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            "train_iou={train_iou:.4f} val_iou={val_iou:.4f}".format(
                epoch=epoch,
                epochs=epochs,
                train_loss=train_metrics["loss"],
                val_loss=val_metrics["loss"],
                train_iou=train_metrics["iou"],
                val_iou=val_metrics["iou"],
            ),
            flush=True,
        )

        if val_metrics["iou"] > best_val_iou:
            best_val_iou = val_metrics["iou"]
            checkpoint_path = output_dir / "best_model.pt"
            torch.save(model.state_dict(), checkpoint_path)

    history_path = output_dir / "history.csv"
    pd.DataFrame(history).to_csv(history_path, index=False)
    return history_path

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
EfficientNet-V2-S two-stage training (head-only then unfreeze blocks) with AMP and imbalance options.

- ImageFolder dataset; stratified 80/20 split.
- Options: class weights (CE), weighted sampler, or focal loss (choose one).
- OneCycleLR with AdamW; label smoothing.
- Early stopping by val loss, per-epoch checkpoints, best_model.pth, stats and plots.
- Optional SWA finishing.

How to run (no CLI needed):
- Edit the USER CONFIG section below (DATA_DIR, CKPT_DIR, learning rates, etc.), then run the file:
        python \
            "e:/Santosh_master_thesis/Understanding_citizenscience_species_segmentation/script/Second part of training/efficientnet_20_classes_train_in_two_stages.py"
"""

import os
import re
import json
import copy
import logging
import shutil
from collections import Counter
from typing import List, Optional

import numpy as np
import matplotlib.pyplot as plt
from PIL import ImageFile

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn

from torchvision import datasets, transforms, models
from torchvision.transforms import RandomPerspective, RandomAffine, RandomGrayscale
from torchvision.models import EfficientNet_V2_S_Weights

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score, balanced_accuracy_score
)
from tqdm.auto import tqdm

# ----------------------------
# USER CONFIG (edit these values)
# ----------------------------
# Paths
# ImageFolder root (folders per class)
DATA_DIR = r"E:/Santosh_master_thesis/classified_output_new"
# where to save checkpoints
CKPT_DIR = r"E:/Santosh_master_thesis/Checkpoints_efficientnet_weightedrandomsampler_two_stages"

# Training params
BATCH_SIZE = 12
IMAGE_SIZE = 640
EPOCHS = 35
HEAD_EPOCHS = 4
UNFREEZE_BLOCKS = 4
PATIENCE = 7
NUM_WORKERS = 4
SEED = 42

# Optimizer / schedulers
HEAD_LR = 5e-4
BACKBONE_LR = 7e-5
WEIGHT_DECAY = 5e-4
LABEL_SMOOTH = 0.02

# Imbalance handling: one of {"none", "class_weights", "sampler", "focal"}
IMBALANCE = "focal"
FOCAL_GAMMA = 1.5

# SWA finishing
USE_SWA = True
SWA_EPOCHS = 5
SWA_LR = 5e-5

# Misc
RENAME_EPOCH_FOLDER = False

# Confusion matrix figure and font settings
CM_FIGSIZE = (18, 14)          # width, height in inches
CM_TICK_FONTSIZE = 12        # tick labels (class names)
CM_ANNOT_FONTSIZE = 8        # numbers inside the cells
CM_TITLE_FONTSIZE = 12       # title font size
CM_AXIS_LABEL_FONTSIZE = 11  # axis label font size

# ----------------------------
# Globals
# ----------------------------
ImageFile.LOAD_TRUNCATED_IMAGES = True
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("train_two_stage")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Will be set in main()
train_tf = None
val_tf_center = None

# ----------------------------
# Helpers
# ----------------------------


def build_transforms(image_size: int):
    global train_tf, val_tf_center
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(0.4, 0.4, 0.4),
        transforms.RandomApply([transforms.GaussianBlur(3)], p=0.2),
        RandomPerspective(distortion_scale=0.2, p=0.2),
        RandomAffine(25, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
        RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.5),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    val_tf_center = transforms.Compose([
        transforms.Resize(int(image_size * 1.1)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def set_seeds(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def freeze_backbone(model: nn.Module):
    for p in model.features.parameters():
        p.requires_grad = False


def unfreeze_last_blocks(model: nn.Module, k: int = 2):
    if k <= 0:
        return
    for m in model.features[-k:]:
        for p in m.parameters():
            p.requires_grad = True


def head_parameters(model: nn.Module):
    return [p for p in model.classifier.parameters() if p.requires_grad]


def backbone_trainable_parameters(model: nn.Module):
    return [p for n, p in model.named_parameters() if "features" in n and p.requires_grad]


def plot_confusion_matrix_png(
    cm: np.ndarray,
    labels,
    out_path: str,
    normalize: bool = False,
    title: Optional[str] = None,
    figsize: tuple = (8, 6),
    tick_fontsize: int = 10,
    annot_fontsize: int = 8,
    title_fontsize: int = 12,
    axis_label_fontsize: int = 11,
):
    plt.figure(figsize=figsize)
    matrix = cm.astype(float)
    if normalize:
        with np.errstate(all='ignore'):
            row_sums = matrix.sum(axis=1, keepdims=True)
            matrix = np.divide(matrix, row_sums, out=np.zeros_like(
                matrix), where=row_sums != 0)
    im = plt.imshow(matrix, interpolation='nearest', cmap=plt.cm.Blues)
    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    try:
        cbar.ax.tick_params(labelsize=tick_fontsize)
    except Exception:
        pass
    ticks = np.arange(len(labels))
    plt.xticks(ticks, labels, rotation=45, ha='right', fontsize=tick_fontsize)
    plt.yticks(ticks, labels, fontsize=tick_fontsize)
    fmt = ".2f" if normalize else ".0f"
    thresh = (matrix.max() if matrix.size else 0) / 2.0 if matrix.size else 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            plt.text(
                j,
                i,
                format(val, fmt),
                ha="center",
                va="center",
                color="white" if val > thresh else "black",
                fontsize=annot_fontsize,
            )
    plt.ylabel('True', fontsize=axis_label_fontsize)
    plt.xlabel('Predicted', fontsize=axis_label_fontsize)
    if title:
        plt.title(title, fontsize=title_fontsize)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def smooth_curve(points, factor=0.8):
    if not points:
        return points
    out, last = [], points[0]
    for p in points:
        last = last * factor + (1 - factor) * p
        out.append(last)
    return out


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction="mean", label_smoothing=0.0):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha  # tensor[K] or None
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(self, logits, target):
        num_classes = logits.size(1)
        # label smoothing via soft targets
        with torch.no_grad():
            true_dist = torch.zeros_like(logits).fill_(
                self.label_smoothing / (num_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(
                1), 1.0 - self.label_smoothing)
        log_probs = F.log_softmax(logits, dim=1)
        probs = log_probs.exp()
        ce = -(true_dist * log_probs).sum(dim=1)  # smoothed CE
        pt = (true_dist * probs).sum(dim=1).clamp(min=1e-8)
        loss = ((1 - pt) ** self.gamma) * ce
        if self.alpha is not None:
            at = self.alpha[target]  # per-sample alpha
            loss = at * loss
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


def get_data_loaders(data_dir: str, batch: int, num_workers: int, use_weighted_sampler: bool, seed: int):
    base = datasets.ImageFolder(data_dir, transform=None)
    targets = [s[1] for s in base.samples]

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_idx, val_idx = next(sss.split(np.zeros(len(targets)), targets))

    train_counts = Counter([targets[i] for i in train_idx])
    val_counts = Counter([targets[i] for i in val_idx])
    logger.info(
        f"Split totals -> Train: {len(train_idx)} | Val: {len(val_idx)}")
    logger.info(f"Train class counts: {train_counts}")
    logger.info(f"Val class counts:   {val_counts}")

    train_set = datasets.ImageFolder(data_dir, transform=train_tf)
    val_set = datasets.ImageFolder(data_dir, transform=val_tf_center)
    train_set.samples = [base.samples[i] for i in train_idx]
    val_set.samples = [base.samples[i] for i in val_idx]

    sampler = None
    if use_weighted_sampler:
        class_sample_counts = np.array(
            [train_counts.get(i, 0) for i in range(len(base.classes))], dtype=np.float32)
        class_weights = 1.0 / np.clip(class_sample_counts, 1.0, None)
        sample_weights = [class_weights[label]
                          for _, label in train_set.samples]
        sampler = WeightedRandomSampler(
            sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_set, batch_size=batch, shuffle=(sampler is None),
                              sampler=sampler, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch,
                            shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, base, train_counts


def swa_finetune(model: nn.Module, criterion, train_loader, val_loader, out_dir: str,
                 swa_epochs: int, swa_lr: float, weight_decay: float):
    logger.info(
        f"Starting SWA finishing for {swa_epochs} epochs at lr={swa_lr}")
    optimizer = AdamW(model.parameters(), lr=swa_lr, weight_decay=weight_decay)
    swa_model = AveragedModel(model)
    swa_scheduler = SWALR(optimizer, swa_lr=swa_lr)
    scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())

    best_swa_loss = float('inf')

    for e in range(1, swa_epochs + 1):
        model.train()
        for inputs, labels in tqdm(train_loader, desc=f"SWA Train {e}/{swa_epochs}", leave=False, dynamic_ncols=True):
            inputs, labels = inputs.to(DEVICE, non_blocking=True), labels.to(
                DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            swa_model.update_parameters(model)
        swa_scheduler.step()

        # Evaluate SWA snapshot
        swa_model.eval()
        val_loss_sum, val_correct, val_total = 0.0, 0, 0
        y_true, y_pred = [], []
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc="SWA Val", leave=False, dynamic_ncols=True):
                inputs, labels = inputs.to(DEVICE, non_blocking=True), labels.to(
                    DEVICE, non_blocking=True)
                outputs = swa_model(inputs)
                loss = criterion(outputs, labels)
                preds = outputs.argmax(1)
                val_loss_sum += loss.item() * inputs.size(0)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                y_true.extend(labels.detach().cpu().tolist())
                y_pred.extend(preds.detach().cpu().tolist())
        val_loss = val_loss_sum / val_total if val_total else float('inf')
        val_acc = val_correct / val_total if val_total else 0.0
        f1_macro = f1_score(y_true, y_pred, average="macro")
        logger.info(
            f"SWA Epoch {e}: VL={val_loss:.4f} VA={val_acc:.4f} F1={f1_macro:.4f}")

        if val_loss < best_swa_loss - 1e-12:
            best_swa_loss = val_loss
            torch.save(swa_model.state_dict(), os.path.join(
                out_dir, f"best_model_swa_{e}_{val_loss:.2f}.pth"))
            torch.save(swa_model.state_dict(), os.path.join(
                out_dir, "best_model_swa.pth"))

    # Update BN
    try:
        update_bn(train_loader, swa_model, device=DEVICE)
    except TypeError:
        update_bn(train_loader, swa_model)

    # Final save
    torch.save(swa_model.state_dict(), os.path.join(
        out_dir, f"best_model_swa_final_{best_swa_loss:.2f}.pth"))
    return swa_model


def train_model(model: nn.Module, criterion, optimizer, scheduler, train_loader, val_loader,
                num_epochs: int, writer: SummaryWriter, root_out_dir: str, class_names: List[str], patience: int,
                best_naming_metric: str, rename_epoch_folder: bool, epoch_start: int = 0):
    scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())

    best_loss = float('inf')
    best_loss_ep = -1
    best_model_wts_loss = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0

    models_dir = os.path.join(root_out_dir, "All_Epoch_Models")
    stats_dir = os.path.join(root_out_dir, "Training_Stats")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(stats_dir, exist_ok=True)

    hist = {"epoch": [], "train_loss": [], "train_acc": [],
            "val_loss": [], "val_acc": [], "val_f1": []}

    for local_epoch in range(num_epochs):
        global_epoch = epoch_start + local_epoch + 1
        logger.info(f"Epoch {global_epoch}/{epoch_start + num_epochs}")
        model.train()
        train_loss_sum, correct, total = 0.0, 0, 0

        for inputs, labels in tqdm(train_loader, desc=f"Train {global_epoch}/{epoch_start + num_epochs}", leave=False, dynamic_ncols=True):
            inputs, labels = inputs.to(DEVICE, non_blocking=True), labels.to(
                DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            prev_scale = scaler.get_scale()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            # Only step the scheduler if optimizer actually stepped (no AMP overflow)
            if scaler.get_scale() >= prev_scale:
                scheduler.step()

            train_loss_sum += loss.item() * inputs.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total if total else 0.0
        train_loss = train_loss_sum / total if total else 0.0

        # Validation
        model.eval()
        val_loss_sum, val_correct, val_total = 0.0, 0, 0
        y_true, y_pred = [], []
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc="Val", leave=False, dynamic_ncols=True):
                inputs, labels = inputs.to(DEVICE, non_blocking=True), labels.to(
                    DEVICE, non_blocking=True)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                preds = outputs.argmax(1)

                val_loss_sum += loss.item() * inputs.size(0)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                y_true.extend(labels.detach().cpu().tolist())
                y_pred.extend(preds.detach().cpu().tolist())

        val_acc = val_correct / val_total if val_total else 0.0
        val_loss = val_loss_sum / val_total if val_total else 0.0
        f1_macro = f1_score(y_true, y_pred, average="macro")
        bal_acc = balanced_accuracy_score(y_true, y_pred)

        hist["epoch"].append(global_epoch)
        hist["train_loss"].append(train_loss)
        hist["train_acc"].append(train_acc)
        hist["val_loss"].append(val_loss)
        hist["val_acc"].append(val_acc)
        hist["val_f1"].append(f1_macro)

        writer.add_scalar("Loss/Train", train_loss, global_epoch)
        writer.add_scalar("Loss/Val",   val_loss,  global_epoch)
        writer.add_scalar("Acc/Train",  train_acc, global_epoch)
        writer.add_scalar("Acc/Val",    val_acc,   global_epoch)
        writer.add_scalar("F1_macro/Val", f1_macro, global_epoch)
        writer.add_scalar("Acc_balanced/Val", bal_acc, global_epoch)

        logger.info(
            f"Epoch {global_epoch}: TL={train_loss:.4f} TA={train_acc:.4f} | VL={val_loss:.4f} VA={val_acc:.4f} | F1={f1_macro:.4f} | BalAcc={bal_acc:.4f}")

        # Save per-epoch checkpoint
        epoch_ckpt = os.path.join(
            models_dir, f"epoch_{global_epoch:02d}_tl{train_loss:.4f}_ta{train_acc:.4f}_vl{val_loss:.4f}_va{val_acc:.4f}.pth")
        torch.save(model.state_dict(), epoch_ckpt)

        # Track best by val loss
        if val_loss < best_loss - 1e-12:
            best_loss = val_loss
            best_loss_ep = global_epoch
            best_model_wts_loss = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts_loss, os.path.join(
                models_dir, f"best_by_loss_ep{global_epoch}_vl{val_loss:.3f}_va{val_acc:.3f}_f1{f1_macro:.3f}.pth"))
            torch.save(best_model_wts_loss, os.path.join(
                root_out_dir, "best_model.pth"))
            try:
                torch.save(best_model_wts_loss, os.path.join(
                    root_out_dir, f"best_model_{global_epoch}_{val_loss:.2f}.pth"))
            except Exception as e:
                logger.warning(
                    f"Failed to save named best model snapshot: {e}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            logger.info(f"Early stopping at epoch {global_epoch}")
            break

    # Final evaluation artifacts
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    stats_dir = os.path.join(root_out_dir, "Training_Stats")
    os.makedirs(stats_dir, exist_ok=True)

    plot_confusion_matrix_png(
        cm,
        class_names,
        os.path.join(stats_dir, "confusion_matrix.png"),
        normalize=False,
        title="Confusion Matrix",
        figsize=CM_FIGSIZE,
        tick_fontsize=CM_TICK_FONTSIZE,
        annot_fontsize=CM_ANNOT_FONTSIZE,
        title_fontsize=CM_TITLE_FONTSIZE,
        axis_label_fontsize=CM_AXIS_LABEL_FONTSIZE,
    )
    plot_confusion_matrix_png(
        cm,
        class_names,
        os.path.join(stats_dir, "confusion_matrix_normalized.png"),
        normalize=True,
        title="Confusion Matrix (Normalized)",
        figsize=CM_FIGSIZE,
        tick_fontsize=CM_TICK_FONTSIZE,
        annot_fontsize=CM_ANNOT_FONTSIZE,
        title_fontsize=CM_TITLE_FONTSIZE,
        axis_label_fontsize=CM_AXIS_LABEL_FONTSIZE,
    )

    report = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True)
    with open(os.path.join(stats_dir, "classification_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Curves (smoothed)
    def _plot(x, ys, labels, title, ylabel, fname):
        plt.figure(figsize=(8, 6))
        for y, lab in zip(ys, labels):
            plt.plot(x, smooth_curve(y), label=lab)
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(stats_dir, fname), dpi=200)
        plt.close()

    _plot(hist["epoch"], [hist["train_loss"], hist["val_loss"]], [
          "Train Loss", "Val Loss"], "Loss", "Loss", "loss_curve.png")
    _plot(hist["epoch"], [hist["train_acc"], hist["val_acc"]], [
          "Train Acc", "Val Acc"], "Accuracy", "Accuracy", "accuracy_curve.png")
    _plot(hist["epoch"], [hist["val_f1"]], ["Val F1 Macro"],
          "Validation F1 Macro", "F1 Macro", "f1_macro_curve.png")

    # Summary JSON
    summary = {
        "best_by_loss_epoch": int(best_loss_ep) if best_loss_ep != -1 else None,
        "best_val_loss": float(best_loss) if best_loss < 1e10 else None,
        "epochs_trained": int(len(hist["epoch"])),
        "final_val_acc": float(hist["val_acc"][-1]) if hist["val_acc"] else None,
        "final_val_f1": float(hist["val_f1"][-1]) if hist["val_f1"] else None,
        "class_to_idx": {c: i for i, c in enumerate(class_names)},
        "early_stopping_by": "val_loss",
    }
    with open(os.path.join(stats_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Save best model with nice base name
    best_epoch_for_name = best_loss_ep if best_loss_ep != - \
        1 else (hist["epoch"][-1] if hist["epoch"] else 0)
    best_metric_for_name = best_loss if best_loss < 1e10 else 0.0
    best_base_name = f"best_model_{best_epoch_for_name}_{best_metric_for_name:.2f}"
    best_model_path_named = os.path.join(root_out_dir, best_base_name + ".pth")
    try:
        torch.save(best_model_wts_loss, best_model_path_named)
        logger.info(f"Saved best model to: {best_model_path_named}")
    except Exception as e:
        logger.warning(f"Failed to save named best model: {e}")

    if rename_epoch_folder:
        try:
            current_models_dir = os.path.join(root_out_dir, "All_Epoch_Models")
            target_models_dir = os.path.join(root_out_dir, best_base_name)
            if os.path.isdir(current_models_dir):
                if os.path.abspath(current_models_dir) != os.path.abspath(target_models_dir):
                    if not os.path.exists(target_models_dir):
                        os.rename(current_models_dir, target_models_dir)
                    else:
                        for fname in os.listdir(current_models_dir):
                            shutil.move(os.path.join(current_models_dir, fname), os.path.join(
                                target_models_dir, fname))
                        shutil.rmtree(current_models_dir, ignore_errors=True)
                logger.info(
                    f"All epoch models located at: {target_models_dir}")
        except Exception as e:
            logger.warning(f"Failed to rename per-epoch models folder: {e}")

    # Return best-by-loss weights loaded
    model.load_state_dict(best_model_wts_loss)
    return model


def main():
    # Echo config
    logger.info("Running with USER CONFIG (edit at top of file)")
    logger.info(f"DATA_DIR={DATA_DIR}")
    logger.info(f"CKPT_DIR={CKPT_DIR}")
    logger.info(
        f"IMAGE_SIZE={IMAGE_SIZE} BATCH_SIZE={BATCH_SIZE} EPOCHS={EPOCHS} HEAD_EPOCHS={HEAD_EPOCHS}")
    logger.info(
        f"LRs: head={HEAD_LR} backbone={BACKBONE_LR} wd={WEIGHT_DECAY} label_smooth={LABEL_SMOOTH}")
    logger.info(
        f"IMBALANCE={IMBALANCE} focal_gamma={FOCAL_GAMMA} USE_SWA={USE_SWA}")

    set_seeds(SEED)

    os.makedirs(CKPT_DIR, exist_ok=True)
    build_transforms(IMAGE_SIZE)

    writer = SummaryWriter(log_dir=os.path.join(
        CKPT_DIR, "Training_Stats", "tensorboard"))
    train_loader, val_loader, base_dataset, train_counts = get_data_loaders(
        DATA_DIR, BATCH_SIZE, NUM_WORKERS, IMBALANCE == "sampler", SEED
    )
    class_names = base_dataset.classes

    # Build model
    model = models.efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(nn.Dropout(
        0.6), nn.Linear(in_features, len(class_names)))
    model.to(DEVICE)

    # Save label mapping
    with open(os.path.join(CKPT_DIR, "class_to_idx.json"), "w", encoding="utf-8") as f:
        json.dump(base_dataset.class_to_idx, f, indent=2)
    with open(os.path.join(CKPT_DIR, "labels.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(class_names))

    # Criterion
    num_classes = len(class_names)
    if IMBALANCE == "class_weights":
        counts = np.array([train_counts.get(i, 0)
                          for i in range(num_classes)], dtype=np.float32)
        inv = 1.0 / np.clip(counts, 1.0, None)
        weights = inv / inv.sum() * num_classes
        class_weights_tensor = torch.tensor(
            weights, dtype=torch.float32, device=DEVICE)
        criterion = nn.CrossEntropyLoss(
            weight=class_weights_tensor, label_smoothing=LABEL_SMOOTH)
    elif IMBALANCE == "focal":
        counts = np.array([train_counts.get(i, 0)
                          for i in range(num_classes)], dtype=np.float32)
        inv = 1.0 / np.clip(counts, 1.0, None)
        alpha_from_counts = torch.tensor(
            inv / inv.sum() * num_classes, dtype=torch.float32, device=DEVICE)
        criterion = FocalLoss(
            gamma=FOCAL_GAMMA, alpha=alpha_from_counts, label_smoothing=LABEL_SMOOTH)
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)

    # Stage 1: freeze backbone (head only)
    freeze_backbone(model)
    optimizer_head = AdamW(head_parameters(
        model), lr=HEAD_LR, weight_decay=WEIGHT_DECAY)
    scheduler_head = torch.optim.lr_scheduler.OneCycleLR(
        optimizer_head, max_lr=HEAD_LR, steps_per_epoch=len(train_loader),
        epochs=HEAD_EPOCHS, pct_start=0.2, final_div_factor=1e3
    )
    model = train_model(model, criterion, optimizer_head, scheduler_head,
                        train_loader, val_loader, HEAD_EPOCHS, writer, CKPT_DIR, class_names,
                        patience=PATIENCE, best_naming_metric="loss", rename_epoch_folder=RENAME_EPOCH_FOLDER, epoch_start=0)

    # Stage 2: unfreeze last K blocks
    unfreeze_last_blocks(model, k=UNFREEZE_BLOCKS)
    backbone_params = backbone_trainable_parameters(model)
    head_params = head_parameters(model)
    param_groups = [{"params": backbone_params, "lr": BACKBONE_LR},
                    {"params": head_params,     "lr": HEAD_LR}]
    optimizer_ft = AdamW(param_groups, weight_decay=WEIGHT_DECAY)
    scheduler_ft = torch.optim.lr_scheduler.OneCycleLR(
        optimizer_ft, max_lr=[BACKBONE_LR,
                              HEAD_LR], steps_per_epoch=len(train_loader),
        epochs=max(1, EPOCHS - HEAD_EPOCHS), pct_start=0.2, final_div_factor=1e4
    )
    model = train_model(model, criterion, optimizer_ft, scheduler_ft,
                        train_loader, val_loader, max(
                            1, EPOCHS - HEAD_EPOCHS), writer,
                        CKPT_DIR, class_names, patience=PATIENCE, best_naming_metric="loss",
                        rename_epoch_folder=RENAME_EPOCH_FOLDER, epoch_start=HEAD_EPOCHS)

    # Optional SWA finishing
    if USE_SWA:
        _ = swa_finetune(model, criterion, train_loader, val_loader, CKPT_DIR,
                         swa_epochs=SWA_EPOCHS, swa_lr=SWA_LR, weight_decay=WEIGHT_DECAY)

    writer.close()
    logger.info("Training complete. Best model saved at CKPT_DIR/best_model.pth")


if __name__ == "__main__":
    main()

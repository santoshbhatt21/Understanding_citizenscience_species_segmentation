#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Train EfficientNetV2-S for citizen-science plant species classification (10 CE tree species).

This version saves the ROOT-LEVEL "best_model" by **validation loss**,
while the All_Epoch_Models folder tracks the BEST SNAPSHOT by **macro-F1** only.
Early stopping still uses macro-F1 (as requested earlier).

Other features kept:
- AMP + GradScaler, non-blocking transfers, optional Mixup and WeightedRandomSampler,
- two-stage default schedule with partial unfreeze,
- SWA finishing with BN update (now compared by **val loss** for root alias replacement),
- history.json and summary.json,
- re-evaluate the root-level best checkpoint to produce best-epoch confusion matrix/report.
"""

import os
import re
import json
import copy
import time
import math
import random
import shutil
import logging
from collections import Counter
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import ImageFile
from tqdm import tqdm

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
    classification_report,
    confusion_matrix,
    f1_score,
    balanced_accuracy_score,
    accuracy_score,
)

# =========================
# Logging / Config
# =========================
ImageFile.LOAD_TRUNCATED_IMAGES = True
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# Set your paths
DATA_PATH = r"E:/Santosh_master_thesis/flat_labeled_Leaves_Others_Trunks_1500_images"
CHECKPOINT_DIR = r"E:/Santosh_master_thesis/Ckpt_efficientnet_v2s_rootLOSS_1500_images"
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# =========================
# Hyperparameters
# =========================
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 12
IMAGE_SIZE = 640

NUM_EPOCHS_TOTAL = 35
HEAD_EPOCHS = 2                 # short head warm-up
UNFREEZE_BLOCKS = 4
PATIENCE = 7                    # early stopping by F1 macro
SINGLE_STAGE = False

HEAD_LR = 5e-4
BACKBONE_LR = 7e-5
WEIGHT_DECAY = 5e-4
LABEL_SMOOTHING = 0.02

USE_MIXUP = True
MIXUP_ALPHA = 0.2
USE_WEIGHTED_SAMPLER = True

USE_SWA = True
SWA_EPOCHS = 5
SWA_LR = 5e-5

# Best checkpoint policy:
# - Root-level "best_model.pth" and "best_model_{epoch}_{val_loss:.2f}.pth" -> **by validation loss** (CHANGED)
# - All_Epoch_Models special snapshot "best_by_f1_..." -> **by macro-F1 only**
RENAME_EPOCH_FOLDER = False
TB_SUBDIR = os.path.join(CHECKPOINT_DIR, "Training_Stats", "tensorboard")

# =========================
# Reproducibility
# =========================


def set_global_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


set_global_seed(SEED)

# =========================
# Transforms
# =========================
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.6, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(0.4, 0.4, 0.4),
    transforms.RandomApply([transforms.GaussianBlur(3)], p=0.2),
    RandomPerspective(distortion_scale=0.1, p=0.15),
    RandomAffine(15, translate=(0.05, 0.05), scale=(0.95, 1.05), shear=5),
    RandomGrayscale(p=0.1),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.5),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize(int(IMAGE_SIZE * 1.1)),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# =========================
# Dataset
# =========================


class RecursiveImageFolder(datasets.ImageFolder):
    def __init__(self, root, transform=None):
        super().__init__(root, transform)
        logger.info(f"Class mapping: {self.class_to_idx}")
        counts = Counter([s[1] for s in self.samples])
        logger.info(f"Samples per class (all data): {counts}")

# =========================
# Helpers
# =========================


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


def smooth_curve(points, factor=0.8):
    if not points:
        return points
    out, last = [], points[0]
    for p in points:
        last = last * factor + (1 - factor) * p
        out.append(last)
    return out


def plot_confusion_matrix_png(cm: np.ndarray, labels, out_path: str, normalize: bool = False, title: Optional[str] = None):
    plt.figure(figsize=(8, 6))
    matrix = cm.astype(float)
    if normalize:
        with np.errstate(all='ignore'):
            row_sums = matrix.sum(axis=1, keepdims=True)
            matrix = np.divide(matrix, row_sums, out=np.zeros_like(
                matrix), where=row_sums != 0)
    im = plt.imshow(matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    ticks = np.arange(len(labels))
    plt.xticks(ticks, labels, rotation=45, ha='right')
    plt.yticks(ticks, labels)
    fmt = ".2f" if normalize else ".0f"
    thresh = (matrix.max() if matrix.size else 0) / 2.0 if matrix.size else 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            plt.text(j, i, format(val, fmt), ha="center", va="center",
                     color="white" if val > thresh else "black")
    plt.ylabel('True')
    plt.xlabel('Predicted')
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

# ============== Mixup ==============


def do_mixup(x: torch.Tensor, y: torch.Tensor, alpha: float = 0.2) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    if alpha <= 0:
        return x, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1. - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

# =========================
# Data loaders (80/20 split)
# =========================


def get_data_loaders(data_dir: str, batch: int, use_weighted_sampler: bool):
    base = RecursiveImageFolder(data_dir, transform=None)
    targets = [s[1] for s in base.samples]

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    train_idx, val_idx = next(sss.split(np.zeros(len(targets)), targets))

    train_counts = Counter([targets[i] for i in train_idx])
    val_counts = Counter([targets[i] for i in val_idx])
    logger.info(
        f"Split totals -> Train: {len(train_idx)} | Val: {len(val_idx)}")
    logger.info(f"Train class counts: {train_counts}")
    logger.info(f"Val class counts:   {val_counts}")

    train_set = RecursiveImageFolder(data_dir, transform=train_transform)
    val_set = RecursiveImageFolder(data_dir, transform=val_transform)
    train_set.samples = [base.samples[i] for i in train_idx]
    val_set.samples = [base.samples[i] for i in val_idx]

    sampler = None
    shuffle = True
    if use_weighted_sampler:
        class_sample_counts = np.array(
            [train_counts.get(i, 0) for i in range(len(base.classes))], dtype=np.float32)
        inv_freq = 1.0 / np.clip(class_sample_counts, 1.0, None)
        sample_weights = [inv_freq[label] for _, label in train_set.samples]
        sampler = WeightedRandomSampler(
            weights=sample_weights, num_samples=len(sample_weights), replacement=True)
        shuffle = False
        logger.info("Using WeightedRandomSampler for class imbalance.")

    train_loader = DataLoader(
        train_set, batch_size=batch, shuffle=shuffle, sampler=sampler,
        num_workers=8, pin_memory=True, persistent_workers=True
    )
    val_loader = DataLoader(
        val_set, batch_size=batch, shuffle=False, num_workers=8,
        pin_memory=True, persistent_workers=True
    )
    return train_loader, val_loader, base, train_counts

# =========================
# Evaluation
# =========================


@torch.no_grad()
def evaluate(model: nn.Module,
             criterion,
             data_loader: DataLoader,
             device,
             compute_cm: bool = False,
             num_classes: Optional[int] = None):
    model.eval()
    val_loss_sum, val_correct, val_total = 0.0, 0, 0
    y_true, y_pred = [], []

    for inputs, labels in data_loader:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        preds = outputs.argmax(1)

        val_loss_sum += loss.item() * inputs.size(0)
        val_correct += (preds == labels).sum().item()
        val_total += labels.size(0)
        y_true.extend(labels.detach().cpu().tolist())
        y_pred.extend(preds.detach().cpu().tolist())

    if val_total == 0:
        return math.inf, 0.0, 0.0, 0.0, [], []

    val_loss = val_loss_sum / val_total
    acc = val_correct / val_total
    f1_macro = f1_score(y_true, y_pred, average="macro")
    bal_acc = balanced_accuracy_score(y_true, y_pred)

    if compute_cm and num_classes is not None:
        cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    else:
        cm = None

    return val_loss, acc, f1_macro, bal_acc, y_true, y_pred, cm


def save_best_epoch_artifacts(y_true, y_pred, class_names, out_root: str, prefix="best"):
    best_dir = os.path.join(out_root, "Training_Stats", "Best_Epoch")
    os.makedirs(best_dir, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    plot_confusion_matrix_png(cm, class_names, os.path.join(best_dir, f"confusion_matrix_{prefix}.png"),
                              normalize=False, title=f"Confusion Matrix ({prefix})")
    plot_confusion_matrix_png(cm, class_names, os.path.join(best_dir, f"confusion_matrix_{prefix}_normalized.png"),
                              normalize=True, title=f"Confusion Matrix ({prefix}, Normalized)")
    report = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True)
    with open(os.path.join(best_dir, f"classification_report_{prefix}.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

# =========================
# Training (two-stage helper)
# =========================


def train_model(model: nn.Module,
                criterion,
                optimizer,
                scheduler,
                train_loader,
                val_loader,
                num_epochs: int,
                writer: SummaryWriter,
                root_out_dir: str,
                class_names: List[str],
                epoch_start: int = 0,
                use_mixup: bool = False,
                mixup_alpha: float = 0.2,
                ):
    """
    Early stopping by macro-F1 (unchanged).
    ROOT best checkpoint is saved by **val loss** (CHANGED).
    All_Epoch_Models special best snapshot is **by macro-F1** only (unchanged).
    """
    scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())

    best_f1 = -1.0
    best_f1_ep = -1
    best_model_wts_f1 = copy.deepcopy(model.state_dict())

    best_loss = math.inf             # CHANGED: track best loss for root-level checkpoint
    best_loss_ep = -1
    best_model_wts_loss = copy.deepcopy(model.state_dict())

    epochs_no_improve = 0

    models_dir = os.path.join(root_out_dir, "All_Epoch_Models")
    stats_dir = os.path.join(root_out_dir, "Training_Stats")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(stats_dir, exist_ok=True)

    hist = {"epoch": [], "train_loss": [], "train_acc": [],
            "val_loss": [], "val_acc": [], "val_f1": [], "val_bal_acc": []}

    num_classes = len(class_names)

    for local_epoch in range(num_epochs):
        global_epoch = epoch_start + local_epoch + 1
        logger.info(f"Epoch {global_epoch}/{epoch_start + num_epochs}")
        model.train()

        train_loss_sum, correct, total = 0.0, 0, 0

        for inputs, labels in tqdm(train_loader, desc=f"Train {global_epoch}", leave=False):
            inputs = inputs.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            if use_mixup and MIXUP_ALPHA > 0.0:
                inputs, targets_a, targets_b, lam = do_mixup(
                    inputs, labels, alpha=mixup_alpha)
                with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                    outputs = model(inputs)
                    loss = lam * criterion(outputs, targets_a) + \
                        (1 - lam) * criterion(outputs, targets_b)
            else:
                with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            train_loss_sum += loss.item() * inputs.size(0)
            preds = outputs.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total if total else 0.0
        train_loss = train_loss_sum / total if total else 0.0

        # Validation
        val_loss, val_acc, f1_macro, bal_acc, y_true, y_pred, _ = evaluate(
            model, criterion, val_loader, DEVICE, compute_cm=False, num_classes=num_classes
        )

        # History & logs
        hist["epoch"].append(global_epoch)
        hist["train_loss"].append(train_loss)
        hist["train_acc"].append(train_acc)
        hist["val_loss"].append(val_loss)
        hist["val_acc"].append(val_acc)
        hist["val_f1"].append(f1_macro)
        hist["val_bal_acc"].append(bal_acc)

        writer.add_scalar("Loss/Train", train_loss, global_epoch)
        writer.add_scalar("Loss/Val",   val_loss,  global_epoch)
        writer.add_scalar("Acc/Train",  train_acc, global_epoch)
        writer.add_scalar("Acc/Val",    val_acc,   global_epoch)
        writer.add_scalar("F1_macro/Val", f1_macro, global_epoch)
        writer.add_scalar("Acc_balanced/Val", bal_acc, global_epoch)

        logger.info(f"Epoch {global_epoch}: TL={train_loss:.4f} TA={train_acc:.4f} | "
                    f"VL={val_loss:.4f} VA={val_acc:.4f} | F1={f1_macro:.4f} | BalAcc={bal_acc:.4f}")

        # Per-epoch checkpoint (filename includes F1 for visibility)
        epoch_ckpt = os.path.join(
            models_dir, f"epoch_{global_epoch:02d}_tl{train_loss:.4f}_ta{train_acc:.4f}_vl{val_loss:.4f}_va{val_acc:.4f}_f1{f1_macro:.4f}.pth"
        )
        torch.save(model.state_dict(), epoch_ckpt)

        # Track BEST by F1 (for All_Epoch_Models snapshot only)
        if f1_macro > best_f1 + 1e-12:
            best_f1 = f1_macro
            best_f1_ep = global_epoch
            best_model_wts_f1 = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts_f1,
                       os.path.join(models_dir, f"best_by_f1_ep{global_epoch}_vf1{f1_macro:.3f}_vl{val_loss:.3f}_va{val_acc:.3f}.pth"))
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # CHANGED: Track BEST by validation loss -> save root-level alias + named snapshot
        if val_loss < best_loss - 1e-12:
            best_loss = val_loss
            best_loss_ep = global_epoch
            best_model_wts_loss = copy.deepcopy(model.state_dict())
            try:
                torch.save(best_model_wts_loss, os.path.join(
                    root_out_dir, "best_model.pth"))
                torch.save(best_model_wts_loss, os.path.join(
                    root_out_dir, f"best_model_{global_epoch}_{best_loss:.2f}.pth"))
            except Exception as e:
                logger.warning(f"Failed to save root best-by-loss model: {e}")

        # Early stopping by F1 macro (unchanged)
        if epochs_no_improve >= PATIENCE:
            logger.info(f"Early stopping (F1 macro) at epoch {global_epoch}")
            break

    # Curves
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

    _plot(hist["epoch"], [hist["train_loss"], hist["val_loss"]],
          ["Train Loss", "Val Loss"], "Loss", "Loss", "loss_curve.png")

    _plot(hist["epoch"], [hist["train_acc"], hist["val_acc"]],
          ["Train Acc", "Val Acc"], "Accuracy", "Accuracy", "accuracy_curve.png")

    _plot(hist["epoch"], [hist["val_f1"]],
          ["Val F1 Macro"], "Validation F1 Macro", "F1 Macro", "f1_curve.png")

    # History JSON
    history_json = {
        "epoch": hist["epoch"],
        "train_loss": hist["train_loss"],
        "train_acc": hist["train_acc"],
        "val_loss": hist["val_loss"],
        "val_acc": hist["val_acc"],
        "val_f1": hist["val_f1"],
        "val_bal_acc": hist["val_bal_acc"],
    }
    with open(os.path.join(stats_dir, "history.json"), "w", encoding="utf-8") as f:
        json.dump(history_json, f, indent=2)

    # Return weights for **F1 best** to continue training (keeps previous behavior)
    model.load_state_dict(best_model_wts_f1)
    return model, best_f1_ep, best_f1, best_loss_ep, best_loss, history_json

# =========================
# SWA finishing (now compare by LOSS for root alias)
# =========================


def swa_finetune(model: nn.Module,
                 criterion,
                 train_loader,
                 val_loader,
                 out_dir: str,
                 class_names: List[str]):
    logger.info(
        f"Starting SWA finishing for {SWA_EPOCHS} epochs at lr={SWA_LR}")
    optimizer = AdamW(model.parameters(), lr=SWA_LR, weight_decay=WEIGHT_DECAY)
    swa_model = AveragedModel(model)
    swa_scheduler = SWALR(optimizer, swa_lr=SWA_LR)

    scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())

    for e in range(1, SWA_EPOCHS + 1):
        model.train()
        for inputs, labels in tqdm(train_loader, desc=f"SWA Train {e}/{SWA_EPOCHS}", leave=False):
            inputs = inputs.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            swa_model.update_parameters(model)
        swa_scheduler.step()

    # Update BN statistics for SWA model
    try:
        update_bn(train_loader, swa_model, device=DEVICE)
    except TypeError:
        update_bn(train_loader, swa_model)

    # Final evaluation after BN update
    swa_model.eval()
    num_classes = len(class_names)
    val_loss, acc, f1_macro, bal_acc, y_true, y_pred, cm = evaluate(
        swa_model, criterion, val_loader, DEVICE, compute_cm=True, num_classes=num_classes
    )

    # Save SWA final model
    try:
        torch.save(swa_model.state_dict(), os.path.join(
            out_dir, f"best_model_swa_final_{val_loss:.2f}.pth"))
        torch.save(swa_model.state_dict(), os.path.join(
            out_dir, "best_model_swa.pth"))
    except Exception:
        pass

    return swa_model, val_loss, f1_macro, (y_true, y_pred)

# =========================
# Main
# =========================


def main():
    writer = SummaryWriter(log_dir=TB_SUBDIR)

    train_loader, val_loader, base_dataset, train_counts = get_data_loaders(
        DATA_PATH, BATCH_SIZE, USE_WEIGHTED_SAMPLER
    )
    class_names = base_dataset.classes
    num_classes = len(class_names)

    # Model
    model = models.efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(nn.Dropout(
        0.6), nn.Linear(in_features, num_classes))
    model.to(DEVICE)

    # Class weights
    train_labels = [lbl for _, lbl in train_loader.dataset.samples]
    counts = Counter(train_labels)
    class_counts = np.array([counts.get(i, 0)
                            for i in range(num_classes)], dtype=np.float32)
    inv = 1.0 / np.clip(class_counts, 1.0, None)
    weights = inv / inv.sum() * num_classes
    class_weights_tensor = torch.tensor(
        weights, dtype=torch.float32, device=DEVICE)

    # Criterion (mixup-aware smoothing)
    label_smoothing = LABEL_SMOOTHING
    if USE_MIXUP and MIXUP_ALPHA > 0:
        label_smoothing = min(label_smoothing, 0.01)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights_tensor, label_smoothing=label_smoothing)

    # Stage 1 / Single-stage
    global_best_f1_ep = -1
    global_best_f1 = -1.0
    global_best_loss_ep = -1
    global_best_loss = math.inf

    if not SINGLE_STAGE:
        freeze_backbone(model)
        optimizer_head = AdamW(head_parameters(
            model), lr=HEAD_LR, weight_decay=WEIGHT_DECAY)
        scheduler_head = torch.optim.lr_scheduler.OneCycleLR(
            optimizer_head, max_lr=HEAD_LR, steps_per_epoch=len(train_loader),
            epochs=HEAD_EPOCHS, pct_start=0.2, final_div_factor=1e3
        )
        model, best_ep_s1, best_f1_s1, best_loss_ep_s1, best_loss_s1, _ = train_model(
            model, criterion, optimizer_head, scheduler_head,
            train_loader, val_loader, HEAD_EPOCHS, writer,
            CHECKPOINT_DIR, class_names, epoch_start=0,
            use_mixup=USE_MIXUP, mixup_alpha=MIXUP_ALPHA
        )
        global_best_f1 = best_f1_s1
        global_best_f1_ep = best_ep_s1
        global_best_loss = best_loss_s1
        global_best_loss_ep = best_loss_ep_s1

        # Stage 2
        unfreeze_last_blocks(model, k=UNFREEZE_BLOCKS)
        backbone_params = backbone_trainable_parameters(model)
        head_params = head_parameters(model)
        param_groups = [
            {"params": backbone_params, "lr": BACKBONE_LR},
            {"params": head_params,     "lr": HEAD_LR},
        ]
        optimizer_ft = AdamW(param_groups, weight_decay=WEIGHT_DECAY)
        scheduler_ft = torch.optim.lr_scheduler.OneCycleLR(
            optimizer_ft, max_lr=[BACKBONE_LR,
                                  HEAD_LR], steps_per_epoch=len(train_loader),
            epochs=max(1, NUM_EPOCHS_TOTAL - HEAD_EPOCHS), pct_start=0.2, final_div_factor=1e4
        )
        model, best_ep_s2, best_f1_s2, best_loss_ep_s2, best_loss_s2, history = train_model(
            model, criterion, optimizer_ft, scheduler_ft,
            train_loader, val_loader, max(
                1, NUM_EPOCHS_TOTAL - HEAD_EPOCHS), writer,
            CHECKPOINT_DIR, class_names, epoch_start=HEAD_EPOCHS,
            use_mixup=USE_MIXUP, mixup_alpha=MIXUP_ALPHA
        )
        # Track globals
        if best_f1_s2 > global_best_f1:
            global_best_f1 = best_f1_s2
            global_best_f1_ep = best_ep_s2
        if best_loss_s2 < global_best_loss:
            global_best_loss = best_loss_s2
            global_best_loss_ep = best_loss_ep_s2
    else:
        # Single-stage
        unfreeze_last_blocks(model, k=len(model.features))  # unfreeze all
        backbone_params = [
            p for n, p in model.named_parameters() if "features" in n]
        head_params = head_parameters(model)
        param_groups = [
            {"params": backbone_params, "lr": BACKBONE_LR * 0.8},
            {"params": head_params,     "lr": HEAD_LR},
        ]
        optimizer = AdamW(param_groups, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=[BACKBONE_LR * 0.8, HEAD_LR], steps_per_epoch=len(train_loader),
            epochs=NUM_EPOCHS_TOTAL, pct_start=0.2, final_div_factor=1e4
        )
        model, best_ep, best_f1, best_loss_ep, best_loss, history = train_model(
            model, criterion, scheduler=scheduler, optimizer=optimizer,
            train_loader=train_loader, val_loader=val_loader,
            num_epochs=NUM_EPOCHS_TOTAL, writer=writer,
            root_out_dir=CHECKPOINT_DIR, class_names=class_names, epoch_start=0,
            use_mixup=USE_MIXUP, mixup_alpha=MIXUP_ALPHA
        )
        global_best_f1_ep, global_best_f1 = best_ep, best_f1
        global_best_loss_ep, global_best_loss = best_loss_ep, best_loss

    # Re-evaluate ROOT-LEVEL best (by loss) and dump artifacts
    best_alias = os.path.join(CHECKPOINT_DIR, "best_model.pth")
    if os.path.exists(best_alias):
        best_weights = torch.load(best_alias, map_location=DEVICE)
        model.load_state_dict(best_weights)
        vl, va, vf1, vbal, y_true, y_pred, _ = evaluate(
            model, criterion, val_loader, DEVICE, compute_cm=False, num_classes=num_classes
        )
        save_best_epoch_artifacts(
            y_true, y_pred, class_names, CHECKPOINT_DIR, prefix="best_loss")
        logger.info(
            f"Root best-by-loss re-eval -> VL={vl:.4f} VA={va:.4f} F1={vf1:.4f} BalAcc={vbal:.4f}")

    # Summary JSON (both F1 and loss info)
    stats_dir = os.path.join(CHECKPOINT_DIR, "Training_Stats")
    os.makedirs(stats_dir, exist_ok=True)
    summary = {
        "root_best_by": "val_loss",
        "root_best_epoch": int(global_best_loss_ep) if global_best_loss_ep is not None else None,
        "root_best_val_loss": float(global_best_loss) if global_best_loss < math.inf else None,
        "best_by_f1_epoch": int(global_best_f1_ep) if global_best_f1_ep is not None else None,
        "best_val_f1": float(global_best_f1) if global_best_f1 >= 0 else None,
        "class_to_idx": {c: i for i, c in enumerate(class_names)},
    }
    with open(os.path.join(stats_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Optional SWA finishing -> compare by LOSS for root alias
    if USE_SWA:
        swa_model, swa_val_loss, swa_f1, (y_true_swa, y_pred_swa) = swa_finetune(
            model, criterion, train_loader, val_loader, CHECKPOINT_DIR, class_names
        )
        prev_best_loss = global_best_loss
        if swa_val_loss < prev_best_loss - 1e-12:
            try:
                torch.save(swa_model.state_dict(), os.path.join(
                    CHECKPOINT_DIR, "best_model.pth"))
                torch.save(swa_model.state_dict(), os.path.join(
                    CHECKPOINT_DIR, f"best_model_swa_best_{swa_val_loss:.2f}.pth"))
            except Exception:
                pass
            # Update artifacts & summary
            save_best_epoch_artifacts(
                y_true_swa, y_pred_swa, class_names, CHECKPOINT_DIR, prefix="best_loss")
            summary["root_best_val_loss"] = float(swa_val_loss)
            # SWA isn't tied to a single epoch
            summary["root_best_epoch"] = None
            with open(os.path.join(stats_dir, "summary.json"), "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
            logger.info(
                f"SWA improved root best loss to {swa_val_loss:.4f}; alias updated.")

    writer.close()


if __name__ == "__main__":
    main()

# --------------------------
# CHANGES REQUESTED (inline)
# --------------------------
# - CHANGED policy: Root-level best checkpoint now uses **validation loss** (best_model.pth and named snapshot).
# - All_Epoch_Models keeps **best_by_f1_...** only for special snapshot (per-epoch checkpoints still saved).
# - Early stopping remains **macro-F1** as before.
# - SWA replacement of root alias is now based on **val loss**, not F1.

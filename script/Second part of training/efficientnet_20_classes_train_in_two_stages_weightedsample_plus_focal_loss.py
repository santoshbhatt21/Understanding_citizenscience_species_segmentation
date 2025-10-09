#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
EfficientNet-V2-S two-stage training (head-only then unfreeze blocks) with AMP and imbalance options.
Manual run only — configure the USER CONFIG below and run this file with Python (no CLI args).

What's new vs. base:
- Early stopping by **macro-F1** or **balanced accuracy** (or val loss).
- Optional **targeted augmentation** for underperforming classes.
- Imbalance mitigation: class-weighted CE, focal loss (with alpha from counts), or weighted sampler (minority oversampling).
- Confusion matrix styling matches the user's sample (font sizes + rotation).
- Saves best checkpoints by **selected early-stop metric** and by **val loss**.
"""

import os
import re
import json
import copy
import logging
import shutil
from collections import Counter
from typing import List, Optional, Sequence, Set, Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt
from PIL import ImageFile, Image

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
DATA_DIR = r"E:/Santosh_master_thesis/classified_Leaves"         # ImageFolder root
# where to save checkpoints, logs, cms
CKPT_DIR = r"E:/Santosh_master_thesis/Checkpoints_efficientnet_Leaves"

# Training params
BATCH_SIZE = 12
IMAGE_SIZE = 640
EPOCHS = 35
# train head-only first (slightly longer for stability)
HEAD_EPOCHS = 6
UNFREEZE_BLOCKS = 4            # then unfreeze last K blocks
NUM_WORKERS = 4
SEED = 42

# Optimizer / schedulers
HEAD_LR = 5e-4
BACKBONE_LR = 7e-5
WEIGHT_DECAY = 5e-4
LABEL_SMOOTH = 0.02

# Early stopping / checkpointing metric (choose one of: "val_loss", "macro_f1", "balanced_accuracy", "val_acc")
EARLY_STOP_METRIC = "macro_f1"
PATIENCE = 9

# Imbalance handling: one of {"none", "class_weights", "sampler", "focal"}
IMBALANCE = "focal"
FOCAL_GAMMA = 2.0

# Targeted augmentation for underperforming classes (by *names* as seen in your ImageFolder)
# Leave list empty to disable.
UNDERPERFORMING_CLASSES: Sequence[str] = [
    # From CM: organ confusions and conifer group
    "Fagus sylvatica Trunks",
    "Abies alba Trunks",
    "Larix decidua Trunks",
    "Picea abies Leaves", "Picea abies Trunks",
    "Pinus sylvestris Leaves", "Pinus sylvestris Trunks",
    "Pseudotsuga menziesii Leaves", "Pseudotsuga menziesii Trunks",
]
# probability to apply the extra augmentation on those classes
EXTRA_AUG_PROB = 0.7

# Multi-task head: species (K-way) + organ (2-way)
USE_MULTITASK_HEAD = True
SPECIES_LOSS_WEIGHT = 1.0
ORGAN_LOSS_WEIGHT = 0.8
# Raise the loss for trunk in organ head a bit to fight imbalance
ORGAN_TRUNK_LOSS_WEIGHT = 1.2

# Oversample trunk images in the sampler (only if IMBALANCE=="sampler")
OVERSAMPLE_TRUNK_IN_SAMPLER = True
TRUNK_OVERSAMPLE_FACTOR = 1.5

# SWA finishing
USE_SWA = True
SWA_EPOCHS = 5
SWA_LR = 5e-5

# Misc
RENAME_EPOCH_FOLDER = False

# Use weighted sampler only in the head stage (recommended if IMBALANCE=="sampler")
SAMPLER_IN_HEAD_ONLY = True

# Optional post-training temperature scaling for calibration
USE_TEMPERATURE_SCALING = False

# Confusion matrix figure and font settings (match user's sample)
CM_FIGSIZE = (16, 14)           # width, height in inches
CM_TICK_FONTSIZE = 12           # tick labels (class names)
CM_ANNOT_FONTSIZE = 10           # numbers inside the cells
CM_TITLE_FONTSIZE = 12          # title font size
CM_AXIS_LABEL_FONTSIZE = 12     # axis label font size
CM_XROTATION = 60               # xtick rotation like sample

# ----------------------------
# Globals
# ----------------------------
ImageFile.LOAD_TRUNCATED_IMAGES = True
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("train_two_stage_improved")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_tf = None
val_tf_eval = None
extra_aug_tf = None

# ----------------------------
# Helpers
# ----------------------------


def build_transforms(image_size: int):
    """Build base train/val transforms and a stronger extra-aug (no ToTensor/Normalize in extra)."""
    global train_tf, val_tf_eval, extra_aug_tf
    # Base training transform
    train_tf = transforms.Compose([
        # Smaller crops to bias toward texture; allow tighter crops
        transforms.RandomResizedCrop(image_size, scale=(0.4, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(0.4, 0.4, 0.4),
        transforms.RandomApply([transforms.GaussianBlur(3)], p=0.2),
        RandomPerspective(distortion_scale=0.2, p=0.2),
        RandomAffine(25, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
        # Slightly increase grayscale to emphasize bark texture
        RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.5),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    # Validation/eval/CAM: resize-only (no center crop)
    val_tf_eval = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    # Extra augmentation (PIL-only transforms; no ToTensor/Normalize here)
    extra_aug_tf = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(0.6, 0.6, 0.6, 0.2),
        RandomPerspective(distortion_scale=0.35, p=1.0),
        RandomAffine(30, translate=(0.15, 0.15), scale=(0.85, 1.15), shear=15),
        RandomGrayscale(p=0.25),
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
    plt.xticks(ticks, labels, rotation=CM_XROTATION,
               ha='right', fontsize=tick_fontsize)
    plt.yticks(ticks, labels, fontsize=tick_fontsize)
    fmt = ".2f" if normalize else ".0f"
    thresh = (matrix.max() if matrix.size else 0) / 2.0 if matrix.size else 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            plt.text(
                j, i, format(val, fmt),
                ha="center", va="center",
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


class TargetedAugImageFolder(datasets.ImageFolder):
    """
    ImageFolder that optionally applies EXTRA augmentation to a set of 'weak' classes.
    extra_aug: PIL-only transforms (no ToTensor/Normalize). Applied with probability EXTRA_AUG_PROB.
    base_transform: standard train transform (includes ToTensor/Normalize).
    """

    def __init__(self, root, base_transform, extra_aug, weak_class_names: Sequence[str], extra_prob: float):
        super().__init__(root, transform=None)
        self.base_transform = base_transform
        self.extra_aug = extra_aug
        self.extra_prob = float(extra_prob)
        # map names->ids; ignore unknown names
        self.weak_ids: Set[int] = set(
            [self.class_to_idx[n] for n in weak_class_names if n in self.class_to_idx])

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)  # PIL
        if isinstance(sample, Image.Image):
            if target in self.weak_ids and np.random.rand() < self.extra_prob:
                sample = self.extra_aug(sample)
        if self.base_transform is not None:
            sample = self.base_transform(sample)
        return sample, target

# ----------------------------
# Multi-task utilities
# ----------------------------


def parse_species_organ_from_classes(class_names: Sequence[str]) -> Tuple[List[str], List[str], List[int], List[int], torch.Tensor]:
    """
    Given 20 combined class names like "Fagus sylvatica Leaves" or "Picea abies Trunks",
    return:
    - species_names: unique species list in encountered order
    - organ_names: ["Leaves", "Trunks"] (if present), or encountered order
    - class_to_species: list[int] of length K
    - class_to_organ: list[int] of length K
    - sporg_to_class: tensor[n_species, n_organs] mapping (species_id, organ_id) -> class_id
    """
    species_list: List[str] = []
    species_to_id: Dict[str, int] = {}
    organ_set: List[str] = []

    class_to_species: List[int] = []
    class_to_organ: List[int] = []

    # pass 1: collect species and organs
    for cname in class_names:
        parts = cname.strip().split()
        if len(parts) < 2:
            raise ValueError(
                f"Class name '{cname}' doesn't look like '<species> <organ>'")
        organ = parts[-1]
        species = " ".join(parts[:-1])
        if species not in species_to_id:
            species_to_id[species] = len(species_list)
            species_list.append(species)
        if organ not in organ_set:
            organ_set.append(organ)

    # Prefer organ order Leaves, Trunks if both present
    if set([o.lower() for o in organ_set]) == {"leaves", "trunks"}:
        organ_names = ["Leaves", "Trunks"]
    else:
        organ_names = organ_set
    organ_to_id = {o: i for i, o in enumerate(organ_names)}

    # pass 2: build mappings
    for cname in class_names:
        parts = cname.strip().split()
        organ = parts[-1]
        species = " ".join(parts[:-1])
        class_to_species.append(species_to_id[species])
        class_to_organ.append(organ_to_id[organ])

    n_species = len(species_list)
    n_organs = len(organ_names)
    sporg_to_class = torch.full(
        (n_species, n_organs), -1, dtype=torch.long, device=DEVICE)
    for class_id, (sp_id, org_id) in enumerate(zip(class_to_species, class_to_organ)):
        sporg_to_class[sp_id, org_id] = class_id

    return species_list, organ_names, class_to_species, class_to_organ, sporg_to_class


class MultiTaskHead(nn.Module):
    def __init__(self, in_features: int, n_species: int, n_organs: int, p: float = 0.6):
        super().__init__()
        self.dropout = nn.Dropout(p)
        self.head_species = nn.Linear(in_features, n_species)
        self.head_organ = nn.Linear(in_features, n_organs)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.dropout(x)
        return self.head_species(x), self.head_organ(x)


def get_data_loaders(
    data_dir: str,
    batch: int,
    num_workers: int,
    use_weighted_sampler: bool,
    seed: int,
    train_idx: Optional[Sequence[int]] = None,
    val_idx: Optional[Sequence[int]] = None,
):
    base = datasets.ImageFolder(data_dir, transform=None)
    targets = [s[1] for s in base.samples]

    if train_idx is None or val_idx is None:
        sss = StratifiedShuffleSplit(
            n_splits=1, test_size=0.2, random_state=seed)
        train_idx, val_idx = next(sss.split(np.zeros(len(targets)), targets))

    train_counts = Counter([targets[i] for i in train_idx])
    val_counts = Counter([targets[i] for i in val_idx])
    logger.info(
        f"Split totals -> Train: {len(train_idx)} | Val: {len(val_idx)}")
    logger.info(f"Train class counts: {train_counts}")
    logger.info(f"Val class counts:   {val_counts}")

    # Datasets
    train_set = TargetedAugImageFolder(
        data_dir, base_transform=train_tf, extra_aug=extra_aug_tf,
        weak_class_names=UNDERPERFORMING_CLASSES, extra_prob=EXTRA_AUG_PROB
    )
    val_set = datasets.ImageFolder(data_dir, transform=val_tf_eval)

    # apply split
    train_set.samples = [base.samples[i] for i in train_idx]
    val_set.samples = [base.samples[i] for i in val_idx]

    # Sampler for minority oversampling if requested
    sampler = None
    if use_weighted_sampler:
        class_sample_counts = np.array(
            [train_counts.get(i, 0) for i in range(len(base.classes))], dtype=np.float32)
        class_weights = 1.0 / np.clip(class_sample_counts, 1.0, None)
        # Optional trunk oversampling multiplier per sample
        _, organ_names, cls2sp, cls2org, _ = parse_species_organ_from_classes(
            base.classes)
        trunk_idx = None
        for i, o in enumerate(organ_names):
            if o.lower().startswith("trunk"):
                trunk_idx = i
        sample_weights = []
        for _, label in train_set.samples:
            w = class_weights[label]
            if OVERSAMPLE_TRUNK_IN_SAMPLER and trunk_idx is not None:
                if cls2org[label] == trunk_idx:
                    w *= TRUNK_OVERSAMPLE_FACTOR
            sample_weights.append(float(w))
        sampler = WeightedRandomSampler(
            sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_set, batch_size=batch, shuffle=(sampler is None),
                              sampler=sampler, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch,
                            shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, base, train_counts, train_idx, val_idx


def swa_finetune(model: nn.Module, criterion, train_loader, val_loader, out_dir: str,
                 swa_epochs: int, swa_lr: float, weight_decay: float, mt: Optional[Dict] = None):
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
                if mt is not None and isinstance(outputs, tuple):
                    # build species/organ targets
                    sp_targets = torch.tensor(
                        [mt["cls2sp"][int(t)] for t in labels.detach().cpu().tolist()], device=labels.device)
                    org_targets = torch.tensor([mt["cls2org"][int(
                        t)] for t in labels.detach().cpu().tolist()], device=labels.device)
                    sp_logits, org_logits = outputs
                    loss = mt["sp_w"] * mt["sp_crit"](sp_logits, sp_targets) + \
                        mt["org_w"] * mt["org_crit"](org_logits, org_targets)
                else:
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
                if mt is not None and isinstance(outputs, tuple):
                    sp_targets = torch.tensor(
                        [mt["cls2sp"][int(t)] for t in labels.detach().cpu().tolist()], device=labels.device)
                    org_targets = torch.tensor([mt["cls2org"][int(
                        t)] for t in labels.detach().cpu().tolist()], device=labels.device)
                    sp_logits, org_logits = outputs
                    loss = mt["sp_w"] * mt["sp_crit"](sp_logits, sp_targets) + \
                        mt["org_w"] * mt["org_crit"](org_logits, org_targets)
                    sp_pred = sp_logits.argmax(1)
                    org_pred = org_logits.argmax(1)
                    preds = mt["sporg2cls"][sp_pred, org_pred].clamp(min=0)
                else:
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


def _metric_value(metric_name: str, val_loss: float, val_acc: float, f1_macro: float, bal_acc: float) -> float:
    if metric_name == "val_loss":
        return val_loss
    elif metric_name == "macro_f1":
        return f1_macro
    elif metric_name == "balanced_accuracy":
        return bal_acc
    elif metric_name == "val_acc":
        return val_acc
    else:
        raise ValueError(f"Unknown EARLY_STOP_METRIC={metric_name}")


def _metric_is_better(metric_name: str, new: float, best: float) -> bool:
    if metric_name == "val_loss":
        return new < best - 1e-12
    else:
        return new > best + 1e-12


def train_model(model: nn.Module, criterion, optimizer, scheduler, train_loader, val_loader,
                num_epochs: int, writer: SummaryWriter, root_out_dir: str, class_names: List[str], patience: int,
                rename_epoch_folder: bool, epoch_start: int = 0, mt: Optional[Dict] = None):
    scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())

    best_loss = float('inf')
    best_loss_ep = -1
    best_model_wts_loss = copy.deepcopy(model.state_dict())

    best_metric = - \
        float('inf') if EARLY_STOP_METRIC != "val_loss" else float('inf')
    best_metric_ep = -1
    best_model_wts_metric = copy.deepcopy(model.state_dict())

    epochs_no_improve = 0

    models_dir = os.path.join(root_out_dir, "All_Epoch_Models")
    stats_dir = os.path.join(root_out_dir, "Training_Stats")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(stats_dir, exist_ok=True)

    hist = {"epoch": [], "train_loss": [], "train_acc": [],
            "val_loss": [], "val_acc": [], "val_f1": [], "val_balacc": []}

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
                if mt is not None and isinstance(outputs, tuple):
                    sp_targets = torch.tensor(
                        [mt["cls2sp"][int(t)] for t in labels.detach().cpu().tolist()], device=labels.device)
                    org_targets = torch.tensor([mt["cls2org"][int(
                        t)] for t in labels.detach().cpu().tolist()], device=labels.device)
                    sp_logits, org_logits = outputs
                    loss = mt["sp_w"] * mt["sp_crit"](sp_logits, sp_targets) + \
                        mt["org_w"] * mt["org_crit"](org_logits, org_targets)
                else:
                    loss = criterion(outputs, labels)
            prev_scale = scaler.get_scale()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() >= prev_scale:
                scheduler.step()

            train_loss_sum += loss.item() * inputs.size(0)
            if mt is not None and isinstance(outputs, tuple):
                sp_logits, org_logits = outputs
                sp_pred = sp_logits.argmax(1)
                org_pred = org_logits.argmax(1)
                preds = mt["sporg2cls"][sp_pred, org_pred].clamp(min=0)
            else:
                preds = outputs.argmax(1)
            correct += (preds == labels).sum().item()
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
                if mt is not None and isinstance(outputs, tuple):
                    sp_targets = torch.tensor(
                        [mt["cls2sp"][int(t)] for t in labels.detach().cpu().tolist()], device=labels.device)
                    org_targets = torch.tensor([mt["cls2org"][int(
                        t)] for t in labels.detach().cpu().tolist()], device=labels.device)
                    sp_logits, org_logits = outputs
                    loss = mt["sp_w"] * mt["sp_crit"](sp_logits, sp_targets) + \
                        mt["org_w"] * mt["org_crit"](org_logits, org_targets)
                    sp_pred = sp_logits.argmax(1)
                    org_pred = org_logits.argmax(1)
                    preds = mt["sporg2cls"][sp_pred, org_pred].clamp(min=0)
                else:
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
        hist["val_balacc"].append(bal_acc)

        writer.add_scalar("Loss/Train", train_loss, global_epoch)
        writer.add_scalar("Loss/Val",   val_loss,  global_epoch)
        writer.add_scalar("Acc/Train",  train_acc, global_epoch)
        writer.add_scalar("Acc/Val",    val_acc,   global_epoch)
        writer.add_scalar("F1_macro/Val", f1_macro, global_epoch)
        writer.add_scalar("Acc_balanced/Val", bal_acc, global_epoch)

        logger.info(
            f"Epoch {global_epoch}: TL={train_loss:.4f} TA={train_acc:.4f} | VL={val_loss:.4f} VA={val_acc:.4f} | F1={f1_macro:.4f} | BalAcc={bal_acc:.4f}")

        # Save per-epoch checkpoint (state_dict only)
        epoch_ckpt = os.path.join(
            models_dir, f"epoch_{global_epoch:02d}_tl{train_loss:.4f}_ta{train_acc:.4f}_vl{val_loss:.4f}_va{val_acc:.4f}.pth")
        torch.save(model.state_dict(), epoch_ckpt)

        # Track best by val loss (always)
        if val_loss < best_loss - 1e-12:
            best_loss = val_loss
            best_loss_ep = global_epoch
            best_model_wts_loss = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts_loss, os.path.join(
                models_dir, f"best_by_loss_ep{global_epoch}_vl{val_loss:.3f}_va{val_acc:.3f}_f1{f1_macro:.3f}.pth"))
            torch.save(best_model_wts_loss, os.path.join(
                root_out_dir, "best_by_loss.pth"))

        # Track best by chosen early stopping metric
        curr_metric = _metric_value(
            EARLY_STOP_METRIC, val_loss, val_acc, f1_macro, bal_acc)
        if _metric_is_better(EARLY_STOP_METRIC, curr_metric, best_metric):
            best_metric = curr_metric
            best_metric_ep = global_epoch
            best_model_wts_metric = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts_metric, os.path.join(
                models_dir, f"best_by_{EARLY_STOP_METRIC}_ep{global_epoch}_{best_metric:.4f}.pth"))
            torch.save(best_model_wts_metric, os.path.join(
                root_out_dir, f"best_by_{EARLY_STOP_METRIC}.pth"))
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            logger.info(
                f"Early stopping (by {EARLY_STOP_METRIC}) at epoch {global_epoch}")
            break

    # Final evaluation artifacts (on last val predictions)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    stats_dir = os.path.join(root_out_dir, "Training_Stats")
    os.makedirs(stats_dir, exist_ok=True)

    plot_confusion_matrix_png(
        cm, class_names, os.path.join(stats_dir, "confusion_matrix.png"),
        normalize=False, title="Confusion Matrix",
        figsize=CM_FIGSIZE, tick_fontsize=CM_TICK_FONTSIZE,
        annot_fontsize=CM_ANNOT_FONTSIZE, title_fontsize=CM_TITLE_FONTSIZE,
        axis_label_fontsize=CM_AXIS_LABEL_FONTSIZE,
    )
    plot_confusion_matrix_png(
        cm, class_names, os.path.join(
            stats_dir, "confusion_matrix_normalized.png"),
        normalize=True, title="Confusion Matrix (Normalized)",
        figsize=CM_FIGSIZE, tick_fontsize=CM_TICK_FONTSIZE,
        annot_fontsize=CM_ANNOT_FONTSIZE, title_fontsize=CM_TITLE_FONTSIZE,
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
        "best_by_metric_epoch": int(best_metric_ep) if best_metric_ep != -1 else None,
        f"best_{EARLY_STOP_METRIC}": float(best_metric) if (best_metric_ep != -1) else None,
        "epochs_trained": int(len(hist["epoch"])),
        "final_val_acc": float(hist["val_acc"][-1]) if hist["val_acc"] else None,
        "final_val_f1": float(hist["val_f1"][-1]) if hist["val_f1"] else None,
        "final_bal_acc": float(hist["val_balacc"][-1]) if hist["val_balacc"] else None,
        "class_to_idx": {c: i for i, c in enumerate(class_names)},
        "early_stopping_by": EARLY_STOP_METRIC,
    }
    with open(os.path.join(stats_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Save best model names
    best_loss_base = f"best_by_loss_ep{best_loss_ep}_{best_loss:.2f}" if best_loss_ep != - \
        1 else "best_by_loss_final"
    try:
        torch.save(best_model_wts_loss, os.path.join(
            root_out_dir, best_loss_base + ".pth"))
        logger.info(
            f"Saved best-by-loss model to: {os.path.join(root_out_dir, best_loss_base + '.pth')}")
    except Exception as e:
        logger.warning(f"Failed to save named best-by-loss model: {e}")

    if best_metric_ep != -1:
        best_metric_base = f"best_by_{EARLY_STOP_METRIC}_ep{best_metric_ep}_{best_metric:.2f}"
        try:
            torch.save(best_model_wts_metric, os.path.join(
                root_out_dir, best_metric_base + ".pth"))
            logger.info(
                f"Saved best-by-{EARLY_STOP_METRIC} model to: {os.path.join(root_out_dir, best_metric_base + '.pth')}")
        except Exception as e:
            logger.warning(
                f"Failed to save named best-by-{EARLY_STOP_METRIC} model: {e}")

    # Return best-by-metric weights loaded
    if best_metric_ep != -1:
        model.load_state_dict(best_model_wts_metric)
    else:
        model.load_state_dict(best_model_wts_loss)
    return model


def run_training():
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
    logger.info(f"EARLY_STOP_METRIC={EARLY_STOP_METRIC}; PATIENCE={PATIENCE}")
    if UNDERPERFORMING_CLASSES:
        logger.info(
            f"Targeted augmentation on classes: {UNDERPERFORMING_CLASSES} (p={EXTRA_AUG_PROB})")

    # Device info
    logger.info(f"DEVICE={DEVICE}; CUDA available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        try:
            name = torch.cuda.get_device_name(0)
        except Exception:
            name = "<unknown>"
        props = torch.cuda.get_device_properties(0)
        total_mem_gb = getattr(props, 'total_memory', 0) / (1024**3)
        logger.info(
            f"GPU: {name}; CC={getattr(props, 'major', '?')}.{getattr(props, 'minor', '?')}; VRAM~{total_mem_gb:.1f} GB")

    set_seeds(SEED)
    os.makedirs(CKPT_DIR, exist_ok=True)
    build_transforms(IMAGE_SIZE)

    writer = SummaryWriter(log_dir=os.path.join(
        CKPT_DIR, "Training_Stats", "tensorboard"))
    train_loader, val_loader, base_dataset, train_counts, train_idx, val_idx = get_data_loaders(
        DATA_DIR, BATCH_SIZE, NUM_WORKERS, IMBALANCE == "sampler", SEED
    )
    class_names = base_dataset.classes

    # Build model
    model = models.efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT)
    in_features = model.classifier[1].in_features

    # Prepare multi-task mapping
    species_names, organ_names, cls2sp, cls2org, sporg2cls = parse_species_organ_from_classes(
        class_names)
    n_species, n_organs = len(species_names), len(organ_names)
    trunk_idx = None
    for i, o in enumerate(organ_names):
        if o.lower().startswith("trunk"):
            trunk_idx = i

    if USE_MULTITASK_HEAD:
        model.classifier = nn.Sequential(nn.Dropout(
            0.6), MultiTaskHead(in_features, n_species, n_organs))
    else:
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
    if USE_MULTITASK_HEAD:
        # species criterion
        if IMBALANCE == "focal":
            species_counts = np.zeros(n_species, dtype=np.float32)
            for cls_id, sp_id in enumerate(cls2sp):
                species_counts[sp_id] += train_counts.get(cls_id, 0)
            inv = 1.0 / np.clip(species_counts, 1.0, None)
            alpha_species = torch.tensor(
                inv / inv.sum() * n_species, dtype=torch.float32, device=DEVICE)
            sp_crit = FocalLoss(
                gamma=FOCAL_GAMMA, alpha=alpha_species, label_smoothing=LABEL_SMOOTH)
        elif IMBALANCE == "class_weights":
            species_counts = np.zeros(n_species, dtype=np.float32)
            for cls_id, sp_id in enumerate(cls2sp):
                species_counts[sp_id] += train_counts.get(cls_id, 0)
            inv = 1.0 / np.clip(species_counts, 1.0, None)
            w = torch.tensor(inv / inv.sum() * n_species,
                             dtype=torch.float32, device=DEVICE)
            sp_crit = nn.CrossEntropyLoss(
                weight=w, label_smoothing=LABEL_SMOOTH)
        else:
            sp_crit = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)

        # organ criterion
        if trunk_idx is not None:
            organ_weights = torch.ones(
                n_organs, dtype=torch.float32, device=DEVICE)
            organ_weights[trunk_idx] = ORGAN_TRUNK_LOSS_WEIGHT
        else:
            organ_weights = None
        if IMBALANCE == "focal":
            organ_counts = np.zeros(n_organs, dtype=np.float32)
            for cls_id, org_id in enumerate(cls2org):
                organ_counts[org_id] += train_counts.get(cls_id, 0)
            inv = 1.0 / np.clip(organ_counts, 1.0, None)
            alpha_organ = torch.tensor(
                inv / inv.sum() * n_organs, dtype=torch.float32, device=DEVICE)
            if organ_weights is not None:
                alpha_organ = alpha_organ * organ_weights
            org_crit = FocalLoss(
                gamma=FOCAL_GAMMA, alpha=alpha_organ, label_smoothing=LABEL_SMOOTH)
        else:
            org_crit = nn.CrossEntropyLoss(
                weight=organ_weights, label_smoothing=LABEL_SMOOTH)

        # package for loops
        mt = {
            "cls2sp": cls2sp,
            "cls2org": cls2org,
            "sporg2cls": sporg2cls,
            "sp_crit": sp_crit,
            "org_crit": org_crit,
            "sp_w": SPECIES_LOSS_WEIGHT,
            "org_w": ORGAN_LOSS_WEIGHT,
        }

        def criterion(outputs, labels):
            sp_targets = torch.tensor(
                [cls2sp[int(t)] for t in labels.detach().cpu().tolist()], device=labels.device)
            org_targets = torch.tensor(
                [cls2org[int(t)] for t in labels.detach().cpu().tolist()], device=labels.device)
            sp_logits, org_logits = outputs
            return SPECIES_LOSS_WEIGHT * sp_crit(sp_logits, sp_targets) + \
                ORGAN_LOSS_WEIGHT * org_crit(org_logits, org_targets)
    else:
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
    model = train_model(
        model, criterion, optimizer_head, scheduler_head,
        train_loader, val_loader, HEAD_EPOCHS, writer, CKPT_DIR, class_names,
        patience=PATIENCE, rename_epoch_folder=RENAME_EPOCH_FOLDER, epoch_start=0, mt=(mt if USE_MULTITASK_HEAD else None)
    )

    # Stage 2: unfreeze last K blocks
    unfreeze_last_blocks(model, k=UNFREEZE_BLOCKS)

    # Optionally rebuild loaders without sampler but same split indices
    if IMBALANCE == "sampler" and SAMPLER_IN_HEAD_ONLY:
        logger.info(
            "Rebuilding loaders for Stage 2 without weighted sampler (same split indices)")
        train_loader, val_loader, _, train_counts_stage2, _, _ = get_data_loaders(
            DATA_DIR, BATCH_SIZE, NUM_WORKERS, False, SEED, train_idx=train_idx, val_idx=val_idx
        )
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
    model = train_model(
        model, criterion, optimizer_ft, scheduler_ft,
        train_loader, val_loader, max(1, EPOCHS - HEAD_EPOCHS), writer,
        CKPT_DIR, class_names, patience=PATIENCE, rename_epoch_folder=RENAME_EPOCH_FOLDER, epoch_start=HEAD_EPOCHS,
        mt=(mt if USE_MULTITASK_HEAD else None)
    )

    # Optional SWA finishing
    if USE_SWA:
        _ = swa_finetune(model, criterion, train_loader, val_loader, CKPT_DIR,
                         swa_epochs=SWA_EPOCHS, swa_lr=SWA_LR, weight_decay=WEIGHT_DECAY,
                         mt=(mt if USE_MULTITASK_HEAD else None))

    # Optional temperature scaling on the validation set for calibration
    if USE_TEMPERATURE_SCALING and not USE_MULTITASK_HEAD:
        try:
            T, nll_before, nll_after = learn_temperature(model, val_loader)
            with open(os.path.join(CKPT_DIR, "temperature.json"), "w", encoding="utf-8") as f:
                json.dump({"temperature": float(T), "val_nll_before": float(
                    nll_before), "val_nll_after": float(nll_after)}, f, indent=2)
            logger.info(
                f"Calibrated temperature T={T:.3f}; NLL before={nll_before:.4f} after={nll_after:.4f}")
        except Exception as e:
            logger.warning(f"Temperature scaling failed: {e}")

    writer.close()
    logger.info(
        "Training complete. Best models saved at CKPT_DIR (best_by_loss.pth and best_by_{EARLY_STOP_METRIC}.pth)")


if __name__ == "__main__":
    # Manual run entry point (no CLI)
    run_training()

# ----------------------------
# Calibration utils
# ----------------------------


def _collect_logits_and_labels(model: nn.Module, loader: DataLoader):
    model.eval()
    all_logits = []
    all_labels = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)
            logits = model(x)
            if isinstance(logits, tuple):
                # For multitask we cannot temperature-scale a single head cleanly here; pick species head
                logits = logits[0]
            all_logits.append(logits.detach())
            all_labels.append(y.detach())
    return torch.cat(all_logits, dim=0), torch.cat(all_labels, dim=0)


def learn_temperature(model: nn.Module, val_loader: DataLoader, max_iters: int = 200, lr: float = 0.01):
    """
    Learn a scalar temperature T>0 that minimizes NLL on the validation set.
    Returns (T_value, nll_before, nll_after).
    """
    logits, labels = _collect_logits_and_labels(model, val_loader)
    temperature = torch.nn.Parameter(torch.ones(1, device=DEVICE))
    optimizer = torch.optim.Adam([temperature], lr=lr)
    nll = nn.CrossEntropyLoss()

    def _nll_at_T(T):
        return nll(logits / T, labels).item()

    nll_before = _nll_at_T(torch.tensor(1.0, device=DEVICE))
    for _ in range(max_iters):
        optimizer.zero_grad(set_to_none=True)
        loss = nll(logits / temperature, labels)
        loss.backward()
        # Project T to positive domain softly
        with torch.no_grad():
            temperature.data.clamp_(min=1e-3, max=100.0)
        optimizer.step()

    nll_after = _nll_at_T(temperature.detach())
    return float(temperature.item()), float(nll_before), float(nll_after)

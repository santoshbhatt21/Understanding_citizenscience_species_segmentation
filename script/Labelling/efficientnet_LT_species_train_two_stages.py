import os
import shutil
import re
import json
import copy
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
)


# =========================
# Logging / Config
# =========================
ImageFile.LOAD_TRUNCATED_IMAGES = True
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# Set these two paths
# Use the flat 20-class output from organs_folders_from_metadata.py
data_path = r"E:/Santosh_master_thesis/LT_species_organ_10_species"  # dataset root (flat 20 classes)
checkpoint_dir = r"E:/Santosh_master_thesis/Checkpoints_species_organ_weighted_random_sampler_focal_loss"
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

os.makedirs(checkpoint_dir, exist_ok=True)

# Training params
batch_size = 12
image_size = 640
num_epochs_total = 50     # total epochs across both stages
HEAD_EPOCHS = 4           # stage 1: head-only epochs
UNFREEZE_BLOCKS = 6      # stage 4: unfreeze last K blocks
patience = 10            # early stopping patience per stage
seed = 42
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# LRs / regularization
HEAD_LR = 1e-4           # safer head LR
BACKBONE_LR = 2e-5       # reduced backbone LR
weight_decay = 5e-4
label_smoothing = 0.02

# ===== Imbalance handling =====
USE_WEIGHTED_SAMPLER = True     # balanced batches for training
LOSS_TYPE = "focal"             # "ce" | "ce_weighted" | "focal"
FOCAL_GAMMA = 2.0               # focusing parameter (1.0–2.0 good)
FOCAL_ALPHA_FROM_COUNTS = True  # use inverse-freq as alpha for focal

# Best model naming preference: "loss", "acc" or "f1"
BEST_NAMING_METRIC = "loss"
# Keep per-epoch checkpoints in a fixed folder without renaming
RENAME_EPOCH_FOLDER = False

# Optional SWA finishing stage
USE_SWA = True
SWA_EPOCHS = 5
SWA_LR = 5e-5

# =========================
# Plotting config (confusion matrix)
# =========================
# Tune these if labels overlap or numbers are too small/large
CM_FIG_SIZE: Tuple[int, int] = (18, 14)
CM_TITLE_FS: int = 20
CM_AXIS_LABEL_FS: int = 18
CM_TICK_FS: int = 14
CM_NUMBER_FS: int = 12
CM_CBAR_FS: int = 10
CM_XTICK_ROT: int = 45

# =========================
# Transforms (no upsampling)
# =========================
train_transform = transforms.Compose([
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
    transforms.RandomErasing(p=0.5),  # must be after ToTensor
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize(int(image_size * 1.1)),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# =========================
# Dataset
# =========================


class OrganSpeciesDataset(torch.utils.data.Dataset):
    """
    Flat 20-class default (no hierarchical):
      root/
        Abies alba leaves/ *.jpg
        Abies alba trunks/ *.jpg
        ... (20 folders)
    Class names are exactly the folder names.

    Optionally supports hierarchical if prefer_flat=False:
      root/Leaves/<species>/*.jpg, root/Trunks/<species>/*.jpg
    """

    def __init__(self, root: str, include_organs: Optional[List[str]] = None, transform=None, prefer_flat: bool = True):
        super().__init__()
        self.root = root
        self.transform = transform
        self.include_organs = include_organs or ["Leaves", "Trunks"]
        self.extensions = (".jpg", ".jpeg", ".png")

        classes: List[str] = []
        class_to_idx: dict = {}
        samples: List[tuple] = []

        # Choose layout
        has_hier = any(os.path.isdir(os.path.join(self.root, d)) for d in ("Leaves", "Trunks"))

        if not prefer_flat and has_hier:
            for organ in self.include_organs:
                organ_dir = os.path.join(self.root, organ)
                if not os.path.isdir(organ_dir):
                    continue
                for species in sorted(os.listdir(organ_dir)):
                    sp_dir = os.path.join(organ_dir, species)
                    if not os.path.isdir(sp_dir):
                        continue
                    class_name = f"{species} {organ.lower()}"
                    if class_name not in class_to_idx:
                        class_to_idx[class_name] = len(classes)
                        classes.append(class_name)
                    cls_idx = class_to_idx[class_name]
                    # Walk recursively and collect images
                    for dirpath, _, filenames in os.walk(sp_dir):
                        for fname in filenames:
                            if fname.lower().endswith(self.extensions):
                                samples.append((os.path.join(dirpath, fname), cls_idx))
        else:
            # Flat layout: each subfolder under root is a class folder
            for class_folder in sorted(os.listdir(self.root)):
                class_dir = os.path.join(self.root, class_folder)
                if not os.path.isdir(class_dir):
                    continue
                class_name = class_folder
                if class_name not in class_to_idx:
                    class_to_idx[class_name] = len(classes)
                    classes.append(class_name)
                cls_idx = class_to_idx[class_name]
                for dirpath, _, filenames in os.walk(class_dir):
                    for fname in filenames:
                        if fname.lower().endswith(self.extensions):
                            samples.append((os.path.join(dirpath, fname), cls_idx))

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples

        logger.info(f"Num classes: {len(self.classes)}")
        logger.info(f"Example classes: {self.classes[:8]}")
        counts = Counter([s[1] for s in self.samples])
        logger.info(f"Samples per class (all data): {counts}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        # Robust image open with retries
        from PIL import Image, UnidentifiedImageError
        for _ in range(5):
            try:
                with Image.open(path) as img:
                    img = img.convert("RGB")
                    if self.transform:
                        img = self.transform(img)
                    return img, label
            except (OSError, UnidentifiedImageError):
                idx = (idx + 1) % len(self.samples)
                path, label = self.samples[idx]
        raise RuntimeError(
            f"Corrupted image or read error at index {idx}: {path}")

# =========================
# Data loaders (80/20 split, no upsampling)
# =========================


def get_data_loaders(data_dir: str, batch: int):
    base = OrganSpeciesDataset(data_dir, transform=None, prefer_flat=True)
    targets = [s[1] for s in base.samples]

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_idx, val_idx = next(sss.split(np.zeros(len(targets)), targets))

    train_counts = Counter([targets[i] for i in train_idx])
    val_counts   = Counter([targets[i] for i in val_idx])
    logger.info(f"Split totals -> Train: {len(train_idx)} | Val: {len(val_idx)}")
    logger.info(f"Train class counts: {train_counts}")
    logger.info(f"Val class counts:   {val_counts}")

    # Build datasets
    train_set = OrganSpeciesDataset(data_dir, transform=train_transform, prefer_flat=True)
    val_set   = OrganSpeciesDataset(data_dir, transform=val_transform,   prefer_flat=True)
    train_set.samples = [base.samples[i] for i in train_idx]
    val_set.samples   = [base.samples[i] for i in val_idx]

    # -------- Sampler (balanced) --------
    if USE_WEIGHTED_SAMPLER:
        num_classes = len(train_set.classes)
        counts_arr = np.zeros(num_classes, dtype=np.float32)
        for _, lbl in train_set.samples:
            counts_arr[lbl] += 1
        inv = 1.0 / np.clip(counts_arr, 1.0, None)
        sample_weights = np.array([inv[lbl] for _, lbl in train_set.samples], dtype=np.float32)
        sampler = WeightedRandomSampler(weights=sample_weights,
                                        num_samples=len(sample_weights),
                                        replacement=True)
        shuffle_flag = False
    else:
        counts_arr = np.zeros(len(train_set.classes), dtype=np.float32)
        for _, lbl in train_set.samples:
            counts_arr[lbl] += 1
        sampler = None
        shuffle_flag = True

    train_loader = DataLoader(train_set, batch_size=batch, sampler=sampler,
                              shuffle=shuffle_flag, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=batch, shuffle=False,
                              num_workers=4, pin_memory=True)

    return train_loader, val_loader, base, counts_arr


# =========================
# Helpers
# =========================
class FocalLoss(nn.Module):
    """
    Multi-class Focal Loss with optional per-class alpha weights.
    Args:
        gamma: focusing parameter (>0). 2.0 is a common choice.
        alpha: Tensor [C] of class weights (e.g., inverse-freq); None -> no alpha.
        reduction: "mean" | "sum" | "none"
        label_smoothing: same semantics as CrossEntropyLoss
    """
    def __init__(self, gamma: float = 2.0, alpha: Optional[torch.Tensor] = None,
                 reduction: str = "mean", label_smoothing: float = 0.0):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # CE with label smoothing -> per-sample loss
        # (we compute focal modulation on the probabilities)
        num_classes = logits.size(1)
        log_probs = F.log_softmax(logits, dim=1)            # [N, C]
        probs = log_probs.exp()                              # [N, C]

        # one-hot w/ smoothing
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.label_smoothing / (num_classes - 1))
            true_dist.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)

        # p_t = sum(one_hot * probs)
        pt = (true_dist * probs).sum(dim=1)                 # [N]
        # focal factor
        focal = (1.0 - pt).clamp(min=1e-8).pow(self.gamma)  # [N]

        # CE per-sample: -sum(true * log_probs)
        ce = -(true_dist * log_probs).sum(dim=1)            # [N]

        loss = focal * ce                                   # [N]

        # alpha weighting by class (on targets)
        if self.alpha is not None:
            a = self.alpha[targets]                         # [N]
            loss = a * loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


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


def next_best_index(root_out_dir: str) -> int:
    pat = re.compile(r"best_model_(\d+)_")
    max_idx = -1
    for f in os.listdir(root_out_dir):
        m = pat.match(f)
        if m:
            max_idx = max(max_idx, int(m.group(1)))
    return max_idx + 1


def reorder_leaves_then_trunks(labels: List[str], cm: np.ndarray):
    """Return labels and confusion matrix reordered as:
    - All '... leaves' classes first (alphabetical by species name)
    - Then all '... trunks' classes (alphabetical by species name)
    Names are kept intact; only order changes.

    If a label doesn't end with 'leaves' or 'trunks', it's appended at the end
    in original order.
    """
    def species_key(name: str) -> str:
        s = name.strip().lower()
        s = s.replace(" leaves", "").replace(" trunks", "")
        return s

    leaves_idx = [i for i, n in enumerate(labels) if n.strip().lower().endswith("leaves")]
    trunks_idx = [i for i, n in enumerate(labels) if n.strip().lower().endswith("trunks")]
    other_idx = [i for i in range(len(labels)) if i not in leaves_idx and i not in trunks_idx]

    leaves_sorted = sorted(leaves_idx, key=lambda i: species_key(labels[i]))
    trunks_sorted = sorted(trunks_idx, key=lambda i: species_key(labels[i]))
    order = leaves_sorted + trunks_sorted + other_idx

    labels_ord = [labels[i] for i in order]
    cm_ord = cm[np.ix_(order, order)] if cm is not None else None
    return labels_ord, cm_ord


def plot_confusion_matrix_png(
    cm: np.ndarray,
    labels,
    out_path: str,
    normalize: bool = False,
    title: Optional[str] = None,
    *,
    fig_size: Tuple[int, int] = (18, 14),
    title_fs: int = 20,
    axis_label_fs: int = 16,
    tick_fs: int = 10,
    number_fs: int = 9,
    cbar_fs: int = 10,
    x_tick_rotation: int = 45,
):
    """Plot and save a confusion matrix PNG with configurable font sizes.

    Args:
        cm: Confusion matrix values (int counts).
        labels: Class names in order.
        out_path: Path to save the figure.
        normalize: If True, row-normalize before plotting.
        title: Optional plot title.
        fig_size: Figure size in inches (w, h).
        title_fs: Font size for the title.
        axis_label_fs: Font size for X/Y axis labels.
        tick_fs: Font size for tick labels.
        number_fs: Font size for the annotated cell numbers.
        cbar_fs: Font size for colorbar tick labels.
        x_tick_rotation: Rotation angle for x tick labels.
    """
    plt.figure(figsize=fig_size)
    matrix = cm.astype(float)
    if normalize:
        with np.errstate(all='ignore'):
            row_sums = matrix.sum(axis=1, keepdims=True)
            matrix = np.divide(matrix, row_sums, out=np.zeros_like(
                matrix), where=row_sums != 0)
    im = plt.imshow(matrix, interpolation='nearest', cmap=plt.cm.Blues)
    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    try:
        cbar.ax.tick_params(labelsize=cbar_fs)
    except Exception:
        pass
    ticks = np.arange(len(labels))
    plt.xticks(ticks, labels, rotation=x_tick_rotation, ha='right', fontsize=tick_fs)
    plt.yticks(ticks, labels, fontsize=tick_fs)
    fmt = ".1f" if normalize else ".0f"
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
                fontsize=number_fs,
                color="white" if val > thresh else "black",
            )
    plt.ylabel('True', fontsize=axis_label_fs)
    plt.xlabel('Predicted', fontsize=axis_label_fs)
    if title:
        plt.title(title, fontsize=title_fs)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def smooth_curve(points, factor=0.8):
    if not points:
        return points
    out, last = [], points[0]
    for p in points:
        last = last * factor + (1 - factor) * p
        out.append(last)
    return out

# =========================
# SWA Finishing
# =========================


def swa_finetune(model: nn.Module,
                 criterion,
                 train_loader,
                 val_loader,
                 out_dir: str,
                 class_names: List[str]):
    logger.info(
        f"Starting SWA finishing for {SWA_EPOCHS} epochs at lr={SWA_LR}")
    optimizer = AdamW(model.parameters(), lr=SWA_LR, weight_decay=weight_decay)
    swa_model = AveragedModel(model)
    swa_scheduler = SWALR(optimizer, swa_lr=SWA_LR)

    best_swa_loss = float('inf')
    stats_dir = os.path.join(out_dir, "Training_Stats")
    os.makedirs(stats_dir, exist_ok=True)

    for e in range(1, SWA_EPOCHS + 1):
        model.train()
        for inputs, labels in tqdm(train_loader, desc=f"SWA Train {e}/{SWA_EPOCHS}", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            swa_model.update_parameters(model)

        swa_scheduler.step()

        # Evaluate SWA model (without BN update yet)
        swa_model.eval()
        val_loss_sum, val_correct, val_total = 0.0, 0, 0
        y_true, y_pred = [], []
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"SWA Val {e}/{SWA_EPOCHS}", leave=False):
                inputs, labels = inputs.to(device), labels.to(device)
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
            # Save SWA snapshot
            try:
                torch.save(swa_model.state_dict(), os.path.join(
                    out_dir, f"best_model_swa_{e}_{val_loss:.2f}.pth"))
                torch.save(swa_model.state_dict(), os.path.join(
                    out_dir, "best_model_swa.pth"))
            except Exception as e:
                logger.warning(f"Failed saving SWA snapshot: {e}")

    # Update BN statistics for SWA model using the training set
    try:
        update_bn(train_loader, swa_model, device=device)
    except TypeError:
        # Older PyTorch versions don't accept device argument
        update_bn(train_loader, swa_model)

    # Final evaluation after BN update
    swa_model.eval()
    val_loss_sum, val_correct, val_total = 0.0, 0, 0
    y_true, y_pred = [], []
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="SWA Final Val", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
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
        f"SWA Final: VL={val_loss:.4f} VA={val_acc:.4f} F1={f1_macro:.4f}")

    # Save final SWA model with metrics
    try:
        torch.save(swa_model.state_dict(), os.path.join(
            out_dir, f"best_model_swa_final_{val_loss:.2f}.pth"))
    except Exception:
        pass

    return swa_model

# =========================
# Training (used for both stages)
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
                epoch_start: int = 0):

    best_loss = float('inf')
    best_f1 = -1.0
    best_acc = -1.0
    best_loss_ep = -1
    best_f1_ep = -1
    best_acc_ep = -1

    best_model_wts_loss = copy.deepcopy(model.state_dict())
    best_model_wts_f1 = copy.deepcopy(model.state_dict())
    best_model_wts_acc = copy.deepcopy(model.state_dict())
    best_improve_idx = next_best_index(root_out_dir)
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

        for inputs, labels in tqdm(train_loader, desc=f"Train {global_epoch}", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
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
            for inputs, labels in tqdm(val_loader, desc=f"Val {global_epoch}", leave=False):
                inputs, labels = inputs.to(device), labels.to(device)
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
            f"Epoch {global_epoch}: TL={train_loss:.4f} TA={train_acc:.4f} | "
            f"VL={val_loss:.4f} VA={val_acc:.4f} | F1={f1_macro:.4f} | BalAcc={bal_acc:.4f}"
        )

        # Per-epoch checkpoint (metrics in name)
        epoch_ckpt = os.path.join(
            models_dir, f"epoch_{global_epoch:02d}_tl{train_loss:.4f}_ta{train_acc:.4f}_vl{val_loss:.4f}_va{val_acc:.4f}.pth"
        )
        torch.save(model.state_dict(), epoch_ckpt)

        # Track best by LOSS
        if val_loss < best_loss - 1e-12:
            best_loss = val_loss
            best_loss_ep = global_epoch
            best_model_wts_loss = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts_loss,
                       os.path.join(models_dir, f"best_by_loss_ep{global_epoch}_vl{val_loss:.3f}_va{val_acc:.3f}_f1{f1_macro:.3f}.pth"))
            torch.save(best_model_wts_loss, os.path.join(
                root_out_dir, "best_model.pth"))  # alias
            # Also save a named snapshot at the checkpoints root for every improvement
            try:
                named_snap = os.path.join(
                    root_out_dir, f"best_model_{global_epoch}_{val_loss:.2f}.pth")
                torch.save(best_model_wts_loss, named_snap)
            except Exception as e:
                logger.warning(
                    f"Failed to save named best model snapshot: {e}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # Track best by F1
        if f1_macro > best_f1 + 1e-12:
            best_f1 = f1_macro
            best_f1_ep = global_epoch
            best_model_wts_f1 = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts_f1,
                       os.path.join(models_dir, f"best_by_f1_ep{global_epoch}_vl{val_loss:.3f}_va{val_acc:.3f}_f1{f1_macro:.3f}.pth"))

        # Track best by Acc
        if val_acc > best_acc + 1e-12:
            best_acc = val_acc
            best_acc_ep = global_epoch
            best_model_wts_acc = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts_acc,
                       os.path.join(models_dir, f"best_by_acc_ep{global_epoch}_vl{val_loss:.3f}_va{val_acc:.3f}_f1{f1_macro:.3f}.pth"))

        if epochs_no_improve >= patience:
            logger.info(f"Early stopping at epoch {global_epoch}")
            break

    # Final evaluation artifacts (confusion matrices + report)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    # Reorder labels and confusion matrix: all leaves (alphabetical) first, then trunks (alphabetical)
    class_names_ordered, cm_ordered = reorder_leaves_then_trunks(class_names, cm)
    stats_dir = os.path.join(root_out_dir, "Training_Stats")
    os.makedirs(stats_dir, exist_ok=True)

    plot_confusion_matrix_png(
    cm_ordered,
    class_names_ordered,
        os.path.join(stats_dir, "confusion_matrix.png"),
        normalize=False,
        title="Confusion Matrix",
        fig_size=CM_FIG_SIZE,
        title_fs=CM_TITLE_FS,
        axis_label_fs=CM_AXIS_LABEL_FS,
        tick_fs=CM_TICK_FS,
        number_fs=CM_NUMBER_FS,
        cbar_fs=CM_CBAR_FS,
        x_tick_rotation=CM_XTICK_ROT,
    )
    plot_confusion_matrix_png(
    cm_ordered,
    class_names_ordered,
        os.path.join(stats_dir, "confusion_matrix_normalized.png"),
        normalize=True,
        title="Confusion Matrix (Normalized)",
        fig_size=CM_FIG_SIZE,
        title_fs=CM_TITLE_FS,
        axis_label_fs=CM_AXIS_LABEL_FS,
        tick_fs=CM_TICK_FS,
        number_fs=CM_NUMBER_FS,
        cbar_fs=CM_CBAR_FS,
        x_tick_rotation=CM_XTICK_ROT,
    )
    # Save numeric confusion matrix and labels for exact relabeling/export later
    try:
        cm_path = os.path.join(stats_dir, "confusion_matrix.json")
        cmn_path = os.path.join(stats_dir, "confusion_matrix_normalized.json")
        with open(cm_path, "w", encoding="utf-8") as f:
            json.dump({"labels": class_names_ordered, "matrix": cm_ordered.tolist()}, f)
        # normalized copy
        with np.errstate(all='ignore'):
            row_sums = cm_ordered.sum(axis=1, keepdims=True)
            cm_norm = np.divide(cm_ordered.astype(float), row_sums, out=np.zeros_like(
                cm_ordered, dtype=float), where=row_sums != 0)
        with open(cmn_path, "w", encoding="utf-8") as f:
            json.dump({"labels": class_names_ordered, "matrix": cm_norm.tolist()}, f)
        # Save labels list exactly as used
        try:
            with open(os.path.join(stats_dir, "labels.txt"), "w", encoding="utf-8") as lf:
                lf.write("\n".join(class_names_ordered))
        except Exception:
            pass
    except Exception as e:
        logger.warning(f"Failed to save confusion_matrix.json: {e}")

    # Classification report should reflect the original label order to align with predictions,
    # but for human readability we also save a version matching the reordered labels.
    report = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True)
    with open(os.path.join(stats_dir, "classification_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    try:
        report_ordered = classification_report(
            y_true, y_pred,
            target_names=class_names_ordered,
            output_dict=True,
            zero_division=0,
        )
        with open(os.path.join(stats_dir, "classification_report_ordered.json"), "w", encoding="utf-8") as f:
            json.dump(report_ordered, f, indent=2)
    except Exception:
        pass

    # Curves (smoothed)
    def _plot(x, ys, labels, title, ylabel, fname):
        plt.figure(figsize=(18, 12))
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

    # Summary JSON
    summary = {
        "best_by_loss_epoch": int(best_loss_ep) if best_loss_ep != -1 else None,
        "best_val_loss": float(best_loss) if best_loss < 1e10 else None,
        "best_by_f1_epoch": int(best_f1_ep) if best_f1_ep != -1 else None,
        "best_val_f1": float(best_f1) if best_f1 > -1 else None,
        "epochs_trained": int(len(hist["epoch"])),
        "final_val_acc": float(hist["val_acc"][-1]) if hist["val_acc"] else None,
        "final_val_f1": float(hist["val_f1"][-1]) if hist["val_f1"] else None,
        "class_to_idx": {c: i for i, c in enumerate(class_names)},
    }
    with open(os.path.join(stats_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Save best model under root_out_dir as: best_model_{epoch}_{f1:.2f}.pth
    # and rename the per-epoch models directory to the same base name.
    # Decide which metric to use for naming
    metric_choice = (BEST_NAMING_METRIC or "loss").lower()
    if metric_choice == "loss" and best_loss_ep != -1:
        best_epoch_for_name = best_loss_ep
        best_metric_for_name = best_loss
        best_weights_for_name = best_model_wts_loss
    elif metric_choice == "f1" and best_f1_ep != -1:
        best_epoch_for_name = best_f1_ep
        best_metric_for_name = best_f1
        best_weights_for_name = best_model_wts_f1
    elif metric_choice == "acc" and best_acc_ep != -1:
        best_epoch_for_name = best_acc_ep
        best_metric_for_name = best_acc
        best_weights_for_name = best_model_wts_acc
    else:
        # Fallback to best by loss if others not available
        best_epoch_for_name = best_loss_ep if best_loss_ep != - \
            1 else (hist["epoch"][-1] if hist["epoch"] else 0)
        # Use corresponding metric value for naming when possible
        if metric_choice == "f1" and hist["val_f1"] and best_epoch_for_name - 1 < len(hist["val_f1"]):
            best_metric_for_name = hist["val_f1"][best_epoch_for_name - 1]
        elif metric_choice == "acc" and hist["val_acc"] and best_epoch_for_name - 1 < len(hist["val_acc"]):
            best_metric_for_name = hist["val_acc"][best_epoch_for_name - 1]
        else:
            best_metric_for_name = best_loss if best_loss < 1e10 else 0.0
        best_weights_for_name = best_model_wts_loss

    best_base_name = f"best_model_{best_epoch_for_name}_{best_metric_for_name:.2f}"
    best_model_path_named = os.path.join(root_out_dir, best_base_name + ".pth")
    try:
        torch.save(best_weights_for_name, best_model_path_named)
        logger.info(f"Saved best model to: {best_model_path_named}")
    except Exception as e:
        logger.warning(f"Failed to save named best model: {e}")

    # Optionally rename/merge per-epoch models folder to match the best model base name
    if RENAME_EPOCH_FOLDER:
        try:
            current_models_dir = os.path.join(root_out_dir, "All_Epoch_Models")
            target_models_dir = os.path.join(root_out_dir, best_base_name)
            if os.path.isdir(current_models_dir):
                if os.path.abspath(current_models_dir) != os.path.abspath(target_models_dir):
                    if not os.path.exists(target_models_dir):
                        try:
                            os.rename(current_models_dir, target_models_dir)
                        except OSError:
                            # Fallback: move files then remove source
                            os.makedirs(target_models_dir, exist_ok=True)
                            for fname in os.listdir(current_models_dir):
                                src = os.path.join(current_models_dir, fname)
                                dst = os.path.join(target_models_dir, fname)
                                try:
                                    shutil.move(src, dst)
                                except Exception:
                                    pass
                            shutil.rmtree(current_models_dir,
                                          ignore_errors=True)
                    else:
                        # Merge contents into existing target then remove source
                        for fname in os.listdir(current_models_dir):
                            src = os.path.join(current_models_dir, fname)
                            dst = os.path.join(target_models_dir, fname)
                            try:
                                shutil.move(src, dst)
                            except Exception:
                                pass
                        shutil.rmtree(current_models_dir, ignore_errors=True)
                    logger.info(
                        f"All epoch models located at: {target_models_dir}")
                else:
                    logger.info(
                        f"All epoch models already named: {target_models_dir}")

            # Consolidate any prior best_model_* folders into the target folder (from previous stage)
            for name in os.listdir(root_out_dir):
                p = os.path.join(root_out_dir, name)
                if p == target_models_dir:
                    continue
                if os.path.isdir(p) and name.startswith("best_model_"):
                    try:
                        for fname in os.listdir(p):
                            src = os.path.join(p, fname)
                            dst = os.path.join(target_models_dir, fname)
                            try:
                                shutil.move(src, dst)
                            except Exception:
                                pass
                        shutil.rmtree(p, ignore_errors=True)
                    except Exception:
                        pass
        except Exception as e:
            logger.warning(f"Failed to rename per-epoch models folder: {e}")

    # Return the model weights corresponding to best loss (same as best_model.pth)
    model.load_state_dict(best_model_wts_loss)
    return model

# =========================
# Main (two-stage, no upsampling)
# =========================


def main():
    writer = SummaryWriter(log_dir=os.path.join(checkpoint_dir, "Training_Stats", "tensorboard"))

    train_loader, val_loader, base_dataset, train_class_counts = get_data_loaders(data_path, batch_size)
    class_names = [str(c).strip() for c in base_dataset.classes]
    num_classes = len(class_names)

    # -------- Model --------
    model = models.efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(nn.Dropout(0.6), nn.Linear(in_features, num_classes))
    model.to(device)

    # -------- Class weights from TRAIN subset --------
    inv = 1.0 / np.clip(train_class_counts, 1.0, None)           # inverse frequency
    weights = inv / inv.sum() * num_classes                      # roughly centered around 1
    class_weights_tensor = torch.tensor(weights, dtype=torch.float32, device=device)

    # -------- Choose loss --------
    if LOSS_TYPE == "ce":
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    elif LOSS_TYPE == "ce_weighted":
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor, label_smoothing=label_smoothing)
    elif LOSS_TYPE == "focal":
        if FOCAL_ALPHA_FROM_COUNTS:
            alpha = class_weights_tensor  # up-weight rare classes
        else:
            alpha = None
        criterion = FocalLoss(gamma=FOCAL_GAMMA, alpha=alpha, reduction="mean",
                              label_smoothing=label_smoothing)
    else:
        raise ValueError(f"Unknown LOSS_TYPE: {LOSS_TYPE}")


    # -------- Stage 1: freeze backbone, train head only --------
    freeze_backbone(model)
    optimizer_head = AdamW(head_parameters(
        model), lr=HEAD_LR, weight_decay=weight_decay)
    scheduler_head = torch.optim.lr_scheduler.OneCycleLR(
        optimizer_head, max_lr=HEAD_LR, steps_per_epoch=len(train_loader),
        epochs=HEAD_EPOCHS, pct_start=0.2, final_div_factor=1e3
    )
    model = train_model(
        model, criterion, optimizer_head, scheduler_head,
        train_loader, val_loader, HEAD_EPOCHS, writer,
        checkpoint_dir, class_names, epoch_start=0
    )

    # -------- Stage 2: unfreeze last K blocks, fine-tune with smaller LR for backbone --------
    unfreeze_last_blocks(model, k=UNFREEZE_BLOCKS)
    backbone_params = backbone_trainable_parameters(model)
    head_params = head_parameters(model)

    param_groups = [
        {"params": backbone_params, "lr": BACKBONE_LR},
        {"params": head_params,     "lr": HEAD_LR},
    ]
    optimizer_ft = AdamW(param_groups, weight_decay=weight_decay)
    scheduler_ft = torch.optim.lr_scheduler.OneCycleLR(
        optimizer_ft, max_lr=[BACKBONE_LR,
                              HEAD_LR], steps_per_epoch=len(train_loader),
        epochs=max(1, num_epochs_total - HEAD_EPOCHS), pct_start=0.2, final_div_factor=1e4
    )
    model = train_model(
        model, criterion, optimizer_ft, scheduler_ft,
        train_loader, val_loader, max(
            1, num_epochs_total - HEAD_EPOCHS), writer,
        checkpoint_dir, class_names, epoch_start=HEAD_EPOCHS
    )

    # -------- Optional SWA finishing --------
    if USE_SWA:
        swa_model = swa_finetune(
            model, criterion, train_loader, val_loader, checkpoint_dir, class_names)
        # Evaluate SWA vs previous best_model.pth by reading summary.json if present
        try:
            stats_dir = os.path.join(checkpoint_dir, "Training_Stats")
            summary_path = os.path.join(stats_dir, "summary.json")
            prev_best = None
            if os.path.exists(summary_path):
                with open(summary_path, "r", encoding="utf-8") as f:
                    prev = json.load(f)
                    prev_best = prev.get("best_val_loss", None)

            # Quick val eval for SWA to compare
            swa_model.eval()
            val_loss_sum, val_total = 0.0, 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = swa_model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss_sum += loss.item() * inputs.size(0)
                    val_total += labels.size(0)
            swa_val_loss = val_loss_sum / \
                val_total if val_total else float('inf')

            # If SWA better, set alias and snapshot
            if prev_best is None or (swa_val_loss is not None and swa_val_loss < prev_best - 1e-12):
                torch.save(swa_model.state_dict(), os.path.join(
                    checkpoint_dir, "best_model.pth"))
                try:
                    torch.save(swa_model.state_dict(), os.path.join(
                        checkpoint_dir, f"best_model_swa_best_{swa_val_loss:.2f}.pth"))
                except Exception:
                    pass
                # Also update summary.json best_val_loss
                if os.path.exists(summary_path):
                    prev["best_val_loss"] = float(swa_val_loss)
                    with open(summary_path, "w", encoding="utf-8") as f:
                        json.dump(prev, f, indent=2)
        except Exception as e:
            logger.warning(f"SWA post-processing failed: {e}")

    writer.close()


if __name__ == "__main__":
    main()

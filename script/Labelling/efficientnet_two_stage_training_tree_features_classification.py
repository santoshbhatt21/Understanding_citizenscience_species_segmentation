import os
import re
import json
import copy
import logging
from collections import Counter
from typing import Optional, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import ImageFile

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

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

from tqdm import tqdm

# =========================
# Logging / Config
# =========================
ImageFile.LOAD_TRUNCATED_IMAGES = True
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# Set these two paths
data_path = r"E:/Santosh_master_thesis/flat_labeled_data_leaves_trunks"
checkpoint_dir = r"E:/Santosh_master_thesis/Checkpoints_tree_organs_two_stages"
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

os.makedirs(checkpoint_dir, exist_ok=True)

# Training params
batch_size = 16
image_size = 512
num_epochs_total = 20     # total epochs across both stages
HEAD_EPOCHS = 4          # stage 1: head-only epochs
UNFREEZE_BLOCKS = 2      # stage 2: unfreeze last K blocks
patience = 5             # early stopping patience per stage
seed = 42
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# LRs / regularization
HEAD_LR = 5e-4           # safer head LR
BACKBONE_LR = 1e-4       # reduced backbone LR
weight_decay = 1e-3
label_smoothing = 0.05

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
class RecursiveImageFolder(datasets.ImageFolder):
    def __init__(self, root, transform=None):
        super().__init__(root, transform)
        logger.info(f"Class mapping: {self.class_to_idx}")
        counts = Counter([s[1] for s in self.samples])
        logger.info(f"Samples per class (all data): {counts}")

# =========================
# Data loaders (80/20 split, no upsampling)
# =========================
def get_data_loaders(data_dir: str, batch: int):
    base = RecursiveImageFolder(data_dir, transform=None)
    targets = [s[1] for s in base.samples]

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_idx, val_idx = next(sss.split(np.zeros(len(targets)), targets))

    train_counts = Counter([targets[i] for i in train_idx])
    val_counts = Counter([targets[i] for i in val_idx])
    logger.info(f"Split totals -> Train: {len(train_idx)} | Val: {len(val_idx)}")
    logger.info(f"Train class counts: {train_counts}")
    logger.info(f"Val class counts:   {val_counts}")

    train_set = RecursiveImageFolder(data_dir, transform=train_transform)
    val_set   = RecursiveImageFolder(data_dir, transform=val_transform)
    train_set.samples = [base.samples[i] for i in train_idx]
    val_set.samples   = [base.samples[i] for i in val_idx]

    train_loader = DataLoader(train_set, batch_size=batch, shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=batch, shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, val_loader, base

# =========================
# Helpers
# =========================
def freeze_backbone(model: nn.Module):
    for p in model.features.parameters():
        p.requires_grad = False

def unfreeze_last_blocks(model: nn.Module, k: int = 2):
    if k <= 0: return
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

def plot_confusion_matrix_png(cm: np.ndarray, labels, out_path: str, normalize: bool = False, title: Optional[str] = None):
    plt.figure(figsize=(8, 6))
    matrix = cm.astype(float)
    if normalize:
        with np.errstate(all='ignore'):
            row_sums = matrix.sum(axis=1, keepdims=True)
            matrix = np.divide(matrix, row_sums, out=np.zeros_like(matrix), where=row_sums != 0)
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

def smooth_curve(points, factor=0.8):
    if not points: return points
    out, last = [], points[0]
    for p in points:
        last = last * factor + (1 - factor) * p
        out.append(last)
    return out

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
    best_f1   = -1.0
    best_loss_ep = -1
    best_f1_ep   = -1

    best_model_wts_loss = copy.deepcopy(model.state_dict())
    best_model_wts_f1   = copy.deepcopy(model.state_dict())
    best_improve_idx = next_best_index(root_out_dir)
    epochs_no_improve = 0

    models_dir = os.path.join(root_out_dir, "All_Epoch_Models")
    stats_dir  = os.path.join(root_out_dir, "Training_Stats")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(stats_dir, exist_ok=True)

    hist = {"epoch": [], "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "val_f1": []}

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
                val_correct  += (preds == labels).sum().item()
                val_total    += labels.size(0)
                y_true.extend(labels.detach().cpu().tolist())
                y_pred.extend(preds.detach().cpu().tolist())

        val_acc  = val_correct / val_total if val_total else 0.0
        val_loss = val_loss_sum / val_total if val_total else 0.0
        f1_macro = f1_score(y_true, y_pred, average="macro")
        bal_acc  = balanced_accuracy_score(y_true, y_pred)

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
            torch.save(best_model_wts_loss, os.path.join(root_out_dir, "best_model.pth"))  # alias
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

        if epochs_no_improve >= patience:
            logger.info(f"Early stopping at epoch {global_epoch}")
            break

    # Final evaluation artifacts (confusion matrices + report)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    stats_dir = os.path.join(root_out_dir, "Training_Stats")
    os.makedirs(stats_dir, exist_ok=True)

    plot_confusion_matrix_png(cm, class_names, os.path.join(stats_dir, "confusion_matrix.png"),
                              normalize=False, title="Confusion Matrix")
    plot_confusion_matrix_png(cm, class_names, os.path.join(stats_dir, "confusion_matrix_normalized.png"),
                              normalize=True, title="Confusion Matrix (Normalized)")

    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    with open(os.path.join(stats_dir, "classification_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Curves (smoothed)
    def _plot(x, ys, labels, title, ylabel, fname):
        plt.figure(figsize=(8, 6))
        for y, lab in zip(ys, labels):
            plt.plot(x, smooth_curve(y), label=lab)
        plt.xlabel("Epoch"); plt.ylabel(ylabel); plt.title(title); plt.legend(); plt.grid(True)
        plt.tight_layout(); plt.savefig(os.path.join(stats_dir, fname), dpi=200); plt.close()

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

    # Return the model weights corresponding to best loss (same as best_model.pth)
    model.load_state_dict(best_model_wts_loss)
    return model

# =========================
# Main (two-stage, no upsampling)
# =========================
def main():
    writer = SummaryWriter(log_dir=os.path.join(checkpoint_dir, "Training_Stats", "tensorboard"))

    train_loader, val_loader, base_dataset = get_data_loaders(data_path, batch_size)
    class_names = base_dataset.classes

    # Model with ImageNet weights and new head
    model = models.efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(nn.Dropout(0.6), nn.Linear(in_features, len(class_names)))
    model.to(device)

    # Optional class weights from TRAIN subset (helpful if imbalanced).
    train_labels = [lbl for _, lbl in train_loader.dataset.samples]
    counts = Counter(train_labels)
    num_classes = len(class_names)
    class_counts = np.array([counts.get(i, 0) for i in range(num_classes)], dtype=np.float32)
    inv = 1.0 / np.clip(class_counts, 1.0, None)
    weights = inv / inv.sum() * num_classes  # roughly centered around 1
    class_weights_tensor = torch.tensor(weights, dtype=torch.float32, device=device)

    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor, label_smoothing=label_smoothing)

    # -------- Stage 1: freeze backbone, train head only --------
    freeze_backbone(model)
    optimizer_head = AdamW(head_parameters(model), lr=HEAD_LR, weight_decay=weight_decay)
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
        optimizer_ft, max_lr=[BACKBONE_LR, HEAD_LR], steps_per_epoch=len(train_loader),
        epochs=max(1, num_epochs_total - HEAD_EPOCHS), pct_start=0.2, final_div_factor=1e4
    )
    model = train_model(
        model, criterion, optimizer_ft, scheduler_ft,
        train_loader, val_loader, max(1, num_epochs_total - HEAD_EPOCHS), writer,
        checkpoint_dir, class_names, epoch_start=HEAD_EPOCHS
    )

    writer.close()

if __name__ == "__main__":
    main()

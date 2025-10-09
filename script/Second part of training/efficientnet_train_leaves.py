# Update the training script to also compute and save *final* confusion matrices
# for the best-by-loss and best-by-macroF1 checkpoints after training.
out_path = "Understanding_citizenscience_species_segmentation/script/Second part of training/efficientnet_train_leaves.py"

#!/usr/bin/env python
# -*- coding: utf-8 -*-

#EfficientNet-V2-S (Leaves) — OneCycleLR + Best-by-MacroF1 + Per-class review + Temperature scaling
# Plus: FINAL confusion matrices for **best-by-loss** and **best-by-F1** models.

# Manual run:
#     python efficientnet_train_leaves_onecycle_f1_temp_bestCM.py
# Edit the CONFIG block first.


import os
import copy
import json
import random
import logging
from datetime import datetime
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.models import EfficientNet_V2_S_Weights
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score, balanced_accuracy_score,
    precision_recall_fscore_support
)
from torch.utils.tensorboard import SummaryWriter
from PIL import Image, ImageFile, UnidentifiedImageError
from tqdm import tqdm
import matplotlib.pyplot as plt

ImageFile.LOAD_TRUNCATED_IMAGES = True
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===================== CONFIG =====================
data_path = r"E:/Santosh_master_thesis/classified_Leaves"
checkpoint_path = r"./Checkpoints_Leaves_OneCycle_F1_Temp_bestCM"
os.makedirs(checkpoint_path, exist_ok=True)

seed = 42
batch_size = 16
image_size = 512
num_img_per_class = None
num_epochs = 30
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

early_stop_patience = 5
use_early_stop = True

# OneCycleLR settings
MAX_LR = 1e-3
PCT_START = 0.2
DIV_FACTOR = 25
FINAL_DIV_FACTOR = 1e3
ANNEAL_STRATEGY = "cos"

label_smoothing = 0.1
weight_decay = 5e-4

# ===================== Utils =====================
def plot_confusion_matrix_png(cm, class_names, out_png, normalized=False,
                              figsize=(14, 12), annot_font=7, tick_font=8, title="Confusion Matrix"):
    mat = cm.astype(float)
    if normalized:
        row_sums = mat.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        mat = mat / row_sums

    plt.figure(figsize=figsize)
    im = plt.imshow(mat, interpolation="nearest", cmap="Blues", vmin=0.0, vmax=(1.0 if normalized else None))
    plt.title(title)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    ticks = np.arange(len(class_names))
    plt.xticks(ticks, class_names, rotation=90, fontsize=tick_font)
    plt.yticks(ticks, class_names, fontsize=tick_font)

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j] if normalized else cm[i, j]
            txt = f"{val:.2f}" if normalized else f"{int(val)}"
            plt.text(j, i, txt, ha="center", va="center", fontsize=annot_font, color="black")

    plt.tight_layout()
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.savefig(out_png, bbox_inches="tight", dpi=200)
    plt.close()

def expected_calibration_error(probs, labels, n_bins=15):
    preds = probs.argmax(axis=1)
    conf = probs.max(axis=1)
    correct = (preds == labels).astype(np.float32)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if mask.sum() == 0:
            continue
        avg_conf = conf[mask].mean()
        acc = correct[mask].mean()
        ece += (mask.mean()) * abs(acc - avg_conf)
    return float(ece)

def plot_reliability_diagram(probs, labels, out_png, n_bins=15):
    preds = probs.argmax(axis=1)
    conf = probs.max(axis=1)
    correct = (preds == labels).astype(np.float32)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_acc = []; bin_conf = []; bin_count = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if mask.sum() == 0:
            bin_acc.append(0.0); bin_conf.append((lo+hi)/2); bin_count.append(0)
        else:
            bin_acc.append(correct[mask].mean())
            bin_conf.append(conf[mask].mean())
            bin_count.append(mask.sum())

    plt.figure(figsize=(5,5))
    plt.plot([0,1],[0,1], linestyle="--", label="Perfect")
    plt.bar(np.linspace(0.5/n_bins, 1-0.5/n_bins, n_bins), bin_acc, width=1/n_bins, alpha=0.6, label="Accuracy")
    plt.plot(np.linspace(0.5/n_bins, 1-0.5/n_bins, n_bins), bin_conf, marker="o", label="Confidence")
    plt.xlabel("Confidence"); plt.ylabel("Accuracy"); plt.title("Reliability Diagram")
    plt.legend(); plt.tight_layout()
    plt.savefig(out_png, dpi=200); plt.close()

# ===================== Data =====================
class RecursiveImageFolder(Dataset):
    def __init__(self, root, transform=None):
        self.samples = []
        self.class_to_idx = {}
        self.transform = transform

        classes = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
        for idx, class_name in enumerate(sorted(classes)):
            class_path = os.path.join(root, class_name)
            self.class_to_idx[class_name] = idx
            for dirpath, dirnames, filenames in os.walk(class_path):
                dirnames.sort()
                filenames = [f for f in filenames if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                filenames.sort()
                for fname in filenames:
                    self.samples.append((os.path.join(dirpath, fname), idx))

        self.classes = list(self.class_to_idx.keys())
        logger.info(f"Found classes: {self.classes}")
        logger.info(f"Images per class: {Counter([lbl for _, lbl in self.samples])}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        max_attempts = 10
        attempts = 0
        j = idx
        while attempts < max_attempts:
            path, label = self.samples[j]
            try:
                with Image.open(path) as img:
                    img = img.convert("RGB")
                    if self.transform:
                        img = self.transform(img)
                    return img, label
            except (OSError, UnidentifiedImageError) as e:
                logger.warning(f"Skipping corrupted image: {path} ({e})")
                j = (j + 1) % len(self.samples)
                attempts += 1
        raise RuntimeError(f"Too many corrupted images around index {idx}")

class SamplesDataset(Dataset):
    def __init__(self, samples, classes, transform=None):
        self.samples = samples
        self.classes = classes
        self.transform = transform

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        with Image.open(path) as img:
            img = img.convert("RGB")
            if self.transform:
                img = self.transform(img)
        return img, label

def get_data_loaders(data_dir, batch_size, num_img_per_class, train_transform, val_transform):
    full_dataset = RecursiveImageFolder(root=data_dir)

    indices = []
    for class_idx in range(len(full_dataset.classes)):
        class_indices = [i for i, (_, lbl) in enumerate(full_dataset.samples) if lbl == class_idx]
        if num_img_per_class is None:
            sampled = class_indices
        else:
            k = min(num_img_per_class, len(class_indices))
            sampled = np.random.choice(class_indices, k, replace=False).tolist()
        indices.extend(sampled)

    np.random.shuffle(indices)
    train_size = int(0.8 * len(indices))
    train_idx, val_idx = indices[:train_size], indices[train_size:]
    assert not set(train_idx).intersection(val_idx), "Train/Val overlap detected!"

    train_samples = [full_dataset.samples[i] for i in train_idx]
    val_samples   = [full_dataset.samples[i] for i in val_idx]

    logger.info(f"Train class counts: {Counter([lbl for _, lbl in train_samples])}")
    logger.info(f"Val class counts:   {Counter([lbl for _, lbl in val_samples])}")

    train_ds = SamplesDataset(train_samples, classes=full_dataset.classes, transform=train_transform)
    val_ds   = SamplesDataset(val_samples,   classes=full_dataset.classes, transform=val_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=8, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
    return train_loader, val_loader, full_dataset

# ===================== Temperature scaling =====================
class _TempScaleWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.0)

    def forward(self, x):
        logits = self.model(x)
        return logits / self.temperature.clamp_min(1e-3)

def fit_temperature(model, loader, device="cpu", max_iter=500, lr=0.01):
    wrapper = _TempScaleWrapper(model).to(device).eval()
    nll = nn.CrossEntropyLoss()
    optimizer = optim.LBFGS([wrapper.temperature], lr=lr, max_iter=max_iter)

    logits_list, labels_list = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            logits_list.append(logits)
            labels_list.append(yb)
    logits = torch.cat(logits_list, dim=0)
    labels = torch.cat(labels_list, dim=0)

    def _eval():
        optimizer.zero_grad(set_to_none=True)
        loss = nll(logits / wrapper.temperature, labels)
        loss.backward()
        return loss

    optimizer.step(_eval)
    T = float(wrapper.temperature.data.item())
    return T

# ===================== Evaluation helper =====================
def evaluate_and_save(model, loader, class_names, out_dir, tag):
    """Evaluate current model weights and save CM & per-class metrics under out_dir/tag_* files."""
    os.makedirs(out_dir, exist_ok=True)
    model.eval()
    y_true, y_pred, logits_all = [], [], []
    val_running_loss, val_total = 0.0, 0
    ce = nn.CrossEntropyLoss(reduction="sum")

    with torch.no_grad():
        for xb, yb in tqdm(loader, desc=f"Evaluate ({tag})"):
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            loss = ce(out, yb)
            val_running_loss += loss.item()
            val_total += yb.size(0)

            pred = out.argmax(1)
            y_true.extend(yb.cpu().tolist())
            y_pred.extend(pred.cpu().tolist())
            logits_all.append(out.cpu())

    logits_all = torch.cat(logits_all, dim=0).numpy()
    probs_all = torch.softmax(torch.tensor(logits_all), dim=1).numpy()

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    acc = (np.array(y_true) == np.array(y_pred)).mean()
    val_loss = val_running_loss / max(val_total, 1)

    # Save artifacts
    plot_confusion_matrix_png(cm, class_names, os.path.join(out_dir, f"{tag}_confusion_matrix.png"),
                              normalized=False, title=f"Confusion Matrix ({tag})")
    plot_confusion_matrix_png(cm, class_names, os.path.join(out_dir, f"{tag}_confusion_matrix_normalized.png"),
                              normalized=True, title=f"Confusion Matrix Normalized ({tag})")
    np.savetxt(os.path.join(out_dir, f"{tag}_confusion_matrix.csv"), cm, fmt="%d", delimiter=",")

    report = classification_report(y_true, y_pred, target_names=class_names, digits=4, output_dict=True)
    with open(os.path.join(out_dir, f"{tag}_classification_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    prec, rec, f1, sup = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
    import pandas as pd
    df = pd.DataFrame({"class": class_names, "precision": prec, "recall": rec, "f1": f1, "support": sup})
    df.sort_values(["f1","support"], ascending=[True, True]).to_csv(
        os.path.join(out_dir, f"{tag}_per_class_metrics_ranked.csv"), index=False
    )

    summary = {"tag": tag, "val_loss": val_loss, "val_acc": float(acc), "macro_f1": float(macro_f1),
               "balanced_acc": float(bal_acc)}
    with open(os.path.join(out_dir, f"{tag}_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary

# ===================== Training =====================
def train_model(model, criterion, optimizer, scheduler, train_loader, val_loader,
                num_epochs, writer, class_names, checkpoint_dir):
    best_loss = float('inf'); best_f1 = -1.0
    best_by_loss_wts = copy.deepcopy(model.state_dict())
    best_by_f1_wts = copy.deepcopy(model.state_dict())
    best_by_loss_epoch = None; best_by_f1_epoch = None
    epochs_no_improve = 0

    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    val_f1s, val_balacc = [], []

    steps_per_epoch = len(train_loader)

    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch+1}/{num_epochs}")

        # Train
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for i, (inputs, labels) in enumerate(tqdm(train_loader, desc=f"Training {epoch+1}/{num_epochs}")):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / max(total, 1)
        train_acc = correct / max(total, 1)
        train_losses.append(train_loss); train_accuracies.append(train_acc)

        # Validate
        model.eval()
        val_running_loss, val_correct, val_total = 0.0, 0, 0
        y_true, y_pred = [], []
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Validate {epoch+1}/{num_epochs}"):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_running_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

                y_true.extend(labels.detach().cpu().tolist())
                y_pred.extend(preds.detach().cpu().tolist())

        val_loss = val_running_loss / max(val_total, 1)
        val_acc = val_correct / max(val_total, 1)
        cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
        macro_f1 = f1_score(y_true, y_pred, average="macro")
        bal_acc = balanced_accuracy_score(y_true, y_pred)

        val_losses.append(val_loss); val_accuracies.append(val_acc)
        val_f1s.append(macro_f1); val_balacc.append(bal_acc)

        # TensorBoard
        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Loss/Val", val_loss, epoch)
        writer.add_scalar("Acc/Train", train_acc, epoch)
        writer.add_scalar("Acc/Val", val_acc, epoch)
        writer.add_scalar("F1/Macro_Val", macro_f1, epoch)
        writer.add_scalar("Acc/Balanced_Val", bal_acc, epoch)

        # Per-epoch checkpointing
        if val_loss < best_loss - 1e-12:
            best_loss = val_loss
            best_by_loss_wts = copy.deepcopy(model.state_dict())
            best_by_loss_epoch = epoch + 1
            torch.save(best_by_loss_wts, os.path.join(checkpoint_dir, f"best_by_loss_ep{best_by_loss_epoch}_{best_loss:.3f}.pth"))
            torch.save(best_by_loss_wts, os.path.join(checkpoint_dir, "best_by_loss.pth"))

        if macro_f1 > best_f1 + 1e-12:
            best_f1 = macro_f1
            best_by_f1_wts = copy.deepcopy(model.state_dict())
            best_by_f1_epoch = epoch + 1
            torch.save(best_by_f1_wts, os.path.join(checkpoint_dir, f"best_by_f1_ep{best_by_f1_epoch}_{best_f1:.3f}.pth"))
            torch.save(best_by_f1_wts, os.path.join(checkpoint_dir, "best_by_f1.pth"))
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        logger.info(f"Train: Loss={train_loss:.4f} Acc={train_acc:.4f} | "
                    f"Val: Loss={val_loss:.4f} Acc={val_acc:.4f} MacroF1={macro_f1:.4f} BalAcc={bal_acc:.4f} | "
                    f"BestF1={best_f1:.4f} (no-improve={epochs_no_improve}/{early_stop_patience})")

        if use_early_stop and epochs_no_improve >= early_stop_patience:
            logger.info(f"Early stopping on Macro-F1 at epoch {epoch+1}.")
            break

    # Save a small training summary
    with open(os.path.join(checkpoint_dir, "training_summary.json"), "w", encoding="utf-8") as f:
        json.dump({
            "best_by_loss_epoch": best_by_loss_epoch,
            "best_val_loss": best_loss,
            "best_by_f1_epoch": best_by_f1_epoch,
            "best_macro_f1": best_f1
        }, f, indent=2)

    return best_by_loss_wts, best_by_f1_wts, best_by_loss_epoch, best_by_f1_epoch

# ===================== Main =====================
def main():
    # Repro
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

    log_dir = os.path.join("runs", "run_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    writer = SummaryWriter(log_dir)

    # Augmentations
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=10),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        transforms.GaussianBlur(3, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        transforms.RandomErasing(p=0.3, value='random'),
    ])
    val_transform = transforms.Compose([
        transforms.Resize(int(image_size * 1.1)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    # Data
    train_loader, val_loader, dataset = get_data_loaders(
        data_path, batch_size, num_img_per_class, train_transform, val_transform
    )

    # Class weights from train subset
    train_labels = [lbl for _, lbl in train_loader.dataset.samples]
    classes_unique = np.arange(len(dataset.classes))
    class_weights = compute_class_weight(class_weight="balanced", classes=classes_unique, y=train_labels)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)

    # Model
    model = models.efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(nn.Dropout(p=0.6), nn.Linear(in_features, len(dataset.classes)))
    model.to(device)

    # Loss / Optim / OneCycleLR
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor, label_smoothing=label_smoothing)
    optimizer = optim.AdamW(model.parameters(), lr=MAX_LR / DIV_FACTOR, weight_decay=weight_decay)
    steps_per_epoch = len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=MAX_LR, epochs=num_epochs, steps_per_epoch=steps_per_epoch,
        pct_start=PCT_START, anneal_strategy=ANNEAL_STRATEGY, div_factor=DIV_FACTOR,
        final_div_factor=FINAL_DIV_FACTOR
    )

    # Train (returns the two best states)
    best_loss_wts, best_f1_wts, ep_loss, ep_f1 = train_model(
        model, criterion, optimizer, scheduler, train_loader, val_loader,
        num_epochs, writer, dataset.classes, checkpoint_path
    )

    # Evaluate and save FINAL CMs for both best models
    # 1) Best-by-loss
    model.load_state_dict(best_loss_wts)
    evaluate_and_save(model, val_loader, dataset.classes, checkpoint_path, tag=f"best_by_loss_ep{ep_loss}")
    # 2) Best-by-F1
    model.load_state_dict(best_f1_wts)
    evaluate_and_save(model, val_loader, dataset.classes, checkpoint_path, tag=f"best_by_f1_ep{ep_f1}")

    # Optional: temperature scaling on the best-by-F1 model (common choice for deployment)
    T = fit_temperature(model, val_loader, device=device)
    # Save T and reliability diagrams
    logits_list, labels_list = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits_list.append(model(xb).cpu())
            labels_list.append(yb.cpu())
    logits = torch.cat(logits_list, dim=0)
    labels = torch.cat(labels_list, dim=0).numpy()
    probs_before = torch.softmax(logits, dim=1).numpy()
    probs_after  = torch.softmax(logits / T, dim=1).numpy()

    calib_dir = os.path.join(checkpoint_path, "calibration")
    os.makedirs(calib_dir, exist_ok=True)
    with open(os.path.join(calib_dir, "temperature.json"), "w", encoding="utf-8") as f:
        json.dump({"temperature": float(T)}, f, indent=2)

    from pathlib import Path as _P
    def expected_calibration_error_np(probs, labels, n_bins=15):
        preds = probs.argmax(axis=1)
        conf = probs.max(axis=1)
        correct = (preds == labels).astype(np.float32)
        bins = np.linspace(0.0, 1.0, n_bins + 1)
        ece = 0.0
        for i in range(n_bins):
            lo, hi = bins[i], bins[i + 1]
            mask = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
            if mask.sum() == 0: continue
            ece += (mask.mean()) * abs(correct[mask].mean() - conf[mask].mean())
        return float(ece)

    ece_before = expected_calibration_error_np(probs_before, labels)
    ece_after  = expected_calibration_error_np(probs_after,  labels)

    def plot_reliability_diagram_np(probs, labels, out_png, n_bins=15):
        preds = probs.argmax(axis=1); conf = probs.max(axis=1)
        correct = (preds == labels).astype(np.float32)
        bins = np.linspace(0.0, 1.0, n_bins + 1)
        xs = np.linspace(0.5/n_bins, 1-0.5/n_bins, n_bins)
        accs = []; confs = []
        for i in range(n_bins):
            lo, hi = bins[i], bins[i + 1]
            mask = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
            if mask.sum() == 0:
                accs.append(0.0); confs.append((lo+hi)/2)
            else:
                accs.append(correct[mask].mean()); confs.append(conf[mask].mean())
        plt.figure(figsize=(5,5))
        plt.plot([0,1],[0,1],"--",label="Perfect")
        plt.bar(xs, accs, width=1/n_bins, alpha=0.6, label="Accuracy")
        plt.plot(xs, confs, marker="o", label="Confidence")
        plt.xlabel("Confidence"); plt.ylabel("Accuracy"); plt.title("Reliability")
        plt.legend(); plt.tight_layout()
        plt.savefig(out_png, dpi=200); plt.close()

    plot_reliability_diagram_np(probs_before, labels, os.path.join(calib_dir, "reliability_before.png"))
    plot_reliability_diagram_np(probs_after,  labels, os.path.join(calib_dir, "reliability_after.png"))
    with open(os.path.join(calib_dir, "ece.json"), "w", encoding="utf-8") as f:
        json.dump({"ece_before": ece_before, "ece_after": ece_after}, f, indent=2)
    logger.info(f"Temperature scaling: T={T:.3f} | ECE before={ece_before:.4f} → after={ece_after:.4f}")

    writer.close()

if __name__ == "__main__":
    main()

# Removed erroneous code writing block.

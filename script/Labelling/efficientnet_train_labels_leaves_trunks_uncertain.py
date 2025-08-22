import os
import copy
import logging
import matplotlib.pyplot as plt
import numpy as np
import json
import random
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.amp import autocast, GradScaler
from contextlib import nullcontext
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, models, transforms
from torchvision.models import EfficientNet_V2_S_Weights
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.tensorboard import SummaryWriter
from PIL import ImageFile
from collections import Counter
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, f1_score

ImageFile.LOAD_TRUNCATED_IMAGES = True
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===================== Config =====================
data_path = "E:/Santosh_master_thesis/Understanding_citizenscience_species_segmentation/split_images_into_leaves_trunks_leaves_uncertain"
checkpoint_dir = "E:/Santosh_master_thesis/Understanding_citizenscience_species_segmentation/Checkpoints_tree_organs"
os.makedirs(checkpoint_dir, exist_ok=True)

batch_size = 16
image_size = 512
num_epochs = 30
patience = 10
use_balanced_sampler = True  # used only if upsampling is disabled
seed = 42

# Upsample training set to at least this many images per class (duplicates allowed).
# Set to None to disable duplication upsampling.
target_train_per_class = 4000  # e.g., Uncertain has 1100 -> will be duplicated to 4000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(seed)

# ===================== Augmentation =====================
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(image_size, scale=(0.6, 1.0), ratio=(0.75, 1.33)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.RandomRotation(20),
    transforms.RandomAffine(degrees=0, translate=(0.1,0.1), scale=(0.8,1.2), shear=10),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
    transforms.GaussianBlur(3, sigma=(0.1, 2.0)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    transforms.RandomErasing(p=0.25, value='random')
])

val_transform = transforms.Compose([
    transforms.Resize(int(image_size * 1.1)),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

# ===================== Dataset =====================
class RecursiveImageFolder(datasets.ImageFolder):
    def __init__(self, root, transform=None):
        super().__init__(root=root, transform=transform)
        logger.info(f"Class to idx mapping: {self.class_to_idx}")
        counts = Counter([t for _, t in self.samples])
        logger.info(f"Images per class: {counts}")

# ===================== Data Loading (Stratified split + optional upsampling) =====================
def get_data_loaders(data_dir, batch_size, train_transform, val_transform, val_ratio=0.2):
    rng = np.random.default_rng(seed)

    full_train = RecursiveImageFolder(data_dir, transform=train_transform)
    targets = [s[1] for s in full_train.samples]

    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
    train_idx, val_idx = next(sss.split(np.zeros(len(targets)), targets))

    base = RecursiveImageFolder(data_dir, transform=None)
    train_ds = RecursiveImageFolder(data_dir, transform=train_transform)
    val_ds   = RecursiveImageFolder(data_dir, transform=val_transform)

    # Original counts on split
    train_counts_orig = Counter([targets[i] for i in train_idx])
    val_counts = Counter([targets[i] for i in val_idx])
    logger.info(f"Train class counts (orig): {train_counts_orig}")
    logger.info(f"Val class counts:          {val_counts}")

    # Optional duplication upsampling to target_train_per_class
    train_idx_expanded = list(train_idx)
    upsampling_applied = False
    if target_train_per_class is not None:
        class_to_indices = {}
        for i in train_idx:
            cls = targets[i]
            class_to_indices.setdefault(cls, []).append(i)

        expanded_indices = []
        for cls, idxs in class_to_indices.items():
            need = max(len(idxs), target_train_per_class)
            if need > len(idxs):
                upsampling_applied = True
                reps = need - len(idxs)
                # Duplicate with random sampling (with replacement)
                extra = rng.choice(idxs, size=reps, replace=True).tolist()
                expanded = idxs + extra
            else:
                expanded = idxs  # already >= target, keep as is
            expanded_indices.extend(expanded)

        rng.shuffle(expanded_indices)
        train_idx_expanded = expanded_indices

        # Log new counts
        train_counts_new = Counter([targets[i] for i in train_idx_expanded])
        logger.info(f"Train class counts (upsampled to >= {target_train_per_class}): {train_counts_new}")

    # Assign samples
    train_ds.samples = [base.samples[i] for i in train_idx_expanded]
    val_ds.samples   = [base.samples[i] for i in val_idx]

    # Sampler: if we already duplicated, do not use weighted sampler
    train_sampler = None
    if (not upsampling_applied) and use_balanced_sampler:
        class_freq = Counter([t for _, t in train_ds.samples])
        weights = {c: 1.0 / class_freq[c] for c in class_freq}
        sample_weights = [weights[t] for _, t in train_ds.samples]
        train_sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              sampler=train_sampler, shuffle=train_sampler is None,
                              num_workers=8, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size*2, shuffle=False,
                              num_workers=8, pin_memory=True)

    return train_loader, val_loader, base, train_idx, val_idx

# ===================== Train / Validate =====================
def train_model(model, criterion, optimizer, scheduler, train_loader, val_loader, num_epochs, writer, root_out_dir, class_names):
    amp_enabled = (device.type == 'cuda')
    best_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0
    scaler = GradScaler('cuda', enabled=amp_enabled) if amp_enabled else GradScaler(enabled=False)

    # Output dirs
    models_dir = os.path.join(root_out_dir, "all_epoch_models")
    stats_dir  = os.path.join(root_out_dir, "training_stats")
    figs_dir   = os.path.join(stats_dir, "figures")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(figs_dir, exist_ok=True)

    hist = {"epoch": [], "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch+1}/{num_epochs}")

        # ---------- Train ----------
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for inputs, labels in tqdm(train_loader, desc=f"Train {epoch+1}", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)

            amp_ctx = autocast('cuda', enabled=amp_enabled) if amp_enabled else nullcontext()
            with amp_ctx:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scheduler.step()
            scaler.update()

            running_loss += loss.item() * inputs.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / max(1, total)
        train_acc = correct / max(1, total)

        # ---------- Validate ----------
        model.eval()
        val_running_loss, val_correct, val_total = 0.0, 0, 0
        y_true_epoch, y_pred_epoch = [], []
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Val {epoch+1}", leave=False):
                inputs, labels = inputs.to(device), labels.to(device)
                amp_ctx = autocast('cuda', enabled=amp_enabled) if amp_enabled else nullcontext()
                with amp_ctx:
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                preds = outputs.argmax(1)
                val_running_loss += loss.item() * inputs.size(0)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                y_true_epoch.extend(labels.detach().cpu().tolist())
                y_pred_epoch.extend(preds.detach().cpu().tolist())

        val_loss = val_running_loss / max(1, val_total)
        val_acc = val_correct / max(1, val_total)

        # ---------- Logging ----------
        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Loss/Val", val_loss, epoch)
        writer.add_scalar("Acc/Train", train_acc, epoch)
        writer.add_scalar("Acc/Val", val_acc, epoch)
        try:
            f1_macro = f1_score(y_true_epoch, y_pred_epoch, average="macro")
            writer.add_scalar("Val/F1_macro", f1_macro, epoch)
        except Exception:
            f1_macro = None

        logger.info(f"TL {train_loss:.4f} TA {train_acc:.4f} | VL {val_loss:.4f} VA {val_acc:.4f} | LR {optimizer.param_groups[0]['lr']:.2e}")

        hist["epoch"].append(epoch+1)
        hist["train_loss"].append(train_loss)
        hist["train_acc"].append(train_acc)
        hist["val_loss"].append(val_loss)
        hist["val_acc"].append(val_acc)

        # ---------- Save: per-epoch model ----------
        epoch_ckpt = os.path.join(models_dir, f"epoch_{epoch+1:02d}_tl{train_loss:.3f}_vl{val_loss:.3f}_va{val_acc:.3f}.pth")
        torch.save(model.state_dict(), epoch_ckpt)

        # ---------- Confusion Matrices (final) ----------
        cm = confusion_matrix(y_true_epoch, y_pred_epoch, labels=list(range(len(class_names))))
        df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
        df_cm.to_csv(os.path.join(figs_dir, f"confusion_matrix_final.csv"))

        with np.errstate(all='ignore'):
            row_sums = cm.sum(axis=1, keepdims=True)
            cm_norm = np.divide(cm.astype(np.float64), row_sums, out=np.zeros_like(cm, dtype=np.float64), where=row_sums != 0)
        df_cm_norm = pd.DataFrame(cm_norm, index=class_names, columns=class_names)
        df_cm_norm.to_csv(os.path.join(figs_dir, f"confusion_matrix_normalized_final.csv"))

        # ---------- Early stopping & best checkpoint ----------
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
            # Save "best so far" in all_epoch_models and also to checkpoint root
            torch.save(model.state_dict(), os.path.join(models_dir, f"best_model_ep{epoch+1}_vl{best_loss:.3f}_va{val_acc:.3f}.pth"))
            torch.save(model.state_dict(), os.path.join(root_out_dir, "best_model.pth"))
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

    # Restore best and save a final snapshot of the best weights
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), os.path.join(models_dir, "best_model_final.pth"))
    torch.save(model.state_dict(), os.path.join(root_out_dir, "best_model.pth"))  # ensure final best at root

    # Save final history CSV & JSON -> training_stats
    hist_df = pd.DataFrame(hist)
    stats_dir = os.path.join(root_out_dir, "training_stats")
    os.makedirs(stats_dir, exist_ok=True)
    hist_df.to_csv(os.path.join(stats_dir, "history.csv"), index=False)
    with open(os.path.join(stats_dir, "history.json"), "w", encoding="utf-8") as f:
        json.dump(hist, f, indent=2)

    # Plots -> training_stats/figures
    plt.figure()
    plt.plot(hist["epoch"], hist["train_loss"], label="Train Loss")
    plt.plot(hist["epoch"], hist["val_loss"], label="Val Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(figs_dir, "loss_curve.png")); plt.close()

    plt.figure()
    plt.plot(hist["epoch"], hist["train_acc"], label="Train Acc")
    plt.plot(hist["epoch"], hist["val_acc"], label="Val Acc")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(figs_dir, "accuracy_curve.png")); plt.close()

    # Final summary JSON -> training_stats
    summary = {
        "best_val_loss": float(best_loss),
        "epochs_trained": len(hist["epoch"]),
        "batch_size": batch_size,
        "image_size": image_size,
        "num_classes": len(class_names),
        "num_epochs_cfg": num_epochs,
        "patience": patience,
        "use_balanced_sampler": use_balanced_sampler,
        "target_train_per_class": target_train_per_class,
        "seed": seed,
        "device": str(device),
        "f1_macro_best": None if f1_macro is None else float(f1_macro)
    }
    with open(os.path.join(stats_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return model

# ===================== Main =====================
def main():
    # Save directly under checkpoint_dir
    root_out_dir = checkpoint_dir
    os.makedirs(root_out_dir, exist_ok=True)

    # TensorBoard logs under training_stats/tb
    stats_dir = os.path.join(root_out_dir, "training_stats")
    os.makedirs(stats_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(stats_dir, "tb"))

    train_loader, val_loader, base_dataset, train_idx, val_idx = get_data_loaders(
        data_path, batch_size, train_transform, val_transform, val_ratio=0.2
    )
    class_names = list(base_dataset.classes)

    # Save class mapping to training_stats
    with open(os.path.join(stats_dir, "classes.json"), "w", encoding="utf-8") as f:
        json.dump(base_dataset.class_to_idx, f, indent=2)

    # Class weights from TRAIN ONLY (original split, before upsampling)
    y_train = [base_dataset.samples[i][1] for i in train_idx]
    classes = np.unique(y_train)
    class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

    # Model
    weights = EfficientNet_V2_S_Weights.DEFAULT
    model = models.efficientnet_v2_s(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.6),
        nn.Linear(in_features, len(base_dataset.classes))
    )
    model.to(device)

    # Loss / Optim / Sched
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor, label_smoothing=0.1)
    optimizer = AdamW(model.parameters(), lr=2e-4, weight_decay=5e-4)
    steps_per_epoch = len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=2e-3, epochs=num_epochs, steps_per_epoch=steps_per_epoch,
        pct_start=0.2, div_factor=10.0, final_div_factor=1e3, anneal_strategy="cos"
    )

    # Train
    train_model(model, criterion, optimizer, scheduler, train_loader, val_loader, num_epochs, writer, root_out_dir, class_names)
    writer.close()

if __name__ == '__main__':
    main()
# This script is designed to train an EfficientNet-V2-S model on a dataset of leaves, trunks, and uncertain images.
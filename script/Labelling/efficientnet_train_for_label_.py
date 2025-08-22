import os
import copy
import json
import logging
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms, models
from torchvision.models import EfficientNet_V2_S_Weights
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm import tqdm
from collections import Counter
import pandas as pd
from PIL import ImageFile

def smooth_curve(points, factor=0.8):
    smoothed = []
    for p in points:
        if smoothed:            smoothed.append(smoothed[-1] * factor + p * (1 - factor))
        else:
            smoothed.append(p)
    return smoothed

ImageFile.LOAD_TRUNCATED_IMAGES = True
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ======= Config =======
data_path = "E:/Santosh_master_thesis/Understanding_citizenscience_species_segmentation/split_images_into_leaves_trunks_leaves_uncertain"
checkpoint_dir = "E:/Santosh_master_thesis/Understanding_citizenscience_species_segmentation/Checkpoints_tree_organs"
os.makedirs(checkpoint_dir, exist_ok=True)
batch_size = 16
image_size = 512
num_epochs = 30
patience = 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 42

# Enforce exactly this many TRAIN samples per class after the 80/20 split
TARGET_TRAIN_PER_CLASS = 4000

# ======= Transforms (Fix: RandomErasing after ToTensor) =======
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(image_size, scale=(0.6, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(0.4, 0.4, 0.4),
    transforms.RandomApply([transforms.GaussianBlur(3)], p=0.2),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.3),  # needs a tensor
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize(int(image_size * 1.1)),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ======= Dataset =======
class RecursiveImageFolder(datasets.ImageFolder):
    def __init__(self, root, transform=None):
        super().__init__(root, transform)
        logger.info(f"Class mapping: {self.class_to_idx}")
        counts = Counter([s[1] for s in self.samples])
        logger.info(f"Samples per class: {counts}")

# ======= Stratified Split + enforce exactly 4000/train/class =======
def get_data_loaders(data_dir, batch_size):
    rng = np.random.default_rng(seed)

    dataset = RecursiveImageFolder(data_dir, transform=None)
    targets = [s[1] for s in dataset.samples]

    # 80/20 stratified split
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_idx, val_idx = next(sss.split(np.zeros(len(targets)), targets))

    # Log original split totals and class counts
    train_counts_orig = Counter([targets[i] for i in train_idx])
    val_counts = Counter([targets[i] for i in val_idx])
    logger.info(f"Split totals -> Train: {len(train_idx)} | Val: {len(val_idx)}")
    logger.info(f"Train class counts (orig): {train_counts_orig}")
    logger.info(f"Val class counts:          {val_counts}")

    # Build per-class index buckets for TRAIN
    class_to_indices = {}
    for i in train_idx:
        c = targets[i]
        class_to_indices.setdefault(c, []).append(i)

    # For each class: if >TARGET → downsample to TARGET; if <TARGET → duplicate to TARGET
    expanded_train_idx = []
    for c in range(len(dataset.classes)):
        idxs = class_to_indices.get(c, [])
        if len(idxs) == 0:
            logger.warning(f"Class {c} has 0 samples in TRAIN split; skipping enforcement for this class.")
            continue
        if len(idxs) > TARGET_TRAIN_PER_CLASS:
            sel = rng.choice(idxs, size=TARGET_TRAIN_PER_CLASS, replace=False).tolist()
            expanded = sel
            logger.info(f"Class {c}: downsampled {len(idxs)} -> {len(expanded)}")
        elif len(idxs) < TARGET_TRAIN_PER_CLASS:
            extras = rng.choice(idxs, size=TARGET_TRAIN_PER_CLASS - len(idxs), replace=True).tolist()
            expanded = idxs + extras
            logger.info(f"Class {c}: upsampled {len(idxs)} -> {len(expanded)}")
        else:
            expanded = idxs
            logger.info(f"Class {c}: already {len(expanded)}")
        expanded_train_idx.extend(expanded)

    rng.shuffle(expanded_train_idx)
    train_counts_final = Counter([targets[i] for i in expanded_train_idx])
    logger.info(f"Train class counts (final, exact {TARGET_TRAIN_PER_CLASS} each): {train_counts_final}")
    logger.info(f"Totals (final) -> Train: {len(expanded_train_idx)} | Val: {len(val_idx)}")

    # Build train/val datasets with transformed samples
    train_set = RecursiveImageFolder(data_dir, transform=train_transform)
    val_set = RecursiveImageFolder(data_dir, transform=val_transform)
    train_set.samples = [dataset.samples[i] for i in expanded_train_idx]
    val_set.samples = [dataset.samples[i] for i in val_idx]

    # DataLoaders
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4)
    return train_loader, val_loader, dataset, train_idx, val_idx

# ======= Training Function =======
def train_model(model, criterion, optimizer, scheduler, train_loader, val_loader, num_epochs, writer, root_out_dir, class_names):
    best_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())
    best_improve_idx = 0  # counts validation-loss improvements
    epochs_no_improve = 0

    models_dir = os.path.join(root_out_dir, "All_Epoch_Models")
    stats_dir = os.path.join(root_out_dir, "Training_Stats")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(stats_dir, exist_ok=True)

    hist = {"epoch": [], "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch+1}/{num_epochs}")
        model.train()
        train_loss, correct, total = 0.0, 0, 0

        for inputs, labels in tqdm(train_loader, desc=f"Train {epoch+1}"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss += loss.item() * inputs.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total if total else 0.0
        train_loss = train_loss / total if total else 0.0

        # ======= Validation =======
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        y_true, y_pred = [], []

        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Val {epoch+1}"):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                preds = outputs.argmax(1)

                val_loss += loss.item() * inputs.size(0)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                y_true.extend(labels.detach().cpu().numpy())
                y_pred.extend(preds.detach().cpu().numpy())

        val_acc = val_correct / val_total if val_total else 0.0
        val_loss = val_loss / val_total if val_total else 0.0

        hist["epoch"].append(epoch + 1)
        hist["train_loss"].append(train_loss)
        hist["train_acc"].append(train_acc)
        hist["val_loss"].append(val_loss)
        hist["val_acc"].append(val_acc)

        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Loss/Val", val_loss, epoch)
        writer.add_scalar("Acc/Train", train_acc, epoch)
        writer.add_scalar("Acc/Val", val_acc, epoch)

        f1_macro = f1_score(y_true, y_pred, average="macro")
        writer.add_scalar("F1_macro/Val", f1_macro, epoch)

        logger.info(f"Epoch {epoch+1}: TL={train_loss:.4f} TA={train_acc:.4f} | VL={val_loss:.4f} VA={val_acc:.4f} | F1={f1_macro:.4f}")

        # Save model for this epoch with both loss and accuracy in filename
        epoch_ckpt = os.path.join(
            models_dir,
            f"epoch_{epoch+1:02d}_tl{train_loss:.4f}_ta{train_acc:.4f}_vl{val_loss:.4f}_va{val_acc:.4f}.pth"
        )
        torch.save(model.state_dict(), epoch_ckpt)

        # Save an improvement snapshot (only when validation loss improves), to checkpoint root
        if val_loss < best_loss - 1e-12:
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            imp_name = f"best_model_{best_improve_idx}_{best_loss:.2f}.pth"
            torch.save(model.state_dict(), os.path.join(root_out_dir, imp_name))
            best_improve_idx += 1
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

    # Save classification report and confusion matrix (last-epoch preds)
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    with open(os.path.join(stats_dir, "classification_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
    df_cm.to_csv(os.path.join(stats_dir, "confusion_matrix.png"))

    with np.errstate(all='ignore'):
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_norm = np.divide(cm.astype(np.float64), row_sums, out=np.zeros_like(cm, dtype=np.float64), where=row_sums != 0)
    pd.DataFrame(cm_norm, index=class_names, columns=class_names)\
        .to_csv(os.path.join(stats_dir, "confusion_matrix_normalized.png"))

    # Save training curves
    plt.figure()
    plt.plot(hist["epoch"], smooth_curve(hist["train_loss"]), label="Train Loss")
    plt.plot(hist["epoch"], smooth_curve(hist["val_loss"]), label="Val Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(stats_dir, "loss_curve.png")); plt.close()

    plt.figure()
    plt.plot(hist["epoch"], hist["train_acc"], label="Train Acc")
    plt.plot(hist["epoch"], hist["val_acc"], label="Val Acc")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(stats_dir, "accuracy_curve.png")); plt.close()

    # Save summary
    summary = {
        "best_val_loss": float(best_loss),
        "epochs_trained": len(hist["epoch"]),
        "f1_macro_last_epoch": float(f1_macro)
    }
    with open(os.path.join(stats_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    return model

# ======= Main =======
def main():
    writer = SummaryWriter(log_dir=os.path.join(checkpoint_dir, "Training_Stats", "tensorboard"))
    train_loader, val_loader, dataset, train_idx, val_idx = get_data_loaders(data_path, batch_size)
    class_names = dataset.classes

    # Compute class weights from TRAIN labels (pre-equalization)
    y_train = [dataset.samples[i][1] for i in train_idx]
    classes = np.unique(y_train)
    class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

    model = models.efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(nn.Dropout(0.4), nn.Linear(in_features, len(class_names)))
    model.to(device)

    criterion = nn.CrossEntropyLoss(weight=None, label_smoothing=0.1)
    optimizer = AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=5e-3, steps_per_epoch=len(train_loader),
        epochs=num_epochs, pct_start=0.2, final_div_factor=1e3
    )

    train_model(model, criterion, optimizer, scheduler, train_loader, val_loader, num_epochs, writer, checkpoint_dir, class_names)
    writer.close()

if __name__ == "__main__":
    main()
    logger.info("Training complete. Checkpoints and stats saved in: %s", checkpoint_dir)

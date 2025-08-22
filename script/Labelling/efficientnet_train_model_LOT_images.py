from tqdm import tqdm  # Make sure this is imported at the top
import os
import copy
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from collections import Counter
from typing import Optional
import torch.optim as optim
from torch.utils.data.sampler import Sampler
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, SubsetRandomSampler, Dataset
from torchvision import datasets, models, transforms
from torchvision.models import EfficientNet_V2_S_Weights
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from torch.utils.tensorboard import SummaryWriter
from PIL import Image, ImageFile, UnidentifiedImageError


ImageFile.LOAD_TRUNCATED_IMAGES = True
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===================== Config =====================
data_path = "E:/Santosh_master_thesis/LOT_all_images_labeled"
checkpoint_path = "./Checkpoints_labeled_LOT"
os.makedirs(checkpoint_path, exist_ok=True)  # ensures directory exists

batch_size = 16
image_size = 640  # Adjust based on your dataset
num_img_per_class = 3000
num_epochs = 30
patience = 6  # Early stopping patience
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training options
best_selection = "loss"  # Always pick best checkpoint by lowest Val loss
use_focal_loss = True
focal_gamma = 2.0
use_mixup = True  # Use mixup augmentation
mixup_alpha = 0.2
use_ema = True
ema_decay = 0.999
use_tta_val = True  # TTA during validation/eval
apply_others_extra_aug = True  # stronger aug only for 'Others'

# Augmentation
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(image_size, scale=(0.5, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.RandomAffine(degrees=15, translate=(
        0.1, 0.1), scale=(0.8, 1.2), shear=10),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
    transforms.GaussianBlur(3, sigma=(0.1, 2.0)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    transforms.RandomErasing(p=0.3, value='random')
])

val_transform = transforms.Compose([
    transforms.Resize(int(image_size * 1.1)),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

# ===================== Data Loading =====================


class RecursiveImageFolder(Dataset):
    def __init__(self, root, transform=None, label_extra_transforms=None):
        self.samples = []
        self.class_to_idx = {}
        self.transform = transform
        self.label_extra_transforms = label_extra_transforms or {}

        for idx, class_name in enumerate(sorted(os.listdir(root))):
            class_path = os.path.join(root, class_name)
            if not os.path.isdir(class_path):
                continue
            self.class_to_idx[class_name] = idx
            for dirpath, dirnames, filenames in os.walk(class_path):
                dirnames.sort()
                filenames.sort()
                for fname in filenames:
                    if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                        self.samples.append(
                            (os.path.join(dirpath, fname), idx))
        self.classes = list(self.class_to_idx.keys())
        print("Class to idx mapping:", self.class_to_idx)
        print("Images per class:", Counter(
            [label for _, label in self.samples]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        max_attempts = 10
        attempts = 0

        while attempts < max_attempts:
            path, label = self.samples[idx]
            try:
                with Image.open(path) as img:
                    img = img.convert("RGB")
                    # optional class-aware extra transforms before base transform
                    if label in self.label_extra_transforms:
                        img = self.label_extra_transforms[label](img)
                    if self.transform:
                        img = self.transform(img)
                    return img, label
            except (OSError, UnidentifiedImageError) as e:
                logger.warning(f"Skipping corrupted image: {path} ({e})")
                # Try next image
                idx = (idx + 1) % len(self.samples)
                attempts += 1

        raise RuntimeError(f"Too many corrupted images around index {idx}")


class BalancedBatchSampler(Sampler):
    """Yields indices so each batch has the same number of samples from each class."""

    def __init__(self, indices_per_class, batch_size):
        self.labels = sorted(indices_per_class.keys())
        self.per_class = len(self.labels)
        assert batch_size % self.per_class == 0, "batch_size must be divisible by #classes"
        self.k = batch_size // self.per_class
        # Shuffle pools per epoch
        self.pools = {c: np.random.permutation(
            v).tolist() for c, v in indices_per_class.items()}
        self.length = min(len(v) for v in indices_per_class.values())
        self.batch_count = self.length // self.k

    def __iter__(self):
        # refresh order
        self.pools = {c: np.random.permutation(
            v).tolist() for c, v in self.pools.items()}
        ptrs = {c: 0 for c in self.labels}
        for _ in range(self.batch_count):
            batch = []
            for c in self.labels:
                start = ptrs[c]
                end = start + self.k
                batch.extend(self.pools[c][start:end])
                ptrs[c] = end
            np.random.shuffle(batch)
            yield from batch

    def __len__(self):
        return self.batch_count * self.per_class * self.k


def get_data_loaders(data_dir, batch_size, num_img_per_class, train_transform, val_transform):
    """
    Build loaders with exactly `num_img_per_class` samples per class.
    - If a class has more than target, cap to target (random without replacement).
    - If a class has fewer than target, oversample with replacement to reach target.
      Augmentations in `train_transform` will add variety during training.
    Split per class into 80% train / 20% val to preserve balance.
    """
    full_dataset = RecursiveImageFolder(root=data_dir)

    if num_img_per_class is None:
        raise ValueError(
            "num_img_per_class must be set (e.g., 3000) to enforce per-class counts.")

    rng = np.random.default_rng()
    per_class_train_indices = []
    per_class_val_indices = []

    # Build dicts: class_idx -> list of indices
    class_indices_by_label = {
        class_idx: [i for i, (_, label) in enumerate(
            full_dataset.samples) if label == class_idx]
        for class_idx in range(len(full_dataset.class_to_idx))
    }

    for class_idx, class_indices in class_indices_by_label.items():
        n = len(class_indices)
        target = int(num_img_per_class)

        if n == 0:
            logger.warning(
                f"Class index {class_idx} has 0 images; cannot sample.")
            continue

        train_target = int(0.8 * target)
        val_target = target - train_target

        if n >= target:
            # First cap to target without replacement, then split into train/val
            selected = rng.choice(class_indices, size=target, replace=False)
            rng.shuffle(selected)
            class_train_idx = selected[:train_target].tolist()
            class_val_idx = selected[train_target:].tolist()
        else:
            # Split UNIQUE indices into 80/20 first to avoid leakage
            class_indices_arr = np.array(class_indices, dtype=int)
            rng.shuffle(class_indices_arr)
            split_point = int(0.8 * n)
            base_train = class_indices_arr[:split_point]
            base_val = class_indices_arr[split_point:]

            # Oversample WITHIN each split to reach targets
            if len(base_train) >= train_target:
                class_train_idx = rng.choice(
                    base_train, size=train_target, replace=False).tolist()
            else:
                deficit = train_target - len(base_train)
                dup = rng.choice(base_train, size=deficit, replace=True) if len(
                    base_train) > 0 else np.array([], dtype=int)
                class_train_idx = np.concatenate([base_train, dup]).tolist()

            if len(base_val) >= val_target:
                class_val_idx = rng.choice(
                    base_val, size=val_target, replace=False).tolist()
            else:
                deficit = val_target - len(base_val)
                dup = rng.choice(base_val, size=deficit, replace=True) if len(
                    base_val) > 0 else np.array([], dtype=int)
                class_val_idx = np.concatenate([base_val, dup]).tolist()

        per_class_train_indices.extend(class_train_idx)
        per_class_val_indices.extend(class_val_idx)

    train_labels = [full_dataset.samples[i][1]
                    for i in per_class_train_indices]
    val_labels = [full_dataset.samples[i][1] for i in per_class_val_indices]
    print("Train class counts (target ~80% of",
          num_img_per_class, "):", Counter(train_labels))
    print("Val class counts (target ~20% of",
          num_img_per_class, "):", Counter(val_labels))

    # Optional: extra augmentation for 'Others' class during training only
    label_extra = {}
    if apply_others_extra_aug:
        # Try to find 'Others' class index (case-sensitive variants)
        others_key = None
        for k in full_dataset.class_to_idx.keys():
            if k.lower() == 'others':
                others_key = k
                break
        if others_key is not None:
            others_idx = full_dataset.class_to_idx[others_key]
            extra_aug = transforms.Compose([
                transforms.RandomApply(
                    [transforms.RandomResizedCrop(image_size, scale=(0.5, 1.0))], p=0.2),
                transforms.RandomApply(
                    [transforms.GaussianBlur(5, sigma=(0.4, 1.6))], p=0.25),
                transforms.ColorJitter(
                    brightness=0.4, contrast=0.5, saturation=0.4, hue=0.04),
                transforms.RandomPerspective(distortion_scale=0.25, p=0.15),
            ])

            label_extra[others_idx] = extra_aug

    train_dataset = RecursiveImageFolder(
        root=data_dir, transform=train_transform, label_extra_transforms=label_extra)
    val_dataset = RecursiveImageFolder(root=data_dir, transform=val_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=SubsetRandomSampler(per_class_train_indices),
        num_workers=8,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=SubsetRandomSampler(per_class_val_indices),
        num_workers=8,
    )

    return train_loader, val_loader, full_dataset


# ===================== Training =====================


def _plot_confusion_matrix(cm, class_names, save_path, normalize=False, title_suffix=""):
    import itertools
    if normalize:
        with np.errstate(all='ignore'):
            cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
            cm_to_show = np.nan_to_num(cm_norm)
    else:
        cm_to_show = cm
    plt.figure(figsize=(6, 5))
    plt.imshow(cm_to_show, interpolation='nearest', cmap=plt.cm.Blues)
    title = f"Confusion Matrix{title_suffix}"
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha='right')
    plt.yticks(tick_marks, class_names)
    fmt = '.2f' if normalize else 'd'
    thresh = cm_to_show.max() / 2.0 if cm_to_show.size > 0 else 0
    for i, j in itertools.product(range(cm_to_show.shape[0]), range(cm_to_show.shape[1])):
        plt.text(j, i, format(cm_to_show[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm_to_show[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight: Optional[torch.Tensor] = None,
                 alpha: Optional[torch.Tensor] = None, reduction: str = 'mean'):
        super().__init__()
        self.gamma = gamma
        self.weight = weight  # class weights (like CE weights)
        self.alpha = alpha    # per-class scaling (focus class)
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = nn.functional.cross_entropy(
            logits, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce)  # prob of correct class
        loss = (1 - pt) ** self.gamma * ce
        if self.alpha is not None:
            # multiply per-sample by alpha[class]
            a = self.alpha.to(logits.device)[targets]
            loss = loss * a
        return loss.mean() if self.reduction == 'mean' else loss.sum()


@torch.no_grad()
def ema_update(ema_model: nn.Module, model: nn.Module, decay: float):
    ema_params = dict(ema_model.named_parameters())
    for n, p in model.named_parameters():
        if n in ema_params:
            ema_params[n].data.mul_(decay).add_(p.data, alpha=1.0 - decay)


def train_model(model, criterion, optimizer, scheduler, train_loader, val_loader, num_epochs, writer, class_names, stats_dir):
    best_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())
    best_epoch = -1
    best_metrics = {}
    epochs_no_improve = 0
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    val_f1_macros = []

    # Prepare output dirs/files
    os.makedirs(stats_dir, exist_ok=True)
    all_epoch_dir = os.path.join(checkpoint_path, "all_epoch_models")
    os.makedirs(all_epoch_dir, exist_ok=True)
    metrics_csv_path = os.path.join(stats_dir, "epoch_metrics.csv")
    if not os.path.exists(metrics_csv_path):
        with open(metrics_csv_path, 'w', encoding='utf-8') as f:
            f.write(
                "epoch,train_loss,train_acc,val_loss,val_acc,val_f1,best_so_far\n")

    # Setup EMA model if enabled
    ema_model = None
    if use_ema:
        ema_model = copy.deepcopy(model)
        for p in ema_model.parameters():
            p.requires_grad_(False)

    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch+1}/{num_epochs}")

        # Train
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for inputs, labels in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()  # Reset gradients

            # Optional MixUp
            if use_mixup:
                # create shuffled targets
                idx = torch.randperm(inputs.size(0), device=inputs.device)
                lam = np.random.beta(mixup_alpha, mixup_alpha)
                mixed = lam * inputs + (1 - lam) * inputs[idx]
                outputs = model(mixed)
                loss = lam * criterion(outputs, labels) + \
                    (1 - lam) * criterion(outputs, labels[idx])
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # OneCycleLR expects per-batch stepping
            try:
                scheduler.step()
            except Exception:
                pass

            # EMA update
            if ema_model is not None:
                ema_update(ema_model, model, ema_decay)

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        print(
            f"Train Loss: {train_loss:.4f} | Train Accuracy: {train_acc:.4f}")

        # Validate
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        all_preds, all_labels_list = [], []
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f"Validation Epoch {epoch+1}")
            for inputs, labels in val_bar:
                inputs, labels = inputs.to(device), labels.to(device)
                eval_model = ema_model if ema_model is not None else model
                if use_tta_val:
                    # average logits across simple flips
                    logits = eval_model(inputs)
                    logits += eval_model(torch.flip(inputs, dims=[3]))
                    logits += eval_model(torch.flip(inputs, dims=[2]))
                    logits += eval_model(torch.flip(inputs, dims=[2, 3]))
                    outputs = logits / 4.0
                else:
                    outputs = eval_model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                all_preds.extend(preds.detach().cpu().tolist())
                all_labels_list.extend(labels.detach().cpu().tolist())

                # Update progress bar with running metrics
                cur_loss = val_loss / max(1, val_total)
                cur_acc = val_correct / max(1, val_total)
                try:
                    cur_f1 = f1_score(all_labels_list, all_preds,
                                      average='macro') if val_total > 0 else 0.0
                except Exception:
                    cur_f1 = 0.0
                val_bar.set_postfix({
                    'loss': f"{cur_loss:.4f}",
                    'acc': f"{cur_acc:.4f}",
                    'f1': f"{cur_f1:.4f}"
                })

        val_loss /= val_total
        val_acc = val_correct / val_total
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        # For ReduceLROnPlateau only; OneCycleLR stepped per-batch above
        # scheduler.step(val_loss)
        val_f1 = f1_score(all_labels_list, all_preds,
                          average='macro') if val_total > 0 else 0.0
        val_f1_macros.append(val_f1)
        print(f"Val Loss: {val_loss:.4f} | Val Accuracy: {val_acc:.4f}")
        print(f"Val F1-macro: {val_f1:.4f}")
        # Log to TensorBoard
        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Loss/Val", val_loss, epoch)
        writer.add_scalar("Acc/Train", train_acc, epoch)
        writer.add_scalar("Acc/Val", val_acc, epoch)
        writer.add_scalar("F1/Val_macro", val_f1, epoch)

        logger.info(
            f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1_macro: {val_f1:.4f}")

        # Save per-epoch checkpoint in a separate folder
        epoch_ckpt_name = f"model_epoch{epoch}_VL{val_loss:.2f}_VA{val_acc:.2f}_F1{val_f1:.2f}.pth"
        torch.save(model.state_dict(), os.path.join(
            all_epoch_dir, epoch_ckpt_name))

        # Append per-epoch metrics to CSV
        with open(metrics_csv_path, 'a', encoding='utf-8') as f:
            f.write(
                f"{epoch},{train_loss:.6f},{train_acc:.6f},{val_loss:.6f},{val_acc:.6f},{val_f1:.6f},")

        is_best = val_loss < best_loss

        if is_best:
            best_loss = val_loss
            # Keep best weights in memory (EMA if enabled)
            if ema_model is not None:
                best_model_wts = copy.deepcopy(ema_model.state_dict())
            else:
                best_model_wts = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
            # Name best model checkpoint with only epoch number and validation loss
            best_ckpt_name = f"best_model_epoch{epoch}_VL{best_loss:.2f}.pth"
            best_path = os.path.join(checkpoint_path, best_ckpt_name)
            # Save EMA weights if enabled; else current model
            if ema_model is not None:
                torch.save(ema_model.state_dict(), best_path)
            else:
                torch.save(model.state_dict(), best_path)
            # Do not save per-epoch confusion matrices; will compute once at end for the final best model.
            best_epoch = epoch
            best_metrics = {
                'epoch': int(epoch),
                'val_loss': float(best_loss),
                'val_acc': float(val_acc),
                'val_f1_macro': float(val_f1),
                'best_model_path': best_path,
            }
            # Mark best_so_far in CSV
            with open(metrics_csv_path, 'a', encoding='utf-8') as f:
                f.write("true\n")
        else:
            epochs_no_improve += 1
            # Not best this epoch -> close CSV line before any potential break
            with open(metrics_csv_path, 'a', encoding='utf-8') as f:
                f.write("false\n")
            if epochs_no_improve >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

    model.load_state_dict(best_model_wts)

    # Save loss curve
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(stats_dir, "loss_curve.png"))
    plt.close()

    # Save loss curve with best epoch marker
    if best_epoch >= 0:
        plt.plot(train_losses, label="Train Loss")
        plt.plot(val_losses, label="Val Loss")
        plt.axvline(best_epoch, color='red', linestyle='--',
                    label=f'Best {best_epoch}')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(stats_dir, "loss_curve_best.png"))
        plt.close()

    # Save accuracy curve
    plt.plot(train_accuracies, label="Train Acc")
    plt.plot(val_accuracies, label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(os.path.join(stats_dir, "accuracy_curve.png"))
    plt.close()

    # Save accuracy curve with best epoch marker
    if best_epoch >= 0:
        plt.plot(train_accuracies, label="Train Acc")
        plt.plot(val_accuracies, label="Val Acc")
        plt.axvline(best_epoch, color='red', linestyle='--',
                    label=f'Best {best_epoch}')
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.savefig(os.path.join(stats_dir, "accuracy_curve_best.png"))
        plt.close()

    # Save F1 macro curve
    if val_f1_macros:
        plt.plot(val_f1_macros, label="Val F1-macro")
        plt.xlabel("Epoch")
        plt.ylabel("F1-macro")
        plt.legend()
        plt.savefig(os.path.join(stats_dir, "f1_macro_curve.png"))
        plt.close()

    # Only one F1 curve saved above; no extra "best" variant

    # After training, compute confusion matrix and classification report ONCE using the final best model
    model.eval()
    all_preds, all_labels_list = [], []
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="Final Eval (best model)"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.detach().cpu().tolist())
            all_labels_list.extend(labels.detach().cpu().tolist())

    cm = confusion_matrix(all_labels_list, all_preds,
                          labels=list(range(len(class_names))))
    _plot_confusion_matrix(cm, class_names, os.path.join(
        stats_dir, "confusion_matrix_best.png"), normalize=False, title_suffix=" (Counts)")
    _plot_confusion_matrix(cm, class_names, os.path.join(
        stats_dir, "confusion_matrix_best_norm.png"), normalize=True, title_suffix=" (Normalized)")
    rep = classification_report(
        all_labels_list, all_preds, target_names=class_names, output_dict=True, zero_division=0)
    others_recall = rep.get('Others', {}).get('recall', float('nan'))
    writer.add_scalar("Recall/Val_Others", others_recall, epoch)

    # Persist the final best classification report as JSON for downstream analysis
    import json
    with open(os.path.join(stats_dir, "classification_report_best.json"), 'w', encoding='utf-8') as f:
        json.dump(rep, f, indent=2)

    # Also write a compact per-class CSV with precision, recall, f1-score, support
    csv_path = os.path.join(stats_dir, "per_class_metrics_best.csv")
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write("class,precision,recall,f1-score,support\n")
        for k, v in rep.items():
            if k in ("accuracy", "macro avg", "weighted avg"):
                continue
            if isinstance(v, dict):
                f.write(
                    f"{k},{v.get('precision','')},{v.get('recall','')},{v.get('f1-score','')},{v.get('support','')}\n")

    # Persist best metrics summary
    if best_metrics:
        import json
        with open(os.path.join(stats_dir, "best_metrics.json"), 'w') as f:
            json.dump(best_metrics, f, indent=2)

    return model
# ===================== Main =====================


def main():
    log_dir = os.path.join(
        "runs", "run_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    writer = SummaryWriter(log_dir)

    train_loader, val_loader, dataset = get_data_loaders(
        data_path, batch_size, num_img_per_class, train_transform, val_transform
    )

    all_labels = [dataset.samples[i][1] for i in train_loader.sampler.indices]
    n_classes = len(dataset.classes)
    if len(all_labels) == 0:
        logger.warning(
            "No training labels found when computing class weights; using uniform weights.")
        class_weights_tensor = torch.ones(
            n_classes, dtype=torch.float).to(device)
    else:
        classes_idx = np.arange(n_classes, dtype=int)
        class_weights = compute_class_weight(
            class_weight="balanced", classes=classes_idx, y=all_labels)
        class_weights_tensor = torch.tensor(
            class_weights, dtype=torch.float).to(device)

    num_classes = len(dataset.classes)
    model = models.efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.6),
        nn.Linear(in_features, num_classes)
    )
    model.to(device)

    # Build alpha: >1.0 for Others, =1.0 for others
    alpha_vec = torch.ones(n_classes, dtype=torch.float)
    # find 'Others' class index
    others_idx = None
    for name, idx in dataset.class_to_idx.items():
        if name.lower() == 'others':
            others_idx = idx
            break
    if others_idx is not None:
        alpha_vec[others_idx] = 1.4  # try 1.15–1.35; start modest

    if use_focal_loss:
        criterion = FocalLoss(gamma=focal_gamma,
                              weight=class_weights_tensor,  # from compute_class_weight
                              alpha=alpha_vec.to(device))
    else:
        criterion = nn.CrossEntropyLoss(
            weight=class_weights_tensor, label_smoothing=0.05)

    optimizer = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=5e-4)
    scheduler = OneCycleLR(
        optimizer, max_lr=2e-4, steps_per_epoch=len(train_loader),
        epochs=num_epochs, pct_start=0.2, final_div_factor=1e3
    )
    stats_dir = os.path.join(checkpoint_path, "training_stats")
    train_model(model, criterion, optimizer, scheduler,
                train_loader, val_loader, num_epochs, writer, dataset.classes, stats_dir)
    writer.close()


if __name__ == '__main__':
    main()

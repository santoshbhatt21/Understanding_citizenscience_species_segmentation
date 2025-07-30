
import os
import copy
import json
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, SubsetRandomSampler, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, models, transforms
from torchvision.models import EfficientNet_V2_S_Weights
from PIL import Image, ImageFile, UnidentifiedImageError
import seaborn as sns

ImageFile.LOAD_TRUNCATED_IMAGES = True
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
torch.backends.cudnn.benchmark = True

# ============ Config ============
data_path = "E:/Santosh_master_thesis/Understanding_citizenscience_species_segmentation/flat_labeled_data"
checkpoint_path = "./checkpoints_leaves_trunks_others"
all_epoch_model_path = os.path.join(checkpoint_path, "all_epoch_models")
stats_path = os.path.join(checkpoint_path, "training_stats")
os.makedirs(checkpoint_path, exist_ok=True)
os.makedirs(all_epoch_model_path, exist_ok=True)
os.makedirs(stats_path, exist_ok=True)

batch_size = 16
image_size = 512
num_img_per_class = 4000
num_classes = 3
num_epochs = 20
patience = 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============ Transforms ============
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(image_size, scale=(0.6, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(0.4, 0.4, 0.4),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

val_transform = transforms.Compose([
    transforms.Resize(image_size + 32),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

class RecursiveImageFolder(Dataset):
    def __init__(self, root, transform=None):
        self.samples = []
        self.class_to_idx = {}
        self.transform = transform
        for idx, class_name in enumerate(sorted(os.listdir(root))):
            class_path = os.path.join(root, class_name)
            if not os.path.isdir(class_path): continue
            self.class_to_idx[class_name] = idx
            for dirpath, _, filenames in os.walk(class_path):
                for fname in filenames:
                    if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                        self.samples.append((os.path.join(dirpath, fname), idx))
        self.classes = list(self.class_to_idx.keys())

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        for _ in range(10):
            try:
                with Image.open(path) as img:
                    img = img.convert("RGB")
                    return self.transform(img), label if self.transform else (img, label)
            except (OSError, UnidentifiedImageError):
                idx = (idx + 1) % len(self.samples)
        raise RuntimeError(f"Corrupted image at index {idx}")

def get_data_loaders(data_dir, batch_size, num_img_per_class, train_transform, val_transform):
    dataset = RecursiveImageFolder(root=data_dir)
    indices = []
    for class_idx in range(len(dataset.class_to_idx)):
        class_indices = [i for i, (_, label) in enumerate(dataset.samples) if label == class_idx]
        sampled = np.random.choice(class_indices, min(num_img_per_class, len(class_indices)), replace=False)
        indices.extend(sampled)
    np.random.shuffle(indices)
    train_size = int(0.7 * len(indices))
    train_idx, val_idx = indices[:train_size], indices[train_size:]
    train_dataset = RecursiveImageFolder(root=data_dir, transform=train_transform)
    val_dataset = RecursiveImageFolder(root=data_dir, transform=val_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=SubsetRandomSampler(train_idx), num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, sampler=SubsetRandomSampler(val_idx), num_workers=4)
    return train_loader, val_loader, dataset

def smooth(data, weight=0.85):
    smoothed = []
    last = data[0]
    for point in data:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

def train_model(model, criterion, optimizer, scheduler, train_loader, val_loader, num_epochs, writer):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')
    epochs_no_improve = 0
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch+1}/{num_epochs}")
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for inputs, labels in tqdm.tqdm(train_loader, desc="Training"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

        train_losses.append(running_loss / total)
        train_accuracies.append(correct / total)

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for inputs, labels in tqdm.tqdm(val_loader, desc="Validation"):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                preds = outputs.argmax(1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_losses.append(val_loss / val_total)
        val_accuracies.append(val_correct / val_total)
        logger.info(f"Val Loss: {val_losses[-1]:.4f}, Val Acc: {val_accuracies[-1]:.4f}")
        scheduler.step(val_losses[-1])

        for i, param_group in enumerate(optimizer.param_groups):
            writer.add_scalar(f"LR/Group_{i}", param_group['lr'], epoch)

        writer.add_scalar("Loss/Train", train_losses[-1], epoch)
        writer.add_scalar("Loss/Val", val_losses[-1], epoch)
        writer.add_scalar("Acc/Train", train_accuracies[-1], epoch)
        writer.add_scalar("Acc/Val", val_accuracies[-1], epoch)

        if val_losses[-1] < best_loss:
            best_loss = val_losses[-1]
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), os.path.join(checkpoint_path, f"best_model_{epoch}_{best_loss:.2f}.pth"))
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logger.info("Early stopping.")
                break

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_losses[-1],
            'val_loss': val_losses[-1],
            'train_acc': train_accuracies[-1],
            'val_acc': val_accuracies[-1]
        }, os.path.join(all_epoch_model_path, f"epoch_{epoch}.pth"))

    model.load_state_dict(best_model_wts)

    with open(os.path.join(stats_path, "best_epoch.txt"), "w") as f:
        f.write(f"Best validation loss: {best_loss:.4f}\n")
        f.write(f"Train Acc: {train_accuracies[train_losses.index(min(train_losses))]:.4f}\n")
        f.write(f"Val Acc: {val_accuracies[val_losses.index(min(val_losses))]:.4f}\n")

    report = classification_report(all_labels, all_preds, output_dict=True)
    with open(os.path.join(stats_path, "classification_report.json"), 'w') as f:
        json.dump(report, f, indent=4)

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(os.path.join(stats_path, "confusion_matrix.png"))
    plt.close()

    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(smooth(train_losses), label="Train Loss")
    plt.plot(smooth(val_losses), label="Val Loss")
    plt.title("Smoothed Loss")
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(smooth(train_accuracies), label="Train Acc")
    plt.plot(smooth(val_accuracies), label="Val Acc")
    plt.title("Smoothed Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(stats_path, "train_val_curves.png"))
    plt.close()

    return model

def main():
    writer = SummaryWriter(log_dir=os.path.join("runs", datetime.now().strftime("run_%Y%m%d_%H%M%S")))
    train_loader, val_loader, dataset = get_data_loaders(data_path, batch_size, num_img_per_class, train_transform, val_transform)
    all_labels = [dataset.samples[i][1] for i in train_loader.sampler.indices]
    class_weights = compute_class_weight('balanced', classes=np.unique(all_labels), y=all_labels)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)

    model = models.efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(nn.Dropout(0.4), nn.Linear(in_features, num_classes))
    model.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor, label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=5e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

    train_model(model, criterion, optimizer, scheduler, train_loader, val_loader, num_epochs, writer)
    writer.close()

if __name__ == '__main__':
    main()

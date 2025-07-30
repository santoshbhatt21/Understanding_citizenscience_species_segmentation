# efficientnet_train_fix.py (Extended with Evaluation + Full Logging)
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
from collections import Counter
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

# ===================== Config =====================
data_path = "E:/Santosh_master_thesis/Understanding_citizenscience_species_segmentation/flat_labeled_data"
checkpoint_path = "./Checkpoints_using_tree_species_classification_code"
all_epoch_model_path = os.path.join(checkpoint_path, "All_Epoch_Models")
stats_path = os.path.join(checkpoint_path, "Training_Stats")
os.makedirs(checkpoint_path, exist_ok=True)
os.makedirs(all_epoch_model_path, exist_ok=True)
os.makedirs(stats_path, exist_ok=True)

batch_size = 16  # Adjust based on your GPU memory
image_size = 512
num_img_per_class = 4000
num_classes = 3  # Leaves, Trunks, Others
num_epochs = 50
patience = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    transforms.RandomErasing(p=0.3, value='random')
])


# ...existing imports and config...

def ten_crop_to_tensor(crops):
    return torch.stack([transforms.ToTensor()(crop) for crop in crops])

def ten_crop_normalize(crops):
    norm = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    return torch.stack([norm(crop) for crop in crops])

val_transform = transforms.Compose([
    transforms.Resize(image_size + 32),
    transforms.TenCrop(image_size),
    transforms.Lambda(ten_crop_to_tensor),
    transforms.Lambda(ten_crop_normalize)
])


class RecursiveImageFolder(Dataset):
    def __init__(self, root, transform=None):
        self.samples = []
        self.class_to_idx = {}
        self.transform = transform
        for idx, class_name in enumerate(sorted(os.listdir(root))):
            class_path = os.path.join(root, class_name)
            if not os.path.isdir(class_path):
                continue
            self.class_to_idx[class_name] = idx
            for dirpath, _, filenames in os.walk(class_path):
                for fname in filenames:
                    if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                        self.samples.append((os.path.join(dirpath, fname), idx))
        self.classes = list(self.class_to_idx.keys())

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        for _ in range(10):
            try:
                with Image.open(path) as img:
                    img = img.convert("RGB")
                    if self.transform:
                        img = self.transform(img)
                    return img, label
            except (OSError, UnidentifiedImageError):
                idx = (idx + 1) % len(self.samples)
        raise RuntimeError(f"Corrupted image at index {idx}")

# ===================== Data Loader =====================
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
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=SubsetRandomSampler(train_idx), num_workers = min(8, os.cpu_count()), pin_memory=True)  # or try 8, 16 depending on your CPU
    val_loader = DataLoader(val_dataset, batch_size=batch_size, sampler=SubsetRandomSampler(val_idx), num_workers = min(8, os.cpu_count()))  # or try 8, 16 depending on your CPU
    return train_loader, val_loader, dataset

# ===================== Training =====================
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

        for inputs, labels in tqdm.tqdm(train_loader, desc="Train"):
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
        for inputs, labels in tqdm.tqdm(val_loader, desc="Val"):
            # Handle TenCrop: inputs shape [batch_size, 10, 3, 512, 512]
            inputs, labels = inputs.to(device), labels.to(device)
            if inputs.dim() == 5:
                bs, ncrops, c, h, w = inputs.size()
                inputs = inputs.view(-1, c, h, w)  # [batch_size*10, 3, 512, 512]
                outputs = model(inputs)  # [batch_size*10, num_classes]
                outputs = outputs.view(bs, ncrops, -1).mean(1)  # [batch_size, num_classes]
                preds = outputs.argmax(1)
                val_loss += criterion(outputs, labels).item() * bs
                val_correct += (preds == labels).sum().item()
                val_total += bs
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
            else:
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
        scheduler.step(val_losses[-1])

        # Logging
        writer.add_scalar("Loss/Train", train_losses[-1], epoch)
        writer.add_scalar("Loss/Val", val_losses[-1], epoch)
        writer.add_scalar("Acc/Train", train_accuracies[-1], epoch)
        writer.add_scalar("Acc/Val", val_accuracies[-1], epoch)

        # Save best
        if val_losses[-1] < best_loss:
            best_loss = val_losses[-1]
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), os.path.join(checkpoint_path, f"best_model_{epoch}_{best_loss:.2f}.pth"))
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logger.info("Early stopping.")
                return model

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

    # Classification report & confusion matrix
    report = classification_report(all_labels, all_preds, output_dict=True)
    with open(os.path.join(stats_path, "classification_report.json"), 'w') as f:
        json.dump(report, f, indent=4)

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.ylabel("True")
    plt.xlabel("Pred")
    plt.tight_layout()
    plt.savefig(os.path.join(stats_path, "confusion_matrix.png"))
    plt.close()

    # Loss/accuracy curves
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.legend(); plt.grid(); plt.title("Loss")
    plt.savefig(os.path.join(stats_path, "loss_curve.png")); plt.close()

    plt.plot(train_accuracies, label='Train Acc')
    plt.plot(val_accuracies, label='Val Acc')
    plt.legend(); plt.grid(); plt.title("Accuracy")
    plt.savefig(os.path.join(stats_path, "accuracy_curve.png")); plt.close()

    return model

# ===================== Main =====================
def main():
    writer = SummaryWriter(log_dir=os.path.join("runs", datetime.now().strftime("run_%Y%m%d_%H%M%S")))
    train_loader, val_loader, dataset = get_data_loaders(data_path, batch_size, num_img_per_class, train_transform, val_transform)
    all_labels = [dataset.samples[i][1] for i in train_loader.sampler.indices]
    class_weights = compute_class_weight('balanced', classes=np.unique(all_labels), y=all_labels)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)

    model = models.efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(nn.Dropout(0.6), nn.Linear(in_features, num_classes))
    model.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor, label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=5e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

    train_model(model, criterion, optimizer, scheduler, train_loader, val_loader, num_epochs, writer)
    writer.close()

if __name__ == '__main__':
    main()

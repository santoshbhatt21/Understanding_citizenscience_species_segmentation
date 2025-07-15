import os
import copy
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
import torch.nn as nn
from collections import Counter
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, SubsetRandomSampler, Dataset
from torchvision import datasets, models, transforms
from torchvision.models import EfficientNet_V2_L_Weights, EfficientNet_V2_S_Weights
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.tensorboard import SummaryWriter
from PIL import Image, ImageFile, UnidentifiedImageError
from collections import Counter

ImageFile.LOAD_TRUNCATED_IMAGES = True
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===================== Config =====================
data_path = "E:/Santosh_master_thesis/Understanding_citizenscience_species_segmentation/Data"
checkpoint_path = "./Checkpoints_4k"
os.makedirs(checkpoint_path, exist_ok=True)  # ensures directory exists

batch_size = 16
image_size = 512  # Adjust based on your dataset
num_img_per_class = 4000
num_classes = 9  # Adjust based on your dataset
num_epochs = 100
patience = 10  # Early stopping patience
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Augmentation
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(image_size, scale=(0.6, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.RandomAffine(degrees=15, translate=(0.1,0.1), scale=(0.8,1.2), shear=10),
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
                    if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                        self.samples.append((os.path.join(dirpath, fname), idx))
        self.classes = list(self.class_to_idx.keys())
        print("Class to idx mapping:", self.class_to_idx)
        print("Images per class:", Counter([label for _, label in self.samples]))

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
                    if self.transform:
                        img = self.transform(img)
                    return img, label
            except (OSError, UnidentifiedImageError) as e:
                logger.warning(f"Skipping corrupted image: {path} ({e})")
                # Try next image
                idx = (idx + 1) % len(self.samples)
                attempts += 1

        raise RuntimeError(f"Too many corrupted images around index {idx}")




def get_data_loaders(data_dir, batch_size, num_img_per_class, train_transform, val_transform):
    full_dataset = RecursiveImageFolder(root=data_dir)

    indices = []
    for class_idx in range(len(full_dataset.class_to_idx)):
        class_indices = [i for i, (_, label) in enumerate(full_dataset.samples) if label == class_idx]
        if num_img_per_class is None:
            sampled = class_indices
        else:
            sampled = np.random.choice(class_indices, min(num_img_per_class, len(class_indices)), replace=False)
        indices.extend(sampled)

    np.random.shuffle(indices)
    train_size = int(0.8 * len(indices))
    train_indices, val_indices = indices[:train_size], indices[train_size:]
    assert len(set(train_indices).intersection(set(val_indices))) == 0, "Train/Val overlap detected!"

    train_labels = [full_dataset.samples[i][1] for i in train_indices]
    val_labels = [full_dataset.samples[i][1] for i in val_indices]
    print("Train class counts:", Counter(train_labels))
    print("Val class counts:", Counter(val_labels))

    train_dataset = RecursiveImageFolder(root=data_dir, transform=train_transform)
    val_dataset = RecursiveImageFolder(root=data_dir, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=SubsetRandomSampler(train_indices), num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, sampler=SubsetRandomSampler(val_indices), num_workers=8)

    return train_loader, val_loader, full_dataset

# ===================== Training =====================
from tqdm import tqdm  # Make sure this is imported at the top

def train_model(model, criterion, optimizer, scheduler, train_loader, val_loader, num_epochs, writer):
    best_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch+1}/{num_epochs}")

        # Train
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for inputs, labels in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        # Validate
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        for inputs, labels in tqdm(val_loader, desc=f"Validation Epoch {epoch+1}"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            #scheduler.step(loss)

            val_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

        val_loss /= val_total
        val_acc = val_correct / val_total
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        #scheduler.step(val_loss)

        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Loss/Val", val_loss, epoch)
        writer.add_scalar("Acc/Train", train_acc, epoch)
        writer.add_scalar("Acc/Val", val_acc, epoch)

        logger.info(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
            torch.save(model.state_dict(), os.path.join(checkpoint_path, f"best_model_{epoch}_{best_loss:.2f}.pth"))
        else:
            epochs_no_improve += 1
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
    plt.savefig("loss_curve.png")
    plt.close()

    # Save accuracy curve
    plt.plot(train_accuracies, label="Train Acc")
    plt.plot(val_accuracies, label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("accuracy_curve.png")
    plt.close()

    return model
# ===================== Main =====================
def main():
    log_dir = os.path.join("runs", "run_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    writer = SummaryWriter(log_dir)

    train_loader, val_loader, dataset = get_data_loaders(
        data_path, batch_size, num_img_per_class, train_transform, val_transform
    )

    all_labels = [dataset.samples[i][1] for i in train_loader.sampler.indices]
    class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(all_labels), y=all_labels)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)

    num_classes = len(dataset.classes)
    model = models.efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(in_features, num_classes)
    )
    model.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=5e-4)
    scheduler = OneCycleLR(optimizer, max_lr=2e-4, epochs=num_epochs, steps_per_epoch=len(train_loader))

    train_model(model, criterion, optimizer, scheduler, train_loader, val_loader, num_epochs, writer)
    writer.close()

if __name__ == '__main__':
    main()

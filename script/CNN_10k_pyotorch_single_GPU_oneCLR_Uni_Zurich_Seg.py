import json
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, SubsetRandomSampler
import numpy as np
from tqdm import tqdm
import logging
import copy
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from sklearn.utils.class_weight import compute_class_weight

# === Setup ===
checkpoint_path = "E:/Santosh_master_thesis/Understanding_citizenscience_species_segmentation/Check_Point"
data_path = "E:/Santosh_master_thesis/Understanding_citizenscience_species_segmentation/iNaturalist"
num_img_per_class = 2000
batch_size = 32
num_epochs = 20
image_size = 256
GPU_index = 'cuda:0'
os.makedirs(checkpoint_path, exist_ok=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()

transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    #transforms.ColorJitter(),
    #transforms.RandomResizedCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    #transforms.RandomErasing(p=0.2, value='random')
])

device = torch.device(GPU_index if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(device)


# === DataLoader ===
def get_data_loaders(data_dir, batch_size, num_img_per_class, transform):
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)

    print("\nClass index mapping:")
    for name, idx in dataset.class_to_idx.items():
        print(f"  {name}: {idx}")

    indices = []
    class_to_indices = {}

    for class_idx in range(len(dataset.classes)):
        class_indices = [i for i, (_, label) in enumerate(dataset.samples) if label == class_idx]
        sampled = np.random.choice(class_indices, num_img_per_class, replace=len(class_indices) < num_img_per_class)
        indices.extend(sampled)
        class_to_indices[class_idx] = sampled

    print("\n5 sample image paths per class:")
    for class_idx, sample_ids in class_to_indices.items():
        print(f"Class: {dataset.classes[class_idx]}")
        for i in np.random.choice(sample_ids, 5, replace=False):
            print(f"  {dataset.samples[i][0]}")

    np.random.shuffle(indices)
    train_size = int(0.8 * len(indices))
    train_indices, val_indices = indices[:train_size], indices[train_size:]

    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(train_indices), num_workers=8)
    val_loader = DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(val_indices), num_workers=8)

    return train_loader, val_loader, dataset


# === Training ===
def train_model(model, criterion, optimizer, scheduler, train_loader, val_loader, num_epochs, writer):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')

    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch}/{num_epochs - 1}")
        model.train()
        running_loss, running_corrects = 0.0, 0

        for inputs, labels in tqdm(train_loader, desc="Training"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)

            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += (preds == labels).sum().item()

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects / len(train_loader.dataset)
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)
        writer.add_scalar('Train/Loss', epoch_loss, epoch)
        writer.add_scalar('Train/Accuracy', epoch_acc, epoch)

        logger.info(f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

        model.eval()
        val_loss, val_corrects = 0.0, 0
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc="Validation"):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                val_loss += loss.item() * inputs.size(0)
                val_corrects += (preds == labels).sum().item()

        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_corrects / len(val_loader.dataset)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        writer.add_scalar('Val/Loss', val_loss, epoch)
        writer.add_scalar('Val/Accuracy', val_acc, epoch)

        logger.info(f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
        print(
            f'Epoch {epoch}/{num_epochs - 1} - Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}')
       
        # Save best model
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            checkpoint_dir = checkpoint_path
            os.makedirs(checkpoint_dir, exist_ok=True)
            model_filename = f'best_model_{epoch}_{best_loss:.2f}.pth'
            torch.save(model.state_dict(), os.path.join(
                checkpoint_dir, model_filename))
            logger.info(
                f"Saved best model checkpoint at epoch {epoch} with validation loss {best_loss:.2f}.")
        # Save model for every epoch
        all_epoch_dir = os.path.join(checkpoint_path, "All_Epoch_Models")
        os.makedirs(all_epoch_dir, exist_ok=True)

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_losses,
            'val_loss': val_losses,
            'train_accuracy': train_accuracies,
            'val_accuracy': val_accuracies
        }

        model_filename = f"epoch_{epoch}_train_{epoch_loss:.4f}_val_{val_loss:.4f}.pth"
        torch.save(checkpoint, os.path.join(all_epoch_dir, model_filename))
    # Final model
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), os.path.join(checkpoint_path, "Final_model.pth"))
    logger.info("Saved final model.")

   # Save training stats in a dedicated folder
    stats_dir = os.path.join(checkpoint_path, "Training_Stats")
    os.makedirs(stats_dir, exist_ok=True)

    stats = {
        "train_loss": train_losses,
        "val_loss": val_losses,
        "train_accuracy": train_accuracies,
        "val_accuracy": val_accuracies
    }
    with open(os.path.join(stats_dir, "training_stats.json"), "w") as f:
        json.dump(stats, f)

    # === Plot Loss and Accuracy Curves ===
    epochs = range(num_epochs)
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Val Loss")
    plt.legend(); plt.grid(); plt.title("Loss")
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label="Train Acc")
    plt.plot(epochs, val_accuracies, label="Val Acc")
    plt.legend(); plt.grid(); plt.title("Accuracy")
    plt.tight_layout()
    plt.savefig(os.path.join(stats_dir, "loss_accuracy_curves.png"))
    plt.show()


# === Main ===
def main():
    log_dir = os.path.join("runs", "run_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    writer = SummaryWriter(log_dir)

    train_loader, val_loader, dataset = get_data_loaders(data_path, batch_size, num_img_per_class, transform)

    num_classes = len(dataset.classes)
    logger.info(f"\nDetected classes: {dataset.classes} (Total: {num_classes})")

    model = models.efficientnet_v2_s(pretrained=True)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = OneCycleLR(optimizer, max_lr=1e-3, epochs=num_epochs, steps_per_epoch=len(train_loader))

    train_model(model, criterion, optimizer, scheduler, train_loader, val_loader, num_epochs, writer)
    writer.close()


if __name__ == "__main__":
    main()

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
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from torchmetrics import Accuracy, MeanMetric

#torch.cuda.empty_cache()
# Paths and constants
checkpoint_path = "E:/Santosh_master_thesis/Understanding_citizenscience_species_segmentation/Check_Point"
data_path = "E:/Santosh_master_thesis/Understanding_citizenscience_species_segmentation/iNaturalist"
num_img_per_class = 2000  # Number of images per class
batch_size = 10  # Batch size for training
num_epochs = 50  # Number of epochs for training
num_classes = 10  # Number of classes in the dataset
image_size = 512  # Manually set image size
GPU_index = 'cuda:0'  # Only one GPU is used

os.makedirs(checkpoint_path, exist_ok=True)
# Initialize logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(),
    # Handles both resizing and cropping
    transforms.RandomResizedCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    transforms.RandomErasing(p=0.2, value='random')
])

device = torch.device(GPU_index if torch.cuda.is_available() else 'cpu')
# torch.cuda.set_device(device)  # Remove the helper function entirely


def get_data_loaders(data_dir, batch_size, num_img_per_class, image_size):

    dataset = ImageFolder(root=data_dir, transform=transform)

    # Count the number of images per class
    class_counts = np.bincount(dataset.targets)  # Directly get counts
    print("Original images per class:", dict(
        zip(dataset.classes, class_counts.tolist())))

    # Sample a specified number of images per class
    indices = []
    for class_idx in range(len(dataset.classes)):
        class_indices = np.where(
            np.array([s[1] for s in dataset.samples]) == class_idx)[0]
        if len(class_indices) < num_img_per_class:
            class_indices = np.random.choice(
                class_indices, num_img_per_class, replace=True)
        else:
            class_indices = np.random.choice(
                class_indices, num_img_per_class, replace=False)
        indices.extend(class_indices)

    # Shuffle and split indices for training and validation
    np.random.shuffle(indices)
    train_size = int(0.8 * len(indices))
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(
        dataset, batch_size=batch_size, sampler=train_sampler, num_workers=4)
    val_loader = DataLoader(dataset, batch_size=batch_size,
                            sampler=val_sampler, num_workers=4)

    # Print summary of number of sampled images per class
    sampled_class_counts = np.bincount(
        [dataset.samples[idx][1] for idx in indices])
    print("Number of images per class after sampling:")
    for class_idx, count in enumerate(sampled_class_counts):
        print(f'Class {dataset.classes[class_idx]}: {count} images')

    return train_loader, val_loader


def train_model(model, criterion, optimizer, scheduler, train_loader, val_loader, num_epochs, device, writer, checkpoint_path, logger):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')
    train_losses = []
    val_losses = []
    for epoch in range(num_epochs):
        logger.info(f'Epoch {epoch}/{num_epochs - 1}')
        logger.info('-' * 10)

        # Training phase
        model.train()
        running_loss = 0.0
        running_corrects = 0

        progress_bar = tqdm(
            train_loader, desc=f"Epoch {epoch}/{num_epochs - 1} Training")
        for batch_idx, (inputs, labels) in enumerate(progress_bar):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                loss.backward()
                optimizer.step()

                scheduler.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data).item()

            writer.add_scalar('Training Loss', loss.item(),
                              epoch * len(train_loader) + batch_idx)
            writer.add_scalar('Learning Rate', scheduler.get_last_lr()[
                              0], epoch * len(train_loader) + batch_idx)

            epoch_loss = running_loss / len(train_loader.dataset)
            epoch_acc = running_corrects / len(train_loader.dataset)

            train_losses.append(epoch_loss)  # <--- Store training loss

            writer.add_scalar('Training Loss', epoch_loss, epoch)
            writer.add_scalar('Training Accuracy', epoch_acc, epoch)

            # Calculate batch accuracy and error rate
            batch_loss = loss.item()
            batch_acc = torch.sum(preds == labels.data).item() / inputs.size(0)

            #Update tqdm description with metrics
            progress_bar.set_postfix({
                'Loss': f'{batch_loss:.4f}',
               'Acc': f'{batch_acc:.4f}'
            })

            writer.add_scalar('Training Loss', batch_loss,
                              epoch * len(train_loader) + batch_idx)
            writer.add_scalar('Learning Rate', scheduler.get_last_lr()[
                              0], epoch * len(train_loader) + batch_idx)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects / len(train_loader.dataset)

        writer.add_scalar('Training Loss', epoch_loss, epoch)
        writer.add_scalar('Training Accuracy', epoch_acc, epoch)

        logger.info(f'Training Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        print(
            f'Epoch {epoch}/{num_epochs - 1} - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}')

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_corrects = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)

                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data).item()

        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_corrects / len(val_loader.dataset)

        val_losses.append(val_loss)  # <--- Store validation loss

        writer.add_scalar('Validation Loss', val_loss, epoch)
        writer.add_scalar('Validation Accuracy', val_acc, epoch)

        logger.info(f'Validation Loss: {val_loss:.4f} Acc: {val_acc:.4f}')
        print(
            f'Epoch {epoch}/{num_epochs - 1} - Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}')

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
        model_filename = f"model_epoch_{epoch}_train_{epoch_loss:.4f}_val_{val_loss:.4f}.pth"
        torch.save(model.state_dict(), os.path.join(
            all_epoch_dir, model_filename))

    model.load_state_dict(best_model_wts)

    # Plot losses
    plt.figure(figsize=(10, 5))
    plt.plot(range(num_epochs), train_losses, label='Training Loss')
    plt.plot(range(num_epochs), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss per Epoch')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(checkpoint_path, "loss_curve.png"))  # Save plot
    plt.show()

    return model


def main():
    writer = SummaryWriter(checkpoint_path)

    data_dir = data_path
    train_loader, val_loader = get_data_loaders(
        data_dir, batch_size, num_img_per_class, image_size)

    model = models.efficientnet_v2_l(pretrained=False)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    # Using AdamW optimizer for better performance
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

    scheduler = OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=len(
        train_loader), epochs=num_epochs)

    model = train_model(model, criterion, optimizer, scheduler, train_loader,
                        val_loader, num_epochs, device, writer, checkpoint_path, logger)

    checkpoint_dir = checkpoint_path
    os.makedirs(checkpoint_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(
        checkpoint_dir, 'Final_model.pth'))
    logger.info("Saved final model.")

    writer.close()


if __name__ == "__main__":
    main()

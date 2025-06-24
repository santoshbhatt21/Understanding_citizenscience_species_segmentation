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

        #epoch_loss = running_loss / len(train_loader.dataset)
        #epoch_acc = running_corrects / len(train_loader.dataset)

        #writer.add_scalar('Training Loss', epoch_loss, epoch)
        #writer.add_scalar('Training Accuracy', epoch_acc, epoch)

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
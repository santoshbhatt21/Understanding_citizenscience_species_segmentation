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
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
from PIL import ImageFile
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from torchmetrics import Accuracy, MeanMetric
import matplotlib.pyplot as plt
import warnings
from typing import Optional

# Allow PIL to load truncated images instead of raising an exception
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Optional: suppress very large image warnings (uncomment if you see DecompressionBombWarning)
# warnings.simplefilter('ignore', Image.DecompressionBombWarning)

# Paths and constants
checkpoint_path = "E:/Santosh_master_thesis/Understanding_citizenscience_species_segmentation/script/Left_arm/Checkpoint_leftarm_4k_oneCLR"
data_path = "E:/Santosh_master_thesis/Understanding_citizenscience_species_segmentation/Data"
num_img_per_class = 4000
batch_size = 16
num_epochs = 150
num_classes = 10
image_size = 512  # Manually set image size
GPU_index = 'cuda:0'

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


def prepare_device():
    """Return the best available device and set CUDA device when applicable.

    Preference order: CUDA (respects GPU_index), Apple MPS (if available), CPU.
    """
    # Prefer CUDA if available and respect the configured GPU_index (e.g., 'cuda:0')
    if torch.cuda.is_available():
        dev = torch.device(GPU_index)
        # Set current CUDA device to ensure ops run on the selected GPU
        torch.cuda.set_device(dev)
        return dev

    # Optional: support Apple Silicon (won't affect Windows/Linux)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def pil_loader_safe(path: str):
    """Robust top-level PIL loader for multiprocessing pickling on Windows.

    Tries to open image; on failure, returns a blank image. Must be top-level so
    that DataLoader workers (spawned processes) can pickle the dataset.
    """
    try:
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')
    except Exception as e:
        logger.warning(
            f"Failed to load image {path}: {e}; substituting a blank image.")
        return Image.new('RGB', (image_size, image_size), color=(0, 0, 0))


def get_data_loaders(data_dir, batch_size, num_img_per_class, image_size):
    dataset = ImageFolder(
        root=data_dir, transform=transform, loader=pil_loader_safe)

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

    # Shuffle sampled indices and split into training and validation
    np.random.shuffle(indices)
    split = int(0.8 * len(indices))
    train_indices = indices[:split]
    val_indices = indices[split:]

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


def _save_training_state(state_path, model, optimizer, scheduler, epoch, best_loss, best_metrics, best_model_wts):
    os.makedirs(os.path.dirname(state_path), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'scheduler_state': scheduler.state_dict(),
        'best_loss': best_loss,
        'best_metrics': best_metrics,
        'best_model_wts': best_model_wts,
    }, state_path)


def _load_training_state(state_path, device):
    if not os.path.isfile(state_path):
        return None
    return torch.load(state_path, map_location=device)


def _compute_confusion_matrix(preds: torch.Tensor, labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Compute confusion matrix tensor of shape [num_classes, num_classes].

    Rows are ground-truth (labels), columns are predictions.
    """
    cm = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    for t, p in zip(labels.view(-1), preds.view(-1)):
        if 0 <= t.item() < num_classes and 0 <= p.item() < num_classes:
            cm[t.long(), p.long()] += 1
    return cm


def _plot_confusion_matrix(cm, class_names, normalize: bool = True, title_suffix: str = ""):
    """Return a matplotlib Figure visualizing the confusion matrix.

    If normalize=True, rows are normalized to sum to 1.
    """
    import numpy as np

    cm = cm.astype(float)
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        # avoid division by zero
        row_sums[row_sums == 0] = 1.0
        cm = cm / row_sums

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel='True label',
        title=f'Confusion Matrix {"(normalized)" if normalize else ""} {title_suffix}'.strip(
        )
    )
    ax.set_xlabel('Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45,
             ha="right", rotation_mode="anchor")

    # Annotate cells
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.0 if cm.size > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=8)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.2)
    return fig


def train_model(model, criterion, optimizer, scheduler, train_loader, val_loader, num_epochs, device, writer, checkpoint_path, logger,
                start_epoch: int = 0, resume_best_loss: Optional[float] = None,
                resume_best_metrics: Optional[dict] = None, resume_best_model_wts: Optional[dict] = None):
    # Initialize best trackers (possibly from resume)
    best_model_wts = copy.deepcopy(
        model.state_dict()) if resume_best_model_wts is None else resume_best_model_wts
    best_loss = float('inf') if resume_best_loss is None else resume_best_loss
    best_metrics = None if resume_best_metrics is None else resume_best_metrics

    # Correct denominators when using samplers: use number of sampled items, not full dataset size
    train_size = len(train_loader.sampler) if getattr(
        train_loader, 'sampler', None) is not None else len(train_loader.dataset)
    val_size = len(val_loader.sampler) if getattr(
        val_loader, 'sampler', None) is not None else len(val_loader.dataset)

    state_path = os.path.join(checkpoint_path, 'training_state.pth')

    for epoch in range(start_epoch, num_epochs):
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

            # Calculate batch accuracy and error rate
            batch_loss = loss.item()
            batch_acc = torch.sum(preds == labels.data).item() / inputs.size(0)

            # Update tqdm description with metrics
            progress_bar.set_postfix({
                'Loss': f'{batch_loss:.4f}',
                'Acc': f'{batch_acc:.4f}'
            })

            writer.add_scalar('Training Loss', batch_loss,
                              epoch * len(train_loader) + batch_idx)
            writer.add_scalar('Learning Rate', scheduler.get_last_lr()[
                              0], epoch * len(train_loader) + batch_idx)

        epoch_loss = running_loss / train_size
        epoch_acc = running_corrects / train_size

        writer.add_scalar('Epoch Training Loss', epoch_loss, epoch)
        writer.add_scalar('Epoch Training Accuracy', epoch_acc, epoch)

        logger.info(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        print(
            f'Epoch {epoch}/{num_epochs - 1} - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}')

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)

                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data).item()

                # Collect predictions and labels for confusion matrix
                all_preds.append(preds.detach().cpu())
                all_labels.append(labels.detach().cpu())

        val_loss = val_loss / val_size
        val_acc = val_corrects / val_size

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

            # Capture confusion matrix and metrics for the best epoch; log later at the end
            try:
                if all_preds and all_labels:
                    preds_epoch = torch.cat(all_preds)
                    labels_epoch = torch.cat(all_labels)
                    cm_t = _compute_confusion_matrix(
                        preds_epoch, labels_epoch, num_classes)
                    cm = cm_t.numpy()

                    class_names = getattr(val_loader.dataset, 'classes', [
                                          str(i) for i in range(num_classes)])

                    # Compute per-class precision/recall/F1
                    tp = np.diag(cm).astype(float)
                    fp = cm.sum(axis=0) - tp
                    fn = cm.sum(axis=1) - tp
                    denom_p = tp + fp
                    denom_r = tp + fn
                    precision = np.divide(
                        tp, denom_p, out=np.zeros_like(tp), where=denom_p != 0)
                    recall = np.divide(
                        tp, denom_r, out=np.zeros_like(tp), where=denom_r != 0)
                    denom_f1 = precision + recall
                    f1 = np.divide(2 * precision * recall, denom_f1,
                                   out=np.zeros_like(precision), where=denom_f1 != 0)

                    best_metrics = {
                        'epoch': epoch,
                        'cm': cm,
                        'class_names': class_names,
                        'precision': precision,
                        'recall': recall,
                        'f1': f1,
                        'macro_precision': float(np.mean(precision)),
                        'macro_recall': float(np.mean(recall)),
                        'macro_f1': float(np.mean(f1)),
                    }
            except Exception as e:
                logger.warning(
                    f"Failed to compute best-epoch confusion matrix/metrics: {e}")

        # Persist training state each epoch (for resume without full rerun)
        _save_training_state(state_path, model, optimizer, scheduler,
                             epoch, best_loss, best_metrics, best_model_wts)

    # If no improvements happened this session but we have a previous best model, compute metrics now
    if best_metrics is None and best_model_wts is not None:
        try:
            current_wts = copy.deepcopy(model.state_dict())
            model.load_state_dict(best_model_wts)
            model.eval()
            all_preds, all_labels = [], []
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    all_preds.append(preds.detach().cpu())
                    all_labels.append(labels.detach().cpu())
            preds_epoch = torch.cat(all_preds)
            labels_epoch = torch.cat(all_labels)
            cm_t = _compute_confusion_matrix(
                preds_epoch, labels_epoch, num_classes)
            cm = cm_t.numpy()
            class_names = getattr(val_loader.dataset, 'classes', [
                                  str(i) for i in range(num_classes)])
            tp = np.diag(cm).astype(float)
            fp = cm.sum(axis=0) - tp
            fn = cm.sum(axis=1) - tp
            denom_p = tp + fp
            denom_r = tp + fn
            precision = np.divide(
                tp, denom_p, out=np.zeros_like(tp), where=denom_p != 0)
            recall = np.divide(
                tp, denom_r, out=np.zeros_like(tp), where=denom_r != 0)
            denom_f1 = precision + recall
            f1 = np.divide(2 * precision * recall, denom_f1,
                           out=np.zeros_like(precision), where=denom_f1 != 0)
            best_metrics = {
                'epoch': -1,  # computed post hoc
                'cm': cm,
                'class_names': class_names,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'macro_precision': float(np.mean(precision)),
                'macro_recall': float(np.mean(recall)),
                'macro_f1': float(np.mean(f1)),
            }
            # restore current weights (we will load best at end anyway)
            model.load_state_dict(current_wts)
        except Exception as e:
            logger.warning(
                f"Failed to compute metrics from best weights at end: {e}")

    # Log best-epoch metrics to TensorBoard once at the end
    if best_metrics is not None:
        try:
            fig_cm_raw = _plot_confusion_matrix(best_metrics['cm'], best_metrics['class_names'], normalize=False,
                                                title_suffix=f"(epoch {best_metrics['epoch']})")
            writer.add_figure('Best/ConfusionMatrix_raw',
                              fig_cm_raw, best_metrics['epoch'])
            plt.close(fig_cm_raw)

            fig_cm_norm = _plot_confusion_matrix(best_metrics['cm'], best_metrics['class_names'], normalize=True,
                                                 title_suffix=f"(epoch {best_metrics['epoch']})")
            writer.add_figure('Best/ConfusionMatrix_norm',
                              fig_cm_norm, best_metrics['epoch'])
            plt.close(fig_cm_norm)

            writer.add_scalar(
                'Best/macro_precision', best_metrics['macro_precision'], best_metrics['epoch'])
            writer.add_scalar(
                'Best/macro_recall', best_metrics['macro_recall'], best_metrics['epoch'])
            writer.add_scalar(
                'Best/macro_f1', best_metrics['macro_f1'], best_metrics['epoch'])

            writer.add_histogram('Best/per_class_precision',
                                 best_metrics['precision'], best_metrics['epoch'])
            writer.add_histogram('Best/per_class_recall',
                                 best_metrics['recall'], best_metrics['epoch'])
            writer.add_histogram('Best/per_class_f1',
                                 best_metrics['f1'], best_metrics['epoch'])
        except Exception as e:
            logger.warning(
                f"Failed to log best-epoch confusion matrix/metrics: {e}")

    model.load_state_dict(best_model_wts)
    return model


def main():
    writer = SummaryWriter(checkpoint_path)
    device = prepare_device()

    data_dir = data_path
    train_loader, val_loader = get_data_loaders(
        data_dir, batch_size, num_img_per_class, image_size)

    model = models.efficientnet_v2_s(pretrained=False)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    # Using AdamW optimizer for better performance
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

    scheduler = OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=len(
        train_loader), epochs=num_epochs)

    # Attempt to resume training state if available
    state_path = os.path.join(checkpoint_path, 'training_state.pth')
    resume_state = _load_training_state(state_path, device)
    if resume_state is not None:
        try:
            try:
                # Strict load (identical architecture)
                model.load_state_dict(resume_state['model_state'])
                strict_loaded = True
            except Exception as e_load:
                # Warm start when architecture changed (e.g., num_classes was modified):
                # load only matching-shaped weights (typically backbone), reinit head.
                logger.warning(
                    f"Strict model load failed ({e_load}); attempting warm start with matching weights only.")
                current_sd = model.state_dict()
                pretrained_sd = resume_state['model_state']
                matched = 0
                for k, v in pretrained_sd.items():
                    if k in current_sd and current_sd[k].shape == v.shape:
                        current_sd[k] = v
                        matched += 1
                model.load_state_dict(current_sd)
                logger.info(
                    f"Warm start loaded {matched} matching parameters.")
                strict_loaded = False

            if strict_loaded:
                optimizer.load_state_dict(resume_state['optimizer_state'])
                scheduler.load_state_dict(resume_state['scheduler_state'])
                start_epoch = int(resume_state.get('epoch', 0)) + 1
                resume_best_loss = float(
                    resume_state.get('best_loss', float('inf')))
                resume_best_metrics = resume_state.get('best_metrics', None)
                resume_best_model_wts = resume_state.get(
                    'best_model_wts', None)
                logger.info(
                    f"Resuming training from epoch {start_epoch} with best_loss={resume_best_loss:.4f}")
            else:
                # Optimizer and scheduler states are invalid when params changed; restart training loop.
                start_epoch = 0
                resume_best_loss = None
                resume_best_metrics = None
                resume_best_model_wts = None
                logger.info(
                    "Restarting from epoch 0 with fresh optimizer/scheduler after warm start.")
        except Exception as e:
            logger.warning(
                f"Failed to load training state, starting fresh: {e}")
            start_epoch = 0
            resume_best_loss = None
            resume_best_metrics = None
            resume_best_model_wts = None
    else:
        start_epoch = 0
        resume_best_loss = None
        resume_best_metrics = None
        resume_best_model_wts = None

    model = train_model(model, criterion, optimizer, scheduler, train_loader,
                        val_loader, num_epochs, device, writer, checkpoint_path, logger,
                        start_epoch=start_epoch,
                        resume_best_loss=resume_best_loss,
                        resume_best_metrics=resume_best_metrics,
                        resume_best_model_wts=resume_best_model_wts)

    checkpoint_dir = checkpoint_path
    os.makedirs(checkpoint_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(
        checkpoint_dir, 'Final_model.pth'))
    logger.info("Saved final model.")

    writer.close()


if __name__ == "__main__":
    main()

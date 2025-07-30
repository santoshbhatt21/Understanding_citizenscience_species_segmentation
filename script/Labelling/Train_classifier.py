import os
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torchvision.models import resnet50
from torch.utils.data import DataLoader
from torch import nn, optim
from tqdm import tqdm

def train():
    # Paths
    data_dir = "E:/Santosh_master_thesis/Understanding_citizenscience_species_segmentation/Labelling_Manual"  # Make sure folder structure is correct
    save_model_path = "resnet50v2_specieswise.pth"
    label_map_path = "label_mapping.txt"

    # Transforms
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    # Dataset
    dataset = ImageFolder(root=data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)  # can set num_workers=0 for safety

    # Save label mapping for inference later
    with open(label_map_path, "w") as f:
        for cls, name in enumerate(dataset.classes):
            f.write(f"{cls}: {name}\n")

    # Model setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = resnet50(weights="IMAGENET1K_V2")
    model.fc = torch.nn.Linear(model.fc.in_features, 9)  # Use 9, not len(label_map)
    model = model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Training loop
    for epoch in range(10):
        model.train()
        running_loss = 0.0

        for inputs, labels in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}: Loss = {running_loss / len(dataloader):.4f}")

    # Save model
    torch.save(model.state_dict(), save_model_path)
    print(f"✅ Model saved to {save_model_path}")


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    train()



'''import os
import torch
import torchvision.transforms as transforms
from torchvision import datasets, models
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# === CONFIG ===
data_dir = "E:/Santosh_master_thesis/Understanding_citizenscience_species_segmentation/Labelling_Manual"
model_save_path = "leaf_trunk_other_classifier_best.pth"
num_classes = 9
batch_size = 32
epochs = 50
lr = 0.001
img_size = 256
patience = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === TRANSFORMS ===
train_transforms = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ToTensor(),
])

val_transforms = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
])

# === DATASET & SPLIT ===
full_dataset = datasets.ImageFolder(data_dir, transform=train_transforms)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
val_dataset.dataset.transform = val_transforms  # set validation transform

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# === MODEL ===
model = models.mobilenet_v2(pretrained=True)
model.classifier[1] = nn.Linear(model.last_channel, num_classes)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# === TRAINING LOOP WITH EARLY STOPPING ===
best_acc = 0.0
trigger = 0
for epoch in range(epochs):
    # --- Training ---
    model.train()
    running_loss, correct = 0.0, 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
    epoch_loss = running_loss / len(train_dataset)
    epoch_acc = correct / len(train_dataset)

    # --- Validation ---
    model.eval()
    val_loss, val_correct = 0.0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            val_correct += (preds == labels).sum().item()
    val_loss = val_loss / len(val_dataset)
    val_acc = val_correct / len(val_dataset)
    scheduler.step()

    print(f"Epoch {epoch+1}/{epochs} | Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

    # Save best model
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), model_save_path)
        print(f"📦 Best model saved at epoch {epoch+1}")
        trigger = 0
    else:
        trigger += 1
        if trigger >= patience:
            print("⏹️ Early stopping triggered.")
            break

# === EVALUATE BEST MODEL ===
print("\n📊 Evaluating best saved model...")
model.load_state_dict(torch.load(model_save_path))
model.eval()

all_preds, all_labels = [], []
with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Confusion matrix and report
class_names = full_dataset.classes
print("\n🔍 Confusion Matrix:\n", confusion_matrix(all_labels, all_preds))
print("\n📝 Classification Report:\n", classification_report(all_labels, all_preds, target_names=class_names))
print("✅ Training completed and best model saved.")
# Save the model architecture for future reference'''
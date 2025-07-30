import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, random_split
from PIL import Image
import shutil

if __name__ == "__main__":
    # ------------------ CONFIG ------------------
    data_dir = "E:/Santosh_master_thesis/Understanding_citizenscience_species_segmentation/Labelling_Manual"
    unlabeled_dir = "E:/Santosh_master_thesis/Understanding_citizenscience_species_segmentation/Data"
    output_dir = "output_sorted"
    batch_size = 16
    epochs = 20
    input_size = 256
    model_path = "universal_resnet50v2.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------ TRANSFORMS ------------------
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # ------------------ DATASET & SPLIT ------------------
    full_dataset = datasets.ImageFolder(data_dir, transform=train_transform)
    num_classes = len(full_dataset.classes)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    val_dataset.dataset.transform = val_transform

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    class_names = full_dataset.classes  # e.g., ['Species1/Leaves', 'Species1/Others', ...]

    # ------------------ MODEL ------------------
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes)
    )
    model = model.to(device)

    # ------------------ TRAIN ------------------
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        model.train()
        running_loss, running_corrects = 0.0, 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = running_corrects.double() / len(train_dataset)
        print(f"Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.4f}")

        # --- Validation ---
        model.eval()
        val_loss, val_corrects = 0.0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)
        val_loss = val_loss / len(val_dataset)
        val_acc = val_corrects.double() / len(val_dataset)
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

    torch.save(model.state_dict(), model_path)
    print("Model saved.")

    # ------------------ PREDICT AND SORT ------------------
    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    predict_transform = val_transform

    with torch.no_grad():
        for fname in os.listdir(unlabeled_dir):
            if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            img_path = os.path.join(unlabeled_dir, fname)
            image = Image.open(img_path).convert('RGB')
            input_tensor = predict_transform(image).unsqueeze(0).to(device)
            output = model(input_tensor)
            _, pred = torch.max(output, 1)
            pred_class = class_names[pred.item()]  # e.g., 'Species1/Leaves'
            target_folder = os.path.join(output_dir, pred_class)
            os.makedirs(target_folder, exist_ok=True)
            shutil.copy(img_path, os.path.join(target_folder, fname))

    print("Prediction and sorting complete.")
import os
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
from tqdm import tqdm

# --- Parameters ---
data_dir = "E:/Santosh_master_thesis/Understanding_citizenscience_species_segmentation/flat_labeled_data"  # Root folder with Leaves/Trunks/Others
save_path = "resnet50v2_leaf_trunk_others.pth"
batch_size = 32
num_epochs = 20
patience = 5
learning_rate = 1e-4
#num_classes = 3  # Leaves, Trunks, Others

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Transforms ---
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], 
                         [0.229, 0.224, 0.225])
])

# --- Dataset ---
train_dataset = datasets.ImageFolder(root=data_dir, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
num_classes = len(train_dataset.classes)  # Automatically detect number of classes

# --- Model ---
model = models.resnet50(weights='IMAGENET1K_V2')
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256, num_classes)
)
model = model.to(device)

# --- Loss & Optimizer ---
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# --- Training Loop ---
model.train()
for epoch in range(num_epochs):
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    acc = correct / total
    print(f"Epoch {epoch+1} Loss: {total_loss:.4f} | Accuracy: {acc:.4f}")

# --- Save Model ---
torch.save(model.state_dict(), save_path)
print(f"Model saved to {save_path}")

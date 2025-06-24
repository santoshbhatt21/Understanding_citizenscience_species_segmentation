import os
import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from tqdm import tqdm

# 1. Set paths and parameters
folder_path = "E:/Santosh_master_thesis/Understanding_citizenscience_species_segmentation/Check_Point/All_Epoch_Models"
data_path = "E:/Santosh_master_thesis/Understanding_citizenscience_species_segmentation/iNaturalist"
num_classes = 10  # Set this to your number of classes
batch_size = 10    # Use a small batch size to avoid OOM
image_size = 512  # Use the same size as during training
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 2. Define your model (adjust if you used a different architecture)
def get_model():
    model = models.efficientnet_v2_l(pretrained=False)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    return model

# 3. Define your validation transforms (should match your training/val transforms)
val_transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

# 4. Load validation dataset
val_dataset = datasets.ImageFolder(root=data_path, transform=val_transform)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# 5. Evaluate each checkpoint
results = []
for filename in sorted(os.listdir(folder_path)):
    if filename.endswith(".pth"):
        model_path = os.path.join(folder_path, filename)
        model = get_model().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Evaluating {filename}"):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        acc = correct / total
        print(f"{filename}: Validation Accuracy = {acc:.4f}")
        results.append((filename, acc))

# 6. Optionally, save results to CSV
import pandas as pd
df = pd.DataFrame(results, columns=["Model", "Val_Accuracy"])
df.to_csv(os.path.join(folder_path, "epoch_model_accuracies.csv"), index=False)
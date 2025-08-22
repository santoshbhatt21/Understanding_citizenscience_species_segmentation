import os, torch
import numpy as np
from collections import Counter
from torchvision import datasets, transforms, models
from torchvision.models import EfficientNet_V2_S_Weights
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

# CONFIG
data_path = r"E:/Santosh_master_thesis/flat_labeled_data_leaves_trunks"
ckpt_path = r"E:/Santosh_master_thesis/Checkpoints_tree_organs_other_emphasis/best_model_XX_YY.pth"  # change
seed = 42
image_size = 512
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transforms
val_transform = transforms.Compose([
    transforms.Resize(int(image_size * 1.1)),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# Base dataset + deterministic split (same as training)
base = datasets.ImageFolder(data_path, transform=None)
targets = [s[1] for s in base.samples]
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
train_idx, val_idx = next(sss.split(np.zeros(len(targets)), targets))

# Build loaders with val-style transforms for both splits
train_eval = datasets.ImageFolder(data_path, transform=val_transform)
val_eval = datasets.ImageFolder(data_path, transform=val_transform)
train_eval.samples = [base.samples[i] for i in train_idx]
val_eval.samples = [base.samples[i] for i in val_idx]
train_loader = DataLoader(train_eval, batch_size=32, shuffle=False, num_workers=4)
val_loader = DataLoader(val_eval, batch_size=32, shuffle=False, num_workers=4)

# Model scaffold (must match training head)
num_classes = len(train_eval.classes)
model = models.efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT)
in_features = model.classifier[1].in_features
model.classifier = torch.nn.Sequential(torch.nn.Dropout(0.5),
                                       torch.nn.Linear(in_features, num_classes))
model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
model.to(device).eval()

def eval_loader(loader):
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            y_true.extend(y.numpy().tolist())
            y_pred.extend(logits.argmax(1).cpu().numpy().tolist())
    acc = accuracy_score(y_true, y_pred)
    bal = balanced_accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average="macro")
    return acc, bal, f1m

ta_eval, ta_bal, ta_f1 = eval_loader(train_loader)   # Train split, eval mode, val transforms
va_eval, va_bal, va_f1 = eval_loader(val_loader)

print(f"Train(eval, val-tfm) Acc={ta_eval:.4f} BalAcc={ta_bal:.4f} F1m={ta_f1:.4f}")
print(f"Val                Acc={va_eval:.4f} BalAcc={va_bal:.4f} F1m={va_f1:.4f}")

# Optional: leakage/duplicate check
train_paths = set(p for p,_ in train_eval.samples)
val_paths = set(p for p,_ in val_eval.samples)
overlap = train_paths & val_paths
print(f"Path overlap between train/val: {len(overlap)}")
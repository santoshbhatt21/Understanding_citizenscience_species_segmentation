import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torchvision import models
import warnings
warnings.filterwarnings("ignore")


# ==========================================================
# CONFIGURATION (EDIT ONLY THESE IF NEEDED)
# ==========================================================

BEST_MODEL_PATH = r"E:/Santosh_master_thesis/Understanding_citizenscience_species_segmentation/script/Left_arm/Checkpoint_leftarm_4k_oneCLR/best_model_148_0.33.pth"
NUM_CLASSES = 10

CLASS_NAMES = [
    "Abies alba Leaves",
    "Acer pseudoplatanus Leaves",
    "Betula pendula Leaves",
    "Fagus sylvatica Leaves",
    "Fraxinus excelsior Leaves",
    "Larix decidua Leaves",
    "Picea abies Leaves",
    "Pinus sylvestris Leaves",
    "Pseudotsuga menziesii Leaves",
    "Quercus rubra Leaves"
]

RESULTS_CSV = r"results.csv"
SAVE_DIR = "post_training_outputs"


# ==========================================================
# LOAD MODEL (EfficientNetV2-S)
# ==========================================================

def load_model(best_model_path):
    model = models.efficientnet_v2_s(weights=None)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, NUM_CLASSES)

    state = torch.load(best_model_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


# ==========================================================
# 1. PLOT TRAINING CURVES
# ==========================================================

def plot_training_curves(df, save_dir):
    # LOSS CURVES
    plt.figure(figsize=(10,5))
    plt.plot(df["epoch"], df["train/box_loss"], label="train/box_loss")
    plt.plot(df["epoch"], df["train/cls_loss"], label="train/cls_loss")
    plt.plot(df["epoch"], df["train/seg_loss"], label="train/seg_loss")
    plt.plot(df["epoch"], df["val/box_loss"], label="val/box_loss")
    plt.plot(df["epoch"], df["val/cls_loss"], label="val/cls_loss")
    plt.plot(df["epoch"], df["val/seg_loss"], label="val/seg_loss")
    plt.title("Training & Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()
    plt.savefig(Path(save_dir)/"loss_curve.png", dpi=200)
    plt.close()

    # ACCURACY
    # If your script logs accuracy separately, add here.
    # Otherwise skip.

    # PRECISION / RECALL
    plt.figure(figsize=(8,5))
    plt.plot(df["epoch"], df["metrics/precision(M)"], label="Precision")
    plt.plot(df["epoch"], df["metrics/recall(M)"], label="Recall")
    plt.title("Precision & Recall Curve")
    plt.xlabel("Epoch")
    plt.legend()
    plt.grid()
    plt.savefig(Path(save_dir)/"precision_recall_curve.png", dpi=200)
    plt.close()

    # F1
    P = df["metrics/precision(M)"]
    R = df["metrics/recall(M)"]
    F1 = 2 * P * R / (P + R + 1e-9)

    plt.figure(figsize=(8,5))
    plt.plot(df["epoch"], F1, label="F1")
    plt.title("F1 Curve")
    plt.xlabel("Epoch")
    plt.grid()
    plt.savefig(Path(save_dir)/"f1_curve.png", dpi=200)
    plt.close()

    # mAP
    plt.figure(figsize=(8,5))
    plt.plot(df["epoch"], df["metrics/mAP50(M)"], label="mAP50")
    plt.plot(df["epoch"], df["metrics/mAP50-95(M)"], label="mAP50-95")
    plt.title("mAP Curves")
    plt.xlabel("Epoch")
    plt.legend()
    plt.grid()
    plt.savefig(Path(save_dir)/"map_curve.png", dpi=200)
    plt.close()


# ==========================================================
# 2. CONFUSION MATRIX
# ==========================================================

def compute_confusion_matrix(model, val_loader, class_names, save_dir):
    all_preds = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.cuda() if torch.cuda.is_available() else images
            outputs = model(images)
            preds = outputs.argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    cm = confusion_matrix(all_labels, all_preds)

    # Raw confusion matrix
    disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
    disp.plot(cmap="Blues", xticks_rotation=45)
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(Path(save_dir)/"confusion_matrix.png", dpi=200)
    plt.close()

    # Normalized
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, None]
    disp = ConfusionMatrixDisplay(cm_norm, display_labels=class_names)
    disp.plot(cmap="Blues", xticks_rotation=45)
    plt.title("Normalized Confusion Matrix")
    plt.tight_layout()
    plt.savefig(Path(save_dir)/"confusion_matrix_normalized.png", dpi=200)
    plt.close()

    return cm, cm_norm


# ==========================================================
# 3. HARDEST SPECIES (lowest recall)
# ==========================================================

def save_hardest_species(cm_norm, class_names, save_dir):
    recall = np.diag(cm_norm)
    sorted_idx = np.argsort(recall)

    with open(Path(save_dir)/"hardest_species.txt", "w") as f:
        f.write("Species Difficulty Ranking\n")
        f.write("--------------------------\n\n")
        for idx in sorted_idx:
            f.write(f"{class_names[idx]} — Recall: {recall[idx]:.3f}\n")


# ==========================================================
# 4. DIAGNOSIS (overfitting / underfitting)
# ==========================================================

def diagnose(df, save_dir):
    t_loss = df["train/cls_loss"].iloc[-1]
    v_loss = df["val/cls_loss"].iloc[-1]
    t_acc = df["metrics/precision(M)"].iloc[-1]
    v_acc = df["metrics/recall(M)"].iloc[-1]

    msg = "=== TRAINING DIAGNOSIS ===\n"

    if v_loss > t_loss * 1.4:
        msg += "Overfitting detected.\n"
    elif v_acc < 0.60:
        msg += "Underfitting detected.\n"
    else:
        msg += "Model is well-balanced.\n"

    with open(Path(save_dir)/"diagnosis.txt", "w") as f:
        f.write(msg)

    print(msg)


# ==========================================================
# MAIN FUNCTION
# ==========================================================

def run_post_training_analysis(val_loader):
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Load logs
    df = pd.read_csv(RESULTS_CSV)

    # Load best model
    print("Loading model...")
    model = load_model(BEST_MODEL_PATH)
    if torch.cuda.is_available():
        model.cuda()

    # Generate plots
    print("Generating curves...")
    plot_training_curves(df, SAVE_DIR)

    # Confusion matrix
    print("Computing confusion matrix...")
    cm, cm_norm = compute_confusion_matrix(model, val_loader, CLASS_NAMES, SAVE_DIR)

    # Hardest species
    save_hardest_species(cm_norm, CLASS_NAMES, SAVE_DIR)

    # Diagnosis
    diagnose(df, SAVE_DIR)

    print("\nAll outputs saved to:", SAVE_DIR)
    print("Done ✔")


# ==========================================================
# USAGE (Call from your training or evaluation script)
# ==========================================================

"""
Example:

from post_training_analysis import run_post_training_analysis

# after defining your val_loader:
run_post_training_analysis(val_loader)

"""


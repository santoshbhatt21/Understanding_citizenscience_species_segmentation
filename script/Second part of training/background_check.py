#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Inspect a PyTorch classifier checkpoint to see how many output classes it was trained with,
so you can tell if a separate 'background' class was included.

Manual run only — edit CONFIG below and run:
    python inspect_classifier_num_classes_manual.py
"""

# =====================
# CONFIG (EDIT ME)
# =====================
CKPT = r"Checkpoints_efficientnet_weightedrandomsampler_two_stages_class_weights_focal_loss/best_by_loss_ep34_0.04.pth"  # .pth file from your training
DATA_DIR = r"E:/Santosh_master_thesis/Understanding_citizenscience_species_segmentation/Data"  # optional: ImageFolder root to compare folder classes
PRINT_SAMPLE_CLASS_NAMES = True

# =====================
# IMPLEMENTATION
# =====================
import os, re, json
from pathlib import Path
import torch
import torch.nn as nn

def load_state(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("state_dict") or ckpt.get("model") or ckpt
    if isinstance(state, dict):
        # Remove possible DistributedDataParallel prefix
        state = {k.replace("module.", ""): v for k, v in state.items()}
    return state

def guess_num_classes_from_state(state_dict):
    """
    Heuristic: find linear layer weights whose shape is [out_features, in_features].
    Prefer layers whose name hints 'classifier', 'fc', 'head', 'last', otherwise pick the
    one with the **smallest** out_features among linear layers (usually the final layer).
    """
    candidates = []
    for name, tensor in state_dict.items():
        if not hasattr(tensor, "shape"):
            continue
        if tensor.ndim == 2:  # linear layer weights
            out_f, in_f = tensor.shape
            score = 0
            lname = name.lower()
            if any(k in lname for k in ["classifier", "fc", "head", "last", "logits"]):
                score += 10
            # prefer small out_features (typical for num_classes)
            score += max(0, 1000 - out_f)
            candidates.append((score, name, out_f, in_f))

    if not candidates:
        return None, []

    candidates.sort(reverse=True)  # higher score first
    top = candidates[0]
    return int(top[2]), candidates

def list_imagefolder_classes(root):
    from torchvision import datasets
    try:
        ds = datasets.ImageFolder(root)
        return ds.classes
    except Exception as e:
        return None

def main():
    print("== Inspecting checkpoint ==")
    print("CKPT:", CKPT)
    state = load_state(CKPT)
    num_classes, cands = guess_num_classes_from_state(state)
    if num_classes is None:
        print("Could not infer num_classes (no 2D linear weights found).")
        return

    print(f"Inferred number of output classes (from final linear weight): {num_classes}")
    print("\nTop classifier-like linear layers found (score, name, out_features, in_features):")
    for score, name, out_f, in_f in cands[:5]:
        print(f"  score={score:4.0f}  {name:40s}  -> [{out_f}, {in_f}]")

    classes = None
    if DATA_DIR and os.path.isdir(DATA_DIR):
        classes = list_imagefolder_classes(DATA_DIR)

    if classes is not None:
        print(f"\nImageFolder classes found in '{DATA_DIR}': {len(classes)}")
        if PRINT_SAMPLE_CLASS_NAMES:
            print("First 10 class names:", classes[:10])
        # Heuristics for 'background'
        has_bg = any(c.lower() in {"background", "bg"} for c in classes)
        if has_bg:
            print("NOTE: Your dataset contains a folder named 'background' (or 'bg').")

        if len(classes) == num_classes:
            print("\n✔ Model output size matches dataset class count.")
        else:
            print(f"\n⚠ Mismatch: model out_classes = {num_classes} vs dataset classes = {len(classes)}")
            print("   This often indicates label misalignment during evaluation or wrong class list used for CM.")
    else:
        print("\n(No ImageFolder provided or could not read classes from it.)")

    print("\nHow to interpret:")
    print("• If inferred classes == K, and you did NOT intend to include 'background', then you trained on K classes only (no background).")
    print("• If inferred classes == K+1 and your dataset includes a 'background' folder, you likely trained WITH background as a class.")
    print("• For standard image classification, background is usually NOT a class; for segmentation, background is often implicit as label 0.")
    print("\nIf your confusion matrix includes 'background' but num_classes == K (no bg), remove 'background' from the class_names you pass to the CM.")

if __name__ == "__main__":
    main()

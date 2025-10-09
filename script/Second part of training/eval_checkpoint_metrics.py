# Create a ready-to-run evaluation script that computes exact F1, balanced accuracy,
# accuracy, and confusion matrices for a given checkpoint + ImageFolder val set.
script_path = "/mnt/data/eval_checkpoint_metrics.py"
code = r'''
import argparse, os, re, json, math, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from sklearn.metrics import (
    f1_score, balanced_accuracy_score, accuracy_score,
    confusion_matrix, classification_report, precision_recall_fscore_support
)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def parse_ckpt_metrics_from_name(path: str):
    name = os.path.basename(path)
    m = re.search(r"epoch_(?P<epoch>\d+)_tl(?P<TL>[0-9.]+)_ta(?P<TA>[0-9.]+)_vl(?P<VL>[0-9.]+)_va(?P<VA>[0-9.]+)\.pth$", name)
    if not m: return {}
    d = {k: float(v) if k != "epoch" else int(v) for k, v in m.groupdict().items()}
    return d

def try_build_model(model_name: str, num_classes: int):
    # Try timm first
    try:
        import timm
        model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
        return model
    except Exception:
        pass
    # Try torchvision next
    try:
        import torchvision.models as tvm
        if hasattr(tvm, model_name):
            ctor = getattr(tvm, model_name)
            model = ctor(weights=None)
            # replace classifier / head
            if hasattr(model, "classifier") and isinstance(model.classifier, nn.Sequential):
                # efficientnet_v2_* in torchvision
                last_in = None
                for m in reversed(model.classifier):
                    if isinstance(m, nn.Linear):
                        last_in = m.in_features
                        break
                if last_in is None:
                    last_in = 1280
                model.classifier[-1] = nn.Linear(last_in, num_classes)
            elif hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
                model.fc = nn.Linear(model.fc.in_features, num_classes)
            elif hasattr(model, "head") and isinstance(model.head, nn.Linear):
                model.head = nn.Linear(model.head.in_features, num_classes)
            else:
                # generic fallback: find final linear layer
                for name, module in model.named_modules():
                    if isinstance(module, nn.Linear) and module.out_features != num_classes:
                        module.out_features = num_classes
                        break
            return model
    except Exception:
        pass
    raise RuntimeError("Could not build model for name='%s'. Install timm or use a torchvision model name." % model_name)

@torch.no_grad()
def evaluate(model, loader, device="cpu", autocast_dtype=None):
    model.eval().to(device)
    y_true, y_pred = [], []
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(dtype=autocast_dtype) if autocast_dtype else torch.inference_mode():
            logits = model(xb)
        preds = logits.argmax(dim=1).cpu().numpy()
        y_pred.append(preds)
        y_true.append(yb.numpy())
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    per_prec, per_rec, per_f1, per_support = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
    return {
        "y_true": y_true, "y_pred": y_pred,
        "acc": float(acc), "macro_f1": float(macro_f1), "bal_acc": float(bal_acc),
        "cm": cm, "per_class": {
            "precision": per_prec.tolist(),
            "recall": per_rec.tolist(),
            "f1": per_f1.tolist(),
            "support": per_support.tolist(),
        }
    }

def save_confusion_matrices(cm, class_names, out_dir: Path, prefix="confusion_matrix"):
    out_dir.mkdir(parents=True, exist_ok=True)
    # Counts
    fig = plt.figure(figsize=(12,9))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=60, ha="right")
    plt.yticks(tick_marks, class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    fig.tight_layout()
    fig.savefig(out_dir / f"{prefix}.png", dpi=200)
    plt.close(fig)

    # Normalized
    with np.errstate(divide="ignore", invalid="ignore"):
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        cm_norm = np.nan_to_num(cm_norm)
    fig = plt.figure(figsize=(12,9))
    plt.imshow(cm_norm, interpolation="nearest", vmin=0.0, vmax=1.0)
    plt.title("Confusion Matrix (Normalized)")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=60, ha="right")
    plt.yticks(tick_marks, class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    fig.tight_layout()
    fig.savefig(out_dir / f"{prefix}_normalized.png", dpi=200)
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Path to .pth checkpoint")
    ap.add_argument("--val_dir", required=True, help="Path to validation folder (ImageFolder)")
    ap.add_argument("--model", default="efficientnet_v2_s", help="Model name for timm/torchvision")
    ap.add_argument("--img-size", type=int, default=384)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--class-names", default=None, help="Optional path to classes.txt (one per line). Defaults to ImageFolder classes.")
    ap.add_argument("--half", action="store_true", help="Use autocast float16 on CUDA")
    args = ap.parse_args()

    out_dir = Path(args.ckpt).with_suffix("")  # folder named after ckpt (without .pth)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Data
    tfm = transforms.Compose([
        transforms.Resize(int(args.img_size * 1.14)),
        transforms.CenterCrop(args.img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    val_ds = datasets.ImageFolder(args.val_dir, transform=tfm)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    # Classes
    if args.class_names and os.path.isfile(args.class_names):
        with open(args.class_names, "r", encoding="utf-8") as f:
            class_names = [line.strip() for line in f if line.strip()]
    else:
        class_names = list(val_ds.classes)

    # Build & load model
    model = try_build_model(args.model, num_classes=len(class_names))

    ckpt = torch.load(args.ckpt, map_location="cpu")
    state = ckpt.get("state_dict") or ckpt.get("model") or ckpt
    if isinstance(state, dict):
        # Remove possible 'module.' prefixes
        new_state = {}
        for k, v in state.items():
            new_state[k.replace("module.", "")] = v
        state = new_state
    model.load_state_dict(state, strict=False)

    # Evaluate
    dtype = torch.float16 if (args.half and args.device.startswith("cuda")) else None
    metrics = evaluate(model, val_loader, device=args.device, autocast_dtype=dtype)

    # Summaries
    from_name = parse_ckpt_metrics_from_name(args.ckpt)
    va_from_name = from_name.get("VA", None)

    summary = {
        "checkpoint": args.ckpt,
        "model": args.model,
        "num_classes": len(class_names),
        "samples": len(val_ds),
        "computed": {
            "val_accuracy": metrics["acc"],
            "macro_f1": metrics["macro_f1"],
            "balanced_accuracy": metrics["bal_acc"],
        },
        "from_filename": from_name,
        "derived": {
            "delta(VA-F1)": (va_from_name - metrics["macro_f1"]) if va_from_name is not None else None,
            "delta(VA-Acc)": (va_from_name - metrics["acc"]) if va_from_name is not None else None
        }
    }

    # Save artifacts
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    np.savetxt(out_dir / "confusion_matrix.csv", metrics["cm"], fmt="%d", delimiter=",")

    # Per-class metrics table
    import pandas as pd
    per_df = pd.DataFrame({
        "class": class_names,
        "precision": metrics["per_class"]["precision"],
        "recall": metrics["per_class"]["recall"],
        "f1": metrics["per_class"]["f1"],
        "support": metrics["per_class"]["support"],
    })
    per_df.to_csv(out_dir / "per_class_metrics.csv", index=False)

    # Confusion matrices
    save_confusion_matrices(metrics["cm"], class_names, out_dir, prefix="confusion_matrix")

    # Print a concise summary
    print(json.dumps(summary, indent=2))
    print(f"\nSaved artifacts to: {out_dir}")

if __name__ == "__main__":
    main()
'''
with open(script_path, "w", encoding="utf-8") as f:
    f.write(code)

script_path

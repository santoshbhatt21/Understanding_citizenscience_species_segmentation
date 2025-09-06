import os
import re
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def find_stats_dir(ckpt_root: str) -> str:
    stats_dir = os.path.join(ckpt_root, "Training_Stats")
    if os.path.isdir(stats_dir):
        return stats_dir
    # fallback: if user passed Training_Stats directly
    if os.path.basename(ckpt_root).lower() == "training_stats":
        return ckpt_root
    raise FileNotFoundError(
        f"Training_Stats folder not found under '{ckpt_root}'.")


def save_as_jpg(png_path: str, jpg_path: str, quality: int = 95):
    with Image.open(png_path) as im:
        rgb = im.convert("RGB")
        rgb.save(jpg_path, format="JPEG", quality=quality)


def plot_confusion(cm: np.ndarray, labels, out_png: str, title: str = None, normalize: bool = False,
                   tick_fontsize: int = 8, cell_fontsize: int = 6, figsize=(12, 10)):
    plt.figure(figsize=figsize)
    mat = cm.astype(float).copy()
    if normalize:
        with np.errstate(all='ignore'):
            row_sums = mat.sum(axis=1, keepdims=True)
            mat = np.divide(mat, row_sums, out=np.zeros_like(
                mat), where=row_sums != 0)
    im = plt.imshow(mat, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    ticks = np.arange(len(labels))
    plt.xticks(ticks, labels, rotation=45, ha='right', fontsize=tick_fontsize)
    plt.yticks(ticks, labels, fontsize=tick_fontsize)
    fmt = ".2f" if normalize else ".0f"
    thresh = (mat.max() if mat.size else 0) / 2.0 if mat.size else 0
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            plt.text(j, i, format(val, fmt), ha="center", va="center", fontsize=cell_fontsize,
                     color="white" if val > thresh else "black")
    plt.ylabel('True', fontsize=tick_fontsize)
    plt.xlabel('Predicted', fontsize=tick_fontsize)
    if title:
        plt.title(title, fontsize=tick_fontsize + 2)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=300)
    plt.close()


def synthesize_diagonal_from_report(report_path: str):
    """Build a diagonal-only confusion matrix approximation from per-class recall/support.
    This is NOT a true confusion matrix. Off-diagonals are unknown and set to 0.
    """
    with open(report_path, 'r', encoding='utf-8') as f:
        report = json.load(f)
    excluded = {"accuracy", "macro avg", "weighted avg"}
    class_items = [(k, v) for k, v in report.items()
                   if isinstance(v, dict) and k not in excluded]
    labels = [k for k, _ in class_items]
    n = len(labels)
    cm = np.zeros((n, n), dtype=float)
    for i, (_, stats) in enumerate(class_items):
        support = float(stats.get("support", 0.0))
        recall = float(stats.get("recall", 0.0))
        tp = recall * support
        cm[i, i] = tp
        # remainder (FN) is unknown which classes received them
        # We leave off-diagonals as 0 to avoid implying specifics.
    return cm, labels


def clean_labels(labels, strip_numeric_prefix: bool = True, replace_underscores: bool = False):
    cleaned = []
    for lab in labels:
        s = str(lab)
        if strip_numeric_prefix:
            s = re.sub(r"^\s*\d+[_\s]+", "", s)
        if replace_underscores:
            s = s.replace("_", " ")
        cleaned.append(s)
    return cleaned


def parse_figsize(s: str):
    try:
        parts = s.lower().split('x')
        if len(parts) == 2:
            return (float(parts[0]), float(parts[1]))
    except Exception:
        pass
    return (12, 10)


def main():
    parser = argparse.ArgumentParser(
        description="Export confusion matrix images from Training_Stats or approximate from classification_report.json")
    parser.add_argument('--ckpt-root', type=str, default=r"E:/Santosh_master_thesis/Checkpoints_LT_organ_species_640",
                        help='Checkpoint root or Training_Stats folder')
    parser.add_argument('--out-prefix', type=str, default="confusion_export",
                        help='Output filename prefix (without extension) placed in Training_Stats')
    parser.add_argument('--approximate-from-report', action='store_true',
                        help='If no confusion image is found, synthesize a diagonal-only approximation from classification_report.json')
    parser.add_argument('--strip-numeric-prefix', action='store_true', default=True,
                        help='Remove leading numeric prefixes like 001_, 002 from class labels when plotting')
    parser.add_argument('--replace-underscores', action='store_true', default=False,
                        help='Replace underscores with spaces in labels for readability')
    parser.add_argument('--tick-fontsize', type=int, default=10,
                        help='Font size for axis tick labels')
    parser.add_argument('--cell-fontsize', type=int, default=8,
                        help='Font size for cell annotations')
    parser.add_argument('--figsize', type=str, default='12x10',
                        help='Figure size as WxH, e.g., 14x12')
    args = parser.parse_args()

    stats_dir = find_stats_dir(args.ckpt_root)

    # 0) If a saved confusion_matrix.json exists, replot the original values with cleaned labels and +2 font sizes
    cm_json_path = os.path.join(stats_dir, "confusion_matrix.json")
    if os.path.exists(cm_json_path):
        try:
            with open(cm_json_path, 'r', encoding='utf-8') as f:
                cm_obj = json.load(f)
            cm = np.array(cm_obj.get('matrix'))
            labels = cm_obj.get('labels') or [str(i)
                                              for i in range(cm.shape[0])]
            labels = clean_labels(labels, strip_numeric_prefix=args.strip_numeric_prefix,
                                  replace_underscores=args.replace_underscores)
            # Raw (counts) confusion matrix with cleaned labels
            out_png = os.path.join(stats_dir, f"{args.out_prefix}_clean.png")
            plot_confusion(cm, labels, out_png,
                           title="Confusion Matrix (Clean Labels)", normalize=False,
                           tick_fontsize=args.tick_fontsize + 2, cell_fontsize=args.cell_fontsize + 2,
                           figsize=parse_figsize(args.figsize))
            out_jpg = os.path.join(stats_dir, f"{args.out_prefix}_clean.jpg")
            save_as_jpg(out_png, out_jpg)

            out_png_norm = os.path.join(
                stats_dir, f"{args.out_prefix}_clean_normalized.png")
            # Plot normalized directly from raw counts to keep data identical and show decimals
            plot_confusion(cm, labels, out_png_norm,
                           title="Confusion Matrix (Normalized, Clean Labels)", normalize=True,
                           tick_fontsize=args.tick_fontsize + 2, cell_fontsize=args.cell_fontsize + 2,
                           figsize=parse_figsize(args.figsize))
            out_jpg_norm = os.path.join(
                stats_dir, f"{args.out_prefix}_clean_normalized.jpg")
            save_as_jpg(out_png_norm, out_jpg_norm)

            print(
                f"Exported confusion matrices with cleaned labels from JSON:\n -> {out_png}\n -> {out_jpg}\n -> {out_png_norm}\n -> {out_jpg_norm}")
            return
        except Exception as e:
            print(
                f"Warning: Failed to use confusion_matrix.json: {e}. Falling back to image-based export.")

    # 1) If confusion_matrix.png exists, copy/export to desired names (PNG and JPG)
    candidates = [
        os.path.join(stats_dir, "confusion_matrix.png"),
        os.path.join(stats_dir, "confusion_matrix_normalized.png"),
    ]
    found = [p for p in candidates if os.path.exists(p)]
    if found:
        # Use the first found as source
        src = found[0]
        out_png = os.path.join(stats_dir, f"{args.out_prefix}.png")
        out_jpg = os.path.join(stats_dir, f"{args.out_prefix}.jpg")
        # If user wants relabeled axes (strip prefixes), try to rebuild labels from report
        report_path = os.path.join(stats_dir, "classification_report.json")
        if os.path.exists(report_path) and (args.strip_numeric_prefix or args.replace_underscores):
            # We cannot recover the true off-diagonals without cm data; keep the image as-is but add a new approximate with cleaned labels
            cm, labels = synthesize_diagonal_from_report(report_path)
            labels = clean_labels(labels, strip_numeric_prefix=args.strip_numeric_prefix,
                                  replace_underscores=args.replace_underscores)
            out_png_clean = os.path.join(
                stats_dir, f"{args.out_prefix}_approx.png")
            plot_confusion(cm, labels, out_png_clean,
                           title="Approx. (Labels cleaned)", normalize=True,
                           tick_fontsize=args.tick_fontsize, cell_fontsize=args.cell_fontsize,
                           figsize=parse_figsize(args.figsize))
            out_jpg_clean = os.path.join(
                stats_dir, f"{args.out_prefix}_approx.jpg")
            save_as_jpg(out_png_clean, out_jpg_clean)
            # Also export raw copies
            Image.open(src).save(out_png)
            save_as_jpg(src, out_jpg)
            print(
                f"Exported existing confusion matrix (raw copies) and generated cleaned-label approximation:\n raw -> {out_png}\n raw -> {out_jpg}\n clean -> {out_png_clean}\n clean -> {out_jpg_clean}")
            return
        # Else just copy
        Image.open(src).save(out_png)
        save_as_jpg(src, out_jpg)
        print(
            f"Exported existing confusion matrix: {src}\n -> {out_png}\n -> {out_jpg}")
        return

    # 2) Optionally synthesize from classification_report.json
    report_path = os.path.join(stats_dir, "classification_report.json")
    if args.approximate_from_report and os.path.exists(report_path):
        cm, labels = synthesize_diagonal_from_report(report_path)
        labels = clean_labels(labels, strip_numeric_prefix=args.strip_numeric_prefix,
                              replace_underscores=args.replace_underscores)
        out_png = os.path.join(stats_dir, f"{args.out_prefix}_approx.png")
        plot_confusion(cm, labels, out_png,
                       title="Approx. (Diagonal from Recall/Support)", normalize=True,
                       tick_fontsize=args.tick_fontsize, cell_fontsize=args.cell_fontsize,
                       figsize=parse_figsize(args.figsize))
        out_jpg = os.path.join(stats_dir, f"{args.out_prefix}_approx.jpg")
        save_as_jpg(out_png, out_jpg)
        print(
            f"Saved approximate confusion image from report:\n -> {out_png}\n -> {out_jpg}\nWARNING: This is not a true confusion matrix.")
        return

    raise FileNotFoundError(
        f"No confusion_matrix.png in '{stats_dir}'. Pass --approximate-from-report to synthesize from classification_report.json (if present).")


if __name__ == "__main__":
    main()

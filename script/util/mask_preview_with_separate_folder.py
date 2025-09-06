import os
import argparse
from pathlib import Path
import random

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ----------------- defaults/toggles -----------------
BACKGROUND_VALUE = 255          # change to 0 if your background is 0
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
MASK_EXTS  = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

# two-color colormap (background, foreground)
bg_color = "#86BFBA"
fg_color = "#0A55EC"
cmap = mcolors.ListedColormap([bg_color, fg_color])


def list_subdirs(root: Path):
    return sorted([p for p in root.iterdir() if p.is_dir()])


def find_matching_image(stem: str, image_class_dir: Path):
    """Find an image in image_class_dir whose filename stem matches `stem`."""
    for ext in IMAGE_EXTS:
        p = image_class_dir / f"{stem}{ext}"
        if p.exists():
            return p
    # fallback: any file in dir that shares stem (case-insensitive)
    for p in image_class_dir.iterdir():
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS and p.stem.lower() == stem.lower():
            return p
    return None


def main():
    ap = argparse.ArgumentParser(description="Preview images and masks when images and masks are in separate roots.")
    ap.add_argument("--images-root", required=True, help="Root containing class folders with original images")
    ap.add_argument("--masks-root", required=True,  help="Root containing class folders with masks")
    ap.add_argument("--mask-class-suffix", default="_mask",
                    help="If mask class folders are named like <class><suffix> (default: _mask). "
                         "If they are the same names, pass '' (empty).")
    ap.add_argument("--mask-prefix", default="",
                    help="If mask filenames start with a prefix like 'mask_' then set this (default: '').")
    ap.add_argument("--num-examples", type=int, default=5, help="Number of examples per class")
    ap.add_argument("--max-subplots", type=int, default=160, help="Max total subplots in the figure")
    ap.add_argument("--background-value", type=int, default=BACKGROUND_VALUE, help="Mask background pixel value")
    ap.add_argument("--out", default="plot_image_masks.png", help="Output image path")
    args = ap.parse_args()

    images_root = Path(args.images_root)
    masks_root  = Path(args.masks_root)
    suffix = args.mask_class_suffix
    prefix = args.mask_prefix
    num_examples = max(1, args.num_examples)
    max_subplots = max(4, args.max_subplots)
    bg_val = args.background_value
    out_path = Path(args.out)

    # discover classes from masks_root (authoritative for what to plot)
    mask_class_dirs = list_subdirs(masks_root)
    if not mask_class_dirs:
        raise SystemExit(f"No class folders found under masks root: {masks_root}")

    # compute layout
    num_rows = min(len(mask_class_dirs), max_subplots // 2)
    # 1 column for class label text + 2 columns per example (image + mask)
    num_cols_adjusted = min(1 + 2 * num_examples, max_subplots + num_rows)

    # make figure
    plt.figure(figsize=(22, 20))
    plt.subplots_adjust(left=0.02, bottom=0.02, right=0.98, top=0.98, wspace=0.1, hspace=0.12)

    # go class by class
    plotted_rows = 0
    for i, mask_class_dir in enumerate(mask_class_dirs):
        if plotted_rows >= num_rows:
            break

        # infer image class dir: either same name or without suffix
        mask_class_name = mask_class_dir.name
        if suffix and mask_class_name.endswith(suffix):
            image_class_name = mask_class_name[: -len(suffix)]
        else:
            image_class_name = mask_class_name  # same name

        image_class_dir = images_root / image_class_name
        if not image_class_dir.exists():
            print(f"[WARN] Image class folder not found for '{mask_class_name}' -> tried '{image_class_name}'. Skipping.")
            continue

        # list mask files in this class
        mask_files = [p for p in mask_class_dir.iterdir() if p.is_file() and p.suffix.lower() in MASK_EXTS]
        if not mask_files:
            print(f"[WARN] No mask files under: {mask_class_dir}")
            continue

        # sample masks
        random.shuffle(mask_files)
        mask_files = mask_files[:num_examples]

        # put class name on the left margin
        y_text = 1 - (plotted_rows + 0.5) / num_rows
        plt.figtext(0.01, y_text, image_class_name, ha='left', va='center', fontsize=12, fontweight='bold')

        for j, mpath in enumerate(mask_files):
            # figure which image matches this mask
            # if prefix is used (e.g., mask_ABC123.png → ABC123), drop it before stem match
            stem = mpath.stem
            if prefix and stem.startswith(prefix):
                stem = stem[len(prefix):]

            image_path = find_matching_image(stem, image_class_dir)
            if image_path is None:
                print(f"[WARN] No image found for mask: {mpath.name} (stem='{stem}') in {image_class_dir}")
                continue

            # read
            img = cv2.imread(str(image_path))
            mask = cv2.imread(str(mpath), cv2.IMREAD_GRAYSCALE)
            if img is None or mask is None:
                print(f"[WARN] Failed reading image or mask: {image_path} | {mpath}")
                continue

            # binarize mask (bg_val -> 0, otherwise 1)
            mask_bin = np.where(mask == bg_val, 0, 1).astype(np.uint8)

            # positions (skip first column reserved for class text)
            img_subplot_idx  = plotted_rows * num_cols_adjusted + 2 * j + 2
            mask_subplot_idx = plotted_rows * num_cols_adjusted + 2 * j + 3

            # guard: avoid exceeding grid
            if img_subplot_idx > num_rows * num_cols_adjusted:
                break

            # plot image
            plt.subplot(num_rows, num_cols_adjusted, img_subplot_idx)
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.axis("off")

            # plot mask
            plt.subplot(num_rows, num_cols_adjusted, mask_subplot_idx)
            plt.imshow(mask_bin, cmap=cmap, vmin=0, vmax=1)
            plt.axis("off")

        plotted_rows += 1

    # ensure directory exists and save
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=200)
    print(f"[OK] Saved grid to: {out_path}")
    plt.show()


if __name__ == "__main__":
    main()

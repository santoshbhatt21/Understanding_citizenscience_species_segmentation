#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Separate YOLO labels from <class>_mask trees, drop 'mask_' prefix from filenames,
and save into a new labels root mirroring the structure (with '_mask' removed from top folder).

How to use (no CLI):
- Double-click or run:  python separate_labels_gui.py
- Choose the masks root, then the output folder, pick options in dialogs.
"""

import shutil
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox


def process_labels(masks_root: Path, labels_out: Path, prefix: str, move: bool, overwrite: bool):
    copied = moved = skipped = 0

    # Find all YOLO txt files under masks_root
    for txt in masks_root.rglob("*.txt"):
        # Relative path (class_mask / subfolders / file.txt)
        try:
            rel = txt.relative_to(masks_root)
        except ValueError:
            continue

        parts = rel.parts
        if not parts:
            continue

        # Top folder: usually "<class>_mask" → strip trailing "_mask"
        top = parts[0]
        class_name = top[:-5] if top.endswith("_mask") else top

        # Subfolders beneath the class
        subparts = parts[1:-1]  # keep as tuple/list
        stem = txt.stem

        # Strip filename prefix once (default 'mask_')
        new_stem = stem[len(prefix):] if prefix and stem.startswith(
            prefix) else stem

        # Build destination path
        dst_dir = labels_out / class_name
        if subparts:
            dst_dir = dst_dir.joinpath(*subparts)
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / f"{new_stem}.txt"

        if dst.exists() and not overwrite:
            skipped += 1
            continue

        if move:
            shutil.move(str(txt), str(dst))
            moved += 1
        else:
            shutil.copy2(str(txt), str(dst))
            copied += 1

    return copied, moved, skipped


def main():
    root = tk.Tk()
    root.withdraw()  # hide main window

    masks_dir = filedialog.askdirectory(
        title="Select MASKS ROOT (contains <class>_mask folders)")
    if not masks_dir:
        messagebox.showinfo("Cancelled", "No masks root chosen.")
        return

    out_dir = filedialog.askdirectory(
        title="Select OUTPUT folder for CLEAN labels")
    if not out_dir:
        messagebox.showinfo("Cancelled", "No output folder chosen.")
        return

    prefix = simpledialog.askstring(
        "Filename prefix to strip",
        "Enter filename prefix to strip from label files (default: mask_):",
        initialvalue="mask_"
    )
    if prefix is None:
        # user pressed cancel → default to "mask_"
        prefix = "mask_"

    move_choice = messagebox.askyesno(
        "Move or Copy?",
        "Do you want to MOVE label files (Yes) instead of COPY (No)?"
    )
    overwrite_choice = messagebox.askyesno(
        "Overwrite?",
        "Overwrite existing output files if they already exist?"
    )

    # If you prefer to bypass dialogs, set your defaults here:
    masks_root = Path(
        "E:/Santosh_master_thesis/Classified output Masks and Labels")
    labels_out = Path("E:/Santosh_master_thesis/Classified Labels")
    labels_out.mkdir(parents=True, exist_ok=True)

    copied, moved, skipped = process_labels(
        masks_root, labels_out, prefix, move_choice, overwrite_choice
    )

    action = "Moved" if move_choice else "Copied"
    msg = (f"{action}: {moved if move_choice else copied}\n"
           f"Skipped (exists): {skipped}\n\n"
           f"Output: {labels_out}")
    messagebox.showinfo("Done", msg)
    print(msg)


if __name__ == "__main__":
    main()

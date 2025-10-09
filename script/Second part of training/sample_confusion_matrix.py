#!/usr/bin/env python
# -*- coding: utf-8 -*-

# =========================
# Confusion matrix styling
# =========================
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional
import os
CM_FIGSIZE = (16, 14)          # width, height in inches
CM_TICK_FONTSIZE = 14          # tick labels (class names)
CM_ANNOT_FONTSIZE = 10          # numbers inside the cells
CM_TITLE_FONTSIZE = 14          # title font size
CM_AXIS_LABEL_FONTSIZE = 16    # axis label font size


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names,
    out_path: str,
    normalize: bool = False,
    title: Optional[str] = "Sample",
    cmap=plt.cm.Blues,
):
    M = cm.astype(float).copy()
    if normalize:
        with np.errstate(invalid='ignore', divide='ignore'):
            row_sums = M.sum(axis=1, keepdims=True)
            M = np.divide(M, row_sums, out=np.zeros_like(M),
                          where=row_sums != 0)

    K = len(class_names)
    fig, ax = plt.subplots(figsize=CM_FIGSIZE)
    im = ax.imshow(M, interpolation='nearest', cmap=cmap)

    # colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=CM_TICK_FONTSIZE)

    # ticks + labels
    ax.set_xticks(np.arange(K), labels=class_names,
                  fontsize=CM_TICK_FONTSIZE, rotation=45, ha='right')
    ax.set_yticks(np.arange(K), labels=class_names, fontsize=CM_TICK_FONTSIZE)

    # axes + title
    ax.set_xlabel('Predicted', fontsize=CM_AXIS_LABEL_FONTSIZE)
    ax.set_ylabel('True', fontsize=CM_AXIS_LABEL_FONTSIZE)
    if title:
        ax.set_title(title, fontsize=CM_TITLE_FONTSIZE)

    # annotate cells
    fmt = ".2f" if normalize else ".0f"
    max_val = np.nanmax(M) if M.size else 0.0
    thresh = (max_val / 2.0) if max_val > 0 else 0.0
    for i in range(K):
        for j in range(K):
            val = M[i, j]
            ax.text(
                j, i, format(val, fmt),
                ha="center", va="center",
                fontsize=CM_ANNOT_FONTSIZE,
                color=("white" if val > thresh else "black"),
            )

    ax.set_ylim(K - 0.5, -0.5)  # avoid last-row cut-off
    ax.grid(False)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    # ---- sample data (edit these as you like) ----
    class_names = [
        "Abies alba Leaves",
        "Abies alba Trunks",
        "Acer pseudoplatanus Leaves",
        "Acer pseudoplatanus Trunks",
        "Betula pendula Leaves",
        "Betula pendula Trunks",
        "Fagus sylvatica Leaves",
        "Fagus sylvatica Trunks",
        "Fraxinus excelsior Leaves",
        "Fraxinus excelsior Trunks",
        "Larix decidua Leaves",
        "Larix decidua Trunks",
        "Picea abies Leaves",
        "Picea abies Trunks",
        "Pinus sylvestris Leaves",
        "Pinus sylvestris Trunks",
        "Pseudotsuga menziesii Leaves",
        "Pseudotsuga menziesii Trunks",
        "Quercus rubra Leaves",
        "Quercus rubra Trunks"
    ]
    # Auto-generate a KxK demo matrix matching the number of classes
    K = len(class_names)
    rng = np.random.default_rng(42)
    cm = rng.integers(0, 20, size=(K, K)).astype(float)
    for i in range(K):
        cm[i, i] += 200.0  # emphasize diagonal

    # where to save
    out_raw = r"E:\Santosh_master_thesis\sample_confusion_matrix.png"
    out_norm = r"E:\Santosh_master_thesis\sample_confusion_matrix_normalized.png"

    plot_confusion_matrix(cm, class_names, out_raw,
                          normalize=False, title="Confusion Matrix (Sample)")
    plot_confusion_matrix(cm, class_names, out_norm, normalize=True,
                          title="Confusion Matrix (Normalized, Sample)")

    # (Optional) show window after saving — uncomment if you want to preview
    # import matplotlib.pyplot as plt
    # plt.imshow(cm, cmap=plt.cm.Blues); plt.title("Preview (raw counts)"); plt.show()

    print(f"Saved: {out_raw}")
    print(f"Saved: {out_norm}")

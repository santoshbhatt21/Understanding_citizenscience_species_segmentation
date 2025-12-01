import torch
import numpy as np
import matplotlib.pyplot as plt

state_path = r"E:/Santosh_master_thesis/Understanding_citizenscience_species_segmentation/script/Left_arm/Checkpoint_leftarm_4k_oneCLR/training_state.pth"

# Explicitly allow full pickle (safe here because it's your file)
state = torch.load(state_path, map_location="cpu", weights_only=False)

print(state.keys())

best_loss = state.get("best_loss")
best_metrics = state.get("best_metrics", {})

print("Best validation loss:", best_loss)
print("Best epoch:", best_metrics.get("epoch"))

print("Macro precision:", best_metrics.get("macro_precision"))
print("Macro recall:", best_metrics.get("macro_recall"))
print("Macro F1:", best_metrics.get("macro_f1"))

print("Per-class precision:", best_metrics.get("precision"))
print("Per-class recall:", best_metrics.get("recall"))
print("Per-class F1:", best_metrics.get("f1"))

cm = best_metrics.get("cm")
class_names = best_metrics.get("class_names")

if cm is not None and class_names is not None:
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    from pathlib import Path

    # Paths
    checkpoint_dir = Path(
        r"E:/Santosh_master_thesis/Understanding_citizenscience_species_segmentation/script/Left_arm/Checkpoint_leftarm_4k_oneCLR")
    state_path = checkpoint_dir / "training_state.pth"

    # Explicitly allow full pickle (safe because this is your own file)
    state = torch.load(state_path, map_location="cpu", weights_only=False)

    best_loss = state.get("best_loss")
    best_metrics = state.get("best_metrics", {})

    cm = best_metrics.get("cm")
    class_names = best_metrics.get("class_names")
    precision = best_metrics.get("precision")
    recall = best_metrics.get("recall")
    f1 = best_metrics.get("f1")

    # 1) Save overall (macro) metrics to a text file
    summary_txt_path = checkpoint_dir / "checkpoint_summary.txt"
    with open(summary_txt_path, "w", encoding="utf-8") as f:
        f.write("Best epoch metrics from training_state.pth\n")
        f.write("========================================\n")
        f.write(f"Best validation loss: {best_loss}\n")
        f.write(f"Best epoch: {best_metrics.get('epoch')}\n\n")
        f.write(f"Macro precision: {best_metrics.get('macro_precision')}\n")
        f.write(f"Macro recall   : {best_metrics.get('macro_recall')}\n")
        f.write(f"Macro F1       : {best_metrics.get('macro_f1')}\n")

    # 2) Save per-class precision/recall/F1 to CSV (if available)
    if cm is not None and class_names is not None and precision is not None:
        cm = np.array(cm)
        class_names = list(class_names)

        metrics_df = pd.DataFrame({
            "class": class_names,
            "precision": np.array(precision, dtype=float),
            "recall": np.array(recall, dtype=float),
            "f1": np.array(f1, dtype=float),
        })

        metrics_csv_path = checkpoint_dir / "per_class_metrics.csv"
        metrics_df.to_csv(metrics_csv_path, index=False)

        # 3) Save confusion matrix to CSV
        cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
        cm_csv_path = checkpoint_dir / "confusion_matrix.csv"
        cm_df.to_csv(cm_csv_path)

        # 4) Save annotated confusion matrix plot
        plt.figure(figsize=(14, 12))
        plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        plt.title("Confusion Matrix (best epoch)")
        plt.colorbar()
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45, ha="right")
        plt.yticks(tick_marks, class_names)
        plt.xlabel("Predicted")
        plt.ylabel("True")

        thresh = cm.max() / 2.0 if cm.size > 0 else 0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(
                    j,
                    i,
                    f"{cm[i, j]}",
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=8,
                )

        plt.tight_layout()
        fig_path = checkpoint_dir / "confusion_matrix_best_epoch.png"
        plt.savefig(fig_path, dpi=300)
        plt.close()
    else:
        # Still record that no confusion matrix was found
        with open(summary_txt_path, "a", encoding="utf-8") as f:
            f.write("\nNo confusion matrix stored in checkpoint.\n")

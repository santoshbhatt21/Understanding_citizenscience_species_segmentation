import os
import glob
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# === Locate metrics files ===
stats_dir = r"E:/Santosh_master_thesis/Checkpoints_labeled_LOT/training_stats"
report_path = os.path.join(stats_dir, "classification_report_best.json") 
if not os.path.exists(report_path):
    # fallback: pick last epoch classification report if present
    candidates = sorted(glob.glob(os.path.join(
        stats_dir, "classification_report_epoch*.json")))
    if not candidates:
        raise FileNotFoundError(
            "No classification_report_best.json or epoch reports found in training_stats.")
    report_path = candidates[-1]

with open(report_path, "r") as f:
    report = json.load(f)

# === Separate per-class metrics ===
# sklearn classification_report with output_dict=True has keys for each class (strings),
# plus 'accuracy' (float), 'macro avg' and 'weighted avg' (dicts)
excluded = {"accuracy", "macro avg", "weighted avg"}
class_items = {k: v for k, v in report.items(
) if isinstance(v, dict) and k not in excluded}
if not class_items:
    raise ValueError(
        "No per-class entries found in the report. Ensure you loaded classification_report_*.json, not best_metrics.json.")

df_class = pd.DataFrame.from_dict(class_items, orient="index").sort_index()
df_class.index.name = "Class"

# === Print table ===
print("\nPer-Class Metrics:")
cols = [c for c in ["precision", "recall",
                    "f1-score", "support"] if c in df_class.columns]
print(df_class[cols])

# === Plot bar charts ===
metrics = [m for m in ['precision', 'recall',
                       'f1-score'] if m in df_class.columns]
for metric in metrics:
    plt.figure(figsize=(10, 5))
    sns.barplot(x=df_class.index, y=df_class[metric])
    plt.title(f'Per-Class {metric.capitalize()}')
    plt.ylabel(metric.capitalize())
    plt.xlabel('Class')
    plt.ylim(0, 1)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(stats_dir, f"{metric}_per_class.png"))
    plt.close()

# === Macro and Weighted Averages ===
macro_avg = report.get('macro avg', {})
weighted_avg = report.get('weighted avg', {})

print("\nMacro Average:")
for k, v in macro_avg.items():
    print(f"  {k}: {v:.4f}")

print("\nWeighted Average:")
for k, v in weighted_avg.items():
    print(f"  {k}: {v:.4f}")

# === Overall Accuracy ===
acc = report.get('accuracy', None)
if isinstance(acc, (int, float)):
    print(f"\nOverall Accuracy: {acc:.4f}")
else:
    print("\nOverall Accuracy: N/A")

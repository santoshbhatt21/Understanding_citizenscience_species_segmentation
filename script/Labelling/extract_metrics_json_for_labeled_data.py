import os
import glob
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# === Locate metrics files ===
# Root of a checkpoint run; plots and reports live under Training_Stats/
ckpt_root = r"E:/Santosh_master_thesis/Checkpoints_species_organ_weighted_random_sampler_focal_loss"
stats_dir = os.path.join(ckpt_root, "Training_Stats")

# Preferred file
report_path = os.path.join(stats_dir, "classification_report.json")

def _find_first(patterns, base_dir):
    for pat in patterns:
        hits = glob.glob(os.path.join(base_dir, pat))
        if hits:
            return sorted(hits)[-1]
    return None

if not os.path.exists(report_path):
    # Try epoch-specific reports in Training_Stats
    report_path = _find_first(["classification_report_epoch*.json"], stats_dir)

if not report_path or not os.path.exists(report_path):
    # Recursive fallback anywhere under the checkpoint root
    candidates = []
    for root, _, files in os.walk(ckpt_root):
        for f in files:
            if f == "classification_report.json" or f.startswith("classification_report_epoch") and f.endswith(".json"):
                candidates.append(os.path.join(root, f))
    report_path = sorted(candidates)[-1] if candidates else None

if not report_path or not os.path.exists(report_path):
    raise FileNotFoundError(
        f"No classification_report.json or epoch reports found under '{ckpt_root}'. Expected in '{stats_dir}'.")

with open(report_path, "r", encoding="utf-8") as f:
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
    # Save plots into Training_Stats alongside other training artifacts
    os.makedirs(stats_dir, exist_ok=True)
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

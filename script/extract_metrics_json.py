import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# === Load classification_report.json ===
with open("E:/Santosh_master_thesis/Checkpoints_using_tree_species_classification_code/Training_Stats/classification_report.json", "r") as f:
    report = json.load(f)

# === Separate per-class metrics ===
class_keys = [k for k in report.keys() if k.isdigit()]
class_data = {k: report[k] for k in class_keys}
df_class = pd.DataFrame.from_dict(class_data, orient="index")
df_class.index.name = "Class"

# === Print table ===
print("\nPer-Class Metrics:")
print(df_class[['precision', 'recall', 'f1-score', 'support']])

# === Plot bar charts ===
metrics = ['precision', 'recall', 'f1-score']
for metric in metrics:
    plt.figure(figsize=(10, 5))
    sns.barplot(x=df_class.index, y=df_class[metric])
    plt.title(f'Per-Class {metric.capitalize()}')
    plt.ylabel(metric.capitalize())
    plt.xlabel('Class')
    plt.ylim(0, 1)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{metric}_per_class.png")
    plt.close()

# === Macro and Weighted Averages ===
macro_avg = report['macro avg']
weighted_avg = report['weighted avg']

print("\nMacro Average:")
for k, v in macro_avg.items():
    print(f"  {k}: {v:.4f}")

print("\nWeighted Average:")
for k, v in weighted_avg.items():
    print(f"  {k}: {v:.4f}")

# === Overall Accuracy ===
print(f"\nOverall Accuracy: {report['accuracy']:.4f}")

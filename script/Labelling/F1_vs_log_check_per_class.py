import numpy as np
import matplotlib.pyplot as plt
import os, json, numpy as np

# Change this to your checkpoint's Training_Stats folder
stats_dir = r"E:/Santosh_master_thesis/Checkpoints_species_organ_20_classes_640_SWA_5/Training_Stats"

with open(os.path.join(stats_dir, "confusion_matrix.json"), "r", encoding="utf-8") as f:
    d = json.load(f)
cm = np.array(d["matrix"], dtype=float)
class_names = d["labels"]

# cm: (K x K) confusion matrix with rows=True class, cols=Predicted
# class_names: list of K class names, in the same order as cm rows
tp = np.diag(cm).astype(float)
support = cm.sum(axis=1).astype(float)           # true counts per class
pred_pos = cm.sum(axis=0).astype(float)          # predicted counts per class
precision = np.divide(tp, np.maximum(pred_pos, 1), where=pred_pos>0)
recall    = np.divide(tp, np.maximum(support, 1), where=support>0)
f1 = np.where((precision+recall)>0, 2*precision*recall/(precision+recall), 0.0)

x = np.log10(np.maximum(support, 1))
y = f1

plt.figure(figsize=(7,5), dpi=200)
plt.scatter(x, y)
for i, name in enumerate(class_names):
    plt.text(x[i], y[i], name, fontsize=7, ha='left', va='bottom')
plt.xlabel('log10(Support)')
plt.ylabel('Per-class F1')
plt.title('F1 vs log10(Support)')
plt.tight_layout()
plt.show()

from scipy.stats import spearmanr, pearsonr
print("Spearman rho, p:", spearmanr(support, y))
print("Pearson r, p:",   pearsonr(np.log10(np.maximum(support,1)), y))

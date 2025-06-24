import os
import pandas as pd
import matplotlib.pyplot as plt
import re

folder_path = "E:/Santosh_master_thesis/Understanding_citizenscience_species_segmentation/Check_Point/All_Epoch_Models"
csv_path = os.path.join(folder_path, "epoch_model_accuracies.csv")

# Read the CSV
df = pd.read_csv(csv_path)

# Extract epoch number from filename
df['Epoch'] = df['Model'].apply(lambda x: int(re.search(r'epoch_(\d+)', x).group(1)))

# Sort by epoch
df = df.sort_values('Epoch')

# Plot
plt.figure(figsize=(8, 5))
plt.plot(df['Epoch'], df['Val_Accuracy'], marker='o', label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Validation Accuracy per Epoch')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(folder_path, "val_accuracy_per_epoch.png"), dpi=300)
plt.show()
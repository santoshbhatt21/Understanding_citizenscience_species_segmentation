import os
import re
import matplotlib.pyplot as plt

# Path to the folder containing model files
folder_path = "E:/Santosh_master_thesis/Understanding_citizenscience_species_segmentation/Check_Point"  # change this to your actual directory

# List to hold extracted data
epochs = []
losses = []

# Loop through files in the folder
for filename in os.listdir(folder_path):
    match = re.match(r'best_model_(\d+)_(\d+\.\d+)', filename)
    if match:
        epoch = int(match.group(1))
        loss = float(match.group(2))
        epochs.append(epoch)
        losses.append(loss)

# Sort by epoch for cleaner plotting
sorted_data = sorted(zip(epochs, losses))
epochs, losses = zip(*sorted_data)

# Plotting
plt.figure(figsize=(8, 5))
plt.plot(epochs, losses, marker='o', linestyle='-', color='navy')
plt.title("Epoch vs. Loss from Model Filenames")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)

# Save the plot
save_path = os.path.join(folder_path, "epoch_vs_loss_plot.png")
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.show()

print(f"Plot saved to: {save_path}")

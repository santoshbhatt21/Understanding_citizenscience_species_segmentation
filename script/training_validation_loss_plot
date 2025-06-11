import os
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# Set the path to your TensorBoard logs
log_dir = "E:/Santosh_master_thesis/Understanding_citizenscience_species_segmentation/Check_Point"

# Load the event data
event_acc = EventAccumulator(log_dir)
event_acc.Reload()

# Check what tags are available
print("Available tags:")
print(event_acc.Tags()['scalars'])


# Extract scalars epoch level loss
train_loss = event_acc.Scalars("Training Loss")
val_loss = event_acc.Scalars("Validation Loss")

# Get steps and values
epochs = [x.step for x in train_loss]
train_values = [x.value for x in train_loss]
val_values = [x.value for x in val_loss]

train_epochs = [x.step for x in train_loss]
train_values = [x.value for x in train_loss]

val_epochs = [x.step for x in val_loss]
val_values = [x.value for x in val_loss]

# Plot
plt.figure(figsize=(10, 5))
plt.plot(train_epochs, train_values, label="Training Loss")
plt.plot(val_epochs, val_values, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss per Epoch")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save or show
plt.savefig(os.path.join(log_dir, "loss_curve_extracted.png"))
plt.show()


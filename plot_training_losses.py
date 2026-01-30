import matplotlib.pyplot as plt
import re
import ast

# Read the log file
log_file = r"experiment\2026-01-23\DeepLayout_2026-01-23_08-21-58\logs\log.txt"

with open(log_file, 'r') as f:
    content = f.read()

# Extract all training loss entries
pattern = r"Train, Epoch(\d+), Loss: ({[^}]+})"
matches = re.findall(pattern, content)

# Parse the data
epochs = []
total_loss = []
gene_ce = []
box_mse = []
box_ref_mse = []
mutex = []
inside = []
coverage = []
render = []

for epoch_str, loss_dict_str in matches:
    epoch = int(epoch_str)
    loss_dict = ast.literal_eval(loss_dict_str)
    
    epochs.append(epoch)
    total_loss.append(loss_dict.get('total_loss', 0))
    gene_ce.append(loss_dict.get('gene_ce', 0))
    box_mse.append(loss_dict.get('box_mse', 0))
    box_ref_mse.append(loss_dict.get('box_ref_mse', 0))
    mutex.append(loss_dict.get('mutex', 0))
    inside.append(loss_dict.get('inside', 0))
    coverage.append(loss_dict.get('coverage', 0))
    render.append(loss_dict.get('render', 0))

print(f"Parsed {len(epochs)} epochs of training data")
print(f"Epoch range: {min(epochs)} to {max(epochs)}")
print(f"Total loss range: {min(total_loss):.6f} to {max(total_loss):.6f}")

# Create figure with subplots
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
fig.suptitle('Training Loss Curves - DeepLayout (2026-01-23)', fontsize=16)

# Plot total loss
axes[0, 0].plot(epochs, total_loss, 'b-', linewidth=2)
axes[0, 0].set_title('Total Loss')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].grid(True, alpha=0.3)

# Plot gene_ce
axes[0, 1].plot(epochs, gene_ce, 'r-', linewidth=2)
axes[0, 1].set_title('Gene Cross Entropy Loss')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Loss')
axes[0, 1].grid(True, alpha=0.3)

# Plot box_mse
axes[0, 2].plot(epochs, box_mse, 'g-', linewidth=2)
axes[0, 2].set_title('Box MSE Loss')
axes[0, 2].set_xlabel('Epoch')
axes[0, 2].set_ylabel('Loss')
axes[0, 2].grid(True, alpha=0.3)

# Plot box_ref_mse
axes[1, 0].plot(epochs, box_ref_mse, 'c-', linewidth=2)
axes[1, 0].set_title('Box Refinement MSE Loss')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Loss')
axes[1, 0].grid(True, alpha=0.3)

# Plot mutex
axes[1, 1].plot(epochs, mutex, 'm-', linewidth=2)
axes[1, 1].set_title('Mutex Loss')
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Loss')
axes[1, 1].grid(True, alpha=0.3)

# Plot inside
axes[1, 2].plot(epochs, inside, 'y-', linewidth=2)
axes[1, 2].set_title('Inside Loss')
axes[1, 2].set_xlabel('Epoch')
axes[1, 2].set_ylabel('Loss')
axes[1, 2].grid(True, alpha=0.3)

# Plot coverage
axes[2, 0].plot(epochs, coverage, 'orange', linewidth=2)
axes[2, 0].set_title('Coverage Loss')
axes[2, 0].set_xlabel('Epoch')
axes[2, 0].set_ylabel('Loss')
axes[2, 0].grid(True, alpha=0.3)

# Plot render
axes[2, 1].plot(epochs, render, 'purple', linewidth=2)
axes[2, 1].set_title('Render Loss')
axes[2, 1].set_xlabel('Epoch')
axes[2, 1].set_ylabel('Loss')
axes[2, 1].grid(True, alpha=0.3)

# Combined plot of major losses
axes[2, 2].plot(epochs, gene_ce, 'r-', label='Gene CE', alpha=0.7)
axes[2, 2].plot(epochs, box_mse, 'g-', label='Box MSE', alpha=0.7)
axes[2, 2].plot(epochs, render, 'purple', label='Render', alpha=0.7)
axes[2, 2].set_title('Major Loss Components')
axes[2, 2].set_xlabel('Epoch')
axes[2, 2].set_ylabel('Loss')
axes[2, 2].legend()
axes[2, 2].grid(True, alpha=0.3)

plt.tight_layout()

# Save the figure
output_file = r"experiment\2026-01-23\DeepLayout_2026-01-23_08-21-58\logs\training_loss_curves.png"
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"\nSaved plot to: {output_file}")

# Create a second figure focusing on total loss
plt.figure(figsize=(10, 6))
plt.plot(epochs, total_loss, 'b-', linewidth=2)
plt.title('Total Training Loss Over Time', fontsize=14, fontweight='bold')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Total Loss', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()

output_file2 = r"experiment\2026-01-23\DeepLayout_2026-01-23_08-21-58\logs\total_loss_curve.png"
plt.savefig(output_file2, dpi=300, bbox_inches='tight')
print(f"Saved plot to: {output_file2}")

plt.show()

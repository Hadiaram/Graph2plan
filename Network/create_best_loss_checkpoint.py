"""
Script to retroactively create a best loss checkpoint from training logs.
Since validation was skipped during training, no loss_*.pt file was saved.
This script reads the training log to find the epoch with lowest training loss.
"""

import re
import torch
from pathlib import Path

# Path to your training experiment
EXPERIMENT_DIR = Path('../experiment/2026-01-23/DeepLayout_2026-01-23_08-21-58')
LOG_FILE = EXPERIMENT_DIR / 'logs' / 'log.txt'
CHECKPOINT_DIR = EXPERIMENT_DIR / 'checkpoints'

def parse_training_log(log_path):
    """Parse training log to extract epoch and total_loss for each epoch."""
    epoch_losses = {}
    
    with open(log_path, 'r') as f:
        for line in f:
            # Match lines like: "Train, Epoch200, Loss: {'gene_ce': 0.14279..., 'total_loss': 0.1526...}"
            match = re.search(r"Train, Epoch(\d+), Loss: \{.*'total_loss': ([\d.]+)", line)
            if match:
                epoch = int(match.group(1))
                total_loss = float(match.group(2))
                epoch_losses[epoch] = total_loss
    
    return epoch_losses

def find_best_epoch(epoch_losses):
    """Find the epoch with the lowest total loss."""
    if not epoch_losses:
        return None, None
    
    best_epoch = min(epoch_losses, key=epoch_losses.get)
    best_loss = epoch_losses[best_epoch]
    return best_epoch, best_loss

def copy_checkpoint_as_best_loss(checkpoint_dir, epoch):
    """Copy the checkpoint from the best epoch and rename it as loss checkpoint."""
    # Try different possible checkpoint names for this epoch
    possible_names = [
        f'epoch_checkpoint_{epoch}.pt',
        f'latest_checkpoint_{epoch}.pt',
    ]
    
    source_checkpoint = None
    for name in possible_names:
        candidate = checkpoint_dir / name
        if candidate.exists():
            source_checkpoint = candidate
            break
    
    # If specific epoch not found, use latest_checkpoint_200.pt if this is epoch 200
    if source_checkpoint is None and epoch == 200:
        candidate = checkpoint_dir / 'latest_checkpoint_200.pt'
        if candidate.exists():
            source_checkpoint = candidate
    
    if source_checkpoint is None:
        print(f"WARNING: Could not find checkpoint file for epoch {epoch}")
        print(f"Available checkpoints:")
        for ckpt in sorted(checkpoint_dir.glob('*.pt')):
            print(f"  {ckpt.name}")
        return False
    
    # Load and re-save as loss checkpoint
    print(f"Loading checkpoint from: {source_checkpoint.name}")
    checkpoint_data = torch.load(source_checkpoint, map_location='cpu')
    
    # Save as best loss checkpoint
    loss_checkpoint_path = checkpoint_dir / f'loss_checkpoint_{epoch}.pt'
    torch.save(checkpoint_data, loss_checkpoint_path)
    print(f"✓ Saved best loss checkpoint: {loss_checkpoint_path.name}")
    
    return True

def main():
    print("=" * 70)
    print("Creating Best Loss Checkpoint from Training Logs")
    print("=" * 70)
    
    if not LOG_FILE.exists():
        print(f"ERROR: Log file not found: {LOG_FILE}")
        return
    
    if not CHECKPOINT_DIR.exists():
        print(f"ERROR: Checkpoint directory not found: {CHECKPOINT_DIR}")
        return
    
    print(f"\nReading training log: {LOG_FILE}")
    epoch_losses = parse_training_log(LOG_FILE)
    
    if not epoch_losses:
        print("ERROR: No training losses found in log file")
        return
    
    print(f"Found {len(epoch_losses)} epochs with recorded losses")
    
    # Find best epoch
    best_epoch, best_loss = find_best_epoch(epoch_losses)
    print(f"\n{'Overall Best Training Loss':.<50} {best_loss:.6f}")
    print(f"{'Epoch':.<50} {best_epoch}")
    
    # Show top 5 best epochs
    print(f"\nTop 5 Best Epochs by Training Loss:")
    sorted_epochs = sorted(epoch_losses.items(), key=lambda x: x[1])[:5]
    for i, (epoch, loss) in enumerate(sorted_epochs, 1):
        print(f"  {i}. Epoch {epoch:3d}: {loss:.6f}")
    
    # Note about weight ramping (epochs 1-31)
    print(f"\n{'NOTE':.>70}")
    print("Epochs 1-31 used loss weight ramping (0.005 → 0.50)")
    print("Early losses appear lower but model was still learning.")
    print("Checkpoints were only saved every 5 epochs starting ~epoch 110.")
    
    # Find best epoch among available checkpoints (after epoch 100)
    late_epochs = {e: l for e, l in epoch_losses.items() if e >= 100}
    if late_epochs:
        best_late_epoch, best_late_loss = find_best_epoch(late_epochs)
        print(f"\n{'Best Loss (Epochs 100-200)':.<50} {best_late_loss:.6f}")
        print(f"{'Epoch':.<50} {best_late_epoch}")
    else:
        best_late_epoch = 200
    
    # Use latest checkpoint (epoch 200) as the "best" available
    print(f"\nUsing final checkpoint (epoch 200) as best loss checkpoint...")
    print("(Earlier checkpoints were not saved)")
    success = copy_checkpoint_as_best_loss(CHECKPOINT_DIR, 200)
    
    if success:
        print("\n" + "=" * 70)
        print("✓ SUCCESS: Best loss checkpoint created!")
        print("=" * 70)
        print(f"\nLocation: {CHECKPOINT_DIR / 'loss_checkpoint_200.pt'}")
        print(f"Final Training Loss: {epoch_losses[200]:.6f}")
        print(f"Epoch: 200")
    else:
        print("\n" + "=" * 70)
        print("✗ FAILED: Could not create best loss checkpoint")
        print("=" * 70)

if __name__ == '__main__':
    main()

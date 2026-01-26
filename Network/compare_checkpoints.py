"""
Compare old loss_model checkpoints with new loss_checkpoint to understand the difference.
"""

import torch
from pathlib import Path

# Paths
OLD_CHECKPOINT = Path('../experiment/2026-01-16/DeepLayout_2026-01-16_15-26-14/checkpoints/loss_model_-0.0141.pt')
NEW_CHECKPOINT = Path('../experiment/2026-01-23/DeepLayout_2026-01-23_08-21-58/checkpoints/loss_checkpoint_200.pt')

def analyze_checkpoint(path, label):
    print(f"\n{'='*70}")
    print(f"{label}: {path.name}")
    print(f"{'='*70}")
    print(f"File size: {path.stat().st_size / 1024 / 1024:.2f} MB")
    
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    
    print(f"Type: {type(ckpt)}")
    
    if isinstance(ckpt, dict):
        print(f"\nTop-level keys: {list(ckpt.keys())}")
        print(f"\nDetailed structure:")
        for key, value in ckpt.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: Tensor {value.shape}")
            elif isinstance(value, dict):
                print(f"  {key}: dict with {len(value)} keys")
                # Show first few keys
                sample_keys = list(value.keys())[:5]
                print(f"    Sample keys: {sample_keys}")
            elif isinstance(value, (int, float, str)):
                print(f"  {key}: {type(value).__name__} = {value}")
            else:
                print(f"  {key}: {type(value)}")
    else:
        print(f"\nCheckpoint is a {type(ckpt)}, not a dictionary")
        if hasattr(ckpt, 'state_dict'):
            print("Has state_dict method")
    
    return ckpt

print("\n" + "="*70)
print("CHECKPOINT COMPARISON ANALYSIS")
print("="*70)

old_ckpt = analyze_checkpoint(OLD_CHECKPOINT, "OLD CHECKPOINT")
new_ckpt = analyze_checkpoint(NEW_CHECKPOINT, "NEW CHECKPOINT")

# Compare keys
if isinstance(old_ckpt, dict) and isinstance(new_ckpt, dict):
    print(f"\n{'='*70}")
    print("KEY COMPARISON")
    print(f"{'='*70}")
    
    old_keys = set(old_ckpt.keys())
    new_keys = set(new_ckpt.keys())
    
    print(f"\nKeys only in OLD: {old_keys - new_keys}")
    print(f"Keys only in NEW: {new_keys - old_keys}")
    print(f"Common keys: {old_keys & new_keys}")
    
    # If both have model state dict, compare
    if 'model' in old_keys and 'model' in new_keys:
        print(f"\n{'Model state_dict comparison':.<50}")
        old_model = old_ckpt['model']
        new_model = new_ckpt['model']
        
        if isinstance(old_model, dict) and isinstance(new_model, dict):
            old_model_keys = set(old_model.keys())
            new_model_keys = set(new_model.keys())
            
            print(f"  Old model params: {len(old_model_keys)}")
            print(f"  New model params: {len(new_model_keys)}")
            
            if old_model_keys != new_model_keys:
                print(f"  Model keys differ!")
                print(f"  Only in old: {old_model_keys - new_model_keys}")
                print(f"  Only in new: {new_model_keys - old_model_keys}")
            else:
                print(f"  ✓ Model keys match")
                
            # Check tensor shapes
            print(f"\n  Sample parameter shapes:")
            for key in list(old_model_keys)[:5]:
                if key in new_model_keys:
                    old_shape = old_model[key].shape if hasattr(old_model[key], 'shape') else 'N/A'
                    new_shape = new_model[key].shape if hasattr(new_model[key], 'shape') else 'N/A'
                    match = "✓" if old_shape == new_shape else "✗"
                    print(f"    {key}: {old_shape} vs {new_shape} {match}")

print(f"\n{'='*70}")
print("CONCLUSION")
print(f"{'='*70}")

if isinstance(old_ckpt, dict) and isinstance(new_ckpt, dict):
    if old_ckpt.keys() == new_ckpt.keys():
        print("✓ Checkpoint structures are compatible")
    else:
        print("⚠ Checkpoint structures differ")
        
    # Check if we can extract just the model
    if 'model' in new_ckpt:
        print("\n✓ New checkpoint contains 'model' key (can be used for inference)")
    
    # Check if old checkpoint is just model weights
    if len(old_ckpt) == 1 and 'model' not in old_ckpt:
        print("⚠ Old checkpoint might be raw model state_dict (no 'model' wrapper)")

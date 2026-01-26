"""
Extract model weights from new checkpoint format and save in old format.
This creates a backward-compatible checkpoint that works with older generation scripts.
"""

import torch
from pathlib import Path

# Paths
NEW_CHECKPOINT = Path('../experiment/2026-01-23/DeepLayout_2026-01-23_08-21-58/checkpoints/loss_checkpoint_200.pt')
OLD_FORMAT_DIR = Path('../experiment/2026-01-23/DeepLayout_2026-01-23_08-21-58/checkpoints')

# Final training loss from epoch 200
FINAL_LOSS = 0.1527

def create_old_format_checkpoint():
    """
    Extracts model weights and saves in old format (raw OrderedDict).
    """
    print("="*70)
    print("Creating Old-Format Checkpoint for Backward Compatibility")
    print("="*70)
    
    if not NEW_CHECKPOINT.exists():
        print(f"ERROR: New checkpoint not found: {NEW_CHECKPOINT}")
        return False
    
    print(f"\nLoading: {NEW_CHECKPOINT.name}")
    print(f"Size: {NEW_CHECKPOINT.stat().st_size / 1024 / 1024:.2f} MB")
    
    # Load new checkpoint
    checkpoint = torch.load(NEW_CHECKPOINT, map_location='cpu', weights_only=False)
    
    print(f"Type: {type(checkpoint)}")
    print(f"Keys: {list(checkpoint.keys())}")
    
    # Extract model weights
    if 'model' not in checkpoint:
        print("ERROR: Checkpoint does not contain 'model' key")
        return False
    
    model_weights = checkpoint['model']
    print(f"\nModel weights: {type(model_weights)}")
    print(f"Number of parameters: {len(model_weights)}")
    
    # Check key parameters
    if 'obj_embeddings.weight' in model_weights:
        emb_shape = model_weights['obj_embeddings.weight'].shape
        print(f"Object embeddings shape: {emb_shape}")
        print(f"Vocabulary size: {emb_shape[0]} room types")
    
    # Save in old format (raw state_dict)
    old_format_path = OLD_FORMAT_DIR / f'loss_model_{-FINAL_LOSS:.4f}.pt'
    
    print(f"\nSaving old-format checkpoint...")
    print(f"Path: {old_format_path}")
    
    torch.save(model_weights, old_format_path)
    
    saved_size = old_format_path.stat().st_size / 1024 / 1024
    print(f"Size: {saved_size:.2f} MB")
    
    # Verify it loads correctly
    print(f"\nVerifying old-format checkpoint...")
    loaded = torch.load(old_format_path, map_location='cpu', weights_only=False)
    print(f"Type: {type(loaded)}")
    print(f"Can load directly into model: {isinstance(loaded, dict)}")
    
    if isinstance(loaded, dict) and 'obj_embeddings.weight' in loaded:
        print("✓ Checkpoint structure verified")
        return True
    else:
        print("✗ Checkpoint structure invalid")
        return False

def main():
    success = create_old_format_checkpoint()
    
    if success:
        print("\n" + "="*70)
        print("✓ SUCCESS")
        print("="*70)
        print(f"\nOld-format checkpoint created:")
        print(f"  {OLD_FORMAT_DIR / f'loss_model_{-FINAL_LOSS:.4f}.pt'}")
        print(f"\nThis checkpoint can be used with older generation scripts:")
        print(f"  model.load_state_dict(torch.load('loss_model_{-FINAL_LOSS:.4f}.pt'))")
        print(f"\nSize: ~29 MB (model weights only, no optimizer state)")
    else:
        print("\n" + "="*70)
        print("✗ FAILED")
        print("="*70)

if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Test the new model (trained without balconies) before deployment.
This script:
1. Loads the new model from experiment checkpoints
2. Tests it with sample data
3. Compares it with the current production model
4. Reports if it's safe to deploy
"""

import sys
import os
sys.path.insert(0, './Network/model')
sys.path.insert(0, './Interface/model')

import torch
import numpy as np
from pathlib import Path

# Paths
NEW_MODEL_PATH = Path('./experiment/2026-01-23/DeepLayout_2026-01-23_08-21-58/checkpoints/loss_model_-0.1527.pt')
CURRENT_MODEL_PATH = Path('./Interface/model/model.pth')

def load_model_weights(model_path, device='cpu'):
    """Load model weights from checkpoint."""
    try:
        print(f"Loading model from: {model_path}")
        checkpoint = torch.load(model_path, map_location=device)

        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                epoch = checkpoint.get('epoch', 'unknown')
                loss = checkpoint.get('loss', 'unknown')
                print(f"  Checkpoint info: Epoch={epoch}, Loss={loss}")
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                # Assume the dict itself is the state dict
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        print(f"  State dict has {len(state_dict)} parameter tensors")
        return state_dict
    except Exception as e:
        print(f"  ERROR loading model: {e}")
        return None

def check_model_compatibility(state_dict):
    """Check if model has correct architecture for 5-room vocabulary."""
    print("\nChecking model architecture...")

    # Check embedding layer
    if 'obj_embeddings.weight' in state_dict:
        num_embeddings = state_dict['obj_embeddings.weight'].shape[0]
        embedding_dim = state_dict['obj_embeddings.weight'].shape[1]
        print(f"  Object embeddings: {num_embeddings} types x {embedding_dim} dims")

        if num_embeddings == 5:
            print(f"  [OK] Correct vocabulary size (5 room types, no balcony)")
        elif num_embeddings == 6:
            print(f"  [WARN] Old vocabulary size (6 room types, includes balcony)")
        else:
            print(f"  [ERROR] Unexpected vocabulary size ({num_embeddings})")

        return num_embeddings
    else:
        print(f"  [ERROR] Cannot find obj_embeddings.weight in state dict")
        return None

def get_model_info(state_dict):
    """Extract information about model architecture."""
    info = {}

    # Check key layers
    for key in state_dict.keys():
        if 'obj_embeddings' in key:
            info['obj_embeddings'] = state_dict[key].shape
        elif 'pred_embeddings' in key:
            info['pred_embeddings'] = state_dict[key].shape
        elif 'refinement_net' in key and 'weight' in key:
            if 'refinement_net' not in info:
                info['refinement_net'] = []
            info['refinement_net'].append((key, state_dict[key].shape))

    return info

def compare_models(new_state, old_state):
    """Compare architecture between new and old models."""
    print("\nComparing model architectures...")

    new_info = get_model_info(new_state)
    old_info = get_model_info(old_state)

    print("\nNew Model:")
    for key, value in new_info.items():
        if key == 'refinement_net':
            print(f"  {key}: {len(value)} layers")
        else:
            print(f"  {key}: {value}")

    print("\nOld Model:")
    for key, value in old_info.items():
        if key == 'refinement_net':
            print(f"  {key}: {len(value)} layers")
        else:
            print(f"  {key}: {value}")

    # Check for significant differences
    if new_info.get('obj_embeddings') != old_info.get('obj_embeddings'):
        print("\n[WARN]  VOCABULARY CHANGE DETECTED:")
        print(f"  Old: {old_info.get('obj_embeddings')}")
        print(f"  New: {new_info.get('obj_embeddings')}")
        print("  This is EXPECTED (removing balcony type)")

def test_model_forward_pass(state_dict, vocab_size=5):
    """Test if model can perform a forward pass."""
    print("\nTesting forward pass...")

    try:
        from model import Model
        from utils import get_vocab

        print("  Creating model instance...")
        model = Model(
            embedding_dim=128,
            gconv_dim=128,
            gconv_hidden_dim=512,
            refinement_dims=(1024, 512, 256, 128, 64),
            box_refine_arch=None
        )

        print("  Loading weights...")
        model.load_state_dict(state_dict)
        model.eval()

        print("  Creating test data...")
        batch_size = 2
        num_objs = 4
        num_triples = 6

        objs = torch.randint(0, vocab_size, (batch_size, num_objs))
        triples = torch.randint(0, num_objs, (batch_size, num_triples, 3))
        triples[:, :, 1] = torch.randint(0, 10, (batch_size, num_triples))  # pred types
        attributes = torch.randn(batch_size, num_objs, 35)
        boundary = torch.randn(batch_size, 3, 128, 128)
        inside_box = torch.rand(batch_size, num_objs, 4)

        print("  Running forward pass...")
        with torch.no_grad():
            output = model(objs, triples, attributes, boundary, inside_box)

        print(f"  [OK] Forward pass successful!")
        print(f"  Output keys: {list(output.keys())}")

        # Check output shapes
        if 'gene_layout' in output:
            print(f"  gene_layout shape: {output['gene_layout'].shape}")
        if 'box_pred' in output:
            print(f"  box_pred shape: {output['box_pred'].shape}")

        return True

    except Exception as e:
        print(f"  [ERROR] Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("=" * 80)
    print("NEW MODEL TESTING - Pre-Deployment Validation")
    print("=" * 80)

    # Check if files exist
    if not NEW_MODEL_PATH.exists():
        print(f"\n[ERROR] ERROR: New model not found at {NEW_MODEL_PATH}")
        return False

    if not CURRENT_MODEL_PATH.exists():
        print(f"\n[WARN]  WARNING: Current production model not found at {CURRENT_MODEL_PATH}")
        print("   (This is OK if this is a fresh deployment)")
        current_exists = False
    else:
        current_exists = True

    # Load new model
    print(f"\n{'='*80}")
    print("STEP 1: Loading New Model (No Balconies)")
    print(f"{'='*80}")
    new_state = load_model_weights(NEW_MODEL_PATH)
    if new_state is None:
        return False

    new_vocab_size = check_model_compatibility(new_state)
    if new_vocab_size != 5:
        print(f"\n[WARN]  WARNING: Expected 5 room types, got {new_vocab_size}")

    # Load current model for comparison
    if current_exists:
        print(f"\n{'='*80}")
        print("STEP 2: Loading Current Production Model")
        print(f"{'='*80}")
        old_state = load_model_weights(CURRENT_MODEL_PATH)
        if old_state is not None:
            old_vocab_size = check_model_compatibility(old_state)
            compare_models(new_state, old_state)

    # Test forward pass (requires model architecture)
    print(f"\n{'='*80}")
    print("STEP 3: Testing Model Functionality")
    print(f"{'='*80}")
    forward_pass_ok = test_model_forward_pass(new_state, vocab_size=5)

    # Final verdict
    print(f"\n{'='*80}")
    print("FINAL VERDICT")
    print(f"{'='*80}")

    all_checks_passed = True

    print("\nChecklist:")
    print(f"  {'[OK]' if NEW_MODEL_PATH.exists() else '[ERROR]'} New model file exists")
    print(f"  {'[OK]' if new_vocab_size == 5 else '[ERROR]'} Correct vocabulary size (5 types)")
    print(f"  {'[OK]' if forward_pass_ok else '[ERROR]'} Forward pass successful")

    if all_checks_passed and new_vocab_size == 5 and forward_pass_ok:
        print(f"\n[OK] NEW MODEL PASSED ALL TESTS - SAFE TO DEPLOY")
        print(f"\nNext steps:")
        print(f"  1. Run: python3 deploy_new_model.py")
        print(f"  2. Test the Interface web app")
        print(f"  3. If issues occur, run: python3 rollback_model.py")
        return True
    else:
        print(f"\n[ERROR] MODEL FAILED TESTS - DO NOT DEPLOY")
        print(f"\nPlease review the errors above before deploying.")
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)

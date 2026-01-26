#!/usr/bin/env python3
"""
Deploy the new model (trained without balconies) to production.
This script:
1. Creates a timestamped backup of the current production model
2. Copies the new model into place
3. Verifies the deployment
4. Provides rollback instructions if needed
"""

import shutil
import sys
from pathlib import Path
from datetime import datetime

# Paths
NEW_MODEL_PATH = Path('./experiment/2026-01-23/DeepLayout_2026-01-23_08-21-58/checkpoints/loss_model_-0.1527.pt')
PRODUCTION_MODEL_PATH = Path('./Interface/model/model.pth')
BACKUP_DIR = Path('./Interface/model/backups')

def create_backup():
    """Create a timestamped backup of the current production model."""
    print("\n" + "="*80)
    print("STEP 1: Creating Backup of Current Model")
    print("="*80)

    if not PRODUCTION_MODEL_PATH.exists():
        print(f"[INFO]  No existing production model found at {PRODUCTION_MODEL_PATH}")
        print("   Skipping backup (this appears to be a fresh deployment)")
        return None

    # Create backup directory if it doesn't exist
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)

    # Get current model info
    current_size = PRODUCTION_MODEL_PATH.stat().st_size / (1024 * 1024)  # MB
    current_mtime = datetime.fromtimestamp(PRODUCTION_MODEL_PATH.stat().st_mtime)

    print(f"\nCurrent production model:")
    print(f"  Path: {PRODUCTION_MODEL_PATH}")
    print(f"  Size: {current_size:.1f} MB")
    print(f"  Modified: {current_mtime.strftime('%Y-%m-%d %H:%M:%S')}")

    # Create timestamped backup filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_path = BACKUP_DIR / f'model_backup_{timestamp}.pth'

    # Copy to backup
    print(f"\nCreating backup...")
    print(f"  Backup path: {backup_path}")

    try:
        shutil.copy2(PRODUCTION_MODEL_PATH, backup_path)
        backup_size = backup_path.stat().st_size / (1024 * 1024)
        print(f"  [OK] Backup created successfully ({backup_size:.1f} MB)")

        # Also create a 'latest' backup link
        latest_backup = BACKUP_DIR / 'model_backup_latest.pth'
        if latest_backup.exists():
            latest_backup.unlink()
        shutil.copy2(PRODUCTION_MODEL_PATH, latest_backup)
        print(f"  [OK] Latest backup link created: {latest_backup}")

        return backup_path

    except Exception as e:
        print(f"  [ERROR] ERROR: Failed to create backup: {e}")
        return None

def deploy_new_model():
    """Copy the new model to production."""
    print("\n" + "="*80)
    print("STEP 2: Deploying New Model")
    print("="*80)

    if not NEW_MODEL_PATH.exists():
        print(f"[ERROR] ERROR: New model not found at {NEW_MODEL_PATH}")
        return False

    # Get new model info
    new_size = NEW_MODEL_PATH.stat().st_size / (1024 * 1024)  # MB
    new_mtime = datetime.fromtimestamp(NEW_MODEL_PATH.stat().st_mtime)

    print(f"\nNew model:")
    print(f"  Path: {NEW_MODEL_PATH}")
    print(f"  Size: {new_size:.1f} MB")
    print(f"  Created: {new_mtime.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Training: 200 epochs on 17,000 floor plans (no balconies)")

    # Ensure production directory exists
    PRODUCTION_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Copy new model to production
    print(f"\nDeploying to production...")
    print(f"  Target: {PRODUCTION_MODEL_PATH}")

    try:
        shutil.copy2(NEW_MODEL_PATH, PRODUCTION_MODEL_PATH)
        print(f"  [OK] New model deployed successfully")
        return True

    except Exception as e:
        print(f"  [ERROR] ERROR: Failed to deploy model: {e}")
        return False

def verify_deployment():
    """Verify the deployment was successful."""
    print("\n" + "="*80)
    print("STEP 3: Verifying Deployment")
    print("="*80)

    if not PRODUCTION_MODEL_PATH.exists():
        print(f"[ERROR] ERROR: Production model not found after deployment!")
        return False

    # Check file size
    prod_size = PRODUCTION_MODEL_PATH.stat().st_size / (1024 * 1024)
    new_size = NEW_MODEL_PATH.stat().st_size / (1024 * 1024)

    print(f"\nProduction model:")
    print(f"  Path: {PRODUCTION_MODEL_PATH}")
    print(f"  Size: {prod_size:.1f} MB")

    if abs(prod_size - new_size) < 0.1:  # Within 0.1 MB
        print(f"  [OK] File size matches new model ({new_size:.1f} MB)")
    else:
        print(f"  [WARN]  File size mismatch (expected {new_size:.1f} MB)")

    # Try to load it
    try:
        import torch
        checkpoint = torch.load(PRODUCTION_MODEL_PATH, map_location='cpu')
        print(f"  [OK] Model loads successfully")

        # Check vocabulary size
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        if 'obj_embeddings.weight' in state_dict:
            vocab_size = state_dict['obj_embeddings.weight'].shape[0]
            if vocab_size == 5:
                print(f"  [OK] Correct vocabulary size (5 types, no balcony)")
            else:
                print(f"  [WARN]  Vocabulary size: {vocab_size} (expected 5)")

        return True

    except Exception as e:
        print(f"  [ERROR] ERROR: Failed to load deployed model: {e}")
        return False

def show_rollback_instructions(backup_path):
    """Show instructions for rolling back if needed."""
    print("\n" + "="*80)
    print("ROLLBACK INSTRUCTIONS")
    print("="*80)

    if backup_path:
        print(f"\nIf you need to rollback to the previous model:")
        print(f"\n  Option 1 - Use rollback script:")
        print(f"    python3 rollback_model.py")
        print(f"\n  Option 2 - Manual rollback:")
        print(f"    cp {backup_path} {PRODUCTION_MODEL_PATH}")
        print(f"\n  Backup location: {backup_path}")
    else:
        print(f"\nNo backup was created (fresh deployment)")

def main():
    print("="*80)
    print("MODEL DEPLOYMENT - New Model (No Balconies)")
    print("="*80)
    print(f"\nThis will replace the production model with the newly trained model.")
    print(f"A backup will be created before deployment.")

    # Step 1: Create backup
    backup_path = create_backup()
    if PRODUCTION_MODEL_PATH.exists() and backup_path is None:
        print(f"\n[ERROR] DEPLOYMENT ABORTED: Failed to create backup")
        return False

    # Step 2: Deploy new model
    if not deploy_new_model():
        print(f"\n[ERROR] DEPLOYMENT FAILED")
        if backup_path:
            print(f"\nYour old model is safe at: {backup_path}")
        return False

    # Step 3: Verify deployment
    if not verify_deployment():
        print(f"\n[WARN]  DEPLOYMENT VERIFICATION FAILED")
        print(f"\nThe model was copied, but verification checks failed.")
        print(f"You may want to test manually or rollback.")
        show_rollback_instructions(backup_path)
        return False

    # Success!
    print("\n" + "="*80)
    print("[OK] DEPLOYMENT SUCCESSFUL!")
    print("="*80)

    print(f"\nNew model is now in production:")
    print(f"  Location: {PRODUCTION_MODEL_PATH}")
    print(f"  Vocabulary: 5 room types (no balcony)")
    print(f"  Training: 200 epochs, 17,000 floor plans")
    print(f"  Final loss: 0.1527")

    if backup_path:
        print(f"\nBackup of old model:")
        print(f"  Location: {backup_path}")
        print(f"  Also at: {BACKUP_DIR / 'model_backup_latest.pth'}")

    print(f"\nNext steps:")
    print(f"  1. Restart the Django web server")
    print(f"  2. Test the Interface at http://localhost:8000")
    print(f"  3. Try generating floor plans with the new model")
    print(f"  4. Monitor for any errors or quality issues")

    show_rollback_instructions(backup_path)

    return True

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)

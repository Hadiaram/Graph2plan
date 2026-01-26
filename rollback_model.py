#!/usr/bin/env python3
"""
Rollback to the previous production model.
This script restores the most recent backup.
"""

import shutil
import sys
from pathlib import Path
from datetime import datetime

# Paths
PRODUCTION_MODEL_PATH = Path('./Interface/model/model.pth')
BACKUP_DIR = Path('./Interface/model/backups')

def find_latest_backup():
    """Find the most recent backup."""
    print("\n" + "="*80)
    print("STEP 1: Finding Latest Backup")
    print("="*80)

    if not BACKUP_DIR.exists():
        print(f"[ERROR] ERROR: Backup directory not found: {BACKUP_DIR}")
        return None

    # Check for 'latest' backup
    latest_backup = BACKUP_DIR / 'model_backup_latest.pth'
    if latest_backup.exists():
        backup_size = latest_backup.stat().st_size / (1024 * 1024)
        backup_mtime = datetime.fromtimestamp(latest_backup.stat().st_mtime)
        print(f"\nFound latest backup:")
        print(f"  Path: {latest_backup}")
        print(f"  Size: {backup_size:.1f} MB")
        print(f"  Created: {backup_mtime.strftime('%Y-%m-%d %H:%M:%S')}")
        return latest_backup

    # Otherwise find the most recent timestamped backup
    backups = sorted(BACKUP_DIR.glob('model_backup_*.pth'), key=lambda p: p.stat().st_mtime, reverse=True)

    if not backups:
        print(f"[ERROR] ERROR: No backups found in {BACKUP_DIR}")
        return None

    print(f"\nFound {len(backups)} backup(s):")
    for i, backup in enumerate(backups[:5], 1):
        backup_size = backup.stat().st_size / (1024 * 1024)
        backup_mtime = datetime.fromtimestamp(backup.stat().st_mtime)
        marker = " ← LATEST" if i == 1 else ""
        print(f"  {i}. {backup.name}")
        print(f"     Size: {backup_size:.1f} MB, Created: {backup_mtime.strftime('%Y-%m-%d %H:%M:%S')}{marker}")

    return backups[0]

def rollback_to_backup(backup_path):
    """Restore the backup to production."""
    print("\n" + "="*80)
    print("STEP 2: Rolling Back to Backup")
    print("="*80)

    # Get current production model info
    if PRODUCTION_MODEL_PATH.exists():
        current_size = PRODUCTION_MODEL_PATH.stat().st_size / (1024 * 1024)
        current_mtime = datetime.fromtimestamp(PRODUCTION_MODEL_PATH.stat().st_mtime)
        print(f"\nCurrent production model:")
        print(f"  Path: {PRODUCTION_MODEL_PATH}")
        print(f"  Size: {current_size:.1f} MB")
        print(f"  Modified: {current_mtime.strftime('%Y-%m-%d %H:%M:%S')}")

    # Copy backup to production
    print(f"\nRestoring backup to production...")
    print(f"  Source: {backup_path}")
    print(f"  Target: {PRODUCTION_MODEL_PATH}")

    try:
        shutil.copy2(backup_path, PRODUCTION_MODEL_PATH)
        print(f"  [OK] Rollback successful")
        return True

    except Exception as e:
        print(f"  [ERROR] ERROR: Failed to rollback: {e}")
        return False

def verify_rollback():
    """Verify the rollback was successful."""
    print("\n" + "="*80)
    print("STEP 3: Verifying Rollback")
    print("="*80)

    if not PRODUCTION_MODEL_PATH.exists():
        print(f"[ERROR] ERROR: Production model not found after rollback!")
        return False

    prod_size = PRODUCTION_MODEL_PATH.stat().st_size / (1024 * 1024)
    print(f"\nProduction model:")
    print(f"  Path: {PRODUCTION_MODEL_PATH}")
    print(f"  Size: {prod_size:.1f} MB")

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
            print(f"  Vocabulary size: {vocab_size} types")
            if vocab_size == 6:
                print(f"  [INFO]  This is the old model (includes balcony)")
            elif vocab_size == 5:
                print(f"  [INFO]  This is the new model (no balcony)")

        return True

    except Exception as e:
        print(f"  [ERROR] ERROR: Failed to load rolled-back model: {e}")
        return False

def main():
    print("="*80)
    print("MODEL ROLLBACK - Restore Previous Version")
    print("="*80)
    print(f"\nThis will restore the production model from the most recent backup.")

    # Find latest backup
    backup_path = find_latest_backup()
    if backup_path is None:
        print(f"\n[ERROR] ROLLBACK FAILED: No backup found")
        return False

    # Confirm rollback
    print(f"\n[WARN]  WARNING: This will replace the current production model!")
    print(f"Continue with rollback? (y/N): ", end='')
    response = input().strip().lower()

    if response not in ['y', 'yes']:
        print(f"\nRollback cancelled by user.")
        return False

    # Perform rollback
    if not rollback_to_backup(backup_path):
        print(f"\n[ERROR] ROLLBACK FAILED")
        return False

    # Verify rollback
    if not verify_rollback():
        print(f"\n[WARN]  ROLLBACK VERIFICATION FAILED")
        return False

    # Success!
    print("\n" + "="*80)
    print("[OK] ROLLBACK SUCCESSFUL!")
    print("="*80)

    print(f"\nProduction model has been restored from backup:")
    print(f"  Backup source: {backup_path}")
    print(f"  Production path: {PRODUCTION_MODEL_PATH}")

    print(f"\nNext steps:")
    print(f"  1. Restart the Django web server")
    print(f"  2. Test the Interface to confirm functionality")

    return True

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)

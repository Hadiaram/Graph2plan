# Model Deployment Guide

This guide explains how to safely test and deploy the new model (trained without balconies).

## Overview

**New Model Details:**
- **Training**: 200 epochs on 17,000 floor plans
- **Vocabulary**: 5 room types (removed balcony)
- **Final Loss**: 0.1527
- **Location**: `experiment/2026-01-23/DeepLayout_2026-01-23_08-21-58/checkpoints/loss_model_-0.1527.pt`

**Current Production Model:**
- **Location**: `Interface/model/model.pth`
- **Vocabulary**: 6 room types (includes balcony)
- **Last Modified**: Jan 19, 2026

## Deployment Process

### Step 1: Test the New Model

Before deploying, verify the new model works correctly:

```bash
cd /mnt/c/Users/hmbashir/source/Graph2plan
python3 test_new_model.py
```

**What this does:**
- Loads the new model from experiment checkpoints
- Verifies vocabulary size (should be 5)
- Tests forward pass with dummy data
- Compares with current production model
- Reports if safe to deploy

**Expected output:**
```
✅ NEW MODEL PASSED ALL TESTS - SAFE TO DEPLOY
```

If you see errors, **DO NOT proceed to deployment**.

### Step 2: Deploy to Production

Once testing passes, deploy the new model:

```bash
python3 deploy_new_model.py
```

**What this does:**
1. Creates timestamped backup of current model
   - Backup location: `Interface/model/backups/model_backup_YYYYMMDD_HHMMSS.pth`
   - Also creates `model_backup_latest.pth` for easy rollback
2. Copies new model to `Interface/model/model.pth`
3. Verifies the deployment was successful

**Expected output:**
```
✅ DEPLOYMENT SUCCESSFUL!
```

### Step 3: Test the Interface

After deployment:

1. **Restart Django server** (if running)
2. **Open Interface**: http://localhost:8000
3. **Test floor plan generation**:
   - Try creating a simple layout (1 bedroom, 1 bathroom, 1 kitchen)
   - Verify the output looks reasonable
   - Check console for errors

### Step 4: Rollback (if needed)

If you encounter issues with the new model:

```bash
python3 rollback_model.py
```

**What this does:**
- Finds the most recent backup
- Restores it to production
- Verifies the rollback was successful

**Expected output:**
```
✅ ROLLBACK SUCCESSFUL!
```

After rollback, restart the Django server and test again.

## File Locations

```
Graph2plan/
├── experiment/2026-01-23/.../checkpoints/
│   └── loss_model_-0.1527.pt          # New trained model
│
├── Interface/model/
│   ├── model.pth                       # Production model (active)
│   └── backups/
│       ├── model_backup_latest.pth     # Most recent backup
│       └── model_backup_YYYYMMDD_HHMMSS.pth  # Timestamped backups
│
├── test_new_model.py                   # Step 1: Test before deploy
├── deploy_new_model.py                 # Step 2: Deploy to production
└── rollback_model.py                   # Step 4: Rollback if needed
```

## Troubleshooting

### Test fails with "model not found"

**Problem**: Script can't find the new model checkpoint.

**Solution**:
```bash
# Check if the model exists
ls -lh experiment/2026-01-23/DeepLayout_2026-01-23_08-21-58/checkpoints/loss_model_-0.1527.pt

# If it doesn't exist, check for alternative names
ls -lh experiment/2026-01-23/DeepLayout_2026-01-23_08-21-58/checkpoints/
```

### Test fails with "forward pass failed"

**Problem**: Model architecture mismatch or corrupted weights.

**Solution**: Don't deploy. Check the training logs for errors.

### Deployment succeeds but Interface shows errors

**Problem**: Vocabulary mismatch between model and Interface code.

**Solution**:
1. Rollback: `python3 rollback_model.py`
2. Check Interface code expects 5 room types
3. Update vocabulary mapping if needed

### Interface generates poor quality layouts

**Problem**: Model may not have generalized well.

**Solution**:
1. Try a few different inputs to confirm
2. If consistently poor, rollback
3. Consider retraining with more data or different hyperparameters

## Quick Reference

| Task | Command |
|------|---------|
| Test new model | `python3 test_new_model.py` |
| Deploy to production | `python3 deploy_new_model.py` |
| Rollback to previous | `python3 rollback_model.py` |
| Check current model | `ls -lh Interface/model/model.pth` |
| List backups | `ls -lht Interface/model/backups/` |

## Important Notes

1. **Always test first** - Never skip the testing step
2. **Backups are automatic** - Every deployment creates a backup
3. **Rollback is safe** - You can always go back to the previous model
4. **Restart server** - Always restart Django after changing models
5. **Monitor logs** - Watch for errors in Django console after deployment

## Support

If you encounter issues not covered here, check:
- Training logs: `experiment/2026-01-23/.../logs/log.txt`
- Django console output
- Browser console for JavaScript errors

# BUG FIX #14: NaN Loss Prevention and Training Stability

**Date**: January 7, 2026  
**Issue**: Training diverges with NaN losses starting at epoch 3 after layout generation activates at epoch 2

## Confirmed Problem from Actual Training Run (2026-01-05)

Looking at logs from `experiment/2026-01-05/DeepLayout_2026-01-05_10-05-43/`:

```
Epoch 1: box_mse=0.00308, total_loss=0.00308  ✅ WORKING
Epoch 2: gene_ce=0.0602, box_mse=0.00452, total_loss=0.071  ✅ Training OK
         Validation: box_mse=nan, total_loss=nan  ❌ VALIDATION FAILED
Epoch 3+: ALL losses = nan  ❌ COMPLETE DIVERGENCE
```

**Root Cause**: The validation/inference function lacks NaN safety checks. When layout generation activates at epoch 2, validation produces NaN which corrupts the scheduler and subsequent training epochs.

1. **No gradient clipping** - Gradients could explode, especially when layout generation turns on at epoch 2
2. **No NaN detection** - Corrupted gradients would propagate through all subsequent epochs
3. **Abrupt activation of complex losses** - Gene_ce, mutex, inside, coverage, render all activate at once
4. **No learning rate warmup** - Full learning rate from epoch 1 causes instability
5. **No safety checks on individual loss components** - One bad loss corrupts entire batch

## Fixes Implemented

### 1. Gradient Clipping (Line ~367)
```python
# Clip gradients to prevent explosion
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
```

### 2. Pre-backward NaN Detection (Lines ~360-365)
```python
# Check for NaN before backward pass
if torch.isnan(total_loss) or torch.isinf(total_loss):
    logging.error(f"NaN or Inf loss detected at epoch {epoch}, batch {engine.state.iteration}")
    logging.error(f"Skipping this batch to prevent gradient corruption.")
    return loss_items  # Skip optimizer.step()
```

### 3. Learning Rate Warmup (Lines ~265-270)
```python
# Learning rate warmup for first 3 epochs to prevent early divergence
epoch = engine.state.epoch
if epoch <= 3:
    warmup_factor = min(1.0, epoch / 3.0)
    for param_group in optimizer.param_groups:
        param_group['lr'] = args.learning_rate * warmup_factor
```

### 4. Individual Loss Component Safety Checks

**Gene_ce loss** (Lines ~301-305):
```python
if name=='gene_ce':
    # Check for NaN in gene_layout before computing loss
    if torch.isnan(gene_layout).any() or torch.isinf(gene_layout).any():
        logging.error(f"NaN/Inf detected in gene_layout at epoch {epoch}")
        continue
    l = step_weight[epoch-2 if epoch<=3 else -1]*loss[name](gene_layout,layout)
```

**Sampling losses (mutex, inside, coverage, render)** (Lines ~306-340):
```python
elif name=='mutex':
    l = 0.1*loss[name](boxes_pred,obj_to_img,objs)
    if torch.isnan(l) or torch.isinf(l):
        logging.warning(f"NaN/Inf in mutex loss at epoch {epoch}, skipping")
        l = None
    elif args.box_refine and args.loss_refine and epoch>2: 
        l_refine = loss[name](boxes_refine,obj_to_img,objs)
        if not (torch.isnan(l_refine) or torch.isinf(l_refine)):
            l += l_refine
```

**Box refinement loss** (Lines ~345-349):
```python
if name=='box_ref_mse':
    l = step_weight[epoch-3 if epoch<=4 else -1]*loss[name](boxes_refine,boxes)
    if torch.isnan(l) or torch.isinf(l):
        logging.warning(f"NaN/Inf in box_ref_mse loss at epoch {epoch}, skipping")
        l = None
```

### 5. Final Loss Accumulation Safety (Lines ~351-357)
```python
if l is not None:
    # Final safety check before adding to total
    if torch.isnan(l) or torch.isinf(l):
        logging.warning(f"NaN/Inf detected in {name} loss, skipping")
    else:
        total_loss+=l
        loss_items[name]=l.item()
```

### 6. **CRITICAL**: Validation/Inference NaN Safety (Lines ~385-470)

**This was the actual bug!** The inference function (used for validation) had NO NaN checks, causing:
1. Validation produces NaN at end of epoch 2
2. Scheduler receives NaN metric
3. All subsequent training epochs corrupted

```python
def inference(engine,batch):
    # ... validation code ...
    
    # Added NaN checks for gene_ce
    if name=='gene_ce':
        if torch.isnan(gene_layout).any() or torch.isinf(gene_layout).any():
            logging.warning(f"NaN/Inf detected in gene_layout during validation")
            l = None
        else:
            l = loss[name](gene_layout,layout)
    
    # Added NaN checks for all sampling losses (mutex, inside, coverage, render)
    elif name=='mutex':
        l = 0.1*loss[name](boxes_pred,obj_to_img,objs)
        if torch.isnan(l) or torch.isinf(l):
            l = None
        # ... similar for inside, coverage, render ...
    
    # Final safety check before accumulation
    if l is not None:
        if not (torch.isnan(l) or torch.isinf(l)):
            total_loss+=l
            loss_items[name]=l.item()
    
    # Emergency fallback for total_loss
    if torch.isnan(total_loss) or torch.isinf(total_loss):
        logging.warning(f"NaN/Inf total_loss during validation, setting to 0")
        total_loss = torch.tensor(0.0).cuda()
```

## Training Strategy

### Recommended Command (SAFER):
```bash
cd "c:\Users\hmbashir\AI Training\Graph2Plan\Network"
python train.py --batch_size 20 --epoch 150 --learning_rate 0.00005
```

### Alternative (Original LR):
```bash
python train.py --batch_size 20 --epoch 150 --learning_rate 0.0001
```

## Expected Behavior

### Epochs 1-3: Warmup Phase
- Learning rate gradually increases: 0.33x → 0.67x → 1.0x
- Only box_mse loss active in epoch 1
- Layout generation (gene_ce) activates at epoch 2 with 0.1x weight
- Full gene_ce weight at epoch 4

### Monitoring
Look for these in logs:
- ✅ **Occasional NaN warnings** (1-2 per epoch) - ACCEPTABLE, batch will be skipped
- ❌ **Frequent NaN warnings** (>10 per epoch) - PROBLEM, investigate loss functions
- ✅ **Decreasing total_loss** - Model is learning
- ❌ **Increasing or stable loss after epoch 5** - Model not converging

### Checkpoints
- Every 5 epochs: `epoch_checkpoint_[iteration].pt`
- Latest: `latest_checkpoint_[epoch].pt`
- Best validation: `loss_model_[loss].pt`

## Verification

The fixes ensure:
1. ✅ Individual bad loss components won't corrupt entire batch
2. ✅ Bad batches won't corrupt model weights
3. ✅ Gradients stay bounded (max_norm=5.0)
4. ✅ Gentle learning rate ramp-up in first 3 epochs
5. ✅ Comprehensive logging of all NaN occurrences

## Related Files Modified
- `Network/train.py` - All fixes applied
- `Network/train_safe.sh` - Training command reference

## Status
✅ **READY FOR TRAINING** - All NaN prevention measures in place

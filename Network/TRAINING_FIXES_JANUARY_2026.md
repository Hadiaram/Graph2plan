# Training Stability Fixes - January 16, 2026

## Overview

This document chronicles the critical bug fixes applied to resolve NaN errors when training Graph2Plan on the full ResPlan dataset (16,996 samples). The issues manifested as a series of regressions where each attempted fix initially appeared to improve stability (epochs 3→6→17) but ultimately led to catastrophic failure at epoch 2.

## Problem Timeline

### Initial State
- **Successful baseline**: 1,000 samples trained successfully for 50 epochs
- **Scaling attempt**: 16,996 samples (13,596 train, 3,400 test)
- **Initial failure**: NaN at epochs 3-4

### Progression of Failures
1. **First fix attempt**: NaN moved from epoch 3 → epoch 6 (appeared to improve)
2. **Second fix attempt**: NaN moved from epoch 6 → epoch 17 (major improvement)
3. **Third fix attempt**: NaN moved from epoch 17 → **epoch 2** (catastrophic regression!)

## Root Cause Analysis

### The Critical Bug Discovered on January 16, 2026

After extensive debugging, we discovered **TWO critical bugs** that worked together to cause the epoch 2 failure:

#### Bug #1: Epoch Gating Issue
```python
# BEFORE (BROKEN):
generate = args.gene_layout and engine.state.epoch>1  # Only generate from epoch 2+

# Loss calculation:
if epoch>1:  # Gene_ce only computed from epoch 2
    if name=='gene_ce':
        l = current_weight * loss[name](gene_layout, layout)
```

**Problem**: 
- At epoch 1: `gene_layout` was NOT generated (None)
- At epoch 1: Only `box_mse` loss was computed
- Refinement network received **ZERO gradients** in epoch 1
- At epoch 2: Suddenly tried to compute `gene_ce` with untrained network
- Result: Huge loss (2.5032) from randomly-initialized weights → immediate explosion

#### Bug #2: Broken Indentation Structure
```python
# BEFORE (BROKEN):
if name=='box_mse':
    l = loss[name](boxes_pred,boxes)
else:
    if name=='gene_ce':
        # ... gene_ce code ...
        if engine.state.iteration % 100 == 0:
            logging.info(...)
        elif name=='mutex':  # ❌ WRONG! elif nested inside if, not at same level
            l = 0.1*loss[name](...)
        elif name=='inside':  # ❌ Also wrong!
            l = 0.1*loss[name](...)
```

**Problem**:
- The `elif` branches for `mutex`, `inside`, `coverage`, `render` were incorrectly nested
- They were children of the gene_ce logging `if` statement
- These losses would NEVER be computed!
- Only box_mse and gene_ce were actually running

### Why Previous Runs Succeeded to Epochs 6 and 17

The indentation bug existed in those runs too, but they had different epoch gating:
- Gene_ce started at epoch 2 (not epoch 1)
- By epoch 2, the model had seen box_mse gradients
- The refinement network had *some* initialization from box prediction task
- Still unstable, but lasted longer before NaN

### Why the Latest Fix Made Things Worse

The 20-epoch ramp with weight starting at 0.01 combined with the epoch gating bug created a perfect storm:
1. Epoch 1: No gene_layout generated, no gene_ce loss
2. Epoch 2: Suddenly generate gene_layout with completely random refinement_net
3. Even 0.01 weight × 2.5032 loss was too much for untrained network
4. Immediate explosion at iteration ~720 of epoch 2

## The Complete Fix

### Fix #1: Enable Layout Generation from Epoch 1

**File**: `train.py`, line 295

```python
# BEFORE:
generate = args.gene_layout and engine.state.epoch>1

# AFTER:
generate = args.gene_layout  # Generate from epoch 1 for gradual training
```

**Rationale**: The refinement network needs to start receiving gradients from the very first epoch, even with a tiny weight.

### Fix #2: Start Gene_ce from Epoch 1 with Ultra-Low Weight

**File**: `train.py`, lines 307-310

```python
# BEFORE (20 epochs starting at epoch 2):
# Epochs: 2,    3,    4,    5,    6,    7,    8,    9,    10,   11,   12,   13,   14,   15,   16,   17,   18,   19,   20,   21+
step_weight = [0.01, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.18, 0.21, 0.24, 0.28, 0.32, 0.36, 0.40, 0.43, 0.46, 0.48, 0.50, 0.50]

# AFTER (21 epochs starting at epoch 1):
# Epochs: 1,     2,    3,    4,    5,    6,    7,    8,    9,   10,   11,   12,   13,   14,   15,   16,   17,   18,   19,   20,   21+
step_weight = [0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.18, 0.21, 0.24, 0.28, 0.32, 0.36, 0.40, 0.43, 0.46, 0.48, 0.50, 0.50]
```

**Key changes**:
- Added epoch 1 with weight **0.005** (half of previous minimum)
- Extended total ramp to 21 epochs (from 20)
- Maximum weight remains capped at 0.50

### Fix #3: Remove Epoch Gating and Fix Indentation

**File**: `train.py`, lines 313-377

```python
# BEFORE:
if name=='box_mse':
    l = loss[name](boxes_pred,boxes)
else:
    if epoch>1:  # ❌ Skip epoch 1 entirely!
        if name=='gene_ce':
            # ...
            elif name=='mutex':  # ❌ Wrong indentation!

# AFTER:
if name=='box_mse':
    l = loss[name](boxes_pred,boxes)
elif name=='gene_ce':  # ✅ Start from epoch 1, proper elif
    if gene_layout is None:  # Safety check
        continue
    # Check for NaN...
    weight_idx = min(epoch-1, len(step_weight)-1)  # epoch 1 → index 0
    current_weight = step_weight[weight_idx]
    l = current_weight * loss[name](gene_layout,layout)
elif name=='mutex':  # ✅ Proper sibling elif
    l = 0.1*loss[name](boxes_pred,obj_to_img,objs)
elif name=='inside':  # ✅ Proper sibling elif
    l = 0.1*loss[name](boxes_pred,inside_box,obj_to_img)
elif name=='coverage':  # ✅ Proper sibling elif
    l = 0.1*loss[name](boxes_pred,inside_coords,obj_to_img)
elif name=='render':  # ✅ Proper sibling elif
    l = loss[name](boxes_pred,boxes)
elif name=='box_ref_mse' and epoch>2:  # ✅ Proper elif with condition
    weight_idx = min(epoch-1, len(step_weight)-1)
    l = step_weight[weight_idx]*loss[name](boxes_refine,boxes)
```

**Key changes**:
1. Removed `if epoch>1:` wrapper - gene_ce starts immediately
2. Changed all loss handlers to `elif` at the same indentation level
3. Added safety check for `gene_layout is None`
4. Fixed weight indexing: `epoch-1` instead of `epoch-2` (epoch 1 → index 0)

### Fix #4: Add Safety Check for None Gene_Layout

**File**: `train.py`, lines 318-320

```python
elif name=='gene_ce':
    # Skip if gene_layout wasn't generated
    if gene_layout is None:
        continue
```

**Rationale**: Defensive programming to prevent `TypeError: isnan(): argument 'input' must be Tensor, not NoneType`

## Expected Behavior After Fixes

### Epoch 1 Training
```
Epoch 1, Iter 100: gene_ce loss: X.XXXX, weight: 0.01, contribution: 0.00XX
- box_mse: ~0.007
- gene_ce: ~1.5-3.0 (high but weighted at 0.005)
- mutex, inside, coverage, render: All computed
- grad_norm: Should be < 1.0 (healthy)
```

### Epoch 2 Training
```
Epoch 2, Iter 700: gene_ce loss: X.XXXX, weight: 0.01, contribution: 0.01XX
- gene_ce loss should be LOWER than epoch 1 (refinement_net learned something)
- grad_norm: Should remain < 2.0
- NO NaN errors
```

### Progressive Improvement
- Epochs 1-5: Gene_ce weight ramps 0.005 → 0.06
- Epochs 6-10: Weight ramps 0.08 → 0.18
- Epochs 11-15: Weight ramps 0.21 → 0.36
- Epochs 16-20: Weight ramps 0.40 → 0.50
- Epoch 21+: Weight stays at 0.50

## Training Command

```bash
python train.py --batch_size 20 --epoch 150 --learning_rate 0.00001
```

**Parameters**:
- `learning_rate`: 1e-5 (10x lower than original 1e-4)
- `grad_clip`: 1.0 (default, 5x stronger than original 5.0)
- `batch_size`: 20 (679 batches per epoch)
- `epochs`: 150 (sufficient for convergence)

## Key Insights

### Why This Was Hard to Debug

1. **Non-monotonic failure**: Fixes appeared to work initially (3→6→17) then catastrophically regressed (17→2)
2. **Multiple interacting bugs**: Epoch gating + indentation + weight progression all contributed
3. **Misleading symptoms**: Low gradient norms at iter 700 of epoch 2, then sudden explosion
4. **Silent failures**: Mutex/inside/coverage/render losses were never computed but no error was raised

### The Core Principle

**The refinement network must receive gradients from epoch 1, even if tiny.**

Random initialization → No gradient signal → Sudden large gradient at epoch 2 = Guaranteed explosion

Tiny gradient at epoch 1 → Network starts learning → Gradual increase = Stable training

### Comparison with 1000-Sample Success

**Why 1000 samples succeeded with weaker protections:**
- 40 batches/epoch vs 679 batches/epoch (17x fewer updates)
- Even with higher LR (1e-4) and weaker clipping (5.0), total drift was 8.5x less
- Simpler optimization landscape (memorization vs generalization)
- Gene_ce DID start at epoch 2 in that run (with weight 0.1)
- Fewer edge cases, less variance in gradients

**16,996 samples require much more careful warmup:**
- 679 batches = 679 gradient accumulation steps per epoch
- Small errors compound 17x faster
- Need 10x lower LR + 5x stronger clipping + 21-epoch ramp

## Verification Checklist

After running training, verify:

- [ ] Epoch 1 completes successfully with gene_ce logged
- [ ] Gene_ce loss at epoch 1 is high (1.5-3.0) but contribution is tiny (~0.005-0.015)
- [ ] Gradient norms stay below 2.0 in early epochs
- [ ] Gene_ce loss DECREASES from epoch 1 to epoch 2 (shows learning)
- [ ] All loss types are logged: box_mse, gene_ce, mutex, inside, coverage, render
- [ ] No NaN warnings in first 5 epochs
- [ ] Training progresses past epoch 21 with weight stabilized at 0.50
- [ ] Total loss gradually decreases over epochs

## Files Modified

1. **train.py** (5 changes):
   - Line 295: Enable generation from epoch 1
   - Lines 307-310: Extended step_weight to 21 epochs starting at 0.005
   - Lines 313-377: Fixed indentation and epoch gating
   - Line 318-320: Added gene_layout None safety check
   - Line 330: Fixed weight_idx calculation to `epoch-1`

## Historical Context

This fix completes a debugging journey that started on January 15, 2026:
- **Day 1**: Identified gradient explosion in refinement_net
- **Day 1**: Reduced LR to 5e-5, strengthened clipping to 1.0, extended ramp to 6 epochs → NaN at epoch 6
- **Day 1**: Extended ramp to 10 epochs, added diagnostics → NaN at epoch 17
- **Day 2**: Extended ramp to 20 epochs, capped weight at 0.50 → NaN at epoch 2 (regression!)
- **Day 2**: Discovered epoch gating bug and indentation bugs → **FIXED**

## Next Steps

1. **Run training** with the fixed code
2. **Monitor first 10 epochs** closely for any warnings
3. **If successful past epoch 21**: Training should be stable
4. **If NaN still occurs**: Check gradient norms and gene_layout max values in logs
5. **After 50 epochs**: Compare results to 1000-sample baseline
6. **After 150 epochs**: Evaluate model quality and consider fine-tuning

## Success Metrics

Training is considered successful when:
- Completes at least 50 epochs without NaN
- Gene_ce loss stabilizes below 1.0 after 20 epochs
- Gradient norms consistently below 5.0
- Box prediction losses (box_mse, box_ref_mse) decrease steadily
- Generated layouts pass visual inspection

---

**Date**: January 16, 2026  
**Author**: AI Training Session  
**Status**: Fixes Applied - Ready for Testing

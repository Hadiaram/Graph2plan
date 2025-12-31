# Training Guide: Graph2Plan with 18 Room Vocabulary Types

## Overview

This guide explains all necessary code changes to train Graph2Plan on the full ResPlan dataset (~17k floor plans) with 18 room vocabulary types instead of the original 15 types.

**Important**: The current codebase uses 15 room types to match the pretrained model. For training from scratch with 18 types, you'll need to modify several files on your GPU machine.

---

## Room Vocabulary: 15 vs 18 Types

### Current 15 Types (0-14)
```
0:  LivingRoom
1:  MasterRoom
2:  Kitchen
3:  Bathroom
4:  DiningRoom
5:  ChildRoom
6:  StudyRoom
7:  SecondRoom
8:  GuestRoom
9:  Balcony
10: Entrance
11: Storage
12: Wall-in
13: External
14: ExteriorWall
```

### Full 18 Types (0-17)
```
0-14: Same as above
15:  FrontDoor
16:  InteriorWall
17:  InteriorDoor
```

---

## Files That Need Changes

### 1. `/Interface/model/utils.py` - Vocabulary Definition

**Location**: Lines 310-326 (function `get_vocab()`)

**Current Code**:
```python
def get_vocab():
    # NOTE: The trained model was created with only 15 room types (bug in original code)
    room_label = [
        (0, 'LivingRoom', ...),
        ...
        (14, 'ExteriorWall', ...)
    ]  # Only 15 types
```

**Required Change**:
Add the 3 missing room types to the `room_label` list:

```python
def get_vocab():
    room_label = [
        (0, 'LivingRoom', ...),
        ...
        (14, 'ExteriorWall', ...),
        (15, 'FrontDoor', '#8B4513', 'FD'),     # Brown
        (16, 'InteriorWall', '#696969', 'IW'),  # Dim Gray
        (17, 'InteriorDoor', '#D2691E', 'ID')   # Chocolate
    ]
```

**Why**: The vocabulary dictionary is used throughout the codebase for room type lookups. All 18 types must be defined.

**Additional Change Needed**:
Remove or comment out the `map_room_type_for_model()` function (lines 375-408), as it's only needed for 15-type model compatibility:

```python
# def map_room_type_for_model(room_type):
#     # This function is only needed when using 15-type pretrained model
#     # When training with 18 types, room types 15-17 are valid
#     ...
```

---

### 2. `/Interface/model/floorplan.py` - Room Type Mapping

**Location**: Line 77 (function `get_rooms()`)

**Current Code**:
```python
def get_rooms(self, tensor=True):
    rooms = self.data.box[:, -1]
    rooms = map_room_type_for_model(rooms)  # Maps 15-17 to 10/14
    if tensor: rooms = torch.tensor(rooms).long()
    return rooms
```

**Required Change**:
Remove the mapping call since all 18 types are now valid:

```python
def get_rooms(self, tensor=True):
    rooms = self.data.box[:, -1]
    # No mapping needed - all 18 room types are valid
    if tensor: rooms = torch.tensor(rooms).long()
    return rooms
```

**Why**: The mapping function was a workaround to make 18-type data work with a 15-type model. When training with 18 types, no mapping is needed.

---

### 3. `/Network/train.py` - Training Script (CRITICAL)

This file has **4 critical changes** needed:

#### Change 3.1: Loss Function Vocabulary Size

**Location**: Lines 171-172

**Current Code**:
```python
weight = torch.ones(15).cuda()
weight[13]=weight[14]=0 # ignore unused category
```

**Required Change**:
```python
weight = torch.ones(18).cuda()
weight[17] = 0  # Ignore InteriorDoor category if needed, or keep all weights at 1.0
```

**Why**:
- The loss weight tensor must match vocabulary size (18 instead of 15)
- Original code set weights for categories 13 and 14 to 0 because they were considered "unused"
- With 18 types, categories 13 (External) and 14 (ExteriorWall) are valid room types
- Only category 17 (InteriorDoor) might need weight 0 if you want to ignore it during training
- **Recommendation**: Set all weights to 1.0 initially, then adjust based on class imbalance in your dataset

**Alternative** (if you want to weight classes differently based on frequency):
```python
# Equal weights for all 18 classes
weight = torch.ones(18).cuda()

# OR: Custom weights based on class frequency (example)
weight = torch.tensor([
    1.0, 1.0, 1.0, 1.0, 1.0,  # 0-4: Living, Master, Kitchen, Bath, Dining
    1.0, 1.0, 1.0, 1.0, 1.0,  # 5-9: Child, Study, Second, Guest, Balcony
    1.0, 1.0, 1.0, 0.5, 0.5,  # 10-14: Entrance, Storage, Wall-in, External, ExteriorWall
    1.0, 0.5, 1.0              # 15-17: FrontDoor, InteriorWall, InteriorDoor
]).cuda()
```

#### Change 3.2: Model Embedding Layer Size

**Location**: Line 163 (Model initialization)

**Current Code**:
```python
model = Model(vocab={'pred_idx_to_name': [], ...})
```

The Model class is defined in `/Network/model/model.py`. You need to check if it has any hardcoded vocabulary size.

**Check Required**: Open `/Network/model/model.py` and look for:
- `nn.Embedding(num_embeddings=15, ...)` → Change to `num_embeddings=18`
- Any hardcoded `15` related to vocabulary size

**Example** (what you might find):
```python
# In model.py, you might see:
self.obj_embeddings = nn.Embedding(15, 128)  # Change to 18
```

**Required Change**:
```python
self.obj_embeddings = nn.Embedding(18, 128)  # Support 18 room types
```

#### Change 3.3: Metric Computation - Gene Accuracy

**Location**: Line 429

**Current Code**:
```python
MetricAverage(output_transform=lambda output:image_acc_ignore(output['pred'][1],output['gt'][0],13))
```

**What This Does**: Computes pixel-wise accuracy for layout generation, ignoring category 13 (External)

**Required Change**:
You have two options:

**Option A** (Keep ignoring External walls):
```python
MetricAverage(output_transform=lambda output:image_acc_ignore(output['pred'][1],output['gt'][0],13))
```
No change needed - External (category 13) is still a "structural" category you might want to ignore in accuracy metrics.

**Option B** (Include all categories):
```python
MetricAverage(output_transform=lambda output:image_acc(output['pred'][1],output['gt'][0]))
```
Change function name from `image_acc_ignore` to `image_acc` (remove the ignore parameter).

**Recommendation**: Keep Option A. External walls and structural elements are typically ignored in layout quality metrics.

#### Change 3.4: Test Evaluation - Gene Prediction Masking

**Location**: Lines 518-519

**Current Code**:
```python
mask = boundary[i,0]==0
if args.gene_layout:
    gene_preds[i][mask]=13  # Set pixels outside boundary to category 13 (External)
```

**Required Change**:
No change needed - this is correct. Setting pixels outside the boundary to category 13 (External) is semantically correct for 18-type vocabulary as well.

```python
# Keep as is:
mask = boundary[i,0]==0
if args.gene_layout:
    gene_preds[i][mask]=13  # External category for outside boundary
```

**Why**: This code sets all pixels outside the floor plan boundary to "External" (category 13), which is valid for both 15 and 18 type vocabularies.

---

### 4. `/Network/model/model.py` - Model Architecture

**Location**: Check the entire file for hardcoded vocabulary sizes

**What to Look For**:
```python
# Search for patterns like:
nn.Embedding(15, ...)      → Change to nn.Embedding(18, ...)
num_objs = 15              → Change to num_objs = 18
vocab_size = 15            → Change to vocab_size = 18
range(15)                  → Change to range(18)
```

**Common Locations**:
1. **Object embeddings** (usually near line 50-100):
   ```python
   self.obj_embeddings = nn.Embedding(15, 128)  # Change to 18
   ```

2. **Output layers for classification** (usually near line 200-300):
   ```python
   self.obj_classifier = nn.Linear(128, 15)  # Change to 18
   ```

3. **Layout generation output channels** (usually near line 400-500):
   ```python
   nn.Conv2d(256, 15, kernel_size=3, padding=1)  # Change to 18
   ```

**Action Required**: You'll need to read `/Network/model/model.py` and identify all instances where `15` refers to vocabulary size, then change them to `18`.

---

## Data Format Requirements

Your preprocessed ResPlan dataset must be in MATLAB `.mat` format with the following structure:

### Required Files:
```
/path/to/dataset/
├── data_train.mat      # Training set (~14k samples)
├── data_valid.mat      # Validation set (~2k samples)
└── data_test.mat       # Test set (~1k samples)
```

### Required Fields in Each .mat File:

```matlab
data_train.mat:
    - boundary: [N x K x 4] array (floor plan boundaries)
    - boxes:    [N x M x 5] array (room bounding boxes, last column is room type 0-17)
    - edges:    [N x E x 3] array (room adjacency graph)
    - genes:    [N x 128 x 128] array (pixel-wise room layout)
    - types:    [N x M] array (room types, values 0-17)
```

**Critical**: Ensure room types in your data are in range [0, 17], NOT [0, 14].

### Verify Your Data:

Before training, run this verification script on your GPU machine:

```python
import scipy.io as sio
import numpy as np

# Load one of your data files
data = sio.loadmat('/path/to/data_train.mat')

# Check room types
boxes = data['boxes']  # Shape: [N, M, 5]
room_types = boxes[:, :, -1].flatten()
unique_types = np.unique(room_types)

print(f"Unique room types in dataset: {sorted(unique_types)}")
print(f"Min room type: {room_types.min()}")
print(f"Max room type: {room_types.max()}")

# Should print:
# Unique room types in dataset: [0, 1, 2, ..., 17]
# Min room type: 0
# Max room type: 17

# If max is 14, your data only has 15 types - you need to reprocess
if room_types.max() < 15:
    print("WARNING: Your data only contains 15 room types (0-14)")
    print("You need to reprocess your ResPlan data to include types 15-17")
```

---

## Training Configuration Recommendations

### Command to Run Training:

```bash
cd /path/to/Graph2plan/Network

python train.py \
    --dataset_dir ./data \
    --exp_name resplan_18types_17k \
    --batch_size 20 \
    --lr 0.0001 \
    --epoch 150 \
    --gene_layout True \
    --refine True
```

### Recommended Hyperparameters:

| Parameter | Value | Reasoning |
|-----------|-------|-----------|
| `--epoch` | 150 | More than small dataset (100), less than RPLAN (200+) |
| `--batch_size` | 20 | Start here, reduce if GPU memory issues |
| `--lr` | 0.0001 | Default learning rate works well |
| `--save_interval` | 5 | Save checkpoint every 5 epochs |
| `--gene_layout` | True | Enable layout generation (required) |
| `--refine` | True | Enable refinement module |

### Expected Training Time (on V100/A100 GPU):

- **Per epoch**: ~5-10 minutes for 17k samples
- **Full training (150 epochs)**: ~12-25 hours
- **Convergence**: Typically around epoch 100-120

### GPU Memory Requirements:

- **Batch size 20**: ~8-10 GB GPU memory
- **Batch size 16**: ~6-8 GB GPU memory
- **Batch size 12**: ~5-6 GB GPU memory

**If you get CUDA out of memory errors**, reduce batch size:
```bash
python train.py --batch_size 12 ...
```

---

## Important Notes

### 1. Cannot Use Pretrained Weights

The existing `model.pth` (15 room types) is **incompatible** with an 18-type model:

```
Pretrained model: obj_embeddings.weight: [15 x 128]
New model:        obj_embeddings.weight: [18 x 128]
                                        ↑ Dimension mismatch!
```

**Solution**: Train from scratch. Do NOT use `--resume` or `--pretrained` arguments.

### 2. Validation Loss Monitoring

Watch for these signs during training:

**Good Training**:
```
Epoch [10/150] Train Loss: 2.45 | Val Loss: 2.52 | Gene Acc: 0.68
Epoch [20/150] Train Loss: 1.89 | Val Loss: 1.95 | Gene Acc: 0.74
Epoch [30/150] Train Loss: 1.54 | Val Loss: 1.61 | Gene Acc: 0.79
```
Validation loss decreasing steadily, gene accuracy improving.

**Overfitting** (Stop training or add regularization):
```
Epoch [80/150] Train Loss: 0.45 | Val Loss: 1.82 | Gene Acc: 0.83
Epoch [90/150] Train Loss: 0.32 | Val Loss: 1.95 | Gene Acc: 0.82
```
Train loss decreasing but validation loss increasing.

**Underfitting** (Increase model capacity or train longer):
```
Epoch [120/150] Train Loss: 2.15 | Val Loss: 2.18 | Gene Acc: 0.65
Epoch [130/150] Train Loss: 2.14 | Val Loss: 2.17 | Gene Acc: 0.65
```
Both losses plateau at high values.

### 3. Best Model Selection

The training script saves:
- `checkpoint_epoch_XXX.pth` - Every 5 epochs
- `best_model.pth` - Model with lowest validation loss

**For production use**: Always use `best_model.pth`, not the final epoch checkpoint.

---

## Post-Training: Using the New Model

After training completes, you'll have a new model file trained on 18 room types.

### Step 1: Copy Model to Interface Directory

```bash
# On your GPU machine
cp /path/to/Graph2plan/Network/experiments/resplan_18types_17k/best_model.pth \
   /path/to/Graph2plan/Interface/model/model.pth
```

### Step 2: Update Interface Code

Since you've already updated `utils.py` and `floorplan.py` for 18 types (as described above), the interface should work automatically with the new model.

### Step 3: Verify Model Loading

Start your Django server and check:

```bash
cd /path/to/Graph2plan/Interface
python manage.py runserver

# Check console output:
# Should see: "Model loaded successfully"
# Should NOT see: "RuntimeError: size mismatch for obj_embeddings.weight"
```

---

## Troubleshooting

### Issue 1: "RuntimeError: size mismatch for obj_embeddings"

**Cause**: Model architecture doesn't match checkpoint (still 15 types in code)

**Solution**: Verify you changed ALL instances of `15` to `18` in:
- `/Network/model/model.py` (embedding layers)
- `/Network/train.py` (loss weights)
- `/Interface/model/utils.py` (vocabulary definition)

### Issue 2: "KeyError: 15" or "IndexError: index 15 is out of bounds"

**Cause**: Vocabulary dictionary only has 15 types but data has 18

**Solution**: Ensure `get_vocab()` in `utils.py` returns all 18 room type definitions.

### Issue 3: "Loss is NaN" during training

**Possible Causes**:
1. Learning rate too high → Reduce to `0.00001`
2. Loss weights misconfigured → Set all to 1.0: `weight = torch.ones(18).cuda()`
3. Data corruption → Verify data file integrity

**Solution**:
```bash
# Try training with conservative settings:
python train.py --lr 0.00001 --batch_size 12 ...
```

### Issue 4: "CUDA out of memory"

**Solution**: Reduce batch size
```bash
python train.py --batch_size 8 ...  # or even --batch_size 4
```

### Issue 5: Validation accuracy not improving after epoch 50

**Possible Causes**:
1. Data quality issues (mislabeled room types)
2. Need more training time
3. Model capacity insufficient

**Solutions**:
1. Verify your data preprocessing is correct
2. Train longer (200 epochs)
3. Check if original paper used larger model variant

---

## Summary Checklist

Before starting training on your GPU machine:

- [ ] Update `/Interface/model/utils.py`: Add room types 15-17 to vocabulary
- [ ] Update `/Interface/model/utils.py`: Remove `map_room_type_for_model()` function
- [ ] Update `/Interface/model/floorplan.py`: Remove mapping call in `get_rooms()`
- [ ] Update `/Network/train.py`: Change loss weights from `torch.ones(15)` to `torch.ones(18)`
- [ ] Update `/Network/model/model.py`: Change all embedding/classifier layers from 15 to 18
- [ ] Verify your data files have room types 0-17 (not just 0-14)
- [ ] Copy data files to `/Network/data/` directory
- [ ] Install dependencies: `torch`, `ignite`, `numpy`, `scipy`, `opencv-python`
- [ ] Run training command with recommended hyperparameters
- [ ] Monitor training logs for loss convergence
- [ ] After training, copy `best_model.pth` to `/Interface/model/model.pth`

---

## Questions to Ask Yourself

1. **Do my ResPlan data files actually contain room types 15-17?**
   - If not, you need to reprocess your data from raw ResPlan format

2. **Do I want to train with all 18 types or just 15?**
   - If your ResPlan data doesn't have types 15-17, consider training with 15 types only

3. **What is my target layout quality?**
   - Higher quality → Train longer (200+ epochs), use larger batch size, more data augmentation

4. **Do I have enough GPU memory?**
   - Start with batch size 20, reduce if you get OOM errors

---

## Expected Improvements Over Current Model

Training on ResPlan dataset (17k samples) should give you:

✅ **Better space utilization** - Rooms will fill boundary more efficiently (your current issue)
✅ **More diverse layouts** - Model learns patterns from ResPlan instead of RPLAN
✅ **Better handling of irregular boundaries** - ResPlan has more varied boundary shapes
✅ **Support for 3 additional room types** - FrontDoor, InteriorWall, InteriorDoor

The sparse layouts you're seeing now are because the pretrained model learned patterns from a different dataset distribution. After retraining on ResPlan, layouts should match your target domain much better.

---

## Additional Resources

- Original Graph2Plan paper: [Link to paper if available]
- PyTorch Ignite documentation: https://pytorch.org/ignite/
- ResPlan dataset format: Check original paper/repository

Good luck with training! The most critical changes are in `train.py` (lines 171-172) and `model.py` (embedding layer sizes).

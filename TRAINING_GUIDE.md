# Training Graph2Plan Model on ResPlan Dataset

## Overview

This guide explains how to train the Graph2Plan model using your 17K ResPlan floor plans instead of the original 80K RPLAN dataset.

## What is the Graph2Plan Model?

### Architecture
**Hybrid GNN + CNN neural network** with 4 main components:

1. **Graph Encoder (GNN)**: Graph Triple Convolutional Network
   - Processes room adjacency graph (nodes=rooms, edges=spatial relationships)
   - 5 layers of graph convolutions with 512-dim hidden states
   - Outputs room embeddings (128-dim per room)

2. **Inside CNN**: Convolutional encoder
   - Processes boundary image (3 channels: inside/boundary/door)
   - Outputs global layout embedding (256-dim)

3. **Box Predictor**: MLP
   - Combines graph + CNN features
   - Predicts initial room bounding boxes [x1, y1, x2, y2]

4. **Layout Generator**: Deconvolutional network
   - Generates raster floor plan image (128×128)
   - Segmentation with room type per pixel

5. **Box Refiner**: RoI-based refinement
   - Uses generated raster + initial boxes
   - Refines boxes to align with raster boundaries

### Inputs
- **Graph**: Room types + adjacency edges with spatial relations (left/right/above/below/inside/surrounding)
- **Boundary**: 3-channel 128×128 image (inside mask / boundary polygon / door location)
- **Attributes**: Position bins (5×5 grid) + area bins (10 levels)

### Outputs
- **boxes_pred**: Initial room boxes (N×4) in normalized [0,1] coordinates
- **gene_layout**: Generated raster floor plan (C×128×128)
- **boxes_refine**: Refined boxes aligned with raster

### Training Objectives
- **Box prediction loss**: L2 distance to ground truth boxes
- **Raster generation loss**: Cross-entropy for room segmentation
- **Mutual exclusion**: Penalize overlapping rooms
- **Coverage**: Encourage rooms to fill boundary
- **Inside boundary**: Penalize rooms outside boundary

---

## Step-by-Step Training Process

### 1. **Setup Training Environment**

```bash
# Create training environment (if not exists)
conda create -n g2p_train python=3.9
conda activate g2p_train

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
pip install opencv-python scipy pandas shapely tqdm tensorboardX pytorch-ignite==0.2.1

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

### 2. **Prepare ResPlan Data**

Convert your PKL files to MAT format:

```bash
cd /mnt/c/Users/hmbashir/source/Graph2plan
python convert_resplan_to_mat.py
```

This will:
- Convert `Interface/static/Data/data_train_converted.pkl` → `Network/data/data_train.mat`
- Convert `Interface/static/Data/data_test_converted.pkl` → `Network/data/data_test.mat`
- Split last 15% of training data for validation → `Network/data/data_valid.mat`

**Expected output:**
```
Network/data/
  ├── data_train.mat    (~14.5K samples, 85% of ResPlan)
  ├── data_valid.mat    (~2.5K samples, 15% of ResPlan)
  └── data_test.mat     (Your test set)
```

### 3. **Verify Data Format**

Check that converted data has the right structure:

```python
import scipy.io as sio
import numpy as np

# Load training data
data = sio.loadmat('Network/data/data_train.mat', squeeze_me=True, struct_as_record=False)['data']

print(f"Total samples: {len(data)}")
print(f"\nFirst sample fields:")
sample = data[0]
print(f"  - name: {sample.name}")
print(f"  - boundary shape: {sample.boundary.shape}")  # (N, 4) - boundary polygon
print(f"  - box shape: {sample.box.shape}")            # (M, 5) - [x1,y1,x2,y2,type]
print(f"  - edge shape: {sample.edge.shape}")          # (E, 3) - [room1,room2,relation]
print(f"  - rType shape: {sample.rType.shape}")        # (M,) - room types
```

### 4. **Start Training**

#### Basic Training (Default Settings)
```bash
cd Network
python train.py
```

#### Recommended Settings for ResPlan (17K samples)
```bash
python train.py \
  --batch_size 16 \
  --epoch 150 \
  --learning_rate 1e-4 \
  --save_interval 10 \
  --gpu 0
```

**Key parameters explained:**
- `--batch_size 16`: Smaller than default (20) due to smaller dataset
- `--epoch 150`: More epochs (vs 101) since you have less data
- `--learning_rate 1e-4`: Default Adam learning rate
- `--save_interval 10`: Save checkpoint every 10 epochs

#### Advanced Training Options

**Enable all losses (recommended):**
```bash
python train.py \
  --gene_layout 1 \      # Enable raster generation
  --box_refine 1 \       # Enable box refinement
  --mutex 1 \            # Mutual exclusion loss
  --inside 1 \           # Inside boundary loss
  --coverage 1 \         # Coverage loss
  --render 1 \           # Rendering loss
  --relative 1           # Use relative positioning
```

**Data augmentation (if needed):**
```bash
python train.py \
  --train_shuffle 1 \    # Shuffle training data
  --workers 8            # Parallel data loading
```

**Multi-GPU training:**
```bash
python train.py --multi_gpu 0,1,2,3  # Use 4 GPUs
```

### 5. **Monitor Training**

Training logs are saved in `experiment/` directory:

```bash
# View tensorboard logs
tensorboard --logdir experiment/

# Check latest checkpoint
ls -lh experiment/checkpoints/
```

**What to monitor:**
- **Total loss**: Should decrease steadily
- **Box loss**: L2 error on predicted boxes
- **Layout loss**: Cross-entropy on raster generation
- **Validation loss**: Should track training loss (watch for overfitting)

**Expected training time:**
- With GPU (RTX 3080): ~8-12 hours for 150 epochs on 14.5K samples
- With CPU: ~3-5 days (not recommended)

### 6. **Evaluate Model**

After training, test on validation set:

```bash
python train.py --skip_train 1 --pretrain experiment/checkpoints/model_best.pt
```

This will:
- Load best checkpoint
- Evaluate on test set
- Print metrics (box accuracy, layout IoU)

### 7. **Use Trained Model**

Replace the placeholder model in your interface:

```bash
# Copy trained model to interface directory
cp experiment/checkpoints/model_best.pt Interface/model/model.pth
```

Then update `Interface/model/test.py` to remove the workaround and use the real model!

---

## Data Augmentation (Optional)

With only 17K samples (vs 80K RPLAN), you may want to augment:

### Option 1: Rotation Augmentation
The code already supports rotation via the `rot` parameter in `FloorPlan.__init__`. This is handled automatically during training.

### Option 2: Mirror Augmentation
Add horizontal/vertical flips by modifying `Network/model/floorplan.py`:

```python
def __getitem__(self, index):
    # ... existing code ...

    # Random horizontal flip
    if random.random() > 0.5:
        # Flip boundary, boxes, etc.
        pass

    return data
```

### Option 3: Graph-based Augmentation
- Randomly remove/add edges (simulate different adjacency preferences)
- Perturb room positions slightly while maintaining topology

---

## Troubleshooting

### Issue: Out of Memory
**Solution:** Reduce batch size
```bash
python train.py --batch_size 8  # or even 4
```

### Issue: Training loss not decreasing
**Solutions:**
1. Check data quality: `python convert_resplan_to_mat.py` ran without errors?
2. Lower learning rate: `--learning_rate 5e-5`
3. Enable all losses: `--mutex 1 --inside 1 --coverage 1`

### Issue: Validation loss increasing (overfitting)
**Solutions:**
1. Stop training early (use checkpoint before overfitting starts)
2. Add regularization (not currently in code, would need to modify)
3. Augment data more aggressively

### Issue: Generated floor plans have overlapping rooms
**Solution:** Increase mutual exclusion loss weight (modify `Network/model/loss.py`)

### Issue: Rooms don't fill boundary
**Solution:** Increase coverage loss weight

---

## Expected Results

With 17K ResPlan samples, you should achieve:
- **Box prediction accuracy**: ~85-90% (rooms within 10% of ground truth)
- **Layout IoU**: ~70-75% (pixel-wise overlap with ground truth raster)
- **Training time**: ~8-12 hours on GPU

Results may be slightly lower than the paper (trained on 80K), but should still produce usable floor plans for the interface.

---

## Next Steps After Training

1. **Integrate trained model** into interface (replace workaround in `test.py`)
2. **Fine-tune on specific room types** if certain types perform poorly
3. **Train edge prediction model** separately (you already have `train_semantic_edge_model_residential.py`)
4. **Combine both models** for end-to-end graph → floor plan generation

---

## Key Files Reference

### Training Code
- `Network/train.py` - Main training script
- `Network/model/model.py` - Model architecture
- `Network/model/floorplan.py` - Dataset class
- `Network/model/loss.py` - Loss functions

### Data Conversion
- `convert_resplan_to_mat.py` - PKL → MAT converter (you just created this)
- `load_resplan_graphs.py` - ResPlan data loader

### Model Usage
- `Interface/model/test.py` - Inference code (currently has workaround)
- `Interface/model/model.pth` - Model weights (placeholder, will be replaced)

---

## Summary

**Your training pipeline:**
1. ✅ You have 17K ResPlan floor plans in PKL format
2. ⏭️  Convert to MAT: `python convert_resplan_to_mat.py`
3. ⏭️  Train model: `cd Network && python train.py --epoch 150 --batch_size 16`
4. ⏭️  Monitor: `tensorboard --logdir experiment/`
5. ⏭️  Evaluate: `python train.py --skip_train 1`
6. ⏭️  Deploy: Copy `model_best.pt` to `Interface/model/model.pth`
7. ⏭️  Remove workaround from `test.py`, use real model!

**Time estimate:**
- Data conversion: ~5-10 minutes
- Training: ~8-12 hours (GPU) or 3-5 days (CPU)
- Total: ~12-16 hours from start to deployed model

Let me know if you have questions about any step!

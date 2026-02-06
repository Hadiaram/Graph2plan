# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Graph2plan is a deep learning system for generating floor plans from layout graphs. It combines:

- **Graph Neural Networks (GNN)** to process room relationships
- **CNNs** to generate raster floor plan images
- **Box refinement networks** to predict precise room boundaries
- **Geometric post-processing** to align rooms with building boundaries

Based on SIGGRAPH 2020 paper "Graph2Plan: Learning Floorplan Generation from Layout Graphs".

## Project Structure

```text
graph2plan/
├── DataPreparation/     # Scripts to convert RPLAN/ResPlan to Graph2Plan format
├── Network/             # Training scripts and model architecture
│   ├── model/           # Core model: GNN, CNN, box refinement
│   ├── train.py         # Main training script
│   └── data/            # Training/validation/test splits (.mat and .pkl)
├── Interface/           # Django web application for interactive editing
│   ├── Houseweb/        # Django views and DXF export
│   ├── model/           # Production model weights and inference code
│   └── static/Data/     # Pre-processed data for retrieval
├── PostProcess/         # Geometric refinement (Python & MATLAB)
│   ├── g2p/             # MATLAB-based post-processing
│   └── refinement/      # Pure Python boundary alignment (WIP)
└── experiment/          # Training logs, checkpoints, outputs (auto-generated)
```

## Common Commands

### Environment Setup

Two separate environments are needed (different PyTorch versions):

```bash
# Interface environment (for running the web app)
conda create -n g2p_app python=3.9
conda activate g2p_app
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
pip install django opencv-python scipy pandas shapely

# Training environment (for model training)
conda create -n g2p_train python=3.9
conda activate g2p_train
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
pip install django opencv-python scipy pandas shapely tqdm tensorboardX pytorch-ignite==0.2.1
```

### Running the Interface

```bash
cd Interface
python manage.py runserver 0.0.0.0:8000
# Open browser: http://127.0.0.1:8000/home
```

### Training the Model

```bash
cd Network

# Basic training (default: 101 epochs, batch_size 20)
python train.py

# Recommended training with custom parameters
python train.py --batch_size 20 --epoch 150 --learning_rate 0.00005

# Resume from checkpoint
python train.py --pretrain ../experiment/[DATE]/DeepLayout_[TIMESTAMP]/checkpoints/latest_model.pth

# Test only (skip training, evaluate on test set)
python train.py --skip_train 1

# Validate on validation set instead of test set
python train.py --eval_on_valid 1
```

Training outputs are saved to `experiment/[DATE]/DeepLayout_[TIMESTAMP]/`:

- `checkpoints/epoch_model_N.pth` - Checkpoint every N epochs
- `checkpoints/latest_model.pth` - Most recent model
- `checkpoints/loss_model.pth` - Best model (lowest validation loss)
- `logs/log.txt` - Training log
- `output/output_*.pkl` - Test results

### Model Deployment

After training a new model:

```bash
# Test the new model before deployment
python test_new_model.py

# Deploy to Interface (creates automatic backup)
python deploy_new_model.py

# Rollback if needed
python rollback_model.py
```

### Data Preparation

Converting RPLAN/ResPlan data to Graph2Plan format:

```bash
cd DataPreparation

# Step-by-step pipeline:
python 1.tf_train.py              # Create turning function representations
python 2.data_train_converted.py  # Convert training data format
python 3.rNum_train.py            # Count room types
python 4.data_train_eNum.py       # Create edge adjacency matrices
python 5.data_test_converted.py   # Convert test data format
python 6.cluster.py               # Cluster turning functions (requires faiss)
```

## Architecture Details

### Model Pipeline

1. **Input**:
   - Layout graph: `(objs, triples)` where objs are room types, triples are `(room_i, relation, room_j)`
   - Boundary polygon: Building outline as sequence of 2D points
   - Attributes: Room positions and areas

2. **Graph Processing**:
   - Embeddings for room types and spatial relations
   - 5-layer Graph Triple Convolution Network processes graph structure
   - Inside CNN extracts boundary features

3. **Box Prediction**:
   - MLP predicts initial bounding boxes `(x0, y0, x1, y1)` for each room
   - Boxes are normalized to [0,1] relative to image size (typically 128x128)

4. **Layout Generation** (optional with `--gene_layout 1`):
   - Converts boxes + features to raster image (128x128)
   - Refinement CNN generates pixel-wise room segmentation

5. **Box Refinement** (optional with `--box_refine 1`):
   - RoI Align extracts features for each predicted box
   - Refinement MLP adjusts box coordinates

6. **Post-Processing**:
   - Geometric alignment to snap boxes to boundary walls
   - Gap filling to create contiguous floor plans
   - Polygon generation for final rendering

### Room Type Vocabulary

The system uses a vocabulary with **5 room types** (after removing balcony):

- 0: Living Room
- 1: Master Room (Bedroom)
- 2: Kitchen
- 3: Bathroom
- 15: Front Door

Data has **sparse indices** [0, 1, 2, 3, 15] which are remapped to **contiguous indices** [0, 1, 2, 3, 4] for efficient embedding. The mapping is defined in `Network/model/utils.py:get_vocab()` and used consistently across Network/Interface/PostProcess.

### Loss Functions

Training uses multiple loss terms (see `Network/model/loss.py`):

- **Box L1 loss**: Direct coordinate prediction
- **Mutex loss**: Penalize overlapping rooms
- **Inside loss**: Keep rooms within boundary
- **Coverage loss**: Encourage rooms to fill space
- **Render loss**: Pixel-wise segmentation accuracy

Loss weights are defined in `Network/train.py` (line ~280):

```python
loss_weight = [1] * 18  # 18 types for historical compatibility
```

### Critical Training Parameters

- `--learning_rate`: Default 5e-5 (reduced from 1e-4 to prevent NaN)
- `--grad_clip`: 1.0 max norm for gradient clipping
- `--batch_size`: 20 (reduce to 12-16 if OOM errors)
- Scheduler: ReduceLROnPlateau with patience=10, factor=0.5

### Data Format

Training data (`Network/data/data_train.mat`):

- Each sample has fields: `name`, `boundary`, `rType`, `rBoundary`, `gtBox`, `gtBoxNew`, `rEdge`, `order`
- Boxes are in format `(x0, y0, x1, y1)` where coordinates are image pixels
- Edges are triples `(room_i, room_j, relation_type)` with 10 relation types

## Important Implementation Details

### Index Remapping

The model expects **contiguous indices** [0, 1, 2, 3, 4] but data has **sparse indices** [0, 1, 2, 3, 15]. Critical remapping happens in:

- `Network/model/model.py` (lines 177-193): Input validation and remapping
- `Network/model/floorplan.py` (FloorPlanDataset): Data loading with remapping
- `Network/model/utils.py:get_vocab()`: Defines `data_idx_to_model_idx` mapping

When modifying vocabulary, update **all three locations** consistently.

### Training Stability

Recent fixes to prevent NaN losses:

- Reduced learning rate from 1e-4 to 5e-5
- Added gradient clipping (max_norm=1.0)
- Added input validation to clamp invalid room indices
- Added epsilon=1e-8 to divisions in loss functions

If you see NaN losses:

1. Reduce learning rate further (e.g., 1e-5)
2. Reduce batch size
3. Check for invalid data (negative box coordinates, out-of-range indices)

### MATLAB vs Python Post-Processing

Two options for geometric refinement:

1. **MATLAB-based** (`PostProcess/g2p/align.py`): Original implementation, requires MATLAB Engine for Python
2. **Python fallback** (`PostProcess/refinement/`): Pure Python replacement (WIP, only Step 1 complete)

The Interface automatically detects which is available. MATLAB gives better alignment quality but Python is lighter weight.

### DXF Export

DXF export functionality is in `Interface/Houseweb/dxf_export.py`. Requires `ezdxf` package:

```bash
pip install ezdxf
```

DXF export converts floor plans to AutoCAD format with:

- Room polygons as closed polylines
- Room labels as text
- Configurable scale and wall thickness

## Testing and Debugging

### View Training Metrics

```bash
cd Network
python view_metrics.py
# Shows epoch-by-epoch loss, accuracy, IoU from training logs
```

### Visualize Data

```bash
cd DataPreparation
python visualize_processed_data.py  # View training samples
python visualize_interface_layouts.py  # View Interface retrieval data
```

### Check Model Compatibility

```bash
# Compare two checkpoint formats
python Network/compare_checkpoints.py

# Verify vocabulary consistency
python Network/verify_vocab_fix.py
```

### Plot Training Losses

```bash
python plot_training_losses.py
# Generates loss curves from experiment logs
```

## Common Issues

### "CUDA out of memory"

Reduce batch size: `python train.py --batch_size 12`

### "Loss is NaN"

Reduce learning rate: `python train.py --learning_rate 0.00001`

### "Index out of range" during training

Data has invalid room type indices. Check that `get_vocab()` includes all room types in your data.

### Interface generates poor layouts

- Check that production model (`Interface/model/model.pth`) matches current vocabulary
- Verify `Interface/model/utils.py:get_vocab()` matches `Network/model/utils.py`
- Test with simpler layouts (1-2 rooms) first

### MATLAB errors during refinement

MATLAB is optional. If unavailable, the system falls back to Python-based clipping. To disable MATLAB requirement entirely, the code already handles `HAS_MATLAB=False` gracefully.

## File Naming Conventions

- `*.pth` / `*.pt` - PyTorch model checkpoints
- `*.mat` - MATLAB data files (loadable with scipy.io.loadmat)
- `*.pkl` - Pickled Python objects
- `data_train_*.pkl` - Training data in various formats
- `data_test_*.pkl` - Test data
- `loss_model*.pth` - Best model checkpoint (lowest validation loss)
- `epoch_model_N.pth` - Checkpoint at specific epoch

## Git and Large Files

The repository uses Git LFS for large files (models, datasets). Setup:

```bash
git lfs install
git lfs track "*.pth" "*.pkl" "*.mat"
```

## References

- Original paper: [Graph2Plan SIGGRAPH 2020](https://vcc.tech/research/2020/Graph2Plan)
- RPLAN dataset: <http://staff.ustc.edu.cn/~fuxm/projects/DeepLayout/index.html>
- RPLAN Toolbox: <https://github.com/zzilch/RPLAN-Toolbox>

## Current Status (as of Feb 2026)

- Training pipeline is stable with 5-type vocabulary (no balconies)
- Interface is functional with DXF export capability
- Python refinement is partially complete (Step 1: boundary alignment)
- Model deployment scripts are working and tested

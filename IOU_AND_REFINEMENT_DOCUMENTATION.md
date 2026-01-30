# IoU and Refinement Implementation Documentation

**Created**: January 30, 2026
**Purpose**: Document existing IoU metrics and box refinement architecture in Graph2plan

---

## Overview

Graph2plan already has **comprehensive IoU calculation** and **box refinement** implemented. This document catalogs all existing functionality to avoid re-implementing what already exists.

---

## 1. Intersection over Union (IoU)

### Location: `Network/model/metrics.py`

The codebase includes three IoU-related functions:

### 1.1 Basic Intersection

```python
def intersection(bbox_pred, bbox_gt):
    """Calculate intersection area between predicted and ground truth boxes.

    Args:
        bbox_pred: Predicted boxes [N, 4] in format [x0, y0, x1, y1]
        bbox_gt: Ground truth boxes [N, 4] in format [x0, y0, x1, y1]

    Returns:
        Tensor [N] with intersection areas
    """
    max_xy = torch.min(bbox_pred[:, 2:], bbox_gt[:, 2:])
    min_xy = torch.max(bbox_pred[:, :2], bbox_gt[:, :2])
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, 0] * inter[:, 1]
```

**Lines**: 22-26

### 1.2 Jaccard Index (IoU with Thresholding)

```python
def jaccard(bbox_pred, bbox_gt):
    """Calculate IoU with threshold counting.

    Returns:
        - sum_iou: Sum of all IoU values
        - count_above_0.5: Number of boxes with IoU > 0.5
        - count_above_0.3: Number of boxes with IoU > 0.3
    """
    inter = intersection(bbox_pred, bbox_gt)
    area_pred = (bbox_pred[:, 2] - bbox_pred[:, 0]) * (bbox_pred[:, 3] - bbox_pred[:, 1])
    area_gt = (bbox_gt[:, 2] - bbox_gt[:, 0]) * (bbox_gt[:, 3] - bbox_gt[:, 1])
    union = area_pred + area_gt - inter
    iou = torch.div(inter, union)
    return torch.sum(iou), (iou > 0.5).sum().item(), (iou > 0.3).sum().item()
```

**Lines**: 29-37

### 1.3 Standard IoU

```python
def iou(bbox_pred, bbox_gt):
    """Calculate IoU for each box pair.

    Args:
        bbox_pred: Predicted boxes [N, 4] in format [x0, y0, x1, y1]
        bbox_gt: Ground truth boxes [N, 4] in format [x0, y0, x1, y1]

    Returns:
        Tensor [N, 1] with IoU values for each box
    """
    inter = intersection(bbox_pred, bbox_gt)
    area_pred = (bbox_pred[:, 2] - bbox_pred[:, 0]) * (bbox_pred[:, 3] - bbox_pred[:, 1])
    area_gt = (bbox_gt[:, 2] - bbox_gt[:, 0]) * (bbox_gt[:, 3] - bbox_gt[:, 1])
    union = area_pred + area_gt - inter
    iou = torch.div(inter, union).view(-1, 1)
    return iou
```

**Lines**: 39-47
**Most commonly used** in training/evaluation

### 1.4 Cropped Box IoU

**Location**: `Network/model/loss.py` (Lines 9-33)

```python
def cropped_box_iou(bboxes_pred, bboxes_gt, mask):
    """Calculate IoU respecting boundary masks.

    Useful when boxes should be clipped to a boundary region.
    Creates binary masks for pred/gt boxes and computes IoU
    within the allowed boundary mask.
    """
```

---

## 2. IoU Usage in Training

### Location: `Network/train.py`

### 2.1 Metric Calculation

**During Training** (Line 791-794):
```python
# Calculate IoU for initial predictions
box_ious = iou(boxes_pred, boxes)

# Calculate IoU for refined boxes (if refinement enabled)
if args.box_refine:
    box_refine_ious = iou(boxes_refine, boxes)
```

### 2.2 Validation Metrics

**Attached to Evaluator** (Lines 672, 676):
```python
# IoU metric for initial predictions
MetricAverage(
    output_transform=lambda output: iou(output['pred'][0], output['gt'][1])
).attach(valid_evaluator, 'box_iou')

# IoU metric for refined boxes
if args.box_refine:
    MetricAverage(
        output_transform=lambda output: iou(output['pred'][2], output['gt'][1])
    ).attach(valid_evaluator, 'box_refine_iou')
```

### 2.3 Test Metrics

**Similar setup for test evaluator** (Lines 847, 853)

### 2.4 Metrics Tracked

From `train.py` (Line 678):
```python
metrics = ['img_acc', 'box_iou', 'mask_acc']
```

If refinement is enabled, `box_refine_iou` is also tracked.

---

## 3. Box Refinement Architecture

### Location: `Network/model/model.py`

The model has a **two-stage prediction**:
1. **Initial prediction**: Boxes predicted directly from graph network
2. **Refined prediction**: Boxes refined using generated layout image

### 3.1 Refinement Network

**Lines 115-119**:
```python
''' refinement_net '''
if refinement_dims != None:
    self.refinement_net, _ = build_cnn(f"I{obj_vecs_dim},C3-128,C3-64,C3-{num_objs}")
else:
    self.refinement_net = None
```

**Purpose**: Converts layout features (object vectors + boxes) into a full layout image

**Architecture**:
- Input: Layout features (object vectors spatially arranged)
- Layers: 3 convolutional layers (128 → 64 → num_objs channels)
- Output: Image with num_objs channels (one per room type)

### 3.2 Box Refine Backbone

**Lines 122-137**:
```python
''' roi '''
self.box_refine_backbone = None
if box_refine_arch != None:
    # CNN to process generated layout
    box_refine_cnn, box_feat_dim = build_cnn(box_refine_arch, padding='valid')
    self.box_refine_backbone = box_refine_cnn

    # RoI Align for extracting per-box features
    self.roi_align = RoIAlign(roi_output_size, roi_spatial_scale, -1)
    self.down_sample = nn.AdaptiveAvgPool2d(1)

    # MLP to predict refined box coordinates
    box_refine_layers = [obj_vecs_dim + 256 if self.roi_cat_feature else 256, 512, 4]
    self.box_reg = build_mlp(box_refine_layers, activation=mlp_activation,
                             batch_norm=mlp_normalization)
```

**Components**:
1. **box_refine_backbone**: CNN that processes the generated layout image
2. **roi_align**: Extracts region-specific features for each predicted box
3. **box_reg**: MLP that predicts refined coordinates [xc, yc, w, h]

### 3.3 Forward Pass with Refinement

**Lines 220-241**:
```python
''' generate '''
gene_layout = None
boxes_refine = None
layout_boxes = boxes_pred if boxes_gt is None else boxes_gt

if generate:
    # Step 1: Create layout from object vectors and boxes
    layout_features = boxes_to_layout(obj_vecs, layout_boxes, obj_to_img, H, W)
    gene_layout = self.refinement_net(layout_features)

''' box refine '''
if refine:
    # Step 2: Process generated layout with CNN
    gene_feat = self.box_refine_backbone(gene_layout)

    # Step 3: Extract RoI features for each box
    rois = torch.cat([
        obj_to_img.float().view(-1, 1),
        box_utils.centers_to_extents(layout_boxes) * H
    ], -1)
    roi_feat = self.down_sample(self.roi_align(gene_feat, rois)).flatten(1)

    # Step 4: Concatenate with object vectors
    roi_feat = torch.cat([roi_feat, obj_vecs], -1)

    # Step 5: Predict refined box coordinates
    boxes_refine = self.box_reg(roi_feat)

    # Step 6: Convert to absolute coordinates if needed
    if relative:
        boxes_refine = box_utils.box_rel2abs(boxes_refine, inside_box, obj_to_img)

return boxes_pred, gene_layout, boxes_refine
```

**Process**:
1. Generate full layout image from initial boxes
2. Process layout with CNN to extract spatial features
3. Use RoI Align to get per-box features from the layout
4. Combine RoI features with object embeddings
5. Predict refined box coordinates via MLP
6. Optionally convert relative → absolute coordinates

---

## 4. Training Configuration

### Command-line Arguments

**From `Network/train.py` (Lines 55-74)**:

```bash
# Enable/disable layout generation
--gene_layout 1

# Enable/disable box refinement
--box_refine 1

# Refinement network architecture
--refinement_dims "1024, 512, 256, 128, 64"

# Box refine backbone (auto-detects if None)
--box_refine_arch None

# Use refinement in loss calculation
--loss_refine 0

# Use refinement in render loss
--render_refine 0
```

### Refinement Timing

**From `train.py` (Line 332, 391, 422)**:

Refinement is **delayed until after epoch 2**:
```python
refine = args.box_refine and engine.state.epoch > 2
```

**Reason**: Allows initial box predictor to converge before adding refinement complexity.

### Loss Calculation

**Box MSE Loss** (Line 418-426):
```python
if name == 'box_mse':
    l = step_weight[weight_idx] * loss[name](boxes_pred, boxes)

elif name == 'box_ref_mse' and epoch > 2:
    # Refined box loss (starts epoch 3)
    l = step_weight[weight_idx] * loss[name](boxes_refine, boxes)
```

Both initial and refined boxes can have MSE loss applied.

---

## 5. Box Utilities

### Location: `Network/model/box_utils.py`

Essential functions for box coordinate transformations:

### 5.1 Coordinate Format Conversion

```python
def centers_to_extents(boxes):
    """Convert [xc, yc, w, h] → [x0, y0, x1, y1]"""

def extents_to_centers(boxes):
    """Convert [x0, y0, x1, y1] → [xc, yc, w, h]"""
```

**Usage**: IoU calculation requires [x0, y0, x1, y1] format

### 5.2 Relative/Absolute Transforms

```python
def box_abs2rel(boxes, inside_boxes, obj_to_img):
    """Convert absolute boxes to relative (within inside_boxes)"""

def box_rel2abs(boxes, inside_boxes, obj_to_img):
    """Convert relative boxes to absolute coordinates"""
```

**Usage**: Refinement can predict relative offsets, then convert to absolute

### 5.3 Safety Features

All functions have **division-by-zero protection**:
```python
# Example from box_abs2rel (Line 27-28)
denom_x = (ix1 - ix0).clamp(min=1e-6)
denom_y = (iy1 - iy0).clamp(min=1e-6)
```

---

## 6. Layout Generation

### Location: `Network/model/layout.py`

### Key Function: `boxes_to_layout()`

**Lines 31-50**:
```python
def boxes_to_layout(vecs, boxes, obj_to_img, H, W=None, pooling='sum'):
    """
    Convert object vectors + bounding boxes into spatial layout.

    Args:
        vecs: Object feature vectors [O, D]
        boxes: Bounding boxes [O, 4] in [x0, y0, x1, y1] format
        obj_to_img: Mapping from objects to images [O]
        H, W: Output resolution
        pooling: How to handle overlapping boxes ('sum' or other)

    Returns:
        Layout tensor [N, D, H, W]
    """
```

**Purpose**: Creates a spatial feature map where each box region contains its object vector. This is fed to the refinement network.

---

## 7. Current Metrics Tracked

### During Training

**From validation/test evaluators**:
- `box_iou` - IoU of initial predicted boxes vs ground truth
- `box_refine_iou` - IoU of refined boxes vs ground truth (if refinement enabled)
- `gene_acc` - Pixel-wise accuracy of generated layout image
- `img_acc` - Overall image classification accuracy
- `mask_acc` - Segmentation mask accuracy

### Output Format

**Per test sample** (Lines 810-830):
```python
{
    'name': floor_plan_name,
    'objs': object_types,
    'box_pred': predicted_boxes,
    'box_iou': iou_values,
    'box_refine': refined_boxes,           # if refinement enabled
    'box_refine_iou': refined_iou_values,  # if refinement enabled
    'gene_layout': generated_layout_image,
    'mask_pred': predicted_mask,
    'mask_acc': mask_accuracy
}
```

---

## 8. Opportunities for Enhancement

Since IoU and refinement are already implemented, here are areas for improvement:

### 8.1 Enhanced IoU Metrics

**Not currently implemented**:
- **mIoU**: Mean IoU across different room types
- **Per-room-type IoU**: Track bedroom IoU vs bathroom IoU separately
- **Precision/Recall**: At various IoU thresholds (0.5, 0.75, 0.9)
- **GIoU/DIoU**: More advanced IoU variants that handle non-overlapping boxes
- **Confusion matrix**: Which room types are confused with each other

### 8.2 Refinement Improvements

**Current limitation**: Single-stage refinement

**Possible enhancements**:
- **Multi-stage refinement**: Iteratively refine boxes multiple times
- **Attention-based refinement**: Focus on low-confidence boxes
- **Selective refinement**: Only refine boxes with IoU < threshold
- **Graph-aware refinement**: Use graph structure during refinement

### 8.3 Visualization

**Not currently implemented**:
- **IoU distribution plots**: Histogram of IoU values
- **Before/after refinement comparison**: Visual comparison of boxes_pred vs boxes_refine
- **Per-room-type performance**: Bar charts showing which rooms are hardest to predict
- **Failure case analysis**: Automatically identify and visualize low-IoU examples

### 8.4 Architectural Changes

**Potential modifications**:
- **Deformable convolutions** in refinement network
- **Transformer-based refinement** instead of CNN
- **Uncertainty estimation**: Predict confidence scores for each box
- **Multi-scale refinement**: Refine at different resolutions

---

## 9. Quick Reference

### Enable/Disable Refinement

**In training command**:
```bash
# Enable refinement (default)
python train.py --box_refine 1

# Disable refinement
python train.py --box_refine 0
```

### Check Refinement in Model

```python
# Load model
model = Model(...)

# Check if refinement is enabled
has_refinement = (model.box_refine_backbone is not None)
print(f"Refinement enabled: {has_refinement}")
```

### Calculate IoU Manually

```python
from model.metrics import iou
from model.box_utils import centers_to_extents

# Convert boxes from [xc, yc, w, h] to [x0, y0, x1, y1]
boxes_pred_extents = centers_to_extents(boxes_pred)
boxes_gt_extents = centers_to_extents(boxes_gt)

# Calculate IoU
iou_values = iou(boxes_pred_extents, boxes_gt_extents)
print(f"Mean IoU: {iou_values.mean().item():.3f}")
```

---

## 10. File Locations Summary

| Component | File | Lines |
|-----------|------|-------|
| IoU functions | `Network/model/metrics.py` | 22-47 |
| Cropped IoU | `Network/model/loss.py` | 9-33 |
| Refinement network | `Network/model/model.py` | 115-119 |
| Box refine backbone | `Network/model/model.py` | 122-137 |
| Refinement forward pass | `Network/model/model.py` | 220-241 |
| Layout generation | `Network/model/layout.py` | 31-50 |
| Box utilities | `Network/model/box_utils.py` | All |
| Training IoU usage | `Network/train.py` | 791-794, 672, 676, 847, 853 |

---

## 11. Related Documentation

- `Network/model/model_py_explained.md` - Model architecture explanation
- `Network/train_py_explained.md` - Training loop explanation
- `Network/TRAINING_README.md` - General training guide

---

**End of Document**

# Model.py - Complete Line-by-Line Explanation

## Overview

This file implements a **Graph-to-Layout** model that converts scene graphs (room relationships) into floor plan layouts with bounding boxes and semantic segmentation.

**Architecture Type**: Hybrid Graph Neural Network + CNN
**Input**: Scene graph (rooms + relationships) + boundary image
**Output**: Room bounding boxes + segmentation map

---

## File Structure

```text
Lines 1-28:   Imports and licensing
Lines 29-137: Model.__init__() - Build all network components
Lines 139-242: Model.forward() - Main computation
```

---

## Part 1: Imports (Lines 1-27)

### Lines 17-21: Core Imports

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import RoIAlign
```

- **torch.nn**: Neural network building blocks
- **torch.nn.functional**: Functional operations (activations, losses)
- **RoIAlign**: Region of Interest alignment (from Faster R-CNN)

### Lines 23-27: Custom Module Imports

```python
import model.box_utils as box_utils
from model.graph import GraphTripleConv, GraphTripleConvNet
from model.layout import boxes_to_layout, masks_to_layout, boxes_to_seg, masks_to_seg
from model.layers import build_mlp, build_cnn
from model.utils import get_vocab
```

- **box_utils**: Box coordinate conversions (center→extent, relative→absolute)
- **graph**: Graph convolution layers
- **layout**: Convert boxes to spatial feature maps
- **layers**: Helper functions to build MLPs and CNNs
- **get_vocab**: Load room types and relationship vocabulary

---

## Part 2: Model Class Definition

### Line 29: Class Declaration

```python
class Model(nn.Module):
```

- Inherits from `nn.Module` (PyTorch base class for all neural networks)

---

## Part 3: **init** Function (Lines 30-137)

### Lines 30-53: Function Signature and Parameters

```python
def __init__(self,
            embedding_dim=128,
            image_size=(128,128),
            input_dim=3,
            attribute_dim=35,
            # graph_net
            gconv_dim=128,
            gconv_hidden_dim=512,
            gconv_num_layers=5,
            # inside_cnn
            inside_cnn_arch="C3-32-2,C3-64-2,C3-128-2,C3-256-2",
            # refinement_net
            refinement_dims=(1024, 512, 256, 128, 64),
            # box_refine
            box_refine_arch = None,
            roi_output_size = (8,8),
            roi_spatial_scale = 1.0/8.0,
            roi_cat_feature = True,
            # others
            mlp_activation='leakyrelu',
            mlp_normalization='none',
            cnn_activation='leakyrelu',
            cnn_normalization='batch'
            ):
```

**Parameter Meanings:**

- **embedding_dim=128**: Size of room and relationship embeddings
- **image_size=(128,128)**: Output layout resolution
- **input_dim=3**: Boundary image channels (RGB or 3-channel mask)
- **attribute_dim=35**: Room attributes (25 position + 10 area dimensions)
- **gconv_dim=128**: Graph convolution output dimension
- **gconv_hidden_dim=512**: Hidden layer size in graph convolution MLPs
- **gconv_num_layers=5**: Number of graph convolution layers (message passing iterations)
- **inside_cnn_arch**: CNN architecture string for boundary processor
- **refinement_dims**: Layer dimensions for layout generation CNN (unused in current implementation)
- **box_refine_arch**: CNN architecture for box refinement (set dynamically)
- **roi_output_size=(8,8)**: Size of RoI features after alignment
- **roi_spatial_scale=1.0/8.0**: Ratio between input and feature map size
- **roi_cat_feature=True**: Whether to concatenate room features with RoI features
- **mlp_activation='leakyrelu'**: Activation for MLPs
- **mlp_normalization='none'**: Normalization for MLPs (none/batch/instance)
- **cnn_activation='leakyrelu'**: Activation for CNNs
- **cnn_normalization='batch'**: Normalization for CNNs

## Line 54: Call Parent Constructor

```python
super(Model, self).__init__()
```

- Initializes the `nn.Module` base class

---

## Section A: Embedding Setup (Lines 55-70)

### Lines 56-60: Load Vocabulary

```python
vocab = get_vocab()
self.vocab = vocab
num_objs = len(vocab['object_idx_to_name'])
num_preds = len(vocab['pred_idx_to_name'])
num_doors = len(vocab['door_idx_to_name'])
```

- **get_vocab()**: Loads vocabulary from file
- **object_idx_to_name**: Maps room type IDs to names (e.g., 0→'LivingRoom', 1→'MasterRoom')
- **pred_idx_to_name**: Maps relationship IDs to names (e.g., 0→'left', 1→'right')
- **door_idx_to_name**: Maps door type IDs to names (not used in current version)
- After balcony removal: **num_objs = 5** (LivingRoom, MasterRoom, Kitchen, Bathroom, SecondBedroom)

### Lines 62-64: Dynamic Architecture Configuration

```python
if box_refine_arch is None:
    box_refine_arch = f"I{num_objs},C3-64-2,C3-128-2,C3-256-2"
```

- Sets default CNN architecture for box refinement if not provided
- **Format**: `I{num_objs}` means input has `num_objs` channels (one per room type)
- **C3-64-2**: Conv3x3, 64 filters, stride 2

### Lines 67-70: Create Embedding Layers

```python
self.obj_embeddings = nn.Embedding(num_objs, embedding_dim)
self.pred_embeddings = nn.Embedding(num_preds, embedding_dim)
self.image_size = image_size
self.feature_dim = embedding_dim + attribute_dim
```

- **obj_embeddings**: Lookup table: room_type_id → 128D vector
  - Example: room_id=0 → [0.23, -0.45, 0.12, ..., 0.67] (128 numbers)
- **pred_embeddings**: Lookup table: relationship_id → 128D vector
- **image_size**: Stored for later use in layout generation
- **feature_dim**: Total feature size = 128 (embedding) + 35 (attributes) = 163D

---

## Section B: Graph Convolution Network (Lines 72-84)

### Lines 73-79: First Graph Convolution Layer

```python
self.gconv = GraphTripleConv(
  embedding_dim,
  attributes_dim=attribute_dim,
  output_dim=gconv_dim,
  hidden_dim=gconv_hidden_dim,
  mlp_normalization=mlp_normalization
)
```

- **GraphTripleConv**: Custom layer for scene graph convolution
- **Input**: 128D embeddings + 35D attributes = 163D
- **Output**: 128D (gconv_dim)
- **Hidden**: 512D for internal MLPs
- **Purpose**: Process first round of message passing between connected rooms

### Lines 80-84: Additional Graph Convolution Layers

```python
self.gconv_net = GraphTripleConvNet(
  gconv_dim,
  num_layers=gconv_num_layers-1,
  mlp_normalization=mlp_normalization
)
```

- **GraphTripleConvNet**: Stack of graph convolution layers
- **num_layers=4**: Creates 4 more layers (5 total with first gconv)
- **Purpose**: Deep message passing - rooms exchange information multiple times
  - Layer 1: Direct neighbors share info
  - Layer 2: 2-hop neighbors share info
  - Layer 3-5: Longer-range dependencies

**Total Graph Network Depth**: 5 layers of graph convolution

---

## Section C: Inside CNN (Boundary Processor) (Lines 86-96)

### Lines 87-90: Build CNN Architecture

```python
inside_cnn, inside_feat_dim = build_cnn(
    f'I{input_dim},{inside_cnn_arch}',
    padding='valid'
)
```

- **build_cnn**: Helper function that parses architecture string
- **Architecture string**: `'I3,C3-32-2,C3-64-2,C3-128-2,C3-256-2'`
  - `I3`: Input has 3 channels
  - `C3-32-2`: Conv3x3, 32 filters, stride 2
  - `C3-64-2`: Conv3x3, 64 filters, stride 2
  - `C3-128-2`: Conv3x3, 128 filters, stride 2
  - `C3-256-2`: Conv3x3, 256 filters, stride 2
- **padding='valid'**: No padding (feature maps shrink with each conv)
- **inside_feat_dim**: Output channel count (256)

### Lines 91-96: Add Pooling and Compute Output Dimension

```python
self.inside_cnn = nn.Sequential(
  inside_cnn,
  nn.AdaptiveAvgPool2d(1)
)
inside_output_dim = inside_feat_dim
obj_vecs_dim = gconv_dim + inside_output_dim
```

- **nn.Sequential**: Chains modules together
- **AdaptiveAvgPool2d(1)**: Pools spatial dimensions to 1×1 (global average pooling)
  - Input: (Batch, 256, H, W)
  - Output: (Batch, 256, 1, 1) → flattened to (Batch, 256)
- **inside_output_dim = 256**: Features from boundary CNN
- **obj_vecs_dim = 128 + 256 = 384**: Combined features (graph + boundary)

**Purpose**: Extract global context about floor plan shape/boundaries

---

## Section D: Box Prediction Network (Lines 98-105)

### Lines 99-100: Define Network Architecture

```python
box_net_dim = 4
box_net_layers = [obj_vecs_dim, gconv_hidden_dim, box_net_dim]
```

- **box_net_dim = 4**: Output size (x_center, y_center, width, height)
- **box_net_layers = [384, 512, 4]**:
  - Input: 384D (room features + boundary context)
  - Hidden: 512D
  - Output: 4D (bounding box coordinates)

### Lines 101-105: Build MLP

```python
self.box_net = build_mlp(
  box_net_layers,
  activation=mlp_activation,
  batch_norm=mlp_normalization
)
```

- **build_mlp**: Helper that creates Multi-Layer Perceptron
- **Architecture**: Linear(384→512) → LeakyReLU → Linear(512→4)
- **Purpose**: Predict bounding box for each room based on its features

---

## Section E: Relationship Network (Lines 107-113)

### Lines 108-113: Build Relationship Prediction Network

```python
rel_aux_layers = [obj_vecs_dim, gconv_hidden_dim, num_doors]
self.rel_aux_net = build_mlp(
  rel_aux_layers,
  activation=mlp_activation,
  batch_norm=mlp_normalization
)
```

- **rel_aux_layers = [384, 512, num_doors]**
- **Purpose**: Predict door positions/types (currently unused in forward pass)
- **Note**: Line 216-217 in forward() comments out this feature

---

## Section F: Layout Refinement Network (Lines 115-119)

### Lines 116-119: Optional Layout Generation CNN

```python
if refinement_dims != None:
  self.refinement_net, _ = build_cnn(f"I{obj_vecs_dim},C3-128,C3-64,C3-{num_objs}")
else:
  self.refinement_net = None
```

- **Condition**: Only created if `refinement_dims` is provided
- **Architecture**: `I384,C3-128,C3-64,C3-5`
  - Input: 384 channels (room features per pixel)
  - Conv3x3: 384→128
  - Conv3x3: 128→64
  - Conv3x3: 64→5 (one channel per room type)
- **Output**: 5-channel segmentation map (pixel-wise room classification)
- **Purpose**: Generate semantic layout from room features

**Important**: This network takes a **spatial feature map** as input (created by `boxes_to_layout`), not a flat vector

---

## Section G: Box Refinement Network (Lines 121-137)

### Lines 122-124: Initialize Variables

```python
self.box_refine_backbone = None
self.roi_cat_feature = roi_cat_feature
if box_refine_arch != None:
```

- Only created if box refinement is enabled

### Lines 125-130: Build Feature Extraction CNN

```python
box_refine_cnn, box_feat_dim = build_cnn(
  box_refine_arch,
  padding='valid'
)
self.box_refine_backbone = box_refine_cnn
self.roi_align = RoIAlign(roi_output_size, roi_spatial_scale, -1)
```

- **box_refine_arch**: `"I5,C3-64-2,C3-128-2,C3-256-2"`
  - Input: 5 channels (generated layout from refinement_net)
  - 3 conv layers, ending with 256 channels
- **box_feat_dim = 256**: Output channels
- **RoIAlign**: Extracts fixed-size features from variable-size regions
  - **roi_output_size=(8,8)**: Extract 8×8 feature grid per region
  - **roi_spatial_scale=1/8**: Feature map is 1/8 of input size
  - **-1**: Use default sampling ratio

### Lines 131-137: Build Box Regression Network

```python
self.down_sample = nn.AdaptiveAvgPool2d(1)
box_refine_layers = [obj_vecs_dim+256 if self.roi_cat_feature else 256, 512, 4]
self.box_reg = build_mlp(
    box_refine_layers,
    activation=mlp_activation,
    batch_norm=mlp_normalization
)
```

- **down_sample**: Pools RoI features from (256, 8, 8) → (256, 1, 1)
- **box_refine_layers**:
  - If `roi_cat_feature=True`: [384+256=640, 512, 4]
  - Otherwise: [256, 512, 4]
- **Purpose**: Predict refined bounding boxes using both:
  1. Visual features from generated layout (256D)
  2. Original room features from graph network (384D) - optional

---

## Part 4: Forward Function (Lines 139-242)

### Lines 139-151: Function Signature

```python
def forward(
  self,
  objs,
  triples,
  boundary,
  obj_to_img=None,
  attributes=None,
  boxes_gt=None,
  generate=False,
  refine=False,
  relative=False,
  inside_box=None
  ):
```

**Parameters:**

- **objs**: (O,) - Room type IDs for all rooms
- **triples**: (T, 3) - Relationship triples [subject_idx, predicate_id, object_idx]
- **boundary**: (B, C, H, W) - Boundary/mask images
- **obj_to_img**: (O,) - Maps each room to its image index
- **attributes**: (O, 35) - Room attributes (position + area)
- **boxes_gt**: (O, 4) - Ground truth boxes (for teacher forcing)
- **generate**: bool - Whether to generate layout segmentation
- **refine**: bool - Whether to refine boxes
- **relative**: bool - Whether boxes are in relative coordinates
- **inside_box**: (B, 4) - Interior bounding box (for relative coordinates)

**Variable Meanings:**

- **O**: Total number of rooms across all images in batch
- **T**: Total number of relationship triples
- **B**: Batch size (number of floor plans)

---

## Section 1: Input Processing (Lines 152-174)

### Lines 165-171: Parse Input Dimensions

```python
O, T = objs.size(0), triples.size(0)
s, p, o = triples.chunk(3, dim=1)           # All have shape (T, 1)
s, p, o = [x.squeeze(1) for x in [s, p, o]] # Now have shape (T,)
edges = torch.stack([s, o], dim=1)          # Shape is (T, 2)
B = boundary.size(0)
H, W = self.image_size
```

- **O**: Number of rooms (e.g., 20 rooms total across 4 floor plans)
- **T**: Number of relationships (e.g., 35 edges connecting rooms)
- **triples.chunk(3, dim=1)**: Splits (T, 3) → 3 tensors of (T, 1)
  - **s**: Subject indices (which room is the source)
  - **p**: Predicate IDs (what relationship)
  - **o**: Object indices (which room is the target)
- **edges**: (T, 2) - Simplified to [source, target] for graph convolution
- **B**: Batch size (number of floor plans)
- **H, W**: Image height and width (128, 128)

**Example:**

```text
triples = [[0, 1, 1],    # Room 0 is "left of" Room 1
           [1, 2, 2],    # Room 1 is "above" Room 2
           [2, 0, 0]]    # Room 2 is "adjacent to" Room 0
After processing:
s = [0, 1, 2]  (subjects)
p = [1, 2, 0]  (predicates)
o = [1, 2, 0]  (objects)
edges = [[0,1], [1,2], [2,0]]
```

### Lines 173-174: Handle Missing obj_to_img

```python
if obj_to_img is None:
  obj_to_img = torch.zeros(O, dtype=objs.dtype, device=objs.device)
```

- If not provided, assumes all rooms belong to image 0

---

## Section 2: Embedding Layer (Lines 176-196)

### Lines 177-193: Validate and Remap Room Indices

```python
num_objs = self.obj_embeddings.num_embeddings
if (objs >= num_objs).any() or (objs < 0).any():
    from model.utils import get_vocab
    vocab = get_vocab()
    index_map = vocab['data_idx_to_model_idx']
    objs_remapped = torch.zeros_like(objs)
    for i, obj_idx in enumerate(objs):
        obj_idx_val = obj_idx.item()
        if obj_idx_val in index_map:
            objs_remapped[i] = index_map[obj_idx_val]
        else:
            objs_remapped[i] = torch.clamp(obj_idx, 0, num_objs - 1)
    objs = objs_remapped
```

- **Critical fix**: Old data has indices [0,1,2,3,15] but model expects [0,1,2,3,4]
- **index_map**: Maps old indices to new ones (e.g., 15→4)
- **Purpose**: Handle legacy data with removed balcony class

### Lines 195-196: Create Embeddings

```python
obj_vecs = self.obj_embeddings(objs)
pred_vecs = self.pred_embeddings(p)
```

- **obj_vecs**: (O, 128) - Room embeddings
- **pred_vecs**: (T, 128) - Relationship embeddings

**What embeddings do:**

```text
Room ID 0 (LivingRoom) → [0.23, -0.45, 0.12, ..., 0.67] (128 numbers)
Room ID 1 (MasterRoom)  → [-0.15, 0.67, -0.23, ..., 0.45]
These are learned during training!
```

---

## Section 3: Add Attributes (Lines 198-201)

### Lines 199-201: Concatenate Attributes

```python
if attributes is not None:
  obj_vecs = torch.cat([obj_vecs, attributes], 1)
obj_vecs_orig = obj_vecs
```

- **attributes**: (O, 35) - Position (25D) + Area (10D)
- **After concatenation**: obj_vecs is (O, 163) = 128 + 35
- **obj_vecs_orig**: Save original features for later use

---

## Section 4: Graph Convolution (Lines 203-205)

### Lines 204-205: Apply Graph Layers

```python
obj_vecs, pred_vecs = self.gconv(obj_vecs, pred_vecs, edges)
obj_vecs, pred_vecs = self.gconv_net(obj_vecs, pred_vecs, edges)
```

- **First call**: First graph convolution layer
- **Second call**: Remaining 4 graph convolution layers
- **Total**: 5 rounds of message passing

**What happens:**

1. Each room looks at its neighbors
2. Aggregates their features
3. Updates its own representation
4. Repeated 5 times → rooms learn about distant neighbors

**After graph convolution:**

- **obj_vecs**: (O, 128) - Updated room features (now context-aware)
- **pred_vecs**: (T, 128) - Updated relationship features

---

## Section 5: Add Boundary Context (Lines 207-209)

### Lines 208-209: Process Boundary and Concatenate

```python
inside_vecs = self.inside_cnn(boundary).view(B, -1)
obj_vecs = torch.cat([obj_vecs, inside_vecs[obj_to_img]], dim=1)
```

- **inside_cnn(boundary)**: Processes boundary image → (B, 256, 1, 1)
- **.view(B, -1)**: Flattens to (B, 256)
- **inside_vecs[obj_to_img]**: Broadcasts boundary features to all rooms in that image
  - If rooms 0-4 belong to image 0, they all get the same boundary vector
- **After concatenation**: obj_vecs is (O, 384) = 128 (graph) + 256 (boundary)

**Purpose**: Add global context about floor plan shape to each room

---

## Section 6: Box Prediction (Lines 211-213)

### Lines 212-213: Predict Initial Boxes

```python
boxes_pred = self.box_net(obj_vecs)
if relative: boxes_pred = box_utils.box_rel2abs(boxes_pred, inside_box, obj_to_img)
```

- **box_net(obj_vecs)**: MLP predicts boxes from room features
  - Input: (O, 384)
  - Output: (O, 4) - [x_center, y_center, width, height]
- **If relative**: Convert from relative (within inner boundary) to absolute coordinates
  - Useful when floor plan has thick walls - rooms positioned relative to interior space

**Box format**: Center + Size representation

- x_center, y_center: Center point (normalized 0-1)
- width, height: Box dimensions (normalized 0-1)

---

## Section 7: Relationship Prediction (Lines 215-217)

### Lines 216-217: (Currently Commented Out)

```python
# rel_scores = self.rel_aux_net(obj_vecs)
```

- **Intended**: Predict door positions using relationship network
- **Status**: Not used in current version

---

## Section 8: Layout Generation (Lines 219-225)

### Lines 220-225: Generate Semantic Segmentation

```python
gene_layout = None
boxes_refine = None
layout_boxes = boxes_pred if boxes_gt is None else boxes_gt
if generate:
  layout_features = boxes_to_layout(obj_vecs, layout_boxes, obj_to_img, H, W)
  gene_layout = self.refinement_net(layout_features)
```

- **layout_boxes**: Use predicted boxes (or ground truth for teacher forcing)
- **boxes_to_layout**: Creates spatial feature map
  - Places room features at their box locations
  - Output: (B, 384, 128, 128) - Room features arranged spatially
- **refinement_net**: CNN that converts features to segmentation
  - Output: (B, 5, 128, 128) - Logits for 5 room types

**What boxes_to_layout does:**

```text
Input: Room 0 features (384D), Room 0 box [0.3, 0.4, 0.2, 0.2]
Output: Feature map where pixels inside box [0.3, 0.4, 0.2, 0.2]
        contain Room 0's features, other pixels are 0 or overlapped
```

---

## Section 9: Box Refinement (Lines 227-241)

### Lines 228-240: Two-Stage Box Refinement

```python
if refine:
  gene_feat = self.box_refine_backbone(gene_layout)
  rois = torch.cat([
    obj_to_img.float().view(-1,1),
    box_utils.centers_to_extents(layout_boxes)*H
  ], -1)
  roi_feat = self.down_sample(self.roi_align(gene_feat, rois)).flatten(1)
  roi_feat = torch.cat([roi_feat, obj_vecs], -1)
  boxes_refine = self.box_reg(roi_feat)
  if relative: boxes_refine = box_utils.box_rel2abs(boxes_refine, inside_box, obj_to_img)
```

**Step-by-step:**

1. **Extract features from generated layout:**

   ```python
   gene_feat = self.box_refine_backbone(gene_layout)
   ```

   - CNN processes segmentation → (B, 256, H/8, W/8)

2. **Prepare RoI specifications:**

   ```python
   rois = torch.cat([obj_to_img.float().view(-1,1),
                     box_utils.centers_to_extents(layout_boxes)*H], -1)
   ```

   - **centers_to_extents**: Converts [x_c, y_c, w, h] → [x0, y0, x1, y1]
   - **Multiply by H**: Scale normalized coords to pixel coords
   - **Format**: [image_idx, x0, y0, x1, y1] for each room
   - Output: (O, 5)

3. **Extract per-room features:**

   ```python
   roi_feat = self.down_sample(self.roi_align(gene_feat, rois)).flatten(1)
   ```

   - **roi_align**: Extracts (256, 8, 8) features for each room's box
   - **down_sample**: Pools to (256, 1, 1)
   - **flatten**: Results in (O, 256)

4. **Combine with original room features:**

   ```python
   roi_feat = torch.cat([roi_feat, obj_vecs], -1)
   ```

   - (O, 256) + (O, 384) = (O, 640)

5. **Predict refined boxes:**

   ```python
   boxes_refine = self.box_reg(roi_feat)
   ```

   - MLP: 640 → 512 → 4
   - Output: (O, 4) refined boxes

**Purpose**: Use visual feedback from generated layout to improve box predictions

---

## Section 10: Return Outputs (Line 242)

### Line 242: Return All Predictions

```python
return boxes_pred, gene_layout, boxes_refine
```

**Returns:**

- **boxes_pred**: (O, 4) - Initial box predictions from graph network
- **gene_layout**: (B, 5, 128, 128) or None - Semantic segmentation (if generate=True)
- **boxes_refine**: (O, 4) or None - Refined boxes (if refine=True)

---

## Summary: Complete Data Flow

```text
Input:
  objs:      [0, 1, 2, 3, 4]              Room types
  triples:   [[0,1,1], [1,2,2], ...]     Relationships
  boundary:  (1, 3, 128, 128)            Boundary image
  attributes: (5, 35)                    Room attributes

Step 1: Embed
  obj_vecs:  (5, 128)                    Room embeddings
  pred_vecs: (T, 128)                    Relation embeddings
  + attributes → (5, 163)

Step 2: Graph Convolution (5 layers)
  obj_vecs:  (5, 128)                    Context-aware features

Step 3: Add Boundary Context
  inside_vecs: (1, 256)                  Boundary features
  obj_vecs:    (5, 384)                  Combined features

Step 4: Predict Boxes
  boxes_pred: (5, 4)                     [x_c, y_c, w, h] for each room

Step 5: Generate Layout (optional)
  layout_features: (1, 384, 128, 128)   Spatial feature map
  gene_layout:     (1, 5, 128, 128)     Segmentation logits

Step 6: Refine Boxes (optional)
  gene_feat:    (1, 256, 16, 16)        Layout features
  roi_feat:     (5, 640)                Per-room features
  boxes_refine: (5, 4)                  Refined boxes

Output:
  boxes_pred, gene_layout, boxes_refine
```

---

## Key Concepts Explained

### 1. Scene Graph Representation

A scene graph represents a floor plan as:

- **Nodes**: Rooms (LivingRoom, MasterRoom, Kitchen, etc.)
- **Edges**: Spatial relationships (left_of, above, adjacent_to)
- **Attributes**: Room properties (position, area)

### 2. Graph Convolution

Similar to how CNNs process images by looking at neighboring pixels, graph convolution processes graphs by looking at neighboring nodes:

- Each room aggregates information from connected rooms
- Multiple layers allow information to propagate across the entire graph
- Rooms learn context-aware representations

### 3. Two-Stage Approach

1. **Stage 1**: Predict boxes from abstract graph
2. **Stage 2**: Generate layout, then refine boxes using visual features
   - Similar to Faster R-CNN's two-stage object detection

### 4. RoI Align

Extracts features from specific regions of a feature map:

- Given a bounding box [x0, y0, x1, y1]
- Extract a fixed-size (8×8) feature grid from that region
- Uses bilinear interpolation for sub-pixel accuracy
- Allows different-sized boxes to produce same-sized features

### 5. Relative vs Absolute Coordinates

- **Absolute**: Boxes measured relative to full image (0-1 normalized)
- **Relative**: Boxes measured relative to inner boundary (interior space)
  - Useful for floor plans with thick walls
  - Separates wall thickness from room layout

---

## Architecture Diagram

```text
                    Input Scene Graph
                           |
    +-----------------+----+----+------------------+
    |                 |         |                  |
  objs            triples   boundary          attributes
 [0,1,2,3]      [[0,1,1],   (B,3,H,W)          (O,35)
                 [1,2,2]]        |
    |                |           |
    v                v           v
Embeddings     Pred Embed    Inside CNN
 (O,128)        (T,128)       (B,256)
    |                |            |
    +--------+-------+            |
             |                    |
             v                    |
   Graph Convolution (5 layers)  |
          (O,128)                 |
             |                    |
             +--------------------+
                      |
                 Concatenate
                   (O,384)
                      |
                 +----+----+
                 |         |
                 v         v
             Box Net   Rel Net (unused)
             (O,4)
                 |
          boxes_to_layout
         (B,384,H,W)
                 |
                 v
         Refinement CNN
          (B,5,H,W)
            gene_layout
                 |
                 v
       Box Refine Backbone
          (B,256,H,W)
                 |
             RoI Align
             (O,256)
                 |
          Concat with obj_vecs
             (O,640)
                 |
                 v
            Box Reg MLP
             (O,4)
           boxes_refine
```

---

## Common Issues & Solutions

### Issue 1: Index Out of Range Error

**Symptom**: `Index out of range in embedding lookup`
**Cause**: Old data has room indices [0,1,2,3,15] but model expects [0,1,2,3,4]
**Solution**: Lines 177-193 remap indices using `data_idx_to_model_idx`

### Issue 2: NaN/Inf During Training

**Symptom**: Losses become NaN after a few epochs
**Cause**: Gradients explode during layout generation
**Solution**: (in train.py)

- Gradual loss weight increase
- Gradient clipping
- Lower learning rate
- Skip batches with NaN

### Issue 3: Memory Issues

**Symptom**: CUDA out of memory
**Cause**: Large feature maps (B, 384, 128, 128)
**Solution**:

- Reduce batch size
- Use smaller image_size
- Disable refinement during debugging

---

## Usage Example

```python
# Create model
model = Model(
    embedding_dim=128,
    image_size=(128, 128),
    gconv_num_layers=5,
    box_refine_arch="I5,C3-64-2,C3-128-2,C3-256-2"
)

# Forward pass
boxes_pred, gene_layout, boxes_refine = model(
    objs=room_ids,           # [0, 1, 2, 3, 4]
    triples=relationships,   # [[0,1,1], [1,2,2], ...]
    boundary=boundary_img,   # (1, 3, 128, 128)
    obj_to_img=obj_img_map, # [0, 0, 0, 0, 0]
    attributes=room_attrs,   # (5, 35)
    generate=True,           # Generate layout
    refine=True,             # Refine boxes
    relative=False           # Use absolute coords
)

# Outputs:
# boxes_pred:   (5, 4) - Initial boxes
# gene_layout:  (1, 5, 128, 128) - Segmentation map
# boxes_refine: (5, 4) - Refined boxes
```

---

## Quick Reference

| Component | Input | Output | Purpose |
| --------- | ----- | ------ | ------- |
| obj_embeddings | (O,) room IDs | (O, 128) | Convert IDs to vectors |
| gconv + gconv_net | (O, 163) features | (O, 128) | Process graph structure |
| inside_cnn | (B, 3, H, W) | (B, 256) | Extract boundary context |
| box_net | (O, 384) | (O, 4) | Predict bounding boxes |
| refinement_net | (B, 384, H, W) | (B, 5, H, W) | Generate segmentation |
| box_refine_backbone | (B, 5, H, W) | (B, 256, H/8, W/8) | Extract layout features |
| roi_align | (B, 256, H/8, W/8) + ROIs | (O, 256, 8, 8) | Extract per-room features |
| box_reg | (O, 640) | (O, 4) | Refine boxes |

---

## Next Steps

1. **Understand graph.py**: See how GraphTripleConv actually works
2. **Understand box_utils.py**: See coordinate transformations
3. **Understand layout.py**: See how boxes_to_layout works
4. **Read train.py**: See how the model is trained
5. **Experiment**: Try forward pass with dummy data

---

### End of model.py explanation

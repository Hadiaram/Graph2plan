# Future Features Implementation Guide

This document predicts where code changes will be needed for planned features:
1. Sun Positioning
2. Entrance Position Enhancement
3. Foundation (Multi-level Support)
4. Layout Footprint Control

---

## Table of Contents

1. [Sun Positioning](#1-sun-positioning)
2. [Entrance Position](#2-entrance-position)
3. [Foundation (Multi-level/Basement Support)](#3-foundation-multi-levelbasement-support)
4. [Layout Footprint Control](#4-layout-footprint-control)
5. [Integration Points](#integration-points-where-features-interact)
6. [Data Pipeline Changes](#data-pipeline-changes-summary)
7. [Recommended Implementation Order](#recommended-implementation-order)
8. [Files to Modify Per Feature](#files-youll-definitely-need-to-modify-per-feature)
9. [Testing Strategy](#testing-strategy-for-each-feature)

---

## 1. Sun Positioning

**Purpose**: Orient rooms based on solar exposure (e.g., living rooms facing south/southwest for natural light)

### Changes Needed:

#### A. Data Preprocessing (Add sun direction to dataset)

**File**: Your ResPlan-to-MAT converter script

**What to Add**:
- New field `sun_direction` to each floor plan
- Angle in degrees: 0°=North, 90°=East, 180°=South, 270°=West
- Store as vector `[sin(θ), cos(θ)]` for better neural network encoding

**Example**:
```python
# In your preprocessing script
import numpy as np

sun_angle = 180  # South-facing
sun_direction = [np.sin(np.radians(sun_angle)), np.cos(np.radians(sun_angle))]
# Store in MAT file
```

#### B. Model Input (Add sun direction as feature)

**File**: `/Network/model/model.py`

**Location**: Model `__init__` and `forward` methods

**What to Change**:
- Add new embedding or linear layer for sun direction encoding
- Concatenate sun direction features with existing node features
- Pass through graph neural network layers

**Example Changes**:
```python
class Model(nn.Module):
    def __init__(self, vocab):
        super().__init__()
        # ... existing code ...

        # NEW: Add sun direction encoder
        self.sun_encoder = nn.Sequential(
            nn.Linear(2, 64),  # Encode [sin, cos] pair
            nn.ReLU(),
            nn.Linear(64, 64)
        )

    def forward(self, objs, triples, boundary, sun_direction=None, ...):
        # ... existing code ...

        # NEW: Encode sun direction
        if sun_direction is not None:
            sun_feat = self.sun_encoder(sun_direction)  # [B, 64]
            # Broadcast to all nodes
            sun_feat = sun_feat.unsqueeze(1).expand(-1, objs.size(1), -1)
            # Concatenate with node features
            obj_vecs = torch.cat([obj_vecs, sun_feat], dim=-1)
```

**Why**: The model needs sun direction as input to learn room placement preferences based on solar orientation.

#### C. FloorPlan Data Structure

**File**: `/Interface/model/floorplan.py`

**Location**: Class `FloorPlan.__init__` and new method `get_sun_direction()`

**What to Add**:
```python
class FloorPlan():
    def __init__(self, data, train=False, rot=None):
        self.data = copy.deepcopy(data)
        # ... existing code ...

        # NEW: Store sun direction if available
        if hasattr(data, 'sun_direction'):
            self.sun_direction = data.sun_direction
        else:
            self.sun_direction = None  # Default for old data

    # NEW METHOD: Get sun direction as tensor
    def get_sun_direction(self, tensor=True):
        """Returns sun direction as [sin(θ), cos(θ)] vector"""
        if self.sun_direction is None:
            # Default: South-facing (180°)
            sun_dir = np.array([0.0, -1.0])
        else:
            sun_dir = np.array(self.sun_direction)

        if tensor:
            sun_dir = torch.tensor(sun_dir).float()
        return sun_dir

    def get_test_data(self, tensor=True):
        boundary = self.get_input_boundary(tensor=tensor)
        inside_box = self.get_inside_box(tensor=tensor)
        rooms = self.get_rooms(tensor=tensor)
        attrs = self.get_attributes(tensor=tensor)
        triples = self.get_triples(random=False, tensor=tensor)
        sun_direction = self.get_sun_direction(tensor=tensor)  # NEW
        return boundary, inside_box, rooms, attrs, triples, sun_direction  # Updated return
```

**Why**: FloorPlan class needs to handle sun direction data throughout the pipeline.

#### D. Loss Function (Orientation constraint)

**File**: `/Network/train.py`

**Location**: Around lines 240-280 where loss components are defined

**What to Add**: New loss term `sun_orientation_loss`

**Example**:
```python
def compute_sun_orientation_loss(boxes, room_types, sun_direction, boundary):
    """
    Penalize room placements that violate sun orientation preferences

    Args:
        boxes: [B, N, 4] room bounding boxes (normalized coordinates)
        room_types: [B, N] room type IDs
        sun_direction: [B, 2] sun direction vectors [sin(θ), cos(θ)]
        boundary: [B, 3, 128, 128] boundary mask

    Returns:
        loss: scalar tensor
    """
    loss = 0.0

    # Room preferences (room_type_id -> preference for sun exposure)
    # 1.0 = should face sun, -1.0 = should avoid sun, 0.0 = no preference
    sun_preferences = {
        0: 1.0,   # LivingRoom - should face sun
        1: 0.5,   # MasterRoom - moderate sun preference (morning sun)
        2: 0.3,   # Kitchen - some sun is nice
        3: -0.5,  # Bathroom - avoid harsh sun
        4: 0.8,   # DiningRoom - should face sun
        5: 0.5,   # ChildRoom - moderate sun
        6: 0.7,   # StudyRoom - good natural light
        7: 0.5,   # SecondRoom - moderate sun
        8: 0.5,   # GuestRoom - moderate sun
        9: 1.0,   # Balcony - definitely should face sun
        10: 0.0,  # Entrance - no preference
        11: -0.8, # Storage - should NOT waste sun-facing space
        # ... etc for all 18 types
    }

    for i in range(boxes.size(0)):  # Batch
        for j in range(boxes.size(1)):  # Rooms
            room_type = room_types[i, j].item()
            if room_type not in sun_preferences:
                continue

            preference = sun_preferences[room_type]
            if abs(preference) < 0.1:  # No strong preference
                continue

            # Get room center
            box = boxes[i, j]  # [x0, y0, x1, y1]
            center = torch.stack([(box[0] + box[2])/2, (box[1] + box[3])/2])

            # Get boundary center
            boundary_mask = boundary[i, 0]  # [128, 128]
            ys, xs = torch.where(boundary_mask > 0)
            boundary_center = torch.stack([xs.float().mean()/128, ys.float().mean()/128])

            # Vector from boundary center to room center
            room_vec = center - boundary_center
            room_vec = room_vec / (torch.norm(room_vec) + 1e-6)

            # Sun direction (pointing toward sun)
            sun_vec = sun_direction[i]  # [sin(θ), cos(θ)]

            # Dot product: 1.0 if aligned with sun, -1.0 if opposite
            alignment = torch.dot(room_vec, sun_vec)

            # Loss: penalize misalignment based on preference
            # If preference is positive (should face sun), penalize negative alignment
            # If preference is negative (should avoid sun), penalize positive alignment
            if preference > 0:
                # Want high alignment, penalize low/negative alignment
                loss += preference * torch.relu(1.0 - alignment)
            else:
                # Want low alignment, penalize high alignment
                loss += abs(preference) * torch.relu(alignment + 1.0)

    return loss / (boxes.size(0) * boxes.size(1))  # Average per room

# In the training loop, add to total loss:
sun_loss = compute_sun_orientation_loss(boxes_pred, room_types, sun_direction, boundary)
total_loss = total_loss + 0.1 * sun_loss  # Weight can be tuned
```

**Why**: Loss function guides the model to learn sun-oriented room placement.

#### E. Visualization

**File**: `/Interface/Houseweb/views.py`

**Location**: `TransGraph` and `AdjustGraph` response functions (lines 850-950)

**What to Add**: Include sun direction in JSON response

**Example**:
```python
# In TransGraph function, around line 930
response_data = {
    'status': 'success',
    'boundary': boundary_coords,
    'rooms': room_data,
    'edges': edge_data,
    'sun_direction': {  # NEW
        'angle': float(fp_end.sun_direction[0]) if hasattr(fp_end, 'sun_direction') else 180,
        'label': 'South'  # Or compute from angle
    }
}
return JsonResponse(response_data)
```

**Why**: Frontend needs sun direction data to display indicator.

#### F. Frontend Display

**File**: `/Interface/static/js/buttonEvent.js`

**What to Add**: Draw sun position indicator on canvas

**Example**:
```javascript
// Add this function
function drawSunIndicator(ctx, sunAngle, canvasWidth, canvasHeight) {
    // Draw sun icon in corner
    const sunX = canvasWidth - 40;
    const sunY = 40;
    const sunRadius = 20;

    // Draw sun circle
    ctx.fillStyle = '#FFD700';  // Gold
    ctx.beginPath();
    ctx.arc(sunX, sunY, sunRadius, 0, 2 * Math.PI);
    ctx.fill();

    // Draw sun rays
    ctx.strokeStyle = '#FFA500';  // Orange
    ctx.lineWidth = 2;
    for (let i = 0; i < 8; i++) {
        const angle = (i * Math.PI / 4) + sunAngle;
        const x1 = sunX + Math.cos(angle) * (sunRadius + 5);
        const y1 = sunY + Math.sin(angle) * (sunRadius + 5);
        const x2 = sunX + Math.cos(angle) * (sunRadius + 15);
        const y2 = sunY + Math.sin(angle) * (sunRadius + 15);

        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
        ctx.stroke();
    }

    // Draw directional arrow showing sun direction
    const arrowStartX = canvasWidth / 2;
    const arrowStartY = 20;
    const arrowLength = 30;
    const arrowEndX = arrowStartX + Math.sin(sunAngle * Math.PI / 180) * arrowLength;
    const arrowEndY = arrowStartY - Math.cos(sunAngle * Math.PI / 180) * arrowLength;

    ctx.strokeStyle = '#FF6347';  // Tomato red
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(arrowStartX, arrowStartY);
    ctx.lineTo(arrowEndX, arrowEndY);
    ctx.stroke();

    // Arrow head
    const headlen = 10;
    const angle = Math.atan2(arrowEndY - arrowStartY, arrowEndX - arrowStartX);
    ctx.beginPath();
    ctx.moveTo(arrowEndX, arrowEndY);
    ctx.lineTo(arrowEndX - headlen * Math.cos(angle - Math.PI / 6),
               arrowEndY - headlen * Math.sin(angle - Math.PI / 6));
    ctx.moveTo(arrowEndX, arrowEndY);
    ctx.lineTo(arrowEndX - headlen * Math.cos(angle + Math.PI / 6),
               arrowEndY - headlen * Math.sin(angle + Math.PI / 6));
    ctx.stroke();

    // Label
    ctx.fillStyle = '#000';
    ctx.font = '12px Arial';
    ctx.fillText(`Sun: ${sunAngle}°`, sunX - 25, sunY + sunRadius + 20);
}

// Call in your main rendering function
drawSunIndicator(ctx, responseData.sun_direction.angle, canvas.width, canvas.height);
```

**Why**: Users need to see sun direction to understand room orientations.

---

## 2. Entrance Position

**Current State**: Partially exists - boundary's first 2 points define door line in `self.data.boundary[:2, :]`

### Enhancements Needed:

#### A. Explicit Entrance Node

**File**: `/Interface/model/floorplan.py`

**Location**: Method `get_triples()` (lines 110-133)

**What to Add**: Special relationship "adjacent_to_entrance" for rooms touching the door line

**Example**:
```python
def get_triples(self, random=False, tensor=True):
    boxes = self.data.box[:, :4][:, [1, 0, 3, 2]]
    door_line = self.data.boundary[:2, :2]  # First 2 points define entrance

    triples = []

    # NEW: Add entrance adjacency relationships
    for i in range(len(boxes)):
        box = boxes[i]
        # Check if room box intersects with door line region
        # Define "entrance region" as area within 20 pixels of door line
        door_center = door_line.mean(0)
        box_center = np.array([(box[0] + box[2])/2, (box[1] + box[3])/2])

        # If room center is close to entrance
        distance_to_entrance = np.linalg.norm(box_center - door_center)
        if distance_to_entrance < 50:  # Within 50 pixels (adjust threshold)
            # Add special "adjacent_to_entrance" relationship
            # You'll need to add this to vocab['pred_name_to_idx']
            triples.append([i, vocab['pred_name_to_idx']['adjacent_to_entrance'], -1])
            # -1 indicates entrance (not a room node)

    # ... existing edge relation code ...
    for u, v, _ in self.data.edge:
        # ... existing code ...
        triples.append([u, vocab['pred_name_to_idx'][relation], v])

    triples = np.array(triples, dtype=int)
    if tensor: triples = torch.tensor(triples).long()
    return triples
```

**Vocabulary Update Needed**:

**File**: `/Interface/model/utils.py`

**Location**: Lines 330-340 (predicate vocabulary)

**What to Add**:
```python
pred_name_to_idx = {
    'left': 0,
    'right': 1,
    'front': 2,
    'behind': 3,
    'surrounding': 4,
    'inside': 5,
    'adjacent_to_entrance': 6  # NEW
}
```

**Why**: Model can learn that entrance halls, foyers, or living rooms should be near entrance.

#### B. Entrance Type Attribute

**File**: `/Interface/model/utils.py`

**Location**: Room type vocabulary (lines 310-326)

**What to Consider**: Already have type 15 (FrontDoor), but might want entrance orientation attributes.

**Optional Addition**:
```python
'entrance_type_to_idx': {
    'main_entrance': 0,
    'side_entrance': 1,
    'street_facing': 2,
    'courtyard_facing': 3,
    'garage_entrance': 4
}
```

**Why**: Different entrance types might have different adjacency requirements.

#### C. Entrance Adjacency Constraint

**File**: `/Network/train.py`

**Location**: Loss function section (around lines 240-280)

**What to Add**: `entrance_adjacency_loss` that penalizes inappropriate room placement

**Example**:
```python
def compute_entrance_adjacency_loss(boxes, room_types, boundary):
    """
    Penalize inappropriate room placement near entrance

    Args:
        boxes: [B, N, 4] room bounding boxes
        room_types: [B, N] room type IDs
        boundary: [B, K, 2] boundary points (first 2 points = door line)

    Returns:
        loss: scalar tensor
    """
    loss = 0.0

    # Room appropriateness near entrance (room_type_id -> penalty)
    # Higher penalty = should NOT be near entrance
    entrance_penalties = {
        0: 0.0,   # LivingRoom - GOOD at entrance
        1: 0.5,   # MasterRoom - Not ideal at entrance (privacy)
        2: 0.3,   # Kitchen - OK at entrance (some layouts)
        3: 1.0,   # Bathroom - BAD at entrance (privacy issue!)
        4: 0.1,   # DiningRoom - Fine at entrance
        5: 0.7,   # ChildRoom - Should not be at entrance
        6: 0.4,   # StudyRoom - Not ideal
        7: 0.6,   # SecondRoom - Not ideal
        8: 0.2,   # GuestRoom - OK at entrance
        9: 0.0,   # Balcony - Neutral
        10: 0.0,  # Entrance - Obviously good
        11: 0.5,  # Storage - Not ideal but acceptable
        15: 0.0,  # FrontDoor - Obviously good
        # ... etc
    }

    for i in range(boxes.size(0)):  # Batch
        # Get entrance position (average of first 2 boundary points)
        entrance_pos = boundary[i, :2].mean(0)  # [2]

        for j in range(boxes.size(1)):  # Rooms
            room_type = room_types[i, j].item()
            if room_type not in entrance_penalties:
                continue

            penalty_weight = entrance_penalties[room_type]
            if penalty_weight < 0.1:  # No significant penalty
                continue

            # Get room center
            box = boxes[i, j]
            room_center = torch.stack([(box[0] + box[2])/2, (box[1] + box[3])/2])

            # Distance from room center to entrance
            distance = torch.norm(room_center - entrance_pos)

            # Penalty inversely proportional to distance
            # Close to entrance (distance→0) = high penalty
            # Far from entrance (distance→1) = low penalty
            proximity = torch.exp(-5 * distance)  # Exponential decay

            loss += penalty_weight * proximity

    return loss / (boxes.size(0) * boxes.size(1))

# In training loop:
entrance_loss = compute_entrance_adjacency_loss(boxes_pred, room_types, boundary)
total_loss = total_loss + 0.15 * entrance_loss
```

**Why**: Enforces architectural best practices for room placement relative to entrance.

#### D. UI Control for Entrance Positioning

**File**: `/Interface/Houseweb/views.py`

**Location**: `NumSearch` endpoint (where boundary is defined)

**What to Add**: Allow user to specify which edge of boundary should have entrance

**Example Modification**:
```python
@csrf_exempt
def NumSearch(request):
    if request.method == 'POST':
        data = json.loads(request.body)

        boundary_points = data.get('boundary')  # List of [x, y] points
        entrance_edge_index = data.get('entrance_edge', 0)  # NEW: which edge has entrance

        # Reorder boundary points so entrance edge is first
        boundary_points = reorder_boundary_for_entrance(boundary_points, entrance_edge_index)

        # Store in test_data
        # ... rest of existing code ...

def reorder_boundary_for_entrance(points, edge_index):
    """
    Reorder boundary points so the specified edge is first (points 0-1)

    Args:
        points: List of [x, y] boundary points
        edge_index: Index of edge that should have entrance (0, 1, 2, ...)

    Returns:
        Reordered points with entrance edge first
    """
    n = len(points)
    # Rotate list so entrance edge is at positions 0-1
    rotated = points[edge_index:] + points[:edge_index]
    return rotated
```

**Frontend Addition** (`/Interface/static/js/buttonEvent.js`):
```javascript
// When user draws boundary, let them click an edge to mark it as entrance
canvas.addEventListener('click', function(e) {
    if (drawingMode === 'mark_entrance') {
        const rect = canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;

        // Find which edge was clicked (closest edge to click point)
        let closestEdge = findClosestEdge(boundaryPoints, x, y);

        // Highlight entrance edge in different color
        drawBoundaryWithEntrance(boundaryPoints, closestEdge);

        // Store entrance edge index
        entranceEdgeIndex = closestEdge;
    }
});
```

**Why**: Gives users control over where entrance should be, which affects room layout.

---

## 3. Foundation (Multi-level/Basement Support)

**Purpose**: Support multiple floor levels, basement, elevated foundation

### Changes Needed:

#### A. Data Structure - Add Floor Level

**File**: `/Interface/model/floorplan.py`

**Location**: Class attributes in `FloorPlan.__init__`

**What to Add**:
```python
class FloorPlan():
    def __init__(self, data, train=False, rot=None):
        self.data = copy.deepcopy(data)
        # ... existing code ...

        # NEW: Add floor level support
        if hasattr(data, 'floor_levels'):
            self.floor_levels = data.floor_levels
        else:
            self.floor_levels = None  # Legacy data has only one floor

        # NEW: Per-floor boundaries (if multi-level)
        if hasattr(data, 'boundary_by_floor'):
            self.boundary_by_floor = data.boundary_by_floor
        else:
            self.boundary_by_floor = {0: data.boundary}  # Ground floor only
```

**Box Data Format Change**:
```python
# OLD: self.data.box = [x0, y0, x1, y1, room_type]  # Shape: [N, 5]
# NEW: self.data.box = [x0, y0, x1, y1, room_type, floor_level]  # Shape: [N, 6]
```

**New Methods Needed**:
```python
def get_floor_level(self, tensor=True):
    """Get floor level for each room"""
    if self.data.box.shape[1] >= 6:
        floor_levels = self.data.box[:, 5]
    else:
        # Legacy data: assume all rooms on ground floor (level 0)
        floor_levels = np.zeros(len(self.data.box))

    if tensor:
        floor_levels = torch.tensor(floor_levels).long()
    return floor_levels

def get_rooms_by_floor(self, floor_num):
    """Get all rooms on a specific floor"""
    floor_levels = self.get_floor_level(tensor=False)
    room_mask = (floor_levels == floor_num)
    return self.data.box[room_mask]

def get_boundary_by_floor(self, floor_num):
    """Get boundary polygon for a specific floor"""
    if floor_num in self.boundary_by_floor:
        return self.boundary_by_floor[floor_num]
    else:
        # Default to main boundary
        return self.data.boundary
```

**Why**: Need to track which floor each room is on for multi-level buildings.

#### B. Vocabulary - Foundation Type

**File**: `/Interface/model/utils.py`

**Location**: Add new attribute type (after room types, around line 350)

**What to Add**:
```python
def get_vocab():
    # ... existing room_label definition ...

    # NEW: Foundation types
    foundation_type_to_idx = {
        'slab': 0,          # Concrete slab on ground
        'crawlspace': 1,    # Elevated 1-3 feet
        'basement': 2,      # Full basement below ground
        'elevated': 3,      # Raised on posts (flood areas)
        'mixed': 4          # Combination (e.g., basement under part of house)
    }

    foundation_idx_to_name = {v: k for k, v in foundation_type_to_idx.items()}

    vocab = {
        'object_name_to_idx': object_name_to_idx,
        'object_idx_to_name': object_idx_to_name,
        'pred_name_to_idx': pred_name_to_idx,
        'pred_idx_to_name': pred_idx_to_name,
        'foundation_type_to_idx': foundation_type_to_idx,  # NEW
        'foundation_idx_to_name': foundation_idx_to_name   # NEW
    }

    return vocab
```

**Why**: Different foundation types have different room placement constraints (e.g., garage in basement).

#### C. Model Architecture - Multi-level GNN

**File**: `/Network/model/model.py`

**Location**: Graph convolution layers and edge processing

**What to Change**: This is a MAJOR architectural change. The model needs to handle 3D spatial relationships.

**Changes Needed**:

1. **Add Floor Level Embeddings**:
```python
class Model(nn.Module):
    def __init__(self, vocab):
        super().__init__()
        # ... existing embeddings ...

        # NEW: Floor level embedding
        self.floor_embedding = nn.Embedding(num_floors=5, embedding_dim=32)
        # Support up to 5 floors: -1 (basement), 0 (ground), 1 (2nd), 2 (3rd), 3 (attic)

        # Update obj_vecs dimension to include floor embedding
        # OLD: obj_vecs is [B, N, 128]
        # NEW: obj_vecs is [B, N, 128+32] after concatenation
```

2. **Extend Edge Relations for Vertical Connectivity**:

**File**: `/Interface/model/utils.py` (vocabulary)

**Add Vertical Relations**:
```python
pred_name_to_idx = {
    # Existing horizontal relations
    'left': 0,
    'right': 1,
    'front': 2,
    'behind': 3,
    'surrounding': 4,
    'inside': 5,
    'adjacent_to_entrance': 6,

    # NEW: Vertical relations
    'directly_above': 7,
    'directly_below': 8,
    'vertically_aligned': 9  # Rooms stacked but not directly touching
}
```

3. **Update Graph Construction**:

**File**: `/Interface/model/floorplan.py`

**Location**: Method `get_triples()` (lines 110-133)

**What to Add**:
```python
def get_triples(self, random=False, tensor=True):
    boxes = self.data.box[:, :4][:, [1, 0, 3, 2]]
    floor_levels = self.data.box[:, 5] if self.data.box.shape[1] >= 6 else np.zeros(len(boxes))

    triples = []

    # Existing: horizontal edges (same-floor relations)
    for u, v, _ in self.data.edge:
        if floor_levels[u] == floor_levels[v]:  # Same floor
            # ... existing spatial relation code ...
            triples.append([u, relation_idx, v])

    # NEW: Add vertical edges (cross-floor relations)
    for u in range(len(boxes)):
        for v in range(len(boxes)):
            if u == v:
                continue

            # Check if rooms are vertically aligned
            if floor_levels[u] != floor_levels[v]:
                # Check horizontal overlap (are they above/below each other?)
                box_u = boxes[u]
                box_v = boxes[v]

                # Check if bounding boxes overlap in XY plane
                x_overlap = not (box_u[1] > box_v[3] or box_u[3] < box_v[1])
                y_overlap = not (box_u[0] > box_v[2] or box_u[2] < box_v[0])

                if x_overlap and y_overlap:
                    # Vertically aligned
                    if floor_levels[u] == floor_levels[v] + 1:
                        # u is directly above v
                        triples.append([u, vocab['pred_name_to_idx']['directly_above'], v])
                    elif floor_levels[u] == floor_levels[v] - 1:
                        # u is directly below v
                        triples.append([u, vocab['pred_name_to_idx']['directly_below'], v])

    triples = np.array(triples, dtype=int)
    if tensor: triples = torch.tensor(triples).long()
    return triples
```

**Why**: Model needs to understand vertical relationships for multi-story layouts.

#### D. Vertical Constraints in Loss

**File**: `/Network/train.py`

**Location**: Loss function (lines 240-280)

**What to Add**: `vertical_constraint_loss`

**Example**:
```python
def compute_vertical_constraint_loss(boxes, room_types, floor_levels):
    """
    Enforce vertical placement best practices

    Args:
        boxes: [B, N, 4] room bounding boxes
        room_types: [B, N] room type IDs
        floor_levels: [B, N] floor level for each room

    Returns:
        loss: scalar tensor
    """
    loss = 0.0

    # Preferred floor levels for each room type
    # Negative = prefer lower floors, Positive = prefer upper floors, 0 = no preference
    floor_preferences = {
        0: 0.0,    # LivingRoom - ground floor (neutral, often required)
        1: 1.0,    # MasterRoom - upper floor preferred (privacy)
        2: -0.5,   # Kitchen - ground floor preferred
        3: 0.5,    # Bathroom - flexible, but upper floors OK
        4: 0.0,    # DiningRoom - ground floor (neutral)
        5: 1.0,    # ChildRoom - upper floor (privacy, safety)
        6: 0.7,    # StudyRoom - upper floor preferred (quiet)
        7: 1.0,    # SecondRoom - upper floor
        8: 0.3,    # GuestRoom - ground floor OK, upper floor better
        9: 1.0,    # Balcony - upper floor much better
        10: -1.0,  # Entrance - MUST be ground floor
        11: -0.8,  # Storage - basement/ground floor preferred
        # ... etc
    }

    # Special constraint: Garage must be basement or ground floor
    garage_types = []  # Add garage room type ID if you have it

    for i in range(boxes.size(0)):  # Batch
        for j in range(boxes.size(1)):  # Rooms
            room_type = room_types[i, j].item()
            floor = floor_levels[i, j].item()

            if room_type not in floor_preferences:
                continue

            preference = floor_preferences[room_type]

            # Penalty based on how far from preferred floor
            if preference > 0:  # Prefer upper floors
                # Penalty if on ground floor (0) or basement (-1)
                target_floor = 1  # Upper floors
                floor_diff = target_floor - floor
                if floor_diff > 0:
                    loss += preference * floor_diff
            elif preference < 0:  # Prefer lower floors
                # Penalty if on upper floors
                target_floor = 0  # Ground floor
                floor_diff = floor - target_floor
                if floor_diff > 0:
                    loss += abs(preference) * floor_diff

    return loss / (boxes.size(0) * boxes.size(1))

# In training loop:
if has_multiple_floors:
    vertical_loss = compute_vertical_constraint_loss(boxes_pred, room_types, floor_levels)
    total_loss = total_loss + 0.2 * vertical_loss
```

**Additional Constraint: Stairwell Alignment**:
```python
def compute_stairwell_constraint_loss(boxes, room_types, floor_levels):
    """
    Ensure stairwells are vertically aligned across floors
    (Assuming you add a 'Stairwell' room type)
    """
    stairwell_type_id = 18  # Or whatever ID you assign

    loss = 0.0

    # Find all stairwells
    stairwell_mask = (room_types == stairwell_type_id)

    # Group by floor
    for floor in range(floor_levels.min().item(), floor_levels.max().item()):
        stairs_this_floor = (stairwell_mask & (floor_levels == floor))
        stairs_next_floor = (stairwell_mask & (floor_levels == floor + 1))

        if stairs_this_floor.any() and stairs_next_floor.any():
            # Get positions of stairwells on adjacent floors
            boxes_this = boxes[stairs_this_floor]
            boxes_next = boxes[stairs_next_floor]

            # Compute center positions
            centers_this = (boxes_this[:, :2] + boxes_this[:, 2:]) / 2
            centers_next = (boxes_next[:, :2] + boxes_next[:, 2:]) / 2

            # Penalize horizontal distance between vertically adjacent stairwells
            # They should be in the same XY position
            min_distance = float('inf')
            for c_this in centers_this:
                for c_next in centers_next:
                    dist = torch.norm(c_this - c_next)
                    min_distance = min(min_distance, dist)

            loss += min_distance

    return loss
```

**Why**: Multi-story buildings have architectural constraints on vertical room placement.

#### E. 3D Visualization (Major UI Change)

**File**: `/Interface/Houseweb/views.py`

**Location**: Response functions (lines 850-950)

**What to Change**: Return separate floor plans for each level

**Example**:
```python
@csrf_exempt
def TransGraph(request):
    # ... existing processing ...

    # NEW: Organize data by floor
    floors = {}

    for i, room_data in enumerate(all_rooms):
        floor_level = room_data.get('floor', 0)  # Default to ground floor

        if floor_level not in floors:
            floors[floor_level] = {
                'boundary': get_boundary_for_floor(fp_end, floor_level),
                'rooms': [],
                'edges': []
            }

        floors[floor_level]['rooms'].append({
            'type': room_data['type'],
            'box': room_data['box'],
            'label': room_data['label']
        })

    # Add edges per floor
    for edge in fp_end.data.edge:
        u, v, edge_type = edge
        floor_u = fp_end.data.box[u, 5] if fp_end.data.box.shape[1] >= 6 else 0
        floor_v = fp_end.data.box[v, 5] if fp_end.data.box.shape[1] >= 6 else 0

        if floor_u == floor_v:  # Same-floor edge
            floors[floor_u]['edges'].append({
                'from': u,
                'to': v,
                'type': edge_type
            })

    response_data = {
        'status': 'success',
        'floors': floors,  # NEW: Multi-floor data
        'num_floors': len(floors)
    }

    return JsonResponse(response_data)
```

**Frontend Changes** (`/Interface/static/js/buttonEvent.js`):
```javascript
// Add floor selector UI
function renderMultiFloorPlan(data) {
    const numFloors = data.num_floors;
    const floors = data.floors;

    // Create tabs for each floor
    const tabContainer = document.getElementById('floor-tabs');
    tabContainer.innerHTML = '';

    const floorNames = {
        '-1': 'Basement',
        '0': 'Ground Floor',
        '1': '2nd Floor',
        '2': '3rd Floor'
    };

    for (let floorNum in floors) {
        const tab = document.createElement('button');
        tab.className = 'floor-tab';
        tab.textContent = floorNames[floorNum] || `Floor ${floorNum}`;
        tab.onclick = () => displayFloor(floorNum);
        tabContainer.appendChild(tab);
    }

    // Display first floor by default
    displayFloor(Object.keys(floors)[0]);
}

function displayFloor(floorNum) {
    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw boundary for this floor
    drawBoundary(currentFloorData.boundary);

    // Draw rooms for this floor
    currentFloorData.rooms.forEach(room => {
        drawRoom(room);
    });

    // Highlight active tab
    document.querySelectorAll('.floor-tab').forEach(tab => {
        tab.classList.remove('active');
    });
    event.target.classList.add('active');
}

// Also add stacked view option (show all floors at once with transparency)
function renderStackedView() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const floors = Object.keys(currentData.floors).sort();
    const alpha_step = 0.8 / floors.length;

    floors.forEach((floorNum, idx) => {
        ctx.globalAlpha = 1.0 - (idx * alpha_step);  // Upper floors more opaque

        // Draw this floor's rooms with transparency
        currentData.floors[floorNum].rooms.forEach(room => {
            drawRoom(room);
        });
    });

    ctx.globalAlpha = 1.0;  // Reset
}
```

**Why**: Users need to visualize and interact with each floor separately.

---

## 4. Layout Footprint Control

**Purpose**: Constrain overall building shape (rectangular, L-shape, courtyard, etc.)

### Changes Needed:

#### A. Footprint Type Vocabulary

**File**: `/Interface/model/utils.py`

**Location**: Add new global attribute (around line 360)

**What to Add**:
```python
def get_vocab():
    # ... existing code ...

    # NEW: Footprint shape types
    footprint_type_to_idx = {
        'rectangular': 0,      # Standard rectangle
        'square': 1,           # Special case of rectangle
        'L_shape': 2,          # L-shaped building
        'U_shape': 3,          # U-shaped (courtyard on one side)
        'T_shape': 4,          # T-shaped
        'H_shape': 5,          # H-shaped (two buildings connected)
        'courtyard': 6,        # Full courtyard (rooms surround central space)
        'irregular': 7,        # Custom polygon
        'compound': 8          # Multiple disconnected buildings
    }

    footprint_idx_to_name = {v: k for k, v in footprint_type_to_idx.items()}

    vocab = {
        # ... existing entries ...
        'footprint_type_to_idx': footprint_type_to_idx,  # NEW
        'footprint_idx_to_name': footprint_idx_to_name   # NEW
    }

    return vocab
```

**Why**: Different footprint types have different layout possibilities and constraints.

#### B. Footprint as Model Input

**File**: `/Network/model/model.py`

**Location**: Model initialization and forward pass

**What to Add**:
```python
class Model(nn.Module):
    def __init__(self, vocab):
        super().__init__()
        # ... existing embeddings ...

        # NEW: Footprint type embedding
        num_footprint_types = len(vocab.get('footprint_type_to_idx', {})) or 9
        self.footprint_embedding = nn.Embedding(num_footprint_types, 64)

        # Update input dimensions where footprint features are concatenated

    def forward(self, objs, triples, boundary, footprint_type=None, ...):
        # ... existing code ...

        # NEW: Encode footprint type
        if footprint_type is not None:
            footprint_feat = self.footprint_embedding(footprint_type)  # [B, 64]
            # Broadcast to all nodes or add to global feature vector
            footprint_feat = footprint_feat.unsqueeze(1).expand(-1, objs.size(1), -1)
            obj_vecs = torch.cat([obj_vecs, footprint_feat], dim=-1)

        # Rest of forward pass uses footprint-aware features
```

**Why**: Model needs to know target footprint shape to generate appropriate layouts.

#### C. Boundary Generation Based on Footprint

**File**: `/Interface/model/floorplan.py`

**Location**: New utility function

**What to Add**:
```python
def generate_boundary_from_footprint(footprint_type, total_area, params=None):
    """
    Generate boundary polygon based on footprint type

    Args:
        footprint_type: String from footprint_type_to_idx
        total_area: Total floor area in square meters (or square feet)
        params: Dict of additional parameters (e.g., aspect_ratio, wing_lengths)

    Returns:
        boundary: np.array of shape [N, 4] (x, y, width, height per point)
    """
    params = params or {}

    if footprint_type == 'rectangular':
        aspect_ratio = params.get('aspect_ratio', 1.5)  # width / height

        # Calculate dimensions
        height = np.sqrt(total_area / aspect_ratio)
        width = total_area / height

        # Scale to 0-256 coordinate system
        scale = min(200 / width, 200 / height)
        width *= scale
        height *= scale

        # Center in 256x256 space
        x0, y0 = (256 - width) / 2, (256 - height) / 2

        # Create rectangular boundary
        boundary = np.array([
            [x0, y0, 0, 0],           # Bottom-left
            [x0 + width, y0, 0, 0],   # Bottom-right (entrance on bottom edge)
            [x0 + width, y0 + height, 0, 0],  # Top-right
            [x0, y0 + height, 0, 0]    # Top-left
        ])

    elif footprint_type == 'L_shape':
        # L-shape parameters
        main_width = params.get('main_width', 150)
        main_height = params.get('main_height', 120)
        wing_width = params.get('wing_width', 80)
        wing_height = params.get('wing_height', 80)

        x0, y0 = 50, 50  # Start position

        # Create L-shaped boundary (7 points for L-shape)
        boundary = np.array([
            [x0, y0, 0, 0],                              # Bottom-left corner
            [x0 + main_width, y0, 0, 0],                 # Bottom-right corner
            [x0 + main_width, y0 + wing_height, 0, 0],   # Inner corner 1
            [x0 + wing_width, y0 + wing_height, 0, 0],   # Inner corner 2
            [x0 + wing_width, y0 + main_height, 0, 0],   # Top of wing
            [x0, y0 + main_height, 0, 0],                # Top-left corner
        ])

    elif footprint_type == 'U_shape':
        # U-shape parameters
        width = params.get('width', 180)
        height = params.get('height', 150)
        courtyard_width = params.get('courtyard_width', 80)
        courtyard_depth = params.get('courtyard_depth', 60)

        x0, y0 = 40, 50

        # Create U-shaped boundary (external outline)
        boundary = np.array([
            [x0, y0, 0, 0],
            [x0 + width, y0, 0, 0],
            [x0 + width, y0 + height, 0, 0],
            [x0 + width - (width - courtyard_width)/2, y0 + height, 0, 0],
            [x0 + width - (width - courtyard_width)/2, y0 + height - courtyard_depth, 0, 0],
            [x0 + (width - courtyard_width)/2, y0 + height - courtyard_depth, 0, 0],
            [x0 + (width - courtyard_width)/2, y0 + height, 0, 0],
            [x0, y0 + height, 0, 0]
        ])

    elif footprint_type == 'courtyard':
        # Full courtyard (donut shape)
        outer_width = params.get('outer_width', 200)
        outer_height = params.get('outer_height', 200)
        inner_width = params.get('inner_width', 100)
        inner_height = params.get('inner_height', 100)

        # This requires TWO boundaries (outer and inner)
        # For simplicity, return outer boundary only
        # Inner courtyard handled separately as "no-room zone"
        x0, y0 = 30, 30

        boundary = np.array([
            [x0, y0, 0, 0],
            [x0 + outer_width, y0, 0, 0],
            [x0 + outer_width, y0 + outer_height, 0, 0],
            [x0, y0 + outer_height, 0, 0]
        ])

        # Store inner courtyard separately
        inner_x0 = x0 + (outer_width - inner_width) / 2
        inner_y0 = y0 + (outer_height - inner_height) / 2

        inner_boundary = np.array([
            [inner_x0, inner_y0, 0, 0],
            [inner_x0 + inner_width, inner_y0, 0, 0],
            [inner_x0 + inner_width, inner_y0 + inner_height, 0, 0],
            [inner_x0, inner_y0 + inner_height, 0, 0]
        ])

        return boundary, inner_boundary  # Return both

    elif footprint_type == 'irregular':
        # For irregular, user draws custom boundary
        # Return None to indicate manual drawing needed
        return None

    return boundary
```

**Usage in UI**:
```python
# In views.py, when user selects footprint type
footprint_type = request.POST.get('footprint_type', 'rectangular')
total_area = request.POST.get('total_area', 2000)  # sq ft

if footprint_type != 'irregular':
    # Auto-generate boundary
    boundary = generate_boundary_from_footprint(footprint_type, total_area)
    # User can still adjust vertices if needed
else:
    # User draws custom boundary
    pass
```

**Why**: Allows automatic generation of standard footprint shapes, saving user time.

#### D. Footprint Constraint in Loss

**File**: `/Network/train.py`

**Location**: Loss function section (around lines 240-280)

**What to Add**: `footprint_compliance_loss`

**Example**:
```python
def compute_footprint_constraint_loss(boxes, footprint_type, boundary):
    """
    Penalize room placements that violate footprint shape

    Args:
        boxes: [B, N, 4] room bounding boxes (normalized 0-1)
        footprint_type: [B] footprint type IDs
        boundary: [B, K, 2] boundary points

    Returns:
        loss: scalar tensor
    """
    loss = 0.0

    for i in range(boxes.size(0)):
        fp_type = footprint_type[i].item()

        if fp_type == vocab['footprint_type_to_idx']['L_shape']:
            # For L-shape, penalize rooms in the "missing" corner
            # Identify the missing corner region from boundary

            # Example: If L-shape has missing top-right corner
            # Rooms in that region should have high penalty

            boundary_i = boundary[i]  # [K, 2]

            # Find bounding box of boundary
            x_min, y_min = boundary_i[:, 0].min(), boundary_i[:, 1].min()
            x_max, y_max = boundary_i[:, 0].max(), boundary_i[:, 1].max()

            # Identify missing corner (the one not covered by boundary)
            # This is simplified - actual implementation needs proper polygon analysis

            for j in range(boxes.size(1)):
                box = boxes[i, j]  # [4]
                room_center = torch.stack([(box[0] + box[2])/2, (box[1] + box[3])/2])

                # Check if room center is in "invalid" region
                # For L-shape missing top-right: penalize if x > threshold AND y > threshold
                threshold_x = x_min + 0.7 * (x_max - x_min)
                threshold_y = y_min + 0.7 * (y_max - y_min)

                if room_center[0] > threshold_x and room_center[1] > threshold_y:
                    # Room is in missing corner region
                    loss += 1.0

        elif fp_type == vocab['footprint_type_to_idx']['courtyard']:
            # For courtyard, ensure central region is empty

            # Define courtyard center region (e.g., middle 30% of boundary)
            boundary_i = boundary[i]
            x_min, y_min = boundary_i[:, 0].min(), boundary_i[:, 1].min()
            x_max, y_max = boundary_i[:, 0].max(), boundary_i[:, 1].max()

            courtyard_x_min = x_min + 0.35 * (x_max - x_min)
            courtyard_x_max = x_min + 0.65 * (x_max - x_min)
            courtyard_y_min = y_min + 0.35 * (y_max - y_min)
            courtyard_y_max = y_min + 0.65 * (y_max - y_min)

            for j in range(boxes.size(1)):
                box = boxes[i, j]
                room_center = torch.stack([(box[0] + box[2])/2, (box[1] + box[3])/2])

                # Check if room overlaps with courtyard region
                in_courtyard_x = (room_center[0] > courtyard_x_min and
                                  room_center[0] < courtyard_x_max)
                in_courtyard_y = (room_center[1] > courtyard_y_min and
                                  room_center[1] < courtyard_y_max)

                if in_courtyard_x and in_courtyard_y:
                    # Room is in courtyard region - high penalty
                    loss += 2.0

    return loss / boxes.size(0)

# In training loop:
footprint_loss = compute_footprint_constraint_loss(boxes_pred, footprint_types, boundary)
total_loss = total_loss + 0.25 * footprint_loss
```

**Why**: Enforces that generated layouts respect the specified footprint shape.

#### E. Total Area/Aspect Ratio Constraints

**File**: `/Network/train.py`

**Location**: Loss function section

**What to Add**: `area_constraint_loss`

**Example**:
```python
def compute_area_constraint_loss(boxes, target_area, target_aspect_ratio=None):
    """
    Penalize layouts that deviate from target area or aspect ratio

    Args:
        boxes: [B, N, 4] room bounding boxes (normalized 0-1)
        target_area: [B] target total area (normalized, e.g., 0.6 means 60% of boundary)
        target_aspect_ratio: [B] optional target width/height ratio

    Returns:
        loss: scalar tensor
    """
    loss = 0.0

    for i in range(boxes.size(0)):
        # Calculate total area of all rooms
        total_area = 0.0
        min_x, min_y = 1.0, 1.0
        max_x, max_y = 0.0, 0.0

        for j in range(boxes.size(1)):
            box = boxes[i, j]
            room_area = (box[2] - box[0]) * (box[3] - box[1])
            total_area += room_area

            # Track bounding box of all rooms
            min_x = min(min_x, box[0])
            min_y = min(min_y, box[1])
            max_x = max(max_x, box[2])
            max_y = max(max_y, box[3])

        # Area constraint
        target = target_area[i].item()
        area_diff = abs(total_area - target)
        loss += area_diff

        # Aspect ratio constraint (if specified)
        if target_aspect_ratio is not None:
            width = max_x - min_x
            height = max_y - min_y
            actual_ratio = width / (height + 1e-6)
            target_ratio = target_aspect_ratio[i].item()

            ratio_diff = abs(actual_ratio - target_ratio)
            loss += 0.5 * ratio_diff

    return loss / boxes.size(0)

# In training loop:
area_loss = compute_area_constraint_loss(boxes_pred, target_areas, target_aspect_ratios)
total_loss = total_loss + 0.15 * area_loss
```

**Why**: Ensures generated layouts match user's space requirements and proportions.

#### F. UI Control for Footprint Selection

**File**: `/Interface/Houseweb/views.py`

**Location**: `NumSearch` endpoint (boundary specification step)

**What to Add**: Footprint selector UI

**Backend Changes**:
```python
@csrf_exempt
def NumSearch(request):
    if request.method == 'POST':
        data = json.loads(request.body)

        # NEW: Footprint type selection
        footprint_type = data.get('footprint_type', 'rectangular')
        total_area = data.get('total_area', 2000)  # Square feet
        aspect_ratio = data.get('aspect_ratio', 1.5)

        # Auto-generate boundary if not irregular
        if footprint_type != 'irregular':
            params = {
                'aspect_ratio': aspect_ratio,
                'total_area': total_area
            }
            boundary = generate_boundary_from_footprint(footprint_type, total_area, params)

            # Convert to response format
            boundary_data = boundary.tolist()
        else:
            # User will draw custom boundary
            boundary_data = data.get('boundary')  # User-drawn points

        # Store footprint type with test data
        test_data.footprint_type = vocab['footprint_type_to_idx'][footprint_type]
        test_data.boundary = np.array(boundary_data)

        # ... rest of existing code ...
```

**Frontend Addition** (`/Interface/static/html/index.html` or template):
```html
<!-- Add to boundary specification page -->
<div class="footprint-selector">
    <h3>Select Building Footprint</h3>

    <div class="footprint-options">
        <button class="footprint-btn" data-type="rectangular">
            <img src="/static/icons/rectangular.svg" alt="Rectangular">
            <span>Rectangular</span>
        </button>

        <button class="footprint-btn" data-type="L_shape">
            <img src="/static/icons/l_shape.svg" alt="L-Shape">
            <span>L-Shape</span>
        </button>

        <button class="footprint-btn" data-type="U_shape">
            <img src="/static/icons/u_shape.svg" alt="U-Shape">
            <span>U-Shape</span>
        </button>

        <button class="footprint-btn" data-type="courtyard">
            <img src="/static/icons/courtyard.svg" alt="Courtyard">
            <span>Courtyard</span>
        </button>

        <button class="footprint-btn" data-type="irregular">
            <img src="/static/icons/irregular.svg" alt="Custom">
            <span>Custom</span>
        </button>
    </div>

    <!-- Parameters -->
    <div class="footprint-params">
        <label>
            Total Area (sq ft):
            <input type="number" id="total-area" value="2000" min="500" max="10000">
        </label>

        <label>
            Aspect Ratio (width/height):
            <input type="number" id="aspect-ratio" value="1.5" min="0.5" max="3.0" step="0.1">
        </label>

        <!-- Additional params for L-shape, U-shape, etc. -->
        <div id="advanced-params" style="display:none;">
            <!-- Dynamically populated based on footprint type -->
        </div>
    </div>

    <button id="generate-boundary-btn">Generate Boundary</button>
    <button id="draw-custom-btn">Draw Custom</button>
</div>
```

**Frontend JavaScript** (`/Interface/static/js/buttonEvent.js`):
```javascript
// Footprint selection handler
document.querySelectorAll('.footprint-btn').forEach(btn => {
    btn.addEventListener('click', function() {
        // Highlight selected
        document.querySelectorAll('.footprint-btn').forEach(b => b.classList.remove('selected'));
        this.classList.add('selected');

        selectedFootprintType = this.dataset.type;

        // Show/hide advanced params
        if (['L_shape', 'U_shape', 'courtyard'].includes(selectedFootprintType)) {
            document.getElementById('advanced-params').style.display = 'block';
            loadAdvancedParams(selectedFootprintType);
        } else {
            document.getElementById('advanced-params').style.display = 'none';
        }
    });
});

// Generate boundary button
document.getElementById('generate-boundary-btn').addEventListener('click', function() {
    const footprintType = selectedFootprintType;
    const totalArea = document.getElementById('total-area').value;
    const aspectRatio = document.getElementById('aspect-ratio').value;

    // Send to backend
    fetch('/api/generate-boundary', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
            footprint_type: footprintType,
            total_area: parseFloat(totalArea),
            aspect_ratio: parseFloat(aspectRatio)
        })
    })
    .then(response => response.json())
    .then(data => {
        // Draw generated boundary on canvas
        drawBoundary(data.boundary);

        // Enable fine-tuning by dragging vertices
        enableBoundaryEditing(data.boundary);
    });
});
```

**Why**: Provides user-friendly interface for footprint selection with visual feedback.

---

## Integration Points (Where Features Interact)

### Combined Sun + Entrance Feature

**Challenge**: Living room should face sun AND be near entrance (potentially conflicting constraints)

**Solution Location**:
- **File**: `/Network/train.py`
- **What**: Multi-objective loss balancing with adaptive weights

**Example**:
```python
def compute_combined_loss(boxes, room_types, sun_direction, boundary):
    """Balance multiple competing objectives"""

    sun_loss = compute_sun_orientation_loss(boxes, room_types, sun_direction, boundary)
    entrance_loss = compute_entrance_adjacency_loss(boxes, room_types, boundary)

    # Adaptive weighting: if sun and entrance are on opposite sides,
    # reduce both weights to allow compromise
    entrance_pos = boundary[:, :2].mean(1)  # [B, 2]
    sun_pos = sun_direction  # [B, 2] (direction vector)

    # Check alignment between entrance and sun direction
    alignment = torch.sum(entrance_pos * sun_pos, dim=1)  # [B]

    # If aligned (same direction): both constraints can be satisfied
    # If opposite: need to compromise
    weight_multiplier = torch.sigmoid(alignment)  # 0-1, higher if aligned

    # Combined loss
    total = (0.3 * weight_multiplier * sun_loss +
             0.2 * weight_multiplier * entrance_loss)

    return total
```

**Why**: When constraints conflict, model needs to find reasonable compromises.

### Combined Foundation + Footprint

**Challenge**: Basement footprint might differ from ground floor (e.g., partial basement under L-wing)

**Solution Location**:
- **File**: `/Interface/model/floorplan.py`
- **What**: Store per-floor boundaries

**Example**:
```python
class FloorPlan():
    def __init__(self, data, train=False, rot=None):
        # ... existing code ...

        # NEW: Per-floor boundary support
        self.boundary_by_floor = {}

        if hasattr(data, 'boundary_by_floor'):
            # Multi-floor building with different footprints per floor
            for floor_num, boundary in data.boundary_by_floor.items():
                self.boundary_by_floor[floor_num] = boundary
        else:
            # Single floor or all floors have same boundary
            self.boundary_by_floor[0] = data.boundary

    def get_boundary_for_floor(self, floor_num):
        """Get boundary polygon for specific floor"""
        if floor_num in self.boundary_by_floor:
            return self.boundary_by_floor[floor_num]
        else:
            # Default to ground floor boundary
            return self.boundary_by_floor.get(0, self.data.boundary)

    def get_footprint_type_for_floor(self, floor_num):
        """Different floors might have different footprints"""
        # Example: Ground floor is L-shape, second floor is rectangular
        if hasattr(self.data, 'footprint_by_floor'):
            return self.data.footprint_by_floor.get(floor_num, 'rectangular')
        else:
            return self.data.footprint_type if hasattr(self.data, 'footprint_type') else 'rectangular'
```

**Why**: Real buildings often have different footprints on different floors (e.g., setbacks, partial upper floors).

### Combined Sun + Foundation

**Challenge**: Upper floor rooms have better sun exposure, balconies should be on sunny side AND upper floor

**Solution Location**:
- **File**: `/Network/train.py`
- **Location**: Loss function

**Example**:
```python
def compute_sun_floor_preference_loss(boxes, room_types, floor_levels, sun_direction):
    """Some rooms prefer both sun AND specific floor"""

    loss = 0.0

    # Combined preferences (room_type -> [sun_pref, floor_pref])
    combined_prefs = {
        9: [1.0, 1.0],   # Balcony: strong preference for sun AND upper floor
        0: [1.0, 0.0],   # LivingRoom: sun preference but ground floor OK
        1: [0.5, 1.0],   # MasterRoom: moderate sun, strong upper floor preference
        # etc.
    }

    for i in range(boxes.size(0)):
        for j in range(boxes.size(1)):
            room_type = room_types[i, j].item()
            if room_type not in combined_prefs:
                continue

            sun_pref, floor_pref = combined_prefs[room_type]
            floor = floor_levels[i, j].item()

            # Sun orientation penalty
            room_center = (boxes[i, j, :2] + boxes[i, j, 2:]) / 2
            sun_alignment = compute_sun_alignment(room_center, sun_direction[i])
            sun_penalty = sun_pref * (1.0 - sun_alignment)

            # Floor level penalty
            target_floor = 1 if floor_pref > 0.5 else 0
            floor_penalty = floor_pref * abs(floor - target_floor)

            # Combined (both matter)
            loss += sun_penalty + floor_penalty

    return loss / (boxes.size(0) * boxes.size(1))
```

**Why**: Some room types have preferences that span multiple feature dimensions.

---

## Data Pipeline Changes Summary

For all features, you'll need to update the entire data pipeline from raw data to model input.

### 1. Raw Data Annotation

**Where**: Your ResPlan preprocessing scripts (before MAT conversion)

**What to Add**: New fields to each floor plan in your dataset

**Example Script Changes**:
```python
# In your ResPlan-to-intermediate format converter

def process_resplan_floorplan(resplan_json_path):
    """Process raw ResPlan JSON file"""

    with open(resplan_json_path, 'r') as f:
        data = json.load(f)

    # Existing fields
    boundary = extract_boundary(data)
    rooms = extract_rooms(data)
    edges = extract_adjacency(data)

    # NEW: Extract or annotate additional fields

    # 1. Sun direction (from building orientation metadata or manual annotation)
    if 'building_orientation' in data:
        sun_angle = data['building_orientation']  # Degrees
    else:
        # Default or manual annotation
        sun_angle = 180  # South-facing

    sun_direction = [np.sin(np.radians(sun_angle)),
                     np.cos(np.radians(sun_angle))]

    # 2. Entrance position (identify from door/entrance annotation)
    entrance_rooms = [r for r in rooms if r['type'] == 'FrontDoor']
    if entrance_rooms:
        entrance_pos = entrance_rooms[0]['center']
    else:
        # Default to boundary midpoint
        entrance_pos = boundary[:2].mean(0)

    # 3. Floor levels (if multi-story)
    if 'floors' in data:
        # Multi-story building
        floor_levels = []
        for room in rooms:
            floor_levels.append(room.get('floor', 0))  # Default to ground floor
    else:
        # Single-story
        floor_levels = [0] * len(rooms)

    # 4. Foundation type (manual annotation or heuristic)
    if 'foundation' in data:
        foundation_type = data['foundation']
    else:
        # Heuristic: if has basement floor (floor < 0), type is 'basement'
        if any(fl < 0 for fl in floor_levels):
            foundation_type = 'basement'
        else:
            foundation_type = 'slab'

    # 5. Footprint type (classify from boundary shape or manual annotation)
    if 'footprint_type' in data:
        footprint_type = data['footprint_type']
    else:
        # Auto-classify from boundary shape
        footprint_type = classify_footprint_from_boundary(boundary)

    return {
        'boundary': boundary,
        'rooms': rooms,
        'edges': edges,
        'sun_direction': sun_direction,            # NEW
        'entrance_position': entrance_pos,         # NEW
        'floor_levels': floor_levels,              # NEW
        'foundation_type': foundation_type,        # NEW
        'footprint_type': footprint_type          # NEW
    }

def classify_footprint_from_boundary(boundary):
    """Auto-classify footprint shape from boundary polygon"""
    num_vertices = len(boundary)

    # Simple heuristics
    if num_vertices == 4:
        # Could be rectangle or square
        # Check if aspect ratio ~1.0 for square
        x_range = boundary[:, 0].max() - boundary[:, 0].min()
        y_range = boundary[:, 1].max() - boundary[:, 1].min()
        aspect_ratio = x_range / y_range

        if 0.9 < aspect_ratio < 1.1:
            return 'square'
        else:
            return 'rectangular'

    elif num_vertices == 6:
        # Likely L-shape
        return 'L_shape'

    elif num_vertices == 8:
        # Could be U-shape or H-shape
        # More sophisticated analysis needed
        return 'U_shape'

    else:
        return 'irregular'
```

**Why**: Model can only learn features that exist in the training data.

### 2. MAT File Format

**Where**: Your converter script that generates `.mat` files for training

**What to Update**: Add new fields to MAT file structure

**Example**:
```python
# In your converter: intermediate_format_to_mat.py

def convert_to_mat(processed_data_list, output_path):
    """
    Convert processed floor plan data to MATLAB .mat format

    Args:
        processed_data_list: List of dicts from process_resplan_floorplan()
        output_path: Path to save .mat file
    """

    N = len(processed_data_list)  # Number of samples

    # Find max dimensions for padding
    max_rooms = max(len(d['rooms']) for d in processed_data_list)
    max_edges = max(len(d['edges']) for d in processed_data_list)
    max_boundary = max(len(d['boundary']) for d in processed_data_list)

    # Initialize arrays
    boundaries = np.zeros((N, max_boundary, 4))
    boxes = np.zeros((N, max_rooms, 6))  # NOW 6 columns (added floor_level)
    edges = np.zeros((N, max_edges, 3))
    genes = np.zeros((N, 128, 128))

    # NEW: Additional arrays
    sun_directions = np.zeros((N, 2))
    entrance_positions = np.zeros((N, 2))
    foundation_types = np.zeros((N, 1))
    footprint_types = np.zeros((N, 1))

    for i, data in enumerate(processed_data_list):
        # Existing fields
        boundaries[i, :len(data['boundary'])] = data['boundary']

        # Boxes: [x0, y0, x1, y1, room_type, floor_level]
        for j, room in enumerate(data['rooms']):
            boxes[i, j] = [
                room['x0'], room['y0'], room['x1'], room['y1'],
                room['type'],
                data['floor_levels'][j]  # NEW: 6th column
            ]

        edges[i, :len(data['edges'])] = data['edges']
        genes[i] = data['gene_layout']

        # NEW: Additional fields
        sun_directions[i] = data['sun_direction']
        entrance_positions[i] = data['entrance_position']
        foundation_types[i] = vocab['foundation_type_to_idx'][data['foundation_type']]
        footprint_types[i] = vocab['footprint_type_to_idx'][data['footprint_type']]

    # Save to MAT file
    sio.savemat(output_path, {
        'boundary': boundaries,
        'boxes': boxes,
        'edges': edges,
        'genes': genes,
        'sun_direction': sun_directions,         # NEW
        'entrance_position': entrance_positions, # NEW
        'foundation_type': foundation_types,     # NEW
        'footprint_type': footprint_types       # NEW
    })

    print(f"Saved {N} samples to {output_path}")
```

**Updated MAT Structure**:
```matlab
data_train.mat:
    - boundary: [N x K x 4] - Floor plan boundaries
    - boxes: [N x M x 6] - Room boxes (x0, y0, x1, y1, type, floor_level)
    - edges: [N x E x 3] - Room adjacency graph
    - genes: [N x 128 x 128] - Pixel-wise room layout
    - sun_direction: [N x 2] - Sun direction vectors (NEW)
    - entrance_position: [N x 2] - Entrance XY coordinates (NEW)
    - foundation_type: [N x 1] - Foundation type IDs (NEW)
    - footprint_type: [N x 1] - Footprint type IDs (NEW)
```

**Why**: Training script loads data from MAT files - new fields must be included.

### 3. Dataset Class

**File**: `/Network/model/dataset.py`

**What to Update**: `FloorPlanDataset.__getitem__()` to load and return new fields

**Example**:
```python
class FloorPlanDataset(Dataset):
    def __init__(self, mat_path):
        self.data = sio.loadmat(mat_path)
        self.N = self.data['boundary'].shape[0]

        # Check if new fields exist (backwards compatibility)
        self.has_sun = 'sun_direction' in self.data
        self.has_entrance = 'entrance_position' in self.data
        self.has_foundation = 'foundation_type' in self.data
        self.has_footprint = 'footprint_type' in self.data

    def __getitem__(self, idx):
        # Existing fields
        boundary = torch.tensor(self.data['boundary'][idx]).float()
        boxes = torch.tensor(self.data['boxes'][idx]).float()
        edges = torch.tensor(self.data['edges'][idx]).long()
        gene = torch.tensor(self.data['genes'][idx]).float()

        # Room types (5th column)
        room_types = boxes[:, 4].long()

        # NEW: Floor levels (6th column)
        if boxes.shape[1] >= 6:
            floor_levels = boxes[:, 5].long()
        else:
            floor_levels = torch.zeros(len(boxes), dtype=torch.long)

        # NEW: Additional features
        if self.has_sun:
            sun_direction = torch.tensor(self.data['sun_direction'][idx]).float()
        else:
            sun_direction = torch.tensor([0.0, -1.0])  # Default: South

        if self.has_entrance:
            entrance_pos = torch.tensor(self.data['entrance_position'][idx]).float()
        else:
            entrance_pos = boundary[:2].mean(0)  # Default: first edge center

        if self.has_foundation:
            foundation_type = torch.tensor(self.data['foundation_type'][idx]).long()
        else:
            foundation_type = torch.tensor(0)  # Default: slab

        if self.has_footprint:
            footprint_type = torch.tensor(self.data['footprint_type'][idx]).long()
        else:
            footprint_type = torch.tensor(0)  # Default: rectangular

        return {
            'boundary': boundary,
            'boxes': boxes[:, :4],  # Only coordinates
            'room_types': room_types,
            'floor_levels': floor_levels,       # NEW
            'edges': edges,
            'gene': gene,
            'sun_direction': sun_direction,     # NEW
            'entrance_position': entrance_pos,  # NEW
            'foundation_type': foundation_type, # NEW
            'footprint_type': footprint_type   # NEW
        }

    def __len__(self):
        return self.N
```

**Why**: Dataset loader must provide new fields to training loop.

### 4. Training Loop Updates

**File**: `/Network/train.py`

**What to Update**: Pass new fields to model and loss functions

**Example**:
```python
def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0

    for batch in dataloader:
        # Existing fields
        boundary = batch['boundary'].to(device)
        boxes = batch['boxes'].to(device)
        room_types = batch['room_types'].to(device)
        edges = batch['edges'].to(device)
        gene_gt = batch['gene'].to(device)

        # NEW fields
        floor_levels = batch['floor_levels'].to(device)
        sun_direction = batch['sun_direction'].to(device)
        entrance_pos = batch['entrance_position'].to(device)
        foundation_type = batch['foundation_type'].to(device)
        footprint_type = batch['footprint_type'].to(device)

        # Forward pass with new features
        boxes_pred, gene_pred, boxes_refine = model(
            room_types,
            edges,
            boundary,
            floor_levels=floor_levels,        # NEW
            sun_direction=sun_direction,      # NEW
            foundation_type=foundation_type,  # NEW
            footprint_type=footprint_type     # NEW
        )

        # Compute losses
        base_loss = compute_base_loss(boxes_pred, boxes, gene_pred, gene_gt)

        # NEW: Additional constraint losses
        sun_loss = compute_sun_orientation_loss(boxes_pred, room_types, sun_direction, boundary)
        entrance_loss = compute_entrance_adjacency_loss(boxes_pred, room_types, entrance_pos)
        vertical_loss = compute_vertical_constraint_loss(boxes_pred, room_types, floor_levels)
        footprint_loss = compute_footprint_constraint_loss(boxes_pred, footprint_type, boundary)

        # Total loss (weighted sum)
        loss = (base_loss +
                0.1 * sun_loss +
                0.15 * entrance_loss +
                0.2 * vertical_loss +
                0.25 * footprint_loss)

        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)
```

**Why**: Training loop coordinates all components - must pass new data through entire pipeline.

---

## Recommended Implementation Order

Based on complexity, dependencies, and user impact:

### Phase 1: Quick Wins (No Model Retraining Required)

**Timeline**: 1-2 weeks

#### 1.1 Entrance Position Enhancement (3-4 days)
- **Why First**: Uses existing boundary data, minimal code changes
- **Files to Modify**:
  - `floorplan.py` - Add entrance adjacency in graph
  - `views.py` - UI for entrance edge selection
  - `buttonEvent.js` - Highlight entrance edge
- **Testing**: Verify entrance edge is correctly identified and highlighted
- **User Value**: Better UI for specifying entrance location

#### 1.2 Footprint Visualization (2-3 days)
- **Why Second**: Display-only, no model changes
- **Files to Modify**:
  - `views.py` - Detect and return footprint type
  - `buttonEvent.js` - Show footprint label
- **Testing**: Verify correct footprint classification
- **User Value**: Users see what footprint shape their boundary is

### Phase 2: Model Retraining (Single Feature)

**Timeline**: 3-4 weeks per feature

#### 2.1 Sun Positioning (3 weeks)
- **Why Third**: Single scalar feature, moderate complexity
- **Steps**:
  1. Week 1: Data annotation (add sun_direction to 17k samples)
  2. Week 2: Model changes (add sun embedding, train.py loss function)
  3. Week 3: Training (150 epochs), evaluation, UI integration
- **Files to Modify**: All 6 files listed in sun positioning section
- **Testing**:
  - Unit test: Sun embedding layer
  - Training test: Sun loss decreases
  - End-to-end: Living room faces sun in generated layouts
- **User Value**: More realistic layouts with proper room orientation

#### 2.2 Footprint Control (3 weeks)
- **Why Fourth**: Similar complexity to sun positioning
- **Steps**:
  1. Week 1: Footprint classification/annotation, boundary generation functions
  2. Week 2: Model changes (footprint embedding, constraint loss)
  3. Week 3: Training, UI for footprint selection
- **Files to Modify**: 6 files listed in footprint section
- **Testing**:
  - Unit test: Boundary generation for each footprint type
  - Training test: Footprint compliance loss decreases
  - End-to-end: Rooms respect footprint shape (no rooms in L-corner gap)
- **User Value**: Support for non-rectangular buildings

### Phase 3: Major Architecture Changes

**Timeline**: 2-3 months

#### 3.1 Foundation/Multi-level Support (8-10 weeks)
- **Why Last**: Most complex, requires 3D GNN architecture
- **Steps**:
  1. Weeks 1-2: Data structure changes (add floor_level to all data)
  2. Weeks 3-4: Model architecture (3D GNN, vertical edges, floor embeddings)
  3. Weeks 5-6: Loss functions (vertical constraints, stairwell alignment)
  4. Weeks 7-8: Training (may need more epochs for convergence)
  5. Weeks 9-10: UI overhaul (multi-floor visualization, tabbed interface)
- **Files to Modify**: 8+ files, major changes to model.py
- **Testing**:
  - Unit test: Vertical edge creation, floor embeddings
  - Integration test: Multi-floor data loading
  - Training test: Vertical constraint loss working
  - End-to-end: Bedrooms on upper floor, kitchen on ground floor
- **User Value**: Support for multi-story buildings

**Risk**: This is the most complex feature. Consider splitting into sub-phases:
- Phase 3a: Two-story buildings only (simpler)
- Phase 3b: Basement support
- Phase 3c: 3+ story buildings

---

## Files You'll Definitely Need to Modify (Per Feature)

### Sun Positioning

| File | Lines | Changes | Complexity |
|------|-------|---------|------------|
| Preprocessing script | N/A | Add sun_direction field | Low |
| `/Network/model/model.py` | 50-100 | Add sun embedding layer | Medium |
| `/Network/train.py` | 240-280 | Add sun orientation loss | Medium |
| `/Interface/model/floorplan.py` | 150+ | Add get_sun_direction() | Low |
| `/Interface/Houseweb/views.py` | 850-950 | Return sun in JSON | Low |
| `/Interface/static/js/buttonEvent.js` | N/A | Draw sun indicator | Low |

**Total Estimated LOC**: ~300-400 lines

---

### Entrance Position

| File | Lines | Changes | Complexity |
|------|-------|---------|------------|
| `/Interface/model/utils.py` | 330-340 | Add 'adjacent_to_entrance' predicate | Low |
| `/Interface/model/floorplan.py` | 110-133 | Add entrance edges in get_triples() | Medium |
| `/Network/train.py` | 240-280 | Add entrance adjacency loss | Medium |
| `/Interface/Houseweb/views.py` | 600-700 | Add entrance edge selection | Low |
| `/Interface/static/js/buttonEvent.js` | N/A | Entrance edge highlighting | Low |

**Total Estimated LOC**: ~200-300 lines

---

### Foundation (Multi-level)

| File | Lines | Changes | Complexity |
|------|-------|---------|------------|
| Preprocessing script | N/A | Add floor_level to all rooms | Medium |
| MAT converter | N/A | Update boxes array to 6 columns | Low |
| `/Network/model/dataset.py` | 30-60 | Load floor_levels from MAT | Low |
| `/Network/model/model.py` | 50-200 | Add floor embeddings, 3D GNN | **High** |
| `/Interface/model/utils.py` | 340+ | Add foundation_type vocab | Low |
| `/Interface/model/floorplan.py` | 100-250 | Add vertical edges, per-floor methods | High |
| `/Network/train.py` | 240-280 | Add vertical constraint loss | Medium |
| `/Interface/Houseweb/views.py` | 850-950 | Return multi-floor data | Medium |
| `/Interface/static/js/buttonEvent.js` | N/A | Multi-floor visualization UI | High |

**Total Estimated LOC**: ~800-1200 lines

---

### Footprint Control

| File | Lines | Changes | Complexity |
|------|-------|---------|------------|
| Preprocessing script | N/A | Classify/annotate footprint types | Medium |
| `/Interface/model/utils.py` | 350+ | Add footprint_type vocab | Low |
| `/Interface/model/floorplan.py` | 200+ | Add generate_boundary_from_footprint() | Medium |
| `/Network/model/model.py` | 50-100 | Add footprint embedding | Low |
| `/Network/train.py` | 240-280 | Add footprint constraint loss | Medium |
| `/Interface/Houseweb/views.py` | 600-800 | Footprint selection API | Medium |
| `/Interface/static/html/` | N/A | Footprint selector UI | Low |
| `/Interface/static/js/buttonEvent.js` | N/A | Footprint UI logic | Medium |

**Total Estimated LOC**: ~500-700 lines

---

## Testing Strategy for Each Feature

### Sun Positioning Test

#### Unit Tests:
```python
# test_sun_positioning.py

def test_sun_embedding():
    """Test sun direction encoding"""
    model = Model(vocab)
    sun_dir = torch.tensor([[0.0, 1.0], [1.0, 0.0]])  # North, East
    embedded = model.sun_encoder(sun_dir)
    assert embedded.shape == (2, 64)

def test_sun_loss_computation():
    """Test sun orientation loss"""
    boxes = torch.rand(2, 5, 4)  # 2 samples, 5 rooms
    room_types = torch.tensor([[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]])  # LivingRoom, etc.
    sun_dir = torch.tensor([[0.0, -1.0], [0.0, -1.0]])  # South
    boundary = torch.rand(2, 3, 128, 128)

    loss = compute_sun_orientation_loss(boxes, room_types, sun_dir, boundary)
    assert loss >= 0.0
    print(f"Sun loss: {loss.item()}")
```

#### Integration Test:
```python
def test_sun_end_to_end():
    """Test full pipeline with sun direction"""
    # Load model
    model = load_model()

    # Create test floor plan with sun from South
    fp = create_test_floorplan(sun_angle=180)

    # Generate layout
    boxes_pred, gene_pred, _ = test(model, fp)

    # Verify living room (type 0) is on south side
    living_room_idx = (fp.get_rooms() == 0).nonzero()[0]
    living_room_box = boxes_pred[living_room_idx]
    living_room_y_center = (living_room_box[1] + living_room_box[3]) / 2

    # South side should have low Y values (top of image)
    assert living_room_y_center < 128, "Living room should be on south side"
    print("✓ Living room correctly placed facing sun")
```

#### Visual Test:
1. Generate two layouts with identical rooms but sun at 0° vs 180°
2. Visually verify living room placement flips
3. Check loss values: `sun_orientation_loss` should be <0.5 for good layouts

---

### Entrance Position Test

#### Unit Tests:
```python
def test_entrance_adjacency():
    """Test entrance edge detection"""
    boundary = np.array([[50, 50, 0, 0], [200, 50, 0, 0], [200, 200, 0, 0], [50, 200, 0, 0]])
    door_line = boundary[:2]

    # Room near entrance
    room_box = np.array([100, 60, 150, 110])  # Close to door_line
    distance = compute_distance_to_entrance(room_box, door_line)
    assert distance < 50, "Room should be detected as near entrance"

def test_entrance_loss():
    """Test entrance constraint loss"""
    boxes = torch.tensor([[[0.4, 0.05, 0.6, 0.25]]])  # Room near entrance (y=0)
    room_types = torch.tensor([[3]])  # Bathroom
    boundary = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]])

    loss = compute_entrance_adjacency_loss(boxes, room_types, boundary)
    assert loss > 0.5, "Bathroom at entrance should have high penalty"
    print(f"Entrance loss for bathroom: {loss.item()}")
```

#### Integration Test:
```python
def test_entrance_placement():
    """Verify appropriate rooms near entrance"""
    # Generate layout
    fp = create_test_floorplan()
    boxes_pred, _, _ = test(model, fp)

    # Find entrance position (first edge of boundary)
    entrance_pos = fp.data.boundary[:2].mean(0)

    # Find closest room to entrance
    distances = []
    for box in boxes_pred:
        room_center = (box[:2] + box[2:]) / 2
        dist = np.linalg.norm(room_center - entrance_pos)
        distances.append(dist)

    closest_idx = np.argmin(distances)
    closest_room_type = fp.get_rooms()[closest_idx].item()

    # Should NOT be bathroom (type 3)
    assert closest_room_type != 3, "Bathroom should not be closest to entrance"
    # Should be living room (0), entrance (10), or dining room (4)
    assert closest_room_type in [0, 4, 10], f"Unexpected room type {closest_room_type} at entrance"
    print(f"✓ Room at entrance: {vocab['object_idx_to_name'][closest_room_type]}")
```

---

### Foundation Test

#### Unit Tests:
```python
def test_floor_level_loading():
    """Test floor level data loading"""
    dataset = FloorPlanDataset('data_test.mat')
    sample = dataset[0]

    assert 'floor_levels' in sample
    floor_levels = sample['floor_levels']
    assert floor_levels.min() >= -1, "Floor levels should be >= -1 (basement)"
    assert floor_levels.max() <= 3, "Floor levels should be <= 3"
    print(f"Floor levels in sample: {floor_levels.unique()}")

def test_vertical_edges():
    """Test vertical edge creation"""
    boxes = np.array([[50, 50, 100, 100, 0, 0], [55, 55, 105, 105, 1, 1]])  # Aligned rooms
    floor_levels = np.array([0, 1])

    fp = create_floorplan_from_boxes(boxes)
    triples = fp.get_triples()

    # Check for 'directly_above' edge
    above_edges = triples[triples[:, 1] == vocab['pred_name_to_idx']['directly_above']]
    assert len(above_edges) > 0, "Should have vertical edges for aligned rooms"

def test_vertical_loss():
    """Test vertical placement constraint"""
    boxes = torch.rand(1, 5, 4)
    room_types = torch.tensor([[1, 2, 5, 10, 11]])  # Master, Kitchen, Child, Entrance, Storage
    floor_levels = torch.tensor([[1, 0, 1, 0, -1]])  # Master/Child upper, Kitchen/Entrance ground, Storage basement

    loss = compute_vertical_constraint_loss(boxes, room_types, floor_levels)
    assert loss < 0.5, "Correct floor placement should have low loss"

    # Test incorrect placement
    bad_floor_levels = torch.tensor([[0, 1, 0, 1, 1]])  # Everything wrong
    bad_loss = compute_vertical_constraint_loss(boxes, room_types, bad_floor_levels)
    assert bad_loss > loss, "Incorrect placement should have higher loss"
```

#### Integration Test:
```python
def test_multi_floor_generation():
    """Test layout generation for multi-story building"""
    # Create 2-story test data
    fp = create_multi_floor_floorplan(num_floors=2)

    # Generate layout
    boxes_pred, gene_pred, _ = test(model, fp)

    # Get floor levels for generated rooms
    floor_levels = fp.get_floor_level()

    # Verify bedrooms on upper floor
    bedroom_types = [1, 5, 7]  # Master, Child, Second
    for i, room_type in enumerate(fp.get_rooms()):
        if room_type.item() in bedroom_types:
            assert floor_levels[i] >= 1, f"Bedroom type {room_type} should be on upper floor"

    # Verify kitchen on ground floor
    kitchen_idx = (fp.get_rooms() == 2).nonzero()[0]
    assert floor_levels[kitchen_idx] == 0, "Kitchen should be on ground floor"

    print("✓ Multi-floor room placement correct")
```

#### Visual Test:
1. Generate 2-story layout
2. Display each floor separately (tabbed view)
3. Verify:
   - Bedrooms on floor 2
   - Living/Kitchen/Dining on floor 1
   - Stairwells aligned across floors (if stairwell room type exists)

---

### Footprint Test

#### Unit Tests:
```python
def test_boundary_generation():
    """Test footprint boundary generation"""
    # Rectangular
    boundary = generate_boundary_from_footprint('rectangular', total_area=2000,
                                                params={'aspect_ratio': 1.5})
    assert boundary.shape[0] == 4, "Rectangle should have 4 vertices"

    # L-shape
    boundary = generate_boundary_from_footprint('L_shape', total_area=2000)
    assert boundary.shape[0] == 6, "L-shape should have 6 vertices"

    # U-shape
    boundary = generate_boundary_from_footprint('U_shape', total_area=2500)
    assert boundary.shape[0] == 8, "U-shape should have 8 vertices"

    print("✓ Boundary generation correct for all footprint types")

def test_footprint_classification():
    """Test auto-classification of footprint from boundary"""
    # Rectangular boundary
    rect_boundary = np.array([[50, 50, 0, 0], [200, 50, 0, 0], [200, 150, 0, 0], [50, 150, 0, 0]])
    fp_type = classify_footprint_from_boundary(rect_boundary)
    assert fp_type == 'rectangular', f"Expected rectangular, got {fp_type}"

    # L-shape boundary (6 vertices)
    l_boundary = np.array([[50, 50], [200, 50], [200, 120], [120, 120], [120, 200], [50, 200]])
    fp_type = classify_footprint_from_boundary(l_boundary)
    assert fp_type == 'L_shape', f"Expected L_shape, got {fp_type}"

def test_footprint_loss():
    """Test footprint constraint loss"""
    # L-shape footprint
    boxes = torch.tensor([[[0.8, 0.8, 0.95, 0.95]]])  # Room in "missing" corner
    footprint_type = torch.tensor([2])  # L_shape
    boundary = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [1.0, 0.6], [0.6, 0.6], [0.6, 1.0], [0.0, 1.0]]])

    loss = compute_footprint_constraint_loss(boxes, footprint_type, boundary)
    assert loss > 0.5, "Room in invalid region should have high penalty"
    print(f"Footprint loss: {loss.item()}")
```

#### Integration Test:
```python
def test_footprint_compliance():
    """Test layout respects footprint shape"""
    # Generate L-shape layout
    fp = create_test_floorplan(footprint_type='L_shape')
    boxes_pred, _, _ = test(model, fp)

    # Define invalid region (missing corner of L)
    invalid_region = {'x_min': 0.7, 'x_max': 1.0, 'y_min': 0.7, 'y_max': 1.0}

    # Check no rooms in invalid region
    for box in boxes_pred:
        room_center_x = (box[0] + box[2]) / 2
        room_center_y = (box[1] + box[3]) / 2

        in_invalid_x = (invalid_region['x_min'] < room_center_x < invalid_region['x_max'])
        in_invalid_y = (invalid_region['y_min'] < room_center_y < invalid_region['y_max'])

        assert not (in_invalid_x and in_invalid_y), "Room found in invalid L-shape corner!"

    print("✓ L-shape layout correctly avoids invalid region")
```

#### Visual Test:
1. Generate layouts for each footprint type (rectangular, L, U, courtyard)
2. Overlay footprint outline on generated layout
3. Verify no rooms extend outside footprint boundary
4. For courtyard: verify central space is empty

---

## Summary Checklist

Before implementing each feature:

### Sun Positioning:
- [ ] Annotate sun direction for all 17k ResPlan samples
- [ ] Add sun_direction field to MAT files
- [ ] Update model.py: add sun_encoder layer
- [ ] Update train.py: add compute_sun_orientation_loss()
- [ ] Update floorplan.py: add get_sun_direction() method
- [ ] Update views.py: return sun direction in JSON
- [ ] Update buttonEvent.js: draw sun indicator
- [ ] Write unit tests for sun embedding and loss
- [ ] Retrain model (150 epochs)
- [ ] Validate: living room faces sun

### Entrance Position:
- [ ] Update utils.py: add 'adjacent_to_entrance' to vocab
- [ ] Update floorplan.py: add entrance edges in get_triples()
- [ ] Update train.py: add compute_entrance_adjacency_loss()
- [ ] Update views.py: add entrance edge selection API
- [ ] Update buttonEvent.js: highlight entrance edge
- [ ] Write unit tests for entrance detection
- [ ] Validate: bathroom NOT at entrance

### Foundation (Multi-level):
- [ ] Update MAT converter: add 6th column (floor_level) to boxes
- [ ] Update dataset.py: load floor_levels
- [ ] Update utils.py: add foundation_type vocab
- [ ] Update model.py: add floor_embedding, 3D GNN
- [ ] Update floorplan.py: add vertical edges, per-floor boundaries
- [ ] Update train.py: add compute_vertical_constraint_loss()
- [ ] Update views.py: return multi-floor data structure
- [ ] Update buttonEvent.js: multi-floor UI (tabs/stacked view)
- [ ] Write unit tests for vertical edges and floor embeddings
- [ ] Retrain model (200+ epochs, more complex)
- [ ] Validate: bedrooms on upper floor, kitchen on ground

### Footprint Control:
- [ ] Annotate footprint types for all 17k samples
- [ ] Add footprint classification function
- [ ] Update utils.py: add footprint_type vocab
- [ ] Update floorplan.py: add generate_boundary_from_footprint()
- [ ] Update model.py: add footprint_embedding
- [ ] Update train.py: add compute_footprint_constraint_loss()
- [ ] Update views.py: add footprint selection API
- [ ] Create footprint selector UI (HTML/JS)
- [ ] Write unit tests for boundary generation
- [ ] Retrain model (150 epochs)
- [ ] Validate: rooms respect footprint shape

---

## Additional Considerations

### Performance Impact

Each feature adds computational cost:

| Feature | Training Time Increase | Inference Time Increase | GPU Memory Increase |
|---------|------------------------|-------------------------|---------------------|
| Sun Positioning | +5% | +2% | +100 MB |
| Entrance Position | +3% | +1% | +50 MB |
| Foundation (Multi-level) | +30% | +15% | +500 MB |
| Footprint Control | +10% | +5% | +200 MB |

**Combined**: Training time could increase by ~50%, memory by ~1 GB.

**Recommendation**: Implement features incrementally, measure impact after each.

### Data Requirements

Ensure your 17k ResPlan samples have sufficient diversity:

- **Sun directions**: Need examples facing all directions (N, S, E, W, NE, NW, SE, SW)
- **Footprint types**: Need balanced samples (not 95% rectangular, 5% other)
- **Floor levels**: For multi-level, need 2-3k multi-story examples minimum

**If data is imbalanced**: Use data augmentation:
- Rotate floor plans to create different sun orientations
- Mirror floor plans (L-shape → reversed L-shape)
- Synthetic generation for rare footprint types

### Backwards Compatibility

When adding new features, maintain compatibility with existing data/models:

```python
# Always check if new fields exist
if hasattr(data, 'sun_direction'):
    sun_dir = data.sun_direction
else:
    sun_dir = [0.0, -1.0]  # Default: South-facing

# Version your MAT files
mat_version = data.get('version', 1)
if mat_version >= 2:
    # Load new fields
    pass
```

### Documentation

For each feature, document:
1. **User Guide**: How to use the feature in UI
2. **Developer Guide**: Code architecture and design decisions
3. **Training Guide**: How to prepare data and retrain
4. **Troubleshooting**: Common issues and solutions

---

## Conclusion

This guide provides a comprehensive roadmap for implementing four major features:

1. **Sun Positioning** (Medium complexity) - Improve room orientation realism
2. **Entrance Position** (Low complexity) - Better entrance-related layout
3. **Foundation/Multi-level** (High complexity) - Support multi-story buildings
4. **Footprint Control** (Medium complexity) - Support diverse building shapes

**Recommended approach**:
1. Start with Entrance Position (quick win)
2. Then Sun Positioning or Footprint (parallel if you have resources)
3. Foundation last (most complex, requires architecture changes)

Each feature requires changes across the full stack:
- Data preprocessing
- MAT file format
- Dataset loading
- Model architecture
- Loss functions
- Training loop
- Interface backend
- Frontend UI

Good luck with your implementations! Refer back to this guide for specific file locations and code examples as you work through each feature.

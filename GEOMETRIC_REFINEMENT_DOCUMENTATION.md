# Geometric Refinement Documentation

**Created**: January 30, 2026
**Purpose**: Document the non-AI (rule-based) geometric refinement system in Graph2plan

---

## Overview

Graph2plan has **two separate refinement systems** that work in sequence:

1. **AI-based refinement** (`Network/model/`) - Neural network learns to refine box predictions
2. **Geometric refinement** (`Interface/align_fp/`) - Rule-based MATLAB system ensures physical validity

This document covers the **geometric refinement system** only. For AI refinement, see `IOU_AND_REFINEMENT_DOCUMENTATION.md`.

---

## Refinement Pipeline

The complete refinement pipeline works as follows:

```
User Input (graph)
    ↓
AI Model Prediction (initial boxes)
    ↓
AI Box Refinement (neural network)
    ↓
Geometric Refinement (MATLAB rules) ← THIS DOCUMENT
    ↓
Final Floor Plan Output
```

**Key insight**: Geometric refinement happens **AFTER** AI refinement, ensuring the final output is physically valid even if the AI makes mistakes.

---

## 1. Location and Files

### Primary Directory
`Interface/align_fp/`

### Key MATLAB Files

| File | Purpose | Lines |
|------|---------|-------|
| `align_fp.m` | Main entry point - orchestrates 4-step refinement | All |
| `align_with_boundary.m` | Step 1: Align boxes with boundary polygon | All |
| `align_neighbor.m` | Step 2: Align adjacent rooms to remove gaps | All |
| `regularize_fp.m` | Step 3: Crop, order, and fill remaining gaps | All |
| `get_room_boundary.m` | Step 4: Generate room polygons for rendering | All |
| `findCommonLine.m` | Utility: Find shared edges between rooms | All |
| `line_line_intersection.m` | Utility: Compute line intersections | All |

### Python Integration Files

| File | Purpose | Lines |
|------|---------|-------|
| `Interface/Houseweb/views.py` | Calls MATLAB via matlab.engine | 1091-1103 |
| `Interface/Houseweb/views.py` | Python fallback when MATLAB unavailable | 39-114 |

---

## 2. The Four-Step Process

### Step 1: Align with Boundary

**File**: `align_with_boundary.m`

**Purpose**: Ensure all room boxes fit within the floor plan boundary

**Process**:
1. Convert boundary polygon to box format
2. For each room box:
   - Check if it extends beyond boundary
   - Snap edges to boundary if within threshold
   - Crop boxes that exceed boundary significantly

**Threshold**: Default 8 pixels

**Output**: Boxes aligned to boundary + list of updated boxes

---

### Step 2: Align Neighbors

**File**: `align_neighbor.m`

**Purpose**: Remove gaps and overlaps between adjacent rooms

**Process**:
1. For each edge in the graph (adjacent room pairs):
   - Find the common boundary line
   - Check if boxes overlap or have a gap
   - If gap/overlap < threshold:
     - Move both boxes to meet at the midpoint
   - If gap/overlap > threshold:
     - Leave as-is (will be filled in Step 3)

**Threshold**: Default 14 pixels (threshold + 6 from main function)

**Output**: Boxes with aligned neighbors

---

### Step 3: Regularize Floor Plan

**File**: `regularize_fp.m`

**Purpose**: Crop to boundary, order by size, and fill remaining gaps

**Process**:

#### 3.1 Crop Boxes (Lines 5-14)
```matlab
% Ensure every box is strictly within boundary using polygon intersection
for i = 1:size(rBox, 1)
    bPoly = [rBox(i,1) rBox(i,2); rBox(i,3) rBox(i,2);
             rBox(i,3) rBox(i,4); rBox(i,1) rBox(i,4)];
    [xi, yi] = polybool('intersection', boundary(:,1), boundary(:,2),
                        bPoly(:,1), bPoly(:,2));
    if ~isempty(xi)
        rBox(i,:) = [min(xi) min(yi) max(xi) max(yi)];
    end
end
```

#### 3.2 Order by Size (Lines 17-39)
```matlab
% Sort rooms by area (largest first)
% This ensures large rooms (living room) render behind small rooms
rArea = (rBox(:,3) - rBox(:,1)) .* (rBox(:,4) - rBox(:,2));
[~, order] = sort(rArea, 'descend');
```

#### 3.3 Fill Gaps (Lines 41-110)
```matlab
% Strategy 1: Expand living room to fill gaps
if livingRoomExists
    expandedLivingRoom = expandToFillGaps(livingRoom, boundary, otherRooms);
end

% Strategy 2: Expand nearest room if no living room
else
    for each gap
        nearestRoom = findNearestRoom(gap, allRooms);
        expandRoom(nearestRoom, gap);
    end
end
```

**Output**: Cropped, ordered boxes with gaps filled

---

### Step 4: Generate Room Boundaries

**File**: `get_room_boundary.m`

**Purpose**: Convert boxes to polygons for rendering

**Process**:
1. For each room box `[x0, y0, x1, y1]`:
   - Create polygon: `[(x0,y0), (x1,y0), (x1,y1), (x0,y1)]`
   - Intersect with boundary to handle irregular shapes
   - Store as room boundary polygon

**Output**: Room polygons ready for visualization

---

## 3. Integration with Production Code

### Location: `Interface/Houseweb/views.py`

### Main Call Site: `Save_Editbox()` Function

**Lines 1091-1103**:

```python
def Save_Editbox(request):
    """
    Save user-edited floor plan after geometric refinement.
    Called when user clicks 'Save' in the web interface.
    """

    # ... extract data from request ...

    # Call MATLAB geometric refinement
    try:
        import matlab.engine
        eng = matlab.engine.start_matlab()
        eng.cd('Interface/align_fp', nargout=0)

        # Main refinement call
        newBox, order, rBoundary = eng.align_fp(
            boundary_matlab,  # Boundary polygon
            rBox_matlab,      # Room boxes from AI
            rType_matlab,     # Room types
            rEdge_matlab,     # Adjacency edges
            fp_matlab,        # Floor plan ID
            threshold,        # Alignment threshold (default 8)
            drawResult,       # Whether to visualize (0 or 1)
            nargout=3
        )

        eng.quit()

    except Exception as e:
        print(f"MATLAB refinement failed: {e}")
        # Fall back to Python implementation
        newBox, order, rBoundary = python_align_fp(...)

    # ... save results to database ...
```

**When it runs**: Every time a user saves an edited floor plan in the web interface

**Input format**:
- `boundary`: Floor plan boundary as Nx2 array `[[x1,y1], [x2,y2], ...]`
- `rBox`: Room boxes as Nx4 array `[[x0,y0,x1,y1], ...]`
- `rType`: Room type indices `[0, 1, 3, ...]` (0=LivingRoom, 1=Bedroom, etc.)
- `rEdge`: Adjacency edges `[[room_i, room_j], ...]`
- `fp`: Floor plan identifier (for debugging/logging)
- `threshold`: Snap threshold in pixels (default 8)
- `drawResult`: 0=no visualization, 1=save debug images

**Output format**:
- `newBox`: Refined boxes Nx4 array `[[x0,y0,x1,y1], ...]`
- `order`: Room ordering for rendering (largest to smallest)
- `rBoundary`: Room polygons for visualization

---

### Python Fallback Implementation

**Location**: `Interface/Houseweb/views.py` (Lines 39-114)

**Purpose**: When MATLAB is not installed, use pure Python implementation

**Function**: `python_align_fp()`

**Capabilities**:
- ✅ Boundary alignment (Step 1)
- ✅ Neighbor alignment (Step 2)
- ✅ Basic regularization (Step 3 - partial)
- ❌ Gap filling (Step 3 - not implemented)
- ✅ Room boundary generation (Step 4)

**Limitations**:
- No sophisticated gap filling (MATLAB version is better)
- Simpler overlap resolution
- No visualization output

**Code structure**:
```python
def python_align_fp(boundary, rBox, rType, rEdge, fp, threshold, drawResult):
    """Pure Python fallback for geometric refinement."""

    # Step 1: Align with boundary
    newBox = align_with_boundary_python(rBox, boundary, threshold)

    # Step 2: Align neighbors
    newBox = align_neighbor_python(newBox, rEdge, threshold)

    # Step 3: Regularize (basic version)
    newBox = crop_to_boundary(newBox, boundary)
    order = sort_by_area(newBox)

    # Step 4: Generate room boundaries
    rBoundary = boxes_to_polygons(newBox, boundary)

    return newBox, order, rBoundary
```

---

## 4. Comparison: AI vs Geometric Refinement

### AI-Based Refinement (`Network/model/`)

**Type**: Neural network (CNN + RoI Align + MLP)

**Training**: Learns from 17,000 floor plan examples

**Input**: Graph structure + object features + initial boxes

**Output**: Refined box coordinates `[xc, yc, w, h]`

**Strengths**:
- Learns complex spatial patterns
- Adapts to dataset-specific conventions
- Can predict better proportions

**Weaknesses**:
- Can produce invalid boxes (outside boundary, gaps, overlaps)
- No guarantees of physical validity
- Requires training data

**When it runs**: During model inference (training or prediction)

---

### Geometric Refinement (`Interface/align_fp/`)

**Type**: Rule-based system (MATLAB + geometry algorithms)

**Training**: None (hand-crafted rules)

**Input**: Boundary + boxes + adjacency graph

**Output**: Refined boxes + room polygons

**Strengths**:
- **Always** produces valid floor plans
- Guarantees boundary compliance
- Fills all gaps
- Resolves all overlaps
- Fast (no neural network inference)

**Weaknesses**:
- Cannot improve proportions or aesthetics
- Uses simple heuristics (nearest room expansion)
- Cannot learn from data

**When it runs**: After AI refinement, when user saves floor plan

---

## 5. The Complete Workflow

### Scenario: User generates a floor plan

```
1. User provides input:
   - Room requirements: 3 bedrooms, 2 bathrooms, 1 kitchen, 1 living room
   - Boundary: Irregular polygon shape

2. AutoAdjustGraph() processes graph:
   - Creates nodes and edges
   - Removes excess rooms (edge-aware)

3. AI Model predicts initial layout:
   - boxes_pred: Initial box predictions

4. AI Refinement (if enabled):
   - boxes_refine: Neural network refines boxes
   - Output: Better proportions, but may have gaps/overlaps

5. User edits in web interface (optional):
   - Move rooms, resize, add/remove

6. User clicks 'Save'

7. Geometric Refinement runs:
   Step 1: Align with boundary (snap to edges)
   Step 2: Align neighbors (close small gaps)
   Step 3: Regularize (crop, order, fill large gaps)
   Step 4: Generate polygons (for rendering)

8. Final output saved to database:
   - Boxes guaranteed to be valid
   - No gaps, no overlaps
   - Fully within boundary
```

---

## 6. Configuration and Tuning

### Threshold Parameter

**Purpose**: Controls how aggressively geometric refinement snaps/aligns boxes

**Default**: 8 pixels

**Effects**:
- **Lower threshold (4-6)**: More conservative, preserves AI predictions, may leave small gaps
- **Higher threshold (10-15)**: Aggressive snapping, removes all gaps, may distort proportions

**Where to change**: `Interface/Houseweb/views.py` (Line 1097)

```python
threshold = 8  # Adjust this value
```

### Neighbor Alignment Threshold

**Purpose**: Controls gap/overlap tolerance between adjacent rooms

**Default**: `threshold + 6` = 14 pixels

**Where defined**: `Interface/align_fp/align_fp.m` (Line 10)

```matlab
[~, newBox, ~] = align_neighbor(newBox, rEdge, updated, threshold+6);
```

### Gap Filling Strategy

**Purpose**: How to fill remaining gaps after alignment

**Current strategy**: Expand living room (or nearest room if no living room)

**Where defined**: `Interface/align_fp/regularize_fp.m` (Lines 41-110)

**Alternatives** (not implemented):
- Create "corridor" rooms for gaps
- Split gaps between adjacent rooms proportionally
- Expand all adjacent rooms equally

---

## 7. Debugging and Visualization

### Enable MATLAB Visualization

Set `drawResult = 1` when calling `align_fp()`:

```python
newBox, order, rBoundary = eng.align_fp(
    boundary_matlab,
    rBox_matlab,
    rType_matlab,
    rEdge_matlab,
    fp_matlab,
    threshold,
    1,  # ← Set to 1 to enable visualization
    nargout=3
)
```

**Output**: MATLAB will save debug images showing:
- Original boxes (before refinement)
- After boundary alignment (Step 1)
- After neighbor alignment (Step 2)
- After regularization (Step 3)
- Final room polygons (Step 4)

### Python Fallback Debugging

Add print statements in `python_align_fp()` (Line 39-114):

```python
print(f"Before boundary alignment: {rBox}")
newBox = align_with_boundary_python(rBox, boundary, threshold)
print(f"After boundary alignment: {newBox}")
# ... etc
```

---

## 8. Known Issues and Limitations

### Issue 1: Gap Filling Heuristic

**Problem**: Always expands living room (or nearest room) to fill gaps

**Limitation**: May create unrealistically large living rooms

**Example**: If AI predicts small gaps between all rooms, living room grows to fill all of them

**Potential fix**: Split gaps proportionally between adjacent rooms

---

### Issue 2: No Learning from User Edits

**Problem**: Geometric refinement uses fixed rules, doesn't learn from user corrections

**Limitation**: If users consistently edit certain room types, system doesn't adapt

**Potential fix**: Log user edits and adjust thresholds or rules based on patterns

---

### Issue 3: MATLAB Dependency

**Problem**: Requires MATLAB installation with matlab.engine

**Limitation**: Adds software dependency, licensing cost

**Workaround**: Python fallback available, but with reduced functionality

**Potential fix**: Reimplement full MATLAB logic in Python (considerable effort)

---

### Issue 4: Fixed Threshold

**Problem**: Single threshold value for all room types and sizes

**Limitation**: May be too aggressive for small rooms, too conservative for large rooms

**Potential fix**: Scale threshold by room size or use per-room-type thresholds

---

## 9. Future Enhancements

### 9.1 Learnable Geometric Refinement

Replace fixed rules with learned refinement:
- Train a small neural network on "before/after" refinement pairs
- Learn optimal thresholds and gap-filling strategies
- Still ensure hard constraints (boundary compliance, no overlaps)

### 9.2 User Preference Learning

Track user edits and adapt refinement:
- Log which rooms users resize most often
- Adjust gap-filling priorities based on user behavior
- Personalize refinement per user

### 9.3 Multi-Stage Refinement

Add iterative refinement loop:
```
1. AI refinement (neural network)
2. Geometric refinement (boundary + neighbors)
3. AI re-refinement (fix any distortions from Step 2)
4. Final geometric validation (ensure validity)
```

### 9.4 Constraint Solver

Replace heuristic gap filling with optimization:
- Formulate as constraint satisfaction problem
- Minimize: distortion from AI predictions
- Subject to: no gaps, no overlaps, boundary compliance
- Solve with quadratic programming or similar

---

## 10. Quick Reference

### Enable/Disable Geometric Refinement

**Currently**: Always enabled when user saves floor plan

**To disable** (for debugging):
```python
# In Interface/Houseweb/views.py, Save_Editbox()
# Comment out the MATLAB call:
# newBox, order, rBoundary = eng.align_fp(...)

# Use AI predictions directly:
newBox = rBox_from_ai
order = list(range(len(newBox)))
rBoundary = boxes_to_polygons(newBox, boundary)
```

### Test Geometric Refinement Standalone

```matlab
% In MATLAB console
cd Interface/align_fp

% Load test data
boundary = load('test_boundary.mat');
rBox = load('test_boxes.mat');
rType = [0, 1, 3, 2];  % Living, Bedroom, Bathroom, Kitchen
rEdge = [0,1; 0,2; 0,3; 1,3];  % Adjacency graph

% Run refinement
[newBox, order, rBoundary] = align_fp(boundary, rBox, rType, rEdge, 'test', 8, 1);

% Check results
disp('Original boxes:');
disp(rBox);
disp('Refined boxes:');
disp(newBox);
```

### Common Troubleshooting

**MATLAB not starting**:
```python
# Error: matlab.engine not found
# Solution: Install MATLAB Engine for Python
cd /path/to/matlab/extern/engines/python
python setup.py install
```

**Refinement too aggressive**:
```python
# Solution: Reduce threshold
threshold = 4  # Instead of 8
```

**Gaps not being filled**:
```python
# Solution: Increase neighbor alignment threshold
# In align_fp.m, change:
# align_neighbor(newBox, rEdge, updated, threshold+6)
# to:
# align_neighbor(newBox, rEdge, updated, threshold+10)
```

---

## 11. File Locations Summary

| Component | File | Purpose |
|-----------|------|---------|
| Main refinement | `Interface/align_fp/align_fp.m` | Orchestrates 4-step process |
| Boundary alignment | `Interface/align_fp/align_with_boundary.m` | Step 1 |
| Neighbor alignment | `Interface/align_fp/align_neighbor.m` | Step 2 |
| Regularization | `Interface/align_fp/regularize_fp.m` | Step 3 (crop, order, fill) |
| Polygon generation | `Interface/align_fp/get_room_boundary.m` | Step 4 |
| Production call | `Interface/Houseweb/views.py` | Lines 1091-1103 |
| Python fallback | `Interface/Houseweb/views.py` | Lines 39-114 |

---

## 12. Related Documentation

- `IOU_AND_REFINEMENT_DOCUMENTATION.md` - AI-based refinement and IoU metrics
- `SESSION_2026-01-30_BUG_FIXES_AND_MODEL_DEPLOYMENT.md` - Session summary
- `MODEL_DEPLOYMENT_GUIDE.md` - Model deployment workflow

---

**End of Document**

# Quick Start: Edge Prediction Implementation

## Purpose

This document provides immediate context for implementing edge prediction in a new session. Read this FIRST, then refer to the detailed guides (beginner or advanced).

---

## Data Location

### ResPlan Dataset Files

**Base Path**: `C:\Users\hmbashir\source\Graph2plan\Interface\static\Data\`

**Files**:

```text
data_train_converted.pkl  - 75k floor plans with nodes, edges, boundaries
data_train_eNum.pkl       - Edge structure vectors [num_floorplans, edge_dim]
rNum_train.npy            - Room count vectors [num_floorplans, 14]
trainTF.pkl               - Turn Functions (boundary shapes)
```

### Already Loaded in Code

**Location**: `Houseweb/views.py` (lines 175-190)

**Global Variables**:

```python
train_data        # List of floor plan objects
trainNameList     # List of floor plan IDs (strings)
trainTF           # Turn function vectors for boundaries
train_data_eNum   # Edge structure vectors
train_data_rNum   # Room count vectors [75000, 14]
```

**How to access**: These are loaded once at Django startup via `getTrainData()` function.

---

## Data Structure Quick Reference

### Floor Plan Object Structure

Each floor plan in `train_data['data']` has:

```python
fp.data.box         # [N, 5] - Room bounding boxes [x1, y1, x2, y2, room_type]
fp.data.edge        # [E, 3] - Edges [src_node, edge_type, dst_node]
fp.data.boundary    # [M, 2] - Outer wall polygon points [x, y]
fp.get_rooms()      # Returns [N] array of room types
fp.get_triples()    # Returns edge array in different format
```

### Room Type Indices (Backend Mapping)

```python
0: LivingRoom
1: MasterRoom (includes all bedroom types)
2: Kitchen
3: Bathroom
4: DiningRoom
5: ChildRoom
6: StudyRoom
7: SecondRoom
8: GuestRoom
9: Balcony
10: Entrance
11: Storage
12: Wall-in
```

**Note**: Frontend uses different mapping (see SESSION_2025_12_23.md for details).

---

## Integration Point: AutoAdjustGraph

### Current Code (No Edge Prediction)

**File**: `Houseweb/views.py`
**Lines**: 1407-1409

```python
# Note: New nodes are added without edges
# Edges will be added later through edge prediction model
# (Previously we auto-connected to nearest neighbor, but that's been removed)
```

### Where to Add Edge Prediction

Replace the above comment with:

```python
# Add edges using edge prediction model
if len(rooms_to_add) > 0:
    # Load model (once at startup)
    if not hasattr(AutoAdjustGraph, 'edge_model'):
        AutoAdjustGraph.edge_model = load_edge_prediction_model()

    # Prepare input: existing graph + new nodes
    graph_data = prepare_graph_for_model(newNode, newEdge)

    # Predict edges for new nodes
    predicted_edges = AutoAdjustGraph.edge_model.predict(
        existing_nodes=newNode[:-len(rooms_to_add)],
        new_nodes=newNode[-len(rooms_to_add):],
        existing_edges=newEdge
    )

    # Add predicted edges
    for u, v in predicted_edges:
        newEdge.append([u, v])
```

**Context**: This runs after user clicks "Auto-Adjust" button, which adds missing rooms.

---

## Quick Data Check (Python Commands)

### Test Data Loading

Run these in a Python shell to verify data access:

```python
# 1. Load data
import pickle
import numpy as np

train_data = pickle.load(open(r'C:\Users\hmbashir\source\Graph2plan\Interface\static\Data\data_train_converted.pkl', 'rb'))

print(f"Number of floor plans: {len(train_data['data'])}")
print(f"Name list length: {len(train_data['nameList'])}")

# 2. Examine one floor plan
fp = train_data['data'][0]
print(f"\nFirst floor plan:")
print(f"  Rooms: {fp.get_rooms(tensor=False)}")
print(f"  Num rooms: {len(fp.data.box)}")
print(f"  Num edges: {len(fp.data.edge)}")
print(f"  Boundary points: {fp.data.boundary.shape}")

# 3. Check room counts
train_data_rNum = np.load(r'C:\Users\hmbashir\source\Graph2plan\Interface\static\Data\rNum_train.npy')
print(f"\nRoom count matrix shape: {train_data_rNum.shape}")  # Should be [75000, 14]
print(f"First floor plan room counts: {train_data_rNum[0]}")

# 4. Examine edges
edges = fp.get_triples(tensor=False)
print(f"\nEdges (first 5):")
for i, edge in enumerate(edges[:5]):
    src, edge_type, dst = edge
    print(f"  Edge {i}: Node {int(src)} <-> Node {int(dst)} (type: {int(edge_type)})")
```

**Expected Output**:

```text
Number of floor plans: ~75000
First floor plan: rooms array with ~8-12 values
Room count matrix shape: (75000, 14)
Edges: list of [src, edge_type, dst] tuples
```

---

## Model Files Location (When Ready)

### Where to Save Trained Model

**Recommended Path**: `C:\Users\hmbashir\source\Graph2plan\Interface\static\models\`

**Files to Save**:

```text
edge_predictor.pth          # Model weights
edge_predictor_config.json  # Hyperparameters (hidden_dim, num_layers, etc.)
```

### Loading in Django

Add to `views.py` at module level:

```python
# At top of views.py, after imports
edge_prediction_model = None

def load_edge_prediction_model():
    """Load edge prediction model (once at startup)"""
    global edge_prediction_model

    if edge_prediction_model is None:
        import torch
        from edge_prediction.models.spatial_gin import SpatialGIN  # Your model

        model = SpatialGIN(input_dim=17, hidden_dim=64, num_layers=3)
        model.load_state_dict(torch.load(
            r'C:\Users\hmbashir\source\Graph2plan\Interface\static\models\edge_predictor.pth',
            map_location='cpu'
        ))
        model.eval()
        edge_prediction_model = model
        print("✓ Edge prediction model loaded successfully")

    return edge_prediction_model
```

---

## Current State of AutoAdjustGraph

### What It Does Now

1. Parses user requirements (roomactarr, roomexaarr, roomnumarr)
2. Counts current rooms by type
3. Determines which rooms to add/remove
4. **Removes excess rooms** (if exact match required)
5. **Adds missing rooms** at random positions (30px from existing nodes)
6. **Does NOT add edges for new nodes**

### What Node Data Looks Like

**Format**: `[[index, roomname, x, y, scalesize], ...]`

**Example**:

```python
newNode = [
    [0, 'LivingRoom', 64.5, 110, 1],
    [1, 'Kitchen', 44.5, 167.5, 1],
    [2, 'MasterRoom', 176.5, 115, 1],
    [3, 'Bathroom', 110, 106, 1],  # New node added by AutoAdjustGraph
]
```

**Edge Format**: `[[u, v], ...]`

**Example**:

```python
newEdge = [
    [0, 1],  # LivingRoom <-> Kitchen
    [0, 2],  # LivingRoom <-> MasterRoom
    [1, 2],  # Kitchen <-> MasterRoom
    # Node 3 (Bathroom) has no edges yet!
]
```

---

## Workflow: Where Edge Prediction Fits

```text
USER FILTERS
  ↓
NumSearch/GraphSearch (returns top 20 matches)
  ↓
USER CLICKS RESULT
  ↓
Transfer (loads floor plan to left panel)
  ↓
USER CLICKS AUTO-ADJUST
  ↓
AutoAdjustGraph:
  1. Remove excess nodes ✓ (already working)
  2. Add missing nodes ✓ (already working)
  3. Predict edges for new nodes ← YOU IMPLEMENT THIS
  ↓
DISPLAY UPDATED GRAPH
  ↓
USER CLICKS LAYOUT (generates actual floor plan)
```

---

## Key Functions Reference

### In views.py

| Function | Lines | Purpose |
| ---------- | ------- | --------- |
| `getTrainData()` | 175-190 | Load ResPlan dataset into global variables |
| `TransGraph()` | 612-664 | Transfer selected floor plan to editing area |
| `AutoAdjustGraph()` | 1258-1420 | Add/remove nodes based on requirements |
| `compute_similarity_scores()` | 259-321 | Rank floor plans by similarity |
| `calculate_room_match_percentage()` | 324-409 | Calculate match % for display |

### In buttonEvent.js

| Function             | Lines   | Purpose                                  |
|----------------------|---------|------------------------------------------|
| `CreateLeftGraph()`  | 752-857 | Handle Transfer and Auto-Adjust buttons  |
| `ListBox()`          | 179-263 | Display search results with match %      |

---

## Development Environment

### Required Libraries (If Not Installed)

```bash
# PyTorch
pip install torch

# PyTorch Geometric (for GNN)
pip install torch-geometric

# Optional: NetworkX for graph utilities
pip install networkx

# Check installation
python -c "import torch; import torch_geometric; print('✓ Ready!')"
```

### Django Server

**Start**: Navigate to `C:\Users\hmbashir\source\Graph2plan\Interface\` and run:

```bash
python manage.py runserver
```

**Access**: <http://127.0.0.1:8000/>

---

## Testing Edge Prediction Integration

### Minimal Test Workflow

1. **Start Django server**
2. **Go to web UI**
3. **Filter**: 2 bedrooms, 1 bathroom, 2 balconies
4. **Search**: Click "Search" button
5. **Select result**: Click on a floor plan (shows on right)
6. **Transfer**: Click "Transfer" button (loads to left)
7. **Auto-Adjust**: Click "Auto-Adjust" button
8. **Check console**: Should see "Predicted edge: Bathroom <-> Kitchen" logs

### Expected Behavior Without Edge Prediction

- New nodes appear isolated (no connections)
- Console shows: "Added Bathroom at (145.3, 187.2), min distance from existing: 42.5px"

### Expected Behavior With Edge Prediction

- New nodes appear connected
- Console shows: "Predicted edge: 3 -> 1 (confidence: 0.87)"
- Graph is fully connected

---

## Troubleshooting

### Data Won't Load

**Error**: `FileNotFoundError: data_train_converted.pkl`

**Fix**: Check path is correct, use raw string: `r'C:\Users\...'`

### Django Can't Import Model

**Error**: `ModuleNotFoundError: No module named 'edge_prediction'`

**Fix**: Add to `sys.path` in views.py:

```python
import sys
sys.path.append(r'C:\Users\hmbashir\source\Graph2plan')
```

### Model Too Slow

**Issue**: Edge prediction takes >5 seconds

**Fix**:

1. Use CPU-optimized inference: `model.eval()` and `torch.no_grad()`
2. Batch predictions for multiple new nodes
3. Consider model quantization

### Edges Look Wrong

**Issue**: Predicted edges cross each other or don't make sense

**Fix**: Add post-processing (see EDGE_PREDICTION_GUIDE_ADVANCED.md, "Problem: Predicted edges cross")

---

## Next Steps

1. ✅ **Verify data access** - Run Python commands above
2. ✅ **Read implementation guide** - Beginner or Advanced based on experience
3. ✅ **Create model training script** - Start with GIN baseline
4. ✅ **Train on subset** - Use 1000 floor plans for debugging
5. ✅ **Integrate with AutoAdjustGraph** - Add code at lines 1407-1409
6. ✅ **Test end-to-end** - Follow testing workflow above

---

## Important Notes

- **Don't modify TransGraph**: It transfers floor plans as-is (no filtering)
- **Room type mapping**: Backend uses 0-12, frontend uses 0-13 (with 13 = total bedrooms)
- **New nodes have no edges**: This is intentional - edge prediction adds them
- **AutoAdjustGraph runs on frontend data**: Gets room requirements from `Num()` function in dealselect.js

---

## File Paths Summary

```text
Code:
  C:\Users\hmbashir\source\Graph2plan\Interface\Houseweb\views.py
  C:\Users\hmbashir\source\Graph2plan\Interface\static\js\buttonEvent.js
  C:\Users\hmbashir\source\Graph2plan\Interface\static\select\dealselect.js

Data:
  C:\Users\hmbashir\source\Graph2plan\Interface\static\Data\data_train_converted.pkl
  C:\Users\hmbashir\source\Graph2plan\Interface\static\Data\rNum_train.npy
  C:\Users\hmbashir\source\Graph2plan\Interface\static\Data\data_train_eNum.pkl

Models (when ready):
  C:\Users\hmbashir\source\Graph2plan\Interface\static\models\edge_predictor.pth

Documentation:
  C:\Users\hmbashir\source\Graph2plan\SessionContext\SESSION_2025_12_23.md
  C:\Users\hmbashir\source\Graph2plan\SessionContext\EDGE_PREDICTION_GUIDE_BEGINNER.md
  C:\Users\hmbashir\source\Graph2plan\SessionContext\EDGE_PREDICTION_GUIDE_ADVANCED.md
  C:\Users\hmbashir\source\Graph2plan\SessionContext\IMPLEMENTATION_NOTES.md
```

---

## Ready to Start?

1. Run the data check commands to verify everything loads
2. Choose your implementation guide (beginner or advanced)
3. Start with training a simple model on a subset
4. Come back when ready to integrate!

Good luck! 🚀

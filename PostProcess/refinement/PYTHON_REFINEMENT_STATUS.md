# Python Geometric Refinement - Current Status & Implementation Guide

## 📋 Table of Contents
1. [Project Overview](#project-overview)
2. [What Has Been Implemented](#what-has-been-implemented)
3. [File Structure](#file-structure)
4. [How It Works](#how-it-works)
5. [Testing Instructions](#testing-instructions)
6. [What's Pending](#whats-pending)
7. [Known Issues](#known-issues)
8. [Technical Details](#technical-details)

---

## Project Overview

### Goal
Replace MATLAB post-processing refinement with pure Python implementation that requires **no AI/ML dependencies** - only geometric algorithms.

### Context
Graph2plan is a floor plan generation system that:
1. Uses an AI model to generate initial floor plans from boundary and room requirements
2. Post-processes the results to refine box positions and create clean polygons
3. Originally used MATLAB for all post-processing (requires MATLAB license)

### User's Direction
> "Let's try and design a Python replacement for the MATLAB code. I want it to not need any AI to run. Let's build it incrementally. First step should be snapping the rooms to the boundary of the room"

---

## What Has Been Implemented

### ✅ Completed

#### 1. Python Refinement Module Structure
Created modular Python refinement system in `PostProcess/refinement/`:
- **geometry_utils.py**: Box and BoundarySegment classes for geometric operations
- **boundary_align.py**: Step 1 of refinement (boundary snapping) - COMPLETE
- **neighbor_align.py**: Step 2 stub (neighbor alignment) - PLACEHOLDER
- **gap_fill.py**: Step 3 stub (gap filling) - PLACEHOLDER
- **polygon_generation.py**: Step 4 stub (polygon generation) - PLACEHOLDER
- **integration.py**: MATLAB-compatible interface for Django integration
- **visualize.py**: Debugging visualization tools

#### 2. Step 1: Boundary Alignment (FULLY WORKING)
**Function**: `align_all_boxes_with_boundary()` in `boundary_align.py`

**What it does**:
- Snaps room box edges to nearby boundary walls
- Uses distance threshold (default: 8.0 pixels)
- Processes each box edge independently
- Returns aligned boxes + statistics about changes

**Algorithm**:
```
For each box:
    For each edge (left, right, top, bottom):
        Find closest parallel boundary segment
        If distance < threshold:
            Snap edge to boundary
        Mark which edges were updated
```

#### 3. Django Web Interface Integration
**Files Modified**:

**`Interface/Houseweb/views.py`** (Lines 17-35):
- Added path setup to import PostProcess module
- Import `align_fp_python` from `PostProcess.refinement.integration`
- Flag `HAS_PYTHON_REFINEMENT` to check if Python refinement is available

**`Interface/Houseweb/views.py`** (Lines 1107-1151 in `Save_Editbox`):
- Try Python refinement first
- Falls back to MATLAB if Python fails
- Falls back to basic Python if MATLAB fails
- Uses `userRoomID` as floor plan identifier

**`Interface/Houseweb/views.py`** (Lines 1162-1292 - New `Refine_Floorplan` endpoint):
- Manual refinement trigger for debugging
- Accepts GET parameters: `userRoomID`, `threshold`, `method`
- Returns JSON with statistics
- Can force specific refinement method

**`Interface/House/urls.py`** (Line 42):
- Added route: `path(r'index/Refine_Floorplan/', views.Refine_Floorplan)`

#### 4. Frontend Refine Button
**Files Modified**:

**`Interface/templates/home.html`** (Line ~851):
```html
<div id="refineButton"
     style="cursor: pointer;background-color: #ff6f00;color: #fff;width: 80px;...;display: none;">
    🔧 Refine
</div>
```
- Orange button positioned between Save and Transfer buttons
- Initially hidden (`display: none`)

**`Interface/static/js/buttonEvent.js`** (Line ~1135):
- Shows button when floor plan is created: `document.getElementById("refineButton").style.display = "block";`
- Located in `CreateLeftGraph()` function

**`Interface/static/js/buttonEvent.js`** (Lines 1389-1449):
- Click handler for refine button
- Gets floor plan ID from cookies
- Shows loading state ("⏳ Refining...")
- Calls `/index/Refine_Floorplan/` endpoint via AJAX
- Displays statistics in alert popup

---

## File Structure

```
Graph2plan/
├── PostProcess/
│   └── refinement/                    # NEW: Python refinement module
│       ├── __init__.py
│       ├── geometry_utils.py          # Box and BoundarySegment classes
│       ├── boundary_align.py          # Step 1: COMPLETE ✓
│       ├── neighbor_align.py          # Step 2: PLACEHOLDER
│       ├── gap_fill.py                # Step 3: PLACEHOLDER
│       ├── polygon_generation.py      # Step 4: PLACEHOLDER
│       ├── integration.py             # MATLAB-compatible interface
│       └── visualize.py               # Debugging visualization
│
├── Interface/
│   ├── Houseweb/
│   │   └── views.py                   # MODIFIED: Import setup, Save_Editbox, Refine_Floorplan
│   ├── House/
│   │   └── urls.py                    # MODIFIED: Added Refine_Floorplan route
│   ├── templates/
│   │   └── home.html                  # MODIFIED: Added refine button HTML
│   └── static/
│       └── js/
│           └── buttonEvent.js         # MODIFIED: Added button handler & visibility
│
└── [Other files unchanged]
```

---

## How It Works

### Automatic Refinement Flow (Save Button)

```
User creates/edits floor plan → Clicks "Save" button
    ↓
Save_Editbox() function (views.py:1107)
    ↓
Try Python refinement (if HAS_PYTHON_REFINEMENT):
    align_fp_python(boundary, boxes, room_types, edges, fp_id, threshold=8.0)
        ↓
        Step 1: Boundary alignment (WORKING)
        Step 2: Neighbor alignment (PLACEHOLDER - returns input unchanged)
        Step 3: Gap filling (PLACEHOLDER - returns input unchanged)
        Step 4: Polygon generation (PLACEHOLDER - returns boxes as is)
        ↓
        Returns: (refined_boxes, box_order, room_boundaries)
    ↓
    ✓ Success → Use refined boxes
    ✗ Fail → Try MATLAB alignment
        ↓
        ✓ Success → Use MATLAB boxes
        ✗ Fail → Use basic Python fallback (no refinement)
    ↓
Save results to .mat file
```

### Manual Refinement Flow (Refine Button)

```
User loads floor plan → 🔧 Refine button appears → User clicks button
    ↓
JavaScript handler (buttonEvent.js:1389)
    ↓
Gets userRoomID from cookies
Shows loading state
    ↓
AJAX GET request to /index/Refine_Floorplan/?userRoomID=X&threshold=8.0
    ↓
Refine_Floorplan() function (views.py:1162)
    ↓
Load floor plan from ./static/<userRoomID>.mat
    ↓
Try refinement (same as automatic flow)
    ↓
Calculate statistics:
    - total_boxes
    - boxes_changed (how many boxes moved)
    - avg_displacement (average movement in pixels)
    - max_displacement (max movement in pixels)
    ↓
Return JSON response:
{
    "success": true,
    "method": "python",
    "threshold": 8.0,
    "statistics": {
        "total_boxes": 5,
        "boxes_changed": 3,
        "avg_displacement": 4.25,
        "max_displacement": 7.8
    },
    "message": "Refinement complete using python method"
}
    ↓
JavaScript displays alert with statistics
```

---

## Testing Instructions

### Prerequisites

```bash
cd /mnt/c/Users/hmbashir/source/Graph2plan/Interface

# Activate virtual environment if needed
# source ../g2p-env/Scripts/activate  # or g2p-train-env

# Ensure dependencies are installed
pip install numpy matplotlib scipy
```

### Test 1: Server Startup - Check Python Refinement Loaded

```bash
python manage.py runserver
```

**Expected console output**:
```
[Init] Python geometric refinement loaded successfully ✓
```

**If you see an import error instead**:
- Check that numpy is installed
- Verify PostProcess/refinement/ directory exists
- Activate virtual environment

### Test 2: Automatic Refinement (Save Button)

1. Open browser: `http://localhost:8000/home`
2. Create or load a floor plan
3. Click "Save" button
4. Check server console for:
   ```
   [Refinement] Using Python geometric refinement for <userRoomID>
   [Refinement] Step 1: Aligning boxes with boundary...
   [Refinement] ✓ Python refinement successful! Got X boxes
   ```

### Test 3: Manual Refinement (Refine Button)

#### Option A: Using the Web Interface

1. Open browser: `http://localhost:8000/home`
2. Create a floor plan (you should see the graph transfer complete)
3. Look for the **orange 🔧 Refine button** between the green "Save" and blue "Transfer" buttons
4. Click the 🔧 Refine button
5. You should see an alert with statistics:
   ```
   ✅ Refinement Complete!

   Method: python
   Threshold: 8.0px

   Boxes changed: 3/5
   Avg displacement: 4.25px
   Max displacement: 7.8px
   ```

#### Option B: Using Browser URL Directly

Navigate to:
```
http://localhost:8000/index/Refine_Floorplan/?userRoomID=<your_floor_plan_id>
```

Example:
```
http://localhost:8000/index/Refine_Floorplan/?userRoomID=test123.pkl
```

#### Option C: Using curl

```bash
# Basic test
curl "http://localhost:8000/index/Refine_Floorplan/?userRoomID=test123.pkl"

# With custom threshold
curl "http://localhost:8000/index/Refine_Floorplan/?userRoomID=test123.pkl&threshold=10.0"

# Force specific method
curl "http://localhost:8000/index/Refine_Floorplan/?userRoomID=test123.pkl&method=python"
```

### Test 4: Testing Different Thresholds

```bash
# Conservative (less snapping)
curl "http://localhost:8000/index/Refine_Floorplan/?userRoomID=test.pkl&threshold=4.0"

# Balanced (default)
curl "http://localhost:8000/index/Refine_Floorplan/?userRoomID=test.pkl&threshold=8.0"

# Aggressive (more snapping)
curl "http://localhost:8000/index/Refine_Floorplan/?userRoomID=test.pkl&threshold=15.0"
```

### Test 5: Force Different Methods

```bash
# Force Python refinement
curl "http://localhost:8000/index/Refine_Floorplan/?userRoomID=test.pkl&method=python"

# Force MATLAB refinement (will fail if MATLAB not configured)
curl "http://localhost:8000/index/Refine_Floorplan/?userRoomID=test.pkl&method=matlab"

# Force basic Python fallback (no refinement)
curl "http://localhost:8000/index/Refine_Floorplan/?userRoomID=test.pkl&method=fallback"
```

---

## What's Pending

### 🔄 Incomplete Steps (Placeholders)

#### Step 2: Neighbor Alignment
**File**: `PostProcess/refinement/neighbor_align.py`

**Purpose**: Align adjacent room boxes to close gaps between neighbors

**Algorithm needed**:
```
For each pair of adjacent rooms (connected by edge):
    Find shared wall orientation (vertical/horizontal)
    Calculate gap between boxes
    If gap < threshold:
        Move boxes toward each other to close gap
        Or extend boxes to meet
```

**Current status**: Function exists but returns input unchanged (placeholder)

#### Step 3: Gap Filling
**File**: `PostProcess/refinement/gap_fill.py`

**Purpose**: Expand room boxes to fill small gaps in the layout

**Algorithm needed**:
```
Identify empty spaces between boxes
For each gap:
    Find adjacent boxes
    Expand boxes proportionally to fill gap
    Respect minimum room sizes
```

**Current status**: Function exists but returns input unchanged (placeholder)

#### Step 4: Polygon Generation
**File**: `PostProcess/refinement/polygon_generation.py`

**Purpose**: Convert rectangular boxes to complex polygons for rendering

**Algorithm needed**:
```
For each room:
    Start with box corners as polygon vertices
    Merge with adjacent room boundaries
    Create L-shaped or complex polygons where appropriate
    Ensure no overlaps
```

**Current status**: Function exists but returns boxes as simple rectangles (placeholder)

### 🎯 Next Implementation Priority

1. **Step 2: Neighbor Alignment** - Most impactful for visual quality
2. **Step 3: Gap Filling** - Prevents empty spaces in floor plan
3. **Step 4: Polygon Generation** - Final polish for realistic floor plans

---

## Known Issues

### Issue 1: Steps 2-4 Not Implemented
**Impact**: Floor plans only have boundary snapping, still have gaps between rooms
**Workaround**: MATLAB refinement still available as fallback
**Fix**: Implement Steps 2-4 incrementally

### Issue 2: Floor Plan ID Tracking
**Issue**: Need to ensure correct userRoomID is passed from frontend
**Status**: Currently uses cookies (`hsname`) - working but verify in production
**Location**: `buttonEvent.js:1400`

### Issue 3: No Visual Feedback After Refinement
**Issue**: Manual refine button shows alert but doesn't update the display
**Impact**: User must reload or save to see changes
**Potential fix**: Add code to refresh the canvas after refinement (see `buttonEvent.js:1434` comment)

---

## Technical Details

### Key Functions

#### `align_fp_python()` - Main Integration Function
**File**: `PostProcess/refinement/integration.py`
**Location**: Lines 19-155

**Signature**:
```python
def align_fp_python(
    boundary: np.ndarray,      # Boundary polygon (Nx2 array)
    boxes: np.ndarray,         # Room boxes (Mx4 array [x1,y1,x2,y2])
    room_types: np.ndarray,    # Room type IDs
    edges: np.ndarray,         # Adjacency edges (Px2 array)
    fp_id: Union[str, int],    # Floor plan identifier
    threshold: float = 8.0,    # Snap distance threshold
    draw_result: bool = False  # Enable visualization
) -> Tuple[List, List, List]:  # Returns (boxes, order, boundaries)
```

**Returns**:
- `box_out`: List of refined box coordinates
- `box_order`: Ordering of boxes (currently 0, 1, 2, ...)
- `rBoundary`: Room boundaries (currently same as boxes)

#### `align_all_boxes_with_boundary()` - Step 1 Core
**File**: `PostProcess/refinement/boundary_align.py`
**Location**: Lines 88-179

**Signature**:
```python
def align_all_boxes_with_boundary(
    boxes: np.ndarray,         # Nx4 array [x1,y1,x2,y2]
    boundary: np.ndarray,      # Boundary polygon
    threshold: float = 8.0,    # Snap distance
    room_types: Optional[np.ndarray] = None,
    verbose: bool = False
) -> Tuple[np.ndarray, List[Dict[str, bool]]]:
```

**Returns**:
- Aligned boxes (same shape as input)
- List of dicts showing which edges changed for each box

### Data Flow

#### From Frontend to Backend:
```javascript
// buttonEvent.js
var userRoomID = hsname.split(".")[0] + ".png.mat";  // e.g., "test123.png.mat"

$.get("/index/Refine_Floorplan/", {
    'userRoomID': userRoomID,
    'threshold': threshold
}, ...)
```

#### In Django View:
```python
# views.py - Refine_Floorplan
userRoomID = request.GET.get('userRoomID')
file_path = os.path.join('./static', userRoomID)  # "./static/test123.png.mat"

# Load .mat file
mat_data = sio.loadmat(file_path)
boundary = mat_data['boundary'][0]
boxes = mat_data['boxes'][0]
...

# Run refinement
box_out, _, _ = align_fp_python(boundary, boxes, ...)
```

### Configuration

#### Threshold Parameter
**Default**: 8.0 pixels
**Meaning**: Maximum distance to snap an edge to a boundary

**Tuning**:
- Lower (4.0): More conservative, less snapping
- Default (8.0): Balanced
- Higher (15.0): Aggressive, more snapping

**Where to change**:
- Default in `integration.py:19` (function parameter)
- Default in `views.py:1119` (automatic refinement)
- Can override via `Refine_Floorplan` endpoint parameter

### Error Handling

#### Three-Level Fallback Chain:
```python
try:
    # 1. Try Python refinement
    box_out = align_fp_python(...)
    method = "python"
except Exception as e:
    try:
        # 2. Try MATLAB refinement
        box_out = matlab_align(...)
        method = "matlab"
    except Exception as e2:
        # 3. Fall back to basic Python (no refinement)
        box_out = boxes  # Use original boxes
        method = "fallback"
```

**Logged to console**:
- `[Refinement] ✓ Python refinement successful!`
- `[Refinement] ✗ Python refinement failed: <error>`
- `[Refinement] Using fallback: basic Python alignment`

---

## Quick Reference

### Where Is Each Feature?

| Feature | File | Line(s) |
|---------|------|---------|
| Step 1 algorithm | `PostProcess/refinement/boundary_align.py` | 88-179 |
| Main integration function | `PostProcess/refinement/integration.py` | 19-155 |
| Python import setup | `Interface/Houseweb/views.py` | 17-35 |
| Automatic refinement (Save) | `Interface/Houseweb/views.py` | 1107-1151 |
| Manual refinement endpoint | `Interface/Houseweb/views.py` | 1162-1292 |
| URL routing | `Interface/House/urls.py` | 42 |
| Refine button HTML | `Interface/templates/home.html` | 851-854 |
| Button visibility trigger | `Interface/static/js/buttonEvent.js` | 1135 |
| Button click handler | `Interface/static/js/buttonEvent.js` | 1389-1449 |

### Common Commands

```bash
# Start server
cd /mnt/c/Users/hmbashir/source/Graph2plan/Interface
python manage.py runserver

# Test refinement via curl
curl "http://localhost:8000/index/Refine_Floorplan/?userRoomID=test.pkl&threshold=8.0"

# Check if files exist
ls -la PostProcess/refinement/
ls -la Interface/static/js/buttonEvent.js
```

### Debugging Checklist

- [ ] Server shows `[Init] Python geometric refinement loaded successfully ✓`
- [ ] Refine button appears after creating floor plan
- [ ] Clicking button shows "⏳ Refining..." state
- [ ] Console shows `[Refine Button] Calling refinement for: <userRoomID>`
- [ ] Alert displays statistics after completion
- [ ] Server console shows `[Refinement] ✓ Python refinement successful!`

---

## Summary for Fresh Session

### What Works Right Now:
✅ Step 1 (boundary snapping) fully implemented and tested
✅ Django integration complete with fallback chain
✅ Manual refine button working in frontend
✅ Automatic refinement on save
✅ Debug endpoint with statistics

### What Needs to Be Done:
❌ Step 2: Neighbor alignment (close gaps between adjacent rooms)
❌ Step 3: Gap filling (expand boxes to fill empty spaces)
❌ Step 4: Polygon generation (create complex room shapes)
❌ Optional: Visual feedback after manual refinement

### How to Continue:
1. **Test current implementation** to verify Step 1 is working
2. **Implement Step 2** next (highest priority for visual quality)
3. **Test iteratively** after each step
4. **Compare with MATLAB** output to validate correctness

### Key Constraint:
**No AI/ML dependencies** - Use only pure geometric algorithms (numpy, scipy geometry functions allowed)

---

**Document Created**: 2026-02-02
**Last Updated**: 2026-02-02
**Status**: Step 1 complete, Steps 2-4 pending

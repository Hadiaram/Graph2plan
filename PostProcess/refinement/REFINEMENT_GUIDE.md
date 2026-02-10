# Graph2Plan Refinement System Guide

This document explains the complete refinement pipeline for aligning room boxes with boundary walls and neighboring rooms.

## Overview

The refinement system is a pure Python replacement for the MATLAB post-processing pipeline. It uses geometric algorithms (no AI/ML) to refine floor plan layouts through multiple passes:

- **Pass 1**: Snap rooms to closest boundary wall (any direction)
- **Pass 2**: Snap rooms to orthogonal axis (perpendicular to Pass 1)
- **Pass 3**: Snap rooms to neighbors + fill boundary gaps + expand living room

## Architecture

```
integration.py          # Orchestrates the multi-pass refinement
    ├── boundary_align.py      # Wall snapping with ray casting
    ├── snap_rooms_to_neighbors()  # Room-to-room alignment
    ├── fill_small_boundary_gaps() # Corner pocket filling
    └── expand_living_room.py  # Living room expansion
```

---

## Pass 1: Initial Wall Snapping

**Goal**: Snap each room to its closest boundary wall.

### Algorithm

1. **Ray Casting** from box center in 4 directions (left, right, up, down)
   - Only considers walls directly visible from the center
   - Filters out occluded walls (solves interior wall problem)

2. **Distance Calculation** to visible walls
   - Uses overlap-aware distance (considers vertical/horizontal overlap)
   - Calculates Euclidean distance to wall segments

3. **Snap Closest Edge**
   - Finds the single closest edge overall
   - Shifts entire box (maintains size, no stretching)
   - Only snaps if distance ≤ threshold (default 50px)

4. **Detachment Prevention**
   - Checks if movement would detach from currently attached walls
   - Tolerance: 0.1px (walls closer than this are "attached")
   - Blocks moves that would pull rooms away from walls

### Key Functions

**`ray_cast_to_walls(box, h_segments, v_segments)`** (boundary_align.py:161-266)
- Casts rays from box center
- Returns dict of visible walls: `{'left': seg, 'right': seg, 'top': seg, 'bottom': seg}`

**`find_closest_segments(box, h_segments, v_segments)`** (boundary_align.py:269-424)
- Uses ray casting results
- Calculates distances only to visible walls
- Returns `EdgeAlignment` objects for each edge

**`align_box_with_boundary(box, boundary, threshold)`** (boundary_align.py:427-632)
- Finds closest edge within threshold
- Checks detachment prevention
- Shifts entire box to snap that edge
- Returns aligned box + dict of updated edges

### Pass 1 Output

Each box may snap to one wall (left, right, top, or bottom). The `updated_edges` dict tracks which edge moved:
```python
{'left': True, 'top': False, 'right': False, 'bottom': False}
```

---

## Pass 2: Orthogonal Snapping

**Goal**: Snap the perpendicular axis that wasn't snapped in Pass 1.

### Algorithm

1. **Determine Axis Exclusion** (FIXED in Feb 2026)
   - **OLD (BUGGY)**: Re-ran Pass 1 on original boxes to guess which axis was snapped
   - **NEW**: Checks which walls the **current boxes** are attached to
   - If attached to left/right wall → exclude horizontal (force vertical snap)
   - If attached to top/bottom wall → exclude vertical (force horizontal snap)
   - If attached to both → exclude horizontal (prioritize vertical)

2. **Snap with Axis Exclusion**
   - Uses same algorithm as Pass 1
   - But only considers edges on the non-excluded axis
   - Still applies detachment prevention

### Key Code

**Axis exclusion logic** (integration.py:74-107):
```python
if refinement_pass == 2:
    # Check which walls the CURRENT boxes are attached to
    for i in range(len(boxes)):
        box = Box.from_array(boxes[i])
        alignments = find_closest_segments(box, h_segments, v_segments)

        attached_left = alignments['left'].distance < 0.1
        attached_right = alignments['right'].distance < 0.1
        attached_top = alignments['top'].distance < 0.1
        attached_bottom = alignments['bottom'].distance < 0.1

        horizontal_attached = attached_left or attached_right
        vertical_attached = attached_top or attached_bottom

        if horizontal_attached:
            per_box_exclude.append('horizontal')  # Force vertical snap
        elif vertical_attached:
            per_box_exclude.append('vertical')  # Force horizontal snap
        else:
            per_box_exclude.append(None)  # Allow both
```

### Why This Fix Matters

**The Problem**: Rooms were moving inward in Pass 2 because the pre-scan determined axis exclusion using original box positions, but Pass 2 operated on Pass 1 refined positions (different locations).

**The Solution**: Check actual current attachments instead of simulating what *would* have happened.

---

## Pass 3: Room-to-Room Snapping + Gap Filling

**Goal**: Align rooms with neighbors and fill small empty spaces.

### Part 1: Room-to-Room Snapping

**Algorithm** (boundary_align.py:745-963):

1. **Skip living room** (type 0) - handled separately

2. **Check wall attachment**
   - Count walls this room is attached to (distance < 0.1px)
   - If attached to 2+ walls:
     - Check if any graph neighbors (non-living) have a gap > 5px
     - If yes: Allow movement to close gap
     - If no: Lock room in place (all neighbors touching)

3. **Find neighboring rooms** from edges/adjacency graph
   - Only considers rooms with graph connections
   - Won't snap to arbitrary nearby rooms

4. **Calculate distances** to each neighbor's edges
   - Checks both aligning edges (our right to their left)
   - And matching edges (our right to their right)
   - Requires overlap (vertical for horizontal snaps, horizontal for vertical)

5. **Snap to closest neighbor**
   - Within threshold (50px)
   - Only if won't detach from walls
   - Shifts entire room

**Key Code** (boundary_align.py:811-857):
```python
# Check how many walls this room is attached to
n_walls_attached = sum(1 for alignment in alignments.values()
                       if alignment.distance < 0.1)

if n_walls_attached >= 2:
    # Check if there are gaps to neighbors (excluding living room)
    has_gap_to_neighbor = False
    gap_threshold = 5.0

    for neighbor_idx in neighbors:
        if room_types[neighbor_idx] == 0:  # Skip living room
            continue

        neighbor_box = Box.from_array(snapped_boxes[neighbor_idx])

        # Calculate minimum distance between boxes
        # ... (gap calculation logic) ...

        if min_dist > gap_threshold:
            has_gap_to_neighbor = True
            break

    if not has_gap_to_neighbor:
        continue  # Lock in place
```

### Part 2: Fill Small Boundary Gaps

**Goal**: Expand rooms to fill small empty corners at the boundary.

**Algorithm** (boundary_align.py:966-1069):

1. **Skip living room** - handled by separate expansion

2. **Iteratively expand edges** (up to 4 iterations)
   - After each expansion, recalculate alignments
   - Find closest edge not yet expanded
   - Expand if distance ≤ 20px and no overlap with other rooms
   - Repeat until no more edges can be expanded

3. **Fills corner pockets**
   - Can expand multiple perpendicular edges (e.g., left + top)
   - Allows filling L-shaped corners

**Key Code** (boundary_align.py:1003-1049):
```python
max_iterations = 4  # One per direction
edges_expanded = []

for iteration in range(max_iterations):
    # Recalculate with current position
    box = Box.from_array(expanded_boxes[i])
    alignments = find_closest_segments(box, h_segments, v_segments)

    # Find closest unexpanded edge within threshold
    best_edge = None
    for edge_name, alignment in alignments.items():
        if edge_name in edges_expanded:
            continue
        if alignment.distance <= gap_threshold and alignment.distance > 0.1:
            if alignment.distance < best_distance:
                best_edge = edge_name
                # ...

    if best_edge is None:
        break  # No more edges to expand

    # Expand the edge (if no overlap)
    # ...
```

### Part 3: Living Room Expansion

**Algorithm** (expand_living_room.py:13-119):

1. Find living room (type 0)
2. Start by expanding to boundary min/max extents
3. Check each other room to see if it blocks expansion
4. Shrink expansion limits if rooms are in the way
5. Apply final expansion

**Note**: This is a simple rectangular expansion - can't fill L-shaped corners blocked by other rooms.

---

## Configuration

### Thresholds

**Wall snapping threshold**: 50px (buttonEvent.js:1416)
```javascript
var threshold = 50.0;
```

**Gap filling threshold**: 20px (integration.py)
```python
gap_filled_boxes = fill_small_boundary_gaps(
    neighbor_aligned_boxes,
    room_types,
    boundary,
    gap_threshold=20.0,  # Fill gaps up to 20 pixels
    verbose=True
)
```

**Room-to-room gap threshold**: 5px (boundary_align.py:815)
```python
gap_threshold = 5.0  # Consider rooms "touching" if closer than this
```

**Detachment tolerance**: 0.1px (boundary_align.py:502)
```python
would_detach, detached_edges = would_detach_from_walls(
    box, edge_name, alignment.snap_value, alignments,
    tolerance=0.1  # Walls closer than this are "attached"
)
```

### Pass Cycling

The refinement cycles through 3 passes:
- Click 1: Pass 1 → Pass 2
- Click 2: Pass 2 → Pass 3
- Click 3: Pass 3 → Pass 1 (cycle repeats)

Pass state is stored in the .mat file: `data['refinement_pass']`

---

## DXF Export Enhancement

### Problem

Original DXF export drew all room polygons as-is, resulting in overlapping geometry that didn't match the visual display.

### Solution (Feb 2026)

Added polygon clipping using Shapely library to export only visible geometry.

**Algorithm** (dxf_export.py:390-456):

1. **Get rendering order** from `order` field
   - Rooms early in order = background (drawn first)
   - Rooms late in order = foreground (drawn last, on top)

2. **For each room** (starting from background):
   - Clip to boundary polygon (intersection)
   - Subtract all rooms drawn later (difference)
   - Export only the resulting visible polygon

3. **Handle complex results**
   - If result is MultiPolygon, take largest piece
   - If result is empty, skip room (fully covered)

**Key Code** (dxf_export.py:390-456):
```python
def _clip_room_polygons(rBoundary, boundary, order):
    # Convert to Shapely polygons
    boundary_poly = Polygon(boundary[:, :2])
    room_polygons = [Polygon(rb[:, :2]) for rb in rBoundary]

    # Convert order to 0-indexed
    order_indices = [int(idx[0]) - 1 for idx in order]

    clipped_rooms = [None] * len(rBoundary)

    for idx, room_idx in enumerate(order_indices):
        visible_polygon = room_polygons[room_idx]

        # Clip to boundary
        visible_polygon = visible_polygon.intersection(boundary_poly)

        # Subtract rooms on top
        for later_idx in order_indices[idx + 1:]:
            visible_polygon = visible_polygon.difference(
                room_polygons[later_idx]
            )

        # Convert back to numpy array
        if not visible_polygon.is_empty:
            clipped_rooms[room_idx] = np.array(
                list(visible_polygon.exterior.coords)
            )

    return clipped_rooms
```

**Result**: DXF files now show clean, non-overlapping geometry that exactly matches the Graph2plan visual output.

---

## Usage

### From Django Interface

1. Load floor plan
2. Click "Refine" button repeatedly:
   - Click 1: Pass 1 (snap to closest wall)
   - Click 2: Pass 2 (snap orthogonal axis)
   - Click 3: Pass 3 (room-to-room + gap fill + living room)
3. Click "Export DXF" for clipped geometry

### From Python

```python
from PostProcess.refinement.integration import align_fp_python

# Pass 1
boxes_pass1, order, boundaries = align_fp_python(
    boundary=boundary,
    boxes=original_boxes,
    room_types=room_types,
    edges=edges,
    fp_id="test",
    threshold=50.0,
    refinement_pass=1,
    expand_living_room=False,
    original_boxes=original_boxes
)

# Pass 2
boxes_pass2, order, boundaries = align_fp_python(
    boundary=boundary,
    boxes=boxes_pass1,  # Use Pass 1 results
    room_types=room_types,
    edges=edges,
    fp_id="test",
    threshold=50.0,
    refinement_pass=2,
    expand_living_room=False,
    original_boxes=original_boxes
)

# Pass 3
boxes_final, order, boundaries = align_fp_python(
    boundary=boundary,
    boxes=boxes_pass2,  # Use Pass 2 results
    room_types=room_types,
    edges=edges,
    fp_id="test",
    threshold=50.0,
    refinement_pass=3,
    expand_living_room=True,  # Only expand in Pass 3
    original_boxes=original_boxes
)
```

---

## Known Limitations

1. **Corner filling doesn't always work** - Complex L-shaped corners may not be filled if the gap is formed by multiple walls that ray casting doesn't "see" properly

2. **Living room expansion is rectangular only** - Can't create L-shaped living rooms to fill corners blocked by other rooms

3. **Ray casting limitations**:
   - Only checks from box center (may miss close walls if center is far)
   - Short wall segments might not be detected if they don't span the center
   - Corners formed by two perpendicular walls are treated as separate walls

4. **Detachment tolerance is fixed** - 0.1px might be too strict for some layouts (e.g., rounding errors causing 0.5px gaps)

---

## Debugging

### Enable Verbose Logging

All main functions have `verbose=True` option:

```python
aligned_boxes, updated_edges = align_all_boxes_with_boundary(
    boxes, boundary, threshold=50.0, room_types=room_types,
    verbose=True  # Prints detailed info for each box
)
```

### Console Output

When verbose=True, you'll see:

```
========================================
BOUNDARY ALIGNMENT - 5 boxes
========================================

--- Box 0: LivingRoom (type=0) ---
  Closest segments:
    ✓ left: distance=3.45, snap_to=0.00
    ✗ top: distance=67.23, snap_to=0.00
    ✗ right: distance=245.67, snap_to=256.00
    ✗ bottom: distance=89.12, snap_to=256.00

  Overall closest edge: left = 3.45px
  ✓ Snapped LEFT edge: shifted entire box LEFT by 3.45px

  Result: Box(0.0, 67.23, 150.0, 180.45)
  Updated edges: ['left']
```

### Common Issues

**"No edges within threshold"** → Increase threshold or check if ray casting is finding the right walls

**"Would detach from walls"** → Box is trying to move away from attached wall, blocked by detachment prevention

**"Attached to 2 walls, locked in place"** → Room won't move in Pass 3 because all neighbors are touching

**"No neighbors in adjacency graph"** → Check edges array, room might not have graph connections

---

## File Structure

```
PostProcess/refinement/
├── integration.py              # Main orchestration (align_fp_python)
├── boundary_align.py           # Wall snapping + ray casting + room-to-room + gap filling
├── expand_living_room.py       # Living room expansion (user-created)
├── geometry_utils.py           # Box, BoundarySegment, utility functions
└── REFINEMENT_GUIDE.md        # This document

Interface/Houseweb/
├── views.py                    # Django endpoint (manual_refine_floorplan)
├── dxf_export.py              # DXF export with clipping
└── buttonEvent.js             # Frontend refine button

Interface/static/js/
└── buttonEvent.js             # Refine button configuration
```

---

## Recent Changes (Feb 2026)

### Pass 2 Fix: Axis Exclusion Based on Current Attachments
- **Problem**: Pass 2 was re-simulating Pass 1 on original boxes, causing wrong axis exclusion
- **Fix**: Now checks which walls the current boxes are actually attached to
- **Result**: Rooms no longer move inward incorrectly in Pass 2

### Pass 3: Room-to-Room Snapping
- **Feature**: Snaps rooms to graph-connected neighbors
- **Smart locking**: Rooms attached to 2+ walls only move if there's a gap to neighbors
- **Gap threshold**: 5px (rooms closer are considered "touching")

### Pass 3: Iterative Gap Filling
- **Feature**: Expands rooms to fill small boundary corners
- **Iterative**: Can expand multiple edges per box to fill L-shaped corners
- **Threshold**: 20px gap limit

### DXF Export: Polygon Clipping
- **Feature**: Uses Shapely to clip overlapping rooms
- **Result**: DXF shows only visible geometry, matching visual display
- **Respects order**: Background rooms have foreground rooms subtracted

---

## Dependencies

- **NumPy**: Array operations, distance calculations
- **SciPy**: Loading .mat files
- **Shapely** (optional): Polygon clipping for DXF export
  - If not installed, DXF export falls back to overlapping geometry
  - Install: `pip install shapely`
- **ezdxf** (optional): DXF file generation
  - Install: `pip install ezdxf`

---

## Future Improvements

1. **Smart corner detection**: Explicitly detect boundary corners and handle them specially
2. **Multi-step gap filling**: Run gap filling multiple times to fill complex corners
3. **L-shaped living room**: Allow living room to expand in complex shapes
4. **Adaptive thresholds**: Different thresholds for different room types or sizes
5. **Better ray casting**: Multiple rays per direction or rays from multiple points
6. **Node movement**: Move graph nodes to follow box movements (attempted but reverted)

---

Last updated: February 6, 2026

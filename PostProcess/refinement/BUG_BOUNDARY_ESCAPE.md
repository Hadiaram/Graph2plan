# Bug Report: Room Escaping Boundary After Pass 3

## What Is Happening

After running the three-pass geometric refinement, one room (visually identified as the
**Kitchen**, room type 2, pink room) extends **below the outer bottom wall** of an
L-shaped floor plan boundary. The room is visibly clipped through the boundary polygon.

Observed per pass:
- **No pass** — Kitchen starts inside the L-shaped boundary at bottom-right. Fine.
- **Pass 1** — Wall snapping. All rooms still inside boundary. Fine.
- **Pass 2** — Orthogonal snapping. All rooms still inside boundary. Fine.
- **Pass 3** — Room-to-room snapping + gap fill + living room expansion. Kitchen escapes
  below the bottom outer wall of the L-shape.

The escape is only visible after Pass 3 completes. Passes 1 and 2 are clean.

---

## System Overview

### Refinement Pipeline

The refinement runs as three separate passes. Each pass calls `align_fp_python()` in
`PostProcess/refinement/integration.py` with `refinement_pass=1/2/3`.

| Pass | What runs |
|------|-----------|
| 1 | `align_all_boxes_with_boundary` — snap each room to the nearest boundary wall |
| 2 | `align_all_boxes_with_boundary` — snap the orthogonal axis (force the other direction) |
| 3 | `resolve_room_overlaps` → `snap_rooms_to_neighbors` → `fill_small_boundary_gaps` → `fill_inter_room_gaps` → *(optionally)* `expand_living_room_to_boundary` → `fill_coverage_gaps` |

### Boundary Format

The boundary array has shape `(N, 4)` with columns `[x, y, orientation, isNew]`.

- `isNew == 0`: original polygon vertices — these define the real building outline.
- `isNew == 1`: interpolated helper vertices inserted by MATLAB preprocessing. These
  points lie along edges but are NOT polygon corners. Including them when constructing
  the polygon test shape distorts the polygon (adds extra co-linear points that can
  effectively "pull" the tested polygon's bounding extent).

`extract_boundary_segments()` already filters `isNew == 1` points with
`boundary[is_new == 0, :2]`. The containment check helpers must do the same.

### Containment Check Helpers

Two helpers in `boundary_align.py` are used to guard moves:

```python
def _point_in_poly(px, py, poly_xy):
    # Standard horizontal-ray ray-casting test

def _box_in_boundary(box, poly_xy, tol=0.5):
    # Returns True only if all 4 inset corners of the box pass _point_in_poly
```

A third helper filters the boundary array for use with both:

```python
def _get_boundary_pts(boundary):
    # Returns boundary[:, :2] filtered to isNew == 0 rows only
    # (same filter used by extract_boundary_segments)
```

---

## What Has Been Tried

### Fix 1 — `_get_boundary_pts` helper + update three call sites

**Problem identified:** Three functions (`snap_rooms_to_neighbors`,
`fill_inter_room_gaps`, `fill_small_boundary_gaps`) were computing the containment
polygon as `boundary[:, :2]` (all rows), which includes `isNew == 1` points and
distorts the polygon.

**Fix applied:** Added `_get_boundary_pts()` helper to `boundary_align.py` and updated
the three sites to use it. Also updated `fill_coverage_gaps` and
`expand_living_room_to_boundary` to use the filtered polygon.

**Result:** Did not fix the visual bug.

---

### Fix 2 — `_box_in_boundary` guard in `fill_coverage_gaps`

**Problem identified:** `fill_coverage_gaps` was expanding rooms to absorb Shapely-
detected uncovered gaps, but had no boundary containment check before applying the
expansion.

**Fix applied:** Added `_box_in_boundary(new_box, bnd_pts)` check before applying each
expansion in `fill_coverage_gaps`.

**Result:** Did not fix the visual bug.

---

### Fix 3 — `_get_boundary_pts_lr` in `expand_living_room_to_boundary`

**Problem identified:** `expand_living_room_to_boundary` was computing expansion limits
using `np.min/max(boundary[:, 0/1])` — the raw boundary including `isNew == 1` points
— potentially inflating the limits beyond the real polygon extents.

**Fix applied:** Added `_get_boundary_pts_lr()` helper in `expand_living_room.py` and
updated the min/max computation.

**Result:** Did not fix the visual bug.

---

### Fix 4 — `_box_in_boundary` guard in `resolve_room_overlaps`

**Problem identified:** `resolve_room_overlaps` runs as the first step of Pass 3. It
resolves overlaps between non-bathroom (type ≠ 3) and non-living (type ≠ 0) rooms.
The Kitchen (type 2) is **not excluded**. The function:

1. Identifies the room with fewer wall attachments as the one to move.
2. Translates it by `snap_delta` to push it edge-to-edge with the overlapping room.
3. Checks only `would_detach_from_walls` — there is **no polygon containment check**.
4. Applies the move unconditionally.

If the Kitchen overlaps a Bedroom and has fewer wall attachments, it is translated
downward (or in any direction) past the boundary with no guard.

**Fix applied:**
- Added `boundary_pts = _get_boundary_pts(boundary)` after the `extract_boundary_segments` call.
- Added `if not _box_in_boundary(new_move_box, boundary_pts): continue` before the
  `resolved_boxes[move_idx] = new_move_box.to_array()` apply line.

**Result:** User confirmed still not fixed. Fix was reverted along with all other
changes at user request.

---

## Current State of the Code

All four fixes above have been **reverted**. The code is back to its state before this
debug session. No changes are present in:

- `PostProcess/refinement/boundary_align.py`
- `PostProcess/refinement/integration.py`
- `PostProcess/refinement/expand_living_room.py`

The `_get_boundary_pts` and `_box_in_boundary` helpers remain in `boundary_align.py`
(they were added in a prior session and not reverted).

---

## What Is Still Unknown

The four fixes targeted every function in Pass 3 that moves a room. All those functions
now have (or had) `_box_in_boundary` guards. Yet the kitchen still escapes. This means
one of the following is true:

### Hypothesis A — `_box_in_boundary` returns the wrong answer

The ray-casting test may be returning `True` (inside) for a box that is visually
outside the L-shaped boundary. Possible reasons:

- The boundary polygon is not a simple closed polygon (self-intersections, duplicate
  points, or the last point does not connect back to the first).
- The boundary winding order causes the ray-casting even-odd rule to misclassify
  the concave notch region.
- `isNew == 1` points are still being included somewhere because the column index for
  `isNew` is not always column 3 (maybe column 2 in some data formats).

### Hypothesis B — A function moves the room before the check runs

The room may already be outside the boundary when Pass 3 starts (carried over from
Pass 2), and the Pass 3 functions are not correcting it — only blocking further escapes.
If the room enters Pass 3 already outside by a small amount, the check on the final
position may still pass (the tolerance is 0.5 px).

### Hypothesis C — `fill_coverage_gaps` uses Shapely's `union(...).bounds`

`fill_coverage_gaps` computes:
```python
combined_bounds = room_polys[room_idx].union(gap).bounds
new_box = Box(combined_bounds[0], combined_bounds[1],
              combined_bounds[2], combined_bounds[3])
```
The `bounds` of the union is the **axis-aligned bounding box** of the union polygon,
not the union polygon itself. For an L-shaped or irregular union, this bounding box
will extend into areas that are outside the boundary. The `_box_in_boundary` check
should catch this — but only if the check is working correctly (see Hypothesis A).

### Hypothesis D — `snap_rooms_to_neighbors` check uses wrong snap value

`snap_rooms_to_neighbors` computes `snap_value` from `new_box` coordinates, not from
the actual boundary snap target. If `new_box` is already outside the boundary, the
`would_detach_from_walls` call uses the wrong reference value, potentially failing to
block the move.

---

## Recommended Next Steps

1. **Verify `_box_in_boundary` for the specific floor plan.** Print the filtered
   boundary polygon points and the kitchen box coordinates at the start of Pass 3.
   Manually test whether `_point_in_poly` returns the expected result for a point
   known to be outside (e.g., one pixel below the bottom boundary edge).

2. **Check the boundary polygon at the point of reading.** Add a temporary print of
   `_get_boundary_pts(boundary)` at the top of `align_fp_python()` and verify the
   points form a proper closed L-shape with no extra vertices.

3. **Check the input boxes at the start of Pass 3.** Print all box coordinates entering
   `align_fp_python` with `refinement_pass=3`. If any box is already outside the
   boundary before any Pass 3 step runs, the escape is from Pass 1 or Pass 2.

4. **Check the `isNew` column index.** Confirm that column index 3 (0-indexed) is
   always the `isNew` flag for the actual data being processed. If the boundary array
   has fewer than 4 columns, `_get_boundary_pts` falls back to `boundary[:, :2]`
   which is correct — but if `isNew` is stored in column 2 instead of column 3, the
   filter is wrong.

---

## Relevant Files

| File | Role |
|------|------|
| `PostProcess/refinement/boundary_align.py` | All geometric refinement functions; `_box_in_boundary`, `_get_boundary_pts`, `resolve_room_overlaps`, `snap_rooms_to_neighbors`, `fill_small_boundary_gaps`, `fill_inter_room_gaps`, `fill_coverage_gaps` |
| `PostProcess/refinement/integration.py` | Orchestrates all passes; calls each function in order |
| `PostProcess/refinement/expand_living_room.py` | Living room expansion step |
| `Interface/Houseweb/views.py` | Calls `align_fp_python()` from the web interface; `Refine_Floorplan` endpoint drives pass selection |

# Bug Report: Snap Translation Pushes Room Outside Boundary

## What Is Happening

During Pass 1 or Pass 2 of wall alignment, a room is snapped toward an interior
boundary wall by translating the **entire box**. Because the translation moves all
four edges simultaneously, the edge **opposite** to the one being snapped can be
pushed past the outer boundary wall on the other side.

The specific case observed: the **Bathroom (type 3)** escapes the outer bottom wall
of an L-shaped floor plan after Pass 2.

---

## Observed Logs

### Pass 1 — Snap RIGHT edge

```
--- Box 4: Bathroom (type=3) ---

Aligning box: Box(112.25, 199.54, 154.83, 252.23)

  Closest segments:
    ✗ left:   distance=999999.00, snap_to=112.25
    ✓ right:  distance=12.27,     snap_to=167.11
    ✓ top:    distance=14.15,     snap_to=213.69
    ✗ bottom: distance=999999.00, snap_to=252.23

  Overall closest edge: right = 12.27px
  ✓ Snapped RIGHT edge: shifted entire box RIGHT by 12.27px

  Result: Box(124.52, 199.54, 167.11, 252.23)
```

The box shifts right. The bottom edge stays at y=252.23. This is fine.

### Pass 2 — Snap TOP edge (horizontal axis excluded)

```
--- Box 4: Bathroom (type=3) ---

Aligning box: Box(124.52, 199.54, 167.11, 252.23)

  ⊘ Skipping left  edge (axis 'horizontal' excluded)
  ⊘ Skipping right edge (axis 'horizontal' excluded)
    top: dist=14.15px, coverage=0.599, score=0.634

  Overall closest edge: top = 14.15px
  ✓ Snapped TOP edge: shifted entire box UP by -14.15px

  Result: Box(124.52, 213.69, 167.11, 266.38)
```

The TOP edge snaps to y=213.69 (the interior wall of the L-step). The entire box
is translated **down** by 14.15 px (y increases in image coordinates). The bottom
edge moves from y=252.23 → y=**266.38**, which is past the outer bottom boundary
wall.

---

## Root Cause

`align_box_with_boundary` always snaps by **rigid translation**: the chosen edge
is moved to the wall, and the opposite edge moves by the same delta.

```
Before:  y1=199.54  y2=252.23   (height = 52.69)
Snap TOP to 213.69:  delta = 213.69 - 199.54 = +14.15
After:   y1=213.69  y2=266.38   (height unchanged = 52.69, but bottom is now outside)
```

The ray cast (`find_closest_segments`) reported `distance=999999` for the bottom
edge in both passes. This means no boundary wall was found **below** the box center
in the downward direction. On an L-shaped boundary, a box positioned in the notch
area may cast a ray downward and exit the concave polygon without hitting a wall —
so the function has no reference coordinate for the outer bottom wall and cannot
guard against the bottom escaping.

### Why `distance=999999` for the bottom

`ray_cast_to_walls` casts a ray from the **box center** in each cardinal direction
and returns the nearest boundary segment whose span covers the ray origin. If the
box is positioned such that its center lies in a part of the L-shape where no
horizontal wall is directly below (e.g. the box spans a step in the boundary), the
downward ray exits the polygon without intersecting a segment. The result is the
sentinel value 999999, meaning "no visible wall found." With no wall reference,
there is nothing to prevent the snap translation from pushing the bottom edge
through the outer wall.

---

## The Fix (Implemented, Then Reverted)

### What Was Added

**`boundary_align.py`** — new helper `_box_fits_in_boundary(box, boundary)`:

- Builds a Shapely Polygon from the boundary using only `isNew==0` vertices.
- Applies a 0.5 px buffer to accept corners that land exactly on a wall.
- Returns `True` if all four corners of `box` are inside (or on) the polygon.
- Falls back to `True` if Shapely raises an exception.

**`align_box_with_boundary`** — check-before-apply pattern for all 4 snap cases:

```python
# Example: TOP snap
delta = aligned_box.y1 - closest_alignment.snap_value
proposed = Box(aligned_box.x1, closest_alignment.snap_value,
               aligned_box.x2, aligned_box.y2 - delta)
if not _box_fits_in_boundary(proposed, boundary):
    # skip — translation would push bottom edge outside
else:
    aligned_box.y1 = closest_alignment.snap_value
    aligned_box.y2 -= delta
    updated_edges['top'] = True
```

### Why It Was Reverted

The fix was reverted at user request before being tested in the browser. The
correctness of the fix (i.e., whether blocking the snap produces acceptable visual
results for the bathroom) was not verified.

---

## Current State of the Code

All changes have been **reverted**. The code is at its pre-session state:

- `boundary_align.py`: no `_box_fits_in_boundary` helper; `align_box_with_boundary`
  applies snaps unconditionally.
- `integration.py`: no diagnostic `_report_violations` calls.

---

## Open Questions

1. **Will blocking the snap leave a gap?** If the TOP snap is blocked because the
   translation would push the bottom outside, the bathroom will not snap to the
   interior wall. This may leave a visible gap between the room and the interior
   step wall. An alternative to blocking is to **resize** (move only the snapped
   edge, keep the opposite edge fixed) rather than translate.

2. **Is the bottom edge already slightly outside at the start of Pass 2?** The
   original bottom edge in Pass 1 is y=252.23. If the outer boundary bottom wall
   is at y=252.00, the box is already 0.23 px outside *before* any snapping. The
   translation in Pass 2 then makes it worse. This should be checked.

3. **Why does `ray_cast_to_walls` miss the outer bottom wall?** If the box center
   is inside the boundary, a downward ray should eventually hit the bottom edge.
   This may be a precision issue, or the wall segment may not span the x-range of
   the box center. Adding a print of the box center coordinates and the horizontal
   boundary segments would confirm this.

---

## Relevant Files

| File | Role |
|------|------|
| `PostProcess/refinement/boundary_align.py` | `align_box_with_boundary` (snap logic), `ray_cast_to_walls`, `find_closest_segments` |
| `PostProcess/refinement/integration.py` | Orchestrates all passes; calls `align_all_boxes_with_boundary` for passes 1 and 2 |

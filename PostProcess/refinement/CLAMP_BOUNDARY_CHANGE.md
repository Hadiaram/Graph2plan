# Change: Boundary Clamp Safety Net

## What Was Added

A final post-processing step that guarantees no room box extends past the building
boundary polygon after refinement. It is purely additive — all existing snapping
functions are unchanged.

---

## Why It Was Added

After three refinement passes, at least one room (the Kitchen, type 2) was visually
extending below the outer bottom wall of an L-shaped boundary. Four earlier attempts
to guard individual functions inside the pipeline all failed or were reverted. Rather
than continuing to modify the internals of those functions, the decision was made to
add a single safety net that runs **after everything else** and corrects any remaining
boundary violations before the result is returned.

---

## How It Works

### New function: `clamp_boxes_to_boundary` (`boundary_align.py`, end of file)

For each room box:

1. **Skip if clean.** Uses the existing `_box_in_boundary` helper to check all four
   inset corners against the filtered boundary polygon (isNew == 0 vertices only). If
   all four corners are inside, the box is untouched.

2. **Find visible walls.** Calls the existing `find_closest_segments`, which internally
   uses `ray_cast_to_walls` to cast rays from the box center in all four directions.
   For each direction, it returns the coordinate of the nearest visible boundary wall
   (`snap_value`). On a concave L-shaped boundary, this correctly finds the inner step
   wall rather than the global bounding box edge.

3. **Clamp protruding edges.** For each of the four edges, if the box edge overshoots
   the wall coordinate, it is pulled inward:

   | Edge   | Condition                   | Fix                                      |
   |--------|-----------------------------|------------------------------------------|
   | Left   | `box.x1 < left_wall.x`      | `new_x1 = left_wall.snap_value`          |
   | Right  | `box.x2 > right_wall.x`     | `new_x2 = right_wall.snap_value`         |
   | Top    | `box.y1 < top_wall.y`       | `new_y1 = top_wall.snap_value`           |
   | Bottom | `box.y2 > bottom_wall.y`    | `new_y2 = bottom_wall.snap_value`        |

   A wall is only used as a clamp target if a real boundary segment was found in that
   direction (`alignment.segment is not None`). If no wall is visible in a direction,
   that edge is left alone.

4. **Validate box.** After clamping, checks that edges haven't crossed
   (`x2 > x1`, `y2 > y1`) and applies a 1px minimum size if they have.

The function logs every box it changes (index, room type, which edges, old → new
coordinates) when `verbose=True`.

### Call site: `align_fp_python` (`integration.py`, Step 6)

```python
# STEP 6: BOUNDARY CLAMP (safety net — runs every pass)
final_boxes = clamp_boxes_to_boundary(
    final_boxes, boundary, room_types=room_types, verbose=True
)
```

This call sits **after** all other steps (Steps 1–5, including `fill_coverage_gaps`)
and **before** polygon generation. It runs on every refinement pass (1, 2, and 3).
On passes where all rooms are already inside the boundary, it is a no-op.

---

## Key Properties

- **Non-invasive.** No existing function was modified. This is a new function added
  at the end of `boundary_align.py` and called once at the end of `align_fp_python`.

- **Concave-polygon aware.** Uses `ray_cast_to_walls` (existing code) rather than
  simple bounding-box clamping (`clip_box_to_boundary` in `geometry_utils.py`). The
  ray cast from the box center finds the correct wall even for L-shaped or stepped
  boundaries.

- **Only shrinks.** The clamp conditions are strictly `box.x1 < wall` /
  `box.x2 > wall` etc. A box edge that is already inside the wall is never moved
  outward.

- **isNew filtering.** Uses `_get_boundary_pts(boundary)` for the `_box_in_boundary`
  pre-check, which filters out `isNew == 1` interpolated helper vertices, matching
  the same filtering used by `extract_boundary_segments`.

---

## Files Changed

| File | Change |
|------|--------|
| `PostProcess/refinement/boundary_align.py` | Added `clamp_boxes_to_boundary()` before the `__main__` block |
| `PostProcess/refinement/integration.py` | Added `clamp_boxes_to_boundary` to import; added Step 6 call after Step 5 |

---

## What Was Not Changed

- `resolve_room_overlaps` — unchanged
- `snap_rooms_to_neighbors` — unchanged
- `fill_small_boundary_gaps` — unchanged
- `fill_inter_room_gaps` — unchanged
- `fill_coverage_gaps` — unchanged
- `expand_living_room_to_boundary` — unchanged
- All Pass 1 / Pass 2 logic — unchanged

---

## Limitations

- The ray cast is from the **box center**. If a box has moved so far that its center
  is outside the boundary, the ray cast may not find the correct wall. The user's
  constraint ("the room is already inside the boundary, at least in part") makes this
  unlikely in practice, but it is a theoretical edge case.

- Only handles **rectilinear boundaries** (horizontal and vertical walls). Diagonal
  boundary segments are skipped by `extract_boundary_segments` and therefore not
  used as clamp targets.

# Session Notes — Graph2plan Debugging & Interface Work

**Date:** February 2026
**Scope:** Boundary alignment debugging, codebase redundancy review, boundary log button in Interface

---

## 1. Snap Translation Bug (Bathroom Escaping Boundary)

### Background

Read `PostProcess/refinement/BUG_SNAP_TRANSLATION_ESCAPE.md`. The bug was already documented from a prior session. Summary:

- During Pass 1 or Pass 2 of wall alignment, `align_box_with_boundary` snaps a box by **rigid translation** — the chosen edge moves to the wall and the opposite edge shifts by the same delta.
- The Bathroom (type=3) in an L-shaped floor plan was already partially outside the boundary before any snapping began (`1641px²` outside).
- Pass 2 snapped the TOP edge to an interior wall, shifting the entire box downward and increasing the escaped area to `2243px²` — making the situation worse.
- The root cause: `ray_cast_to_walls` returns `distance=999999` for the bottom edge when the box sits in the notch of an L-shape (no wall is directly below the box center), so no guard exists to prevent the opposite edge from escaping.

### Diagnostic Logging Added (then reverted)

A `_log_bathroom_diagnostic` helper and a `_compute_escape_area` helper were added to `boundary_align.py` to track:
- All four corner positions before and after each snap
- Movement direction and magnitude (left/right/up/down)
- Box dimensions
- Shapely-based boundary escape area (px²) before and after

These were gated on `room_types[i] == 3` and always printed regardless of the `verbose` flag.

A **guard** was also added to `align_box_with_boundary` to block any snap that would increase the escaped area. The selection loop was also changed to prefer the snap candidate that **minimises escape area** (rather than purely minimising distance or maximising coverage heuristic).

### Reverted

All changes to `boundary_align.py` were reverted at user request before testing. The file is back to its pre-session state:
- No `_compute_escape_area`
- No `_log_bathroom_diagnostic`
- Original coverage/distance selection loop
- Original direct snap application (no guard)

### Open Issues

1. The bathroom starts partially outside the boundary before any snapping — this is the model's initial placement, not a snapping error. The snap then makes it worse.
2. The alternative fix (resize only the snapped edge, keep opposite edge fixed) was discussed but not implemented.
3. The `ray_cast_to_walls` miss for the bottom edge in L-shaped plans is unresolved — likely because the box center is in a notch where no horizontal wall segment spans the center's X coordinate.

---

## 2. Codebase Redundancy Review

A full scan of the repository identified the following redundancies (no changes made):

### High Priority
| Issue | Location |
|---|---|
| `get_vocab()` defined 3× with **diverging vocabularies** | `Network/model/utils.py`, `Interface/model/utils.py`, `PostProcess/g2p/utils.py` |
| `room_type_names` dict copy-pasted **8 times** | `PostProcess/refinement/boundary_align.py` |

### Medium Priority
| Issue | Location |
|---|---|
| 6 geometry/utility functions duplicated across 3 `utils.py` files | `*/utils.py` × 3 |
| `label2index()` / `index2label()` are identity functions (return input unchanged) | `Interface/model/utils.py`, `PostProcess/g2p/utils.py` |
| Two "Step 4" labels + dead TODO stubs | `PostProcess/refinement/integration.py` |

### Low Priority
| Issue | Location |
|---|---|
| `room_label` tuple defined twice identically | `Interface/model/utils.py`, `PostProcess/g2p/utils.py` |
| Commented-out old function implementations | `Interface/model/utils.py` ~line 185, `PostProcess/g2p/utils.py` ~line 180 |
| Dead alias `room_label_original = room_label`, never used | `Interface/model/utils.py` line 25 |

### Critical Note on `get_vocab()`
`PostProcess/g2p/utils.py` returns a **15-type vocabulary** while `Network/` and `Interface/` both return a **5-type vocabulary**. This is not just redundancy — it is a potential type mismatch source if the two pipelines are ever connected without careful conversion.

---

## 3. Boundary Image Extraction (Test Set)

Identified the relevant scripts in `DataPreparation/`:

| Script | Purpose |
|---|---|
| `reextract_test_boundaries.py` | Re-extracts boundary coordinates from `ResPlan.pkl` into `data_test_converted.pkl` |
| `generate_floorplan_images.py` | Renders boundary outlines as PNG files → `Interface/static/Data/Img/` |
| `check_boundary_integrity.py` | Validates room boundaries don't extend beyond outer boundary |
| `check_test_boundaries.py` | Quick check for the "all boundaries identical" bug |

To generate boundary images for the test set (boundary outline only, room rendering is commented out in the script):

```bash
cd DataPreparation
python generate_floorplan_images.py
```

Reads from `../Interface/static/Data/data_test_converted.pkl`, saves PNGs to `../Interface/static/Data/Img/`.

---

## 4. Interface — Log Boundary Button

A new diagnostic button was added to the Interface. It is **separate from the Refine and Export DXF buttons** and does not affect any floor plan data.

### Files Changed

| File | Change |
|---|---|
| `Interface/House/urls.py` | Added route `index/Log_Boundaries/` |
| `Interface/Houseweb/views.py` | Added `Log_Boundaries(request)` view function |
| `Interface/templates/home.html` | Added `🔍 Log Boundary` button (teal, `margin-left: 680px`) |
| `Interface/static/js/buttonEvent.js` | Show logic + click handler added |

### Button Behaviour

- Appears alongside Refine and Export DXF once a floor plan is loaded (after Transfer)
- Calls `GET /index/Log_Boundaries/?userRoomID=<id>`
- On success: prints full detail to the **server console** and **browser console** (F12), shows a brief summary alert
- Does **not** modify any floor plan data

### What the Log Shows

**Server console and browser console both output:**

1. **Boundary points** — index, x, y, direction code, direction name, isNew flag
2. **Wall segments** — each wall as `(x1,y1) → (x2,y2)` with length and direction
3. **Rooms** — index, type name, bounding box coords, center, nearest boundary point index + distance, whether inside boundary extents, and (if Shapely available) escape area in px²

**Browser alert summary** includes:
- Boundary point count and extents
- Warning count for rooms outside boundary extents
- Warning count for rooms with non-zero escape area

### Coordinate System Note

The boundary data uses **image coordinates: Y increases downward** (0,0 = top-left). The interface SVG renders in the same convention. However, the floor plan visual appears **Y-flipped** relative to spatial/mathematical convention (Y increases upward).

This means:
- "Snapped UP" in the refinement logs → box moves to **higher Y values** in data → appears to move **down** visually
- "Snapped DOWN" in the refinement logs → appears to move **up** visually
- Wall direction `up ↑` (code 1) in the boundary log → runs **downward** on screen

**Root cause of the flip:** The ResPlan dataset stores boundary coordinates via Shapely, which follows GIS/CAD convention (Y up). The SVG renders them with Y down (screen convention) without correction. `generate_floorplan_images.py` compensates with an explicit `ax.invert_yaxis()` call; the interface SVG never received the equivalent correction.

---

## 5. Relevant File Paths

```
Interface/
  House/urls.py                          ← URL routing
  Houseweb/views.py                      ← All view functions incl. Log_Boundaries
  templates/home.html                    ← Button definitions
  static/js/buttonEvent.js              ← Button handlers

PostProcess/refinement/
  boundary_align.py                      ← align_box_with_boundary, ray_cast_to_walls
  integration.py                         ← align_fp_python, orchestrates all passes
  BUG_SNAP_TRANSLATION_ESCAPE.md         ← Snap bug documentation

DataPreparation/
  reextract_test_boundaries.py           ← Re-extract boundaries into pkl
  generate_floorplan_images.py           ← Render boundary PNGs
  check_boundary_integrity.py            ← Validation
```

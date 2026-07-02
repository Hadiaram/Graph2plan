"""
Hotel logic layer — structural grid allocator for the mixed logic+diffusion approach.

Takes a building boundary (outer floor envelope) and a list of room types from the
user's bubble diagram, and returns per-room slot bounding boxes that respect the
hotel's double-loaded corridor structure.

The diffusion model then generates actual polygon geometry within those slots
rather than starting from pure noise.

Defaults are inferred from the 4star_L1_1.json training floor:
  bay_width   ≈ 17 px  (structural bay for a standard guestroom)
  wing_depth  ≈ 37 px  (depth of each guestroom wing)
  corridor    ≈ 46 %   of total floor height
  core        ≈ 26 %   of total floor width (lifts, stairs, service rooms)
"""

import numpy as np

# ── Defaults from training data ────────────────────────────────────────────────
DEFAULT_BAY_WIDTH_PX   = 17.0
DEFAULT_WING_DEPTH_PX  = 37.0
DEFAULT_CORRIDOR_FRAC  = 0.46   # fraction of total floor height
DEFAULT_CORE_FRAC      = 0.26   # fraction of total floor width

# Room types that live in the corridor/service core (not in the guestroom wings)
CORE_TYPES  = frozenset({3, 4, 5, 6, 7, 8})
GUEST_TYPES = frozenset({0, 1, 2})


# ── Bay width inference ────────────────────────────────────────────────────────

def infer_bay_width(room_polygons):
    """
    Estimate the structural bay width by clustering the left-edge (min-x) positions
    of guestroom polygons.  Returns the median inter-column gap in pixels.

    room_polygons: list of Px2 numpy arrays (one per room, already filtered to
                   guest types or passed in full — non-guest rooms are rare enough
                   that they don't significantly skew the distribution).
    """
    if not room_polygons:
        return DEFAULT_BAY_WIDTH_PX

    min_xs = sorted({round(float(np.array(p)[:, 0].min()), 1) for p in room_polygons})
    if len(min_xs) < 2:
        return DEFAULT_BAY_WIDTH_PX

    diffs = np.diff(min_xs)
    # Keep only gaps in the plausible bay-width range (10–25 px); the large gap
    # across the service core (~50 px) is an outlier and is excluded here.
    bay_diffs = diffs[(diffs >= 10) & (diffs <= 25)]
    return float(np.median(bay_diffs)) if len(bay_diffs) > 0 else DEFAULT_BAY_WIDTH_PX


# ── Slot allocator ─────────────────────────────────────────────────────────────

def allocate_hotel_slots(boundary, g2p_types,
                         bay_width_px=DEFAULT_BAY_WIDTH_PX,
                         corridor_fraction=DEFAULT_CORRIDOR_FRAC,
                         core_fraction=DEFAULT_CORE_FRAC,
                         wing_depth_px=DEFAULT_WING_DEPTH_PX):
    """
    Allocate a slot (bounding box) for every room in g2p_types.

    boundary    : Nx4 or Nx2 array — outer floor boundary polygon.
                  If the boundary is the inner corridor polygon (very narrow
                  height compared to width), the function extends it by
                  wing_depth_px on each side automatically.
    g2p_types   : list[int] — room type per room, in bubble-diagram order.
    bay_width_px: structural bay width in canvas pixels.

    Returns list[dict] with keys {type, x1, y1, x2, y2, zone} in the same
    order as g2p_types, or None if the boundary is too small to accommodate
    all rooms (caller should fall back to unconstrained generation).
    """
    b = np.array(boundary, dtype=float)
    pts = b[:, :2]

    x_min, x_max = float(pts[:, 0].min()), float(pts[:, 0].max())
    y_min, y_max = float(pts[:, 1].min()), float(pts[:, 1].max())
    floor_w = x_max - x_min
    floor_h = y_max - y_min

    # ── Distinguish outer boundary from inner corridor boundary ────────────────
    # A narrow strip (h < 40 % of w) is treated as the corridor polygon;
    # extend it by wing_depth_px above and below to get the full floor zone.
    if floor_h < floor_w * 0.40:
        corridor_y_lo = y_min
        corridor_y_hi = y_max
        outer_y_min   = y_min - wing_depth_px
        outer_y_max   = y_max + wing_depth_px
    else:
        y_mid = (y_min + y_max) / 2.0
        half  = floor_h * corridor_fraction / 2.0
        corridor_y_lo = y_mid - half
        corridor_y_hi = y_mid + half
        outer_y_min   = y_min
        outer_y_max   = y_max

    top_y1 = outer_y_min
    top_y2 = corridor_y_lo
    bot_y1 = corridor_y_hi
    bot_y2 = outer_y_max

    # ── Count available guest bays ─────────────────────────────────────────────
    # Wing bays span the FULL floor width — the service core only occupies the
    # corridor zone vertically and does not create a horizontal gap in the wings.
    total_bays = max(1, int(floor_w / bay_width_px))

    # ── Separate rooms by category ─────────────────────────────────────────────
    core_rooms  = [(i, t) for i, t in enumerate(g2p_types) if t in CORE_TYPES]
    guest_rooms = [(i, t) for i, t in enumerate(g2p_types) if t in GUEST_TYPES]

    total_guest_slots = total_bays * 2   # top + bottom per bay
    if len(guest_rooms) > total_guest_slots:
        return None   # boundary too small — caller should fall back

    slots = [None] * len(g2p_types)

    # ── Service / corridor room slots ──────────────────────────────────────────
    n_core = len(core_rooms)
    if n_core > 0:
        # Subdivide the full corridor x-span into n_core equal columns.
        slot_w = floor_w / n_core
        for k, (room_idx, rtype) in enumerate(core_rooms):
            slots[room_idx] = {
                'type': rtype,
                'x1': x_min + k * slot_w,       'y1': corridor_y_lo,
                'x2': x_min + (k + 1) * slot_w, 'y2': corridor_y_hi,
                'zone': 'core',
            }

    # ── Guest room slots (top+bottom interleaved, left to right) ──────────────
    bay_slots = []
    for b_idx in range(total_bays):
        bx1 = x_min + b_idx * bay_width_px
        bx2 = bx1 + bay_width_px
        bay_slots.append(('top', bx1, top_y1, bx2, top_y2))
        bay_slots.append(('bot', bx1, bot_y1, bx2, bot_y2))

    for slot_idx, (room_idx, rtype) in enumerate(guest_rooms):
        zone, bx1, by1, bx2, by2 = bay_slots[slot_idx]
        slots[room_idx] = {
            'type': rtype,
            'x1': bx1, 'y1': by1,
            'x2': bx2, 'y2': by2,
            'zone': zone,
        }

    if any(s is None for s in slots):
        return None

    return slots


# ── Slot → polygon conversion ──────────────────────────────────────────────────

def slots_to_polygons(slots):
    """
    Convert slot dicts to 4-corner polygon lists in [0, 255] canvas space.
    Returns list[list[[x,y]]] in the same order as slots.
    """
    return [
        [
            [s['x1'], s['y1']],
            [s['x2'], s['y1']],
            [s['x2'], s['y2']],
            [s['x1'], s['y2']],
        ]
        for s in slots
    ]

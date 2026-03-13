"""
postprocess.py — Heuristic post-processing passes for floor plan layouts.

Separate from solver.py (the CP-SAT optimizer) so each pass is easy to
isolate, debug, and extend independently.
"""

import numpy as np

try:
    from shapely.geometry import Polygon, LineString, MultiLineString
    from shapely.ops import unary_union
    _SHAPELY_OK = True
except ImportError:
    _SHAPELY_OK = False

# ---------------------------------------------------------------------------
# Room metadata
# ---------------------------------------------------------------------------

_ROOM_NAMES = {
    0: 'LivingRoom', 1: 'MasterRoom', 2: 'Kitchen', 3: 'Bathroom',
    4: 'DiningRoom', 5: 'ChildRoom',  6: 'StudyRoom', 7: 'SecondRoom',
    8: 'GuestRoom',  9: 'Balcony',   10: 'Entrance', 11: 'Storage',
    12: 'Wall-in',  13: 'External',  14: 'ExteriorWall', 15: 'FrontDoor',
    16: 'InteriorWall', 17: 'InteriorDoor',
}

# Minimum room dimensions in pixels (w, h).
MIN_ROOM_SIZES = {
    0:  (50, 40),   # LivingRoom
    1:  (40, 35),   # MasterRoom
    2:  (30, 25),   # Kitchen
    3:  (20, 20),   # Bathroom
    4:  (25, 25),   # DiningRoom
    5:  (35, 30),   # ChildRoom
    6:  (30, 25),   # StudyRoom
    7:  (35, 30),   # SecondRoom
    8:  (35, 30),   # GuestRoom
    9:  (20, 20),   # Balcony
    10: (20, 20),   # Entrance
    11: (15, 15),   # Storage
}
_DEFAULT_MIN = (15, 15)

MIN_DOOR_PX = 30

# Types that represent "open / structural" elements and should not be treated
# as room blockers when checking connectivity to the LivingRoom.
_OPEN_TYPES = {0, 13, 14, 15}   # LivingRoom, External, ExteriorWall, FrontDoor


def _room_label(i, types):
    t = int(types[i]) if i < len(types) else -1
    return f"room[{i}] {_ROOM_NAMES.get(t, f'type{t}')}"


def _min_wh(i, types):
    t = int(types[i]) if i < len(types) else -1
    return MIN_ROOM_SIZES.get(t, _DEFAULT_MIN)


def _current_shared(boxes, u, v, tol=3.0):
    """Return the current shared wall length between two rooms (0 if not touching)."""
    ax0, ay0, ax1, ay1 = boxes[u]
    bx0, by0, bx1, by1 = boxes[v]
    if abs(ax1 - bx0) <= tol or abs(bx1 - ax0) <= tol:
        return max(0.0, min(ay1, by1) - max(ay0, by0))
    if abs(ay1 - by0) <= tol or abs(by1 - ay0) <= tol:
        return max(0.0, min(ax1, bx1) - max(ax0, bx0))
    return 0.0


def _best_direction(boxes, u, v):
    """
    Try all four possible connection directions and return the best one —
    the direction where the gap is smallest and non-negative (rooms are on
    the correct side of each other and closest to touching).
    """
    ux0, uy0, ux1, uy1 = boxes[u]
    vx0, vy0, vx1, vy1 = boxes[v]

    y_ov = max(0.0, min(uy1, vy1) - max(uy0, vy0))
    x_ov = max(0.0, min(ux1, vx1) - max(ux0, vx0))

    options = [
        ('lr', 2, 0, vx0 - ux1, y_ov),   # u left of v
        ('lr', 0, 2, ux0 - vx1, y_ov),   # u right of v
        ('tb', 3, 1, vy0 - uy1, x_ov),   # u above v
        ('tb', 1, 3, uy0 - vy1, x_ov),   # u below v
    ]

    non_neg = [o for o in options if o[3] >= 0]
    if non_neg:
        return min(non_neg, key=lambda o: (o[3], -o[4]))
    else:
        return max(options, key=lambda o: o[3])


# ---------------------------------------------------------------------------
# LivingRoom connectivity helpers — Shapely-based
#
# In Graph2plan the LivingRoom bounding box intentionally extends beyond the
# house boundary, so every other room is geometrically "inside" the LR box.
# Shared-edge checks are meaningless for LR edges.
#
# Instead we compute:
#   lr_space = boundary_polygon.difference(union_of_all_solid_rooms)
# which is the exact open floor area.  A wall of room R has a traversal
# opening wherever it intersects lr_space.
# ---------------------------------------------------------------------------

def _box_poly(x0, y0, x1, y1):
    return Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])


def _build_lr_space(boxes, r, K, types, boundary):
    """LR space = boundary polygon minus all solid non-LR rooms (except r)."""
    bnd_poly = Polygon(np.asarray(boundary, dtype=float)[:, :2])
    blocker_polys = [
        _box_poly(*boxes[k])
        for k in range(K)
        if k != r and int(types[k]) not in _OPEN_TYPES
    ]
    if blocker_polys:
        return bnd_poly.difference(unary_union(blocker_polys))
    return bnd_poly


def _max_opening_length(geom):
    """Longest contiguous free segment from a Shapely line/multiline geometry."""
    if geom is None or geom.is_empty:
        return 0.0
    if geom.geom_type == 'LineString':
        return geom.length
    if geom.geom_type == 'MultiLineString':
        return max(g.length for g in geom.geoms)
    if geom.geom_type == 'GeometryCollection':
        lengths = [_max_opening_length(g) for g in geom.geoms]
        return max(lengths) if lengths else 0.0
    return 0.0


def _lr_free_walls(boxes, r, K, types, boundary, adj_tol=8.0, **_kwargs):
    """
    Shapely-based free wall computation for room r.

    Builds lr_space = boundary.difference(all_solid_rooms_except_r), then
    intersects each of r's 4 wall segments with lr_space to find the exact
    free (traversable) length on each side.

    Returns list of (side, max_free_length, blocker_indices) sorted best-first.
    side: 0=left, 1=top, 2=right, 3=bottom
    """
    rx0, ry0, rx1, ry1 = [float(v) for v in boxes[r]]
    bnd_arr2 = np.asarray(boundary, dtype=float)[:, :2]
    bnd_poly = Polygon(bnd_arr2)
    lr_space = _build_lr_space(boxes, r, K, types, boundary)

    # Exterior walls lie exactly on the house boundary.  Buffer the boundary
    # line slightly and subtract it so those segments don't count as openings.
    exterior_strip = bnd_poly.boundary.buffer(1.0)

    # Shift each probe line 0.5px outward so it sits just outside room r.
    # This avoids the Shapely boundary-coincidence problem where a wall that
    # exactly shares an edge with a subtracted neighbour still appears "free"
    # (lines on the boundary of a difference polygon are included in intersection).
    eps = 0.5
    wall_lines = {
        0: LineString([(rx0 - eps, ry0), (rx0 - eps, ry1)]),   # left
        1: LineString([(rx0, ry0 - eps), (rx1, ry0 - eps)]),   # top
        2: LineString([(rx1 + eps, ry0), (rx1 + eps, ry1)]),   # right
        3: LineString([(rx0, ry1 + eps), (rx1, ry1 + eps)]),   # bottom
    }

    results = []
    for side, wall_line in wall_lines.items():
        opening = wall_line.intersection(lr_space)
        # Remove any portion sitting on the house boundary (exterior wall)
        opening = opening.difference(exterior_strip)
        free = _max_opening_length(opening)
        if free < 0.5:
            continue  # wall is exterior or fully blocked

        # Identify adjacent blocker rooms for this side (for shrink targeting)
        blocker_idxs = []
        for k in range(K):
            if k == r or int(types[k]) in _OPEN_TYPES:
                continue
            kx0, ky0, kx1, ky1 = boxes[k]
            if side == 0 and abs(kx1 - rx0) <= adj_tol and min(ky1, ry1) - max(ky0, ry0) > 0.5:
                blocker_idxs.append(k)
            elif side == 2 and abs(kx0 - rx1) <= adj_tol and min(ky1, ry1) - max(ky0, ry0) > 0.5:
                blocker_idxs.append(k)
            elif side == 1 and abs(ky1 - ry0) <= adj_tol and min(kx1, rx1) - max(kx0, rx0) > 0.5:
                blocker_idxs.append(k)
            elif side == 3 and abs(ky0 - ry1) <= adj_tol and min(kx1, rx1) - max(kx0, rx0) > 0.5:
                blocker_idxs.append(k)

        results.append((side, free, blocker_idxs))

    results.sort(key=lambda x: x[1], reverse=True)
    return results


def _try_slide_to_boundary(boxes, r, K, types, bnd_arr, min_door, **_kwargs):
    """
    Try to slide room r to a boundary wall using Shapely to verify the path.

    For each direction (left/up/right/down):
      - Build the corridor rectangle between R's edge and the boundary edge
      - Trim it to the actual boundary polygon (handles L-shapes)
      - Check the corridor is clear of all other solid rooms
      - Slide distance = trimmed corridor extent in that direction
      - Require slide_dist >= min_door (the freed opposite side gets that gap)
      - Pick the SHORTEST valid slide (nearest boundary wall)

    Returns True if a slide was applied.
    """
    bnd_arr = np.asarray(bnd_arr, dtype=float)
    bnd_poly = Polygon(bnd_arr[:, :2])
    rx0, ry0, rx1, ry1 = [float(v) for v in boxes[r]]
    bx0 = float(bnd_arr[:, 0].min()); bx1 = float(bnd_arr[:, 0].max())
    by0 = float(bnd_arr[:, 1].min()); by1 = float(bnd_arr[:, 1].max())

    obstacle_polys = [
        _box_poly(*boxes[k])
        for k in range(K)
        if k != r and int(types[k]) not in _OPEN_TYPES
    ]
    obstacles = unary_union(obstacle_polys) if obstacle_polys else None

    dir_names = {0: 'left', 1: 'up', 2: 'right', 3: 'down'}
    opp_names = {0: 'right', 1: 'bottom', 2: 'left', 3: 'top'}

    candidates = []
    for d in range(4):
        # Raw corridor from R's edge to the bounding-box boundary edge
        if d == 0:
            raw = _box_poly(bx0, ry0, rx0, ry1)
        elif d == 1:
            raw = _box_poly(rx0, by0, rx1, ry0)
        elif d == 2:
            raw = _box_poly(rx1, ry0, bx1, ry1)
        else:
            raw = _box_poly(rx0, ry1, rx1, by1)

        # Trim to actual boundary (handles L-shaped cutoffs)
        corridor = raw.intersection(bnd_poly)
        if corridor.is_empty or corridor.area < 1.0:
            continue

        # Corridor must be clear of all other solid rooms
        if obstacles is not None:
            overlap = obstacles.intersection(corridor)
            if not overlap.is_empty and overlap.area > 1.0:
                print(f"[FIX ROOMS]   slide {dir_names[d]}: blocked by room in corridor")
                continue

        # Actual slide distance = corridor extent in the slide direction
        cb = corridor.bounds  # (minx, miny, maxx, maxy)
        if d == 0:
            slide_dist = rx0 - cb[0]
        elif d == 1:
            slide_dist = ry0 - cb[1]
        elif d == 2:
            slide_dist = cb[2] - rx1
        else:
            slide_dist = cb[3] - ry1

        if slide_dist < min_door:
            continue  # freed gap would be too narrow for a door

        candidates.append((slide_dist, d))
        print(f"[FIX ROOMS]   slide {dir_names[d]}: clear corridor, dist={slide_dist:.1f}px")

    if not candidates:
        return False

    # Shortest slide = nearest boundary wall = least disruption
    best_dist, best_dir = min(candidates, key=lambda x: x[0])
    dx = [-1, 0, 1, 0][best_dir]
    dy = [0, -1, 0, 1][best_dir]
    boxes[r][0] += dx * best_dist
    boxes[r][2] += dx * best_dist
    boxes[r][1] += dy * best_dist
    boxes[r][3] += dy * best_dist
    print(f"[FIX ROOMS]   Slid {_room_label(r, types)} {dir_names[best_dir]} "
          f"by {best_dist:.1f}px → touches boundary, opens {opp_names[best_dir]} side to LR")
    return True


def _fix_lr_edge(boxes, r, K, types, bnd, min_door):
    """
    Ensure room r has at least min_door px of free interior wall opening onto
    LivingRoom space.

    Strategy (in order):
      1. Perpendicular shrink of the biggest blocker on the best side.
      2. If still blocked, slide room r toward the boundary wall with the most
         available space — this frees the opposite side without creating corridors.

    Side selection: prefer the side that needs the smallest blocker shrink to
    reach min_door (least disruption).  If a side is already >= min_door it is
    reported as OK immediately.
    """
    bnd = np.asarray(bnd, dtype=float)
    walls = _lr_free_walls(boxes, r, K, types, bnd)
    side_names = {0: 'left', 1: 'top', 2: 'right', 3: 'bottom'}

    if not walls:
        print(f"[FIX ROOMS]   {_room_label(r, types)} ↔ LivingRoom: "
              f"all sides exterior — cannot open to LR")
        return

    # Log all candidate sides for diagnostics
    for side, free, blockers in walls:
        blocker_labels = [_room_label(b, types) for b in blockers]
        print(f"[FIX ROOMS]   side={side_names[side]:6s} free={free:.1f}px "
              f"blockers={blocker_labels}")

    # Check if any side already has enough free wall
    for side, free, blockers in walls:
        if free >= min_door:
            print(f"[FIX ROOMS]   {_room_label(r, types)} ↔ LivingRoom: "
                  f"OK — {free:.1f}px free on {side_names[side]} side")
            return

    # No side is free enough.  Pick the side where we need the least shrink
    # from its biggest blocker to reach min_door.
    def _max_shrinkable(side, blockers):
        """How much can the biggest blocker on this side give us via perpendicular shrink?
        Side 0/2 (vertical wall) → shrink blocker in Y (height).
        Side 1/3 (horizontal wall) → shrink blocker in X (width)."""
        best = 0.0
        for k in blockers:
            kx0, ky0, kx1, ky1 = boxes[k]
            min_w, min_h = _min_wh(k, types)
            if side in (1, 3):   # horizontal wall → shrink blocker width (X)
                best = max(best, max(0.0, (kx1 - kx0) - min_w))
            else:                # vertical wall → shrink blocker height (Y)
                best = max(best, max(0.0, (ky1 - ky0) - min_h))
        return best

    # Score each side: (needed shrink, whether it is achievable)
    candidates = []
    for side, free, blockers in walls:
        needed = min_door - free
        achievable = _max_shrinkable(side, blockers)
        candidates.append((side, free, blockers, needed, achievable))

    # Prefer fixable sides; within those, prefer minimum needed shrink.
    # Tiebreaker: prefer horizontal walls (sides 0/2 = left/right) over
    # vertical walls (sides 1/3 = top/bottom) — rooms are more naturally
    # entered from the side than through the ceiling/floor.
    fixable = [(s, f, bl, n, a) for s, f, bl, n, a in candidates if a >= n]
    pool = fixable if fixable else candidates   # fall back to any side

    def _sort_key(x):
        side, free, blockers, needed, achievable = x
        is_vertical_wall = 1 if side in (1, 3) else 0  # prefer 0/2
        return (needed, is_vertical_wall)

    # Try perpendicular shrinks across ALL candidate sides (in priority order)
    # until one achieves min_door.  A single side may have multiple blockers
    # (e.g. two rooms together covering the full wall); after shrinking the
    # primary blocker we re-check and, if still insufficient, keep shrinking
    # secondary blockers on the same side before moving on to the next side.

    def _shrink_one_blocker(side, blockers):
        """Perpendicular-shrink the blocker that covers the most of the chosen
        wall, at whichever end exposes the largest corner.  Returns True if
        any shrink was applied."""
        if not blockers:
            return False

        rx0_, ry0_, rx1_, ry1_ = boxes[r]
        wall_lo = ry0_ if side in (0, 2) else rx0_
        wall_hi = ry1_ if side in (0, 2) else rx1_

        def _cover(k):
            kx0, ky0, kx1, ky1 = boxes[k]
            if side in (0, 2):
                return min(ky1, wall_hi) - max(ky0, wall_lo)
            else:
                return min(kx1, wall_hi) - max(kx0, wall_lo)

        kb = max(blockers, key=_cover)
        kx0, ky0, kx1, ky1 = boxes[kb]
        min_w, min_h = _min_wh(kb, types)

        if side in (1, 3):   # horizontal wall → shrink blocker in X
            cur_left  = max(0.0, kx0 - rx0_)
            cur_right = max(0.0, rx1_ - kx1)
            nl = max(0.0, min_door - cur_left)
            nr = max(0.0, min_door - cur_right)
            can = max(0.0, (kx1 - kx0) - min_w)
            use_l = (nl <= nr) if (can >= nl or can >= nr) else (can >= nl)
            shrink = min(nl if use_l else nr, can)
            if shrink > 0:
                if use_l:
                    boxes[kb][0] += shrink
                    print(f"[FIX ROOMS]   Shrunk {_room_label(kb, types)} "
                          f"left edge by {shrink:.1f}px (perp, opens {side_names[side]} of {_room_label(r, types)})")
                else:
                    boxes[kb][2] -= shrink
                    print(f"[FIX ROOMS]   Shrunk {_room_label(kb, types)} "
                          f"right edge by {shrink:.1f}px (perp, opens {side_names[side]} of {_room_label(r, types)})")
                return True

        else:                # vertical wall → shrink blocker in Y
            cur_top = max(0.0, ky0 - ry0_)
            cur_bot = max(0.0, ry1_ - ky1)
            nt = max(0.0, min_door - cur_top)
            nb = max(0.0, min_door - cur_bot)
            can = max(0.0, (ky1 - ky0) - min_h)
            use_t = (nt <= nb) if (can >= nt or can >= nb) else (can >= nt)
            shrink = min(nt if use_t else nb, can)
            if shrink > 0:
                if use_t:
                    boxes[kb][1] += shrink
                    print(f"[FIX ROOMS]   Shrunk {_room_label(kb, types)} "
                          f"top edge by {shrink:.1f}px (perp, opens {side_names[side]} of {_room_label(r, types)})")
                else:
                    boxes[kb][3] -= shrink
                    print(f"[FIX ROOMS]   Shrunk {_room_label(kb, types)} "
                          f"bottom edge by {shrink:.1f}px (perp, opens {side_names[side]} of {_room_label(r, types)})")
                return True

        return False

    print(f"[FIX ROOMS]   {_room_label(r, types)} ↔ LivingRoom: BLOCKED — "
          f"trying {len(pool)} candidate side(s)…")

    for side_candidate, free_c, blockers_c, needed_c, _ in sorted(pool, key=_sort_key):
        if not blockers_c:
            # No blocker — extend room r on this side directly
            rx0, ry0, rx1, ry1 = boxes[r]
            if side_candidate == 0:
                boxes[r][0] = rx0 - needed_c
                print(f"[FIX ROOMS]   Extended {_room_label(r, types)} left by {needed_c:.1f}px")
            elif side_candidate == 2:
                boxes[r][2] = rx1 + needed_c
                print(f"[FIX ROOMS]   Extended {_room_label(r, types)} right by {needed_c:.1f}px")
            elif side_candidate == 1:
                boxes[r][1] = ry0 - needed_c
                print(f"[FIX ROOMS]   Extended {_room_label(r, types)} top by {needed_c:.1f}px")
            else:
                boxes[r][3] = ry1 + needed_c
                print(f"[FIX ROOMS]   Extended {_room_label(r, types)} bottom by {needed_c:.1f}px")
            chk = _lr_free_walls(boxes, r, K, types, bnd)
            if max((w[1] for w in chk), default=0.0) >= min_door:
                print(f"[FIX ROOMS]   ✓ Fixed via extension")
                return
            continue

        # Try shrinking blockers one at a time (biggest first) until sufficient
        remaining_blockers = list(blockers_c)
        max_attempts = len(remaining_blockers)
        for _ in range(max_attempts):
            chk = _lr_free_walls(boxes, r, K, types, bnd)
            if max((w[1] for w in chk), default=0.0) >= min_door:
                print(f"[FIX ROOMS]   ✓ Fixed via perpendicular shrink on "
                      f"{side_names[side_candidate]} side")
                return
            # Re-read current blockers for this side from the live walls
            live = {s: bl for s, _, bl in chk}
            current_blockers = live.get(side_candidate, remaining_blockers)
            if not current_blockers:
                break
            applied = _shrink_one_blocker(side_candidate, current_blockers)
            if not applied:
                break  # can't shrink further on this side

        # Check again after all shrink attempts for this side
        chk = _lr_free_walls(boxes, r, K, types, bnd)
        if max((w[1] for w in chk), default=0.0) >= min_door:
            print(f"[FIX ROOMS]   ✓ Fixed via perpendicular shrink on "
                  f"{side_names[side_candidate]} side")
            return

    # All perpendicular shrink attempts exhausted
    chk_pre = _lr_free_walls(boxes, r, K, types, bnd)
    pre_best = max((w[1] for w in chk_pre), default=0.0)
    if pre_best >= min_door:
        print(f"[FIX ROOMS]   ✓ Fixed: max free={pre_best:.1f}px")
    else:
        print(f"[FIX ROOMS]   ✗ Could not open {_room_label(r, types)} to LR — "
              f"best={pre_best:.1f}px after all shrink attempts")


# ---------------------------------------------------------------------------
# Main pass
# ---------------------------------------------------------------------------

def fix_room_connectivity(boxes, types, edges, boundary, min_door=MIN_DOOR_PX, max_fix=40.0):
    """
    Heuristic pass: use graph edges to find room pairs that should be
    connected and alter the layout so they share a wall >= min_door px.

    Pre-conditioning: runs Snap → Expand LR → Snap before the connectivity
    checks.  This settles the layout into a stable state where rooms are flush
    against walls and the LivingRoom fills available space, making the
    subsequent targeted shrinks much more reliable.

    Two connectivity strategies:

    - LivingRoom edges: LR's bounding box intentionally extends beyond the
      house boundary in Graph2plan, so geometric shared-edge checks are
      meaningless.  Instead we verify that the non-LR room has at least
      min_door px of free interior wall (not blocked by another solid room).
      If not, the biggest blocker is shrunk.

    - Non-LR edges: standard shared-wall check; closes small gaps / small
      overlaps (within max_fix) and ensures perpendicular overlap >= min_door.
    """
    boxes = np.array(boxes, dtype=float)
    types = np.array(types, dtype=int).flatten()
    K = len(boxes)

    if edges is None or len(edges) == 0:
        print("[FIX ROOMS] No edges — nothing to check.")
        return boxes.astype(int)

    edge_arr = np.array(edges)
    if edge_arr.ndim == 1:
        edge_arr = edge_arr.reshape(1, -1)

    print(f"[FIX ROOMS] Checking {len(edge_arr)} connections (min_door={min_door}px)...")

    for edge in edge_arr:
        u, v = int(edge[0]), int(edge[1])
        if u >= K or v >= K:
            continue

        u_is_lr = (int(types[u]) == 0)
        v_is_lr = (int(types[v]) == 0)

        # ------------------------------------------------------------------
        # LivingRoom edges — probe-based check.
        # LR underlies all rooms so shared-wall geometry is meaningless.
        # Instead, verify that room r has >= min_door px of wall facing
        # visible LivingRoom space (inside boundary, not inside another room).
        # If blocked, shrink the blocker perpendicularly — no corridors.
        # ------------------------------------------------------------------
        if u_is_lr or v_is_lr:
            r = v if u_is_lr else u
            # Skip non-habitable elements (FrontDoor, ExteriorWall, etc.)
            if int(types[r]) in _OPEN_TYPES:
                print(f"[FIX ROOMS] {_room_label(u, types)} ↔ {_room_label(v, types)}: "
                      f"skipped (non-habitable type)")
                continue

            # Measure opening before any fix
            walls_pre = _lr_free_walls(boxes, r, K, types, boundary)
            best_pre = max((w[1] for w in walls_pre), default=0.0)
            status_pre = "OK" if best_pre >= min_door else "BLOCKED"
            side_names_local = {0: 'left', 1: 'top', 2: 'right', 3: 'bottom'}
            best_side_pre = walls_pre[0][0] if walls_pre else None
            print(f"[FIX ROOMS] ── CONNECTION: {_room_label(r, types)} ↔ LivingRoom ──")
            print(f"[FIX ROOMS]    Visual opening  : {best_pre:.1f}px "
                  f"(best side: {side_names_local.get(best_side_pre, '?')})")
            print(f"[FIX ROOMS]    Minimum required: {min_door}px")
            print(f"[FIX ROOMS]    Status          : {status_pre}")

            _fix_lr_edge(boxes, r, K, types, boundary, min_door)

            # Measure again after fix
            walls_post = _lr_free_walls(boxes, r, K, types, boundary)
            best_post = max((w[1] for w in walls_post), default=0.0)
            status_post = "OK" if best_post >= min_door else "STILL BLOCKED"
            print(f"[FIX ROOMS]    Opening after fix: {best_post:.1f}px  → {status_post}")
            continue

        # ------------------------------------------------------------------
        # Non-LR edge — shared-wall approach
        # ------------------------------------------------------------------
        shared = _current_shared(boxes, u, v)
        print(f"[FIX ROOMS] ── CONNECTION: {_room_label(u, types)} ↔ {_room_label(v, types)} ──")
        print(f"[FIX ROOMS]    Shared wall     : {shared:.1f}px")
        print(f"[FIX ROOMS]    Minimum required: {min_door}px")
        print(f"[FIX ROOMS]    Status          : {'OK' if shared >= min_door else 'BLOCKED'}")
        if shared >= min_door:
            continue

        axis, u_near, v_near, gap, perp_ov = _best_direction(boxes, u, v)
        print(f"[FIX ROOMS]    Gap to close    : {gap:.1f}px  (perp overlap: {perp_ov:.1f}px)")

        if abs(gap) > max_fix:
            print(f"[FIX ROOMS]   Skipping — gap/overlap {gap:.1f}px exceeds max_fix={max_fix:.0f}px")
            continue

        ux0, uy0, ux1, uy1 = boxes[u]
        vx0, vy0, vx1, vy1 = boxes[v]

        # Step 0: resolve small overlap
        if gap < 0:
            print(f"[FIX ROOMS]   Resolving overlap of {-gap:.1f}px...")
            uw, uh = ux1 - ux0, uy1 - uy0
            vw, vh = vx1 - vx0, vy1 - vy0
            u_smaller = (uw * uh) <= (vw * vh)
            if axis == 'lr':
                if u_near == 2:
                    boxes[u][2] = vx0 if u_smaller else (boxes[v].__setitem__(0, ux1) or boxes[v][0])
                    if not u_smaller:
                        boxes[v][0] = ux1
                    else:
                        boxes[u][2] = vx0
                else:
                    if u_smaller:
                        boxes[u][0] = vx1
                    else:
                        boxes[v][2] = ux0
            else:
                if u_near == 3:
                    if u_smaller:
                        boxes[u][3] = vy0
                    else:
                        boxes[v][1] = uy1
                else:
                    if u_smaller:
                        boxes[u][1] = vy1
                    else:
                        boxes[v][3] = uy0
            ux0, uy0, ux1, uy1 = boxes[u]
            vx0, vy0, vx1, vy1 = boxes[v]
            gap = 0.0

        # Step 1: close the gap
        if gap > 0:
            print(f"[FIX ROOMS]   Closing gap of {gap:.1f}px...")
            if axis == 'lr':
                x_lo = ux1 if u_near == 2 else vx1
                x_hi = vx0 if u_near == 2 else ux0
                y_lo = min(uy0, vy0)
                y_hi = max(uy1, vy1)
            else:
                y_lo = uy1 if u_near == 3 else vy1
                y_hi = vy0 if u_near == 3 else uy0
                x_lo = min(ux0, vx0)
                x_hi = max(ux1, vx1)

            remaining = gap
            for k in range(K):
                if k in (u, v) or remaining <= 0:
                    continue
                kx0, ky0, kx1, ky1 = boxes[k]
                if axis == 'lr':
                    in_gap = (kx0 < x_hi + 1 and kx1 > x_lo - 1
                              and min(ky1, y_hi) - max(ky0, y_lo) > 0)
                    if not in_gap:
                        continue
                    min_w, _ = _min_wh(k, types)
                    shrink = min(remaining, max(0.0, (kx1 - kx0) - min_w))
                    if shrink > 0:
                        boxes[k][0 if u_near == 2 else 2] += (shrink if u_near == 2 else -shrink)
                        remaining -= shrink
                        print(f"[FIX ROOMS]   Shrunk {_room_label(k, types)} "
                              f"in x by {shrink:.1f}px — {remaining:.1f}px left")
                else:
                    in_gap = (ky0 < y_hi + 1 and ky1 > y_lo - 1
                              and min(kx1, x_hi) - max(kx0, x_lo) > 0)
                    if not in_gap:
                        continue
                    _, min_h = _min_wh(k, types)
                    shrink = min(remaining, max(0.0, (ky1 - ky0) - min_h))
                    if shrink > 0:
                        boxes[k][1 if u_near == 3 else 3] += (shrink if u_near == 3 else -shrink)
                        remaining -= shrink
                        print(f"[FIX ROOMS]   Shrunk {_room_label(k, types)} "
                              f"in y by {shrink:.1f}px — {remaining:.1f}px left")

            if remaining > 0:
                uw, uh = ux1 - ux0, uy1 - uy0
                vw, vh = vx1 - vx0, vy1 - vy0
                u_smaller = (uw * uh) <= (vw * vh)
                if axis == 'lr':
                    if u_near == 2:
                        if u_smaller:
                            boxes[u][2] = vx0
                            print(f"[FIX ROOMS]   Extended {_room_label(u, types)} right")
                        else:
                            boxes[v][0] = ux1
                            print(f"[FIX ROOMS]   Extended {_room_label(v, types)} left")
                    else:
                        if u_smaller:
                            boxes[u][0] = vx1
                            print(f"[FIX ROOMS]   Extended {_room_label(u, types)} left")
                        else:
                            boxes[v][2] = ux0
                            print(f"[FIX ROOMS]   Extended {_room_label(v, types)} right")
                else:
                    if u_near == 3:
                        if u_smaller:
                            boxes[u][3] = vy0
                            print(f"[FIX ROOMS]   Extended {_room_label(u, types)} bottom")
                        else:
                            boxes[v][1] = uy1
                            print(f"[FIX ROOMS]   Extended {_room_label(v, types)} top")
                    else:
                        if u_smaller:
                            boxes[u][1] = vy1
                            print(f"[FIX ROOMS]   Extended {_room_label(u, types)} top")
                        else:
                            boxes[v][3] = uy0
                            print(f"[FIX ROOMS]   Extended {_room_label(v, types)} bottom")

        # Step 2: ensure perpendicular overlap >= min_door
        ux0, uy0, ux1, uy1 = boxes[u]
        vx0, vy0, vx1, vy1 = boxes[v]
        if axis == 'lr':
            overlap = max(0.0, min(uy1, vy1) - max(uy0, vy0))
        else:
            overlap = max(0.0, min(ux1, vx1) - max(ux0, vx0))

        if overlap < min_door:
            needed = min_door - overlap
            uw, uh = ux1 - ux0, uy1 - uy0
            vw, vh = vx1 - vx0, vy1 - vy0
            u_smaller = (uw * uh) <= (vw * vh)
            if axis == 'lr':
                if u_smaller:
                    boxes[u][3 if uy1 <= vy1 else 1] += (needed if uy1 <= vy1 else -needed)
                    boxes[u][3] = min(boxes[u][3], vy1)
                    boxes[u][1] = max(boxes[u][1], vy0)
                else:
                    boxes[v][3 if vy1 <= uy1 else 1] += (needed if vy1 <= uy1 else -needed)
                    boxes[v][3] = min(boxes[v][3], uy1)
                    boxes[v][1] = max(boxes[v][1], uy0)
            else:
                if u_smaller:
                    boxes[u][2 if ux1 <= vx1 else 0] += (needed if ux1 <= vx1 else -needed)
                    boxes[u][2] = min(boxes[u][2], vx1)
                    boxes[u][0] = max(boxes[u][0], vx0)
                else:
                    boxes[v][2 if vx1 <= ux1 else 0] += (needed if vx1 <= ux1 else -needed)
                    boxes[v][2] = min(boxes[v][2], ux1)
                    boxes[v][0] = max(boxes[v][0], ux0)
            print(f"[FIX ROOMS]   Extended smaller room by {needed:.1f}px for perp overlap")

        new_shared = _current_shared(boxes, u, v)
        if new_shared >= min_door:
            print(f"[FIX ROOMS]   ✓ Fixed: shared={new_shared:.1f}px")
        else:
            print(f"[FIX ROOMS]   ✗ Still {new_shared:.1f}px")

    print("[FIX ROOMS] Done.")
    return boxes.astype(int)

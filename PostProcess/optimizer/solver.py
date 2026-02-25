"""
Constrained layout optimizer for Graph2Plan.

Stage 1 — OR-Tools CP-SAT:
  Hard constraints:  non-overlap, boundary containment, spatial ordering
  Objective:         minimize L1 displacement from neural network output

Inputs  (from app.align output):
  boxes    (K, 4)  int  [x0, y0, x1, y1]  pixel coords in 256x256 space
  types    (K,)    int  room type indices
  edges    (E, 3)  int  [room_u, room_v, predicate_id]
  boundary (N, 4)  int  [x, y, dir, isNew]

Output:
  optimized_boxes  (K, 4)  int  [x0, y0, x1, y1]
"""

import numpy as np
import warnings

# Predicate indices from g2p/utils.py get_vocab()
_LEFT_ABOVE  = 0
_LEFT_BELOW  = 1
_LEFT_OF     = 2
_ABOVE       = 3
_INSIDE      = 4   # skipped
_SURROUNDING = 5   # skipped
_BELOW       = 6
_RIGHT_OF    = 7
_RIGHT_ABOVE = 8
_RIGHT_BELOW = 9

CANVAS       = 256
MIN_ROOM_PX  = 10   # ~0.7 m — small floor to allow solver flexibility


def _poly_x_range_at_y(vertices, y):
    """
    Horizontal scan of a closed polygon at height y.
    Returns (x_lo, x_hi) of the polygon interior, or None if y is outside.
    vertices: list of [x, y] pairs (closed — last == first, or auto-closed).
    """
    xs = []
    n = len(vertices)
    for i in range(n - 1):
        x1, y1 = float(vertices[i][0]),   float(vertices[i][1])
        x2, y2 = float(vertices[i+1][0]), float(vertices[i+1][1])
        if y1 == y2:
            continue                          # horizontal edge — skip
        if min(y1, y2) < y <= max(y1, y2):
            xi = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            xs.append(xi)
    if len(xs) < 2:
        return None
    return min(xs), max(xs)


def optimize_layout(boxes, types, edges, boundary=None, timeout=5.0):
    """
    Run the CP-SAT layout optimizer.

    Returns
    -------
    optimized_boxes : np.ndarray  shape (K, 4)  dtype int
    status          : str  'OPTIMAL' | 'FEASIBLE' | 'FAILED'
    """
    try:
        from ortools.sat.python import cp_model
    except ImportError:
        warnings.warn(
            "ortools not installed. Run: pip install ortools\n"
            "Returning original boxes without optimization.",
            UserWarning,
        )
        return np.array(boxes, dtype=int), 'FAILED'

    boxes = np.array(boxes, dtype=int)
    K     = len(boxes)

    if K == 0:
        return boxes, 'OPTIMAL'

    # --- Boundary bounding box and polygon vertices --------------------------
    if boundary is not None and len(boundary) > 0:
        bnd = np.array(boundary)
        bx0 = int(np.min(bnd[:, 0]))
        by0 = int(np.min(bnd[:, 1]))
        bx1 = int(np.max(bnd[:, 0]))
        by1 = int(np.max(bnd[:, 1]))
        # Build ordered polygon vertex list for x-range scan
        poly_verts = bnd[:, :2].tolist()          # [[x,y], ...]
        if poly_verts[0] != poly_verts[-1]:
            poly_verts.append(poly_verts[0])      # close the polygon
    else:
        bx0, by0, bx1, by1 = 0, 0, CANVAS - 1, CANVAS - 1
        poly_verts = None

    W = bx1 - bx0
    H = by1 - by0

    # --- Initial positions (clamped to valid range) --------------------------
    x0 = [int(np.clip(boxes[i, 0], bx0, bx1 - MIN_ROOM_PX)) for i in range(K)]
    y0 = [int(np.clip(boxes[i, 1], by0, by1 - MIN_ROOM_PX)) for i in range(K)]
    w0 = [int(np.clip(boxes[i, 2] - boxes[i, 0], MIN_ROOM_PX, W)) for i in range(K)]
    h0 = [int(np.clip(boxes[i, 3] - boxes[i, 1], MIN_ROOM_PX, H)) for i in range(K)]

    # --- CP-SAT model --------------------------------------------------------
    model = cp_model.CpModel()

    x = [model.NewIntVar(bx0, bx1 - MIN_ROOM_PX, f'x_{i}') for i in range(K)]
    y = [model.NewIntVar(by0, by1 - MIN_ROOM_PX, f'y_{i}') for i in range(K)]
    w = [model.NewIntVar(MIN_ROOM_PX, W,           f'w_{i}') for i in range(K)]
    h = [model.NewIntVar(MIN_ROOM_PX, H,           f'h_{i}') for i in range(K)]

    # Derived end-coordinate variables (needed by NewIntervalVar)
    xe = [model.NewIntVar(bx0, bx1, f'xe_{i}') for i in range(K)]
    ye = [model.NewIntVar(by0, by1, f'ye_{i}') for i in range(K)]

    for i in range(K):
        model.Add(xe[i] == x[i] + w[i])
        model.Add(ye[i] == y[i] + h[i])
        # Stay inside boundary
        model.Add(xe[i] <= bx1)
        model.Add(ye[i] <= by1)

    # --- Identify intentional overlaps from edge predicates ------------------
    # Any two rooms connected by an edge in the graph are neighbours and may
    # intentionally overlap to form composite shapes (e.g. Kitchen + LivingRoom
    # sharing a wall region).  Non-adjacent rooms must never overlap.
    #
    # inside (4): u is inside v  → add containment constraint
    # surrounding (5): u surrounds v → add containment constraint
    contained_in = {}   # contained_in[u] = v  means u must sit inside v
    intentional  = set()  # set of (min,max) index pairs allowed to overlap

    for edge in edges:
        u, v, pred = int(edge[0]), int(edge[1]), int(edge[2])
        if u >= K or v >= K:
            continue
        # Every graph edge allows overlap between the two rooms
        intentional.add((min(u, v), max(u, v)))
        if pred == _INSIDE:
            contained_in[u] = v
        elif pred == _SURROUNDING:
            contained_in[v] = u

    # --- Non-overlap: pairwise, skipping intentional pairs -------------------
    # For the pairs that must not overlap we add a disjunctive constraint:
    # at least one of the four separation conditions must hold.
    for i in range(K):
        for j in range(i + 1, K):
            if (i, j) in intentional:
                continue    # intentional overlap — handled below
            b_il = model.NewBoolVar(f'sep_{i}_{j}_iL')   # i is left of j
            b_ir = model.NewBoolVar(f'sep_{i}_{j}_iR')   # i is right of j
            b_ia = model.NewBoolVar(f'sep_{i}_{j}_iA')   # i is above j
            b_ib = model.NewBoolVar(f'sep_{i}_{j}_iB')   # i is below j
            model.Add(xe[i] <= x[j]).OnlyEnforceIf(b_il)
            model.Add(xe[j] <= x[i]).OnlyEnforceIf(b_ir)
            model.Add(ye[i] <= y[j]).OnlyEnforceIf(b_ia)
            model.Add(ye[j] <= y[i]).OnlyEnforceIf(b_ib)
            model.AddBoolOr([b_il, b_ir, b_ia, b_ib])

    # --- Containment for inside/surrounding pairs ----------------------------
    for inner, outer in contained_in.items():
        model.Add(x[inner]  >= x[outer])
        model.Add(xe[inner] <= xe[outer])
        model.Add(y[inner]  >= y[outer])
        model.Add(ye[inner] <= ye[outer])

    # --- Spatial ordering from edge predicates -------------------------------
    # We use 2*center comparisons to avoid fractions:
    #   2 * center_x[i]  =  2*x[i] + w[i]
    for edge in edges:
        u, v, pred = int(edge[0]), int(edge[1]), int(edge[2])
        if u >= K or v >= K:
            continue

        cx_u = 2 * x[u] + w[u]
        cx_v = 2 * x[v] + w[v]
        cy_u = 2 * y[u] + h[u]
        cy_v = 2 * y[v] + h[v]

        if pred == _LEFT_OF:
            model.Add(cx_u <= cx_v)
        elif pred == _RIGHT_OF:
            model.Add(cx_u >= cx_v)
        elif pred == _ABOVE:
            model.Add(cy_u <= cy_v)
        elif pred == _BELOW:
            model.Add(cy_u >= cy_v)
        elif pred == _LEFT_ABOVE:
            model.Add(cx_u <= cx_v)
            model.Add(cy_u <= cy_v)
        elif pred == _LEFT_BELOW:
            model.Add(cx_u <= cx_v)
            model.Add(cy_u >= cy_v)
        elif pred == _RIGHT_ABOVE:
            model.Add(cx_u >= cx_v)
            model.Add(cy_u <= cy_v)
        elif pred == _RIGHT_BELOW:
            model.Add(cx_u >= cx_v)
            model.Add(cy_u >= cy_v)
        # _INSIDE and _SURROUNDING handled above as containment

    # --- Wall-sharing for adjacent pairs -------------------------------------
    CONTAINMENT_RATIO = 0.60
    wall_constraints_added = 0
    size_ratio_inner = set()   # rooms forced inside a larger room by size ratio

    for edge in edges:
        u, v, pred = int(edge[0]), int(edge[1]), int(edge[2])
        if u >= K or v >= K:
            continue
        if pred in (_INSIDE, _SURROUNDING):
            continue   # already handled by containment block above
        if int(types[u]) == 0 or int(types[v]) == 0:
            continue   # exclude LivingRoom

        area_u = w0[u] * h0[u]
        area_v = w0[v] * h0[v]

        if area_u == 0 or area_v == 0:
            continue

        if area_u <= area_v:
            inner, outer = u, v
            ratio = area_u / area_v
        else:
            inner, outer = v, u
            ratio = area_v / area_u

        if ratio < CONTAINMENT_RATIO:
            model.Add(x[inner]  >= x[outer])
            model.Add(xe[inner] <= xe[outer])
            model.Add(y[inner]  >= y[outer])
            model.Add(ye[inner] <= ye[outer])
            size_ratio_inner.add(inner)
            wall_constraints_added += 1
        else:
            if pred == _LEFT_OF:
                model.Add(xe[u] == x[v])
            elif pred == _RIGHT_OF:
                model.Add(xe[v] == x[u])
            elif pred == _ABOVE:
                model.Add(ye[u] == y[v])
            elif pred == _BELOW:
                model.Add(ye[v] == y[u])
            wall_constraints_added += 1

    # --- Inward push: rooms whose original position overhangs a boundary wall --
    # Uses the raw neural-network boxes (before clipping) to detect overhang.
    # Any edge that sticks out is hard-snapped to that wall.  The displacement
    # objective then preserves the room's size and slides it inward.
    #
    # Excluded: LivingRoom (type 0) and rooms already pinned inside another.
    push_excluded = set(contained_in.keys()) | size_ratio_inner
    push_inward   = set()   # rooms handled here; skipped in threshold-snap below
    push_added    = 0

    for i in range(K):
        if int(types[i]) == 0 or i in push_excluded:
            continue

        orig_x0 = int(boxes[i, 0])
        orig_y0 = int(boxes[i, 1])
        orig_x1 = int(boxes[i, 2])
        orig_y1 = int(boxes[i, 3])

        pushed = False

        if orig_x1 > bx1:
            model.Add(xe[i] == bx1)
            pushed = True
        if orig_x0 < bx0:
            model.Add(x[i] == bx0)
            pushed = True
        if orig_y1 > by1:
            model.Add(ye[i] == by1)
            pushed = True
        if orig_y0 < by0:
            model.Add(y[i] == by0)
            pushed = True

        if poly_verts is not None:
            cy = (orig_y0 + orig_y1) / 2.0
            xr = _poly_x_range_at_y(poly_verts, cy)
            if xr is not None:
                px_lo, px_hi = int(np.floor(xr[0])), int(np.ceil(xr[1]))
                if orig_x0 < px_lo:
                    model.Add(x[i] >= px_lo)
                    pushed = True
                if orig_x1 > px_hi:
                    model.Add(xe[i] <= px_hi)
                    pushed = True

        if pushed:
            push_inward.add(i)
            push_added += 1

    # --- Boundary wall snapping (threshold-based) ----------------------------
    SNAP_THRESHOLD = max(20, min(W, H) // 8)
    snap_excluded  = push_excluded | push_inward
    snap_added     = 0

    for i in range(K):
        if int(types[i]) == 0:
            continue
        if i in snap_excluded:
            continue

        xi0, yi0 = x0[i], y0[i]
        xi1, yi1 = x0[i] + w0[i], y0[i] + h0[i]

        dist_left   = xi0 - bx0
        dist_right  = bx1 - xi1
        dist_top    = yi0 - by0
        dist_bottom = by1 - yi1

        if min(dist_left, dist_right, dist_top, dist_bottom) > SNAP_THRESHOLD:
            continue

        if dist_left <= SNAP_THRESHOLD:
            model.Add(x[i] == bx0)
        if dist_right <= SNAP_THRESHOLD:
            model.Add(xe[i] == bx1)
        if dist_top <= SNAP_THRESHOLD:
            model.Add(y[i] == by0)
        if dist_bottom <= SNAP_THRESHOLD:
            model.Add(ye[i] == by1)
        snap_added += 1

    # --- Gap closing ---------------------------------------------------------
    # For any two rooms that are nearly touching (gap < GAP_CLOSE_PX) and
    # already aligned in the perpendicular direction (they share wall length),
    # force them to actually touch.
    #
    # Only applied to non-intentional pairs (intentional pairs can overlap and
    # are handled by the containment / wall-sharing constraints above).
    GAP_CLOSE_PX = 15
    gap_closed = 0

    for i in range(K):
        for j in range(i + 1, K):
            if (i, j) in intentional:
                continue

            xi0, yi0 = x0[i], y0[i]
            xi1, yi1 = x0[i] + w0[i], y0[i] + h0[i]
            xj0, yj0 = x0[j], y0[j]
            xj1, yj1 = x0[j] + w0[j], y0[j] + h0[j]

            # Horizontal gap: i is to the left of j
            h_gap = xj0 - xi1
            if 0 < h_gap <= GAP_CLOSE_PX:
                y_overlap = min(yi1, yj1) - max(yi0, yj0)
                if y_overlap > 0:
                    model.Add(xe[i] == x[j])
                    gap_closed += 1
                    continue

            h_gap_rev = xi0 - xj1
            if 0 < h_gap_rev <= GAP_CLOSE_PX:
                y_overlap = min(yi1, yj1) - max(yi0, yj0)
                if y_overlap > 0:
                    model.Add(xe[j] == x[i])
                    gap_closed += 1
                    continue

            v_gap = yj0 - yi1
            if 0 < v_gap <= GAP_CLOSE_PX:
                x_overlap = min(xi1, xj1) - max(xi0, xj0)
                if x_overlap > 0:
                    model.Add(ye[i] == y[j])
                    gap_closed += 1
                    continue

            v_gap_rev = yi0 - yj1
            if 0 < v_gap_rev <= GAP_CLOSE_PX:
                x_overlap = min(xi1, xj1) - max(xi0, xj0)
                if x_overlap > 0:
                    model.Add(ye[j] == y[i])
                    gap_closed += 1

    # --- Objective: minimize L1 displacement from initial boxes --------------
    dx = [model.NewIntVar(0, W, f'dx_{i}') for i in range(K)]
    dy = [model.NewIntVar(0, H, f'dy_{i}') for i in range(K)]
    dw = [model.NewIntVar(0, W, f'dw_{i}') for i in range(K)]
    dh = [model.NewIntVar(0, H, f'dh_{i}') for i in range(K)]

    for i in range(K):
        model.AddAbsEquality(dx[i], x[i] - x0[i])
        model.AddAbsEquality(dy[i], y[i] - y0[i])
        model.AddAbsEquality(dw[i], w[i] - w0[i])
        model.AddAbsEquality(dh[i], h[i] - h0[i])

    model.Minimize(sum(dx) + sum(dy) + sum(dw) + sum(dh))

    # --- Solve ---------------------------------------------------------------
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds  = timeout
    solver.parameters.num_search_workers   = 4

    status_code = solver.Solve(model)

    if status_code in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        result = np.zeros((K, 4), dtype=int)
        for i in range(K):
            xi = solver.Value(x[i])
            yi = solver.Value(y[i])
            wi = solver.Value(w[i])
            hi = solver.Value(h[i])
            result[i] = [xi, yi, xi + wi, yi + hi]
        status = 'OPTIMAL' if status_code == cp_model.OPTIMAL else 'FEASIBLE'
        print(f"  [SOLVER] Status: {status}")
        return result, status

    # Solver failed — return original boxes clipped to boundary
    print("  [SOLVER] Status: FAILED — returning original boxes unchanged")
    warnings.warn("CP-SAT solver failed to find a feasible solution. "
                  "Returning original boxes clipped to boundary.", UserWarning)
    result = boxes.copy()
    result[:, 0] = np.clip(result[:, 0], bx0, bx1)
    result[:, 1] = np.clip(result[:, 1], by0, by1)
    result[:, 2] = np.clip(result[:, 2], bx0, bx1)
    result[:, 3] = np.clip(result[:, 3], by0, by1)
    return result, 'FAILED'


def expand_living_room(boxes, types, boundary):
    """
    Expand the LivingRoom (type 0) to fill empty space in all four directions.

    For each direction the algorithm:
      1. Finds rooms that are on that side of the LivingRoom *and* overlap with
         it in the perpendicular axis (i.e., rooms that would be in the path of
         the expansion).
      2. Shifts that group of rooms as a unit toward the boundary wall until the
         outermost room in the group touches the wall.
      3. Extends the LivingRoom edge to touch the nearest shifted room (or all
         the way to the wall when nothing is blocking).

    Directions are processed sequentially (left → right → top → bottom).
    The full 4-direction cycle repeats until LR's bounds stop changing
    (convergence), allowing chain reactions: e.g. expanding RIGHT may expose
    new rooms in the BOTTOM blocking set that weren't in LR's original x-range.

    Returns
    -------
    expanded_boxes : np.ndarray  shape (K, 4)  dtype int
    """
    LR_TYPE = 0
    boxes = np.array(boxes, dtype=float)
    types = np.array(types, dtype=int)
    K = len(boxes)

    lr_indices = np.where(types == LR_TYPE)[0]
    if len(lr_indices) == 0:
        return boxes.astype(int)
    lr_idx = lr_indices[0]

    bnd  = np.array(boundary)
    bx0  = float(np.min(bnd[:, 0]))
    by0  = float(np.min(bnd[:, 1]))
    bx1  = float(np.max(bnd[:, 0]))
    by1  = float(np.max(bnd[:, 1]))

    print(f"  [EXPAND] LR (room {lr_idx}): {boxes[lr_idx].tolist()}")
    print(f"  [EXPAND] Boundary: x=[{bx0},{bx1}] y=[{by0},{by1}]")
    # Print all non-LR rooms for reference
    for i in range(K):
        if i != lr_idx:
            print(f"  [EXPAND] Room {i} (type {types[i]}): {boxes[i].tolist()}")

    moved_rooms = set()   # rooms fixed after being pushed in a previous pass
    MAX_PASSES = 20
    for pass_num in range(1, MAX_PASSES + 1):
        prev_lr    = boxes[lr_idx].copy()
        pass_start = boxes.copy()           # snapshot to detect which rooms move this pass
        print(f"  [EXPAND] === PASS {pass_num} (fixed={sorted(moved_rooms)}) ===")

        for direction in ('left', 'right', 'top', 'bottom'):
            lx0, ly0, lx1, ly1 = (float(v) for v in boxes[lr_idx])
            print(f"  [EXPAND] --- {direction.upper()} --- LR now: [{lx0},{ly0},{lx1},{ly1}]")

            if direction == 'left':
                blocking = [
                    i for i in range(K) if i != lr_idx
                    and i not in moved_rooms
                    and boxes[i][2] <= lx0
                    and min(boxes[i][3], ly1) - max(boxes[i][1], ly0) > 0
                ]
                print(f"  [EXPAND] blocking={blocking}")
                if not blocking:
                    print(f"  [EXPAND] no blockers → LR.x0 = {bx0}")
                    boxes[lr_idx][0] = bx0
                else:
                    # Greedy compaction: sort outermost (smallest x0) first, pack toward bx0
                    srt = sorted(blocking, key=lambda i: boxes[i][0])
                    ws  = {i: float(boxes[i][2] - boxes[i][0]) for i in srt}
                    placed = {}
                    for i in srt:
                        pos = bx0
                        for j in placed:
                            if min(boxes[i][3], boxes[j][3]) - max(boxes[i][1], boxes[j][1]) > 0:
                                pos = max(pos, placed[j] + ws[j])
                        placed[i] = pos
                    for i in srt:
                        boxes[i][0] = placed[i]
                        boxes[i][2] = placed[i] + ws[i]
                        print(f"  [EXPAND]   room {i} → {boxes[i].tolist()}")
                    new_x0 = max(placed[i] + ws[i] for i in srt)
                    print(f"  [EXPAND] LR.x0: {lx0} → {new_x0}")
                    boxes[lr_idx][0] = new_x0

            elif direction == 'right':
                blocking = [
                    i for i in range(K) if i != lr_idx
                    and i not in moved_rooms
                    and boxes[i][0] >= lx1
                    and min(boxes[i][3], ly1) - max(boxes[i][1], ly0) > 0
                ]
                print(f"  [EXPAND] blocking={blocking}")
                if not blocking:
                    print(f"  [EXPAND] no blockers → LR.x1 = {bx1}")
                    boxes[lr_idx][2] = bx1
                else:
                    # Greedy compaction: sort outermost (largest x1) first, pack toward bx1
                    srt = sorted(blocking, key=lambda i: -boxes[i][2])
                    ws  = {i: float(boxes[i][2] - boxes[i][0]) for i in srt}
                    placed = {}   # i -> new x1
                    for i in srt:
                        pos = bx1
                        for j in placed:
                            if min(boxes[i][3], boxes[j][3]) - max(boxes[i][1], boxes[j][1]) > 0:
                                pos = min(pos, placed[j] - ws[j])
                        placed[i] = pos
                    for i in srt:
                        boxes[i][2] = placed[i]
                        boxes[i][0] = placed[i] - ws[i]
                        print(f"  [EXPAND]   room {i} → {boxes[i].tolist()}")
                    new_x1 = min(placed[i] - ws[i] for i in srt)
                    print(f"  [EXPAND] LR.x1: {lx1} → {new_x1}")
                    boxes[lr_idx][2] = new_x1

            elif direction == 'top':
                blocking = [
                    i for i in range(K) if i != lr_idx
                    and i not in moved_rooms
                    and boxes[i][3] <= ly0
                    and min(boxes[i][2], lx1) - max(boxes[i][0], lx0) > 0
                ]
                print(f"  [EXPAND] blocking={blocking}")
                if not blocking:
                    print(f"  [EXPAND] no blockers → LR.y0 = {by0}")
                    boxes[lr_idx][1] = by0
                else:
                    # Greedy compaction: sort outermost (smallest y0) first, pack toward by0
                    srt = sorted(blocking, key=lambda i: boxes[i][1])
                    hs  = {i: float(boxes[i][3] - boxes[i][1]) for i in srt}
                    placed = {}   # i -> new y0
                    for i in srt:
                        pos = by0
                        for j in placed:
                            if min(boxes[i][2], boxes[j][2]) - max(boxes[i][0], boxes[j][0]) > 0:
                                pos = max(pos, placed[j] + hs[j])
                        placed[i] = pos
                    for i in srt:
                        boxes[i][1] = placed[i]
                        boxes[i][3] = placed[i] + hs[i]
                        print(f"  [EXPAND]   room {i} → {boxes[i].tolist()}")
                    new_y0 = max(placed[i] + hs[i] for i in srt)
                    print(f"  [EXPAND] LR.y0: {ly0} → {new_y0}")
                    boxes[lr_idx][1] = new_y0

            elif direction == 'bottom':
                blocking = [
                    i for i in range(K) if i != lr_idx
                    and i not in moved_rooms
                    and boxes[i][1] >= ly1
                    and min(boxes[i][2], lx1) - max(boxes[i][0], lx0) > 0
                ]
                print(f"  [EXPAND] blocking={blocking}")
                if not blocking:
                    print(f"  [EXPAND] no blockers → LR.y1 = {by1}")
                    boxes[lr_idx][3] = by1
                else:
                    # Greedy compaction: sort outermost (largest y1) first, pack toward by1
                    srt = sorted(blocking, key=lambda i: -boxes[i][3])
                    hs  = {i: float(boxes[i][3] - boxes[i][1]) for i in srt}
                    placed = {}   # i -> new y1
                    for i in srt:
                        pos = by1
                        for j in placed:
                            if min(boxes[i][2], boxes[j][2]) - max(boxes[i][0], boxes[j][0]) > 0:
                                pos = min(pos, placed[j] - hs[j])
                        placed[i] = pos
                    for i in srt:
                        boxes[i][3] = placed[i]
                        boxes[i][1] = placed[i] - hs[i]
                        print(f"  [EXPAND]   room {i} → {boxes[i].tolist()}")
                    new_y1 = min(placed[i] - hs[i] for i in srt)
                    print(f"  [EXPAND] LR.y1: {ly1} → {new_y1}")
                    boxes[lr_idx][3] = new_y1

        # Rooms that moved this pass become fixed for all subsequent passes
        for i in range(K):
            if i != lr_idx and not np.allclose(boxes[i], pass_start[i]):
                moved_rooms.add(i)

        if np.allclose(boxes[lr_idx], prev_lr):
            print(f"  [EXPAND] Converged after {pass_num} pass(es)")
            break
    else:
        print(f"  [EXPAND] Warning: reached max passes ({MAX_PASSES}) without convergence")

    print(f"  [EXPAND] Final LR: {boxes[lr_idx].tolist()}")
    return boxes.astype(int)


def boxes_to_boundaries(boxes):
    """
    Convert optimized [x0, y0, x1, y1] boxes to rectangular room boundary
    polygons (5-point closed rectangles), matching the format expected by
    app.decorate / add_door_window.

    Returns list of np.ndarray, each shape (5, 2).
    """
    boundaries = []
    for box in boxes:
        x0, y0, x1, y1 = box
        poly = np.array([
            [x0, y0],
            [x1, y0],
            [x1, y1],
            [x0, y1],
            [x0, y0],
        ], dtype=float)
        boundaries.append(poly)
    return boundaries

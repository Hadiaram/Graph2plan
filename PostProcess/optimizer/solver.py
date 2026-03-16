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
MIN_ROOM_PX  = 10   # fallback minimum when type is unknown

# Per-type minimum (width, height) in pixels enforced by the CP-SAT optimizer.
# Keeps rooms from being crushed to slivers while still giving the solver room
# to adjust sizes.  Structural/open types use the global fallback.
_MIN_WH = {
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


def _poly_y_range_at_x(vertices, x):
    """
    Vertical scan of a closed polygon at column x.
    Returns (y_lo, y_hi) of the polygon interior, or None if x is outside.
    vertices: list of [x, y] pairs (closed — last == first, or auto-closed).
    """
    ys = []
    n = len(vertices)
    for i in range(n - 1):
        x1, y1 = float(vertices[i][0]),   float(vertices[i][1])
        x2, y2 = float(vertices[i+1][0]), float(vertices[i+1][1])
        if x1 == x2:
            continue                          # vertical edge — skip
        if min(x1, x2) < x <= max(x1, x2):
            yi = y1 + (x - x1) * (y2 - y1) / (x2 - x1)
            ys.append(yi)
    if len(ys) < 2:
        return None
    return min(ys), max(ys)


def optimize_layout(boxes, types, edges, boundary=None, timeout=5.0):
    """
    Run the CP-SAT layout optimizer.

    Returns
    -------
    optimized_boxes : np.ndarray  shape (K, 4)  dtype int
    status          : str  'OPTIMAL' | 'FEASIBLE' | 'FAILED'
    """
    try:
        from ortools.sat.python import cp_model #type: ignore
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

    # Per-room minimum dimensions from type table
    types_arr = np.array(types).flatten().astype(int) if types is not None else np.zeros(K, dtype=int)
    min_w = [_MIN_WH.get(int(types_arr[i]), (MIN_ROOM_PX, MIN_ROOM_PX))[0] for i in range(K)]
    min_h = [_MIN_WH.get(int(types_arr[i]), (MIN_ROOM_PX, MIN_ROOM_PX))[1] for i in range(K)]

    # --- Initial positions (clamped to valid range) --------------------------
    x0 = [int(np.clip(boxes[i, 0], bx0, bx1 - min_w[i])) for i in range(K)]
    y0 = [int(np.clip(boxes[i, 1], by0, by1 - min_h[i])) for i in range(K)]
    w0 = [int(np.clip(boxes[i, 2] - boxes[i, 0], min_w[i], W)) for i in range(K)]
    h0 = [int(np.clip(boxes[i, 3] - boxes[i, 1], min_h[i], H)) for i in range(K)]

    # --- CP-SAT model --------------------------------------------------------
    model = cp_model.CpModel()

    x = [model.NewIntVar(bx0, bx1 - min_w[i], f'x_{i}') for i in range(K)]
    y = [model.NewIntVar(by0, by1 - min_h[i], f'y_{i}') for i in range(K)]
    w = [model.NewIntVar(min_w[i], W,          f'w_{i}') for i in range(K)]
    h = [model.NewIntVar(min_h[i], H,          f'h_{i}') for i in range(K)]

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


def expand_living_room(boxes, types, boundary, edges=None):
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

    WALL_TOL = 2.0
    wall_left_attached   = set(i for i in range(K) if i != lr_idx and float(boxes[i][0]) - bx0 <= WALL_TOL)
    wall_right_attached  = set(i for i in range(K) if i != lr_idx and bx1 - float(boxes[i][2]) <= WALL_TOL)
    wall_top_attached    = set(i for i in range(K) if i != lr_idx and float(boxes[i][1]) - by0 <= WALL_TOL)
    wall_bottom_attached = set(i for i in range(K) if i != lr_idx and by1 - float(boxes[i][3]) <= WALL_TOL)
    wall_attached_all    = wall_left_attached | wall_right_attached | wall_top_attached | wall_bottom_attached
    print(f"  [EXPAND] Wall-attached rooms: {sorted(wall_attached_all)}")

    # Build a set of edge-connected room pairs for quick lookup
    edge_pairs = set()
    if edges is not None:
        for e in edges:
            u, v = int(e[0]), int(e[1])
            if u != v:
                edge_pairs.add((min(u, v), max(u, v)))

    def _rigid_offsets(push_chain, axis):
        """For rooms in push_chain that share a graph edge AND significantly
        overlap, return {child_idx: (parent_idx, offset)} where offset is
        child_pos - parent_pos in the given axis (0=x, 1=y).
        Parent is always the larger-area room."""
        offsets = {}
        chain_list = list(push_chain)
        for ci in range(len(chain_list)):
            for cj in range(ci + 1, len(chain_list)):
                u, v = chain_list[ci], chain_list[cj]
                if (min(u, v), max(u, v)) not in edge_pairs:
                    continue
                ux0, uy0, ux1, uy1 = boxes[u]
                vx0, vy0, vx1, vy1 = boxes[v]
                ox = max(0.0, min(ux1, vx1) - max(ux0, vx0))
                oy = max(0.0, min(uy1, vy1) - max(uy0, vy0))
                overlap = ox * oy
                min_area = min((ux1 - ux0) * (uy1 - uy0),
                               (vx1 - vx0) * (vy1 - vy0))
                if min_area <= 0 or overlap / min_area < 0.5:
                    continue
                # Parent = larger room, child = smaller room
                u_area = (ux1 - ux0) * (uy1 - uy0)
                v_area = (vx1 - vx0) * (vy1 - vy0)
                parent, child = (u, v) if u_area >= v_area else (v, u)
                offset = float(boxes[child][axis]) - float(boxes[parent][axis])
                offsets[child] = (parent, offset)
                print(f"  [EXPAND]   rigid pair: room {parent} (parent) + "
                      f"room {child} (child), axis={axis}, offset={offset:.1f}")
        return offsets

    moved_rooms = set()   # rooms fixed after being pushed in a previous pass
    MAX_PASSES = 20
    for pass_num in range(1, MAX_PASSES + 1):
        prev_lr      = boxes[lr_idx].copy()
        pass_blocked = set()   # rooms that appeared in any blocking set this pass
        print(f"  [EXPAND] === PASS {pass_num} (fixed={sorted(moved_rooms)}) ===")

        for direction in ('left', 'right', 'top', 'bottom'):
            lx0, ly0, lx1, ly1 = (float(v) for v in boxes[lr_idx])
            print(f"  [EXPAND] --- {direction.upper()} --- LR now: [{lx0},{ly0},{lx1},{ly1}]")

            if direction == 'left':
                # Unified blocking: rooms whose left edge is left of LR's left edge (outside OR straddling)
                initial_blocking = [
                    i for i in range(K) if i != lr_idx
                    and i not in moved_rooms
                    and float(boxes[i][0]) < lx0
                    and min(float(boxes[i][3]), ly1) - max(float(boxes[i][1]), ly0) > 0
                ]
                print(f"  [EXPAND] blocking={initial_blocking}")
                if not initial_blocking:
                    print(f"  [EXPAND] no blockers → LR.x0 = {bx0}")
                    boxes[lr_idx][0] = bx0
                else:
                    push_chain = set(initial_blocking)
                    frontier = list(initial_blocking)
                    while frontier:
                        ri = frontier.pop()
                        for j in range(K):
                            if j == lr_idx or j in push_chain or j in moved_rooms or j in wall_attached_all:
                                continue
                            if (float(boxes[j][0]) < float(boxes[ri][0])
                                    and min(float(boxes[j][3]), float(boxes[ri][3])) - max(float(boxes[j][1]), float(boxes[ri][1])) > 0):
                                push_chain.add(j)
                                frontier.append(j)
                    srt = sorted(push_chain, key=lambda i: boxes[i][0])
                    ws = {i: float(boxes[i][2] - boxes[i][0]) for i in srt}
                    orig_lo = {i: float(boxes[i][0]) for i in srt}
                    rigid = _rigid_offsets(push_chain, axis=0)
                    placed = {}
                    for i in srt:
                        if i in rigid:
                            parent, offset = rigid[i]
                            if parent in placed:
                                placed[i] = placed[parent] + offset
                                continue
                        pos = bx0
                        for j in placed:
                            if min(float(boxes[i][3]), float(boxes[j][3])) - max(float(boxes[i][1]), float(boxes[j][1])) > 0:
                                pos = max(pos, placed[j] + ws[j])
                        for j in range(K):
                            if j == lr_idx or j in push_chain:
                                continue
                            if (float(boxes[j][0]) < orig_lo[i]
                                    and min(float(boxes[i][3]), float(boxes[j][3])) - max(float(boxes[i][1]), float(boxes[j][1])) > 0):
                                pos = max(pos, float(boxes[j][2]))
                        placed[i] = pos
                    for i in srt:
                        boxes[i][0] = placed[i]
                        boxes[i][2] = placed[i] + ws[i]
                        print(f"  [EXPAND]   room {i} → {boxes[i].tolist()}")
                    new_x0 = max(placed[i] + ws[i] for i in initial_blocking)
                    print(f"  [EXPAND] LR.x0: {lx0} → {new_x0}")
                    boxes[lr_idx][0] = new_x0
                    pass_blocked.update(push_chain)

            elif direction == 'right':
                # Unified blocking: rooms whose right edge extends past LR's right edge (outside OR straddling)
                initial_blocking = [
                    i for i in range(K) if i != lr_idx
                    and i not in moved_rooms
                    and float(boxes[i][2]) > lx1
                    and min(float(boxes[i][3]), ly1) - max(float(boxes[i][1]), ly0) > 0
                ]
                print(f"  [EXPAND] blocking={initial_blocking}")
                if not initial_blocking:
                    print(f"  [EXPAND] no blockers → LR.x1 = {bx1}")
                    boxes[lr_idx][2] = bx1
                else:
                    push_chain = set(initial_blocking)
                    frontier = list(initial_blocking)
                    while frontier:
                        ri = frontier.pop()
                        for j in range(K):
                            if j == lr_idx or j in push_chain or j in moved_rooms or j in wall_attached_all:
                                continue
                            if (float(boxes[j][0]) > float(boxes[ri][0])
                                    and min(float(boxes[j][3]), float(boxes[ri][3])) - max(float(boxes[j][1]), float(boxes[ri][1])) > 0):
                                push_chain.add(j)
                                frontier.append(j)
                    srt = sorted(push_chain, key=lambda i: -boxes[i][2])
                    ws = {i: float(boxes[i][2] - boxes[i][0]) for i in srt}
                    orig_lo = {i: float(boxes[i][0]) for i in srt}
                    # For RIGHT, placed stores x1; rigid offset on x1
                    rigid = {c: (p, float(boxes[c][2]) - float(boxes[p][2]))
                             for c, (p, _) in _rigid_offsets(push_chain, axis=0).items()}
                    placed = {}
                    for i in srt:
                        if i in rigid:
                            parent, offset = rigid[i]
                            if parent in placed:
                                placed[i] = placed[parent] + offset
                                continue
                        pos = bx1
                        for j in placed:
                            if min(float(boxes[i][3]), float(boxes[j][3])) - max(float(boxes[i][1]), float(boxes[j][1])) > 0:
                                pos = min(pos, placed[j] - ws[j])
                        for j in range(K):
                            if j == lr_idx or j in push_chain:
                                continue
                            if (float(boxes[j][0]) > orig_lo[i]
                                    and min(float(boxes[i][3]), float(boxes[j][3])) - max(float(boxes[i][1]), float(boxes[j][1])) > 0):
                                pos = min(pos, float(boxes[j][0]))
                        placed[i] = pos
                    for i in srt:
                        boxes[i][2] = placed[i]
                        boxes[i][0] = placed[i] - ws[i]
                        print(f"  [EXPAND]   room {i} → {boxes[i].tolist()}")
                    new_x1 = min(placed[i] - ws[i] for i in initial_blocking)
                    print(f"  [EXPAND] LR.x1: {lx1} → {new_x1}")
                    boxes[lr_idx][2] = new_x1
                    pass_blocked.update(push_chain)

            elif direction == 'top':
                # Unified blocking: rooms whose top edge is above LR's top edge (outside OR straddling)
                initial_blocking = [
                    i for i in range(K) if i != lr_idx
                    and i not in moved_rooms
                    and float(boxes[i][1]) < ly0
                    and min(float(boxes[i][2]), lx1) - max(float(boxes[i][0]), lx0) > 0
                ]
                print(f"  [EXPAND] blocking={initial_blocking}")
                if not initial_blocking:
                    print(f"  [EXPAND] no blockers → LR.y0 = {by0}")
                    boxes[lr_idx][1] = by0
                else:
                    push_chain = set(initial_blocking)
                    frontier = list(initial_blocking)
                    while frontier:
                        ri = frontier.pop()
                        for j in range(K):
                            if j == lr_idx or j in push_chain or j in moved_rooms or j in wall_attached_all:
                                continue
                            if (float(boxes[j][1]) < float(boxes[ri][1])
                                    and min(float(boxes[j][2]), float(boxes[ri][2])) - max(float(boxes[j][0]), float(boxes[ri][0])) > 0):
                                push_chain.add(j)
                                frontier.append(j)
                    srt = sorted(push_chain, key=lambda i: boxes[i][1])
                    hs = {i: float(boxes[i][3] - boxes[i][1]) for i in srt}
                    orig_lo = {i: float(boxes[i][1]) for i in srt}
                    rigid = _rigid_offsets(push_chain, axis=1)
                    placed = {}
                    for i in srt:
                        if i in rigid:
                            parent, offset = rigid[i]
                            if parent in placed:
                                placed[i] = placed[parent] + offset
                                continue
                        pos = by0
                        for j in placed:
                            if min(float(boxes[i][2]), float(boxes[j][2])) - max(float(boxes[i][0]), float(boxes[j][0])) > 0:
                                pos = max(pos, placed[j] + hs[j])
                        for j in range(K):
                            if j == lr_idx or j in push_chain:
                                continue
                            if (float(boxes[j][1]) < orig_lo[i]
                                    and min(float(boxes[i][2]), float(boxes[j][2])) - max(float(boxes[i][0]), float(boxes[j][0])) > 0):
                                pos = max(pos, float(boxes[j][3]))
                        placed[i] = pos
                    for i in srt:
                        boxes[i][1] = placed[i]
                        boxes[i][3] = placed[i] + hs[i]
                        print(f"  [EXPAND]   room {i} → {boxes[i].tolist()}")
                    new_y0 = max(placed[i] + hs[i] for i in initial_blocking)
                    print(f"  [EXPAND] LR.y0: {ly0} → {new_y0}")
                    boxes[lr_idx][1] = new_y0
                    pass_blocked.update(push_chain)

            elif direction == 'bottom':
                # Unified blocking: rooms whose bottom edge extends past LR's bottom edge (outside OR straddling)
                initial_blocking = [
                    i for i in range(K) if i != lr_idx
                    and i not in moved_rooms
                    and float(boxes[i][3]) > ly1
                    and min(float(boxes[i][2]), lx1) - max(float(boxes[i][0]), lx0) > 0
                ]
                print(f"  [EXPAND] blocking={initial_blocking}")
                if not initial_blocking:
                    print(f"  [EXPAND] no blockers → LR.y1 = {by1}")
                    boxes[lr_idx][3] = by1
                else:
                    push_chain = set(initial_blocking)
                    frontier = list(initial_blocking)
                    while frontier:
                        ri = frontier.pop()
                        for j in range(K):
                            if j == lr_idx or j in push_chain or j in moved_rooms or j in wall_attached_all:
                                continue
                            if (float(boxes[j][1]) > float(boxes[ri][1])
                                    and min(float(boxes[j][2]), float(boxes[ri][2])) - max(float(boxes[j][0]), float(boxes[ri][0])) > 0):
                                push_chain.add(j)
                                frontier.append(j)
                    srt = sorted(push_chain, key=lambda i: -boxes[i][3])
                    hs = {i: float(boxes[i][3] - boxes[i][1]) for i in srt}
                    orig_lo = {i: float(boxes[i][1]) for i in srt}
                    # For BOTTOM, placed stores y1; rigid offset on y1
                    rigid = {c: (p, float(boxes[c][3]) - float(boxes[p][3]))
                             for c, (p, _) in _rigid_offsets(push_chain, axis=1).items()}
                    placed = {}
                    for i in srt:
                        if i in rigid:
                            parent, offset = rigid[i]
                            if parent in placed:
                                placed[i] = placed[parent] + offset
                                continue
                        pos = by1
                        for j in placed:
                            if min(float(boxes[i][2]), float(boxes[j][2])) - max(float(boxes[i][0]), float(boxes[j][0])) > 0:
                                pos = min(pos, placed[j] - hs[j])
                        for j in range(K):
                            if j == lr_idx or j in push_chain:
                                continue
                            if (float(boxes[j][1]) > orig_lo[i]
                                    and min(float(boxes[i][2]), float(boxes[j][2])) - max(float(boxes[i][0]), float(boxes[j][0])) > 0):
                                pos = min(pos, float(boxes[j][1]))
                        placed[i] = pos
                    for i in srt:
                        boxes[i][3] = placed[i]
                        boxes[i][1] = placed[i] - hs[i]
                        print(f"  [EXPAND]   room {i} → {boxes[i].tolist()}")
                    new_y1 = min(placed[i] - hs[i] for i in initial_blocking)
                    print(f"  [EXPAND] LR.y1: {ly1} → {new_y1}")
                    boxes[lr_idx][3] = new_y1
                    pass_blocked.update(push_chain)

        # Every room that was a blocker this pass becomes fixed for subsequent passes,
        # whether it physically moved or was already at the boundary wall.
        # Wall-attached rooms are never promoted — they remain permanent blockers forever.
        moved_rooms |= pass_blocked
        moved_rooms -= wall_attached_all

        if np.allclose(boxes[lr_idx], prev_lr):
            print(f"  [EXPAND] Converged after {pass_num} pass(es)")
            break
    else:
        print(f"  [EXPAND] Warning: reached max passes ({MAX_PASSES}) without convergence")

    print(f"  [EXPAND] Final LR: {boxes[lr_idx].tolist()}")
    return boxes.astype(int)


def snap_single_edge_rooms(boxes, types, boundary, snap_tol=2.0, max_gap=30.0):
    """
    Post-expansion pass: any non-LR room connected on fewer than 2 sides gets
    snapped to its nearest available contact point (wall or adjacent room edge),
    closing small gaps and producing a tighter layout.

    A side is "connected" if its edge is within snap_tol of a boundary wall or
    another room's facing edge AND there is positive overlap in the perpendicular
    axis.  Only gaps <= max_gap are eligible.  Moves are validated to avoid
    creating overlaps.  Repeats until stable (max 10 iterations).
    """
    boxes = np.array(boxes, dtype=float)
    types = np.array(types, dtype=int)
    K = len(boxes)

    bnd  = np.array(boundary)
    bx0  = float(np.min(bnd[:, 0]))
    by0  = float(np.min(bnd[:, 1]))
    bx1  = float(np.max(bnd[:, 0]))
    by1  = float(np.max(bnd[:, 1]))

    # Build closed polygon vertex list for actual wall position queries.
    # This is critical for non-rectangular (e.g., L-shaped) boundaries:
    # bx0/bx1/by0/by1 are the bounding box extremes, but the actual wall
    # at any given y-level may be much closer to the rooms.
    poly_verts = bnd[:, :2].tolist()
    if poly_verts[0] != poly_verts[-1]:
        poly_verts.append(poly_verts[0])  # close the polygon

    def actual_walls(x0, y0, x1, y1):
        """Return actual (left, right, top, bottom) wall at this room's midpoint."""
        cy = (y0 + y1) / 2.0
        cx = (x0 + x1) / 2.0
        xr = _poly_x_range_at_y(poly_verts, cy)
        yr = _poly_y_range_at_x(poly_verts, cx)
        lw = xr[0] if xr else bx0
        rw = xr[1] if xr else bx1
        tw = yr[0] if yr else by0
        bw = yr[1] if yr else by1
        return lw, rw, tw, bw

    LR_TYPE = 0

    _ROOM_NAMES = {0: 'LivingRoom', 1: 'MasterRoom', 2: 'Kitchen', 3: 'Bathroom',
                   4: 'DiningRoom', 5: 'ChildRoom', 6: 'StudyRoom', 7: 'SecondRoom',
                   8: 'GuestRoom', 9: 'Balcony', 10: 'Entrance', 11: 'Storage',
                   12: 'Wall', 13: 'Exterior', 14: 'LivingDining', 15: 'FrontDoor'}

    # Types excluded from both connectivity detection and snap targets.
    # Types excluded from connectivity checks: LR (0) creates artificial adjacencies
    # after expansion; Entrance (10) and unknown type 15 are non-habitable elements.
    _EXCL_CONNECTIVITY = {0, 10, 15}

    # Types excluded from being snap targets: Entrance (10), Wall (12), Exterior (13),
    # unknown type 15.  LR (0) is intentionally NOT excluded — rooms should be able
    # to snap toward LR edges to close small gaps.
    _EXCL_TARGETS = {10, 12, 13, 15}

    def room_label(i):
        return f"room[{i}] {_ROOM_NAMES.get(int(types[i]), str(types[i]))}"

    def yov(i, j):
        return (min(float(boxes[i][3]), float(boxes[j][3]))
                - max(float(boxes[i][1]), float(boxes[j][1]))) > 0

    def xov(i, j):
        return (min(float(boxes[i][2]), float(boxes[j][2]))
                - max(float(boxes[i][0]), float(boxes[j][0]))) > 0

    def connected_sides(i):
        x0, y0, x1, y1 = (float(boxes[i][k]) for k in range(4))
        lw, rw, tw, bw = actual_walls(x0, y0, x1, y1)
        cl = abs(x0 - lw) <= snap_tol or any(
            abs(x0 - float(boxes[j][2])) <= snap_tol and yov(i, j)
            for j in range(K) if j != i and int(types[j]) not in _EXCL_CONNECTIVITY)
        cr = abs(x1 - rw) <= snap_tol or any(
            abs(x1 - float(boxes[j][0])) <= snap_tol and yov(i, j)
            for j in range(K) if j != i and int(types[j]) not in _EXCL_CONNECTIVITY)
        ct = abs(y0 - tw) <= snap_tol or any(
            abs(y0 - float(boxes[j][3])) <= snap_tol and xov(i, j)
            for j in range(K) if j != i and int(types[j]) not in _EXCL_CONNECTIVITY)
        cb = abs(y1 - bw) <= snap_tol or any(
            abs(y1 - float(boxes[j][1])) <= snap_tol and xov(i, j)
            for j in range(K) if j != i and int(types[j]) not in _EXCL_CONNECTIVITY)
        return cl, cr, ct, cb

    def no_overlap(i, nx0, ny0, nx1, ny1):
        # Reject if outside the bounding box
        if nx0 < bx0 - 1.0 or nx1 > bx1 + 1.0 or ny0 < by0 - 1.0 or ny1 > by1 + 1.0:
            return False
        # Reject if outside the actual polygon boundary at the new room's midpoint.
        # This catches cases where bx1/by1 extends beyond the actual L-shaped wall.
        nxr = _poly_x_range_at_y(poly_verts, (ny0 + ny1) / 2.0)
        nyr = _poly_y_range_at_x(poly_verts, (nx0 + nx1) / 2.0)
        if nxr and (nx0 < nxr[0] - 1.0 or nx1 > nxr[1] + 1.0):
            return False
        if nyr and (ny0 < nyr[0] - 1.0 or ny1 > nyr[1] + 1.0):
            return False
        for j in range(K):
            if j == i:
                continue
            xo = min(nx1, float(boxes[j][2])) - max(nx0, float(boxes[j][0]))
            yo = min(ny1, float(boxes[j][3])) - max(ny0, float(boxes[j][1]))
            if xo > 1.0 and yo > 1.0:
                return False
        return True

    for iteration in range(10):
        print(f"[SNAP] === Iteration {iteration + 1} ===")
        any_moved = False
        for i in range(K):
            if int(types[i]) == LR_TYPE:
                continue
            x0, y0, x1, y1 = (float(boxes[i][k]) for k in range(4))
            w, h = x1 - x0, y1 - y0
            cl, cr, ct, cb = connected_sides(i)
            sides_count = sum([cl, cr, ct, cb])
            print(f"  [SNAP] {room_label(i)} pos=({x0:.0f},{y0:.0f},{x1:.0f},{y1:.0f}) "
                  f"connected=[L:{cl} R:{cr} T:{ct} B:{cb}] ({sides_count} sides)")
            if sides_count >= 2:
                print(f"  [SNAP]   → SKIP (already connected on {sides_count} sides)")
                continue

            # Collect candidate snaps: (gap, dx, dy, label)
            cands = []
            rejected = []

            # Compute actual wall positions using the polygon (not just bounding box).
            # For L-shaped boundaries, bx1/by1 may extend beyond the actual wall at
            # this room's y/x level — using actual walls prevents snapping into voids.
            aw_left, aw_right, aw_top, aw_bot = actual_walls(x0, y0, x1, y1)

            # Wall-anchor preservation: if the room already touches a boundary wall,
            # don't snap in the direction that would pull it away from that wall.
            # This prevents oscillation when a gap to LR exists on the opposite side.
            wall_left   = abs(x0 - aw_left)  <= snap_tol
            wall_right  = abs(x1 - aw_right) <= snap_tol
            wall_top    = abs(y0 - aw_top)   <= snap_tol
            wall_bottom = abs(y1 - aw_bot)   <= snap_tol

            if not cl and not wall_right:
                gap = x0 - aw_left
                if gap <= 0:
                    print(f"  [SNAP]   LEFT: gap={gap:.1f} (at or past wall, skipping)")
                elif gap > max_gap:
                    print(f"  [SNAP]   LEFT wall: gap={gap:.1f} > max_gap={max_gap}, skipped")
                elif not no_overlap(i, aw_left, y0, aw_left + w, y1):
                    rejected.append(f"left-wall (gap={gap:.1f}, overlap blocked)")
                else:
                    cands.append((gap, aw_left - x0, 0.0, 'left-wall'))
                for j in range(K):
                    if j == i or int(types[j]) in _EXCL_TARGETS or int(types[j]) > 14:
                        continue
                    jx1 = float(boxes[j][2])
                    if jx1 < x0:
                        gap = x0 - jx1
                        if not yov(i, j):
                            pass  # no y-overlap, ignore silently
                        elif gap > max_gap:
                            print(f"  [SNAP]   LEFT {room_label(j)}: gap={gap:.1f} > max_gap={max_gap}, skipped")
                        elif not no_overlap(i, jx1, y0, jx1 + w, y1):
                            rejected.append(f"{room_label(j)} left (gap={gap:.1f}, overlap blocked)")
                        else:
                            cands.append((gap, jx1 - x0, 0.0, room_label(j)))

            if not cr and not wall_left:
                gap = aw_right - x1
                if gap <= 0:
                    print(f"  [SNAP]   RIGHT: gap={gap:.1f} (at or past wall, skipping)")
                elif gap > max_gap:
                    print(f"  [SNAP]   RIGHT wall: gap={gap:.1f} > max_gap={max_gap}, skipped")
                elif not no_overlap(i, aw_right - w, y0, aw_right, y1):
                    rejected.append(f"right-wall (gap={gap:.1f}, overlap blocked)")
                else:
                    cands.append((gap, aw_right - x1, 0.0, 'right-wall'))
                for j in range(K):
                    if j == i or int(types[j]) in _EXCL_TARGETS or int(types[j]) > 14:
                        continue
                    jx0 = float(boxes[j][0])
                    if jx0 > x1:
                        gap = jx0 - x1
                        if not yov(i, j):
                            pass
                        elif gap > max_gap:
                            print(f"  [SNAP]   RIGHT {room_label(j)}: gap={gap:.1f} > max_gap={max_gap}, skipped")
                        elif not no_overlap(i, jx0 - w, y0, jx0, y1):
                            rejected.append(f"{room_label(j)} right (gap={gap:.1f}, overlap blocked)")
                        else:
                            cands.append((gap, jx0 - x1, 0.0, room_label(j)))

            if not ct and not wall_bottom:
                gap = y0 - aw_top
                if gap <= 0:
                    print(f"  [SNAP]   TOP: gap={gap:.1f} (at or past wall, skipping)")
                elif gap > max_gap:
                    print(f"  [SNAP]   TOP wall: gap={gap:.1f} > max_gap={max_gap}, skipped")
                elif not no_overlap(i, x0, aw_top, x1, aw_top + h):
                    rejected.append(f"top-wall (gap={gap:.1f}, overlap blocked)")
                else:
                    cands.append((gap, 0.0, aw_top - y0, 'top-wall'))
                for j in range(K):
                    if j == i or int(types[j]) in _EXCL_TARGETS or int(types[j]) > 14:
                        continue
                    jy1 = float(boxes[j][3])
                    if jy1 < y0:
                        gap = y0 - jy1
                        if not xov(i, j):
                            pass
                        elif gap > max_gap:
                            print(f"  [SNAP]   TOP {room_label(j)}: gap={gap:.1f} > max_gap={max_gap}, skipped")
                        elif not no_overlap(i, x0, jy1, x1, jy1 + h):
                            rejected.append(f"{room_label(j)} top (gap={gap:.1f}, overlap blocked)")
                        else:
                            cands.append((gap, 0.0, jy1 - y0, room_label(j)))

            if not cb and not wall_top:
                gap = aw_bot - y1
                if gap <= 0:
                    print(f"  [SNAP]   BOT: gap={gap:.1f} (at or past wall, skipping)")
                elif gap > max_gap:
                    print(f"  [SNAP]   BOT wall: gap={gap:.1f} > max_gap={max_gap}, skipped")
                elif not no_overlap(i, x0, aw_bot - h, x1, aw_bot):
                    rejected.append(f"bot-wall (gap={gap:.1f}, overlap blocked)")
                else:
                    cands.append((gap, 0.0, aw_bot - y1, 'bot-wall'))
                for j in range(K):
                    if j == i or int(types[j]) in _EXCL_TARGETS or int(types[j]) > 14:
                        continue
                    jy0 = float(boxes[j][1])
                    if jy0 > y1:
                        gap = jy0 - y1
                        if not xov(i, j):
                            pass
                        elif gap > max_gap:
                            print(f"  [SNAP]   BOT {room_label(j)}: gap={gap:.1f} > max_gap={max_gap}, skipped")
                        elif not no_overlap(i, x0, jy0 - h, x1, jy0):
                            rejected.append(f"{room_label(j)} bot (gap={gap:.1f}, overlap blocked)")
                        else:
                            cands.append((gap, 0.0, jy0 - y1, room_label(j)))

            if rejected:
                print(f"  [SNAP]   Rejected candidates: {rejected}")

            if not cands:
                print(f"  [SNAP]   → NO valid snap candidates found")
                continue

            print(f"  [SNAP]   Candidates: {[(t, f'{g:.1f}px') for g, _, _, t in cands]}")
            gap, dx, dy, target = min(cands, key=lambda c: c[0])
            print(f"  [SNAP]   → SNAP to {target} (gap={gap:.1f}px, move=({dx:.1f},{dy:.1f}))")
            boxes[i][0] += dx
            boxes[i][2] += dx
            boxes[i][1] += dy
            boxes[i][3] += dy
            any_moved = True

        if not any_moved:
            print(f"[SNAP] Stable after {iteration + 1} iteration(s)")
            break

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


def align_walls(boxes, types, tol=6.0):
    """Snap wall coordinates that are within `tol` pixels of each other to a
    common value, eliminating sub-pixel gaps that fill_wall_gaps would otherwise
    try to close (and potentially overshoot, causing 1-pixel overlaps).

    Only solid non-LR rooms are considered (types not in {0,12,13,14,15}).
    For each axis (x and y) all left/right or top/bottom wall coordinates are
    collected, clustered by proximity, and each cluster is snapped to its
    integer median value.
    """
    EXCL = {0, 12, 13, 14, 15}
    boxes = [list(b) for b in boxes]
    K = len(boxes)

    def _snap_axis(coords_by_room, dim):
        """coords_by_room: list of (room_idx, coord_slot, value).
           dim unused — kept for clarity."""
        vals = sorted(set(v for _, _, v in coords_by_room))
        if not vals:
            return
        # Build clusters: greedily merge values within tol of the previous
        clusters = []
        cur = [vals[0]]
        for v in vals[1:]:
            if v - cur[-1] <= tol:
                cur.append(v)
            else:
                clusters.append(cur)
                cur = [v]
        clusters.append(cur)

        # Map each original value to its cluster median (rounded to int)
        snap_map = {}
        for cl in clusters:
            if len(cl) < 2:
                continue                      # lone value — nothing to snap
            target = int(round(sorted(cl)[len(cl) // 2]))
            for v in cl:
                snap_map[v] = target

        for room_idx, slot, val in coords_by_room:
            if val in snap_map:
                boxes[room_idx][slot] = snap_map[val]

    # Collect x-coords (slots 0 and 2) and y-coords (slots 1 and 3)
    x_coords = []
    y_coords = []
    for i in range(K):
        if int(types[i]) in EXCL:
            continue
        x_coords.append((i, 0, boxes[i][0]))
        x_coords.append((i, 2, boxes[i][2]))
        y_coords.append((i, 1, boxes[i][1]))
        y_coords.append((i, 3, boxes[i][3]))

    _snap_axis(x_coords, 'x')
    _snap_axis(y_coords, 'y')

    snapped = sum(
        1 for i in range(K) if int(types[i]) not in EXCL
        and any(boxes[i][s] != list(boxes[i])[s] for s in range(4))
    )
    print(f"[ALIGN WALLS] Done — tolerance={tol}px")
    return [np.array(b) for b in boxes]


def adjust_boundary_to_rooms(boxes, types, boundary):
    """
    After post-processing heuristics, some non-LR rooms may extend beyond the
    original boundary polygon.  This function shifts each wall of the boundary
    outward (never inward) to encompass any such rooms.

    LivingRoom (type 0) is excluded — its bounding box intentionally extends
    past the house boundary and must not drive expansion.

    Unlike a simple bounding-box check, this uses the actual boundary polygon
    geometry (_poly_x_range_at_y / _poly_y_range_at_x) to detect which wall
    each room is outside of.  This correctly handles L-shaped and other
    non-rectangular boundaries: a room in the notch corner of an L-shape is
    detected as outside even though it sits within the overall bounding box.

    For each non-LR room, the midpoint of each side is scanned against the
    polygon to find the actual wall position at that level.  If the room
    extends past that wall, the wall's required new position is recorded.
    All boundary vertices sharing that wall coordinate (within EPS tolerance)
    are then shifted to the new position.

    Parameters
    ----------
    boxes    : array-like (K, 4)  [x0, y0, x1, y1]
    types    : array-like (K,)    room type indices
    boundary : array-like (N, 4)  [x, y, dir, isNew]

    Returns
    -------
    new_boundary : np.ndarray  (N, 4)
    """
    LR_TYPE = 0
    boxes = np.array(boxes, dtype=float)
    types = np.array(types, dtype=int).flatten()
    bnd   = np.array(boundary, dtype=float)

    poly_verts = bnd[:, :2].tolist()
    if poly_verts[0] != poly_verts[-1]:
        poly_verts.append(poly_verts[0])

    solid_mask = types != LR_TYPE
    if not np.any(solid_mask):
        print("[ADJUST BOUNDARY] No non-LR rooms found — boundary unchanged.")
        return bnd

    EPS = 0.5

    # x_expansions: old_x_wall → required_new_x  (vertical walls)
    # y_expansions: old_y_wall → required_new_y  (horizontal walls)
    x_expansions = {}
    y_expansions = {}

    for i in np.where(solid_mask)[0]:
        rx0, ry0, rx1, ry1 = (float(boxes[i][k]) for k in range(4))
        cy = (ry0 + ry1) / 2.0
        cx = (rx0 + rx1) / 2.0

        # --- LEFT / RIGHT: scan the polygon at the room's y-midpoint ---
        xr = _poly_x_range_at_y(poly_verts, cy)
        if xr:
            left_wall, right_wall = xr
            if rx0 < left_wall - EPS:
                print(f"[ADJUST BOUNDARY] Room {i}: LEFT extends to {rx0:.1f}, wall at {left_wall:.1f}")
                if left_wall not in x_expansions or rx0 < x_expansions[left_wall]:
                    x_expansions[left_wall] = rx0
            if rx1 > right_wall + EPS:
                print(f"[ADJUST BOUNDARY] Room {i}: RIGHT extends to {rx1:.1f}, wall at {right_wall:.1f}")
                if right_wall not in x_expansions or rx1 > x_expansions[right_wall]:
                    x_expansions[right_wall] = rx1

        # --- TOP / BOTTOM: scan the polygon at the room's x-midpoint ---
        yr = _poly_y_range_at_x(poly_verts, cx)
        if yr:
            top_wall, bot_wall = yr
            if ry0 < top_wall - EPS:
                print(f"[ADJUST BOUNDARY] Room {i}: TOP extends to {ry0:.1f}, wall at {top_wall:.1f}")
                if top_wall not in y_expansions or ry0 < y_expansions[top_wall]:
                    y_expansions[top_wall] = ry0
            if ry1 > bot_wall + EPS:
                print(f"[ADJUST BOUNDARY] Room {i}: BOTTOM extends to {ry1:.1f}, wall at {bot_wall:.1f}")
                if bot_wall not in y_expansions or ry1 > y_expansions[bot_wall]:
                    y_expansions[bot_wall] = ry1

    if not x_expansions and not y_expansions:
        print("[ADJUST BOUNDARY] All rooms are inside the boundary — no adjustment needed.")
        return bnd

    # Apply expansions: shift every boundary vertex that sits on an old wall
    # coordinate to the new required position.
    new_bnd = bnd.copy()
    for idx in range(len(bnd)):
        vx, vy = float(bnd[idx, 0]), float(bnd[idx, 1])
        for old_x, new_x in x_expansions.items():
            if abs(vx - old_x) < EPS:
                new_bnd[idx, 0] = new_x
                break
        for old_y, new_y in y_expansions.items():
            if abs(vy - old_y) < EPS:
                new_bnd[idx, 1] = new_y
                break

    print(f"[ADJUST BOUNDARY] Applied {len(x_expansions)} x-wall and "
          f"{len(y_expansions)} y-wall expansions.")
    return new_bnd


def fill_wall_gaps(boxes, types, boundary, gap_threshold=10.0, room_gap_threshold=5.0):
    """Expand rooms to fill small gaps between their edges and the exterior boundary wall.

    For each room and each of its 4 sides, if the gap to the **actual** boundary wall
    facing that side (queried at the room's midpoint via polygon scan) is <= gap_threshold
    AND no other room occupies that gap region, expand the room's edge to that wall.

    Using the polygon scan rather than the global bounding box correctly handles
    non-rectangular boundaries (L-shapes, T-shapes, etc.) where the global min/max
    x or y is not the wall that faces a particular room.
    """
    boxes = [[float(b[0]), float(b[1]), float(b[2]), float(b[3])] for b in boxes]
    K = len(boxes)
    boundary_arr = np.array(boundary)

    # Global extents — only used for logging and fallback
    bx0_g = float(np.min(boundary_arr[:, 0]))
    bx1_g = float(np.max(boundary_arr[:, 0]))
    by0_g = float(np.min(boundary_arr[:, 1]))
    by1_g = float(np.max(boundary_arr[:, 1]))

    # Build closed polygon vertex list from boundary (x, y columns only)
    poly_verts = boundary_arr[:, :2].tolist()
    if poly_verts[0] != poly_verts[-1]:
        poly_verts.append(poly_verts[0])

    _ROOM_NAMES = {
        0: 'LivingRoom', 1: 'MasterRoom', 2: 'Kitchen', 3: 'Bathroom',
        4: 'DiningRoom', 5: 'ChildRoom', 6: 'StudyRoom', 7: 'SecondRoom',
        8: 'GuestRoom', 9: 'Balcony', 10: 'Entrance', 11: 'Storage',
        12: 'Wall-in', 13: 'External', 14: 'ExteriorWall', 15: 'FrontDoor',
        16: 'InteriorWall', 17: 'InteriorDoor',
    }

    def room_label(i):
        t = int(types[i])
        name = _ROOM_NAMES.get(t, f'type{t}')
        return f"room[{i}] {name}"

    def find_blocker(i, check_fn):
        for j in range(K):
            # LivingRoom (type 0) is excluded — after expansion it touches walls
            # and should not prevent other rooms from filling small gaps
            if j != i and int(types[j]) != 0 and check_fn(j):
                return j
        return None

    print(f"[FILL GAPS] Boundary bbox: x=[{bx0_g:.1f}, {bx1_g:.1f}] y=[{by0_g:.1f}, {by1_g:.1f}], threshold={gap_threshold}px")

    for i in range(K):
        x0, y0, x1, y1 = boxes[i]
        name = room_label(i)
        cy = (y0 + y1) / 2.0   # room y-midpoint — used to query left/right walls
        cx = (x0 + x1) / 2.0   # room x-midpoint — used to query top/bottom walls

        # Actual left and right walls at the room's y-midpoint
        x_range = _poly_x_range_at_y(poly_verts, cy)
        wall_left  = x_range[0] if x_range else bx0_g
        wall_right = x_range[1] if x_range else bx1_g

        # Actual top and bottom walls at the room's x-midpoint
        y_range = _poly_y_range_at_x(poly_verts, cx)
        wall_top    = y_range[0] if y_range else by0_g
        wall_bottom = y_range[1] if y_range else by1_g

        # LEFT
        gap = x0 - wall_left
        if gap <= 0:
            pass  # already at or past the wall
        elif gap > gap_threshold:
            print(f"[FILL GAPS]   {name} LEFT:   gap={gap:.2f}px (wall={wall_left:.1f}) > threshold, skipping")
        else:
            blocker = find_blocker(i, lambda j: (
                min(y1, boxes[j][3]) - max(y0, boxes[j][1]) > 1.0
                and boxes[j][0] > wall_left + 0.5   # not already at the wall
                and boxes[j][0] < x0
                and boxes[j][2] > wall_left
            ))
            if blocker is not None:
                print(f"[FILL GAPS]   {name} LEFT:   gap={gap:.2f}px (wall={wall_left:.1f}), blocked by {room_label(blocker)}")
            else:
                print(f"[FILL GAPS]   {name} LEFT:   gap={gap:.2f}px (wall={wall_left:.1f}) → expanding to wall")
                boxes[i][0] = wall_left
                x0 = wall_left

        # RIGHT
        gap = wall_right - x1
        if gap <= 0:
            pass
        elif gap > gap_threshold:
            print(f"[FILL GAPS]   {name} RIGHT:  gap={gap:.2f}px (wall={wall_right:.1f}) > threshold, skipping")
        else:
            blocker = find_blocker(i, lambda j: (
                min(y1, boxes[j][3]) - max(y0, boxes[j][1]) > 1.0
                and boxes[j][2] < wall_right - 0.5   # not already at the wall
                and boxes[j][2] > x1
                and boxes[j][0] < wall_right
            ))
            if blocker is not None:
                print(f"[FILL GAPS]   {name} RIGHT:  gap={gap:.2f}px (wall={wall_right:.1f}), blocked by {room_label(blocker)}")
            else:
                print(f"[FILL GAPS]   {name} RIGHT:  gap={gap:.2f}px (wall={wall_right:.1f}) → expanding to wall")
                boxes[i][2] = wall_right
                x1 = wall_right

        # TOP
        gap = y0 - wall_top
        if gap <= 0:
            pass
        elif gap > gap_threshold:
            print(f"[FILL GAPS]   {name} TOP:    gap={gap:.2f}px (wall={wall_top:.1f}) > threshold, skipping")
        else:
            blocker = find_blocker(i, lambda j: (
                min(x1, boxes[j][2]) - max(x0, boxes[j][0]) > 1.0
                and boxes[j][1] > wall_top + 0.5   # not already at the wall
                and boxes[j][1] < y0
                and boxes[j][3] > wall_top
            ))
            if blocker is not None:
                print(f"[FILL GAPS]   {name} TOP:    gap={gap:.2f}px (wall={wall_top:.1f}), blocked by {room_label(blocker)}")
            else:
                print(f"[FILL GAPS]   {name} TOP:    gap={gap:.2f}px (wall={wall_top:.1f}) → expanding to wall")
                boxes[i][1] = wall_top
                y0 = wall_top

        # BOTTOM
        gap = wall_bottom - y1
        if gap <= 0:
            pass
        elif gap > gap_threshold:
            print(f"[FILL GAPS]   {name} BOTTOM: gap={gap:.2f}px (wall={wall_bottom:.1f}) > threshold, skipping")
        else:
            blocker = find_blocker(i, lambda j: (
                min(x1, boxes[j][2]) - max(x0, boxes[j][0]) > 1.0
                and boxes[j][3] < wall_bottom - 0.5   # not already at the wall
                and boxes[j][3] > y1
                and boxes[j][1] < wall_bottom
            ))
            if blocker is not None:
                print(f"[FILL GAPS]   {name} BOTTOM: gap={gap:.2f}px (wall={wall_bottom:.1f}), blocked by {room_label(blocker)}")
            else:
                print(f"[FILL GAPS]   {name} BOTTOM: gap={gap:.2f}px (wall={wall_bottom:.1f}) → expanding to wall")
                boxes[i][3] = wall_bottom
                y1 = wall_bottom

    # Second pass: room-to-room gaps
    # For each room and each direction, find the nearest neighbouring room
    # (with perpendicular overlap) and close the gap if it is <= gap_threshold.
    # LivingRoom and structural elements are excluded as both source and target.
    EXCL = {0, 12, 13, 14, 15}
    print(f"[FILL GAPS] Room-to-room pass (threshold={room_gap_threshold}px)...")

    for i in range(K):
        if int(types[i]) in EXCL:
            continue
        x0, y0, x1, y1 = boxes[i]
        name = room_label(i)

        # LEFT — find nearest room to the left with y-overlap
        best_j, best_gap = None, room_gap_threshold + 1
        for j in range(K):
            if j == i or int(types[j]) in EXCL:
                continue
            jx0, jy0, jx1, jy1 = boxes[j]
            if min(y1, jy1) - max(y0, jy0) > 0 and 0 < x0 - jx1 <= room_gap_threshold:
                if x0 - jx1 < best_gap:
                    best_gap, best_j = x0 - jx1, j
        if best_j is not None:
            jx0, jy0, jx1, jy1 = boxes[best_j]
            print(f"[FILL GAPS]   {name} LEFT:   room-to-room gap={best_gap:.2f}px "
                  f"to {room_label(best_j)} → snapping")
            boxes[i][0] = jx1          # snap directly — no rounding overlap
            x0 = boxes[i][0]

        # RIGHT — find nearest room to the right with y-overlap
        best_j, best_gap = None, room_gap_threshold + 1
        for j in range(K):
            if j == i or int(types[j]) in EXCL:
                continue
            jx0, jy0, jx1, jy1 = boxes[j]
            if min(y1, jy1) - max(y0, jy0) > 0 and 0 < jx0 - x1 <= room_gap_threshold:
                if jx0 - x1 < best_gap:
                    best_gap, best_j = jx0 - x1, j
        if best_j is not None:
            jx0, jy0, jx1, jy1 = boxes[best_j]
            print(f"[FILL GAPS]   {name} RIGHT:  room-to-room gap={best_gap:.2f}px "
                  f"to {room_label(best_j)} → snapping")
            boxes[i][2] = jx0
            x1 = boxes[i][2]

        # TOP — find nearest room above with x-overlap
        best_j, best_gap = None, room_gap_threshold + 1
        for j in range(K):
            if j == i or int(types[j]) in EXCL:
                continue
            jx0, jy0, jx1, jy1 = boxes[j]
            if min(x1, jx1) - max(x0, jx0) > 0 and 0 < y0 - jy1 <= room_gap_threshold:
                if y0 - jy1 < best_gap:
                    best_gap, best_j = y0 - jy1, j
        if best_j is not None:
            jx0, jy0, jx1, jy1 = boxes[best_j]
            print(f"[FILL GAPS]   {name} TOP:    room-to-room gap={best_gap:.2f}px "
                  f"to {room_label(best_j)} → snapping")
            boxes[i][1] = jy1
            y0 = boxes[i][1]

        # BOTTOM — find nearest room below with x-overlap
        best_j, best_gap = None, room_gap_threshold + 1
        for j in range(K):
            if j == i or int(types[j]) in EXCL:
                continue
            jx0, jy0, jx1, jy1 = boxes[j]
            if min(x1, jx1) - max(x0, jx0) > 0 and 0 < jy0 - y1 <= room_gap_threshold:
                if jy0 - y1 < best_gap:
                    best_gap, best_j = jy0 - y1, j
        if best_j is not None:
            jx0, jy0, jx1, jy1 = boxes[best_j]
            print(f"[FILL GAPS]   {name} BOTTOM: room-to-room gap={best_gap:.2f}px "
                  f"to {room_label(best_j)} → snapping")
            boxes[i][3] = jy0
            y1 = boxes[i][3]

    print(f"[FILL GAPS] Done.")
    return [np.array(b) for b in boxes]


def fill_living_room(boxes, types, boundary):
    """
    Expand the Living Room (type 0) to cover the entire boundary polygon,
    without moving or resizing any other room.  Other rooms are drawn on top
    in the DXF export, so visual overlap is intentional.

    The LR is expanded to the actual polygon walls (queried at the LR midpoint)
    and iterated until stable so that the midpoint query converges as the LR
    grows toward the full boundary extent.
    """
    boxes = [list(b) for b in boxes]
    types = list(types)
    K = len(boxes)

    lr_idx = next((i for i in range(K) if int(types[i]) == 0), None)
    if lr_idx is None:
        print("[FILL LR] No LivingRoom found — skipping.")
        return [np.array(b) for b in boxes]

    boundary_arr = np.array(boundary)
    bx0 = float(np.min(boundary_arr[:, 0]))
    by0 = float(np.min(boundary_arr[:, 1]))
    bx1 = float(np.max(boundary_arr[:, 0]))
    by1 = float(np.max(boundary_arr[:, 1]))

    lx0, ly0, lx1, ly1 = (float(boxes[lr_idx][k]) for k in range(4))
    print(f"[FILL LR] Expanding LR from ({lx0:.0f},{ly0:.0f},{lx1:.0f},{ly1:.0f}) "
          f"to bounding box ({bx0:.0f},{by0:.0f},{bx1:.0f},{by1:.0f})")
    boxes[lr_idx][0] = bx0
    boxes[lr_idx][1] = by0
    boxes[lr_idx][2] = bx1
    boxes[lr_idx][3] = by1

    return [np.array(b) for b in boxes]

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

    # --- Boundary bounding box -----------------------------------------------
    if boundary is not None and len(boundary) > 0:
        bnd = np.array(boundary)
        bx0 = int(np.min(bnd[:, 0]))
        by0 = int(np.min(bnd[:, 1]))
        bx1 = int(np.max(bnd[:, 0]))
        by1 = int(np.max(bnd[:, 1]))
    else:
        bx0, by0, bx1, by1 = 0, 0, CANVAS - 1, CANVAS - 1

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

    # --- Objective: minimize L1 displacement from initial boxes --------------
    # |x[i] - x0[i]|  encoded via AbsEquality auxiliary variable
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
        return result, status

    # Solver failed — return original boxes clipped to boundary
    warnings.warn("CP-SAT solver failed to find a feasible solution. "
                  "Returning original boxes clipped to boundary.", UserWarning)
    result = boxes.copy()
    result[:, 0] = np.clip(result[:, 0], bx0, bx1)
    result[:, 1] = np.clip(result[:, 1], by0, by1)
    result[:, 2] = np.clip(result[:, 2], bx0, bx1)
    result[:, 3] = np.clip(result[:, 3], by0, by1)
    return result, 'FAILED'


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

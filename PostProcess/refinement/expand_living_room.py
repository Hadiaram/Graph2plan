"""
Living room expansion module - expands the living room to fill boundary space.

This module expands the living room (public area) to fill any empty space
between the room boxes and the boundary walls.
"""

import numpy as np


def expand_living_room_to_boundary(boxes: np.ndarray,
                                   room_types: np.ndarray,
                                   boundary: np.ndarray,
                                   verbose: bool = False) -> np.ndarray:
    """
    Expand the living room box to fill the full boundary bounding rectangle.

    The living room is the background layer — other rooms are drawn on top of
    it, so overlapping is intentional and correct. The SVG clipPath handles
    clamping everything to the actual boundary polygon.

    All boundary vertices are used for the extents (including isNew==1 points,
    which are real structural corners of the polygon).

    Args:
        boxes: Nx4 array of boxes [[x1, y1, x2, y2], ...]
        room_types: Array of room type indices
        boundary: Boundary polygon (Nx2 or Nx4 array)
        verbose: Print debug information

    Returns:
        expanded_boxes: Nx4 array with living room expanded
    """
    living_room_indices = np.where(room_types == 0)[0]

    if len(living_room_indices) == 0:
        if verbose:
            print("  No living room found (type=0), skipping expansion")
        return boxes.copy()

    living_room_idx = living_room_indices[0]

    if verbose:
        print(f"\n{'='*60}")
        print(f"LIVING ROOM EXPANSION")
        print(f"{'='*60}")
        print(f"  Living room index: {living_room_idx}")
        print(f"  Original box: {boxes[living_room_idx]}")

    # Use all boundary vertices (isNew flag does not matter — those points are
    # real corners and must be included to get correct extents).
    bnd_pts = boundary[:, :2]
    boundary_x_min = float(np.min(bnd_pts[:, 0]))
    boundary_x_max = float(np.max(bnd_pts[:, 0]))
    boundary_y_min = float(np.min(bnd_pts[:, 1]))
    boundary_y_max = float(np.max(bnd_pts[:, 1]))

    if verbose:
        print(f"  Boundary extents: X=[{boundary_x_min:.2f}, {boundary_x_max:.2f}], Y=[{boundary_y_min:.2f}, {boundary_y_max:.2f}]")

    expanded_boxes = boxes.copy()
    lr_box = boxes[living_room_idx].copy()

    # Start with full boundary extents on all four sides, then pull back any
    # side where a bathroom (type=3) is currently touching the living room.
    # Bathrooms need a traversable connection to the living room — if the
    # living room expands past them they become inaccessible interior pockets.
    # Other room types (bedrooms, kitchens, etc.) are not restricted: they are
    # large enough to touch a boundary wall independently and don't rely on a
    # specific living-room edge for access.
    TOUCH_TOL = 1.0  # px — edge-coordinate tolerance for "touching"

    new_x1 = boundary_x_min
    new_x2 = boundary_x_max
    new_y1 = boundary_y_min
    new_y2 = boundary_y_max

    for i in range(len(boxes)):
        if room_types[i] != 3:          # bathrooms only
            continue
        b = boxes[i]
        # Axis-overlap helpers (rooms must actually share a segment, not just
        # be at the same coordinate in a completely different region).
        y_overlap = min(b[3], lr_box[3]) - max(b[1], lr_box[1]) > 0
        x_overlap = min(b[2], lr_box[2]) - max(b[0], lr_box[0]) > 0

        # Bathroom to the LEFT  → its right edge touches LR's left edge
        if abs(b[2] - lr_box[0]) <= TOUCH_TOL and y_overlap:
            new_x1 = lr_box[0]
            if verbose:
                print(f"  Box {i} (Bathroom) touches LR left edge  → left  expansion blocked")

        # Bathroom to the RIGHT → its left edge touches LR's right edge
        if abs(b[0] - lr_box[2]) <= TOUCH_TOL and y_overlap:
            new_x2 = lr_box[2]
            if verbose:
                print(f"  Box {i} (Bathroom) touches LR right edge → right expansion blocked")

        # Bathroom ABOVE        → its bottom edge touches LR's top edge
        if abs(b[3] - lr_box[1]) <= TOUCH_TOL and x_overlap:
            new_y1 = lr_box[1]
            if verbose:
                print(f"  Box {i} (Bathroom) touches LR top edge   → top   expansion blocked")

        # Bathroom BELOW        → its top edge touches LR's bottom edge
        if abs(b[1] - lr_box[3]) <= TOUCH_TOL and x_overlap:
            new_y2 = lr_box[3]
            if verbose:
                print(f"  Box {i} (Bathroom) touches LR bottom edge→ bottom expansion blocked")

    expanded_boxes[living_room_idx] = np.array([new_x1, new_y1, new_x2, new_y2])

    if verbose:
        original_area = (lr_box[2] - lr_box[0]) * (lr_box[3] - lr_box[1])
        new_area = (new_x2 - new_x1) * (new_y2 - new_y1)
        print(f"  Expanded box: [{new_x1:.2f}, {new_y1:.2f}, {new_x2:.2f}, {new_y2:.2f}]")
        print(f"  Area change: {original_area:.2f} → {new_area:.2f} (+{((new_area/original_area - 1) * 100):.1f}%)")
        print(f"{'='*60}\n")

    return expanded_boxes

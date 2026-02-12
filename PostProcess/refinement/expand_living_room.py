"""
Living room expansion module - expands the living room to fill boundary space.

This module expands the living room (public area) to fill any empty space
between the room boxes and the boundary walls.
"""

import numpy as np
from typing import Tuple, List
from .geometry_utils import Box, extract_boundary_segments


def _get_boundary_pts_lr(boundary: np.ndarray) -> np.ndarray:
    """Return only the original (non-isNew) boundary vertices as Nx2 array."""
    if boundary.ndim == 2 and boundary.shape[1] >= 4:
        mask = (boundary[:, 3] == 0)
        return boundary[mask, :2]
    return boundary[:, :2]


def expand_living_room_to_boundary(boxes: np.ndarray, 
                                   room_types: np.ndarray,
                                   boundary: np.ndarray,
                                   verbose: bool = False) -> np.ndarray:
    """
    Expand the living room box to fill space up to the boundary.
    
    This function:
    1. Finds the living room (room_type = 0)
    2. Expands it to the min/max boundary coordinates
    3. Ensures it doesn't overlap with other rooms (keeps other rooms intact)
    
    Args:
        boxes: Nx4 array of boxes [[x1, y1, x2, y2], ...]
        room_types: Array of room type indices
        boundary: Boundary polygon
        verbose: Print debug information
        
    Returns:
        expanded_boxes: Nx4 array with living room expanded
    """
    
    # Find living room (type 0)
    living_room_indices = np.where(room_types == 0)[0]
    
    if len(living_room_indices) == 0:
        if verbose:
            print("  No living room found (type=0), skipping expansion")
        return boxes.copy()
    
    living_room_idx = living_room_indices[0]  # Take first living room if multiple
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"LIVING ROOM EXPANSION")
        print(f"{'='*60}")
        print(f"  Living room index: {living_room_idx}")
        print(f"  Original box: {boxes[living_room_idx]}")
    
    # Get boundary extents — use only original (non-isNew) vertices so that
    # interpolated helper points don't inflate the expansion limits.
    bnd_pts = _get_boundary_pts_lr(boundary)
    boundary_x_min = np.min(bnd_pts[:, 0])
    boundary_x_max = np.max(bnd_pts[:, 0])
    boundary_y_min = np.min(bnd_pts[:, 1])
    boundary_y_max = np.max(bnd_pts[:, 1])
    
    if verbose:
        print(f"  Boundary extents: X=[{boundary_x_min:.2f}, {boundary_x_max:.2f}], Y=[{boundary_y_min:.2f}, {boundary_y_max:.2f}]")
    
    # Start with current living room box
    expanded_boxes = boxes.copy()
    lr_box = boxes[living_room_idx].copy()
    lr_x1, lr_y1, lr_x2, lr_y2 = lr_box
    
    # Start by expanding to boundary in each direction
    new_x1 = boundary_x_min
    new_x2 = boundary_x_max
    new_y1 = boundary_y_min
    new_y2 = boundary_y_max
    
    # Check each other room to see if it blocks expansion in any direction
    for i, other_box in enumerate(boxes):
        if i == living_room_idx:
            continue  # Skip the living room itself
            
        ox1, oy1, ox2, oy2 = other_box
        
        # Check vertical overlap (needed for horizontal expansion)
        vertical_overlap = not (oy2 <= lr_y1 or oy1 >= lr_y2)
        
        # Check horizontal overlap (needed for vertical expansion)
        horizontal_overlap = not (ox2 <= lr_x1 or ox1 >= lr_x2)
        
        # LEFT expansion: if other room is to the left and has vertical overlap
        if vertical_overlap and ox2 > new_x1 and ox2 <= lr_x1:
            new_x1 = max(new_x1, ox2)
            if verbose:
                print(f"  Room {i} (type={room_types[i]}) blocks left expansion at x={ox2:.2f}")
        
        # RIGHT expansion: if other room is to the right and has vertical overlap
        if vertical_overlap and ox1 >= lr_x2 and ox1 < new_x2:
            new_x2 = min(new_x2, ox1)
            if verbose:
                print(f"  Room {i} (type={room_types[i]}) blocks right expansion at x={ox1:.2f}")
        
        # TOP expansion: if other room is above and has horizontal overlap
        if horizontal_overlap and oy2 > new_y1 and oy2 <= lr_y1:
            new_y1 = max(new_y1, oy2)
            if verbose:
                print(f"  Room {i} (type={room_types[i]}) blocks top expansion at y={oy2:.2f}")
        
        # BOTTOM expansion: if other room is below and has horizontal overlap
        if horizontal_overlap and oy1 >= lr_y2 and oy1 < new_y2:
            new_y2 = min(new_y2, oy1)
            if verbose:
                print(f"  Room {i} (type={room_types[i]}) blocks bottom expansion at y={oy1:.2f}")
    
    # Apply the expansion
    expanded_boxes[living_room_idx] = np.array([new_x1, new_y1, new_x2, new_y2])
    
    if verbose:
        print(f"  Expanded box: [{new_x1:.2f}, {new_y1:.2f}, {new_x2:.2f}, {new_y2:.2f}]")
        original_area = (lr_box[2] - lr_box[0]) * (lr_box[3] - lr_box[1])
        new_area = (new_x2 - new_x1) * (new_y2 - new_y1)
        print(f"  Area change: {original_area:.2f} → {new_area:.2f} (+{((new_area/original_area - 1) * 100):.1f}%)")
        print(f"{'='*60}\n")
    
    return expanded_boxes

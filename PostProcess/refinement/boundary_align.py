"""
Boundary alignment module for post-processing refinement.

This module implements Step 1 of the refinement pipeline:
snapping room boxes to boundary walls when they are close enough.
"""

import numpy as np
from typing import Tuple, List, Optional, Dict
from .geometry_utils import Box, BoundarySegment, extract_boundary_segments, distance_point_to_segment


class EdgeAlignment:
    """
    Stores information about aligning a box edge with a boundary segment.

    Attributes:
        edge_name: 'left', 'top', 'right', or 'bottom'
        distance: Distance from box edge to boundary segment
        snap_value: Coordinate value to snap to
        segment: The boundary segment to snap to
    """

    def __init__(self, edge_name: str, distance: float,
                 snap_value: float, segment: Optional[BoundarySegment] = None):
        self.edge_name = edge_name
        self.distance = distance
        self.snap_value = snap_value
        self.segment = segment

    def __repr__(self):
        return f"EdgeAlignment({self.edge_name}, dist={self.distance:.2f}, snap_to={self.snap_value:.2f})"


def would_detach_from_walls(box: Box, edge_to_snap: str, snap_value: float,
                           current_alignments: Dict[str, EdgeAlignment],
                           tolerance: float = 0.1) -> Tuple[bool, List[str]]:
    """
    Check if snapping an edge would detach the box from walls it's currently attached to.
    
    Args:
        box: Current box position
        edge_to_snap: Which edge we want to snap ('left', 'right', 'top', 'bottom')
        snap_value: The coordinate value we want to snap to
        current_alignments: Current alignment info for all edges
        tolerance: Distance threshold to consider "attached" (default 0.1px)
        
    Returns:
        (would_detach, detached_edges): True if movement would detach, and list of affected edges
    """
    # Calculate the movement delta
    if edge_to_snap == 'left':
        delta_x = snap_value - box.x1
        delta_y = 0
    elif edge_to_snap == 'right':
        delta_x = snap_value - box.x2
        delta_y = 0
    elif edge_to_snap == 'top':
        delta_x = 0
        delta_y = snap_value - box.y1
    elif edge_to_snap == 'bottom':
        delta_x = 0
        delta_y = snap_value - box.y2
    else:
        return False, []
    
    # Create hypothetical new box position
    new_box = Box(
        box.x1 + delta_x,
        box.y1 + delta_y,
        box.x2 + delta_x,
        box.y2 + delta_y
    )
    
    detached_edges = []
    
    # Check each edge that is currently attached
    for edge_name, alignment in current_alignments.items():
        # Skip the edge we're trying to snap (it's intentionally moving)
        if edge_name == edge_to_snap:
            continue
            
        # Is this edge currently attached to a wall?
        if alignment.distance < tolerance:
            # Calculate new distance after the move
            if edge_name == 'left':
                new_distance = abs(new_box.x1 - alignment.snap_value)
            elif edge_name == 'right':
                new_distance = abs(new_box.x2 - alignment.snap_value)
            elif edge_name == 'top':
                new_distance = abs(new_box.y1 - alignment.snap_value)
            elif edge_name == 'bottom':
                new_distance = abs(new_box.y2 - alignment.snap_value)
            else:
                continue
            
            # Would this edge become detached?
            if new_distance > tolerance:
                detached_edges.append(edge_name)
    
    return len(detached_edges) > 0, detached_edges


def calculate_boundary_coverage(box: Box, edge_to_snap: str, snap_value: float,
                                boundary_extents: Tuple[float, float, float, float]) -> float:
    """
    Calculate how much of the boundary space a box would cover after snapping.
    Returns a score where higher is better (more coverage = less empty space).
    
    Args:
        box: Current box position
        edge_to_snap: Which edge we want to snap
        snap_value: The coordinate value we want to snap to
        boundary_extents: (min_x, max_x, min_y, max_y) of the boundary
        
    Returns:
        Coverage score (0-1, higher is better)
    """
    min_x, max_x, min_y, max_y = boundary_extents
    boundary_width = max_x - min_x
    boundary_height = max_y - min_y
    
    # Calculate the new box position after snapping
    if edge_to_snap == 'left':
        new_x1, new_x2 = snap_value, box.x2 + (snap_value - box.x1)
        new_y1, new_y2 = box.y1, box.y2
    elif edge_to_snap == 'right':
        new_x1, new_x2 = box.x1 + (snap_value - box.x2), snap_value
        new_y1, new_y2 = box.y1, box.y2
    elif edge_to_snap == 'top':
        new_x1, new_x2 = box.x1, box.x2
        new_y1, new_y2 = snap_value, box.y2 + (snap_value - box.y1)
    elif edge_to_snap == 'bottom':
        new_x1, new_x2 = box.x1, box.x2
        new_y1, new_y2 = box.y1 + (snap_value - box.y2), snap_value
    else:
        return 0.0
    
    # Calculate what percentage of boundary space this would cover
    # Higher percentage = better coverage = less empty space
    x_coverage = (new_x2 - new_x1) / boundary_width if boundary_width > 0 else 0
    y_coverage = (new_y2 - new_y1) / boundary_height if boundary_height > 0 else 0
    
    # Also consider how close the edges are to the boundary extents
    # Penalize distance from boundary edges
    left_gap = (new_x1 - min_x) / boundary_width if boundary_width > 0 else 0
    right_gap = (max_x - new_x2) / boundary_width if boundary_width > 0 else 0
    top_gap = (new_y1 - min_y) / boundary_height if boundary_height > 0 else 0
    bottom_gap = (max_y - new_y2) / boundary_height if boundary_height > 0 else 0
    
    # Total gap (lower is better, so we'll subtract this from 1)
    total_gap = left_gap + right_gap + top_gap + bottom_gap
    
    # Coverage score: combination of size and gap filling
    # Prioritize filling gaps to boundary edges
    coverage_score = 1.0 - (total_gap / 4.0)
    
    return coverage_score


def ray_cast_to_walls(box: Box, h_segments: List[BoundarySegment],
                      v_segments: List[BoundarySegment]) -> Dict[str, Optional[BoundarySegment]]:
    """
    Cast rays from box center to find the first wall hit in each direction.

    This filters out occluded walls by only considering walls that are directly
    visible from the box center via horizontal/vertical ray casting.

    Args:
        box: The box to cast rays from
        h_segments: List of horizontal boundary segments
        v_segments: List of vertical boundary segments

    Returns:
        Dictionary mapping direction to the first wall segment hit:
        {
            'left': BoundarySegment or None,
            'right': BoundarySegment or None,
            'top': BoundarySegment or None,
            'bottom': BoundarySegment or None
        }
    """
    center_x, center_y = box.center

    visible_walls = {
        'left': None,
        'right': None,
        'top': None,
        'bottom': None
    }

    # Ray cast LEFT: Find closest vertical wall to the left of center
    min_dist_left = float('inf')
    for seg in v_segments:
        seg_x = seg.x1  # Vertical segment has constant x

        # Wall must be to the left of center
        if seg_x >= center_x:
            continue

        # Check if ray at center_y intersects with segment's y range
        seg_y_min = min(seg.y1, seg.y2)
        seg_y_max = max(seg.y1, seg.y2)

        if seg_y_min <= center_y <= seg_y_max:
            # Ray hits this wall
            dist = center_x - seg_x
            if dist < min_dist_left:
                min_dist_left = dist
                visible_walls['left'] = seg

    # Ray cast RIGHT: Find closest vertical wall to the right of center
    min_dist_right = float('inf')
    for seg in v_segments:
        seg_x = seg.x1

        # Wall must be to the right of center
        if seg_x <= center_x:
            continue

        seg_y_min = min(seg.y1, seg.y2)
        seg_y_max = max(seg.y1, seg.y2)

        if seg_y_min <= center_y <= seg_y_max:
            dist = seg_x - center_x
            if dist < min_dist_right:
                min_dist_right = dist
                visible_walls['right'] = seg

    # Ray cast UP (TOP): Find closest horizontal wall above center
    min_dist_top = float('inf')
    for seg in h_segments:
        seg_y = seg.y1  # Horizontal segment has constant y

        # Wall must be above center (lower y value)
        if seg_y >= center_y:
            continue

        seg_x_min = min(seg.x1, seg.x2)
        seg_x_max = max(seg.x1, seg.x2)

        if seg_x_min <= center_x <= seg_x_max:
            dist = center_y - seg_y
            if dist < min_dist_top:
                min_dist_top = dist
                visible_walls['top'] = seg

    # Ray cast DOWN (BOTTOM): Find closest horizontal wall below center
    min_dist_bottom = float('inf')
    for seg in h_segments:
        seg_y = seg.y1

        # Wall must be below center (higher y value)
        if seg_y <= center_y:
            continue

        seg_x_min = min(seg.x1, seg.x2)
        seg_x_max = max(seg.x1, seg.x2)

        if seg_x_min <= center_x <= seg_x_max:
            dist = seg_y - center_y
            if dist < min_dist_bottom:
                min_dist_bottom = dist
                visible_walls['bottom'] = seg

    return visible_walls


def find_closest_segments(box: Box, h_segments: List[BoundarySegment],
                          v_segments: List[BoundarySegment]) -> Dict[str, EdgeAlignment]:
    """
    Find the closest boundary segment for each edge of a box.

    This implements the MATLAB find_close_seg() function with ray casting
    to filter out occluded walls.

    Args:
        box: The box to find alignments for
        h_segments: List of horizontal boundary segments
        v_segments: List of vertical boundary segments

    Returns:
        Dictionary mapping edge names to EdgeAlignment objects:
        {
            'left': EdgeAlignment(...),
            'top': EdgeAlignment(...),
            'right': EdgeAlignment(...),
            'bottom': EdgeAlignment(...)
        }
    """
    alignments = {}

    # Initialize with large distances
    MAX_DIST = 999999.0

    # STEP 1: Ray cast from box center to find visible walls only
    visible_walls = ray_cast_to_walls(box, h_segments, v_segments)

    # STEP 2: Calculate distances to visible walls only
    # Find closest vertical segment for left and right edges
    best_left = EdgeAlignment('left', MAX_DIST, box.x1)
    best_right = EdgeAlignment('right', MAX_DIST, box.x2)

    # Only check the visible wall for left edge (from ray casting)
    if visible_walls['left'] is not None:
        seg = visible_walls['left']

        # Calculate vertical overlap
        seg_y_min = min(seg.y1, seg.y2)
        seg_y_max = max(seg.y1, seg.y2)
        box_y_min = box.y1
        box_y_max = box.y2

        overlap_y_start = max(seg_y_min, box_y_min)
        overlap_y_end = min(seg_y_max, box_y_max)

        if overlap_y_end < overlap_y_start:
            if seg_y_max <= box_y_min:
                v_dist = box_y_min - seg_y_max
            elif seg_y_min >= box_y_max:
                v_dist = seg_y_min - box_y_max
            else:
                v_dist = 0
        else:
            v_dist = 0

        seg_x = seg.x1
        h_dist_left = abs(box.x1 - seg_x)
        dist_left = np.sqrt(h_dist_left**2 + v_dist**2)
        best_left = EdgeAlignment('left', dist_left, seg_x, seg)

    # Only check the visible wall for right edge (from ray casting)
    if visible_walls['right'] is not None:
        seg = visible_walls['right']

        seg_y_min = min(seg.y1, seg.y2)
        seg_y_max = max(seg.y1, seg.y2)
        box_y_min = box.y1
        box_y_max = box.y2

        overlap_y_start = max(seg_y_min, box_y_min)
        overlap_y_end = min(seg_y_max, box_y_max)

        if overlap_y_end < overlap_y_start:
            if seg_y_max <= box_y_min:
                v_dist = box_y_min - seg_y_max
            elif seg_y_min >= box_y_max:
                v_dist = seg_y_min - box_y_max
            else:
                v_dist = 0
        else:
            v_dist = 0

        seg_x = seg.x1
        h_dist_right = abs(seg_x - box.x2)
        dist_right = np.sqrt(h_dist_right**2 + v_dist**2)
        best_right = EdgeAlignment('right', dist_right, seg_x, seg)

    alignments['left'] = best_left
    alignments['right'] = best_right

    # Find closest horizontal segment for top and bottom edges
    best_top = EdgeAlignment('top', MAX_DIST, box.y1)
    best_bottom = EdgeAlignment('bottom', MAX_DIST, box.y2)

    # Only check the visible wall for top edge (from ray casting)
    if visible_walls['top'] is not None:
        seg = visible_walls['top']

        # Calculate horizontal overlap
        seg_x_min = min(seg.x1, seg.x2)
        seg_x_max = max(seg.x1, seg.x2)
        box_x_min = box.x1
        box_x_max = box.x2

        overlap_x_start = max(seg_x_min, box_x_min)
        overlap_x_end = min(seg_x_max, box_x_max)

        if overlap_x_end < overlap_x_start:
            if seg_x_max <= box_x_min:
                h_dist = box_x_min - seg_x_max
            elif seg_x_min >= box_x_max:
                h_dist = seg_x_min - box_x_max
            else:
                h_dist = 0
        else:
            h_dist = 0

        seg_y = seg.y1
        v_dist_top = abs(box.y1 - seg_y)
        dist_top = np.sqrt(v_dist_top**2 + h_dist**2)
        best_top = EdgeAlignment('top', dist_top, seg_y, seg)

    # Only check the visible wall for bottom edge (from ray casting)
    if visible_walls['bottom'] is not None:
        seg = visible_walls['bottom']

        seg_x_min = min(seg.x1, seg.x2)
        seg_x_max = max(seg.x1, seg.x2)
        box_x_min = box.x1
        box_x_max = box.x2

        overlap_x_start = max(seg_x_min, box_x_min)
        overlap_x_end = min(seg_x_max, box_x_max)

        if overlap_x_end < overlap_x_start:
            if seg_x_max <= box_x_min:
                h_dist = box_x_min - seg_x_max
            elif seg_x_min >= box_x_max:
                h_dist = seg_x_min - box_x_max
            else:
                h_dist = 0
        else:
            h_dist = 0

        seg_y = seg.y1
        v_dist_bottom = abs(seg_y - box.y2)
        dist_bottom = np.sqrt(v_dist_bottom**2 + h_dist**2)
        best_bottom = EdgeAlignment('bottom', dist_bottom, seg_y, seg)

    alignments['top'] = best_top
    alignments['bottom'] = best_bottom

    return alignments


def align_box_with_boundary(box: Box, boundary: np.ndarray,
                           threshold: float = 8.0,
                           verbose: bool = False,
                           exclude_axis: Optional[str] = None,
                           refinement_pass: int = 1) -> Tuple[Box, Dict[str, bool]]:
    """
    Align a box with the boundary by snapping edges within threshold.

    This implements the MATLAB align_with_boundary() function with ray casting.

    Args:
        box: Box to align
        boundary: Boundary polygon (Nx2 or Nx4 array)
        threshold: Maximum distance for snapping (pixels)
        verbose: Print debug information
        exclude_axis: Optional axis to exclude from snapping ('horizontal' or 'vertical')
                     Used for two-pass refinement to snap orthogonal directions separately
        refinement_pass: Current refinement pass (1 or 2). Pass 2 prefers movements that cover empty space.

    Returns:
        aligned_box: Box with edges snapped to boundary
        updated_edges: Dict indicating which edges were updated
                      {'left': bool, 'top': bool, 'right': bool, 'bottom': bool}
    """
    # Extract boundary segments
    h_segments, v_segments = extract_boundary_segments(boundary)
    
    # Calculate boundary extents for coverage calculation
    all_x = [seg.x1 for seg in h_segments + v_segments] + [seg.x2 for seg in h_segments + v_segments]
    all_y = [seg.y1 for seg in h_segments + v_segments] + [seg.y2 for seg in h_segments + v_segments]
    boundary_extents = (min(all_x), max(all_x), min(all_y), max(all_y))

    if verbose:
        print(f"\nAligning box: {box}")
        print(f"  Boundary has {len(h_segments)} horizontal and {len(v_segments)} vertical segments")
        print(f"  Threshold: {threshold} pixels")

    # Find closest segments for each edge
    alignments = find_closest_segments(box, h_segments, v_segments)

    if verbose:
        print("\n  Closest segments:")
        for edge_name, alignment in alignments.items():
            within = "✓" if alignment.distance <= threshold else "✗"
            print(f"    {within} {edge_name}: distance={alignment.distance:.2f}, snap_to={alignment.snap_value:.2f}")

    # Create new box with snapped edges
    aligned_box = Box(box.x1, box.y1, box.x2, box.y2)
    original_width = box.x2 - box.x1
    original_height = box.y2 - box.y1

    updated_edges = {
        'left': False,
        'top': False,
        'right': False,
        'bottom': False
    }

    # NEW LOGIC: Find the SINGLE closest edge overall, then shift entire box
    # This prevents extension/conflicts by only snapping one edge at a time

    # Filter edges based on exclude_axis parameter
    available_edges = {}
    for edge_name, alignment in alignments.items():
        # Determine axis for this edge
        edge_axis = 'horizontal' if edge_name in ['left', 'right'] else 'vertical'
        
        # Skip this edge if its axis is excluded
        if exclude_axis is not None and edge_axis == exclude_axis:
            if verbose:
                print(f"  ⊘ Skipping {edge_name} edge (axis '{edge_axis}' excluded)")
            continue
        
        # Check if snapping this edge would detach the box from walls it's currently attached to
        would_detach, detached_edges = would_detach_from_walls(
            box, edge_name, alignment.snap_value, alignments, tolerance=0.1
        )
        if would_detach:
            if verbose:
                print(f"  ⊗ Skipping {edge_name} edge (would detach from: {', '.join(detached_edges)})")
            continue
            
        available_edges[edge_name] = alignment

    if not available_edges:
        if verbose:
            print(f"  ⊗ No edges available after axis filtering (excluded: {exclude_axis})")
        updated_edges = {'left': False, 'top': False, 'right': False, 'bottom': False}
        return aligned_box, updated_edges

    # Find which edge is closest to any wall (from available edges only)
    # In pass 2, also consider coverage score to prefer movements that fill empty space
    closest_edge = None
    closest_distance = float('inf')  # Start with infinity to find true minimum
    closest_alignment = None
    best_coverage = -1.0  # Track best coverage for pass 2

    for edge_name, alignment in available_edges.items():
        # Skip edges that are already perfectly aligned (already snapped)
        if alignment.distance < 0.1:  # Consider anything under 0.1 pixels as "already aligned"
            if verbose:
                print(f"  ⊙ Skipping {edge_name} edge (already aligned at {alignment.distance:.2f}px)")
            continue
        
        # Calculate coverage score for this movement (used in pass 2)
        coverage = calculate_boundary_coverage(box, edge_name, alignment.snap_value, boundary_extents)
        
        # Selection logic depends on pass
        if refinement_pass == 2:
            # Pass 2: Prefer movements that maximize coverage, but still within reasonable distance
            # Use a weighted score: prioritize coverage, but penalize extreme distances
            # Only consider edges within threshold
            if alignment.distance <= threshold:
                # Weighted score: 70% coverage, 30% inverse distance
                # Normalize distance to 0-1 range (closer = higher score)
                distance_score = 1.0 - (alignment.distance / threshold)
                combined_score = 0.7 * coverage + 0.3 * distance_score
                
                if verbose:
                    print(f"    {edge_name}: dist={alignment.distance:.2f}px, coverage={coverage:.3f}, score={combined_score:.3f}")
                
                # Pick the edge with best combined score
                if combined_score > best_coverage:
                    best_coverage = combined_score
                    closest_distance = alignment.distance
                    closest_edge = edge_name
                    closest_alignment = alignment
        else:
            # Pass 1: Just pick closest edge (original behavior)
            if alignment.distance < closest_distance:
                closest_distance = alignment.distance
                closest_edge = edge_name
                closest_alignment = alignment

    if verbose and closest_edge:
        print(f"\n  Overall closest edge: {closest_edge} = {closest_distance:.2f}px")

    # Only snap if the closest edge is within threshold
    if closest_edge is not None and closest_distance <= threshold:
        if closest_edge == 'left':
            # Calculate how much to move
            delta = aligned_box.x1 - closest_alignment.snap_value
            # Move both edges by the same amount in the same direction
            aligned_box.x1 = closest_alignment.snap_value
            aligned_box.x2 -= delta
            updated_edges['left'] = True
            if verbose:
                print(f"  ✓ Snapped LEFT edge: shifted entire box LEFT by {delta:.2f}px")

        elif closest_edge == 'right':
            # Calculate how much to move
            delta = closest_alignment.snap_value - aligned_box.x2
            # Move both edges by the same amount in the same direction
            aligned_box.x2 = closest_alignment.snap_value
            aligned_box.x1 += delta
            updated_edges['right'] = True
            if verbose:
                print(f"  ✓ Snapped RIGHT edge: shifted entire box RIGHT by {delta:.2f}px")

        elif closest_edge == 'top':
            # Calculate how much to move
            delta = aligned_box.y1 - closest_alignment.snap_value
            # Move both edges by the same amount in the same direction
            aligned_box.y1 = closest_alignment.snap_value
            aligned_box.y2 -= delta
            updated_edges['top'] = True
            if verbose:
                print(f"  ✓ Snapped TOP edge: shifted entire box UP by {delta:.2f}px")

        elif closest_edge == 'bottom':
            # Calculate how much to move
            delta = closest_alignment.snap_value - aligned_box.y2
            # Move both edges by the same amount in the same direction
            aligned_box.y2 = closest_alignment.snap_value
            aligned_box.y1 += delta
            updated_edges['bottom'] = True
            if verbose:
                print(f"  ✓ Snapped BOTTOM edge: shifted entire box DOWN by {delta:.2f}px")
    else:
        if verbose:
            if closest_edge is not None:
                print(f"  ⊗ No edges within threshold (closest: {closest_edge}={closest_distance:.2f}px > {threshold}px)")
            else:
                print(f"  ⊗ No edges within threshold")

    # Ensure box is valid (x2 > x1, y2 > y1)
    if aligned_box.x2 <= aligned_box.x1:
        aligned_box.x2 = aligned_box.x1 + 1
        if verbose:
            print(f"  ⚠ Fixed invalid width")

    if aligned_box.y2 <= aligned_box.y1:
        aligned_box.y2 = aligned_box.y1 + 1
        if verbose:
            print(f"  ⚠ Fixed invalid height")

    if verbose:
        final_width = aligned_box.x2 - aligned_box.x1
        final_height = aligned_box.y2 - aligned_box.y1
        print(f"\n  Result: {aligned_box}")
        print(f"  Updated edges: {[k for k, v in updated_edges.items() if v]}")
        print(f"  Size check: Original({original_width:.2f} × {original_height:.2f}) → Final({final_width:.2f} × {final_height:.2f})")
        if abs(final_width - original_width) > 0.01 or abs(final_height - original_height) > 0.01:
            print(f"  ⚠️  WARNING: Box size changed! This should not happen.")

    return aligned_box, updated_edges


def align_all_boxes_with_boundary(boxes: np.ndarray, boundary: np.ndarray,
                                  threshold: float = 8.0,
                                  room_types: Optional[np.ndarray] = None,
                                  verbose: bool = False,
                                  exclude_axis: Optional[any] = None,
                                  refinement_pass: int = 1) -> Tuple[np.ndarray, List[Dict[str, bool]]]:
    """
    Align all boxes with the boundary.

    Args:
        boxes: Nx4 array of boxes [[x1, y1, x2, y2], ...]
        boundary: Boundary polygon
        threshold: Snapping threshold (pixels)
        room_types: Optional array of room type indices
        verbose: Print debug information
        exclude_axis: Optional axis to exclude from snapping. Can be:
                     - None: No exclusion (default)
                     - str: 'horizontal' or 'vertical' to exclude for all boxes
                     - List[str]: Per-box exclusion (list of 'horizontal', 'vertical', or None)
        refinement_pass: Current refinement pass (1 or 2)

    Returns:
        aligned_boxes: Nx4 array of aligned boxes
        all_updated_edges: List of updated_edges dicts for each box
    """
    n_boxes = len(boxes)
    aligned_boxes = np.zeros_like(boxes)
    all_updated_edges = []

    if verbose:
        print(f"\n{'='*60}")
        if exclude_axis:
            print(f"BOUNDARY ALIGNMENT - {n_boxes} boxes (excluding {exclude_axis} axis)")
        else:
            print(f"BOUNDARY ALIGNMENT - {n_boxes} boxes")
        print(f"{'='*60}")

    for i in range(n_boxes):
        box = Box.from_array(boxes[i])

        if verbose:
            if room_types is not None:
                # Map room type to name for better debugging
                room_type_id = room_types[i]
                room_type_names = {
                    0: "LivingRoom",
                    1: "MasterRoom",
                    2: "Kitchen",
                    3: "Bathroom",
                    4: "DiningRoom",
                    5: "ChildRoom",
                    6: "StudyRoom",
                    7: "SecondRoom",
                    8: "GuestRoom",
                    9: "Balcony",
                    10: "Entrance",
                    11: "Storage",
                    12: "Wall"
                }
                room_name = room_type_names.get(room_type_id, f"Unknown({room_type_id})")
                print(f"\n--- Box {i}: {room_name} (type={room_type_id}) ---")
            else:
                print(f"\n--- Box {i} ---")

        # Determine the exclude_axis for this specific box
        box_exclude_axis = None
        if isinstance(exclude_axis, list):
            # Per-box exclusion list
            box_exclude_axis = exclude_axis[i] if i < len(exclude_axis) else None
        elif isinstance(exclude_axis, str):
            # Global exclusion string
            box_exclude_axis = exclude_axis
        # else: exclude_axis is None, so box_exclude_axis stays None

        aligned_box, updated_edges = align_box_with_boundary(
            box, boundary, threshold, verbose=verbose, exclude_axis=box_exclude_axis,
            refinement_pass=refinement_pass
        )

        aligned_boxes[i] = aligned_box.to_array()
        all_updated_edges.append(updated_edges)

    if verbose:
        print(f"\n{'='*60}")
        n_updated = sum(1 for edges in all_updated_edges if any(edges.values()))
        print(f"✓ Aligned {n_updated}/{n_boxes} boxes")

        # Show which boxes didn't snap
        not_snapped = []
        for i, edges in enumerate(all_updated_edges):
            if not any(edges.values()):
                if room_types is not None:
                    room_type_id = room_types[i]
                    room_type_names = {0: "LivingRoom", 1: "MasterRoom", 2: "Kitchen", 3: "Bathroom",
                                     4: "DiningRoom", 5: "ChildRoom", 6: "StudyRoom", 7: "SecondRoom",
                                     8: "GuestRoom", 9: "Balcony", 10: "Entrance", 11: "Storage", 12: "Wall"}
                    room_name = room_type_names.get(room_type_id, f"Unknown({room_type_id})")
                    not_snapped.append(f"Box {i} ({room_name})")
                else:
                    not_snapped.append(f"Box {i}")

        if not_snapped:
            print(f"⚠️  Boxes NOT snapped (all edges > {threshold}px from walls):")
            for box_desc in not_snapped:
                print(f"   - {box_desc}")

        print(f"{'='*60}\n")

    return aligned_boxes, all_updated_edges


if __name__ == "__main__":
    print("Testing boundary alignment...")

    # Create a simple rectangular boundary
    boundary = np.array([
        [0, 0],      # Bottom-left
        [100, 0],    # Bottom-right
        [100, 100],  # Top-right
        [0, 100]     # Top-left
    ])

    print(f"\nBoundary: {boundary.shape[0]} vertices")
    print(f"  Extents: (0, 0) to (100, 100)")

    # Test case 1: Box close to left wall
    print("\n" + "="*60)
    print("TEST 1: Box close to left wall (should snap)")
    box1 = Box(5, 20, 30, 50)
    aligned1, updated1 = align_box_with_boundary(box1, boundary, threshold=8.0, verbose=True)

    # Test case 2: Box far from walls
    print("\n" + "="*60)
    print("TEST 2: Box far from walls (should not snap)")
    box2 = Box(40, 40, 60, 60)
    aligned2, updated2 = align_box_with_boundary(box2, boundary, threshold=8.0, verbose=True)

    # Test case 3: Box very close to top-right corner
    print("\n" + "="*60)
    print("TEST 3: Box close to corner (should snap both edges)")
    box3 = Box(75, 94, 97, 120)  # Near top-right, extends beyond
    aligned3, updated3 = align_box_with_boundary(box3, boundary, threshold=8.0, verbose=True)

    # Test case 4: Multiple boxes
    print("\n" + "="*60)
    print("TEST 4: Multiple boxes at once")
    boxes = np.array([
        [3, 10, 25, 40],    # Close to left
        [75, 3, 97, 30],    # Close to right and top
        [40, 40, 60, 60],   # Far from walls
    ])
    room_types = np.array([1, 2, 0])  # Bedroom, Kitchen, Living Room

    aligned_all, updated_all = align_all_boxes_with_boundary(
        boxes, boundary, threshold=8.0, room_types=room_types, verbose=True
    )

    print("\n✓ Boundary alignment tests complete!")

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


def _would_separate_graph_neighbors(idx: int, new_box: Box,
                                    current_boxes: np.ndarray,
                                    edges: np.ndarray,
                                    tol: float = 1.0) -> bool:
    """
    Return True if placing new_box at index idx would separate it from any
    graph-connected neighbour it is currently in contact with.

    "In contact" covers two cases:
      - Edge-touching: rooms share an axis-aligned edge within tol pixels
      - Overlapping: rooms have positive overlap in both x and y (e.g. a
        bathroom hosted inside a bedroom)

    If the move would break that contact entirely (no edge-touch AND no
    overlap remaining), the snap is rejected.

    Args:
        idx: Index of the room being moved
        new_box: Proposed new Box position
        current_boxes: Current Nx4 array of all rooms (before the move)
        edges: Graph adjacency edges, shape (K, 2) or (K, 3)
        tol: Distance tolerance to consider rooms "touching" (pixels)

    Returns:
        True if the snap would separate a currently-contacting graph neighbour
    """
    for edge in edges:
        a, b = int(edge[0]), int(edge[1])
        if a == idx:
            nb_idx = b
        elif b == idx:
            nb_idx = a
        else:
            continue

        if nb_idx >= len(current_boxes):
            continue

        cur = Box.from_array(current_boxes[idx])
        nb = Box.from_array(current_boxes[nb_idx])

        # Current contact — edge-touching or overlapping
        v_overlap_cur = min(cur.y2, nb.y2) - max(cur.y1, nb.y1)
        h_overlap_cur = min(cur.x2, nb.x2) - max(cur.x1, nb.x1)

        touching_h = ((abs(cur.x2 - nb.x1) <= tol or abs(nb.x2 - cur.x1) <= tol)
                      and v_overlap_cur > 0)
        touching_v = ((abs(cur.y2 - nb.y1) <= tol or abs(nb.y2 - cur.y1) <= tol)
                      and h_overlap_cur > 0)
        overlapping = h_overlap_cur > 0 and v_overlap_cur > 0

        in_contact = touching_h or touching_v or overlapping
        if not in_contact:
            continue  # Not in contact — no constraint needed

        # Would they still be in contact after the move?
        v_overlap_new = min(new_box.y2, nb.y2) - max(new_box.y1, nb.y1)
        h_overlap_new = min(new_box.x2, nb.x2) - max(new_box.x1, nb.x1)

        still_h = ((abs(new_box.x2 - nb.x1) <= tol or abs(nb.x2 - new_box.x1) <= tol)
                   and v_overlap_new > 0)
        still_v = ((abs(new_box.y2 - nb.y1) <= tol or abs(nb.y2 - new_box.y1) <= tol)
                   and h_overlap_new > 0)
        still_overlapping = h_overlap_new > 0 and v_overlap_new > 0

        still_in_contact = still_h or still_v or still_overlapping

        if not still_in_contact:
            return True

    return False


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

    # After snapping, check if the box stays within boundary bounding box.
    # If snapping pushed any edge outside, REJECT the snap entirely.
    #
    # Why reject rather than clamp: clamping applies a correction in the opposite
    # direction of the snap, which can push the room PAST its original position
    # (further outside than it started).  Rejecting is safer — the room stays at
    # its current (Pass 1) position which was already validated.
    if any(updated_edges.values()):
        bnd_x_min, bnd_x_max, bnd_y_min, bnd_y_max = boundary_extents
        outside = (aligned_box.x1 < bnd_x_min - 0.5 or
                   aligned_box.x2 > bnd_x_max + 0.5 or
                   aligned_box.y1 < bnd_y_min - 0.5 or
                   aligned_box.y2 > bnd_y_max + 0.5)
        if outside:
            if verbose:
                print(f"  ⚠️  Snap would push box outside boundary extents — rejecting snap "
                      f"(x=[{aligned_box.x1:.1f},{aligned_box.x2:.1f}] "
                      f"y=[{aligned_box.y1:.1f},{aligned_box.y2:.1f}] "
                      f"vs bnd x=[{bnd_x_min:.1f},{bnd_x_max:.1f}] y=[{bnd_y_min:.1f},{bnd_y_max:.1f}])")
            aligned_box = Box(box.x1, box.y1, box.x2, box.y2)
            updated_edges = {'left': False, 'top': False, 'right': False, 'bottom': False}

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
                                  edges: Optional[np.ndarray] = None,
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
        edges: Optional graph adjacency edges, shape (K, 2) or (K, 3).
               When provided, any snap that would separate a currently-touching
               graph-connected room is rejected.
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
    aligned_boxes = boxes.copy()  # Keep current positions so graph-neighbor check sees real coords
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

        if verbose:
            print(f"  [snap] Box {i}: box_exclude_axis={box_exclude_axis!r}, "
                  f"coords before=({box.x1:.1f},{box.y1:.1f},{box.x2:.1f},{box.y2:.1f})")

        aligned_box, updated_edges = align_box_with_boundary(
            box, boundary, threshold, verbose=verbose, exclude_axis=box_exclude_axis,
            refinement_pass=refinement_pass
        )

        # Reject the snap if it would pull this room away from a touching graph neighbour
        if edges is not None and any(updated_edges.values()):
            if _would_separate_graph_neighbors(i, aligned_box, aligned_boxes, edges):
                if verbose:
                    print(f"  ⊙ Snap rejected: would separate a touching graph neighbour")
                aligned_box = box
                updated_edges = {'left': False, 'top': False, 'right': False, 'bottom': False}

        if verbose:
            snapped = [k for k, v in updated_edges.items() if v]
            if snapped:
                print(f"  [snap] Box {i}: snapped {snapped} → coords after=({aligned_box.x1:.1f},{aligned_box.y1:.1f},{aligned_box.x2:.1f},{aligned_box.y2:.1f})")
            else:
                print(f"  [snap] Box {i}: no snap applied")

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


def resolve_room_overlaps(boxes: np.ndarray,
                          room_types: np.ndarray,
                          boundary: np.ndarray,
                          edges: Optional[np.ndarray] = None,
                          verbose: bool = False) -> np.ndarray:
    """
    Resolve overlapping rooms by snapping them edge-to-edge.

    The living room (type 0) is excluded — it is a background layer and
    intentionally overlaps every other room.

    All other room pairs are checked.  If the two rooms share a graph edge
    their overlap is intentional (e.g. en-suite bathroom inside its bedroom)
    and is left alone.  If they have no graph edge they are independent spaces
    and the overlap is resolved by moving the less-wall-attached room.

    A proposed move is rejected (and the other room tried) if it would:
    - detach the room from a wall it is currently touching
    - separate a graph-connected neighbour
    - push the room outside the boundary bounding box

    Args:
        boxes: Nx4 array of boxes [[x1, y1, x2, y2], ...]
        room_types: Room type indices (0=Living, 3=Bathroom, etc.)
        boundary: Boundary polygon for wall attachment checking and extents
        edges: Adjacency graph — connected pairs may overlap; also used to
               prevent separating graph neighbours
        verbose: Print debug information

    Returns:
        resolved_boxes: Nx4 array with overlaps resolved
    """
    n_boxes = len(boxes)
    resolved_boxes = boxes.copy()

    h_segments, v_segments = extract_boundary_segments(boundary)

    # Boundary bounding box — used to prevent resolved moves from escaping
    bnd_pts = boundary[:, :2]
    bnd_x_min = float(np.min(bnd_pts[:, 0]))
    bnd_x_max = float(np.max(bnd_pts[:, 0]))
    bnd_y_min = float(np.min(bnd_pts[:, 1]))
    bnd_y_max = float(np.max(bnd_pts[:, 1]))

    # Build a set of connected room-index pairs so we can skip them quickly.
    # Rooms that share a graph edge may intentionally overlap (e.g. en-suite
    # bathroom inside its bedroom). Store as (min, max) tuples for
    # order-independent lookup.
    connected_pairs: set = set()
    if edges is not None:
        for edge in edges:
            a, b = int(edge[0]), int(edge[1])
            if a < n_boxes and b < n_boxes:
                connected_pairs.add((min(a, b), max(a, b)))

    if verbose:
        print(f"\n{'='*60}")
        print(f"RESOLVING ROOM OVERLAPS - {n_boxes} boxes")
        print(f"  Boundary extents: X=[{bnd_x_min:.2f}, {bnd_x_max:.2f}], Y=[{bnd_y_min:.2f}, {bnd_y_max:.2f}]")
        print(f"{'='*60}")

    room_type_names = {0: "LivingRoom", 1: "MasterRoom", 2: "Kitchen", 3: "Bathroom",
                       4: "DiningRoom", 5: "ChildRoom", 6: "StudyRoom", 7: "SecondRoom",
                       8: "GuestRoom", 9: "Balcony", 10: "Entrance", 11: "Storage", 12: "Wall"}

    n_resolved = 0

    # Check every pair of rooms for overlaps
    for i in range(n_boxes):
        # Living room is the background layer — skip entirely
        if room_types[i] == 0:
            continue

        box_i = Box.from_array(resolved_boxes[i])

        for j in range(i + 1, n_boxes):
            # Living room is the background layer — skip
            if room_types[j] == 0:
                continue

            # Connected rooms may intentionally overlap (one accessed via the
            # other, e.g. en-suite bathroom inside bedroom)
            if (min(i, j), max(i, j)) in connected_pairs:
                continue

            box_j = Box.from_array(resolved_boxes[j])

            # Calculate overlap
            overlap_x = min(box_i.x2, box_j.x2) - max(box_i.x1, box_j.x1)
            overlap_y = min(box_i.y2, box_j.y2) - max(box_i.y1, box_j.y1)

            if overlap_x <= 0 or overlap_y <= 0:
                continue  # No overlap

            if verbose:
                name_i = room_type_names.get(room_types[i], f"Type{room_types[i]}")
                name_j = room_type_names.get(room_types[j], f"Type{room_types[j]}")
                print(f"\n  Overlap: Box {i} ({name_i}) ↔ Box {j} ({name_j})")
                print(f"    Overlap: x={overlap_x:.2f}px, y={overlap_y:.2f}px")

            # Determine which room to move: prefer moving the one with fewer wall attachments
            alignments_i = find_closest_segments(box_i, h_segments, v_segments)
            alignments_j = find_closest_segments(box_j, h_segments, v_segments)
            walls_i = sum(1 for a in alignments_i.values() if a.distance < 0.1)
            walls_j = sum(1 for a in alignments_j.values() if a.distance < 0.1)

            # Build candidate move orders.
            # Order by fewer-walls room first, then try the alternate axis as
            # a fallback so that rooms stuck between others on the primary axis
            # can escape horizontally (or vertically).
            # Axis 'min' = minimum-overlap axis (preferred), 'alt' = the other.
            min_axis = 'horizontal' if overlap_x <= overlap_y else 'vertical'
            alt_axis = 'vertical' if min_axis == 'horizontal' else 'horizontal'

            if walls_j < walls_i:
                room_order = [(j, i, box_j, box_i, alignments_j),
                              (i, j, box_i, box_j, alignments_i)]
            else:
                room_order = [(i, j, box_i, box_j, alignments_i),
                              (j, i, box_j, box_i, alignments_j)]

            # Try min-axis first for both rooms, then alt-axis for both rooms
            candidates = [(r + (min_axis,)) for r in room_order] + \
                         [(r + (alt_axis,)) for r in room_order]

            resolved = False
            for move_idx, stay_idx, move_box, stay_box, move_alignments, axis in candidates:
                # Determine snap direction along the chosen axis
                if axis == 'horizontal':
                    move_center_x = (move_box.x1 + move_box.x2) / 2
                    stay_center_x = (stay_box.x1 + stay_box.x2) / 2

                    if move_center_x < stay_center_x:
                        snap_delta = stay_box.x1 - move_box.x2
                        edge_to_check = 'right'
                        snap_value = stay_box.x1
                    else:
                        snap_delta = stay_box.x2 - move_box.x1
                        edge_to_check = 'left'
                        snap_value = stay_box.x2

                    new_move_box = Box(move_box.x1 + snap_delta, move_box.y1,
                                      move_box.x2 + snap_delta, move_box.y2)
                else:
                    move_center_y = (move_box.y1 + move_box.y2) / 2
                    stay_center_y = (stay_box.y1 + stay_box.y2) / 2

                    if move_center_y < stay_center_y:
                        snap_delta = stay_box.y1 - move_box.y2
                        edge_to_check = 'bottom'
                        snap_value = stay_box.y1
                    else:
                        snap_delta = stay_box.y2 - move_box.y1
                        edge_to_check = 'top'
                        snap_value = stay_box.y2

                    new_move_box = Box(move_box.x1, move_box.y1 + snap_delta,
                                      move_box.x2, move_box.y2 + snap_delta)

                # Check if this would detach from walls
                would_detach, detached_edges = would_detach_from_walls(
                    move_box, edge_to_check, snap_value, move_alignments, tolerance=0.1
                )

                if would_detach:
                    if verbose:
                        name_move = room_type_names.get(room_types[move_idx], f"Type{room_types[move_idx]}")
                        print(f"    ⊗ Cannot move Box {move_idx} ({name_move}) - would detach from: {', '.join(detached_edges)}, trying other room...")
                    continue  # Try the other room in the pair

                # Reject if this move would separate a graph-connected neighbour
                if edges is not None and _would_separate_graph_neighbors(
                        move_idx, new_move_box, resolved_boxes, edges):
                    if verbose:
                        name_move = room_type_names.get(room_types[move_idx], f"Type{room_types[move_idx]}")
                        print(f"    ⊙ Cannot move Box {move_idx} ({name_move}) - would separate a graph neighbour, trying other room...")
                    continue  # Try the other room in the pair

                # Reject if this move would push the room outside the boundary
                if (new_move_box.x1 < bnd_x_min or new_move_box.x2 > bnd_x_max or
                        new_move_box.y1 < bnd_y_min or new_move_box.y2 > bnd_y_max):
                    if verbose:
                        name_move = room_type_names.get(room_types[move_idx], f"Type{room_types[move_idx]}")
                        print(f"    ⊗ Cannot move Box {move_idx} ({name_move}) - would escape boundary "
                              f"[{new_move_box.x1:.1f},{new_move_box.y1:.1f},{new_move_box.x2:.1f},{new_move_box.y2:.1f}], trying other room...")
                    continue  # Try the other room in the pair

                # Apply the resolution
                resolved_boxes[move_idx] = new_move_box.to_array()

                # Update local box references so subsequent pairs use new positions
                if move_idx == i:
                    box_i = new_move_box
                else:
                    box_j = new_move_box

                n_resolved += 1
                resolved = True
                if verbose:
                    name_move = room_type_names.get(room_types[move_idx], f"Type{room_types[move_idx]}")
                    axis_label = f"({axis})" if axis != min_axis else ""
                    print(f"    ✓ Moved Box {move_idx} ({name_move}) by {snap_delta:.2f}px {axis_label} to resolve overlap")
                break  # Overlap resolved, move on

            if not resolved and verbose:
                name_i = room_type_names.get(room_types[i], f"Type{room_types[i]}")
                name_j = room_type_names.get(room_types[j], f"Type{room_types[j]}")
                print(f"    ✗ Could not resolve overlap between Box {i} ({name_i}) and Box {j} ({name_j}) - both rooms locked by walls")

    if verbose:
        print(f"\n{'='*60}")
        print(f"✓ Resolved {n_resolved} overlapping room pairs")
        print(f"{'='*60}\n")

    return resolved_boxes


def snap_rooms_to_neighbors(boxes: np.ndarray,
                            room_types: np.ndarray,
                            edges: np.ndarray,
                            boundary: np.ndarray,
                            threshold: float = 8.0,
                            verbose: bool = False) -> np.ndarray:
    """
    Pass 3: Snap rooms to their nearest neighboring rooms.

    This function moves rooms to align with adjacent room boundaries, but only if:
    1. The room is NOT a living room (type 0)
    2. The room is connected to fewer than 2 walls (if connected to 2+ walls, it's locked)
    3. The movement won't detach the room from walls it's currently attached to

    Args:
        boxes: Nx4 array of boxes [[x1, y1, x2, y2], ...]
        room_types: Room type indices (0=Living, 1=Bedroom, etc.)
        edges: Adjacency graph, shape (K, 2) or (K, 3)
               Format: [[room_i, room_j], ...] or [[room_i, room_j, spatial_type], ...]
        boundary: Boundary polygon for wall attachment checking
        threshold: Snapping threshold in pixels
        verbose: Print debug information

    Returns:
        snapped_boxes: Nx4 array of boxes with room-to-room snapping applied
    """
    n_boxes = len(boxes)
    snapped_boxes = boxes.copy()

    # Extract boundary segments for wall attachment checking
    h_segments, v_segments = extract_boundary_segments(boundary)

    if verbose:
        print(f"\n{'='*60}")
        print(f"ROOM-TO-ROOM SNAPPING - {n_boxes} boxes")
        print(f"{'='*60}")

    # Build adjacency list from edges
    adjacency = {i: set() for i in range(n_boxes)}
    for edge in edges:
        room_i, room_j = int(edge[0]), int(edge[1])
        if room_i < n_boxes and room_j < n_boxes:
            adjacency[room_i].add(room_j)
            adjacency[room_j].add(room_i)

    # --- Same-type overlap resolution ---
    # If two rooms of the same type currently overlap, push the more-mobile
    # one along the axis of minimum overlap until they share an edge.
    # Runs over ALL pairs (not just graph neighbours) before the main pass.
    rtype_names = {0: "LivingRoom", 1: "MasterRoom", 2: "Kitchen", 3: "Bathroom",
                   4: "DiningRoom", 5: "ChildRoom", 6: "StudyRoom", 7: "SecondRoom",
                   8: "GuestRoom", 9: "Balcony", 10: "Entrance", 11: "Storage", 12: "Wall"}
    n_same_type_resolved = 0
    for i in range(n_boxes):
        if room_types[i] == 0:
            continue  # Living room is the background layer
        box_i = Box.from_array(snapped_boxes[i])
        for j in range(i + 1, n_boxes):
            if room_types[j] != room_types[i]:
                continue  # Only same-type pairs
            box_j = Box.from_array(snapped_boxes[j])
            overlap_x = min(box_i.x2, box_j.x2) - max(box_i.x1, box_j.x1)
            overlap_y = min(box_i.y2, box_j.y2) - max(box_i.y1, box_j.y1)
            if overlap_x <= 0 or overlap_y <= 0:
                continue  # Not overlapping
            if verbose:
                rname = rtype_names.get(room_types[i], f"Type{room_types[i]}")
                print(f"\n  [Same-type overlap] Box {i} ({rname}) ↔ Box {j} ({rname})"
                      f"  x={overlap_x:.2f}px  y={overlap_y:.2f}px")
            aligns_i = find_closest_segments(box_i, h_segments, v_segments)
            aligns_j = find_closest_segments(box_j, h_segments, v_segments)
            walls_i = sum(1 for a in aligns_i.values() if a.distance < 0.1)
            walls_j = sum(1 for a in aligns_j.values() if a.distance < 0.1)
            # Choose resolution axis (minimum overlap)
            if overlap_x <= overlap_y:
                # Horizontal: determine which room is left vs right
                if (box_i.x1 + box_i.x2) < (box_j.x1 + box_j.x2):
                    delta_i = box_j.x1 - box_i.x2   # move i left  (negative)
                    delta_j = box_i.x2 - box_j.x1   # move j right (positive)
                else:
                    delta_i = box_j.x2 - box_i.x1   # move i right (positive)
                    delta_j = box_i.x1 - box_j.x2   # move j left  (negative)
                axis = 'horizontal'
            else:
                # Vertical: determine which room is above vs below
                if (box_i.y1 + box_i.y2) < (box_j.y1 + box_j.y2):
                    delta_i = box_j.y1 - box_i.y2   # move i up   (negative)
                    delta_j = box_i.y2 - box_j.y1   # move j down (positive)
                else:
                    delta_i = box_j.y2 - box_i.y1   # move i down (positive)
                    delta_j = box_i.y1 - box_j.y2   # move j up   (negative)
                axis = 'vertical'
            # Prefer moving the room with fewer wall attachments
            if walls_j <= walls_i:
                move_idx, delta, move_box, move_aligns = j, delta_j, box_j, aligns_j
            else:
                move_idx, delta, move_box, move_aligns = i, delta_i, box_i, aligns_i
            if axis == 'horizontal':
                new_box = Box(move_box.x1 + delta, move_box.y1,
                              move_box.x2 + delta, move_box.y2)
                edge_to_check = 'right' if delta > 0 else 'left'
                snap_val = new_box.x2 if delta > 0 else new_box.x1
            else:
                new_box = Box(move_box.x1, move_box.y1 + delta,
                              move_box.x2, move_box.y2 + delta)
                edge_to_check = 'bottom' if delta > 0 else 'top'
                snap_val = new_box.y2 if delta > 0 else new_box.y1
            would_detach, detached = would_detach_from_walls(
                move_box, edge_to_check, snap_val, move_aligns, tolerance=0.1
            )
            if would_detach:
                if verbose:
                    rname = rtype_names.get(room_types[move_idx], f"Type{room_types[move_idx]}")
                    print(f"    ⊗ Cannot move Box {move_idx} ({rname}) - would detach from: {', '.join(detached)}")
                continue
            snapped_boxes[move_idx] = new_box.to_array()
            n_same_type_resolved += 1
            # Keep local box references in sync for subsequent j iterations
            if move_idx == i:
                box_i = new_box
            else:
                box_j = new_box
            if verbose:
                rname = rtype_names.get(room_types[move_idx], f"Type{room_types[move_idx]}")
                print(f"    ✓ Moved Box {move_idx} ({rname}) by {abs(delta):.2f}px to share edge")

    if verbose and n_same_type_resolved > 0:
        print(f"\n  [Same-type overlap] Resolved {n_same_type_resolved} pair(s)")

    # Process each room
    n_snapped = 0
    for i in range(n_boxes):
        box = Box.from_array(snapped_boxes[i])
        room_type = room_types[i]

        # Skip living room (type 0)
        if room_type == 0:
            if verbose:
                print(f"\n--- Box {i}: LivingRoom ---")
                print(f"  ⊘ Skipping (living room excluded from room-to-room snapping)")
            continue

        if verbose:
            room_type_names = {0: "LivingRoom", 1: "MasterRoom", 2: "Kitchen", 3: "Bathroom",
                             4: "DiningRoom", 5: "ChildRoom", 6: "StudyRoom", 7: "SecondRoom",
                             8: "GuestRoom", 9: "Balcony", 10: "Entrance", 11: "Storage", 12: "Wall"}
            room_name = room_type_names.get(room_type, f"Unknown({room_type})")
            print(f"\n--- Box {i}: {room_name} ---")

        # Check how many walls this room is attached to
        alignments = find_closest_segments(box, h_segments, v_segments)
        n_walls_attached = sum(1 for alignment in alignments.values() if alignment.distance < 0.1)

        # Find neighbor rooms (needed before the n_walls_attached check below)
        neighbors = adjacency.get(i, set())

        if n_walls_attached >= 2:
            # Check if there are any non-touching graph neighbors (excluding living room)
            # If there's a gap to a neighbor, allow movement to close the gap
            has_gap_to_neighbor = False
            gap_threshold = 5.0  # Consider rooms "touching" if closer than this

            for neighbor_idx in neighbors:
                # Skip living room — unless this room is a bathroom.
                # Bathrooms need a direct connection to the living room, so a
                # gap toward the living room should unlock them even when they
                # are touching 2 walls.
                if room_types[neighbor_idx] == 0 and room_type != 3:
                    continue

                neighbor_box = Box.from_array(snapped_boxes[neighbor_idx])

                # Calculate minimum distance between the two boxes
                # If boxes overlap or touch, distance is 0 or negative
                # If there's a gap, distance is positive

                # Check horizontal gap
                if box.x2 < neighbor_box.x1:  # Box is to the left of neighbor
                    h_gap = neighbor_box.x1 - box.x2
                elif neighbor_box.x2 < box.x1:  # Box is to the right of neighbor
                    h_gap = box.x1 - neighbor_box.x2
                else:  # Horizontal overlap
                    h_gap = 0

                # Check vertical gap
                if box.y2 < neighbor_box.y1:  # Box is above neighbor
                    v_gap = neighbor_box.y1 - box.y2
                elif neighbor_box.y2 < box.y1:  # Box is below neighbor
                    v_gap = box.y1 - neighbor_box.y2
                else:  # Vertical overlap
                    v_gap = 0

                # Minimum distance is the gap (0 if touching/overlapping)
                min_dist = max(h_gap, v_gap) if (h_gap > 0 and v_gap > 0) else max(h_gap, v_gap, 0)

                # If there's a significant gap, allow movement
                if min_dist > gap_threshold:
                    has_gap_to_neighbor = True
                    if verbose:
                        print(f"    Gap to neighbor {neighbor_idx}: {min_dist:.2f}px")
                    break

            if not has_gap_to_neighbor:
                if verbose:
                    print(f"  ⊗ Skipping (attached to {n_walls_attached} walls, all non-living neighbors touching)")
                continue
            else:
                if verbose:
                    print(f"  ✓ Attached to {n_walls_attached} walls but has gap to neighbor, allowing movement")

        if not neighbors:
            if verbose:
                print(f"  ⊗ Skipping (no neighbors in adjacency graph)")
            continue

        if verbose:
            print(f"  Neighbors: {list(neighbors)}, Attached to {n_walls_attached} wall(s)")

        # Find closest neighbor edge
        best_edge = None
        best_distance = float('inf')
        best_snap_delta = None
        best_axis = None
        best_bypass_axis = False
        best_neighbor_box = None
        best_already_overlapping = False

        # For bathrooms connected to a living room, prioritise the LR snap.
        # Only fall back to non-LR neighbours if no LR snap is found.
        # This prevents a closer bedroom from winning the distance comparison
        # and causing the bathroom to snap toward the bedroom instead of the LR.
        if room_type == 3:
            lr_nbrs = [nb for nb in neighbors if room_types[nb] == 0]
            other_nbrs = [nb for nb in neighbors if room_types[nb] != 0]
            ordered_neighbors = lr_nbrs + other_nbrs
            lr_boundary = len(lr_nbrs)
        else:
            ordered_neighbors = list(neighbors)
            lr_boundary = len(ordered_neighbors)

        for _enum_idx, neighbor_idx in enumerate(ordered_neighbors):
            # Bathrooms: once we've found a snap from an LR neighbour,
            # don't let a closer non-LR neighbour override it.
            if room_type == 3 and _enum_idx >= lr_boundary and best_edge is not None:
                break
            neighbor_box = Box.from_array(snapped_boxes[neighbor_idx])

            # Check if already overlapping with this neighbor
            already_overlapping = (
                box.x1 < neighbor_box.x2 and box.x2 > neighbor_box.x1 and
                box.y1 < neighbor_box.y2 and box.y2 > neighbor_box.y1
            )
            # Bathrooms not already inside a room should only use "aligning" snaps
            # (edge-to-edge), not "matching" snaps (same-edge-to-same-edge), which
            # would pull the bathroom inside the neighbor.
            allow_matching_snaps = not (room_type == 3 and not already_overlapping)

            # Bathrooms always snap to a graph-connected living room regardless of
            # distance — they must stay accessible from the living room, so the
            # normal threshold is waived for this specific pair.
            # Exception: if the bathroom already overlaps the living room, it is
            # already accessible and snapping would only move it to align with
            # the living room's far edge (potentially pushing it toward another
            # room). In that case skip the living room as a snap target entirely.
            if room_type == 3 and room_types[neighbor_idx] == 0 and already_overlapping:
                continue
            eff_threshold = (
                float('inf')
                if room_type == 3 and room_types[neighbor_idx] == 0
                else threshold
            )

            # For bathroom→living room snaps the axis-overlap guards are bypassed
            # entirely.  The living room is typically far away and diagonally
            # placed, so the guards would always block the snap even though the
            # threshold is already infinite.  The bathroom must reach the living
            # room regardless of whether they currently share any axis extent.
            bypass_axis = (room_type == 3 and room_types[neighbor_idx] == 0)

            # Calculate distances to each edge of the neighbor
            # Left edge of neighbor (vertical line at neighbor.x1)
            if bypass_axis or (box.y2 > neighbor_box.y1 and box.y1 < neighbor_box.y2):  # Vertical overlap
                dist_to_left = abs(box.x2 - neighbor_box.x1)  # Our right edge to their left edge
                dist_from_left = abs(box.x1 - neighbor_box.x1)  # Our left edge to their left edge

                if dist_to_left < best_distance and dist_to_left <= eff_threshold:
                    best_distance = dist_to_left
                    best_edge = 'right_to_left'
                    best_snap_delta = neighbor_box.x1 - box.x2
                    best_axis = 'horizontal'
                    best_bypass_axis = bypass_axis
                    best_neighbor_box = neighbor_box
                    best_already_overlapping = already_overlapping

                if allow_matching_snaps and dist_from_left < best_distance and dist_from_left <= eff_threshold:
                    best_distance = dist_from_left
                    best_edge = 'left_to_left'
                    best_snap_delta = neighbor_box.x1 - box.x1
                    best_axis = 'horizontal'
                    best_bypass_axis = bypass_axis
                    best_neighbor_box = neighbor_box
                    best_already_overlapping = already_overlapping

            # Right edge of neighbor (vertical line at neighbor.x2)
            if bypass_axis or (box.y2 > neighbor_box.y1 and box.y1 < neighbor_box.y2):  # Vertical overlap
                dist_to_right = abs(box.x1 - neighbor_box.x2)  # Our left edge to their right edge
                dist_from_right = abs(box.x2 - neighbor_box.x2)  # Our right edge to their right edge

                if dist_to_right < best_distance and dist_to_right <= eff_threshold:
                    best_distance = dist_to_right
                    best_edge = 'left_to_right'
                    best_snap_delta = neighbor_box.x2 - box.x1
                    best_axis = 'horizontal'
                    best_bypass_axis = bypass_axis
                    best_neighbor_box = neighbor_box
                    best_already_overlapping = already_overlapping

                if allow_matching_snaps and dist_from_right < best_distance and dist_from_right <= eff_threshold:
                    best_distance = dist_from_right
                    best_edge = 'right_to_right'
                    best_snap_delta = neighbor_box.x2 - box.x2
                    best_axis = 'horizontal'
                    best_bypass_axis = bypass_axis
                    best_neighbor_box = neighbor_box
                    best_already_overlapping = already_overlapping

            # Top edge of neighbor (horizontal line at neighbor.y1)
            if bypass_axis or (box.x2 > neighbor_box.x1 and box.x1 < neighbor_box.x2):  # Horizontal overlap
                dist_to_top = abs(box.y2 - neighbor_box.y1)  # Our bottom edge to their top edge
                dist_from_top = abs(box.y1 - neighbor_box.y1)  # Our top edge to their top edge

                if dist_to_top < best_distance and dist_to_top <= eff_threshold:
                    best_distance = dist_to_top
                    best_edge = 'bottom_to_top'
                    best_snap_delta = neighbor_box.y1 - box.y2
                    best_axis = 'vertical'
                    best_bypass_axis = bypass_axis
                    best_neighbor_box = neighbor_box
                    best_already_overlapping = already_overlapping

                if allow_matching_snaps and dist_from_top < best_distance and dist_from_top <= eff_threshold:
                    best_distance = dist_from_top
                    best_edge = 'top_to_top'
                    best_snap_delta = neighbor_box.y1 - box.y1
                    best_axis = 'vertical'
                    best_bypass_axis = bypass_axis
                    best_neighbor_box = neighbor_box
                    best_already_overlapping = already_overlapping

            # Bottom edge of neighbor (horizontal line at neighbor.y2)
            if bypass_axis or (box.x2 > neighbor_box.x1 and box.x1 < neighbor_box.x2):  # Horizontal overlap
                dist_to_bottom = abs(box.y1 - neighbor_box.y2)  # Our top edge to their bottom edge
                dist_from_bottom = abs(box.y2 - neighbor_box.y2)  # Our bottom edge to their bottom edge

                if dist_to_bottom < best_distance and dist_to_bottom <= eff_threshold:
                    best_distance = dist_to_bottom
                    best_edge = 'top_to_bottom'
                    best_snap_delta = neighbor_box.y2 - box.y1
                    best_axis = 'vertical'
                    best_bypass_axis = bypass_axis
                    best_neighbor_box = neighbor_box
                    best_already_overlapping = already_overlapping

                if allow_matching_snaps and dist_from_bottom < best_distance and dist_from_bottom <= eff_threshold:
                    best_distance = dist_from_bottom
                    best_edge = 'bottom_to_bottom'
                    best_snap_delta = neighbor_box.y2 - box.y2
                    best_axis = 'vertical'
                    best_bypass_axis = bypass_axis
                    best_neighbor_box = neighbor_box
                    best_already_overlapping = already_overlapping

        if best_edge is None:
            if verbose:
                print(f"  ⊗ No neighbor edges within threshold ({threshold}px)")
            continue

        if verbose:
            print(f"  Closest neighbor edge: {best_edge}, distance={best_distance:.2f}px, delta={best_snap_delta:.2f}px")

        # Calculate new box position after snapping.
        # For bathroom→LR diagonal snaps: if the rooms have no perpendicular overlap
        # (they are diagonally placed), a single-axis snap lands the bathroom at LR's
        # corner — touching at one point but sharing no wall segment.  Apply a small
        # perpendicular correction so the bathroom enters the LR corner by 2px,
        # creating an actual shared wall segment for door placement.
        perp_correction = 0
        if best_bypass_axis and not best_already_overlapping and best_neighbor_box is not None:
            if best_axis == 'vertical':
                x_overlap = min(box.x2, best_neighbor_box.x2) - max(box.x1, best_neighbor_box.x1)
                if x_overlap <= 0:
                    if box.x1 >= best_neighbor_box.x2:    # bathroom to the right of LR
                        perp_correction = best_neighbor_box.x2 - box.x1 - 2
                    elif box.x2 <= best_neighbor_box.x1:  # bathroom to the left of LR
                        perp_correction = best_neighbor_box.x1 - box.x2 + 2
            else:  # horizontal
                y_overlap = min(box.y2, best_neighbor_box.y2) - max(box.y1, best_neighbor_box.y1)
                if y_overlap <= 0:
                    if box.y1 >= best_neighbor_box.y2:    # bathroom below LR
                        perp_correction = best_neighbor_box.y2 - box.y1 - 2
                    elif box.y2 <= best_neighbor_box.y1:  # bathroom above LR
                        perp_correction = best_neighbor_box.y1 - box.y2 + 2

        if perp_correction != 0 and verbose:
            print(f"  ↗ Diagonal correction applied: {perp_correction:.1f}px on perpendicular axis")

        if best_axis == 'horizontal':
            new_box = Box(box.x1 + best_snap_delta, box.y1 + perp_correction,
                         box.x2 + best_snap_delta, box.y2 + perp_correction)
        else:  # vertical
            new_box = Box(box.x1 + perp_correction, box.y1 + best_snap_delta,
                         box.x2 + perp_correction, box.y2 + best_snap_delta)

        # Reject if this snap would separate a currently-touching graph neighbour
        if _would_separate_graph_neighbors(i, new_box, snapped_boxes, edges):
            if verbose:
                print(f"  ⊙ Snap rejected: would separate a touching graph neighbour")
            continue

        # Check if this would detach from walls
        # We need to check which edge corresponds to the snap direction
        if best_axis == 'horizontal':
            # Horizontal movement - check if we're detaching from left/right walls
            if best_snap_delta > 0:
                edge_to_check = 'right'
            else:
                edge_to_check = 'left'
        else:
            # Vertical movement - check if we're detaching from top/bottom walls
            if best_snap_delta > 0:
                edge_to_check = 'bottom'
            else:
                edge_to_check = 'top'

        # Use the wall snap value from current alignments
        snap_value = new_box.x1 if edge_to_check == 'left' else \
                     new_box.x2 if edge_to_check == 'right' else \
                     new_box.y1 if edge_to_check == 'top' else new_box.y2

        would_detach, detached_edges = would_detach_from_walls(
            box, edge_to_check, snap_value, alignments, tolerance=0.1
        )

        if would_detach:
            if verbose:
                print(f"  ⊗ Skipping (would detach from walls: {', '.join(detached_edges)})")
            continue

        # Check if the snap would create new overlaps with non-bathroom rooms.
        # For bathrooms: instead of blocking, shrink the box so its leading edge
        # touches the blocker's face (the non-leading edge still moves with the snap).
        # For other room types: block as before.
        blocker_box = None
        blocker_idx = -1
        room_type_names_local = {0: "LivingRoom", 1: "MasterRoom", 2: "Kitchen",
                                  3: "Bathroom", 4: "DiningRoom", 5: "ChildRoom"}
        for k in range(len(snapped_boxes)):
            if k == i:
                continue
            if room_types[k] == 3:  # Bathrooms can overlap anything
                continue
            if room_types[k] == 0:  # Living room handled separately
                continue

            other_box = Box.from_array(snapped_boxes[k])

            new_overlaps = (new_box.x1 < other_box.x2 and new_box.x2 > other_box.x1 and
                            new_box.y1 < other_box.y2 and new_box.y2 > other_box.y1)
            currently_overlaps = (box.x1 < other_box.x2 and box.x2 > other_box.x1 and
                                  box.y1 < other_box.y2 and box.y2 > other_box.y1)

            if new_overlaps and not currently_overlaps:
                blocker_box = other_box
                blocker_idx = k
                if verbose:
                    name_k = room_type_names_local.get(room_types[k], f"Type{room_types[k]}")
                    print(f"  ⚠ Snap would create new overlap with Box {k} ({name_k})")
                break

        if blocker_box is not None:
            if room_type == 3:
                # Bathroom: shrink to fit — clamp the leading edge to the blocker's
                # face while the trailing edge still moves with the snap.
                if best_axis == 'vertical':
                    if best_snap_delta > 0:  # Moving down — clamp bottom to blocker top
                        shrunk = Box(new_box.x1, new_box.y1, new_box.x2, blocker_box.y1)
                    else:                    # Moving up   — clamp top to blocker bottom
                        shrunk = Box(new_box.x1, blocker_box.y2, new_box.x2, new_box.y2)
                else:
                    if best_snap_delta > 0:  # Moving right — clamp right to blocker left
                        shrunk = Box(new_box.x1, new_box.y1, blocker_box.x1, new_box.y2)
                    else:                    # Moving left  — clamp left to blocker right
                        shrunk = Box(blocker_box.x2, new_box.y1, new_box.x2, new_box.y2)

                if shrunk.x2 > shrunk.x1 and shrunk.y2 > shrunk.y1:
                    snapped_boxes[i] = shrunk.to_array()
                    n_snapped += 1
                    if verbose:
                        name_k = room_type_names_local.get(room_types[blocker_idx],
                                                           f"Type{room_types[blocker_idx]}")
                        print(f"  ✓ Bathroom shrunk to touch Box {blocker_idx} ({name_k}) edge "
                              f"(moved {abs(best_snap_delta):.2f}px, clamped to fit)")
                else:
                    if verbose:
                        print(f"  ⊗ Skipping (shrunk box would have zero size)")
            else:
                if verbose:
                    name_k = room_type_names_local.get(room_types[blocker_idx],
                                                       f"Type{room_types[blocker_idx]}")
                    print(f"  ⊗ Skipping (would create new overlap with Box {blocker_idx} ({name_k}))")
            continue

        # Apply the snap
        snapped_boxes[i] = new_box.to_array()
        n_snapped += 1

        if verbose:
            print(f"  ✓ Snapped to neighbor (moved {abs(best_snap_delta):.2f}px along {best_axis} axis)")

    if verbose:
        print(f"\n{'='*60}")
        print(f"✓ Snapped {n_snapped}/{n_boxes} boxes to neighbors")
        print(f"{'='*60}\n")

    return snapped_boxes


def snap_bathroom_to_nearest_clear_wall(
        boxes: np.ndarray,
        room_types: np.ndarray,
        edges: np.ndarray,
        boundary: np.ndarray,
        verbose: bool = False) -> np.ndarray:
    """
    For each bathroom that is graph-connected to the living room, find the
    closest boundary wall with a clear line of sight (no other room in between)
    and snap the bathroom flush to it.  Distance is unlimited.

    The LR itself counts as a blocker, so the bathroom will not be moved through
    the room it is already adjacent to — only free directions are considered.

    Args:
        boxes:      Nx4 array [[x1, y1, x2, y2], ...]
        room_types: N-length array of room type indices (0=LR, 3=Bathroom, ...)
        edges:      Graph adjacency edges, shape (K, 2) or (K, 3)
        boundary:   Boundary polygon, shape (M, 2+)
        verbose:    Print debug info

    Returns:
        Updated Nx4 box array.
    """
    n_boxes = len(boxes)
    result = boxes.copy()
    h_segments, v_segments = extract_boundary_segments(boundary)

    # Build the set of bathrooms that have a graph edge to any living room
    lr_connected_bathrooms: set = set()
    for edge in edges:
        a, b = int(edge[0]), int(edge[1])
        if a < n_boxes and b < n_boxes:
            if room_types[a] == 3 and room_types[b] == 0:
                lr_connected_bathrooms.add(a)
            elif room_types[b] == 3 and room_types[a] == 0:
                lr_connected_bathrooms.add(b)

    room_type_names = {
        0: "LivingRoom", 1: "MasterRoom", 2: "Kitchen", 3: "Bathroom",
        4: "DiningRoom", 5: "ChildRoom", 6: "StudyRoom", 7: "SecondRoom",
        8: "GuestRoom", 9: "Balcony", 10: "Entrance", 11: "Storage", 12: "Wall",
    }

    def _rname(t: int) -> str:
        return room_type_names.get(int(t), f"type{t}")

    if verbose:
        print(f"\n{'='*60}")
        print(f"BATHROOM→WALL ANCHORING  "
              f"({len(lr_connected_bathrooms)} LR-connected bathroom(s))")
        print(f"{'='*60}")

    for i in lr_connected_bathrooms:
        box = Box.from_array(result[i])

        if verbose:
            print(f"\n--- Box {i}: Bathroom  "
                  f"({box.x1:.1f}, {box.y1:.1f}, {box.x2:.1f}, {box.y2:.1f}) ---")

        # Ray-cast from bathroom centre to find boundary walls in 4 directions
        walls = ray_cast_to_walls(box, h_segments, v_segments)

        # Pre-build other boxes once (all rooms, including LR, are potential blockers)
        other_boxes_with_idx = [(k, Box.from_array(result[k]))
                                for k in range(n_boxes) if k != i]
        other_boxes = [rb for _, rb in other_boxes_with_idx]

        # Collect the LR boxes that this bathroom is graph-connected to
        connected_lr_boxes = []
        for edge in edges:
            a, b = int(edge[0]), int(edge[1])
            if a == i and b < n_boxes and room_types[b] == 0:
                connected_lr_boxes.append(Box.from_array(result[b]))
            elif b == i and a < n_boxes and room_types[a] == 0:
                connected_lr_boxes.append(Box.from_array(result[a]))

        best_direction = None
        best_gap = float('inf')
        best_dx = 0.0
        best_dy = 0.0

        for direction, wall_seg in walls.items():
            if wall_seg is None:
                continue

            if direction == 'left':
                wall_coord = wall_seg.x1
                gap = box.x1 - wall_coord
                dx, dy = wall_coord - box.x1, 0.0
                # Blocker: any room with x-extent overlapping [wall_coord, box.x1]
                # and y-extent overlapping the bathroom
                blockers = [
                    (k, rb) for k, rb in other_boxes_with_idx
                    if room_types[k] != 0  # LR expands in Pass 5; not a real blocker
                    and rb.x2 > wall_coord and rb.x1 < box.x1
                    and rb.y1 < box.y2 and rb.y2 > box.y1
                ]
            elif direction == 'right':
                wall_coord = wall_seg.x1
                gap = wall_coord - box.x2
                dx, dy = wall_coord - box.x2, 0.0
                blockers = [
                    (k, rb) for k, rb in other_boxes_with_idx
                    if room_types[k] != 0
                    and rb.x1 < wall_coord and rb.x2 > box.x2
                    and rb.y1 < box.y2 and rb.y2 > box.y1
                ]
            elif direction == 'top':
                wall_coord = wall_seg.y1
                gap = box.y1 - wall_coord
                dx, dy = 0.0, wall_coord - box.y1
                blockers = [
                    (k, rb) for k, rb in other_boxes_with_idx
                    if room_types[k] != 0
                    and rb.y2 > wall_coord and rb.y1 < box.y1
                    and rb.x1 < box.x2 and rb.x2 > box.x1
                ]
            else:  # bottom
                wall_coord = wall_seg.y1
                gap = wall_coord - box.y2
                dx, dy = 0.0, wall_coord - box.y2
                blockers = [
                    (k, rb) for k, rb in other_boxes_with_idx
                    if room_types[k] != 0
                    and rb.y1 < wall_coord and rb.y2 > box.y2
                    and rb.x1 < box.x2 and rb.x2 > box.x1
                ]

            blocked = bool(blockers)

            if gap < 0:
                if verbose:
                    print(f"  ⊗ {direction}: bathroom already past wall (gap={gap:.1f}px)")
                continue

            if blocked:
                if verbose:
                    blocker_strs = [
                        f"room {k} {_rname(room_types[k])}"
                        f" ({rb.x1:.0f},{rb.y1:.0f}→{rb.x2:.0f},{rb.y2:.0f})"
                        for k, rb in blockers
                    ]
                    print(f"  ⊗ {direction}: blocked by {', '.join(blocker_strs)}"
                          f"  [gap to wall={gap:.1f}px]")
                continue

            # Check that the new position still touches at least one connected LR
            if connected_lr_boxes:
                tentative = Box(box.x1 + dx, box.y1 + dy, box.x2 + dx, box.y2 + dy)
                tol = 2.0
                still_connected = any(
                    (abs(tentative.y1 - lr.y2) <= tol
                     and min(tentative.x2, lr.x2) - max(tentative.x1, lr.x1) > 0)
                    or (abs(tentative.y2 - lr.y1) <= tol
                        and min(tentative.x2, lr.x2) - max(tentative.x1, lr.x1) > 0)
                    or (abs(tentative.x1 - lr.x2) <= tol
                        and min(tentative.y2, lr.y2) - max(tentative.y1, lr.y1) > 0)
                    or (abs(tentative.x2 - lr.x1) <= tol
                        and min(tentative.y2, lr.y2) - max(tentative.y1, lr.y1) > 0)
                    or (tentative.x1 < lr.x2 and tentative.x2 > lr.x1
                        and tentative.y1 < lr.y2 and tentative.y2 > lr.y1)
                    for lr in connected_lr_boxes
                )
                if not still_connected:
                    if verbose:
                        lr_gaps = []
                        for lr in connected_lr_boxes:
                            tentative = Box(box.x1 + dx, box.y1 + dy,
                                            box.x2 + dx, box.y2 + dy)
                            edge_gaps = {
                                'top':    tentative.y1 - lr.y2,
                                'bottom': lr.y1 - tentative.y2,
                                'left':   tentative.x1 - lr.x2,
                                'right':  lr.x1 - tentative.x2,
                            }
                            closest = min(edge_gaps, key=lambda d: abs(edge_gaps[d]))
                            lr_gaps.append(
                                f"LivingRoom({lr.x1:.0f},{lr.y1:.0f}→{lr.x2:.0f},{lr.y2:.0f})"
                                f" nearest={closest} gap={edge_gaps[closest]:.1f}px"
                            )
                        print(f"  ⊗ {direction}: would disconnect from LR"
                              f"  [tentative=({tentative.x1:.0f},{tentative.y1:.0f}"
                              f"→{tentative.x2:.0f},{tentative.y2:.0f})]"
                              f"  {'; '.join(lr_gaps)}")
                    continue

            if verbose:
                print(f"  ✓ {direction}: clear path, gap={gap:.1f}px")

            if gap < best_gap:
                best_gap = gap
                best_direction = direction
                best_dx = dx
                best_dy = dy

        if best_direction is None:
            if verbose:
                print(f"  ⊗ No clear path to any wall — skipping")
            continue

        new_box = Box(box.x1 + best_dx, box.y1 + best_dy,
                      box.x2 + best_dx, box.y2 + best_dy)
        result[i] = new_box.to_array()

        if verbose:
            print(f"  → Snapped {best_direction} by {best_gap:.1f}px  →  "
                  f"({new_box.x1:.1f}, {new_box.y1:.1f}, "
                  f"{new_box.x2:.1f}, {new_box.y2:.1f})")

    return result


def snap_rooms_to_fill_gaps(
        boxes: np.ndarray,
        room_types: np.ndarray,
        boundary: np.ndarray,
        gap_threshold: float = 20.0,
        wall_tol: float = 3.0,
        verbose: bool = False) -> np.ndarray:
    """
    For each non-LR room that is attached to fewer than 2 boundary walls, find
    the direction with the largest open gap to a boundary wall and snap toward it.

    The snap is only allowed along an axis that has no existing wall attachment —
    moving along a locked axis would pull the room away from its current wall.

    Args:
        boxes:         Nx4 array [[x1, y1, x2, y2], ...]
        room_types:    N-length array of room type indices (0=LR, 3=Bathroom, ...)
        boundary:      Boundary polygon, shape (M, 2+)
        gap_threshold: Only snap if the gap to the wall exceeds this (pixels)
        wall_tol:      Tolerance for detecting existing wall attachment (pixels)
        verbose:       Print debug info

    Returns:
        Updated Nx4 box array.
    """
    n_boxes = len(boxes)
    result = boxes.copy()
    h_segments, v_segments = extract_boundary_segments(boundary)

    room_type_names = {
        0: "LivingRoom", 1: "MasterRoom", 2: "Kitchen", 3: "Bathroom",
        4: "DiningRoom", 5: "ChildRoom", 6: "StudyRoom", 7: "SecondRoom",
        8: "GuestRoom", 9: "Balcony", 10: "Entrance", 11: "Storage", 12: "Wall",
    }

    def _rname(t: int) -> str:
        return room_type_names.get(int(t), f"type{t}")

    def get_wall_attachments(box: Box) -> set:
        """Return set of directions where box edge is flush with a boundary segment."""
        attached = set()
        min_overlap = 5.0  # require at least 5px of shared edge
        for seg in h_segments:
            if abs(box.y1 - seg.y1) <= wall_tol:
                if min(box.x2, max(seg.x1, seg.x2)) - max(box.x1, min(seg.x1, seg.x2)) >= min_overlap:
                    attached.add('top')
            if abs(box.y2 - seg.y1) <= wall_tol:
                if min(box.x2, max(seg.x1, seg.x2)) - max(box.x1, min(seg.x1, seg.x2)) >= min_overlap:
                    attached.add('bottom')
        for seg in v_segments:
            if abs(box.x1 - seg.x1) <= wall_tol:
                if min(box.y2, max(seg.y1, seg.y2)) - max(box.y1, min(seg.y1, seg.y2)) >= min_overlap:
                    attached.add('left')
            if abs(box.x2 - seg.x1) <= wall_tol:
                if min(box.y2, max(seg.y1, seg.y2)) - max(box.y1, min(seg.y1, seg.y2)) >= min_overlap:
                    attached.add('right')
        return attached

    def _is_perp_overlap(a: Box, b: Box, snap_dir: str, tol: float = 5.0) -> bool:
        """Bypass b only when a is already deeply inside b in the snap direction.

        For a horizontal snap (left/right): bypass only if there is already
        significant x-overlap (> tol px) AND y-overlap.  This means b has
        already penetrated well into a's x-range — snapping a further in the
        same direction does not create a *new* overlap.

        Adjacent rooms (x_overlap = 0) and rooms that merely touch (x_overlap
        ≤ tol) are NEVER bypassed — they are real blockers that prevent the
        snap from moving a through them.

        tol=5 avoids triggering on 1-3 px rounding contact from prior passes.
        """
        x_overlap = max(0.0, min(a.x2, b.x2) - max(a.x1, b.x1))
        y_overlap = max(0.0, min(a.y2, b.y2) - max(a.y1, b.y1))
        if snap_dir in ('left', 'right'):
            return x_overlap > tol and y_overlap > 0
        else:  # top / bottom
            return y_overlap > tol and x_overlap > 0

    if verbose:
        print(f"\n{'='*60}")
        print(f"GAP-FILL WALL SNAPPING  (gap_threshold={gap_threshold}px)")
        print(f"{'='*60}")

    for i in range(n_boxes):
        # Skip LR — it already fills the boundary via Pass 5
        if room_types[i] == 0:
            continue

        box = Box.from_array(result[i])
        attached = get_wall_attachments(box)

        if len(attached) >= 2:
            if verbose:
                print(f"\n  Box {i} {_rname(room_types[i])}: "
                      f"already attached to {attached} — skip")
            continue

        if verbose:
            print(f"\n  Box {i} {_rname(room_types[i])}: "
                  f"attached to {attached or 'none'}, looking for gaps...")

        # Snapping left/right shifts x → would break any left/right attachment
        # Snapping top/bottom shifts y → would break any top/bottom attachment
        x_locked = bool(attached & {'left', 'right'})
        y_locked = bool(attached & {'top', 'bottom'})

        walls = ray_cast_to_walls(box, h_segments, v_segments)
        other_boxes_with_idx = [(k, Box.from_array(result[k]))
                                for k in range(n_boxes) if k != i]

        best_direction = None
        best_gap = gap_threshold  # only pick directions with gap > threshold
        best_dx = 0.0
        best_dy = 0.0

        for direction, wall_seg in walls.items():
            if wall_seg is None:
                continue

            # Skip if moving along this axis would break an existing attachment
            if direction in ('left', 'right') and x_locked:
                if verbose:
                    print(f"    ⊗ {direction}: x-axis locked by "
                          f"{attached & {'left', 'right'}}")
                continue
            if direction in ('top', 'bottom') and y_locked:
                if verbose:
                    print(f"    ⊗ {direction}: y-axis locked by "
                          f"{attached & {'top', 'bottom'}}")
                continue

            if direction == 'left':
                wall_coord = wall_seg.x1
                gap = box.x1 - wall_coord
                dx, dy = wall_coord - box.x1, 0.0
                in_corridor = [
                    (k, rb) for k, rb in other_boxes_with_idx
                    if room_types[k] != 0
                    and rb.x2 > wall_coord and rb.x1 < box.x1
                    and rb.y1 < box.y2 and rb.y2 > box.y1
                ]
            elif direction == 'right':
                wall_coord = wall_seg.x1
                gap = wall_coord - box.x2
                dx, dy = wall_coord - box.x2, 0.0
                in_corridor = [
                    (k, rb) for k, rb in other_boxes_with_idx
                    if room_types[k] != 0
                    and rb.x1 < wall_coord and rb.x2 > box.x2
                    and rb.y1 < box.y2 and rb.y2 > box.y1
                ]
            elif direction == 'top':
                wall_coord = wall_seg.y1
                gap = box.y1 - wall_coord
                dx, dy = 0.0, wall_coord - box.y1
                in_corridor = [
                    (k, rb) for k, rb in other_boxes_with_idx
                    if room_types[k] != 0
                    and rb.y2 > wall_coord and rb.y1 < box.y1
                    and rb.x1 < box.x2 and rb.x2 > box.x1
                ]
            else:  # bottom
                wall_coord = wall_seg.y1
                gap = wall_coord - box.y2
                dx, dy = 0.0, wall_coord - box.y2
                in_corridor = [
                    (k, rb) for k, rb in other_boxes_with_idx
                    if room_types[k] != 0
                    and rb.y1 < wall_coord and rb.y2 > box.y2
                    and rb.x1 < box.x2 and rb.x2 > box.x1
                ]

            # Split corridor rooms into pre-existing overlaps (bypassed) and
            # true blockers (rooms that don't yet intersect the moving room).
            # Moving perpendicular to an existing overlap doesn't make it worse,
            # so pre-existing overlapping rooms are not treated as blockers.
            bypassed = [(k, rb) for k, rb in in_corridor
                        if _is_perp_overlap(box, rb, direction)]
            blockers  = [(k, rb) for k, rb in in_corridor
                         if not _is_perp_overlap(box, rb, direction)]

            if gap < 0:
                if verbose:
                    print(f"    ⊗ {direction}: already past wall (gap={gap:.1f}px)")
                continue

            if gap <= gap_threshold:
                if verbose:
                    print(f"    ⊗ {direction}: gap too small "
                          f"({gap:.1f}px ≤ {gap_threshold}px)")
                continue

            if verbose and bypassed:
                bypass_strs = [
                    f"room {k} {_rname(room_types[k])}"
                    f" ({rb.x1:.0f},{rb.y1:.0f}→{rb.x2:.0f},{rb.y2:.0f})"
                    f" [pre-existing overlap]"
                    for k, rb in bypassed
                ]
                print(f"    ~ {direction}: bypassing {', '.join(bypass_strs)}")

            if blockers:
                if verbose:
                    blocker_strs = [
                        f"room {k} {_rname(room_types[k])}"
                        f" ({rb.x1:.0f},{rb.y1:.0f}→{rb.x2:.0f},{rb.y2:.0f})"
                        for k, rb in blockers
                    ]
                    print(f"    ⊗ {direction}: blocked by {', '.join(blocker_strs)}"
                          f"  [gap={gap:.1f}px]")
                continue

            if verbose:
                print(f"    ✓ {direction}: gap={gap:.1f}px — eligible")

            if gap > best_gap:
                best_gap = gap
                best_direction = direction
                best_dx = dx
                best_dy = dy

        if best_direction is None:
            if verbose:
                print(f"    → No eligible direction found")
            continue

        new_box = Box(box.x1 + best_dx, box.y1 + best_dy,
                      box.x2 + best_dx, box.y2 + best_dy)
        result[i] = new_box.to_array()

        if verbose:
            print(f"    → Snapped {best_direction} by {best_gap:.1f}px  →  "
                  f"({new_box.x1:.1f},{new_box.y1:.1f},"
                  f"{new_box.x2:.1f},{new_box.y2:.1f})")

    return result


def enforce_graph_adjacency(
        boxes_before: np.ndarray,
        boxes_after: np.ndarray,
        edges: np.ndarray,
        room_types: np.ndarray,
        touch_tol_before: float = 15.0,
        touch_tol_after: float = 3.0,
        verbose: bool = False) -> np.ndarray:
    """
    After any refinement pass, ensure graph-connected room pairs remain adjacent.

    For each edge (i, j): if the two rooms were close before the pass (within
    touch_tol_before) but are now separated (gap > touch_tol_after), the smaller
    room is face-snapped to the bigger room's new position.  The adjacency
    direction (which face of the bigger room the smaller room touches) is
    determined from the pre-pass state, then the smaller room is translated
    so that exact face-to-face contact is restored.

    Args:
        boxes_before:       Nx4 boxes at pass entry
        boxes_after:        Nx4 boxes at pass exit (will be modified)
        edges:              Graph adjacency array, shape (K, 2) or (K, 3)
        room_types:         N-length room type array
        touch_tol_before:   Proximity threshold for "were they adjacent?" (px)
        touch_tol_after:    Proximity threshold for "still adjacent?" (px)
        verbose:            Print debug info

    Returns:
        Updated Nx4 box array with co-movement applied.
    """
    if edges is None or len(edges) == 0:
        return boxes_after.copy()

    result = boxes_after.copy()
    n = len(boxes_before)

    room_type_names = {
        0: "LivingRoom", 1: "MasterRoom", 2: "Kitchen", 3: "Bathroom",
        4: "DiningRoom", 5: "ChildRoom", 6: "StudyRoom", 7: "SecondRoom",
        8: "GuestRoom", 9: "Balcony", 10: "Entrance", 11: "Storage", 12: "Wall",
    }

    def _rname(t: int) -> str:
        return room_type_names.get(int(t), f"type{t}")

    def _is_close(a: np.ndarray, b: np.ndarray, tol: float) -> bool:
        """Return True if boxes are touching, overlapping, or within tol px."""
        h_ov = min(a[2], b[2]) - max(a[0], b[0])
        v_ov = min(a[3], b[3]) - max(a[1], b[1])
        if h_ov > 0 and v_ov > 0:
            return True  # overlapping
        if (abs(a[0] - b[2]) <= tol or abs(a[2] - b[0]) <= tol) and v_ov > 0:
            return True  # vertical face
        if (abs(a[1] - b[3]) <= tol or abs(a[3] - b[1]) <= tol) and h_ov > 0:
            return True  # horizontal face
        return False

    def _adj_direction(small: np.ndarray, big: np.ndarray, tol: float) -> str:
        """
        Determine which face of 'big' the 'small' box was adjacent to before
        the pass.  Returns 'left', 'right', 'top', or 'bottom' (the face of
        'big' that 'small' was touching/nearest to).
        """
        if abs(small[2] - big[0]) <= tol:  return 'left'    # small.x2 ≈ big.x1
        if abs(small[0] - big[2]) <= tol:  return 'right'   # small.x1 ≈ big.x2
        if abs(small[3] - big[1]) <= tol:  return 'top'     # small.y2 ≈ big.y1
        if abs(small[1] - big[3]) <= tol:  return 'bottom'  # small.y1 ≈ big.y2
        # Overlapping — fall back to centroid direction
        cx_s = (small[0] + small[2]) / 2.0
        cy_s = (small[1] + small[3]) / 2.0
        cx_b = (big[0] + big[2]) / 2.0
        cy_b = (big[1] + big[3]) / 2.0
        dx, dy = cx_b - cx_s, cy_b - cy_s
        if abs(dx) >= abs(dy):
            return 'right' if dx > 0 else 'left'
        return 'bottom' if dy > 0 else 'top'

    def _snap_to_face(small: np.ndarray, big: np.ndarray, face: str) -> np.ndarray:
        """
        Translate 'small' so it touches 'big' on the given face.
        'face' is the face of 'big' that 'small' should be pressed against.
        """
        s = small.copy()
        if face == 'left':      # small's right edge → big's left edge
            dx = big[0] - s[2]
            return np.array([s[0]+dx, s[1], s[2]+dx, s[3]])
        elif face == 'right':   # small's left edge → big's right edge
            dx = big[2] - s[0]
            return np.array([s[0]+dx, s[1], s[2]+dx, s[3]])
        elif face == 'top':     # small's bottom edge → big's top edge
            dy = big[1] - s[3]
            return np.array([s[0], s[1]+dy, s[2], s[3]+dy])
        else:                   # 'bottom': small's top edge → big's bottom edge
            dy = big[3] - s[1]
            return np.array([s[0], s[1]+dy, s[2], s[3]+dy])

    def _area(box: np.ndarray) -> float:
        return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])

    if verbose:
        print(f"\n  [enforce_graph_adjacency] Checking {len(edges)} graph edges...")

    for edge in edges:
        ei, ej = int(edge[0]), int(edge[1])
        if not (0 <= ei < n and 0 <= ej < n):
            continue

        # Were they adjacent (or close) before the pass?
        if not _is_close(boxes_before[ei], boxes_before[ej], touch_tol_before):
            continue  # Not adjacent before — nothing to enforce

        # Are they still adjacent after the pass?
        if _is_close(result[ei], result[ej], touch_tol_after):
            continue  # Still adjacent — all good

        # Separated — determine which is smaller (mover) and bigger (anchor)
        area_i = _area(boxes_before[ei])
        area_j = _area(boxes_before[ej])
        if area_i <= area_j:
            smaller_idx, bigger_idx = ei, ej
        else:
            smaller_idx, bigger_idx = ej, ei

        # Determine which face of the bigger room the smaller room was against
        face = _adj_direction(boxes_before[smaller_idx], boxes_before[bigger_idx],
                              tol=touch_tol_before)

        # Snap the smaller room's face directly to the bigger room's new position
        new_small = _snap_to_face(result[smaller_idx], result[bigger_idx], face)
        result[smaller_idx] = new_small

        if verbose:
            print(f"  [enforce_graph_adjacency] Edge ({ei},{ej}): "
                  f"{_rname(room_types[smaller_idx])} {smaller_idx} snapped to "
                  f"{face!r} face of {_rname(room_types[bigger_idx])} {bigger_idx} "
                  f"→ ({new_small[0]:.1f},{new_small[1]:.1f},"
                  f"{new_small[2]:.1f},{new_small[3]:.1f})")

    return result


def close_small_gaps(
        boxes: np.ndarray,
        room_types: np.ndarray,
        boundary: np.ndarray,
        gap_threshold: float = 20.0,
        verbose: bool = False) -> np.ndarray:
    """
    For each room, find small gaps (< gap_threshold px) to the nearest obstacle
    (boundary wall or adjacent room edge) in each of the 4 directions and extend
    the room's edge to close them.

    Ray-casting is done from 3 points per edge face — the two corners and the
    midpoint — so partial adjacency that the center ray misses is still detected.

    Extensions are only applied when the gap target lies inside (or on) the
    boundary, so rooms cannot grow outside the floor-plan outline.

    Args:
        boxes:         Nx4 array [[x1, y1, x2, y2], ...]
        room_types:    N-length array of room type indices
        boundary:      Boundary polygon, shape (M, 2+)
        gap_threshold: Maximum gap size to close (pixels)
        verbose:       Print debug info

    Returns:
        Updated Nx4 box array.
    """
    n_boxes = len(boxes)
    result = boxes.copy()
    h_segments, v_segments = extract_boundary_segments(boundary)

    room_type_names = {
        0: "LivingRoom", 1: "MasterRoom", 2: "Kitchen", 3: "Bathroom",
        4: "DiningRoom", 5: "ChildRoom", 6: "StudyRoom", 7: "SecondRoom",
        8: "GuestRoom", 9: "Balcony", 10: "Entrance", 11: "Storage", 12: "Wall",
    }

    def _rname(t: int) -> str:
        return room_type_names.get(int(t), f"type{t}")

    if verbose:
        print(f"\n{'='*60}")
        print(f"SMALL GAP CLOSING  (gap_threshold={gap_threshold}px)")
        print(f"{'='*60}")

    for i in range(n_boxes):
        box = Box.from_array(result[i])
        cx, cy = box.center
        other_boxes = [(k, Box.from_array(result[k]))
                       for k in range(n_boxes) if k != i]

        if verbose:
            print(f"\n  Box {i} {_rname(room_types[i])}: "
                  f"({box.x1:.0f},{box.y1:.0f}→{box.x2:.0f},{box.y2:.0f})")

        # ------------------------------------------------------------------
        # RIGHT  — scan from (x2, y1), (x2, cy), (x2, y2)
        # ------------------------------------------------------------------
        scan_ys = [box.y1, cy, box.y2]
        min_gap, target = float('inf'), None
        for y in scan_ys:
            for seg in v_segments:
                if seg.x1 > box.x2:
                    sy1, sy2 = min(seg.y1, seg.y2), max(seg.y1, seg.y2)
                    if sy1 <= y <= sy2 and seg.x1 - box.x2 < min_gap:
                        min_gap, target = seg.x1 - box.x2, seg.x1
            for k, rb in other_boxes:
                if rb.x1 > box.x2 and rb.y1 <= y <= rb.y2:
                    if rb.x1 - box.x2 < min_gap:
                        min_gap, target = rb.x1 - box.x2, rb.x1
        if 0 < min_gap < gap_threshold:
            if verbose:
                print(f"    → extend right {min_gap:.1f}px → x2={target:.0f}")
            box = Box(box.x1, box.y1, target, box.y2)
            result[i] = box.to_array()

        # ------------------------------------------------------------------
        # LEFT  — scan from (x1, y1), (x1, cy), (x1, y2)
        # ------------------------------------------------------------------
        min_gap, target = float('inf'), None
        for y in scan_ys:
            for seg in v_segments:
                if seg.x1 < box.x1:
                    sy1, sy2 = min(seg.y1, seg.y2), max(seg.y1, seg.y2)
                    if sy1 <= y <= sy2 and box.x1 - seg.x1 < min_gap:
                        min_gap, target = box.x1 - seg.x1, seg.x1
            for k, rb in other_boxes:
                if rb.x2 < box.x1 and rb.y1 <= y <= rb.y2:
                    if box.x1 - rb.x2 < min_gap:
                        min_gap, target = box.x1 - rb.x2, rb.x2
        if 0 < min_gap < gap_threshold:
            if verbose:
                print(f"    → extend left  {min_gap:.1f}px → x1={target:.0f}")
            box = Box(target, box.y1, box.x2, box.y2)
            result[i] = box.to_array()

        # ------------------------------------------------------------------
        # BOTTOM  — scan from (x1, y2), (cx, y2), (x2, y2)
        # ------------------------------------------------------------------
        scan_xs = [box.x1, cx, box.x2]
        min_gap, target = float('inf'), None
        for x in scan_xs:
            for seg in h_segments:
                if seg.y1 > box.y2:
                    sx1, sx2 = min(seg.x1, seg.x2), max(seg.x1, seg.x2)
                    if sx1 <= x <= sx2 and seg.y1 - box.y2 < min_gap:
                        min_gap, target = seg.y1 - box.y2, seg.y1
            for k, rb in other_boxes:
                if rb.y1 > box.y2 and rb.x1 <= x <= rb.x2:
                    if rb.y1 - box.y2 < min_gap:
                        min_gap, target = rb.y1 - box.y2, rb.y1
        if 0 < min_gap < gap_threshold:
            if verbose:
                print(f"    → extend bottom {min_gap:.1f}px → y2={target:.0f}")
            box = Box(box.x1, box.y1, box.x2, target)
            result[i] = box.to_array()

        # ------------------------------------------------------------------
        # TOP  — scan from (x1, y1), (cx, y1), (x2, y1)
        # ------------------------------------------------------------------
        min_gap, target = float('inf'), None
        for x in scan_xs:
            for seg in h_segments:
                if seg.y1 < box.y1:
                    sx1, sx2 = min(seg.x1, seg.x2), max(seg.x1, seg.x2)
                    if sx1 <= x <= sx2 and box.y1 - seg.y1 < min_gap:
                        min_gap, target = box.y1 - seg.y1, seg.y1
            for k, rb in other_boxes:
                if rb.y2 < box.y1 and rb.x1 <= x <= rb.x2:
                    if box.y1 - rb.y2 < min_gap:
                        min_gap, target = box.y1 - rb.y2, rb.y2
        if 0 < min_gap < gap_threshold:
            if verbose:
                print(f"    → extend top   {min_gap:.1f}px → y1={target:.0f}")
            box = Box(box.x1, target, box.x2, box.y2)
            result[i] = box.to_array()

    return result


def fill_inter_room_gaps(boxes: np.ndarray,
                         room_types: np.ndarray,
                         boundary: np.ndarray,
                         gap_threshold: float = 20.0,
                         edges: Optional[np.ndarray] = None,
                         verbose: bool = False) -> np.ndarray:
    """
    Expand rooms to fill small gaps between adjacent non-living rooms.

    Scans every pair of rooms for gaps in the horizontal and vertical directions.
    When two rooms are close but not touching (gap <= gap_threshold) and they
    overlap in the perpendicular axis, the room with fewer wall attachments has
    its edge expanded to close the gap.

    This catches coverage gaps that are not against a boundary wall (which
    fill_small_boundary_gaps cannot detect) and are not closed by
    snap_rooms_to_neighbors (which translates whole rooms).

    Args:
        boxes: Nx4 array of boxes [[x1, y1, x2, y2], ...]
        room_types: Room type indices (0=Living, 3=Bathroom, etc.)
        boundary: Boundary polygon
        gap_threshold: Maximum gap size to fill (pixels)
        verbose: Print debug information

    Returns:
        filled_boxes: Nx4 array with inter-room gaps filled
    """
    n_boxes = len(boxes)
    filled_boxes = boxes.copy()
    h_segments, v_segments = extract_boundary_segments(boundary)
    n_expanded = 0

    room_type_names = {0: "LivingRoom", 1: "MasterRoom", 2: "Kitchen", 3: "Bathroom",
                       4: "DiningRoom", 5: "ChildRoom", 6: "StudyRoom", 7: "SecondRoom",
                       8: "GuestRoom", 9: "Balcony", 10: "Entrance", 11: "Storage"}

    if verbose:
        print(f"\n{'='*60}")
        print(f"INTER-ROOM GAP FILL - {n_boxes} boxes, threshold: {gap_threshold:.1f}px")
        print(f"{'='*60}")

    # Up to 3 passes to handle cascading fills
    for _ in range(3):
        changed = False

        for i in range(n_boxes):
            if room_types[i] == 0:
                continue  # Living room handled separately

            box_i = Box.from_array(filled_boxes[i])
            align_i = find_closest_segments(box_i, h_segments, v_segments)
            walls_i = sum(1 for a in align_i.values() if a.distance < 0.1)

            for j in range(n_boxes):
                if j == i or room_types[j] == 0:
                    continue

                box_j = Box.from_array(filled_boxes[j])

                # --- Horizontal gap: i is to the left of j ---
                h_gap = box_j.x1 - box_i.x2
                if 0 < h_gap <= gap_threshold:
                    v_overlap = min(box_i.y2, box_j.y2) - max(box_i.y1, box_j.y1)
                    if v_overlap > 0:
                        align_j = find_closest_segments(box_j, h_segments, v_segments)
                        walls_j = sum(1 for a in align_j.values() if a.distance < 0.1)

                        # Snap the room with fewer wall attachments (translate whole box)
                        if walls_i <= walls_j:
                            # Translate i right to touch j's left edge
                            new_box = Box(box_i.x1 + h_gap, box_i.y1, box_j.x1, box_i.y2)
                            expand_idx = i
                        else:
                            # Translate j left to touch i's right edge
                            new_box = Box(box_i.x2, box_j.y1, box_j.x2 - h_gap, box_j.y2)
                            expand_idx = j

                        # Reject if this snap would separate a currently-touching graph neighbour
                        if edges is not None and _would_separate_graph_neighbors(
                                expand_idx, new_box, filled_boxes, edges):
                            if verbose:
                                name = room_type_names.get(room_types[expand_idx],
                                                           f"Type{room_types[expand_idx]}")
                                print(f"  ⊙ Cannot translate Box {expand_idx} ({name}) "
                                      f"(would separate a touching graph neighbour)")
                            continue

                        # Check that expansion doesn't create new overlaps
                        orig = Box.from_array(filled_boxes[expand_idx])
                        ok = True
                        for k in range(n_boxes):
                            if k == expand_idx or room_types[k] == 3 or room_types[k] == 0:
                                continue
                            other = Box.from_array(filled_boxes[k])
                            new_overlaps = (new_box.x1 < other.x2 and new_box.x2 > other.x1 and
                                            new_box.y1 < other.y2 and new_box.y2 > other.y1)
                            was_overlapping = (orig.x1 < other.x2 and orig.x2 > other.x1 and
                                               orig.y1 < other.y2 and orig.y2 > other.y1)
                            if new_overlaps and not was_overlapping:
                                ok = False
                                break

                        if ok:
                            name = room_type_names.get(room_types[expand_idx], f"Type{room_types[expand_idx]}")
                            other_idx = j if expand_idx == i else i
                            if verbose:
                                print(f"  ✓ Expanded Box {expand_idx} ({name}) right/left by "
                                      f"{h_gap:.1f}px to close gap with Box {other_idx}")
                            filled_boxes[expand_idx] = new_box.to_array()
                            # Refresh box_i if we just changed it
                            if expand_idx == i:
                                box_i = new_box
                            n_expanded += 1
                            changed = True

                # --- Vertical gap: i is above j ---
                v_gap = box_j.y1 - box_i.y2
                if 0 < v_gap <= gap_threshold:
                    h_overlap = min(box_i.x2, box_j.x2) - max(box_i.x1, box_j.x1)
                    if h_overlap > 0:
                        align_j = find_closest_segments(box_j, h_segments, v_segments)
                        walls_j = sum(1 for a in align_j.values() if a.distance < 0.1)

                        if walls_i <= walls_j:
                            # Translate i down to touch j's top edge
                            new_box = Box(box_i.x1, box_i.y1 + v_gap, box_i.x2, box_j.y1)
                            expand_idx = i
                        else:
                            # Translate j up to touch i's bottom edge
                            new_box = Box(box_j.x1, box_i.y2, box_j.x2, box_j.y2 - v_gap)
                            expand_idx = j

                        # Reject if this snap would separate a currently-touching graph neighbour
                        if edges is not None and _would_separate_graph_neighbors(
                                expand_idx, new_box, filled_boxes, edges):
                            if verbose:
                                name = room_type_names.get(room_types[expand_idx],
                                                           f"Type{room_types[expand_idx]}")
                                print(f"  ⊙ Cannot translate Box {expand_idx} ({name}) "
                                      f"(would separate a touching graph neighbour)")
                            continue

                        orig = Box.from_array(filled_boxes[expand_idx])
                        ok = True
                        for k in range(n_boxes):
                            if k == expand_idx or room_types[k] == 3 or room_types[k] == 0:
                                continue
                            other = Box.from_array(filled_boxes[k])
                            new_overlaps = (new_box.x1 < other.x2 and new_box.x2 > other.x1 and
                                            new_box.y1 < other.y2 and new_box.y2 > other.y1)
                            was_overlapping = (orig.x1 < other.x2 and orig.x2 > other.x1 and
                                               orig.y1 < other.y2 and orig.y2 > other.y1)
                            if new_overlaps and not was_overlapping:
                                ok = False
                                break

                        if ok:
                            name = room_type_names.get(room_types[expand_idx], f"Type{room_types[expand_idx]}")
                            other_idx = j if expand_idx == i else i
                            if verbose:
                                print(f"  ✓ Expanded Box {expand_idx} ({name}) top/bottom by "
                                      f"{v_gap:.1f}px to close gap with Box {other_idx}")
                            filled_boxes[expand_idx] = new_box.to_array()
                            if expand_idx == i:
                                box_i = new_box
                            n_expanded += 1
                            changed = True

        if not changed:
            break

    if verbose:
        print(f"\n{'='*60}")
        print(f"✓ Filled {n_expanded} inter-room gaps")
        print(f"{'='*60}\n")

    return filled_boxes


def fill_small_boundary_gaps(boxes: np.ndarray,
                             room_types: np.ndarray,
                             boundary: np.ndarray,
                             gap_threshold: float = 20.0,
                             edges: Optional[np.ndarray] = None,
                             verbose: bool = False) -> np.ndarray:
    """
    Expand rooms to fill small gaps at the boundary corners.

    This function expands room edges that are close to the boundary (within gap_threshold)
    to fill small empty corners. Only runs on non-living-room boxes.

    Args:
        boxes: Nx4 array of boxes [[x1, y1, x2, y2], ...]
        room_types: Room type indices (0=Living, 1=Bedroom, etc.)
        boundary: Boundary polygon
        gap_threshold: Maximum gap distance to fill (pixels)
        verbose: Print debug information

    Returns:
        expanded_boxes: Nx4 array of boxes with gaps filled
    """
    n_boxes = len(boxes)
    expanded_boxes = boxes.copy()

    # Extract boundary segments
    h_segments, v_segments = extract_boundary_segments(boundary)

    if verbose:
        print(f"\n{'='*60}")
        print(f"FILLING SMALL BOUNDARY GAPS - {n_boxes} boxes")
        print(f"  Gap threshold: {gap_threshold}px")
        print(f"{'='*60}")

    n_expanded = 0
    for i in range(n_boxes):
        box = Box.from_array(expanded_boxes[i])
        room_type = room_types[i]

        # Skip living room (it will be expanded separately)
        if room_type == 0:
            if verbose:
                print(f"\n--- Box {i}: LivingRoom ---")
                print(f"  ⊘ Skipping (living room)")
            continue

        if verbose:
            room_type_names = {0: "LivingRoom", 1: "MasterRoom", 2: "Kitchen", 3: "Bathroom",
                             4: "DiningRoom", 5: "ChildRoom", 6: "StudyRoom", 7: "SecondRoom",
                             8: "GuestRoom", 9: "Balcony", 10: "Entrance", 11: "Storage", 12: "Wall"}
            room_name = room_type_names.get(room_type, f"Unknown({room_type})")
            print(f"\n--- Box {i}: {room_name} ---")

        # Iteratively expand multiple edges to fill corner pockets
        # Keep trying to expand edges until no more expansions are possible
        max_iterations = 4  # Maximum number of edges that can be expanded (one per direction)
        edges_expanded = []

        for iteration in range(max_iterations):
            # Recalculate alignments with current box position
            box = Box.from_array(expanded_boxes[i])
            alignments = find_closest_segments(box, h_segments, v_segments)

            # Find the closest edge that can be expanded
            best_edge = None
            best_alignment = None
            best_distance = float('inf')

            for edge_name, alignment in alignments.items():
                # Skip edges already expanded
                if edge_name in edges_expanded:
                    continue

                # Only consider edges close to boundary but not already attached
                if alignment.distance <= gap_threshold and alignment.distance > 0.1:
                    if alignment.distance < best_distance:
                        best_distance = alignment.distance
                        best_edge = edge_name
                        best_alignment = alignment

            # If no more edges can be expanded, stop
            if best_edge is None:
                break

            # Snap whole box to wall (translate, preserving room size)
            if best_edge == 'left':
                delta = best_alignment.snap_value - box.x1
                new_box = Box(best_alignment.snap_value, box.y1, box.x2 + delta, box.y2)
            elif best_edge == 'right':
                delta = best_alignment.snap_value - box.x2
                new_box = Box(box.x1 + delta, box.y1, best_alignment.snap_value, box.y2)
            elif best_edge == 'top':
                delta = best_alignment.snap_value - box.y1
                new_box = Box(box.x1, best_alignment.snap_value, box.x2, box.y2 + delta)
            elif best_edge == 'bottom':
                delta = best_alignment.snap_value - box.y2
                new_box = Box(box.x1, box.y1 + delta, box.x2, best_alignment.snap_value)
            else:
                break

            # Reject if this snap would separate a currently-touching graph neighbour
            if edges is not None and _would_separate_graph_neighbors(i, new_box, expanded_boxes, edges):
                if verbose:
                    print(f"  ⊙ Cannot expand {best_edge} (would separate a touching graph neighbour)")
                edges_expanded.append(best_edge)
                continue

            # Check for overlap with other rooms
            would_overlap = False
            for j in range(n_boxes):
                if i == j:
                    continue
                other_box = Box.from_array(expanded_boxes[j])

                # Check if boxes overlap
                if (new_box.x1 < other_box.x2 and new_box.x2 > other_box.x1 and
                    new_box.y1 < other_box.y2 and new_box.y2 > other_box.y1):
                    would_overlap = True
                    break

            if not would_overlap:
                # Apply the expansion
                expanded_boxes[i] = new_box.to_array()
                edges_expanded.append(best_edge)
                if verbose:
                    print(f"  ✓ Expanded {best_edge} edge by {best_distance:.2f}px to fill gap")
            else:
                # Can't expand this edge, mark it as tried
                edges_expanded.append(best_edge)
                if verbose:
                    print(f"  ⊗ Cannot expand {best_edge} (would overlap with other room)")

        # Update counter
        if len(edges_expanded) > 0:
            n_expanded += 1

        if len(edges_expanded) == 0 and verbose:
            print(f"  ⊗ No small gaps to fill (all edges either attached or too far)")

    if verbose:
        print(f"\n{'='*60}")
        print(f"✓ Expanded {n_expanded} boxes to fill boundary gaps")
        print(f"{'='*60}\n")

    return expanded_boxes


def fill_coverage_gaps(boxes: np.ndarray,
                       room_types: np.ndarray,
                       boundary: np.ndarray,
                       min_gap_area: float = 4.0,
                       verbose: bool = False) -> np.ndarray:
    """
    Detect and fill any areas inside the boundary not covered by any room.

    After living room expansion, computes the exact uncovered regions as
    (boundary polygon - union of all room polygons) using Shapely. For each
    gap, finds adjacent non-living rooms and expands the best candidate's
    bounding box to absorb the gap.

    This is geometry-based and does not use ray casting, so it catches corner
    pockets that ray-casting-based functions miss (e.g. step-wall notches where
    the room center is above the visible wall segment).

    Requires Shapely. Returns boxes unchanged if Shapely is not available.

    Args:
        boxes: Nx4 array of boxes [[x1, y1, x2, y2], ...]
        room_types: Room type indices (0=Living, 3=Bathroom, etc.)
        boundary: Boundary polygon (Nx2 or Nx4 array)
        min_gap_area: Ignore gaps smaller than this many px² (floating-point noise)
        verbose: Print debug information

    Returns:
        filled_boxes: Nx4 array with coverage gaps filled
    """
    try:
        from shapely.geometry import Polygon
        from shapely.ops import unary_union
    except ImportError:
        if verbose:
            print("  Shapely not available, skipping coverage gap fill")
        return boxes.copy()

    n_boxes = len(boxes)
    filled_boxes = boxes.copy()

    h_segments, v_segments = extract_boundary_segments(boundary)

    room_type_names = {0: "LivingRoom", 1: "MasterRoom", 2: "Kitchen", 3: "Bathroom",
                       4: "DiningRoom", 5: "ChildRoom", 6: "StudyRoom", 7: "SecondRoom",
                       8: "GuestRoom", 9: "Balcony", 10: "Entrance", 11: "Storage", 12: "Wall"}

    if verbose:
        print(f"\n{'='*60}")
        print(f"COVERAGE GAP FILL - {n_boxes} boxes")
        print(f"{'='*60}")

    def make_poly(b):
        x1, y1, x2, y2 = float(b[0]), float(b[1]), float(b[2]), float(b[3])
        return Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])

    # Build boundary polygon
    bnd_pts = boundary[:, :2].astype(float)
    boundary_poly = Polygon(bnd_pts)
    if not boundary_poly.is_valid:
        boundary_poly = boundary_poly.buffer(0)

    # Build room polygons
    room_polys = [make_poly(b) for b in filled_boxes]

    # Compute uncovered area inside the boundary
    covered = unary_union(room_polys)
    uncovered = boundary_poly.difference(covered)

    if uncovered.is_empty:
        if verbose:
            print("  No coverage gaps found")
        return filled_boxes

    # Collect individual gap polygons above the minimum area threshold
    gaps = []
    if hasattr(uncovered, 'geoms'):
        for g in uncovered.geoms:
            if g.area >= min_gap_area:
                gaps.append(g)
    elif uncovered.area >= min_gap_area:
        gaps = [uncovered]

    if not gaps:
        if verbose:
            print("  No significant coverage gaps found")
        return filled_boxes

    if verbose:
        print(f"  Found {len(gaps)} gap(s) to fill")

    n_filled = 0
    for gap_idx, gap in enumerate(gaps):
        bx = gap.bounds  # (minx, miny, maxx, maxy)
        if verbose:
            print(f"\n  Gap {gap_idx}: area={gap.area:.1f}px², "
                  f"bounds=[{bx[0]:.1f},{bx[1]:.1f} → {bx[2]:.1f},{bx[3]:.1f}]")

        # Find non-living rooms that touch or are within 1px of the gap
        gap_expanded = gap.buffer(1.0)
        adjacent = []
        for i in range(n_boxes):
            if room_types[i] == 0:
                continue  # Living room already expanded - skip as candidate
            if not room_polys[i].intersects(gap_expanded):
                continue

            # Measure how much boundary the room shares with the gap
            shared = room_polys[i].boundary.intersection(gap.boundary)
            shared_length = shared.length if not shared.is_empty else 0.0

            # Count current wall attachments (fewer = more free to expand)
            box_i = Box.from_array(filled_boxes[i])
            alignments = find_closest_segments(box_i, h_segments, v_segments)
            walls = sum(1 for a in alignments.values() if a.distance < 0.1)

            adjacent.append((i, shared_length, walls))

        if not adjacent:
            if verbose:
                print(f"    No adjacent non-living rooms found")
            continue

        # Prefer the room with fewer wall attachments (more free to move),
        # breaking ties by longest shared edge with the gap
        adjacent.sort(key=lambda x: (x[2], -x[1]))

        filled = False
        for room_idx, shared_len, walls in adjacent:
            # Expand the room's bounding box to include the entire gap
            combined_bounds = room_polys[room_idx].union(gap).bounds
            new_box = Box(combined_bounds[0], combined_bounds[1],
                          combined_bounds[2], combined_bounds[3])
            new_poly = make_poly(new_box.to_array())

            # Verify the expansion doesn't create meaningful overlap with other rooms
            ok = True
            for k in range(n_boxes):
                if k == room_idx or room_types[k] == 0 or room_types[k] == 3:
                    continue  # Skip self, living room, and bathrooms (bathrooms are hosted inside rooms)
                intersection_area = new_poly.intersection(room_polys[k]).area
                if intersection_area > 1.0:  # >1px² = real overlap, not just touching
                    ok = False
                    if verbose:
                        name_k = room_type_names.get(room_types[k], f"Type{room_types[k]}")
                        print(f"    ✗ Box {room_idx} expansion blocked by Box {k} ({name_k}), "
                              f"overlap={intersection_area:.1f}px²")
                    break

            if ok:
                filled_boxes[room_idx] = new_box.to_array()
                room_polys[room_idx] = new_poly  # Update so later gaps see the new position
                n_filled += 1
                filled = True
                if verbose:
                    name = room_type_names.get(room_types[room_idx], f"Type{room_types[room_idx]}")
                    print(f"    ✓ Expanded Box {room_idx} ({name}) "
                          f"(walls={walls}, shared={shared_len:.1f}px)")
                break

        if not filled and verbose:
            print(f"    ✗ Could not fill gap (all adjacent rooms blocked)")

    if verbose:
        print(f"\n{'='*60}")
        print(f"✓ Filled {n_filled}/{len(gaps)} coverage gap(s)")
        print(f"{'='*60}\n")

    return filled_boxes


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

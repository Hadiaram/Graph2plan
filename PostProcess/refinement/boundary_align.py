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


def find_closest_segments(box: Box, h_segments: List[BoundarySegment],
                          v_segments: List[BoundarySegment]) -> Dict[str, EdgeAlignment]:
    """
    Find the closest boundary segment for each edge of a box.

    This implements the MATLAB find_close_seg() function.

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

    # Find closest vertical segment for left and right edges
    best_left = EdgeAlignment('left', MAX_DIST, box.x1)
    best_right = EdgeAlignment('right', MAX_DIST, box.x2)

    for seg in v_segments:
        # For vertical segments, check alignment with box left/right edges

        # Calculate vertical overlap
        seg_y_min = min(seg.y1, seg.y2)
        seg_y_max = max(seg.y1, seg.y2)
        box_y_min = box.y1
        box_y_max = box.y2

        # Check if there's vertical overlap
        overlap_y_start = max(seg_y_min, box_y_min)
        overlap_y_end = min(seg_y_max, box_y_max)

        if overlap_y_end < overlap_y_start:
            # No vertical overlap - calculate distance to nearest point
            if seg_y_max <= box_y_min:
                v_dist = box_y_min - seg_y_max
            elif seg_y_min >= box_y_max:
                v_dist = seg_y_min - box_y_max
            else:
                v_dist = 0
        else:
            # Has vertical overlap
            v_dist = 0

        # Get segment x-coordinate (same for both ends since vertical)
        seg_x = seg.x1

        # Distance to left edge
        h_dist_left = box.x1 - seg_x
        if h_dist_left > 0:  # Segment is to the left of box
            dist_left = np.sqrt(h_dist_left**2 + v_dist**2)
            if dist_left < best_left.distance:
                best_left = EdgeAlignment('left', dist_left, seg_x, seg)

        # Distance to right edge
        h_dist_right = seg_x - box.x2
        if h_dist_right > 0:  # Segment is to the right of box
            dist_right = np.sqrt(h_dist_right**2 + v_dist**2)
            if dist_right < best_right.distance:
                best_right = EdgeAlignment('right', dist_right, seg_x, seg)

    alignments['left'] = best_left
    alignments['right'] = best_right

    # Find closest horizontal segment for top and bottom edges
    best_top = EdgeAlignment('top', MAX_DIST, box.y1)
    best_bottom = EdgeAlignment('bottom', MAX_DIST, box.y2)

    for seg in h_segments:
        # For horizontal segments, check alignment with box top/bottom edges

        # Calculate horizontal overlap
        seg_x_min = min(seg.x1, seg.x2)
        seg_x_max = max(seg.x1, seg.x2)
        box_x_min = box.x1
        box_x_max = box.x2

        # Check if there's horizontal overlap
        overlap_x_start = max(seg_x_min, box_x_min)
        overlap_x_end = min(seg_x_max, box_x_max)

        if overlap_x_end < overlap_x_start:
            # No horizontal overlap - calculate distance to nearest point
            if seg_x_max <= box_x_min:
                h_dist = box_x_min - seg_x_max
            elif seg_x_min >= box_x_max:
                h_dist = seg_x_min - box_x_max
            else:
                h_dist = 0
        else:
            # Has horizontal overlap
            h_dist = 0

        # Get segment y-coordinate (same for both ends since horizontal)
        seg_y = seg.y1

        # Distance to top edge
        v_dist_top = box.y1 - seg_y
        if v_dist_top > 0:  # Segment is above box
            dist_top = np.sqrt(v_dist_top**2 + h_dist**2)
            if dist_top < best_top.distance:
                best_top = EdgeAlignment('top', dist_top, seg_y, seg)

        # Distance to bottom edge
        v_dist_bottom = seg_y - box.y2
        if v_dist_bottom > 0:  # Segment is below box
            dist_bottom = np.sqrt(v_dist_bottom**2 + h_dist**2)
            if dist_bottom < best_bottom.distance:
                best_bottom = EdgeAlignment('bottom', dist_bottom, seg_y, seg)

    alignments['top'] = best_top
    alignments['bottom'] = best_bottom

    return alignments


def align_box_with_boundary(box: Box, boundary: np.ndarray,
                           threshold: float = 8.0,
                           verbose: bool = False) -> Tuple[Box, Dict[str, bool]]:
    """
    Align a box with the boundary by snapping edges within threshold.

    This implements the MATLAB align_with_boundary() function.

    Args:
        box: Box to align
        boundary: Boundary polygon (Nx2 or Nx4 array)
        threshold: Maximum distance for snapping (pixels)
        verbose: Print debug information

    Returns:
        aligned_box: Box with edges snapped to boundary
        updated_edges: Dict indicating which edges were updated
                      {'left': bool, 'top': bool, 'right': bool, 'bottom': bool}
    """
    # Extract boundary segments
    h_segments, v_segments = extract_boundary_segments(boundary)

    if verbose:
        print(f"\nAligning box: {box}")
        print(f"  Boundary has {len(h_segments)} horizontal and {len(v_segments)} vertical segments")
        print(f"  Threshold: {threshold} pixels")

    # Find closest segments for each edge
    alignments = find_closest_segments(box, h_segments, v_segments)

    if verbose:
        print("\n  Closest segments:")
        for edge_name, alignment in alignments.items():
            print(f"    {edge_name}: distance={alignment.distance:.2f}, snap_to={alignment.snap_value:.2f}")

    # Create new box with snapped edges
    aligned_box = Box(box.x1, box.y1, box.x2, box.y2)
    updated_edges = {
        'left': False,
        'top': False,
        'right': False,
        'bottom': False
    }

    # Snap each edge if within threshold
    for edge_name, alignment in alignments.items():
        if alignment.distance <= threshold:
            if edge_name == 'left':
                aligned_box.x1 = alignment.snap_value
                updated_edges['left'] = True
            elif edge_name == 'top':
                aligned_box.y1 = alignment.snap_value
                updated_edges['top'] = True
            elif edge_name == 'right':
                aligned_box.x2 = alignment.snap_value
                updated_edges['right'] = True
            elif edge_name == 'bottom':
                aligned_box.y2 = alignment.snap_value
                updated_edges['bottom'] = True

            if verbose:
                print(f"  ✓ Snapped {edge_name} edge (dist={alignment.distance:.2f})")

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
        print(f"\n  Result: {aligned_box}")
        print(f"  Updated edges: {[k for k, v in updated_edges.items() if v]}")

    return aligned_box, updated_edges


def align_all_boxes_with_boundary(boxes: np.ndarray, boundary: np.ndarray,
                                  threshold: float = 8.0,
                                  room_types: Optional[np.ndarray] = None,
                                  verbose: bool = False) -> Tuple[np.ndarray, List[Dict[str, bool]]]:
    """
    Align all boxes with the boundary.

    Args:
        boxes: Nx4 array of boxes [[x1, y1, x2, y2], ...]
        boundary: Boundary polygon
        threshold: Snapping threshold (pixels)
        room_types: Optional array of room type indices
        verbose: Print debug information

    Returns:
        aligned_boxes: Nx4 array of aligned boxes
        all_updated_edges: List of updated_edges dicts for each box
    """
    n_boxes = len(boxes)
    aligned_boxes = np.zeros_like(boxes)
    all_updated_edges = []

    if verbose:
        print(f"\n{'='*60}")
        print(f"BOUNDARY ALIGNMENT - {n_boxes} boxes")
        print(f"{'='*60}")

    for i in range(n_boxes):
        box = Box.from_array(boxes[i])

        if verbose:
            room_type = room_types[i] if room_types is not None else i
            print(f"\n--- Box {i} (type={room_type}) ---")

        aligned_box, updated_edges = align_box_with_boundary(
            box, boundary, threshold, verbose=verbose
        )

        aligned_boxes[i] = aligned_box.to_array()
        all_updated_edges.append(updated_edges)

    if verbose:
        print(f"\n{'='*60}")
        n_updated = sum(1 for edges in all_updated_edges if any(edges.values()))
        print(f"✓ Aligned {n_updated}/{n_boxes} boxes")
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

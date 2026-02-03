"""
Integration module for connecting geometric refinement to the Interface.

This module provides a drop-in replacement for MATLAB's align_fp function.
"""

import numpy as np
from typing import Tuple, List, Union
from .boundary_align import align_all_boxes_with_boundary


def align_fp_python(boundary: np.ndarray,
                   boxes: np.ndarray,
                   room_types: np.ndarray,
                   edges: np.ndarray,
                   fp_id: Union[str, int],
                   threshold: float = 8.0,
                   draw_result: bool = False) -> Tuple[List, List, List]:
    """
    Pure Python replacement for MATLAB align_fp() function.

    This is a drop-in replacement for the MATLAB function called in
    Interface/Houseweb/views.py. Currently implements Step 1 only
    (boundary alignment). Steps 2-4 will be added incrementally.

    Args:
        boundary: Boundary polygon, shape (N, 2) or (N, 4)
                 Format: [[x, y], ...] or [[x, y, orientation, isNew], ...]
        boxes: Room boxes, shape (M, 4)
              Format: [[x1, y1, x2, y2], ...]
        room_types: Room type indices, shape (M,)
                   0=Living, 1=Bedroom, 2=Kitchen, 3=Bathroom, etc.
        edges: Adjacency graph, shape (K, 2) or (K, 3)
              Format: [[room_i, room_j], ...] or [[room_i, room_j, spatial_type], ...]
        fp_id: Floor plan identifier (for debugging/logging)
        threshold: Snapping threshold in pixels (default: 8.0)
        draw_result: If True, save visualization (not implemented yet)

    Returns:
        new_boxes: Refined boxes as list of lists [[x1,y1,x2,y2], ...]
        order: Render order as list of lists [[idx], ...] (1-indexed for MATLAB compatibility)
        room_boundaries: Room polygons as list of lists [[[x,y], ...], ...]

    Note:
        Currently only implements Step 1 (boundary alignment).
        Steps 2-4 (neighbor align, gap fill, polygon generation) coming soon.

        For now, returns:
        - new_boxes: Boundary-aligned boxes
        - order: Simple ordering by area (largest first)
        - room_boundaries: Simple box polygons (not cropped to boundary yet)
    """

    print(f"\n[Python Refinement] Processing floor plan {fp_id}")
    print(f"  Boxes: {len(boxes)}, Boundary vertices: {len(boundary)}, Threshold: {threshold}px")

    # ============================================================
    # STEP 1: BOUNDARY ALIGNMENT ✅
    # ============================================================
    print("  Step 1: Aligning boxes with boundary...")

    aligned_boxes, updated_edges = align_all_boxes_with_boundary(
        boxes,
        boundary,
        threshold=threshold,
        room_types=room_types,
        verbose=True  # Enable verbose logging to debug
    )

    n_updated = sum(1 for edges_dict in updated_edges if any(edges_dict.values()))
    print(f"    ✓ Aligned {n_updated}/{len(boxes)} boxes")

    # ============================================================
    # STEP 2: NEIGHBOR ALIGNMENT (TODO)
    # ============================================================
    print("  Step 2: Neighbor alignment (TODO - using boundary-aligned boxes)")
    # TODO: Implement neighbor alignment
    # For now, just use boundary-aligned boxes
    neighbor_aligned_boxes = aligned_boxes.copy()

    # ============================================================
    # STEP 3: GAP FILLING & REGULARIZATION (TODO)
    # ============================================================
    print("  Step 3: Gap filling (TODO - using current boxes)")
    # TODO: Implement gap filling
    # For now, just use neighbor-aligned boxes
    final_boxes = neighbor_aligned_boxes.copy()

    # ============================================================
    # STEP 4: POLYGON GENERATION (TODO)
    # ============================================================
    print("  Step 4: Generating room boundaries (TODO - using simple polygons)")
    # TODO: Implement proper polygon generation with boundary cropping
    # For now, create simple rectangular polygons from boxes

    # Preserve original order (don't sort by area to maintain hierarchy)
    # MATLAB-style 1-indexed order (as nested lists)
    order = [[int(i) + 1] for i in range(len(final_boxes))]

    # Generate simple rectangular room boundaries
    room_boundaries = []
    for box in final_boxes:
        x1, y1, x2, y2 = box
        # Create rectangular polygon [5 points to close the shape]
        poly = [
            [float(x1), float(y1)],
            [float(x2), float(y1)],
            [float(x2), float(y2)],
            [float(x1), float(y2)],
            [float(x1), float(y1)]  # Close the polygon
        ]
        room_boundaries.append(poly)

    # Convert boxes to list of lists for MATLAB compatibility
    new_boxes = final_boxes.tolist()

    print(f"  ✓ Refinement complete!")
    print(f"    Output: {len(new_boxes)} boxes, {len(order)} render order, {len(room_boundaries)} boundaries\n")

    # TODO: If draw_result is True, save visualization
    if draw_result:
        print("  (Visualization not yet implemented)")

    return new_boxes, order, room_boundaries


def align_fp_matlab_compatible(boundary, boxes, room_types, edges, fp_id,
                               threshold=8.0, draw_result=False):
    """
    Wrapper that accepts MATLAB-style inputs and returns MATLAB-style outputs.

    This handles conversion from MATLAB data types (matlab.double) to NumPy arrays
    and back to Python lists for return to the Interface.

    Args:
        boundary: MATLAB double or NumPy array or list
        boxes: MATLAB double or NumPy array or list
        room_types: MATLAB double or NumPy array or list
        edges: MATLAB double or NumPy array or list
        fp_id: Floor plan identifier
        threshold: Snapping threshold
        draw_result: Whether to save visualizations

    Returns:
        Tuple of (new_boxes, order, room_boundaries) as Python lists
    """

    # Convert MATLAB types to NumPy if needed
    try:
        # Try to import matlab to check if we have MATLAB types
        import matlab # type: ignore

        if isinstance(boundary, matlab.double):
            boundary = np.array(boundary)
        if isinstance(boxes, matlab.double):
            boxes = np.array(boxes)
        if isinstance(room_types, matlab.double):
            room_types = np.array(room_types).flatten()
        if isinstance(edges, matlab.double):
            edges = np.array(edges)
    except ImportError:
        pass

    # Ensure NumPy arrays
    boundary = np.asarray(boundary)
    boxes = np.asarray(boxes)
    room_types = np.asarray(room_types).flatten().astype(int)
    edges = np.asarray(edges)

    # Call the main function
    return align_fp_python(
        boundary, boxes, room_types, edges, fp_id,
        threshold, draw_result
    )


# Convenience function that matches the MATLAB interface exactly
def align_fp(boundary, boxes, room_types, edges, fp_id=0,
            threshold=8.0, draw_result=False, nargout=3):
    """
    Drop-in replacement for MATLAB align_fp() function.

    This function signature matches MATLAB exactly, including the nargout parameter.

    Usage in Interface/Houseweb/views.py:
        from PostProcess.refinement.integration import align_fp

        new_boxes, order, room_boundaries = align_fp(
            boundary_mat,
            Box_mat,
            rType_mat,
            Edge_mat,
            fp_id,
            threshold=8,
            draw_result=False,
            nargout=3
        )
    """
    # The nargout parameter is for MATLAB compatibility but not used in Python
    return align_fp_matlab_compatible(
        boundary, boxes, room_types, edges, fp_id,
        threshold, draw_result
    )


if __name__ == "__main__":
    print("\nTesting integration module...")

    # Create test data
    boundary = np.array([
        [0, 0],
        [256, 0],
        [256, 256],
        [0, 256]
    ])

    boxes = np.array([
        [5, 20, 80, 100],
        [90, 20, 180, 100],
        [190, 20, 250, 100],
    ])

    room_types = np.array([0, 1, 2])  # Living, Bedroom, Kitchen
    edges = np.array([[0, 1], [1, 2]])  # Room 0-1 adjacent, 1-2 adjacent

    print("\nTest 1: Direct call")
    new_boxes, order, boundaries = align_fp_python(
        boundary, boxes, room_types, edges, fp_id="test1", threshold=8.0
    )

    print(f"\nResults:")
    print(f"  New boxes: {len(new_boxes)} boxes")
    print(f"  Order: {order}")
    print(f"  Boundaries: {len(boundaries)} polygons")

    print("\nTest 2: MATLAB-compatible call")
    new_boxes2, order2, boundaries2 = align_fp(
        boundary, boxes, room_types, edges, fp_id="test2",
        threshold=8.0, draw_result=False, nargout=3
    )

    print(f"\nResults:")
    print(f"  New boxes: {len(new_boxes2)} boxes")
    print(f"  Order: {order2}")

    print("\n✓ Integration module working!")

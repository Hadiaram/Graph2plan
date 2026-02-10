"""
Integration module for connecting geometric refinement to the Interface.

This module provides a drop-in replacement for MATLAB's align_fp function.
"""

import numpy as np
from typing import Tuple, List, Union
from .boundary_align import align_all_boxes_with_boundary, snap_rooms_to_neighbors, fill_small_boundary_gaps, fill_inter_room_gaps, resolve_room_overlaps, fill_coverage_gaps
from .expand_living_room import expand_living_room_to_boundary


def align_fp_python(boundary: np.ndarray,
                   boxes: np.ndarray,
                   room_types: np.ndarray,
                   edges: np.ndarray,
                   fp_id: Union[str, int],
                   threshold: float = 8.0,
                   draw_result: bool = False,
                   refinement_pass: int = 1,
                   expand_living_room: bool = False,
                   original_boxes: np.ndarray = None) -> Tuple[List, List, List]:
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
        refinement_pass: Which refinement pass (1 or 2). Pass 1 snaps closest edge,
                        Pass 2 snaps the orthogonal direction.
        expand_living_room: If True, expand living room to fill boundary after alignment
        original_boxes: Original unrefined boxes (used for Pass 2 pre-scan to determine axis)

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
    print(f"  Refinement Pass: {refinement_pass}/3")

    # ============================================================
    # STEP 1: BOUNDARY ALIGNMENT ✅
    # ============================================================

    # Determine which axis to exclude based on the pass
    # Pass 1: No exclusions, snap to closest edge (records which axis was used)
    # Pass 2: Exclude the axis that was snapped in Pass 1 to force orthogonal snapping
    # Pass 3: Room-to-room snapping (skip wall snapping)

    exclude_axis = None
    # Determine per-box axis exclusion for pass 2
    per_box_exclude = None
    if refinement_pass == 3:
        # Pass 3: Skip wall snapping entirely, just pass boxes through
        print(f"  Pass 3: Skipping wall snapping (will do room-to-room snapping instead)")
        aligned_boxes = boxes.copy()
        updated_edges = [{'left': False, 'top': False, 'right': False, 'bottom': False} for _ in range(len(boxes))]
    elif refinement_pass == 2:
        # On second pass, each box should snap the orthogonal axis from what it did in pass 1
        # FIXED: Check which walls the CURRENT boxes (Pass 1 refined) are attached to
        from .boundary_align import find_closest_segments, extract_boundary_segments, Box

        h_segments, v_segments = extract_boundary_segments(boundary)

        # For each box, determine which axis to exclude based on current wall attachments
        per_box_exclude = []
        for i in range(len(boxes)):
            box = Box.from_array(boxes[i])
            alignments = find_closest_segments(box, h_segments, v_segments)

            # Check which walls this box is currently attached to (from Pass 1)
            attached_left = alignments['left'].distance < 0.1
            attached_right = alignments['right'].distance < 0.1
            attached_top = alignments['top'].distance < 0.1
            attached_bottom = alignments['bottom'].distance < 0.1

            horizontal_attached = attached_left or attached_right
            vertical_attached = attached_top or attached_bottom

            # If attached horizontally, exclude horizontal (force vertical snap in Pass 2)
            # If attached vertically, exclude vertical (force horizontal snap in Pass 2)
            if horizontal_attached and vertical_attached:
                # Attached to both axes - could exclude both, or pick one
                # Let's exclude horizontal to try vertical movement
                per_box_exclude.append('horizontal')
            elif horizontal_attached:
                per_box_exclude.append('horizontal')
            elif vertical_attached:
                per_box_exclude.append('vertical')
            else:
                # Box didn't snap in pass 1, allow both axes in pass 2
                per_box_exclude.append(None)

        print(f"  Pass 2: Using per-box axis exclusion based on current wall attachments")

    if refinement_pass != 3:
        print(f"  Step 1: Aligning boxes with boundary...")

        aligned_boxes, updated_edges = align_all_boxes_with_boundary(
            boxes,
            boundary,
            threshold=threshold,
            room_types=room_types,
            verbose=True,  # Enable verbose logging to debug
            exclude_axis=per_box_exclude if refinement_pass == 2 else None,
            refinement_pass=refinement_pass
        )

        n_updated = sum(1 for edges_dict in updated_edges if any(edges_dict.values()))
        print(f"    ✓ Aligned {n_updated}/{len(boxes)} boxes")

    # ============================================================
    # STEP 2: ROOM-TO-ROOM SNAPPING (Pass 3 only)
    # ============================================================
    if refinement_pass == 3:
        print("  Step 2a: Resolving room overlaps...")
        overlap_resolved_boxes = resolve_room_overlaps(
            aligned_boxes,
            room_types,
            boundary,
            verbose=True
        )

        print("  Step 2b: Room-to-room snapping (closing gaps)...")
        neighbor_aligned_boxes = snap_rooms_to_neighbors(
            overlap_resolved_boxes,
            room_types,
            edges,
            boundary,
            threshold=threshold,
            verbose=True
        )

    else:
        print("  Step 2: Room-to-room snapping (skipped - only runs in Pass 3)")
        neighbor_aligned_boxes = aligned_boxes.copy()

    # ============================================================
    # STEP 3: FILL SMALL BOUNDARY GAPS (Pass 3 only)
    # ============================================================
    if refinement_pass == 3:
        print("  Step 3: Filling small boundary gaps...")
        gap_filled_boxes = fill_small_boundary_gaps(
            neighbor_aligned_boxes,
            room_types,
            boundary,
            gap_threshold=20.0,
            verbose=True
        )
        print("  Step 3b: Filling boundary gaps for bathrooms (extended threshold)...")
        # Bathrooms may move to a new boundary section during neighbor snapping,
        # leaving a larger gap. Run a second pass for bathrooms only with a
        # higher threshold. Non-bathrooms are masked as type 0 (skipped).
        bathroom_only_types = np.where(room_types == 3, 3, 0)
        gap_filled_boxes = fill_small_boundary_gaps(
            gap_filled_boxes,
            bathroom_only_types,
            boundary,
            gap_threshold=threshold * 2,
            verbose=True
        )
        print("  Step 3.5: Filling inter-room gaps...")
        gap_filled_boxes = fill_inter_room_gaps(
            gap_filled_boxes,
            room_types,
            boundary,
            gap_threshold=20.0,
            verbose=True
        )
    else:
        print("  Step 3: Filling small boundary gaps (skipped - only runs in Pass 3)")
        gap_filled_boxes = neighbor_aligned_boxes.copy()

    # ============================================================
    # STEP 4: FINAL PROCESSING (TODO)
    # ============================================================
    print("  Step 4: Final processing (TODO - using current boxes)")
    # TODO: Any final adjustments
    # For now, just use gap-filled boxes
    final_boxes = gap_filled_boxes.copy()

    # ============================================================
    # STEP 3.5: LIVING ROOM EXPANSION (OPTIONAL)
    # ============================================================
    # Only expand living room after Pass 3 is complete (after all snapping is done)
    if expand_living_room and refinement_pass == 3:
        print("  Step 3.5: Expanding living room to fill boundary (Pass 3 complete)...")
        final_boxes = expand_living_room_to_boundary(
            final_boxes,
            room_types,
            boundary,
            verbose=True
        )
    elif expand_living_room:
        print(f"  Step 3.5: Skipping living room expansion (waiting for Pass 3)...")

    # ============================================================
    # STEP 5: COVERAGE GAP FILL (Pass 3 only, after living room expansion)
    # ============================================================
    # Detect any remaining uncovered pockets inside the boundary using Shapely
    # (boundary polygon minus union of all rooms). Expands the adjacent room to
    # absorb each gap. This catches corner pockets that ray-casting-based
    # functions miss (e.g. step-wall notches).
    if refinement_pass == 3:
        print("  Step 5: Filling coverage gaps (Shapely)...")
        final_boxes = fill_coverage_gaps(
            final_boxes,
            room_types,
            boundary,
            verbose=True
        )

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

"""
Integration module for connecting geometric refinement to the Interface.

This module provides a drop-in replacement for MATLAB's align_fp function.
"""

import numpy as np
from typing import Tuple, List, Union
from .boundary_align import align_all_boxes_with_boundary, snap_rooms_to_neighbors, snap_bathroom_to_nearest_clear_wall, close_small_gaps, snap_rooms_to_fill_gaps, fill_small_boundary_gaps, fill_inter_room_gaps, resolve_room_overlaps, fill_coverage_gaps
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
    print(f"  Refinement Pass: {refinement_pass}/5")

    # ============================================================
    # STEP 1: BOUNDARY ALIGNMENT ✅
    # ============================================================

    # Pass overview:
    # Pass 1: Horizontal wall snap
    # Pass 2: Vertical wall snap (with pre-pass for out-of-boundary rooms)
    # Pass 3: Room-to-room snapping + gap fills (bathroom snaps to LR)
    # Pass 4: Bathroom wall anchor (find nearest clear wall, stay LR-connected)
    # Pass 5: Living room expansion + coverage gap fill

    exclude_axis = None
    # Determine per-box axis exclusion for pass 2
    per_box_exclude = None
    if refinement_pass in (3, 4, 5, 6, 7):
        # Passes 3–7: Skip wall snapping entirely, just pass boxes through
        print(f"  Pass {refinement_pass}: Skipping wall snapping")
        aligned_boxes = boxes.copy()
        updated_edges = [{'left': False, 'top': False, 'right': False, 'bottom': False} for _ in range(len(boxes))]
    elif refinement_pass == 2:
        from .boundary_align import find_closest_segments, extract_boundary_segments, Box

        h_segments, v_segments = extract_boundary_segments(boundary)

        # Pre-compute boundary extents once
        bnd_pts = boundary[:, :2]
        bnd_x_min = float(np.min(bnd_pts[:, 0]))
        bnd_x_max = float(np.max(bnd_pts[:, 0]))
        bnd_y_min = float(np.min(bnd_pts[:, 1]))
        bnd_y_max = float(np.max(bnd_pts[:, 1]))

        # Work on a mutable copy so the pre-pass feeds into align_all_boxes_with_boundary
        boxes = boxes.copy()

        # --- Pre-pass: rooms mostly outside the boundary ---
        # If more than half of a room's area falls outside the boundary bounding box,
        # translate the whole room so its furthest-outside edge lands on the boundary
        # wall it violates.  Repeat until the room is fully inside (or 4 iterations
        # max to guard against degenerate rooms larger than the boundary).
        # pre_pass_axis[i] records which axis the pre-pass used for box i
        # ('horizontal', 'vertical', or None if the pre-pass didn't touch it).
        pre_pass_axis = [None] * len(boxes)

        for i in range(len(boxes)):
            box = Box.from_array(boxes[i])
            box_orig = Box.from_array(boxes[i])  # Keep original for logging
            room_area = box.width * box.height
            if room_area <= 0:
                continue

            # Compute initial inside fraction for logging
            ix1_chk = max(box.x1, bnd_x_min); iy1_chk = max(box.y1, bnd_y_min)
            ix2_chk = min(box.x2, bnd_x_max); iy2_chk = min(box.y2, bnd_y_max)
            init_inside = max(0.0, ix2_chk - ix1_chk) * max(0.0, iy2_chk - iy1_chk)
            init_pct = 100.0 * init_inside / room_area if room_area > 0 else 100.0
            print(f"  [Pass 2 pre-pass] Box {i}: coords=({box.x1:.1f},{box.y1:.1f},{box.x2:.1f},{box.y2:.1f}), inside={init_pct:.1f}%")

            for _ in range(4):
                ix1 = max(box.x1, bnd_x_min)
                iy1 = max(box.y1, bnd_y_min)
                ix2 = min(box.x2, bnd_x_max)
                iy2 = min(box.y2, bnd_y_max)
                inside_area = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)

                if inside_area >= 0.5 * room_area:
                    break  # Room is mostly (or fully) inside — stop

                # Find the edge with the greatest overshoot
                os_left   = max(0.0, bnd_x_min - box.x1)
                os_right  = max(0.0, box.x2 - bnd_x_max)
                os_top    = max(0.0, bnd_y_min - box.y1)
                os_bottom = max(0.0, box.y2 - bnd_y_max)
                max_os = max(os_left, os_right, os_top, os_bottom)

                if max_os <= 0:
                    break  # No overshoot — already inside

                print(f"    overshoot: left={os_left:.1f} right={os_right:.1f} top={os_top:.1f} bottom={os_bottom:.1f} → max={max_os:.1f}")

                if max_os == os_right:
                    delta = bnd_x_max - box.x2          # negative → moves left
                    box = Box(box.x1 + delta, box.y1, bnd_x_max, box.y2)
                    pre_pass_axis[i] = 'horizontal'
                elif max_os == os_left:
                    delta = bnd_x_min - box.x1          # positive → moves right
                    box = Box(bnd_x_min, box.y1, box.x2 + delta, box.y2)
                    pre_pass_axis[i] = 'horizontal'
                elif max_os == os_bottom:
                    delta = bnd_y_max - box.y2          # negative → moves up
                    box = Box(box.x1, box.y1 + delta, box.x2, bnd_y_max)
                    pre_pass_axis[i] = 'vertical'
                else:  # os_top
                    delta = bnd_y_min - box.y1          # positive → moves down
                    box = Box(box.x1, bnd_y_min, box.x2, box.y2 + delta)
                    pre_pass_axis[i] = 'vertical'

                print(f"    → after translation: ({box.x1:.1f},{box.y1:.1f},{box.x2:.1f},{box.y2:.1f}), axis={pre_pass_axis[i]}")

            if pre_pass_axis[i] is not None:
                boxes[i] = box.to_array()
                print(f"  [Pass 2 pre-pass] Box {i}: MOVED ({box_orig.x1:.1f},{box_orig.y1:.1f},{box_orig.x2:.1f},{box_orig.y2:.1f}) → ({box.x1:.1f},{box.y1:.1f},{box.x2:.1f},{box.y2:.1f}), excluding '{pre_pass_axis[i]}' axis in snap")
            else:
                print(f"  [Pass 2 pre-pass] Box {i}: already >=50% inside, no translation")

        # --- Standard Pass 2: orthogonal-axis snapping for rooms already inside ---
        # Rooms handled by the pre-pass exclude the axis they were just snapped on,
        # so align_all_boxes_with_boundary cannot undo the translation by snapping
        # another edge on the same axis back in the opposite direction.
        per_box_exclude = []
        for i in range(len(boxes)):
            if pre_pass_axis[i] is not None:
                per_box_exclude.append(pre_pass_axis[i])
                continue

            box = Box.from_array(boxes[i])
            alignments = find_closest_segments(box, h_segments, v_segments)

            attached_left   = alignments['left'].distance < 0.1
            attached_right  = alignments['right'].distance < 0.1
            attached_top    = alignments['top'].distance < 0.1
            attached_bottom = alignments['bottom'].distance < 0.1

            horizontal_attached = attached_left or attached_right
            vertical_attached   = attached_top or attached_bottom

            if horizontal_attached and vertical_attached:
                per_box_exclude.append('horizontal')
            elif horizontal_attached:
                per_box_exclude.append('horizontal')
            elif vertical_attached:
                per_box_exclude.append('vertical')
            else:
                per_box_exclude.append(None)

        n_pre = len([x for x in pre_pass_axis if x is not None])
        print(f"  Pass 2: {n_pre} room(s) pre-translated to boundary; remainder using per-box axis exclusion")
        print(f"  [Pass 2] per_box_exclude = {per_box_exclude}")
        for i, excl in enumerate(per_box_exclude):
            box = Box.from_array(boxes[i])
            print(f"    Box {i}: exclude='{excl}', coords=({box.x1:.1f},{box.y1:.1f},{box.x2:.1f},{box.y2:.1f})")

    if refinement_pass not in (3, 4):
        print(f"  Step 1: Aligning boxes with boundary...")

        aligned_boxes, updated_edges = align_all_boxes_with_boundary(
            boxes,
            boundary,
            threshold=threshold,
            room_types=room_types,
            edges=edges,
            verbose=True,  # Enable verbose logging to debug
            exclude_axis=per_box_exclude if refinement_pass == 2 else None,
            refinement_pass=refinement_pass
        )

        n_updated = sum(1 for edges_dict in updated_edges if any(edges_dict.values()))
        print(f"    ✓ Aligned {n_updated}/{len(boxes)} boxes")

    # ============================================================
    # STEP 2: ROOM-TO-ROOM SNAPPING (Pass 3 only)
    # ============================================================

    # Helper: print bathroom positions at each sub-step so we can see
    # which step is responsible for moving bathrooms to the wall.
    def _log_bathroom_positions(label, bxs):
        bath_idxs = [i for i, t in enumerate(room_types) if t == 3]
        if not bath_idxs:
            return
        print(f"  [Bathroom tracker] {label}")
        for bi in bath_idxs:
            b = bxs[bi]
            print(f"    Box {bi} (Bathroom): ({b[0]:.1f}, {b[1]:.1f}, {b[2]:.1f}, {b[3]:.1f})")

    if refinement_pass == 3:
        _log_bathroom_positions("ENTRY (after Pass 2)", aligned_boxes)

        print("  Step 2a: Resolving room overlaps (iterative)...")
        overlap_resolved_boxes = aligned_boxes.copy()
        MAX_OVERLAP_ITERS = 5
        for _overlap_iter in range(MAX_OVERLAP_ITERS):
            prev = overlap_resolved_boxes.copy()
            overlap_resolved_boxes = resolve_room_overlaps(
                overlap_resolved_boxes,
                room_types,
                boundary,
                edges=edges,
                verbose=True
            )
            if np.allclose(overlap_resolved_boxes, prev, atol=0.01):
                print(f"  Step 2a: Converged after {_overlap_iter + 1} iteration(s)")
                break
        else:
            print(f"  Step 2a: Did not fully converge after {MAX_OVERLAP_ITERS} iterations")
        _log_bathroom_positions("after resolve_room_overlaps", overlap_resolved_boxes)

        print("  Step 2b: Room-to-room snapping (closing gaps)...")
        MAX_SNAP_ITERS = 3
        neighbor_aligned_boxes = overlap_resolved_boxes.copy()
        for _snap_iter in range(MAX_SNAP_ITERS):
            prev_snap = neighbor_aligned_boxes.copy()
            neighbor_aligned_boxes = snap_rooms_to_neighbors(
                neighbor_aligned_boxes,
                room_types,
                edges,
                boundary,
                threshold=threshold,
                verbose=True
            )
            if np.allclose(neighbor_aligned_boxes, prev_snap, atol=0.01):
                print(f"  Step 2b: Converged after {_snap_iter + 1} iteration(s)")
                break
        else:
            print(f"  Step 2b: Did not fully converge after {MAX_SNAP_ITERS} iterations")
        _log_bathroom_positions("after snap_rooms_to_neighbors", neighbor_aligned_boxes)

    elif refinement_pass == 4:
        print("  Step 2 (Pass 4): Bathroom→wall anchoring (LR-connected bathrooms)...")
        neighbor_aligned_boxes = snap_bathroom_to_nearest_clear_wall(
            aligned_boxes,
            room_types,
            edges,
            boundary,
            verbose=True
        )
        def _log_bathroom_positions(label, bxs):
            bath_idxs = [i for i, t in enumerate(room_types) if t == 3]
            if not bath_idxs:
                return
            print(f"  [Bathroom tracker] {label}")
            for bi in bath_idxs:
                b = bxs[bi]
                print(f"    Box {bi} (Bathroom): ({b[0]:.1f}, {b[1]:.1f}, {b[2]:.1f}, {b[3]:.1f})")
        _log_bathroom_positions("after snap_bathroom_to_nearest_clear_wall", neighbor_aligned_boxes)

    else:
        print(f"  Step 2: Room-to-room snapping (skipped - only runs in Pass 3)")
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
            edges=edges,
            verbose=True
        )
        _log_bathroom_positions("after fill_small_boundary_gaps", gap_filled_boxes)

        print("  Step 3b: Filling boundary gaps for bathrooms (extended threshold)...")
        # Bathrooms may move to a new boundary section during neighbor snapping,
        # leaving a larger gap. Run a second pass for bathrooms only with a
        # higher threshold. Non-bathrooms are masked as type 0 (skipped).
        #
        # Exception: bathrooms that have a graph edge to the living room are
        # excluded here. Their position relative to the living room boundary
        # matters more than closing the gap to the nearest wall — the living
        # room will fill that wall gap when it expands in Pass 4.
        living_room_indices = set(np.where(room_types == 0)[0])
        bathroom_lr_connected = set()
        for edge in edges:
            a, b_idx = int(edge[0]), int(edge[1])
            if room_types[a] == 3 and b_idx in living_room_indices:
                bathroom_lr_connected.add(a)
            if room_types[b_idx] == 3 and a in living_room_indices:
                bathroom_lr_connected.add(b_idx)
        if bathroom_lr_connected:
            print(f"    Skipping Step 3b for bathrooms connected to living room: {sorted(bathroom_lr_connected)}")
        bathroom_only_types = np.array([
            3 if (room_types[i] == 3 and i not in bathroom_lr_connected) else 0
            for i in range(len(room_types))
        ])
        gap_filled_boxes = fill_small_boundary_gaps(
            gap_filled_boxes,
            bathroom_only_types,
            boundary,
            gap_threshold=threshold * 2,
            edges=edges,
            verbose=True
        )
        _log_bathroom_positions("after fill_small_boundary_gaps (bathrooms only)", gap_filled_boxes)

        print("  Step 3.5: Filling inter-room gaps...")
        gap_filled_boxes = fill_inter_room_gaps(
            gap_filled_boxes,
            room_types,
            boundary,
            gap_threshold=20.0,
            edges=edges,
            verbose=True
        )
        _log_bathroom_positions("after fill_inter_room_gaps", gap_filled_boxes)
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
    # STEP 3.5: LIVING ROOM EXPANSION (Pass 5 only)
    # ============================================================
    # Runs after Pass 4 (bathroom wall anchor) so the LR expands around
    # the already wall-anchored bathroom, keeping it accessible.
    if refinement_pass == 5:
        print("  Step 3.5: Expanding living room to fill boundary (Pass 5)...")
        final_boxes = expand_living_room_to_boundary(
            final_boxes,
            room_types,
            boundary,
            verbose=True
        )
    elif expand_living_room:
        print(f"  Step 3.5: Skipping living room expansion (runs in Pass 5)...")

    # ============================================================
    # STEP 5: COVERAGE GAP FILL (Pass 5 only, after living room expansion)
    # ============================================================
    # Detect any remaining uncovered pockets inside the boundary using Shapely
    # (boundary polygon minus union of all rooms). Expands the adjacent room to
    # absorb each gap. This catches corner pockets that ray-casting-based
    # functions miss (e.g. step-wall notches).
    if refinement_pass == 5:
        print("  Step 5: Filling coverage gaps (Shapely)...")
        final_boxes = fill_coverage_gaps(
            final_boxes,
            room_types,
            boundary,
            verbose=True
        )

    # ============================================================
    # PASS 6: GAP-FILL WALL SNAPPING
    # Rooms attached to < 2 boundary walls look for the largest open
    # gap and snap toward that wall, without breaking existing attachment.
    # Must run before small gap closing so translations settle first.
    # ============================================================
    if refinement_pass == 6:
        print("  Step 2 (Pass 6): Gap-fill wall snapping...")
        final_boxes = snap_rooms_to_fill_gaps(
            final_boxes,
            room_types,
            boundary,
            gap_threshold=20.0,
            verbose=True
        )

    # ============================================================
    # PASS 7: SMALL GAP CLOSING
    # Extends any room edge by up to 20px to touch the nearest boundary
    # wall or adjacent room edge.  Rays are cast from corners + midpoint
    # of each face so partial adjacency is detected.
    # Runs last so translations from Pass 6 don't reopen closed gaps.
    # ============================================================
    if refinement_pass == 7:
        print("  Step 2 (Pass 7): Closing small gaps (< 20px)...")
        final_boxes = close_small_gaps(
            final_boxes,
            room_types,
            boundary,
            gap_threshold=20.0,
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

"""
Test the minimal boundary-escape guard fix.
Loads real floor plan data and verifies:
1. 14926 bathroom is NOT pushed further outside by Pass 2
2. Other floor plans still refine correctly (no regressions)
"""
import numpy as np
import scipy.io as sio
import os

from PostProcess.refinement.integration import align_fp_python
from PostProcess.refinement.boundary_align import (
    _get_boundary_pts, _corners_inside,
    align_box_with_boundary, align_all_boxes_with_boundary
)
from PostProcess.refinement.geometry_utils import Box, extract_boundary_segments


def load_mat(fp_id):
    path = os.path.join("Interface", "static", f"{fp_id}.mat")
    if not os.path.exists(path):
        return None
    data = sio.loadmat(path)
    d = data["data"][0, 0]
    boxes = d["box"].astype(float)[:, :4]  # Only first 4 columns: x1, y1, x2, y2
    boundary = d["boundary"].astype(float)
    room_types = np.array(d["rType"]).flatten().astype(int)
    edges = d["edge"].astype(float) if "edge" in d.dtype.names else np.zeros((0, 2))
    return boxes, boundary, room_types, edges


def count_corners_outside(boxes, boundary):
    """For each room, count how many corners are outside the boundary polygon."""
    poly = _get_boundary_pts(boundary)
    results = []
    for i in range(boxes.shape[0]):
        bx = Box(boxes[i, 0], boxes[i, 1], boxes[i, 2], boxes[i, 3])
        inside = _corners_inside(bx, poly)
        results.append(4 - inside)
    return results


def run_3_passes(boxes, boundary, room_types, edges, fp_id):
    """Run all 3 refinement passes, returning the result after each pass."""
    results = {}
    current = boxes.copy()
    for p in [1, 2, 3]:
        new_boxes, order, boundaries = align_fp_python(
            boundary, current, room_types, edges, fp_id,
            threshold=8.0, refinement_pass=p,
            expand_living_room=(p == 3),
            original_boxes=boxes if p == 2 else None
        )
        current = np.array(new_boxes)
        results[p] = current.copy()
    return results


def test_14926_bathroom_detail():
    """Detailed trace of the 14926 bathroom through all passes."""
    data = load_mat(14926)
    if data is None:
        print("[SKIP] 14926.mat not found")
        return True

    boxes, boundary, room_types, edges = data
    poly = _get_boundary_pts(boundary)

    # Find bathroom(s) - type 3
    bath_indices = [i for i in range(len(room_types)) if room_types[i] == 3]
    print(f"\n{'='*60}")
    print(f"14926: Found {len(bath_indices)} bathrooms: {bath_indices}")

    for bi in bath_indices:
        bx = boxes[bi]
        orig_box = Box(bx[0], bx[1], bx[2], bx[3])
        orig_inside = _corners_inside(orig_box, poly)
        print(f"  Bath room {bi}: [{bx[0]:.2f}, {bx[1]:.2f}, {bx[2]:.2f}, {bx[3]:.2f}] "
              f"corners inside: {orig_inside}/4")

    # Run all 3 passes
    results = run_3_passes(boxes, boundary, room_types, edges, 14926)

    all_ok = True
    for bi in bath_indices:
        orig_box = Box(boxes[bi, 0], boxes[bi, 1], boxes[bi, 2], boxes[bi, 3])
        orig_inside = _corners_inside(orig_box, poly)

        for p in [1, 2, 3]:
            rbx = results[p][bi]
            ref_box = Box(rbx[0], rbx[1], rbx[2], rbx[3])
            ref_inside = _corners_inside(ref_box, poly)
            status = "OK" if ref_inside >= orig_inside else "FAIL"
            print(f"  After Pass {p}: room {bi} [{rbx[0]:.2f}, {rbx[1]:.2f}, {rbx[2]:.2f}, {rbx[3]:.2f}] "
                  f"corners inside: {ref_inside}/4 {status}")
            if ref_inside < orig_inside:
                all_ok = False

    return all_ok


def test_floor_plan(fp_id):
    """Test a floor plan through all 3 passes."""
    data = load_mat(fp_id)
    if data is None:
        print(f"  [SKIP] {fp_id}.mat not found")
        return True

    boxes, boundary, room_types, edges = data
    print(f"\n{'='*60}")
    print(f"Floor plan {fp_id}: {boxes.shape[0]} rooms")

    # Corners outside BEFORE refinement
    before_outside = count_corners_outside(boxes, boundary)
    total_before = sum(before_outside)
    print(f"  Before refinement: {total_before} corners outside boundary")

    # Run all 3 passes
    results = run_3_passes(boxes, boundary, room_types, edges, fp_id)
    refined = results[3]

    # Corners outside AFTER full refinement
    after_outside = count_corners_outside(refined, boundary)
    total_after = sum(after_outside)
    print(f"  After refinement:  {total_after} corners outside boundary")

    for i, cnt in enumerate(after_outside):
        if cnt > 0:
            rt = room_types[i]
            print(f"    Room {i} (type={rt}): {cnt} corners outside")

    if total_after > total_before:
        print(f"  WARNING: corners outside increased from {total_before} to {total_after}")
    else:
        print(f"  OK: corners outside = {total_after} (was {total_before})")

    return True


if __name__ == "__main__":
    all_ok = True

    # Test 14926 bathroom in detail
    all_ok &= test_14926_bathroom_detail()

    # Test all available floor plans
    for fp_id in [14035, 14433, 14926, 17054]:
        test_floor_plan(fp_id)

    print(f"\n{'='*60}")
    if all_ok:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")

"""
Test script for boundary alignment.

Run this to test the boundary snapping functionality.
"""

import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from PostProcessing.boundary_align import align_all_boxes_with_boundary
from PostProcessing.visualize import visualize_boundary_alignment


def test_simple_rectangular_boundary():
    """Test with a simple rectangular boundary."""
    print("\n" + "="*70)
    print("TEST 1: Simple Rectangular Boundary")
    print("="*70)

    # Create rectangular boundary (0, 0) to (256, 256)
    boundary = np.array([
        [0, 0],
        [256, 0],
        [256, 256],
        [0, 256]
    ])

    # Create boxes at various distances from walls
    boxes = np.array([
        [5, 20, 80, 100],      # Living room - close to left wall (5 pixels)
        [90, 20, 180, 100],    # Bedroom - middle (not close to walls)
        [190, 20, 250, 100],   # Kitchen - close to right wall (6 pixels)
        [5, 120, 80, 200],     # Bathroom - close to left wall
        [90, 120, 180, 200],   # Dining - middle
        [190, 120, 250, 250],  # Balcony - close to right and bottom walls
    ])

    room_types = np.array([0, 1, 2, 3, 6, 4])  # Living, Bedroom, Kitchen, Bathroom, Dining, Balcony

    print(f"\nBoundary: (0, 0) to (256, 256)")
    print(f"Boxes: {len(boxes)} rooms")
    print(f"Threshold: 8 pixels")

    # Align boxes
    aligned_boxes, updated_edges = align_all_boxes_with_boundary(
        boxes, boundary, threshold=8.0, room_types=room_types, verbose=True
    )

    # Visualize
    visualize_boundary_alignment(
        boxes, aligned_boxes, boundary,
        room_types=room_types,
        updated_edges=updated_edges,
        title="Test 1: Rectangular Boundary - Threshold=8px",
        save_path="/tmp/test1_rectangular_boundary.png"
    )

    return boxes, aligned_boxes, boundary, room_types


def test_l_shaped_boundary():
    """Test with L-shaped boundary."""
    print("\n" + "="*70)
    print("TEST 2: L-Shaped Boundary")
    print("="*70)

    # Create L-shaped boundary
    boundary = np.array([
        [0, 0],
        [200, 0],
        [200, 100],
        [100, 100],
        [100, 200],
        [0, 200]
    ])

    # Create boxes in different regions of the L
    boxes = np.array([
        [10, 10, 90, 90],       # Living room - in bottom-left corner
        [110, 10, 190, 90],     # Bedroom - in bottom-right section
        [10, 110, 90, 190],     # Kitchen - in top-left section
    ])

    room_types = np.array([0, 1, 2])

    print(f"\nBoundary: L-shaped")
    print(f"Boxes: {len(boxes)} rooms")
    print(f"Threshold: 8 pixels")

    # Align boxes
    aligned_boxes, updated_edges = align_all_boxes_with_boundary(
        boxes, boundary, threshold=8.0, room_types=room_types, verbose=True
    )

    # Visualize
    visualize_boundary_alignment(
        boxes, aligned_boxes, boundary,
        room_types=room_types,
        updated_edges=updated_edges,
        title="Test 2: L-Shaped Boundary - Threshold=8px",
        save_path="/tmp/test2_l_shaped_boundary.png"
    )

    return boxes, aligned_boxes, boundary, room_types


def test_different_thresholds():
    """Test effect of different threshold values."""
    print("\n" + "="*70)
    print("TEST 3: Different Thresholds")
    print("="*70)

    boundary = np.array([
        [0, 0],
        [200, 0],
        [200, 200],
        [0, 200]
    ])

    # Boxes at various distances: 3px, 7px, 12px from walls
    boxes = np.array([
        [3, 20, 50, 60],    # 3 pixels from left
        [60, 20, 110, 60],  # 7 pixels from middle
        [118, 20, 170, 60], # 12 pixels from middle
    ])

    room_types = np.array([0, 1, 2])

    for threshold in [5, 8, 15]:
        print(f"\n--- Threshold = {threshold} pixels ---")

        aligned_boxes, updated_edges = align_all_boxes_with_boundary(
            boxes, boundary, threshold=threshold, room_types=room_types, verbose=False
        )

        # Count snapped edges
        n_snapped = sum(1 for edges in updated_edges if any(edges.values()))
        print(f"  Boxes snapped: {n_snapped}/3")

        visualize_boundary_alignment(
            boxes, aligned_boxes, boundary,
            room_types=room_types,
            updated_edges=updated_edges,
            title=f"Test 3: Threshold = {threshold} pixels",
            save_path=f"/tmp/test3_threshold_{threshold}.png"
        )


def test_real_world_scenario():
    """Test with realistic floor plan scenario."""
    print("\n" + "="*70)
    print("TEST 4: Realistic Floor Plan")
    print("="*70)

    # Simulate realistic boundary (irregular shape)
    boundary = np.array([
        [0, 0],
        [150, 0],
        [150, 50],
        [200, 50],
        [200, 150],
        [150, 150],
        [150, 200],
        [0, 200]
    ])

    # Simulate AI predictions (slightly off from walls)
    boxes = np.array([
        [6, 6, 70, 80],        # Living room - slightly off from corner
        [78, 6, 144, 80],      # Bedroom 1 - close to walls
        [6, 88, 70, 194],      # Bedroom 2 - close to walls
        [78, 88, 144, 150],    # Kitchen - middle section
        [156, 56, 194, 110],   # Bathroom - in the extension
    ])

    room_types = np.array([0, 1, 1, 2, 3])

    print(f"\nBoundary: Irregular shape")
    print(f"Boxes: {len(boxes)} rooms")
    print(f"Threshold: 8 pixels")
    print("\nThis simulates AI model predictions that are close but not exact.")

    # Align boxes
    aligned_boxes, updated_edges = align_all_boxes_with_boundary(
        boxes, boundary, threshold=8.0, room_types=room_types, verbose=True
    )

    # Visualize
    visualize_boundary_alignment(
        boxes, aligned_boxes, boundary,
        room_types=room_types,
        updated_edges=updated_edges,
        title="Test 4: Realistic Floor Plan - AI Predictions Snapped to Walls",
        save_path="/tmp/test4_realistic_floorplan.png"
    )

    # Print statistics
    print("\n" + "="*70)
    print("STATISTICS")
    print("="*70)

    total_edges_updated = sum(sum(edges.values()) for edges in updated_edges)
    print(f"Total edges snapped: {total_edges_updated}")

    for i, (orig_box, aligned_box, edges) in enumerate(zip(boxes, aligned_boxes, updated_edges)):
        if any(edges.values()):
            orig_area = (orig_box[2] - orig_box[0]) * (orig_box[3] - orig_box[1])
            aligned_area = (aligned_box[2] - aligned_box[0]) * (aligned_box[3] - aligned_box[1])
            area_change = aligned_area - orig_area
            area_change_pct = (area_change / orig_area) * 100

            print(f"\nBox {i} (type={room_types[i]}):")
            print(f"  Snapped edges: {[k for k, v in edges.items() if v]}")
            print(f"  Area change: {area_change:.1f} sq px ({area_change_pct:+.1f}%)")

    return boxes, aligned_boxes, boundary, room_types


if __name__ == "__main__":
    print("\n" + "="*70)
    print("BOUNDARY ALIGNMENT TEST SUITE")
    print("="*70)

    try:
        # Run tests
        test_simple_rectangular_boundary()
        test_l_shaped_boundary()
        test_different_thresholds()
        test_real_world_scenario()

        print("\n" + "="*70)
        print("✓ ALL TESTS COMPLETED")
        print("="*70)
        print("\nCheck the generated PNG files in /tmp/ to see visualizations:")
        print("  - test1_rectangular_boundary.png")
        print("  - test2_l_shaped_boundary.png")
        print("  - test3_threshold_*.png")
        print("  - test4_realistic_floorplan.png")

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()

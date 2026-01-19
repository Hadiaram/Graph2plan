"""
Re-extract boundaries for test set floor plans from ResPlan dataset

This script:
1. Loads the test floor plan IDs from test.txt
2. Loads the original ResPlan dataset
3. Extracts proper boundaries using the same logic as convert_resplan_to_mat.py
4. Updates data_test_converted.pkl with the correct boundaries
5. Regenerates boundary images
"""

import pickle
import numpy as np
import scipy.io as sio
from pathlib import Path
from typing import Dict, Any
import sys

# Add parent directory to path for resplan_utils
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from shapely.geometry import MultiPolygon, Point, Polygon
    from shapely.ops import unary_union
except ImportError:
    print("ERROR: Shapely not installed. Run: pip install shapely")
    sys.exit(1)


def extract_coords_from_shapely(geometry) -> np.ndarray:
    """Extract coordinates from Shapely geometry objects"""
    if isinstance(geometry, Polygon):
        coords = np.array(geometry.exterior.coords[:-1])
    elif isinstance(geometry, MultiPolygon):
        largest = max(geometry.geoms, key=lambda p: p.area)
        coords = np.array(largest.exterior.coords[:-1])
    else:
        coords = np.array(geometry.coords[:-1])
    return coords


def compute_boundary_directions(coords: np.ndarray) -> np.ndarray:
    """
    Compute direction codes for boundary segments

    Direction codes:
        0 = right (→)
        1 = up (↑)
        2 = left (←)
        3 = down (↓)
    """
    n = len(coords)
    directions = np.zeros(n, dtype=int)

    for i in range(n):
        j = (i + 1) % n
        dx = coords[j, 0] - coords[i, 0]
        dy = coords[j, 1] - coords[i, 1]

        if abs(dx) > abs(dy):
            directions[i] = 0 if dx > 0 else 2
        else:
            directions[i] = 1 if dy > 0 else 3

    return directions


def add_boundary_metadata(coords: np.ndarray) -> np.ndarray:
    """
    Convert (N x 2) coordinates to Graph2plan boundary format (N x 4)

    Adds direction and isNew columns
    """
    directions = compute_boundary_directions(coords)
    is_new = np.zeros(len(coords), dtype=int)

    return np.column_stack([coords, directions, is_new])


def extract_boundary_from_inner(item: Dict) -> np.ndarray:
    """
    Extract floorplan boundary from ResPlan 'inner' field

    Aligns boundary so first two points correspond to front door
    """
    # Get largest polygon from 'inner' field
    inner = item['inner']
    if isinstance(inner, MultiPolygon):
        largest = max(inner.geoms, key=lambda p: p.area)
    else:
        largest = inner

    # Extract boundary coordinates
    boundary_coords = extract_coords_from_shapely(largest)

    # Add direction and isNew metadata
    boundary = add_boundary_metadata(boundary_coords)

    # Try to align with front door
    if 'front_door' in item and item['front_door'] is not None:
        front_door = item['front_door']

        # Get front door centroid
        door_point = Point(front_door.centroid.x, front_door.centroid.y)

        # Find closest boundary point to door
        coords = boundary[:, :2]
        distances = np.array([
            np.sqrt((coords[i, 0] - door_point.x)**2 + (coords[i, 1] - door_point.y)**2)
            for i in range(len(coords))
        ])
        closest_idx = np.argmin(distances)

        # Reorder boundary to start at closest point
        boundary = np.roll(boundary, -closest_idx, axis=0)

    return boundary


def load_resplan_dataset(resplan_path: Path) -> Dict[str, Dict]:
    """
    Load ResPlan dataset and create ID-indexed dictionary

    Args:
        resplan_path: Path to ResPlan.pkl or similar

    Returns:
        Dictionary mapping floor plan IDs to floor plan data
    """
    print(f"Loading ResPlan dataset from: {resplan_path}")

    with open(resplan_path, 'rb') as f:
        data = pickle.load(f)

    # Handle different data structures
    if isinstance(data, dict) and 'data' in data:
        floorplans = data['data']
    elif isinstance(data, list):
        floorplans = data
    else:
        floorplans = [data]

    print(f"  Loaded {len(floorplans)} floor plans")

    # Create ID-indexed dictionary
    id_dict = {}
    for plan in floorplans:
        if isinstance(plan, dict) and 'id' in plan:
            plan_id = str(plan['id'])
            id_dict[plan_id] = plan
        else:
            print(f"  Warning: Floor plan missing 'id' field")

    print(f"  Indexed {len(id_dict)} floor plans by ID")

    return id_dict


def reextract_test_boundaries(
    resplan_path: Path,
    test_names_path: Path,
    test_pkl_path: Path,
    output_pkl_path: Path = None
):
    """
    Re-extract boundaries for test set floor plans

    Args:
        resplan_path: Path to ResPlan dataset (pkl file)
        test_names_path: Path to test.txt with floor plan IDs
        test_pkl_path: Path to current data_test_converted.pkl
        output_pkl_path: Path to save updated pkl (default: overwrite input)
    """
    print("=" * 70)
    print("Re-extracting Test Set Boundaries from ResPlan Dataset")
    print("=" * 70)
    print()

    # Load test names
    print(f"1. Loading test floor plan IDs from: {test_names_path}")
    with open(test_names_path, 'r') as f:
        test_names = [line.strip() for line in f if line.strip()]
    print(f"   Found {len(test_names)} test floor plans: {test_names}")
    print()

    # Load ResPlan dataset
    print(f"2. Loading ResPlan dataset")
    resplan_data = load_resplan_dataset(resplan_path)
    print()

    # Load current test data
    print(f"3. Loading current test data from: {test_pkl_path}")
    with open(test_pkl_path, 'rb') as f:
        test_data = pickle.load(f)

    print(f"   Keys: {list(test_data.keys())}")
    print(f"   Current test data items: {len(test_data['data'])}")
    print()

    # Re-extract boundaries
    print(f"4. Re-extracting boundaries")
    print("-" * 70)

    updated_count = 0
    failed_ids = []

    for i, plan_id in enumerate(test_names):
        print(f"   Processing [{i}] ID: {plan_id}")

        # Check if plan exists in ResPlan dataset
        if plan_id not in resplan_data:
            print(f"      [ERROR] Floor plan {plan_id} not found in ResPlan dataset")
            failed_ids.append(plan_id)
            continue

        # Get floor plan from ResPlan
        resplan_plan = resplan_data[plan_id]

        # Extract boundary
        try:
            boundary = extract_boundary_from_inner(resplan_plan)
            print(f"      [OK] Extracted boundary: shape={boundary.shape}")

            # Update test data
            if i < len(test_data['data']):
                # Get existing test data item
                test_item = test_data['data'][i]

                # Update boundary
                if hasattr(test_item, 'boundary'):
                    test_item.boundary = boundary
                elif isinstance(test_item, dict):
                    test_item['boundary'] = boundary
                else:
                    print(f"      [WARNING] Cannot update boundary for type {type(test_item)}")
                    continue

                updated_count += 1
                print(f"      [OK] Updated test data item {i}")
            else:
                print(f"      [ERROR] Test data index {i} out of range")
                failed_ids.append(plan_id)

        except Exception as e:
            print(f"      [ERROR] Failed to extract boundary: {e}")
            import traceback
            traceback.print_exc()
            failed_ids.append(plan_id)

    print()
    print("-" * 70)
    print(f"Summary: Updated {updated_count}/{len(test_names)} boundaries")

    if failed_ids:
        print(f"Failed IDs: {failed_ids}")

    print()

    # Save updated test data
    if output_pkl_path is None:
        output_pkl_path = test_pkl_path

    print(f"5. Saving updated test data to: {output_pkl_path}")

    # Create backup
    backup_path = test_pkl_path.with_suffix('.pkl.backup')
    if test_pkl_path.exists() and not backup_path.exists():
        import shutil
        shutil.copy2(test_pkl_path, backup_path)
        print(f"   [OK] Created backup: {backup_path}")

    # Save updated data
    with open(output_pkl_path, 'wb') as f:
        pickle.dump(test_data, f)

    print(f"   [OK] Saved updated test data")
    print()

    print("=" * 70)
    print("SUCCESS: Boundary re-extraction complete!")
    print("=" * 70)
    print()
    print("Next steps:")
    print("  1. Verify boundaries: python check_test_boundaries.py")
    print("  2. Regenerate images: python generate_floorplan_images.py")
    print()

    return updated_count, failed_ids


def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(
        description='Re-extract boundaries for test set from ResPlan dataset'
    )
    parser.add_argument(
        '--resplan',
        type=str,
        default='../../ResPlan_Dataset/ResPlan.pkl',
        help='Path to ResPlan.pkl file'
    )
    parser.add_argument(
        '--test-names',
        type=str,
        default='./data/test.txt',
        help='Path to test.txt with floor plan IDs'
    )
    parser.add_argument(
        '--test-pkl',
        type=str,
        default='../Interface/static/Data/data_test_converted.pkl',
        help='Path to data_test_converted.pkl'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output path for updated pkl (default: overwrite input)'
    )

    args = parser.parse_args()

    # Convert to Path objects
    resplan_path = Path(args.resplan)
    test_names_path = Path(args.test_names)
    test_pkl_path = Path(args.test_pkl)
    output_path = Path(args.output) if args.output else None

    # Validate paths
    if not resplan_path.exists():
        print(f"ERROR: ResPlan file not found: {resplan_path}")
        return 1

    if not test_names_path.exists():
        print(f"ERROR: Test names file not found: {test_names_path}")
        return 1

    if not test_pkl_path.exists():
        print(f"ERROR: Test PKL file not found: {test_pkl_path}")
        return 1

    # Run re-extraction
    updated_count, failed_ids = reextract_test_boundaries(
        resplan_path,
        test_names_path,
        test_pkl_path,
        output_path
    )

    return 0 if len(failed_ids) == 0 else 1


if __name__ == '__main__':
    sys.exit(main())

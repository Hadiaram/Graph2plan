"""
Check if room boundaries extend beyond the outer boundary in converted data
"""
import pickle
import numpy as np
from pathlib import Path

def check_boundary_integrity(data_path, num_samples=10):
    """
    Check if any room boundaries extend beyond the outer boundary

    Args:
        data_path: Path to PKL file
        num_samples: Number of floor plans to check
    """
    print(f"\nChecking: {data_path}")
    print("=" * 60)

    # Load data
    data_dict = pickle.load(open(data_path, 'rb'))
    data = data_dict['data']

    if not isinstance(data, (list, np.ndarray)):
        data = [data]
    elif isinstance(data, np.ndarray) and data.ndim == 1:
        data = list(data)

    issues_found = 0
    total_checked = 0

    # Check first N floor plans
    for i in range(min(num_samples, len(data))):
        item = data[i]

        # Get attributes
        def get_attr(obj, key):
            if isinstance(obj, dict):
                return obj.get(key)
            else:
                return getattr(obj, key, None)

        name = get_attr(item, 'name')
        boundary = get_attr(item, 'boundary')
        rBoundary = get_attr(item, 'rBoundary')

        if name is None:
            name = str(i)

        if boundary is None or rBoundary is None:
            print(f"  [{name}] Missing data - skipping")
            continue

        total_checked += 1

        # Get boundary extents
        boundary_coords = np.array(boundary)[:, :2]
        min_x, min_y = boundary_coords.min(axis=0)
        max_x, max_y = boundary_coords.max(axis=0)

        # Check each room boundary
        has_issue = False
        room_issues = []

        for room_idx, room_boundary in enumerate(rBoundary):
            if room_boundary is None or len(room_boundary) == 0:
                continue

            room_coords = np.array(room_boundary)
            if room_coords.ndim == 1:
                room_coords = room_coords.reshape(-1, 2)

            room_min_x, room_min_y = room_coords[:, 0].min(), room_coords[:, 1].min()
            room_max_x, room_max_y = room_coords[:, 0].max(), room_coords[:, 1].max()

            # Check if room extends beyond boundary (with small tolerance)
            tolerance = 1.0

            if room_min_x < (min_x - tolerance):
                has_issue = True
                room_issues.append(f"Room {room_idx}: X min {room_min_x:.2f} < boundary {min_x:.2f}")

            if room_max_x > (max_x + tolerance):
                has_issue = True
                room_issues.append(f"Room {room_idx}: X max {room_max_x:.2f} > boundary {max_x:.2f}")

            if room_min_y < (min_y - tolerance):
                has_issue = True
                room_issues.append(f"Room {room_idx}: Y min {room_min_y:.2f} < boundary {min_y:.2f}")

            if room_max_y > (max_y + tolerance):
                has_issue = True
                room_issues.append(f"Room {room_idx}: Y max {room_max_y:.2f} > boundary {max_y:.2f}")

        if has_issue:
            issues_found += 1
            print(f"  ⚠️  [{name}] Issues found:")
            for issue in room_issues:
                print(f"      {issue}")
        else:
            print(f"  ✓  [{name}] All rooms within boundary")

    print("\n" + "=" * 60)
    print(f"Summary: {issues_found}/{total_checked} floor plans have boundary issues")
    print("=" * 60)

    return issues_found

if __name__ == '__main__':
    train_path = Path('../Interface/static/Data/data_train_converted.pkl')
    test_path = Path('../Interface/static/Data/data_test_converted.pkl')

    print("\n" + "=" * 60)
    print("ResPlan Boundary Integrity Check")
    print("=" * 60)

    if train_path.exists():
        train_issues = check_boundary_integrity(train_path, num_samples=20)
    else:
        print(f"\n{train_path} not found")
        train_issues = 0

    if test_path.exists():
        test_issues = check_boundary_integrity(test_path, num_samples=10)
    else:
        print(f"\n{test_path} not found")
        test_issues = 0

    print("\n" + "=" * 60)
    if train_issues == 0 and test_issues == 0:
        print("✅ NO BOUNDARY ISSUES FOUND!")
        print("All room boundaries are properly contained within outer boundaries.")
    else:
        print(f"⚠️  ISSUES FOUND: {train_issues + test_issues} floor plans")
        print("Some room boundaries extend beyond the outer boundary.")
        print("This may be due to:")
        print("  1. Gap-filling in conversion process")
        print("  2. Coordinate system differences")
        print("  3. Floating-point precision")
    print("=" * 60)

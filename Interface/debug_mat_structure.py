#!/usr/bin/env python3
"""Debug script to inspect .mat file structure"""

import scipy.io as sio
import numpy as np

# Load the mat file
data = sio.loadmat("./static/14926.mat")
fp_data = data['data'][0, 0]

print("=" * 60)
print("MAT File Structure for 14926.mat")
print("=" * 60)

# Print all fields
print("\nAvailable fields:")
for field_name in fp_data.dtype.names:
    field_data = fp_data[field_name]
    if isinstance(field_data, np.ndarray):
        print(f"  {field_name:20s} shape: {field_data.shape}, dtype: {field_data.dtype}")
    else:
        print(f"  {field_name:20s} type: {type(field_data)}")

# Check critical fields for DXF export
print("\n" + "=" * 60)
print("Critical Fields for DXF Export")
print("=" * 60)

# Boundary
if hasattr(fp_data, 'boundary') or 'boundary' in fp_data.dtype.names:
    boundary = fp_data['boundary']
    print(f"\n✓ boundary: {boundary.shape}")
    print(f"  First 3 points: {boundary[:3]}")

# Room Types
if hasattr(fp_data, 'rType') or 'rType' in fp_data.dtype.names:
    rType = fp_data['rType']
    print(f"\n✓ rType: {rType.shape}")
    print(f"  Values: {rType.flatten()}")
else:
    print("\n✗ rType: NOT FOUND")

# Room Boundaries
if hasattr(fp_data, 'rBoundary') or 'rBoundary' in fp_data.dtype.names:
    rBoundary = fp_data['rBoundary']
    print(f"\n✓ rBoundary: shape={rBoundary.shape}, type={type(rBoundary)}")
    if len(rBoundary) > 0:
        print(f"  Number of room boundaries: {len(rBoundary[0])}")
else:
    print("\n✗ rBoundary: NOT FOUND")

# Boxes
if hasattr(fp_data, 'newBox') or 'newBox' in fp_data.dtype.names:
    newBox = fp_data['newBox']
    print(f"\n✓ newBox: {newBox.shape}")
    print(f"  First 3 boxes: {newBox[:3]}")
elif hasattr(fp_data, 'box') or 'box' in fp_data.dtype.names:
    box = fp_data['box']
    print(f"\n✓ box: {box.shape}")
    print(f"  First 3 boxes: {box[:3]}")
else:
    print("\n✗ newBox/box: NOT FOUND")

# Windows
if hasattr(fp_data, 'windows') or 'windows' in fp_data.dtype.names:
    windows = fp_data['windows']
    print(f"\n✓ windows: {windows.shape}")
    if len(windows) > 0:
        print(f"  Number of windows: {len(windows)}")
else:
    print("\n✗ windows: NOT FOUND")

# Edges
if hasattr(fp_data, 'rEdge') or 'rEdge' in fp_data.dtype.names:
    rEdge = fp_data['rEdge']
    print(f"\n✓ rEdge: {rEdge.shape}")
elif hasattr(fp_data, 'edge') or 'edge' in fp_data.dtype.names:
    edge = fp_data['edge']
    print(f"\n✓ edge: {edge.shape}")

print("\n" + "=" * 60)

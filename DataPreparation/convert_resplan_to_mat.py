"""
Convert ResPlan dataset (pkl format) to Graph2plan data.mat format

This script converts a ResPlan dataset stored as a single pkl file
into the data.mat format required by Graph2plan.

Usage:
    1. First run in inspection mode to see your data structure:
       python convert_resplan_to_mat.py --inspect your_resplan.pkl

    2. Then customize the conversion function and run:
       python convert_resplan_to_mat.py --convert your_resplan.pkl output_data.mat
"""

import pickle
import numpy as np
import scipy.io as sio
from pathlib import Path
import argparse
from typing import List, Dict, Any
from collections import namedtuple

# Define the expected Graph2plan data structure
# Each data item should have these fields:
DataItem = namedtuple('DataItem', [
    'name',       # str: filename/id
    'boundary',   # array: (x,y,dir,isNew) - first two points indicate front door
    'order',      # array: room order for visualization
    'rType',      # array: room category indices
    'rBoundary',  # list of arrays: (x,y) boundary points for each room
    'gtBox',      # array: (y0,x0,y1,x1) bounding boxes from original data
    'gtBoxNew',   # array: (x0,y0,x1,y1) bounding boxes after gap filling
    'rEdge'       # array: (u,v,r) room edges with relative position
])

# Room categories mapping (from Graph2plan)
ROOM_CATEGORIES = {
    'LivingRoom': 0,
    'MasterRoom': 1,
    'Kitchen': 2,
    'Bathroom': 3,
    'DiningRoom': 4,
    'ChildRoom': 5,
    'StudyRoom': 6,
    'SecondRoom': 7,
    'GuestRoom': 8,
    'Balcony': 9,
    'Entrance': 10,
    'Storage': 11,
    'Wall-in': 12,
    'External': 13,
    'ExteriorWall': 14,
    'FrontDoor': 15,
    'InteriorWall': 16,
    'InteriorDoor': 17
}

# Edge types (relative positions)
EDGE_TYPES = {
    'left-above': 0,
    'left-below': 1,
    'left-of': 2,
    'above': 3,
    'inside': 4,
    'surrounding': 5,
    'below': 6,
    'right-of': 7,
    'right-above': 8,
    'right-below': 9
}


def inspect_pkl_structure(pkl_path: str, num_samples: int = 3):
    """
    Inspect the structure of the ResPlan pkl file
    """
    print(f"\n{'='*60}")
    print(f"Inspecting: {pkl_path}")
    print(f"{'='*60}\n")

    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    print(f"Data type: {type(data)}")

    if isinstance(data, dict):
        print(f"\nDictionary keys: {list(data.keys())}")
        for key, value in data.items():
            print(f"  {key}: {type(value)}, ", end="")
            if isinstance(value, (list, np.ndarray)):
                print(f"length={len(value)}")
            else:
                print(f"value={value}")

    # Try to access the actual floorplan data
    if isinstance(data, dict) and 'data' in data:
        floorplans = data['data']
    elif isinstance(data, list):
        floorplans = data
    else:
        floorplans = [data]

    print(f"\nTotal number of floorplans: {len(floorplans)}")
    print(f"\nInspecting first {min(num_samples, len(floorplans))} samples:")
    print("-" * 60)

    for i in range(min(num_samples, len(floorplans))):
        print(f"\nSample {i}:")
        sample = floorplans[i]
        print(f"  Type: {type(sample)}")

        if isinstance(sample, dict):
            print(f"  Keys: {list(sample.keys())}")
            for key, value in sample.items():
                print(f"    {key}: {type(value)}", end="")
                if isinstance(value, np.ndarray):
                    print(f", shape={value.shape}, dtype={value.dtype}")
                elif isinstance(value, list):
                    print(f", length={len(value)}")
                    if len(value) > 0:
                        print(f"      First element type: {type(value[0])}")
                else:
                    print(f", value={value}")

        elif hasattr(sample, '__dict__'):
            print(f"  Attributes: {list(vars(sample).keys())}")
            for key, value in vars(sample).items():
                print(f"    {key}: {type(value)}", end="")
                if isinstance(value, np.ndarray):
                    print(f", shape={value.shape}, dtype={value.dtype}")
                elif isinstance(value, list):
                    print(f", length={len(value)}")
                else:
                    print(f", value={value}")
        else:
            print(f"  Value: {sample}")

    print("\n" + "="*60)
    print("NEXT STEPS:")
    print("="*60)
    print("1. Review the structure above")
    print("2. Update the convert_item() function to map your fields")
    print("3. Run with --convert flag to perform conversion")
    print("="*60 + "\n")


def compute_edge_relations(boxes: np.ndarray) -> np.ndarray:
    """
    Compute relative position relationships between rooms

    Args:
        boxes: (N, 4) array of boxes in format (x0, y0, x1, y1)

    Returns:
        edges: (M, 3) array of (u, v, relation_type)
    """
    edges = []
    n_rooms = len(boxes)

    for i in range(n_rooms):
        for j in range(n_rooms):
            if i == j:
                continue

            box_i = boxes[i]
            box_j = boxes[j]

            # Compute centers
            cx_i = (box_i[0] + box_i[2]) / 2
            cy_i = (box_i[1] + box_i[3]) / 2
            cx_j = (box_j[0] + box_j[2]) / 2
            cy_j = (box_j[1] + box_j[3]) / 2

            # Check if boxes overlap (are adjacent)
            overlap_x = not (box_i[2] < box_j[0] or box_j[2] < box_i[0])
            overlap_y = not (box_i[3] < box_j[1] or box_j[3] < box_i[1])

            if not (overlap_x or overlap_y):
                continue  # Not adjacent

            # Determine relative position
            dx = cx_j - cx_i
            dy = cy_j - cy_i

            # Inside/surrounding check
            if (box_j[0] >= box_i[0] and box_j[2] <= box_i[2] and
                box_j[1] >= box_i[1] and box_j[3] <= box_i[3]):
                relation = 4  # inside
            elif (box_i[0] >= box_j[0] and box_i[2] <= box_j[2] and
                  box_i[1] >= box_j[1] and box_i[3] <= box_j[3]):
                relation = 5  # surrounding
            # Directional relations
            elif abs(dx) > abs(dy):  # Horizontal relation dominant
                if dx > 0:  # j is to the right of i
                    if dy > 0:
                        relation = 8  # right-below
                    elif dy < 0:
                        relation = 9  # right-above
                    else:
                        relation = 7  # right-of
                else:  # j is to the left of i
                    if dy > 0:
                        relation = 1  # left-below
                    elif dy < 0:
                        relation = 0  # left-above
                    else:
                        relation = 2  # left-of
            else:  # Vertical relation dominant
                if dy > 0:
                    relation = 6  # below
                else:
                    relation = 3  # above

            edges.append([i, j, relation])

    return np.array(edges) if edges else np.zeros((0, 3))


def convert_item(item: Any, index: int) -> Dict[str, Any]:
    """
    Convert a single ResPlan item to Graph2plan format

    CUSTOMIZE THIS FUNCTION based on your data structure!

    Args:
        item: Single floorplan from your ResPlan dataset
        index: Index for generating a unique name if needed

    Returns:
        Dictionary with Graph2plan fields
    """
    converted = {}

    # ============================================================
    # CUSTOMIZE THESE MAPPINGS BASED ON YOUR DATA STRUCTURE
    # ============================================================

    # Example mappings - UPDATE THESE!
    # If your data is a dict, use: item['field_name']
    # If your data is an object, use: item.field_name

    # 1. NAME (string identifier)
    if hasattr(item, 'name'):
        converted['name'] = item.name
    elif isinstance(item, dict) and 'name' in item:
        converted['name'] = item['name']
    elif isinstance(item, dict) and 'id' in item:
        converted['name'] = str(item['id'])
    else:
        converted['name'] = f'floorplan_{index:05d}'

    # 2. BOUNDARY (N x 4: x, y, direction, isNew)
    # First two points should indicate front door
    if hasattr(item, 'boundary'):
        boundary = item.boundary
    elif isinstance(item, dict) and 'boundary' in item:
        boundary = item['boundary']
    else:
        # If no boundary, you may need to compute from room boundaries
        # This is a placeholder - you'll need to implement this
        raise ValueError("Boundary not found - please implement boundary extraction")

    # Ensure boundary has 4 columns
    if boundary.shape[1] == 2:
        # Add direction (0=right) and isNew (0=corner) columns
        n_points = len(boundary)
        dirs = np.zeros((n_points, 1))
        is_new = np.zeros((n_points, 1))
        boundary = np.hstack([boundary, dirs, is_new])

    converted['boundary'] = boundary.astype(float)

    # 3. ROOM TYPES (N,) - indices into ROOM_CATEGORIES
    if hasattr(item, 'rType'):
        r_type = item.rType
    elif isinstance(item, dict) and 'room_types' in item:
        r_type = item['room_types']
    elif isinstance(item, dict) and 'rType' in item:
        r_type = item['rType']
    else:
        raise ValueError("Room types not found - please implement room type extraction")

    converted['rType'] = np.array(r_type).astype(int)

    # 4. ROOM BOUNDARIES - list of (M x 2) arrays for each room
    if hasattr(item, 'rBoundary'):
        r_boundary = item.rBoundary
    elif isinstance(item, dict) and 'room_boundaries' in item:
        r_boundary = item['room_boundaries']
    elif isinstance(item, dict) and 'rBoundary' in item:
        r_boundary = item['rBoundary']
    else:
        # Placeholder - may need to extract from boxes
        r_boundary = []

    converted['rBoundary'] = [np.array(rb).astype(float) for rb in r_boundary]

    # 5. BOUNDING BOXES - (N x 4)
    # gtBox format: (y0, x0, y1, x1)
    # gtBoxNew format: (x0, y0, x1, y1)
    if hasattr(item, 'gtBox'):
        gt_box = item.gtBox
    elif isinstance(item, dict) and 'boxes' in item:
        boxes = np.array(item['boxes'])
        # Assuming boxes are in (x0, y0, x1, y1) format
        # Convert to gtBox format (y0, x0, y1, x1)
        gt_box = boxes[:, [1, 0, 3, 2]]
    elif isinstance(item, dict) and 'gtBox' in item:
        gt_box = item['gtBox']
    else:
        raise ValueError("Bounding boxes not found - please implement box extraction")

    converted['gtBox'] = np.array(gt_box).astype(float)

    # gtBoxNew: convert to (x0, y0, x1, y1) if needed
    if hasattr(item, 'gtBoxNew'):
        gt_box_new = item.gtBoxNew
    elif isinstance(item, dict) and 'gtBoxNew' in item:
        gt_box_new = item['gtBoxNew']
    else:
        # Convert from gtBox (y0, x0, y1, x1) to (x0, y0, x1, y1)
        gt_box_new = gt_box[:, [1, 0, 3, 2]]

    converted['gtBoxNew'] = np.array(gt_box_new).astype(float)

    # 6. ROOM EDGES (M x 3: u, v, relation_type)
    if hasattr(item, 'rEdge'):
        r_edge = item.rEdge
    elif isinstance(item, dict) and 'edges' in item:
        r_edge = item['edges']
    elif isinstance(item, dict) and 'rEdge' in item:
        r_edge = item['rEdge']
    else:
        # Compute edges from bounding boxes
        r_edge = compute_edge_relations(converted['gtBoxNew'])

    converted['rEdge'] = np.array(r_edge).astype(float)

    # 7. ORDER (N,) - rendering order
    if hasattr(item, 'order'):
        order = item.order
    elif isinstance(item, dict) and 'order' in item:
        order = item['order']
    else:
        # Default: order by room area (larger rooms drawn first)
        boxes = converted['gtBoxNew']
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        order = np.argsort(-areas)  # Descending order

    converted['order'] = np.array(order).astype(float)

    return converted


def convert_resplan_to_mat(pkl_path: str, output_path: str):
    """
    Convert ResPlan pkl file to Graph2plan data.mat format
    """
    print(f"\nLoading ResPlan data from: {pkl_path}")

    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    # Extract floorplan list
    if isinstance(data, dict) and 'data' in data:
        floorplans = data['data']
    elif isinstance(data, list):
        floorplans = data
    else:
        floorplans = [data]

    print(f"Found {len(floorplans)} floorplans")
    print(f"\nConverting to Graph2plan format...")

    converted_data = []
    failed_indices = []

    for i, item in enumerate(floorplans):
        try:
            converted = convert_item(item, i)

            # Create structured array for MATLAB
            dt = np.dtype([
                ('name', 'O'),
                ('boundary', 'O'),
                ('order', 'O'),
                ('rType', 'O'),
                ('rBoundary', 'O'),
                ('gtBox', 'O'),
                ('gtBoxNew', 'O'),
                ('rEdge', 'O')
            ])

            structured_item = np.empty(1, dtype=dt)
            structured_item['name'] = converted['name']
            structured_item['boundary'] = converted['boundary']
            structured_item['order'] = converted['order']
            structured_item['rType'] = converted['rType']
            structured_item['rBoundary'] = converted['rBoundary']
            structured_item['gtBox'] = converted['gtBox']
            structured_item['gtBoxNew'] = converted['gtBoxNew']
            structured_item['rEdge'] = converted['rEdge']

            converted_data.append(structured_item[0])

            if (i + 1) % 1000 == 0:
                print(f"  Processed {i + 1}/{len(floorplans)}")

        except Exception as e:
            print(f"  Failed to convert item {i}: {e}")
            failed_indices.append(i)

    print(f"\nSuccessfully converted: {len(converted_data)}/{len(floorplans)}")
    if failed_indices:
        print(f"Failed indices: {failed_indices[:10]}{'...' if len(failed_indices) > 10 else ''}")

    # Save as .mat file
    print(f"\nSaving to: {output_path}")
    converted_array = np.array(converted_data)
    sio.savemat(output_path, {'data': converted_array},
                format='5', oned_as='row')

    print(f"✓ Conversion complete!")
    print(f"\nNext steps:")
    print(f"1. Update DataPreparation/config.py to point to {output_path}")
    print(f"2. Run the data preparation scripts 1-6 in order")


def main():
    parser = argparse.ArgumentParser(
        description='Convert ResPlan pkl to Graph2plan data.mat format'
    )
    parser.add_argument('input_pkl', type=str,
                       help='Path to ResPlan pkl file')
    parser.add_argument('--inspect', action='store_true',
                       help='Inspect the structure of the pkl file')
    parser.add_argument('--convert', action='store_true',
                       help='Convert the pkl file to data.mat')
    parser.add_argument('--output', type=str, default='data.mat',
                       help='Output path for data.mat (default: data.mat)')
    parser.add_argument('--samples', type=int, default=3,
                       help='Number of samples to inspect (default: 3)')

    args = parser.parse_args()

    if not Path(args.input_pkl).exists():
        print(f"Error: File not found: {args.input_pkl}")
        return

    if args.inspect:
        inspect_pkl_structure(args.input_pkl, args.samples)
    elif args.convert:
        convert_resplan_to_mat(args.input_pkl, args.output)
    else:
        print("Please specify either --inspect or --convert")
        parser.print_help()


if __name__ == '__main__':
    main()

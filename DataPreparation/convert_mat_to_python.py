"""
Convert MATLAB .mat files to Python-friendly formats

Supports conversion to:
- .pkl (pickle): Python native format, preserves all data types
- .json: Universal text format, human-readable
- .npz: Numpy archive, efficient for numerical data
- .csv: Tabular format (for floorplan metadata)

Usage:
    # Inspect the .mat file structure
    python convert_mat_to_python.py --inspect data.mat

    # Convert to pickle (recommended)
    python convert_mat_to_python.py --to-pickle data.mat output.pkl

    # Convert to JSON (human-readable)
    python convert_mat_to_python.py --to-json data.mat output.json

    # Convert to numpy archive
    python convert_mat_to_python.py --to-npz data.mat output.npz

    # Convert to CSV metadata
    python convert_mat_to_python.py --to-csv data.mat output.csv
"""

import pickle
import json
import numpy as np
import scipy.io as sio
import pandas as pd
from pathlib import Path
import argparse
from typing import Any, Dict, List
import warnings


def mat_to_dict(mat_data: Any) -> Dict:
    """
    Recursively convert MATLAB structs to Python dictionaries

    Handles:
    - MATLAB structs → dict
    - MATLAB arrays → numpy arrays
    - MATLAB cell arrays → lists
    """
    if isinstance(mat_data, np.ndarray):
        # Check if it's a structured array (MATLAB struct)
        if mat_data.dtype.names:
            # Convert structured array to list of dicts
            result = []
            for item in mat_data.flat:
                item_dict = {}
                for field in mat_data.dtype.names:
                    field_data = item[field]
                    if isinstance(field_data, np.ndarray) and field_data.dtype == object:
                        # Handle cell arrays
                        if field_data.size == 0:
                            item_dict[field] = []
                        elif field_data.size == 1:
                            item_dict[field] = mat_to_dict(field_data.item())
                        else:
                            item_dict[field] = [mat_to_dict(x) for x in field_data.flat]
                    else:
                        item_dict[field] = mat_to_dict(field_data)
                result.append(item_dict)
            return result if len(result) > 1 else result[0] if result else {}

        # Regular numpy array
        if mat_data.dtype == object:
            # Object array (cell array in MATLAB)
            if mat_data.size == 0:
                return []
            elif mat_data.size == 1:
                return mat_to_dict(mat_data.item())
            else:
                return [mat_to_dict(x) for x in mat_data.flat]
        else:
            # Numeric array - keep as numpy array
            return mat_data

    return mat_data


def inspect_mat_file(mat_path: str, verbose: bool = True):
    """
    Inspect the structure of a .mat file
    """
    print(f"\n{'='*70}")
    print(f"Inspecting: {mat_path}")
    print(f"{'='*70}\n")

    # Load with scipy
    mat_data = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)

    print("Top-level keys:")
    for key in mat_data.keys():
        if not key.startswith('__'):
            value = mat_data[key]
            print(f"  {key}: {type(value)}", end="")
            if isinstance(value, np.ndarray):
                print(f", shape={value.shape}, dtype={value.dtype}")
            else:
                print()

    # Focus on 'data' if it exists (Graph2plan format)
    if 'data' in mat_data:
        data = mat_data['data']
        print(f"\n'data' field details:")
        print(f"  Type: {type(data)}")
        print(f"  Shape: {data.shape if hasattr(data, 'shape') else 'N/A'}")

        if hasattr(data, 'dtype') and hasattr(data.dtype, 'names'):
            print(f"  Fields: {data.dtype.names}")

        # Try to access first item
        if isinstance(data, np.ndarray):
            if data.size > 0:
                print(f"\n  First item structure:")
                first_item = data.flat[0] if data.size > 1 else data.item()

                if hasattr(first_item, '__dict__'):
                    print(f"    Attributes: {list(vars(first_item).keys())}")
                    for attr_name in vars(first_item).keys():
                        attr = getattr(first_item, attr_name)
                        print(f"      {attr_name}: {type(attr)}", end="")
                        if isinstance(attr, np.ndarray):
                            print(f", shape={attr.shape}, dtype={attr.dtype}")
                        elif isinstance(attr, list):
                            print(f", length={len(attr)}")
                        else:
                            print()

                if verbose and data.size <= 3:
                    print(f"\n  Sample items (first {data.size}):")
                    for i, item in enumerate(data.flat):
                        print(f"\n    Item {i}:")
                        if hasattr(item, '__dict__'):
                            for attr_name, attr_value in vars(item).items():
                                print(f"      {attr_name}: {attr_value}")

    print("\n" + "="*70 + "\n")
    return mat_data


def convert_mat_to_pickle(mat_path: str, output_path: str):
    """
    Convert .mat file to pickle format
    """
    print(f"Loading: {mat_path}")
    mat_data = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)

    # Convert to Python-friendly format
    python_data = {}
    for key, value in mat_data.items():
        if not key.startswith('__'):
            python_data[key] = mat_to_dict(value)

    print(f"Saving to: {output_path}")
    with open(output_path, 'wb') as f:
        pickle.dump(python_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"✓ Conversion complete!")
    print(f"  Load with: pickle.load(open('{output_path}', 'rb'))")


def convert_mat_to_json(mat_path: str, output_path: str, indent: int = 2):
    """
    Convert .mat file to JSON format

    Note: JSON doesn't support numpy arrays directly, so they're converted to lists
    """
    print(f"Loading: {mat_path}")
    mat_data = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)

    # Convert to JSON-serializable format
    def make_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(item) for item in obj]
        elif hasattr(obj, '__dict__'):
            return {k: make_serializable(v) for k, v in vars(obj).items()}
        else:
            return obj

    json_data = {}
    for key, value in mat_data.items():
        if not key.startswith('__'):
            json_data[key] = make_serializable(mat_to_dict(value))

    print(f"Saving to: {output_path}")
    with open(output_path, 'w') as f:
        json.dump(json_data, f, indent=indent)

    print(f"✓ Conversion complete!")
    print(f"  Load with: json.load(open('{output_path}', 'r'))")


def convert_mat_to_npz(mat_path: str, output_path: str):
    """
    Convert .mat file to numpy .npz archive

    Best for numerical data; complex nested structures may not convert well
    """
    print(f"Loading: {mat_path}")
    mat_data = sio.loadmat(mat_path, squeeze_me=False, struct_as_record=True)

    # Filter out MATLAB metadata
    arrays = {k: v for k, v in mat_data.items() if not k.startswith('__')}

    print(f"Saving to: {output_path}")
    np.savez_compressed(output_path, **arrays)

    print(f"✓ Conversion complete!")
    print(f"  Load with: np.load('{output_path}', allow_pickle=True)")


def convert_mat_to_csv(mat_path: str, output_path: str):
    """
    Convert .mat file to CSV format (metadata only)

    Extracts key metadata from Graph2plan data structure into a flat table
    """
    print(f"Loading: {mat_path}")
    mat_data = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)

    if 'data' not in mat_data:
        print("Warning: 'data' field not found in .mat file")
        return

    data = mat_data['data']

    # Extract metadata into a flat structure
    records = []
    for i, item in enumerate(data.flat if hasattr(data, 'flat') else [data]):
        record = {'index': i}

        # Extract scalar fields
        if hasattr(item, 'name'):
            record['name'] = item.name

        # Count-based fields
        if hasattr(item, 'rType'):
            r_type = item.rType
            if isinstance(r_type, np.ndarray):
                record['num_rooms'] = len(r_type)
                # Count each room type
                for room_idx in range(18):
                    count = np.sum(r_type == room_idx)
                    if count > 0:
                        record[f'room_type_{room_idx}_count'] = count

        if hasattr(item, 'boundary'):
            boundary = item.boundary
            if isinstance(boundary, np.ndarray):
                record['num_boundary_points'] = len(boundary)

        if hasattr(item, 'rEdge'):
            r_edge = item.rEdge
            if isinstance(r_edge, np.ndarray):
                record['num_edges'] = len(r_edge)

        # Bounding box statistics
        if hasattr(item, 'gtBoxNew'):
            boxes = item.gtBoxNew
            if isinstance(boxes, np.ndarray) and len(boxes) > 0:
                areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                record['total_area'] = float(np.sum(areas))
                record['avg_room_area'] = float(np.mean(areas))
                record['min_room_area'] = float(np.min(areas))
                record['max_room_area'] = float(np.max(areas))

        records.append(record)

    # Create DataFrame
    df = pd.DataFrame(records)

    print(f"Saving to: {output_path}")
    df.to_csv(output_path, index=False)

    print(f"✓ Conversion complete!")
    print(f"  Extracted {len(records)} records with {len(df.columns)} columns")
    print(f"  Load with: pd.read_csv('{output_path}')")


def load_and_explore_pickle(pkl_path: str):
    """
    Load and explore a pickle file
    """
    print(f"Loading: {pkl_path}")
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    print(f"Type: {type(data)}")

    if isinstance(data, dict):
        print(f"Keys: {list(data.keys())}")

        if 'data' in data:
            floorplans = data['data']
            print(f"Number of floorplans: {len(floorplans)}")

            if len(floorplans) > 0:
                print(f"\nFirst floorplan:")
                first = floorplans[0]
                if isinstance(first, dict):
                    for key, value in first.items():
                        print(f"  {key}: {type(value)}", end="")
                        if isinstance(value, np.ndarray):
                            print(f", shape={value.shape}")
                        elif isinstance(value, list):
                            print(f", length={len(value)}")
                        else:
                            print()

    return data


def main():
    parser = argparse.ArgumentParser(
        description='Convert MATLAB .mat files to Python-friendly formats',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Inspect a .mat file
  python convert_mat_to_python.py data.mat --inspect

  # Convert to pickle
  python convert_mat_to_python.py data.mat --to-pickle output.pkl

  # Convert to JSON
  python convert_mat_to_python.py data.mat --to-json output.json

  # Explore a pickle file
  python convert_mat_to_python.py data.pkl --explore-pickle
        """
    )

    parser.add_argument('input_file', help='Input file (.mat or .pkl)')
    parser.add_argument('--inspect', action='store_true',
                       help='Inspect file structure')
    parser.add_argument('--verbose', action='store_true',
                       help='Show detailed information (with --inspect)')
    parser.add_argument('--to-pickle', metavar='OUTPUT',
                       help='Convert to pickle format')
    parser.add_argument('--to-json', metavar='OUTPUT',
                       help='Convert to JSON format')
    parser.add_argument('--to-npz', metavar='OUTPUT',
                       help='Convert to numpy .npz format')
    parser.add_argument('--to-csv', metavar='OUTPUT',
                       help='Convert to CSV metadata format')
    parser.add_argument('--explore-pickle', action='store_true',
                       help='Explore a pickle file')

    args = parser.parse_args()

    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Error: File not found: {args.input_file}")
        return

    if args.inspect:
        if input_path.suffix == '.mat':
            inspect_mat_file(str(input_path), args.verbose)
        else:
            print(f"Inspect mode only works with .mat files")

    elif args.explore_pickle:
        if input_path.suffix == '.pkl':
            load_and_explore_pickle(str(input_path))
        else:
            print(f"Explore mode only works with .pkl files")

    elif args.to_pickle:
        convert_mat_to_pickle(str(input_path), args.to_pickle)

    elif args.to_json:
        convert_mat_to_json(str(input_path), args.to_json)

    elif args.to_npz:
        convert_mat_to_npz(str(input_path), args.to_npz)

    elif args.to_csv:
        convert_mat_to_csv(str(input_path), args.to_csv)

    else:
        print("Please specify an operation (--inspect, --to-pickle, --to-json, --to-npz, or --to-csv)")
        parser.print_help()


if __name__ == '__main__':
    main()

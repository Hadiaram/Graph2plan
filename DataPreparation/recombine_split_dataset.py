"""
Recombine split floor plan files back into a single dataset

This script takes multiple individual floor plan files and combines them
into a single dataset file for convenience.

Usage:
    # Preview what files will be combined
    python recombine_split_dataset.py --input-dir ./floorplans --preview

    # Recombine into a single pickle file
    python recombine_split_dataset.py --input-dir ./floorplans --output combined.pkl

    # Recombine with custom file pattern
    python recombine_split_dataset.py --input-dir ./floorplans --pattern "*.pkl" --output combined.pkl

    # Recombine and save as multiple formats
    python recombine_split_dataset.py --input-dir ./floorplans --output combined.pkl --also-json --also-mat
"""

import pickle
import json
import numpy as np
import scipy.io as sio
from pathlib import Path
import argparse
from typing import List, Dict, Any
from tqdm import tqdm
import warnings


def find_files(input_dir: str, pattern: str = "*", recursive: bool = False) -> List[Path]:
    """
    Find all matching files in directory
    """
    input_path = Path(input_dir)

    if not input_path.exists():
        raise ValueError(f"Directory not found: {input_dir}")

    if recursive:
        files = sorted(input_path.rglob(pattern))
    else:
        files = sorted(input_path.glob(pattern))

    # Filter out directories
    files = [f for f in files if f.is_file()]

    return files


def load_single_file(file_path: Path) -> Any:
    """
    Load a single floor plan file (pkl, json, or npy)
    """
    suffix = file_path.suffix.lower()

    try:
        if suffix == '.pkl':
            with open(file_path, 'rb') as f:
                return pickle.load(f)

        elif suffix == '.json':
            with open(file_path, 'r') as f:
                return json.load(f)

        elif suffix == '.npy':
            return np.load(str(file_path), allow_pickle=True)

        elif suffix == '.npz':
            data = np.load(str(file_path), allow_pickle=True)
            # Convert to dict
            return {key: data[key] for key in data.keys()}

        elif suffix == '.mat':
            return sio.loadmat(str(file_path), squeeze_me=True, struct_as_record=False)

        else:
            warnings.warn(f"Unknown file type: {suffix}, attempting pickle load")
            with open(file_path, 'rb') as f:
                return pickle.load(f)

    except Exception as e:
        warnings.warn(f"Failed to load {file_path}: {e}")
        return None


def preview_files(input_dir: str, pattern: str = "*", recursive: bool = False):
    """
    Preview files that will be combined
    """
    print(f"\n{'='*70}")
    print(f"Preview: Files in {input_dir}")
    print(f"{'='*70}\n")

    files = find_files(input_dir, pattern, recursive)

    print(f"Found {len(files)} files matching pattern '{pattern}'")

    if len(files) == 0:
        print("\nNo files found!")
        return

    # Group by extension
    by_extension = {}
    for f in files:
        ext = f.suffix.lower()
        if ext not in by_extension:
            by_extension[ext] = []
        by_extension[ext].append(f)

    print(f"\nFile types:")
    for ext, file_list in sorted(by_extension.items()):
        print(f"  {ext}: {len(file_list)} files")

    print(f"\nFirst 10 files:")
    for i, f in enumerate(files[:10]):
        print(f"  {i+1}. {f.name} ({f.stat().st_size / 1024:.1f} KB)")

    if len(files) > 10:
        print(f"  ... and {len(files) - 10} more")

    print(f"\nLast 5 files:")
    for i, f in enumerate(files[-5:], start=len(files)-4):
        print(f"  {i}. {f.name} ({f.stat().st_size / 1024:.1f} KB)")

    # Try to load first file to show structure
    print(f"\n{'='*70}")
    print("Sample structure (first file):")
    print(f"{'='*70}\n")

    first_data = load_single_file(files[0])
    if first_data is not None:
        print(f"Type: {type(first_data)}")

        if isinstance(first_data, dict):
            print(f"Keys: {list(first_data.keys())}")
            for key, value in list(first_data.items())[:5]:
                print(f"  {key}: {type(value)}", end="")
                if isinstance(value, np.ndarray):
                    print(f", shape={value.shape}")
                elif isinstance(value, list):
                    print(f", length={len(value)}")
                else:
                    print()

        elif isinstance(first_data, np.ndarray):
            print(f"Shape: {first_data.shape}")
            print(f"Dtype: {first_data.dtype}")

        elif isinstance(first_data, list):
            print(f"Length: {len(first_data)}")
            if len(first_data) > 0:
                print(f"First element type: {type(first_data[0])}")

    print("\n" + "="*70 + "\n")


def recombine_files(input_dir: str, output_path: str, pattern: str = "*",
                   recursive: bool = False, data_key: str = None):
    """
    Recombine split files into a single dataset

    Args:
        input_dir: Directory containing split files
        output_path: Path for combined output file
        pattern: File pattern to match (e.g., "*.pkl", "floorplan_*.json")
        recursive: Search subdirectories recursively
        data_key: If files are dicts, extract this key (e.g., 'data')
    """
    print(f"\n{'='*70}")
    print(f"Recombining files from: {input_dir}")
    print(f"{'='*70}\n")

    files = find_files(input_dir, pattern, recursive)

    print(f"Found {len(files)} files to combine")

    if len(files) == 0:
        print("No files found!")
        return

    combined_data = []
    failed_files = []

    print("\nLoading files...")
    for file_path in tqdm(files):
        data = load_single_file(file_path)

        if data is None:
            failed_files.append(file_path.name)
            continue

        # Extract data if it's wrapped in a dict
        if isinstance(data, dict) and data_key:
            if data_key in data:
                data = data[data_key]
            else:
                warnings.warn(f"Key '{data_key}' not found in {file_path.name}, using whole dict")

        # Handle different data types
        if isinstance(data, list):
            combined_data.extend(data)
        elif isinstance(data, np.ndarray):
            if data.size == 1:
                combined_data.append(data.item())
            else:
                combined_data.extend(data.flat)
        else:
            # Single item
            combined_data.append(data)

    print(f"\n✓ Successfully loaded {len(combined_data)} floor plans")

    if failed_files:
        print(f"✗ Failed to load {len(failed_files)} files:")
        for fname in failed_files[:10]:
            print(f"    {fname}")
        if len(failed_files) > 10:
            print(f"    ... and {len(failed_files) - 10} more")

    # Save combined data
    print(f"\nSaving combined dataset to: {output_path}")

    output_file = Path(output_path)
    output_suffix = output_file.suffix.lower()

    if output_suffix == '.pkl':
        with open(output_path, 'wb') as f:
            pickle.dump({'data': combined_data}, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"✓ Saved as pickle")

    elif output_suffix == '.json':
        # Convert numpy arrays to lists for JSON
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
            else:
                return obj

        json_data = make_serializable(combined_data)
        with open(output_path, 'w') as f:
            json.dump({'data': json_data}, f, indent=2)
        print(f"✓ Saved as JSON")

    elif output_suffix == '.npy':
        np.save(output_path, np.array(combined_data, dtype=object))
        print(f"✓ Saved as numpy array")

    elif output_suffix == '.npz':
        np.savez_compressed(output_path, data=np.array(combined_data, dtype=object))
        print(f"✓ Saved as numpy archive")

    else:
        warnings.warn(f"Unknown output format {output_suffix}, saving as pickle")
        with open(output_path, 'wb') as f:
            pickle.dump({'data': combined_data}, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"✓ Saved as pickle")

    print(f"\n{'='*70}")
    print(f"Summary:")
    print(f"{'='*70}")
    print(f"  Input files: {len(files)}")
    print(f"  Successfully loaded: {len(files) - len(failed_files)}")
    print(f"  Failed: {len(failed_files)}")
    print(f"  Total floor plans: {len(combined_data)}")
    print(f"  Output: {output_path}")
    print(f"{'='*70}\n")

    return combined_data


def save_additional_formats(data: List, base_path: str, save_json: bool = False,
                           save_mat: bool = False, save_npz: bool = False):
    """
    Save the combined data in additional formats
    """
    base = Path(base_path).stem
    parent = Path(base_path).parent

    if save_json:
        json_path = parent / f"{base}.json"
        print(f"\nSaving as JSON: {json_path}")

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
            else:
                return obj

        json_data = make_serializable(data)
        with open(json_path, 'w') as f:
            json.dump({'data': json_data}, f, indent=2)
        print(f"✓ Saved JSON: {json_path}")

    if save_npz:
        npz_path = parent / f"{base}.npz"
        print(f"\nSaving as NPZ: {npz_path}")
        np.savez_compressed(npz_path, data=np.array(data, dtype=object))
        print(f"✓ Saved NPZ: {npz_path}")

    if save_mat:
        mat_path = parent / f"{base}.mat"
        print(f"\nSaving as MAT: {mat_path}")
        # Convert to MATLAB-compatible format
        sio.savemat(mat_path, {'data': data}, format='5', oned_as='row')
        print(f"✓ Saved MAT: {mat_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Recombine split floor plan files into a single dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Preview files
  python recombine_split_dataset.py --input-dir ./split_floorplans --preview

  # Combine all pkl files
  python recombine_split_dataset.py --input-dir ./split_floorplans --pattern "*.pkl" --output combined.pkl

  # Combine with multiple output formats
  python recombine_split_dataset.py --input-dir ./split_floorplans --output resplan.pkl --also-json --also-mat

  # Recursive search in subdirectories
  python recombine_split_dataset.py --input-dir ./data --recursive --output combined.pkl
        """
    )

    parser.add_argument('--input-dir', '-i', required=True,
                       help='Directory containing split files')
    parser.add_argument('--output', '-o',
                       help='Output file path (e.g., combined.pkl)')
    parser.add_argument('--pattern', '-p', default='*',
                       help='File pattern to match (default: "*", matches all files)')
    parser.add_argument('--recursive', '-r', action='store_true',
                       help='Search subdirectories recursively')
    parser.add_argument('--data-key', '-k',
                       help='Extract this key from dict files (e.g., "data")')
    parser.add_argument('--preview', action='store_true',
                       help='Preview files without combining')
    parser.add_argument('--also-json', action='store_true',
                       help='Also save as JSON')
    parser.add_argument('--also-mat', action='store_true',
                       help='Also save as MATLAB .mat')
    parser.add_argument('--also-npz', action='store_true',
                       help='Also save as numpy .npz')

    args = parser.parse_args()

    if args.preview:
        preview_files(args.input_dir, args.pattern, args.recursive)
    elif args.output:
        combined_data = recombine_files(
            args.input_dir,
            args.output,
            args.pattern,
            args.recursive,
            args.data_key
        )

        # Save additional formats if requested
        if combined_data and (args.also_json or args.also_mat or args.also_npz):
            save_additional_formats(
                combined_data,
                args.output,
                args.also_json,
                args.also_mat,
                args.also_npz
            )
    else:
        parser.print_help()
        print("\nError: Must specify either --preview or --output")


if __name__ == '__main__':
    main()

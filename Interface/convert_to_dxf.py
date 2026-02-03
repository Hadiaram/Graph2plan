#!/usr/bin/env python3
"""
Batch convert .mat floor plan files to DXF format.

Usage:
    python convert_to_dxf.py                    # Convert all .mat files in static/
    python convert_to_dxf.py --file 14926.mat   # Convert single file
    python convert_to_dxf.py --scale 0.01       # Use 1 pixel = 1 cm scale
    python convert_to_dxf.py --output ./dxf     # Custom output directory
"""

import argparse
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from Houseweb.dxf_export import save_floorplan_dxf, batch_export_dxf, HAS_EZDXF
    import scipy.io as sio
    import numpy as np
except ImportError as e:
    print(f"ERROR: Failed to import required modules: {e}")
    print("\nPlease install required packages:")
    print("  pip install ezdxf scipy numpy")
    sys.exit(1)


def convert_single_file(mat_file, output_file=None, scale=1.0, wall_thickness=3.0):
    """Convert a single .mat file to DXF."""
    if not HAS_EZDXF:
        print("ERROR: ezdxf not installed. Install with: pip install ezdxf")
        return False

    if not os.path.exists(mat_file):
        print(f"ERROR: File not found: {mat_file}")
        return False

    # Generate output filename if not provided
    if output_file is None:
        output_file = mat_file.replace('.mat', '.dxf')

    try:
        print(f"Loading {mat_file}...")
        data = sio.loadmat(mat_file)
        fp_data = data['data'][0, 0]

        print(f"Exporting to {output_file}...")
        print(f"  Scale: {scale}")
        print(f"  Wall thickness: {wall_thickness}")

        success = save_floorplan_dxf(
            fp_data,
            output_file,
            scale=scale,
            wall_thickness=wall_thickness,
            include_labels=True,
            include_dimensions=False
        )

        if success:
            print(f"✓ Successfully converted {mat_file} -> {output_file}")
            # Print file size
            size_mat = os.path.getsize(mat_file) / 1024
            size_dxf = os.path.getsize(output_file) / 1024
            print(f"  .mat size: {size_mat:.1f} KB")
            print(f"  .dxf size: {size_dxf:.1f} KB")
            return True
        else:
            print(f"✗ Failed to convert {mat_file}")
            return False

    except Exception as e:
        print(f"✗ ERROR converting {mat_file}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Convert Graph2Plan .mat files to DXF format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python convert_to_dxf.py
      Convert all .mat files in static/ directory

  python convert_to_dxf.py --file static/14926.mat
      Convert single file

  python convert_to_dxf.py --scale 0.01 --output ./dxf_export
      Convert all files with 1 pixel = 1 cm scale

  python convert_to_dxf.py --file 14926.mat --scale 0.001
      Convert with millimeter scale (1 px = 1 mm)
        """
    )

    parser.add_argument(
        '--file', '-f',
        help='Single .mat file to convert (default: convert all in static/)'
    )

    parser.add_argument(
        '--input-dir', '-i',
        default='./static',
        help='Input directory containing .mat files (default: ./static)'
    )

    parser.add_argument(
        '--output', '-o',
        help='Output directory or file (default: same as input)'
    )

    parser.add_argument(
        '--scale', '-s',
        type=float,
        default=1.0,
        help='Scale factor: 1.0=pixels, 0.01=cm, 0.001=mm, 0.0254=inches (default: 1.0)'
    )

    parser.add_argument(
        '--wall-thickness', '-w',
        type=float,
        default=3.0,
        help='Wall thickness in drawing units (default: 3.0)'
    )

    parser.add_argument(
        '--list-only', '-l',
        action='store_true',
        help='List .mat files without converting'
    )

    args = parser.parse_args()

    # Check if ezdxf is available
    if not HAS_EZDXF:
        print("=" * 60)
        print("ERROR: ezdxf library not installed")
        print("=" * 60)
        print("\nDXF export requires the ezdxf library.")
        print("Install it with:")
        print("\n  pip install ezdxf")
        print("\nor:")
        print("\n  conda install -c conda-forge ezdxf")
        print("\n" + "=" * 60)
        return 1

    print("=" * 60)
    print("Graph2Plan .mat to DXF Converter")
    print("=" * 60)
    print()

    # Single file conversion
    if args.file:
        output_file = args.output if args.output else args.file.replace('.mat', '.dxf')
        success = convert_single_file(
            args.file,
            output_file,
            scale=args.scale,
            wall_thickness=args.wall_thickness
        )
        return 0 if success else 1

    # Batch conversion
    input_dir = args.input_dir
    output_dir = args.output if args.output else input_dir

    if not os.path.exists(input_dir):
        print(f"ERROR: Input directory not found: {input_dir}")
        return 1

    # Find all .mat files
    mat_files = [f for f in os.listdir(input_dir) if f.endswith('.mat')]

    if not mat_files:
        print(f"No .mat files found in {input_dir}")
        return 1

    print(f"Found {len(mat_files)} .mat files in {input_dir}")
    print()

    if args.list_only:
        print("Files to convert:")
        for mat_file in sorted(mat_files):
            mat_path = os.path.join(input_dir, mat_file)
            size = os.path.getsize(mat_path) / 1024
            print(f"  {mat_file:30s} ({size:7.1f} KB)")
        print()
        print(f"Total: {len(mat_files)} files")
        return 0

    # Create output directory if needed
    if output_dir != input_dir:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Output directory: {output_dir}")
        print()

    # Convert all files
    successful = []
    failed = []

    for i, mat_file in enumerate(mat_files, 1):
        print(f"[{i}/{len(mat_files)}] Processing {mat_file}...")

        mat_path = os.path.join(input_dir, mat_file)
        dxf_file = mat_file.replace('.mat', '.dxf')
        dxf_path = os.path.join(output_dir, dxf_file)

        if convert_single_file(mat_path, dxf_path, args.scale, args.wall_thickness):
            successful.append(mat_file)
        else:
            failed.append(mat_file)

        print()

    # Summary
    print("=" * 60)
    print("Conversion Summary")
    print("=" * 60)
    print(f"Total files: {len(mat_files)}")
    print(f"Successful:  {len(successful)} ({len(successful)/len(mat_files)*100:.1f}%)")
    print(f"Failed:      {len(failed)} ({len(failed)/len(mat_files)*100:.1f}%)")
    print()

    if failed:
        print("Failed files:")
        for mat_file in failed:
            print(f"  - {mat_file}")
        print()

    if successful:
        # Calculate total sizes
        total_mat_size = sum(os.path.getsize(os.path.join(input_dir, f)) for f in successful) / 1024
        total_dxf_size = sum(os.path.getsize(os.path.join(output_dir, f.replace('.mat', '.dxf'))) for f in successful) / 1024

        print(f"Total .mat size: {total_mat_size:.1f} KB")
        print(f"Total .dxf size: {total_dxf_size:.1f} KB")
        print()

    print(f"Output directory: {os.path.abspath(output_dir)}")

    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())

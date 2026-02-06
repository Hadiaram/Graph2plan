"""
DXF Export Module for Graph2Plan Floor Plans

Converts floor plan data to AutoCAD DXF format for use in CAD software.
Supports walls, rooms, windows, doors, and room labels.

Usage:
    from Houseweb.dxf_export import save_floorplan_dxf
    save_floorplan_dxf(fp_data, "output.dxf")
"""

import numpy as np
import warnings

# Try to import ezdxf (install with: pip install ezdxf)
HAS_EZDXF = False
try:
    import ezdxf #type: ignore
    from ezdxf.enums import TextEntityAlignment #type: ignore
    HAS_EZDXF = True
except ImportError:
    warnings.warn(
        "ezdxf not installed. DXF export disabled. "
        "Install with: pip install ezdxf",
        UserWarning
    )


def _get_field(fp_data, field_name):
    """
    Safely get a field from fp_data, handling both structured arrays and objects.

    Args:
        fp_data: Floor plan data (structured array or object)
        field_name: Name of field to retrieve

    Returns:
        Field data if found, None otherwise
    """
    # Method 1: Structured array access (from .mat files)
    if hasattr(fp_data, 'dtype') and hasattr(fp_data.dtype, 'names') and fp_data.dtype.names:
        if field_name in fp_data.dtype.names:
            return fp_data[field_name]
    # Method 2: Direct attribute access
    elif hasattr(fp_data, field_name):
        return getattr(fp_data, field_name)

    return None


def save_floorplan_dxf(fp_data, filepath, scale=1.0, wall_thickness=3.0,
                       include_labels=True, include_dimensions=False):
    """
    Save floor plan data as DXF file.

    Args:
        fp_data: Floor plan data object with attributes:
            - boundary: Exterior boundary coordinates (N x 2 or N x 4)
            - rBoundary: List of room boundary polygons (optional, preferred)
            - newBox: Refined room boxes (N x 4: x1, y1, x2, y2)
            - rType: Room type indices (N,)
            - windows: Window data (N x 6: indx, x, y, w, h, r)
        filepath: Output DXF file path
        scale: Scale factor (default 1.0 = pixels, use 0.01 for cm)
        wall_thickness: Wall thickness in drawing units (default 3.0)
        include_labels: Add room type labels (default True)
        include_dimensions: Add dimension lines (default False)

    Returns:
        bool: True if successful, False if ezdxf not available
    """
    if not HAS_EZDXF:
        print("ERROR: ezdxf not installed. Cannot export to DXF.")
        print("Install with: pip install ezdxf")
        return False

    try:
        # Create new DXF document (AutoCAD 2018 format)
        doc = ezdxf.new('R2018', setup=True)
        msp = doc.modelspace()

        # Define layers with colors
        doc.layers.add('EXTERIOR_WALL', color=7)      # White/gray
        doc.layers.add('INTERIOR_WALL', color=8)      # Dark gray
        doc.layers.add('ROOMS', color=3)              # Green
        doc.layers.add('WINDOWS', color=4)            # Cyan
        doc.layers.add('DOORS', color=6)              # Magenta
        doc.layers.add('LABELS', color=2)             # Yellow
        doc.layers.add('DIMENSIONS', color=1)         # Red

        # 1. Draw exterior boundary (main wall outline)
        boundary = _get_field(fp_data, 'boundary')
        if boundary is not None:
            boundary = np.array(boundary)
            if len(boundary) > 0:
                # Extract x, y coordinates (handle both N x 2 and N x 4 formats)
                if boundary.shape[1] >= 2:
                    exterior_points = [(float(x) * scale, float(y) * scale)
                                      for x, y in boundary[:, :2]]

                    # Close the polygon if not already closed
                    if exterior_points[0] != exterior_points[-1]:
                        exterior_points.append(exterior_points[0])

                    # Draw exterior wall as thick polyline
                    msp.add_lwpolyline(
                        exterior_points,
                        dxfattribs={
                            'layer': 'EXTERIOR_WALL',
                            'const_width': wall_thickness * scale
                        }
                    )
                    print(f"[DXF] Drew exterior boundary with {len(exterior_points)} points")

        # 2. Draw room boundaries (preferred over boxes for accuracy)
        rBoundary = _get_field(fp_data, 'rBoundary')
        if rBoundary is not None and len(rBoundary) > 0:
            room_labels = _get_room_labels(fp_data)

            for i, rb in enumerate(rBoundary):
                if isinstance(rb, np.ndarray) and len(rb) > 0:
                    room_points = [(float(x) * scale, float(y) * scale)
                                  for x, y in rb]

                    # Close the polygon
                    if room_points[0] != room_points[-1]:
                        room_points.append(room_points[0])

                    # Draw room boundary
                    msp.add_lwpolyline(
                        room_points,
                        dxfattribs={
                            'layer': 'INTERIOR_WALL',
                            'const_width': wall_thickness * scale * 0.5  # Thinner than exterior
                        }
                    )

                    # Add room label at centroid
                    if include_labels and i < len(room_labels):
                        centroid = _calculate_centroid(rb)
                        msp.add_text(
                            room_labels[i],
                            dxfattribs={
                                'layer': 'LABELS',
                                'height': 5.0 * scale,
                            }
                        ).set_placement(
                            (centroid[0] * scale, centroid[1] * scale),
                            align=TextEntityAlignment.MIDDLE_CENTER
                        )

            print(f"[DXF] Drew {len(rBoundary)} room boundaries with labels")

        # 3. Fallback: Draw room boxes if rBoundary not available
        else:
            newBox = _get_field(fp_data, 'newBox')
            if newBox is not None:
                boxes = np.array(newBox)
                room_labels = _get_room_labels(fp_data)

                for i, box in enumerate(boxes):
                    if len(box) >= 4:
                        x1, y1, x2, y2 = box[:4]

                        # Draw room as rectangle
                        room_points = [
                            (float(x1) * scale, float(y1) * scale),
                            (float(x2) * scale, float(y1) * scale),
                            (float(x2) * scale, float(y2) * scale),
                            (float(x1) * scale, float(y2) * scale),
                            (float(x1) * scale, float(y1) * scale)  # Close
                        ]

                        msp.add_lwpolyline(
                            room_points,
                            dxfattribs={
                                'layer': 'ROOMS',
                                'const_width': wall_thickness * scale * 0.5
                            }
                        )

                        # Add room label at center
                        if include_labels and i < len(room_labels):
                            center_x = (float(x1) + float(x2)) / 2 * scale
                            center_y = (float(y1) + float(y2)) / 2 * scale
                            msp.add_text(
                                room_labels[i],
                                dxfattribs={
                                    'layer': 'LABELS',
                                    'height': 5.0 * scale,
                                }
                            ).set_placement(
                                (center_x, center_y),
                                align=TextEntityAlignment.MIDDLE_CENTER
                            )

                print(f"[DXF] Drew {len(boxes)} room boxes (fallback)")

        # 4. Draw windows
        windows = _get_field(fp_data, 'windows')
        if windows is not None:
            windows = np.array(windows)
            window_count = 0

            for window in windows:
                if len(window) >= 5:
                    indx, x, y, w, h = window[:5]

                    # Draw horizontal window
                    if w != 0:
                        x1 = (float(x) + 2) * scale
                        y1 = (float(y) - 2) * scale
                        x2 = (float(x) + float(w) - 2) * scale
                        y2 = y1

                        msp.add_line(
                            (x1, y1), (x2, y2),
                            dxfattribs={
                                'layer': 'WINDOWS',
                                'lineweight': 25  # Thicker line for windows
                            }
                        )
                        window_count += 1

                    # Draw vertical window
                    if h != 0:
                        x1 = (float(x) - 2) * scale
                        y1 = float(y) * scale
                        x2 = x1
                        y2 = (float(y) + float(h)) * scale

                        msp.add_line(
                            (x1, y1), (x2, y2),
                            dxfattribs={
                                'layer': 'WINDOWS',
                                'lineweight': 25
                            }
                        )
                        window_count += 1

            print(f"[DXF] Drew {window_count} windows")

        # 5. Draw door (first two boundary points)
        boundary = _get_field(fp_data, 'boundary')
        if boundary is not None:
            boundary = np.array(boundary)
            if len(boundary) >= 2:
                door_p1 = (float(boundary[0][0]) * scale, float(boundary[0][1]) * scale)
                door_p2 = (float(boundary[1][0]) * scale, float(boundary[1][1]) * scale)

                # Draw door as special line
                msp.add_line(
                    door_p1, door_p2,
                    dxfattribs={
                        'layer': 'DOORS',
                        'lineweight': 35  # Extra thick for doors
                    }
                )

                # Add door arc (swing indicator)
                door_angle = np.arctan2(door_p2[1] - door_p1[1],
                                       door_p2[0] - door_p1[0])
                door_angle_deg = np.degrees(door_angle)

                msp.add_arc(
                    center=door_p1,
                    radius=np.linalg.norm([door_p2[0] - door_p1[0],
                                          door_p2[1] - door_p1[1]]),
                    start_angle=door_angle_deg,
                    end_angle=door_angle_deg + 90,
                    dxfattribs={'layer': 'DOORS'}
                )

                print(f"[DXF] Drew door at entry")

        # 6. Add optional dimensions
        if include_dimensions:
            boundary = _get_field(fp_data, 'boundary')
            if boundary is not None:
                boundary = np.array(boundary)
                if len(boundary) > 0:
                    # Calculate overall dimensions
                    x_coords = boundary[:, 0]
                    y_coords = boundary[:, 1]
                    width = (np.max(x_coords) - np.min(x_coords)) * scale
                    height = (np.max(y_coords) - np.min(y_coords)) * scale

                    # Add dimension text
                    dim_text = f"Width: {width:.2f} x Height: {height:.2f}"
                    msp.add_text(
                        dim_text,
                        dxfattribs={
                            'layer': 'DIMENSIONS',
                            'height': 3.0 * scale,
                        }
                    ).set_placement(
                        (float(np.min(x_coords)) * scale,
                         float(np.max(y_coords) + 10) * scale),
                        align=TextEntityAlignment.BOTTOM_LEFT
                    )

        # Save the DXF file
        doc.saveas(filepath)
        print(f"[DXF] Successfully saved floor plan to {filepath}")
        print(f"[DXF] Scale: {scale}, Wall thickness: {wall_thickness * scale}")
        return True

    except Exception as e:
        print(f"[DXF] ERROR saving floor plan: {e}")
        import traceback
        traceback.print_exc()
        return False


def _get_room_labels(fp_data):
    """Extract room labels from floor plan data."""
    room_labels = []

    # Check if rType exists in structured array (scipy.io.loadmat format)
    has_rType = False
    rType_data = None

    # Method 1: Structured array access (from .mat files)
    if hasattr(fp_data, 'dtype') and hasattr(fp_data.dtype, 'names') and fp_data.dtype.names:
        if 'rType' in fp_data.dtype.names:
            has_rType = True
            rType_data = fp_data['rType']
            print(f"[DXF Debug] rType found via dtype.names, shape: {rType_data.shape}")
    # Method 2: Direct attribute access (for regular objects)
    elif hasattr(fp_data, 'rType') and fp_data.rType is not None:
        has_rType = True
        rType_data = fp_data.rType
        print(f"[DXF Debug] rType found via attribute, shape: {rType_data.shape}")

    if has_rType and rType_data is not None:
        try:
            # Import room label mapping
            import model.utils as mdul

            # Flatten to handle both 1D and 2D arrays
            rType = np.array(rType_data).flatten()
            print(f"[DXF Debug] rType flattened: {rType}, length: {len(rType)}")

            for room_type in rType:
                room_idx = int(room_type)
                if room_idx < len(mdul.room_label):
                    room_labels.append(mdul.room_label[room_idx][1])
                else:
                    # Handle unknown room types
                    print(f"[DXF] Warning: Unknown room type index {room_idx}, using 'Room{room_idx}'")
                    room_labels.append(f"Room{room_idx}")

            print(f"[DXF Debug] Room labels extracted: {room_labels}")
        except Exception as e:
            print(f"[DXF] Warning: Could not load room labels: {e}")
            import traceback
            traceback.print_exc()
            room_labels = [f"Room{i}" for i in range(len(rType_data.flatten()))]
    else:
        print(f"[DXF Debug] rType NOT found in fp_data!")

    return room_labels


def _calculate_centroid(polygon):
    """Calculate centroid of a polygon."""
    polygon = np.array(polygon)
    if len(polygon) == 0:
        return (0, 0)

    x_coords = polygon[:, 0]
    y_coords = polygon[:, 1]

    return (float(np.mean(x_coords)), float(np.mean(y_coords)))


def batch_export_dxf(mat_directory, output_directory, scale=1.0):
    """
    Batch export all .mat files in a directory to DXF.

    Args:
        mat_directory: Directory containing .mat files
        output_directory: Directory to save DXF files
        scale: Scale factor for output

    Returns:
        List of successfully exported files
    """
    import os
    import scipy.io as sio

    if not HAS_EZDXF:
        print("ERROR: ezdxf not installed. Cannot batch export.")
        return []

    os.makedirs(output_directory, exist_ok=True)

    mat_files = [f for f in os.listdir(mat_directory) if f.endswith('.mat')]
    successful = []

    print(f"[DXF Batch] Found {len(mat_files)} .mat files to convert")

    for mat_file in mat_files:
        try:
            mat_path = os.path.join(mat_directory, mat_file)
            dxf_file = mat_file.replace('.mat', '.dxf')
            dxf_path = os.path.join(output_directory, dxf_file)

            # Load .mat file
            data = sio.loadmat(mat_path)
            fp_data = data['data'][0, 0]

            # Export to DXF
            if save_floorplan_dxf(fp_data, dxf_path, scale=scale):
                successful.append(dxf_file)
                print(f"[DXF Batch] ✓ Converted {mat_file} -> {dxf_file}")
            else:
                print(f"[DXF Batch] ✗ Failed to convert {mat_file}")

        except Exception as e:
            print(f"[DXF Batch] ✗ Error converting {mat_file}: {e}")

    print(f"[DXF Batch] Completed: {len(successful)}/{len(mat_files)} successful")
    return successful


if __name__ == "__main__":
    # Test example
    print("DXF Export Module")
    print(f"ezdxf available: {HAS_EZDXF}")

    if HAS_EZDXF:
        print(f"ezdxf version: {ezdxf.__version__}")
        print("\nUsage:")
        print("  from Houseweb.dxf_export import save_floorplan_dxf")
        print("  save_floorplan_dxf(fp_data, 'output.dxf')")

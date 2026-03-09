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

# Try to import shapely for polygon clipping
HAS_SHAPELY = False
try:
    from shapely.geometry import Polygon, MultiPolygon #type: ignore
    from shapely.ops import unary_union #type: ignore
    HAS_SHAPELY = True
except ImportError:
    warnings.warn(
        "shapely not installed. DXF export will show overlapping rooms. "
        "Install with: pip install shapely for proper clipping",
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
        raise RuntimeError("ezdxf not installed. Install with: pip install ezdxf")

    print(f"[DXF] save_floorplan_dxf called, fp_data type={type(fp_data).__name__}", flush=True)

    # No outer try/except — let exceptions propagate to Export_DXF for proper error reporting
    if True:
        # Create new DXF document (AutoCAD 2018 format)
        doc = ezdxf.new('R2018', setup=True)
        msp = doc.modelspace()

        # Define layers with colors
        doc.layers.new('EXTERIOR_WALL', dxfattribs={'color': 7})   # White/gray
        doc.layers.new('INTERIOR_WALL', dxfattribs={'color': 8})   # Dark gray
        doc.layers.new('ROOMS', dxfattribs={'color': 3})           # Green
        doc.layers.new('WINDOWS', dxfattribs={'color': 4})         # Cyan
        doc.layers.new('DOORS', dxfattribs={'color': 6})           # Magenta
        doc.layers.new('LABELS', dxfattribs={'color': 2})          # Yellow
        doc.layers.new('DIMENSIONS', dxfattribs={'color': 1})      # Red

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

            # Clip each room to the exterior boundary only
            boundary_array = _get_field(fp_data, 'boundary')
            if boundary_array is not None:
                clipped_rBoundary = _clip_room_polygons(rBoundary, np.array(boundary_array), None)
            else:
                clipped_rBoundary = list(rBoundary)

            rooms_drawn = 0
            for i, rb in enumerate(clipped_rBoundary):
                if rb is None:
                    continue
                if not isinstance(rb, np.ndarray) or len(rb) == 0:
                    continue

                room_points = [(float(x) * scale, float(y) * scale) for x, y in rb]

                # Close the polygon
                if room_points[0] != room_points[-1]:
                    room_points.append(room_points[0])

                msp.add_lwpolyline(
                    room_points,
                    dxfattribs={
                        'layer': 'INTERIOR_WALL',
                        'const_width': wall_thickness * scale * 0.5
                    }
                )

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
                rooms_drawn += 1

            print(f"[DXF] Drew {rooms_drawn}/{len(rBoundary)} room boundaries")

        # 3. Fallback: Draw room boxes if rBoundary not available
        else:
            newBox = _get_field(fp_data, 'newBox')
            if newBox is not None:
                boxes = np.array(newBox)
                room_labels = _get_room_labels(fp_data)

                # Get rendering order
                order = _get_field(fp_data, 'order')
                if order is not None and len(order) > 0:
                    try:
                        order_indices = [int(idx[0]) - 1 for idx in order if len(idx) > 0]
                        print(f"[DXF] Using rendering order: {order_indices}")
                    except:
                        order_indices = list(range(len(boxes)))
                        print(f"[DXF] Warning: Could not parse order, using default sequence")
                else:
                    order_indices = list(range(len(boxes)))
                    print(f"[DXF] No rendering order found, using default sequence")

                # Convert boxes to polygons for clipping
                box_polygons = []
                for box in boxes:
                    if len(box) >= 4:
                        x1, y1, x2, y2 = box[:4]
                        box_poly = np.array([
                            [x1, y1],
                            [x2, y1],
                            [x2, y2],
                            [x1, y2],
                            [x1, y1]
                        ])
                        box_polygons.append(box_poly)
                    else:
                        box_polygons.append(None)

                # Clip boxes
                boundary_array = _get_field(fp_data, 'boundary')
                if boundary_array is not None:
                    clipped_boxes = _clip_room_polygons(box_polygons, np.array(boundary_array), order)
                else:
                    clipped_boxes = box_polygons

                # Draw clipped boxes in rendering order
                rooms_drawn = 0
                for i in order_indices:
                    if i < len(clipped_boxes) and clipped_boxes[i] is not None:
                        clipped_poly = clipped_boxes[i]
                        if isinstance(clipped_poly, np.ndarray) and len(clipped_poly) > 0:
                            room_points = [(float(x) * scale, float(y) * scale)
                                          for x, y in clipped_poly]

                            # Close if needed
                            if room_points[0] != room_points[-1]:
                                room_points.append(room_points[0])

                            msp.add_lwpolyline(
                                room_points,
                                dxfattribs={
                                    'layer': 'ROOMS',
                                    'const_width': wall_thickness * scale * 0.5
                                }
                            )

                            # Add room label at centroid of clipped polygon
                            if include_labels and i < len(room_labels):
                                centroid = _calculate_centroid(clipped_poly)
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
                            rooms_drawn += 1

                print(f"[DXF] Drew {rooms_drawn}/{len(boxes)} room boxes (clipped, no overlaps)")

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
        print(f"[DXF] Section 5: drawing front door...", flush=True)
        boundary = _get_field(fp_data, 'boundary')
        if boundary is not None:
            boundary = np.array(boundary)
            if len(boundary) >= 2:
                try:
                    door_p1 = (float(boundary[0][0]) * scale, float(boundary[0][1]) * scale)
                    door_p2 = (float(boundary[1][0]) * scale, float(boundary[1][1]) * scale)

                    msp.add_line(door_p1, door_p2,
                                 dxfattribs={'layer': 'DOORS', 'lineweight': 35})

                    door_radius = np.linalg.norm([door_p2[0] - door_p1[0],
                                                  door_p2[1] - door_p1[1]])
                    if door_radius > 0:
                        door_angle_deg = np.degrees(
                            np.arctan2(door_p2[1] - door_p1[1], door_p2[0] - door_p1[0]))
                        msp.add_arc(center=door_p1, radius=door_radius,
                                    start_angle=door_angle_deg,
                                    end_angle=door_angle_deg + 90,
                                    dxfattribs={'layer': 'DOORS'})
                    print(f"[DXF] Drew door at entry", flush=True)
                except Exception as _door_err:
                    print(f"[DXF] Warning: front door drawing failed: {_door_err}", flush=True)

        # 5b. Interior doors — one per graph edge.
        #
        # For each edge (u, v) in rEdge, score all 4 candidate face-to-face
        # walls by overlap / (gap + 1).  The highest score wins regardless of
        # whether the boxes touch or one sits inside the other (e.g. LR after
        # fill_living_room).  No Shapely needed.
        rEdge_data  = _get_field(fp_data, 'rEdge')
        newBox_data = _get_field(fp_data, 'newBox')
        rType_data  = _get_field(fp_data, 'rType')

        if rEdge_data is not None and newBox_data is not None:
            _edges = np.array(rEdge_data)
            _boxes = [np.array(b, dtype=float).flatten()[:4] for b in newBox_data]
            _types = np.array(rType_data).flatten().astype(int) if rType_data is not None else None
            _K     = len(_boxes)
            _EXCL  = {12, 13, 14, 15}
            _DOOR  = 30.0   # max door width in pixels
            _seen  = set()
            _found = 0
            _skipped = 0
            _failed  = 0

            print(f"[DXF] Interior doors: processing {len(_edges)} graph edges, "
                  f"{_K} boxes", flush=True)

            for _e in _edges:
                _u, _v = int(_e[0]), int(_e[1])
                if _u >= _K or _v >= _K or _u == _v:
                    continue
                _pair = (min(_u, _v), max(_u, _v))
                if _pair in _seen:
                    continue
                _seen.add(_pair)

                if _types is not None and (_types[_u] in _EXCL or _types[_v] in _EXCL):
                    _skipped += 1
                    continue

                _bu = _boxes[_u]   # x0, y0, x1, y1
                _bv = _boxes[_v]

                # Detect if one box is entirely inside the other.
                # This happens after fill_living_room expands LR to the full
                # bounding box — the normal gap-based candidates break because
                # LR's faces end up at the exterior boundary.
                _EPS_IN  = 2.0
                _u_in_v  = (_bu[0] >= _bv[0] - _EPS_IN and _bu[2] <= _bv[2] + _EPS_IN and
                             _bu[1] >= _bv[1] - _EPS_IN and _bu[3] <= _bv[3] + _EPS_IN)
                _v_in_u  = (_bv[0] >= _bu[0] - _EPS_IN and _bv[2] <= _bu[2] + _EPS_IN and
                             _bv[1] >= _bu[1] - _EPS_IN and _bv[3] <= _bu[3] + _EPS_IN)

                _best       = None
                _best_score = -1.0

                if _u_in_v or _v_in_u:
                    # One box contains the other.  Use the *inner* box's four
                    # faces as candidates and keep only faces that:
                    #   (a) do NOT touch the outer box's boundary (exterior wall)
                    #   (b) have no other room directly adjacent to them
                    # The surviving face opens onto the visible interior area.
                    _inner     = _bu if _u_in_v else _bv
                    _inner_idx = _u  if _u_in_v else _v
                    _outer     = _bv if _u_in_v else _bu
                    _outer_idx = _v  if _u_in_v else _u
                    _EPS_ADJ   = 3.0

                    for _fkind, _fcoord, _fs0, _fs1 in [
                        ('vert',  _inner[0], _inner[1], _inner[3]),  # left
                        ('vert',  _inner[2], _inner[1], _inner[3]),  # right
                        ('horiz', _inner[1], _inner[0], _inner[2]),  # top
                        ('horiz', _inner[3], _inner[0], _inner[2]),  # bottom
                    ]:
                        # (a) skip faces flush with the outer (= exterior) boundary
                        if _fkind == 'vert':
                            if (abs(_fcoord - _outer[0]) < _EPS_ADJ or
                                    abs(_fcoord - _outer[2]) < _EPS_ADJ):
                                continue
                        else:
                            if (abs(_fcoord - _outer[1]) < _EPS_ADJ or
                                    abs(_fcoord - _outer[3]) < _EPS_ADJ):
                                continue

                        # (b) collect blocked intervals, find the longest unblocked
                        #     portion — a partial blocker doesn't disqualify the face
                        _blk_ivs = []
                        for _k in range(_K):
                            if _k == _inner_idx or _k == _outer_idx:
                                continue
                            if _types is not None and _types[_k] in _EXCL:
                                continue
                            _bk = _boxes[_k]
                            if _fkind == 'vert':
                                if (abs(_bk[0] - _fcoord) < _EPS_ADJ or
                                        abs(_bk[2] - _fcoord) < _EPS_ADJ):
                                    _s = max(_bk[1], _fs0)
                                    _e = min(_bk[3], _fs1)
                                    if _e > _s:
                                        _blk_ivs.append((_s, _e))
                            else:
                                if (abs(_bk[1] - _fcoord) < _EPS_ADJ or
                                        abs(_bk[3] - _fcoord) < _EPS_ADJ):
                                    _s = max(_bk[0], _fs0)
                                    _e = min(_bk[2], _fs1)
                                    if _e > _s:
                                        _blk_ivs.append((_s, _e))

                        _ub_ivs  = _subtract_intervals((_fs0, _fs1), _blk_ivs)
                        _best_ub = max(_ub_ivs, key=lambda _iv: _iv[1] - _iv[0]) if _ub_ivs else None

                        if _best_ub is None or (_best_ub[1] - _best_ub[0]) < 10:
                            continue

                        _ov0_face, _ov1_face = _best_ub
                        _score = _ov1_face - _ov0_face
                        if _score > _best_score:
                            _best_score = _score
                            _best = (_fkind, _fcoord, _ov0_face, _ov1_face)

                else:
                    # Normal case: four candidate face-to-face walls, scored by
                    # overlap / (gap + 1).
                    for _kind, _coord, _s0u, _s1u, _s0v, _s1v, _gap in [
                        # u's right  meets v's left
                        ('vert',  _bv[0], _bu[1], _bu[3], _bv[1], _bv[3], abs(_bu[2] - _bv[0])),
                        # v's right  meets u's left
                        ('vert',  _bv[2], _bu[1], _bu[3], _bv[1], _bv[3], abs(_bv[2] - _bu[0])),
                        # u's bottom meets v's top
                        ('horiz', _bv[1], _bu[0], _bu[2], _bv[0], _bv[2], abs(_bu[3] - _bv[1])),
                        # v's bottom meets u's top
                        ('horiz', _bv[3], _bu[0], _bu[2], _bv[0], _bv[2], abs(_bv[3] - _bu[1])),
                    ]:
                        _ov = max(0.0, min(_s1u, _s1v) - max(_s0u, _s0v))
                        if _ov > 0:
                            _score = _ov / (_gap + 1.0)
                            if _score > _best_score:
                                _best_score = _score
                                _best = (_kind, _coord,
                                         max(_s0u, _s0v), min(_s1u, _s1v))

                if _best is None:
                    _failed += 1
                    continue

                _kind, _coord, _ov0, _ov1 = _best
                _dw  = min(_DOOR, (_ov1 - _ov0) * 0.7)
                _mid = (_ov0 + _ov1) / 2.0

                if _kind == 'vert':
                    _p1  = (_coord * scale, (_mid - _dw/2) * scale)
                    _p2  = (_coord * scale, (_mid + _dw/2) * scale)
                    _rad = _dw * scale
                    msp.add_line(_p1, _p2,
                                 dxfattribs={"layer": "DOORS", "lineweight": 35})
                    msp.add_line(_p1, (_p1[0] + _rad, _p1[1]),
                                 dxfattribs={"layer": "DOORS", "lineweight": 18})
                    msp.add_arc(center=_p1, radius=_rad,
                                start_angle=0.0, end_angle=90.0,
                                dxfattribs={"layer": "DOORS"})
                else:
                    _p1  = ((_mid - _dw/2) * scale, _coord * scale)
                    _p2  = ((_mid + _dw/2) * scale, _coord * scale)
                    _rad = _dw * scale
                    msp.add_line(_p1, _p2,
                                 dxfattribs={"layer": "DOORS", "lineweight": 35})
                    msp.add_line(_p1, (_p1[0], _p1[1] + _rad),
                                 dxfattribs={"layer": "DOORS", "lineweight": 18})
                    msp.add_arc(center=_p1, radius=_rad,
                                start_angle=0.0, end_angle=90.0,
                                dxfattribs={"layer": "DOORS"})
                _found += 1

            print(f"[DXF] Interior doors: {_found} drawn, {_skipped} skipped "
                  f"(wall/ext type), {_failed} no overlap", flush=True)


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


def build_floorplan_dxf_bytes(fp_data, scale=1.0, wall_thickness=3.0,
                               include_labels=True, include_dimensions=False):
    """
    Build a DXF document for fp_data and return its content as bytes.
    Raises an exception on failure (no try/except — let caller handle it).
    """
    import io
    import tempfile, os
    # Reuse save_floorplan_dxf by writing to a temp file, then read back bytes
    with tempfile.NamedTemporaryFile(suffix='.dxf', delete=False) as tmp:
        tmp_path = tmp.name
    try:
        save_floorplan_dxf(fp_data, tmp_path, scale=scale,
                           wall_thickness=wall_thickness,
                           include_labels=include_labels,
                           include_dimensions=include_dimensions)
        with open(tmp_path, 'rb') as f:
            return f.read()
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


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
        shape_info = getattr(rType_data, 'shape', type(rType_data).__name__)
        print(f"[DXF Debug] rType found via attribute, shape/type: {shape_info}")

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


def _subtract_intervals(total, blocks):
    """Return list of (start, end) intervals remaining after removing blocks from total."""
    if not blocks:
        return [total]
    merged = sorted(blocks)
    # merge overlapping blocks
    fused = [list(merged[0])]
    for s, e in merged[1:]:
        if s <= fused[-1][1]:
            fused[-1][1] = max(fused[-1][1], e)
        else:
            fused.append([s, e])
    result = []
    cur = total[0]
    for s, e in fused:
        if s > cur:
            result.append((cur, min(s, total[1])))
        cur = max(cur, e)
    if cur < total[1]:
        result.append((cur, total[1]))
    return result


def _calculate_centroid(polygon):
    """Calculate centroid of a polygon."""
    polygon = np.array(polygon)
    if len(polygon) == 0:
        return (0, 0)

    x_coords = polygon[:, 0]
    y_coords = polygon[:, 1]

    return (float(np.mean(x_coords)), float(np.mean(y_coords)))


def _clip_room_polygons(rBoundary, boundary, order):
    """
    Clip room polygons to the exterior boundary only.

    For DXF export each room is drawn at its actual shape — CAD software
    handles overlapping polylines correctly.  We only remove the parts of
    each room that fall outside the exterior boundary polygon.

    Args:
        rBoundary: List of room boundary polygons
        boundary: Exterior boundary polygon
        order: Rendering order (1-indexed) — kept for API compatibility

    Returns:
        List of clipped polygons (same length as rBoundary, None if a room
        ends up completely outside the boundary)
    """
    if not HAS_SHAPELY:
        print("[DXF] Warning: Shapely not available, skipping boundary clip")
        return rBoundary

    try:
        boundary_poly = Polygon(boundary[:, :2])
        clipped_rooms = []

        for rb in rBoundary:
            if not isinstance(rb, np.ndarray) or len(rb) == 0:
                clipped_rooms.append(None)
                continue

            try:
                room_poly = Polygon(rb[:, :2] if rb.ndim == 2 and rb.shape[1] >= 2 else rb)

                if not room_poly.is_valid or not boundary_poly.is_valid:
                    # Keep original polygon if validity check fails
                    clipped_rooms.append(rb)
                    continue

                clipped = room_poly.intersection(boundary_poly)

                if clipped.is_empty:
                    clipped_rooms.append(None)
                elif isinstance(clipped, Polygon):
                    clipped_rooms.append(np.array(list(clipped.exterior.coords)))
                elif isinstance(clipped, MultiPolygon):
                    largest = max(clipped.geoms, key=lambda p: p.area)
                    clipped_rooms.append(np.array(list(largest.exterior.coords)))
                else:
                    # GeometryCollection or other — fall back to original
                    clipped_rooms.append(rb)

            except Exception:
                clipped_rooms.append(rb)

        drawn = sum(1 for r in clipped_rooms if r is not None)
        print(f"[DXF] Boundary-clipped {drawn}/{len(rBoundary)} rooms")
        return clipped_rooms

    except Exception as e:
        print(f"[DXF] Warning: Failed to clip polygons: {e}")
        import traceback
        traceback.print_exc()
        return rBoundary


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

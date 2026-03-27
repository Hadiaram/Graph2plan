"""
DXF Import Module for Graph2Plan

Parses a DXF boundary file and converts it to the internal boundary format
(N x 4 array: x, y, dir, isNew) used throughout the pipeline.
"""

import numpy as np

# DXF INSUNITS codes -> mm conversion factor
_INSUNITS_TO_MM = {
    0:  1.0,       # Unitless — assume mm
    1:  25.4,      # Inches
    2:  304.8,     # Feet
    4:  1.0,       # Millimeters
    5:  10.0,      # Centimeters
    6:  1000.0,    # Meters
}

_TARGET_SIZE = 230.0   # pixels to fill within the 256x256 canvas
_CANVAS      = 256.0


def parse_dxf_boundary(file_bytes):
    """
    Extract boundary polygon vertices from DXF bytes.

    Returns:
        pts          : list of (x, y) tuples in original DXF units
        units_to_mm  : conversion factor from DXF units to millimetres
    """
    try:
        import ezdxf  # type: ignore
    except ImportError:
        raise RuntimeError("ezdxf is required for DXF import. Install with: pip install ezdxf")

    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix='.dxf', delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    try:
        doc = ezdxf.readfile(tmp_path)
    finally:
        os.unlink(tmp_path)
    msp = doc.modelspace()

    # Determine unit conversion factor from DXF header
    insunits = doc.header.get('$INSUNITS', 0)
    units_to_mm = _INSUNITS_TO_MM.get(int(insunits), 1.0)

    # Debug: log all entity types present
    all_types = {}
    for e in msp:
        t = e.dxftype()
        all_types[t] = all_types.get(t, 0) + 1
    print(f"[DXF] Entity types in modelspace: {all_types}")

    pts = None

    # 1. Try LWPOLYLINE first (most common in modern DXF)
    lwpolylines = list(msp.query('LWPOLYLINE'))
    print(f"[DXF] LWPOLYLINE count: {len(lwpolylines)}")
    if lwpolylines:
        best = max(lwpolylines, key=lambda p: len(list(p.vertices())))
        pts = [(v[0], v[1]) for v in best.vertices()]
        print(f"[DXF] Using LWPOLYLINE with {len(pts)} vertices")

    # 2. Try POLYLINE (older DXF format)
    if pts is None:
        polylines = [e for e in msp.query('POLYLINE') if e.is_2d_polyline or e.is_3d_polyline]
        print(f"[DXF] POLYLINE count: {len(polylines)}")
        if polylines:
            best = max(polylines, key=lambda p: len(list(p.vertices())))
            pts = [(v.dxf.location.x, v.dxf.location.y) for v in best.vertices()]
            print(f"[DXF] Using POLYLINE with {len(pts)} vertices")

    # 3. Try assembling LINE entities into a closed loop
    if pts is None:
        lines = list(msp.query('LINE'))
        print(f"[DXF] LINE count: {len(lines)}")
        if lines:
            pts = _assemble_lines(lines)
            print(f"[DXF] LINE assembly result: {len(pts) if pts else 'failed'}")

    if not pts or len(pts) < 3:
        raise ValueError(f"No valid boundary polygon found. Entity types present: {all_types}")

    # Remove duplicate closing point if present
    if len(pts) > 1 and _pts_equal(pts[0], pts[-1]):
        pts = pts[:-1]

    return pts, units_to_mm


def normalize_to_256(pts, units_to_mm):
    """
    Uniformly scale pts (preserving aspect ratio) to fit within _TARGET_SIZE
    pixels, centered on the _CANVAS x _CANVAS grid.

    Returns:
        norm_pts         : list of (x, y) in pixel space
        scale_mm_per_px  : how many mm one pixel represents
    """
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    width  = max_x - min_x
    height = max_y - min_y

    if width <= 0 or height <= 0:
        raise ValueError("Boundary has zero width or height.")

    px_per_unit = _TARGET_SIZE / max(width, height)   # pixels per DXF unit
    offset_x = (_CANVAS - width  * px_per_unit) / 2.0
    offset_y = (_CANVAS - height * px_per_unit) / 2.0

    norm_pts = [
        ((p[0] - min_x) * px_per_unit + offset_x,
         (p[1] - min_y) * px_per_unit + offset_y)
        for p in pts
    ]

    # scale_mm_per_px: real-world mm represented by one pixel
    scale_mm_per_px = units_to_mm / px_per_unit

    return norm_pts, scale_mm_per_px


def compute_dirs(pts):
    """
    Compute direction codes for each vertex based on the wall it starts.
      0 = rightward  (dx > 0)
      1 = downward   (dy > 0)
      2 = leftward   (dx < 0)
      3 = upward     (dy < 0)
    """
    n = len(pts)
    dirs = []
    for i in range(n):
        j = (i + 1) % n
        dx = pts[j][0] - pts[i][0]
        dy = pts[j][1] - pts[i][1]
        if abs(dx) >= abs(dy):
            dirs.append(0 if dx >= 0 else 2)
        else:
            dirs.append(1 if dy >= 0 else 3)
    return dirs


def build_boundary_array(pts, door_wall_idx):
    """
    Build the N x 4 boundary array [x, y, dir, isNew] with the front door
    wall at index 0 (i.e. pts[0] and pts[1] are the door endpoints).

    door_wall_idx: index of the wall whose start vertex becomes pts[0].
    """
    n = len(pts)
    # Reorder so the selected wall starts at index 0
    pts = pts[door_wall_idx:] + pts[:door_wall_idx]
    dirs = compute_dirs(pts)
    return np.array(
        [[pts[i][0], pts[i][1], dirs[i], 0] for i in range(n)],
        dtype=float
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _pts_equal(a, b, tol=1e-6):
    return abs(a[0] - b[0]) < tol and abs(a[1] - b[1]) < tol


def _assemble_lines(lines):
    """
    Attempt to chain LINE entities into a single closed polygon.
    Returns list of (x, y) or None if assembly fails.
    """
    segments = [
        ((e.dxf.start.x, e.dxf.start.y), (e.dxf.end.x, e.dxf.end.y))
        for e in lines
    ]
    if not segments:
        return None

    chain = list(segments[0])
    remaining = list(segments[1:])

    for _ in range(len(remaining)):
        tail = chain[-1]
        found = False
        for i, (a, b) in enumerate(remaining):
            if _pts_equal(tail, a):
                chain.append(b)
                remaining.pop(i)
                found = True
                break
            elif _pts_equal(tail, b):
                chain.append(a)
                remaining.pop(i)
                found = True
                break
        if not found:
            break

    # Check closed
    if len(chain) >= 4 and _pts_equal(chain[0], chain[-1]):
        return chain[:-1]
    return None

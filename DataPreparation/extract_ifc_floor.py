"""
Stage 0b of the hotel floor-level pipeline.

IFC-based drop-in replacement for the Revit Python Shell script
(revit_scripts/extract_revit_floor.py).  Reads an .ifc file exported from
Revit (or any BIM tool) and writes the same JSON schema consumed by
extract_hotel_floor.py.

No Revit license is needed after the one-time IFC export.  Runs on any
machine with ifcopenshell installed.

Output schema (identical to extract_revit_floor.py):
-----------------------------------------------------
{
  "star_rating": 4,
  "floor": "L1",
  "rooms": [
    {"id": 0, "name": "BR", "area_m2": 34.25,
     "geometry": [[x, y], ...], "number": "20"}
  ],
  "doors": [{"from": 0, "to": 1}]   # to=-1 = corridor / outside
}

Requirements
------------
  pip install ifcopenshell numpy

Usage
-----
  python extract_ifc_floor.py hotel.ifc --list-storeys
  python extract_ifc_floor.py hotel.ifc --storey "01-F1-FFL"
  python extract_ifc_floor.py hotel.ifc --storey "01-F1-FFL" --out raw/floor_L1.json
  python extract_ifc_floor.py hotel.ifc --storey "01-F1-FFL" --star 4 --label L1
  python extract_ifc_floor.py hotel.ifc --all-storeys --out-dir raw_floors/ --star 4

After extraction, feed the output into the normal Stage 1 pipeline:
  python extract_hotel_floor.py raw_floors/ --out processed_floors/
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

try:
    import ifcopenshell
    import ifcopenshell.util.placement
    import ifcopenshell.util.element
    import ifcopenshell.util.unit
except ImportError:
    print("[ERROR] ifcopenshell not installed.  Run:  pip install ifcopenshell")
    sys.exit(1)

# Points sampled per curved segment (circles, arcs)
CURVE_POINTS = 8


# =============================================================================
# Unit conversion
# =============================================================================

def _unit_scale_to_m(ifc):
    """Return factor: IFC project length unit → metres."""
    try:
        return ifcopenshell.util.unit.calculate_unit_scale(ifc)
    except Exception:
        pass
    # Fallback: parse IfcUnitAssignment directly
    for ua in ifc.by_type("IfcUnitAssignment"):
        for unit in ua.Units:
            if not hasattr(unit, "UnitType") or unit.UnitType != "LENGTHUNIT":
                continue
            if unit.is_a("IfcSIUnit"):
                return {
                    "MILLI": 0.001, "CENTI": 0.01, None: 1.0,
                    "KILO": 1000.0,
                }.get(getattr(unit, "Prefix", None), 1.0)
            if unit.is_a("IfcConversionBasedUnit"):
                n = (unit.Name or "").upper()
                if "FOOT" in n:
                    return 0.3048
                if "INCH" in n:
                    return 0.0254
    return 1.0  # default: assume project unit is metres


# =============================================================================
# Matrix helpers
# =============================================================================

def _world_matrix(placement):
    """4×4 world transformation matrix from IfcObjectPlacement (or None)."""
    if placement is None:
        return np.eye(4)
    return ifcopenshell.util.placement.get_local_placement(placement)


def _ax2_matrix(ax2):
    """4×4 matrix from IfcAxis2Placement2D or IfcAxis2Placement3D."""
    m = np.eye(4)
    if ax2 is None:
        return m
    loc = ax2.Location.Coordinates
    m[0, 3] = float(loc[0])
    m[1, 3] = float(loc[1])
    if ax2.is_a("IfcAxis2Placement3D"):
        m[2, 3] = float(loc[2]) if len(loc) > 2 else 0.0
        if ax2.Axis and ax2.RefDirection:
            z = np.array(list(ax2.Axis.DirectionRatios[:3]), dtype=float)
            x = np.array(list(ax2.RefDirection.DirectionRatios[:3]), dtype=float)
            nz = np.linalg.norm(z)
            nx = np.linalg.norm(x)
            if nz > 0 and nx > 0:
                z /= nz
                x -= np.dot(x, z) * z
                x /= np.linalg.norm(x)
                y = np.cross(z, x)
                m[:3, 0] = x
                m[:3, 1] = y
                m[:3, 2] = z
    else:  # IfcAxis2Placement2D
        if hasattr(ax2, "RefDirection") and ax2.RefDirection:
            d = ax2.RefDirection.DirectionRatios
            dx, dy = float(d[0]), float(d[1])
            m[0, 0] = dx;  m[1, 0] = dy
            m[0, 1] = -dy; m[1, 1] = dx
    return m


def _apply_xy(pts_2d, matrix):
    """Transform [(x,y), ...] through a 4×4 matrix → [(x,y), ...]."""
    out = []
    for x, y in pts_2d:
        p = np.array([float(x), float(y), 0.0, 1.0])
        w = matrix @ p
        out.append((float(w[0]), float(w[1])))
    return out


# =============================================================================
# Curve / profile geometry parsers  (no geometry engine required)
# =============================================================================

def _parse_curve(curve):
    """
    Extract [(x, y), ...] from common IfcCurve subtypes.
    Returns None if the type is not handled.
    """
    if curve is None:
        return None
    t = curve.is_a()

    if t == "IfcPolyline":
        return [(float(p.Coordinates[0]), float(p.Coordinates[1]))
                for p in curve.Points]

    if t == "IfcIndexedPolyCurve":
        cl = curve.Points.CoordList
        return [(float(c[0]), float(c[1])) for c in cl]

    if t in ("IfcCompositeCurve",):
        pts = []
        for seg in curve.Segments:
            pc = getattr(seg, "ParentCurve", None) or getattr(seg, "Curve", None)
            sub = _parse_curve(pc)
            if sub:
                pts.extend(sub[:-1])   # skip last point to avoid duplicates at joints
        return pts or None

    if t == "IfcTrimmedCurve":
        return _parse_curve(curve.BasisCurve)

    if t == "IfcCircle":
        cx = float(curve.Position.Location.Coordinates[0])
        cy = float(curve.Position.Location.Coordinates[1])
        r  = float(curve.Radius)
        n  = CURVE_POINTS * 4
        return [(cx + r * np.cos(2 * np.pi * i / n),
                 cy + r * np.sin(2 * np.pi * i / n))
                for i in range(n)]

    return None


def _parse_profile(profile):
    """Extract footprint [(x, y), ...] from an IfcProfileDef."""
    if profile is None:
        return None
    t = profile.is_a()

    if t == "IfcArbitraryClosedProfileDef":
        return _parse_curve(profile.OuterCurve)

    if t == "IfcRectangleProfileDef":
        xh = float(profile.XDim) / 2.0
        yh = float(profile.YDim) / 2.0
        base = [(-xh, -yh), (xh, -yh), (xh, yh), (-xh, yh)]
        if getattr(profile, "Position", None):
            base = _apply_xy(base, _ax2_matrix(profile.Position))
        return base

    if t == "IfcCompositeProfileDef":
        for sub in profile.Profiles:
            pts = _parse_profile(sub)
            if pts:
                return pts

    return None


def _body_footprint(items):
    """
    Try to extract a 2D footprint from Body representation items.
    Handles IfcExtrudedAreaSolid and common Boolean wrappers.
    Returns [(x, y), ...] or None.
    """
    for item in items:
        t = item.is_a()

        if t == "IfcExtrudedAreaSolid":
            pts = _parse_profile(item.SweptArea)
            if pts:
                if getattr(item, "Position", None):
                    pts = _apply_xy(pts, _ax2_matrix(item.Position))
                return pts

        if t in ("IfcBooleanResult", "IfcBooleanClippingResult"):
            pts = _body_footprint([item.FirstOperand])
            if pts:
                return pts

        if t == "IfcMappedItem":
            src_items = list(item.MappingSource.MappedRepresentation.Items)
            pts = _body_footprint(src_items)
            if pts:
                origin_m = _ax2_matrix(item.MappingSource.MappingOrigin) \
                           if hasattr(item.MappingSource, "MappingOrigin") else np.eye(4)
                target_m = _ax2_matrix(item.MappingTarget) \
                           if item.MappingTarget else np.eye(4)
                pts = _apply_xy(pts, origin_m)
                pts = _apply_xy(pts, target_m)
                return pts

    return None


# =============================================================================
# Geometry engine fallback  (requires ifcopenshell.geom / pythonocc)
# =============================================================================

def _geom_footprint(space):
    """
    Last-resort geometry extraction using ifcopenshell.geom.
    Returns [(x, y), ...] in IFC project units, or None.
    """
    try:
        import ifcopenshell.geom

        settings = ifcopenshell.geom.settings()
        settings.set(settings.USE_WORLD_COORDS, True)
        settings.set(settings.WELD_VERTICES, False)

        shape = ifcopenshell.geom.create_shape(settings, space)
        verts = np.array(shape.geometry.verts).reshape(-1, 3)
        faces = np.array(shape.geometry.faces).reshape(-1, 3)

        # Find bottom face (minimum Z)
        min_z = float(verts[:, 2].min())
        tol   = max(abs(min_z) * 0.001, 0.01)

        is_bottom = np.abs(verts[:, 2] - min_z) < tol

        # Collect boundary edges from bottom-face triangles
        edge_count = {}
        for face in faces:
            if all(is_bottom[v] for v in face):
                for k in range(3):
                    e = (min(int(face[k]), int(face[(k + 1) % 3])),
                         max(int(face[k]), int(face[(k + 1) % 3])))
                    edge_count[e] = edge_count.get(e, 0) + 1

        boundary_edges = [e for e, c in edge_count.items() if c == 1]
        if not boundary_edges:
            return None

        # Walk the boundary polygon
        adj = {}
        for a, b in boundary_edges:
            adj.setdefault(a, []).append(b)
            adj.setdefault(b, []).append(a)

        start  = boundary_edges[0][0]
        path   = [start]
        visited = {start}
        curr   = start
        while True:
            nexts = [n for n in adj.get(curr, []) if n not in visited]
            if not nexts:
                break
            curr = nexts[0]
            path.append(curr)
            visited.add(curr)

        return [(float(verts[i, 0]), float(verts[i, 1])) for i in path]

    except Exception:
        return None


# =============================================================================
# Room polygon extraction
# =============================================================================

def _room_polygon(space, to_mm):
    """
    Return the room polygon as [[x_mm, y_mm], ...].
    Tries FootPrint rep → Body rep → geometry engine fallback.
    Returns [] if nothing works.
    """
    if not space.Representation:
        return []

    world_m = _world_matrix(space.ObjectPlacement)

    for rep in space.Representation.Representations:
        rid = rep.RepresentationIdentifier or ""

        # 1. FootPrint: 2-D curve in local space
        if rid == "FootPrint":
            for item in rep.Items:
                pts = _parse_curve(item)
                if pts:
                    world_pts = _apply_xy(pts, world_m)
                    return [[round(x * to_mm, 4), round(y * to_mm, 4)]
                            for x, y in world_pts]

        # 2. Body: extruded profile
        if rid in ("Body", "Body/Clipping", ""):
            pts = _body_footprint(list(rep.Items))
            if pts:
                world_pts = _apply_xy(pts, world_m)
                return [[round(x * to_mm, 4), round(y * to_mm, 4)]
                        for x, y in world_pts]

    # 3. Geometry engine fallback (requires pythonocc)
    pts = _geom_footprint(space)
    if pts:
        return [[round(x * to_mm, 4), round(y * to_mm, 4)] for x, y in pts]

    return []


# =============================================================================
# Room metadata
# =============================================================================

def _room_name_number(space):
    """
    Return (name, number) for an IfcSpace.

    Revit IFC export convention:
      IfcSpace.Name     → room number  (e.g. "301")
      IfcSpace.LongName → room type    (e.g. "BEDROOM")
    """
    long_name = (getattr(space, "LongName", None) or "").strip()
    name_attr = (getattr(space, "Name",     None) or "").strip()

    room_name   = long_name or name_attr or "Unknown"
    room_number = name_attr

    # Override from property sets when available
    try:
        psets = ifcopenshell.util.element.get_psets(space)
        for pset in psets.values():
            if isinstance(pset, dict):
                if pset.get("Name"):
                    room_name   = str(pset["Name"])
                if pset.get("Number"):
                    room_number = str(pset["Number"])
    except Exception:
        pass

    return room_name, room_number


def _room_area_m2(space, unit_to_m):
    """
    Return room area in m² from IFC quantity sets.
    Returns None if no quantity is found.
    """
    try:
        psets = ifcopenshell.util.element.get_psets(space, qtos_only=True)
        for qto in psets.values():
            if not isinstance(qto, dict):
                continue
            for key in ("NetFloorArea", "GrossFloorArea", "NetArea", "GrossArea", "Area"):
                val = qto.get(key)
                if val is not None:
                    # quantity is in project area units → convert to m²
                    return round(float(val) * (unit_to_m ** 2), 3)
    except Exception:
        pass
    return None


def _estimate_area_m2(polygon_mm):
    """Shoelace area of a polygon given in mm → m²."""
    if len(polygon_mm) < 3:
        return 0.0
    pts = polygon_mm
    n   = len(pts)
    area_mm2 = abs(sum(
        pts[i][0] * pts[(i + 1) % n][1] - pts[(i + 1) % n][0] * pts[i][1]
        for i in range(n)
    )) / 2.0
    return round(area_mm2 / 1e6, 3)   # mm² → m²


# =============================================================================
# Door connectivity
# =============================================================================

def _extract_doors(ifc, space_to_local):
    """
    Build [{from, to}, ...] door list using IfcRelSpaceBoundary.
    Falls back to IfcRelConnectsSpaces if no boundary relations are found.
    """
    door_spaces = {}   # ifc door id → [local_id, ...]

    for rel in ifc.by_type("IfcRelSpaceBoundary"):
        elem = rel.RelatedBuildingElement
        if elem is None or not elem.is_a("IfcDoor"):
            continue
        space = rel.RelatingSpace
        if space not in space_to_local:
            continue
        did = elem.id()
        lid = space_to_local[space]
        if lid not in door_spaces.get(did, []):
            door_spaces.setdefault(did, []).append(lid)

    doors_out = []
    seen      = set()

    for did, space_ids in door_spaces.items():
        if len(space_ids) >= 2:
            f, t = space_ids[0], space_ids[1]
        else:
            f, t = space_ids[0], -1   # exterior door

        key = (f, t)
        if key not in seen:
            seen.add(key)
            doors_out.append({"from": f, "to": t})

    # Supplement with IfcRelConnectsSpaces (some exporters use this instead)
    for rel in ifc.by_type("IfcRelConnectsSpaces"):
        s1 = rel.RelatingSpace
        s2 = rel.RelatedSpace
        if s1 not in space_to_local or s2 not in space_to_local:
            continue
        f = space_to_local[s1]
        t = space_to_local[s2]
        key = (min(f, t), max(f, t))
        if key not in seen:
            seen.add(key)
            doors_out.append({"from": f, "to": t})

    return doors_out


# =============================================================================
# Storey helpers
# =============================================================================

def _get_storeys(ifc):
    """All IfcBuildingStorey elements, sorted by elevation (ascending)."""
    storeys = ifc.by_type("IfcBuildingStorey")
    try:
        storeys = sorted(storeys, key=lambda s: float(s.Elevation or 0))
    except Exception:
        pass
    return storeys


def _spaces_in_storey(storey):
    """IfcSpace elements directly contained in (or decomposing) a storey."""
    spaces = []
    seen   = set()

    for rel in (storey.ContainsElements or []):
        for elem in rel.RelatedElements:
            if elem.is_a("IfcSpace") and elem.id() not in seen:
                seen.add(elem.id())
                spaces.append(elem)

    for rel in (storey.IsDecomposedBy or []):
        for obj in rel.RelatedObjects:
            if obj.is_a("IfcSpace") and obj.id() not in seen:
                seen.add(obj.id())
                spaces.append(obj)

    return spaces


def _storey_label(storey):
    return (getattr(storey, "Name", None) or
            getattr(storey, "LongName", None) or
            "unknown").strip()


# =============================================================================
# Core extraction
# =============================================================================

def extract_storey(ifc, storey, star_rating, floor_label, to_mm, unit_to_m):
    """
    Extract rooms and doors for one IfcBuildingStorey.
    Returns output dict (same schema as extract_revit_floor.py) or None.
    """
    s_name = _storey_label(storey)
    print("Extracting storey: {0}".format(s_name))
    print("=" * 50)

    spaces = _spaces_in_storey(storey)
    if not spaces:
        print("  [WARN] No IfcSpace elements found in storey.")
        return None

    # Stable ordering by room number (mirrors the Revit script sort)
    spaces.sort(key=lambda s: _room_name_number(s)[1])
    print("Rooms in storey: {0}".format(len(spaces)))

    space_to_local = {s: i for i, s in enumerate(spaces)}

    rooms_out = []
    for local_id, space in enumerate(spaces):
        room_name, room_number = _room_name_number(space)

        polygon = _room_polygon(space, to_mm)

        area = _room_area_m2(space, unit_to_m)
        if area is None:
            area = _estimate_area_m2(polygon)

        rooms_out.append({
            "id":      local_id,
            "name":    room_name,
            "area_m2": area,
            "geometry": polygon,
            "number":  room_number,
        })

        print("  [{0:>3}] #{1:<6} '{2}'  {3:.2f} m2  {4} pts".format(
            local_id,
            room_number[:6],
            room_name[:25],
            area,
            len(polygon),
        ))

    doors_out = _extract_doors(ifc, space_to_local)
    print("")
    print("Doors extracted: {0}".format(len(doors_out)))

    return {
        "star_rating": star_rating,
        "floor":       floor_label,
        "rooms":       rooms_out,
        "doors":       doors_out,
    }


# =============================================================================
# Output
# =============================================================================

def _write(data, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print("")
    print("Wrote: {0}  ({1} rooms, {2} doors)".format(
        out_path, len(data["rooms"]), len(data["doors"])))
    print("")
    print("Next step:")
    print('  python extract_hotel_floor.py "{0}"'.format(out_path))


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Stage 0b: Extract hotel floor data from IFC → raw JSON "
            "for extract_hotel_floor.py."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python extract_ifc_floor.py hotel.ifc --list-storeys\n"
            "  python extract_ifc_floor.py hotel.ifc --storey '01-F1-FFL' --star 4 --label L1\n"
            "  python extract_ifc_floor.py hotel.ifc --all-storeys --out-dir raw_floors/ --star 4\n"
        ),
    )
    parser.add_argument("ifc_file",
                        help="Path to the .ifc file")
    parser.add_argument("--storey", default="",
                        help="Storey name to extract (partial, case-insensitive). "
                             "Omit to use the first storey.")
    parser.add_argument("--list-storeys", action="store_true",
                        help="List all storeys in the file and exit.")
    parser.add_argument("--all-storeys", action="store_true",
                        help="Extract every storey as a separate JSON file.")
    parser.add_argument("--out", default="",
                        help="Output JSON path (single-storey mode). "
                             "Default: <ifc_stem>_<storey>.json")
    parser.add_argument("--out-dir", default="raw_floors",
                        help="Output directory for --all-storeys (default: raw_floors/)")
    parser.add_argument("--star", type=int, default=4,
                        help="Hotel star rating written to JSON (default: 4)")
    parser.add_argument("--label", default="",
                        help="Floor label string written to JSON. "
                             "Default: storey Name from IFC.")
    parser.add_argument("--unit", choices=["mm", "m"], default="mm",
                        help="Coordinate output unit (default: mm — matches Revit script)")
    args = parser.parse_args()

    ifc_path = Path(args.ifc_file)
    if not ifc_path.exists():
        print("[ERROR] File not found: {0}".format(ifc_path))
        sys.exit(1)

    print("Loading IFC: {0}".format(ifc_path))
    ifc = ifcopenshell.open(str(ifc_path))

    unit_to_m = _unit_scale_to_m(ifc)
    to_mm     = unit_to_m * 1000.0 if args.unit == "mm" else unit_to_m

    storeys = _get_storeys(ifc)

    # --list-storeys ----------------------------------------------------------------
    if args.list_storeys:
        print("")
        print("Storeys in {0}:".format(ifc_path.name))
        for s in storeys:
            n_sp = len(_spaces_in_storey(s))
            print("  '{0}'  (elevation={1})  {2} space(s)".format(
                _storey_label(s), s.Elevation or 0, n_sp))
        sys.exit(0)

    # --all-storeys -----------------------------------------------------------------
    if args.all_storeys:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for storey in storeys:
            s_name = _storey_label(storey)
            data   = extract_storey(ifc, storey, args.star, s_name, to_mm, unit_to_m)
            if data:
                safe = s_name.replace("/", "_").replace(" ", "_")
                _write(data, out_dir / "{0}_{1}.json".format(ifc_path.stem, safe))
        return

    # Single storey -----------------------------------------------------------------
    if args.storey:
        target  = args.storey.lower()
        matched = [s for s in storeys
                   if target in _storey_label(s).lower()]
        if not matched:
            print("[ERROR] No storey matching '{0}'.".format(args.storey))
            print("Run with --list-storeys to see available names.")
            sys.exit(1)
        storey = matched[0]
    elif storeys:
        storey = storeys[0]
        print("[INFO] No --storey given; using first storey: '{0}'".format(
            _storey_label(storey)))
    else:
        print("[ERROR] No IfcBuildingStorey found in file.")
        sys.exit(1)

    s_name = _storey_label(storey)
    label  = args.label or s_name
    data   = extract_storey(ifc, storey, args.star, label, to_mm, unit_to_m)

    if not data:
        print("[ERROR] No rooms extracted.")
        sys.exit(1)

    if args.out:
        out_path = Path(args.out)
    else:
        safe     = s_name.replace("/", "_").replace(" ", "_")
        out_path = Path("{0}_{1}.json".format(ifc_path.stem, safe))

    _write(data, out_path)


if __name__ == "__main__":
    main()

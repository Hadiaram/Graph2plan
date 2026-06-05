# extract_revit_hotel.py
#
# Revit Python Shell script — IronPython 2.7 compatible
# Run this script from inside Revit via the Revit Python Shell (RPS) panel.
# It exports all placed hotel room units on the active view's floor to our
# BIM JSON schema, which is then processed by extract_hotel_keys.py.
#
# Usage
# -----
#   1. Open the Revit Python Shell panel (Add-Ins tab)
#   2. Open this file in the RPS editor  (File → Open)
#   3. Edit the CONFIGURATION section below
#   4. Click Run (or press F5)
#   5. The output JSON is written to OUTPUT_PATH
#
# Then continue with the DataPreparation pipeline:
#   python extract_hotel_keys.py "<OUTPUT_PATH>" --out_dir extracted_keys/

# =============================================================================
# CONFIGURATION — edit before running
# =============================================================================

# Where to save the exported BIM JSON.  Use a raw string (r"...") on Windows.
OUTPUT_PATH = r"C:\Users\hmbashir\source\BIM Exports\hotel_floor.json"

# Star rating of this floor (4 or 5)
STAR_RATING = 5

# Floor label for this export (e.g. "L1", "L2", "Floor 3")
FLOOR_LABEL = "L1"

# --- Key (module) assignment ---
#
# "grid_bay"      : derive the key ID automatically from the enclosing grid lines
#                   e.g. a room between grids E-D (N/S) and 10-11 (E/W) → "E-D/10-11"
#                   Use this when running on multiple sheets — the same bay label
#                   appears in both exports so merge_hotel_sheets.py can join them.
# "parameter"     : read a shared parameter named KEY_PARAM_NAME from each room
# "number_prefix" : use the first KEY_PREFIX_CHARS characters of the room's
#                   Room Number as the key ID
#
KEY_METHOD       = "grid_bay"
KEY_PREFIX_CHARS = 3                # used only when KEY_METHOD == "number_prefix"
KEY_PARAM_NAME   = "Key ID"         # used only when KEY_METHOD == "parameter"

# --- Group assignment within a standard (split) key ---
#
# A "standard" key holds TWO independent bookable rooms (group A and B) that
# share a structural module but have no connecting door between them.
#
# "parameter" : read a shared parameter named GROUP_PARAM_NAME from each room
#               (add "Group" parameter to rooms, set value "A" or "B" in Revit)
# "spatial"   : auto-split by X coordinate — left half = A, right half = B
# "suffix"    : look for "A" or "B" in the room number after the key prefix
# "none"      : treat all rooms in every key as a single unit (no splitting)
#
GROUP_METHOD     = "parameter"
GROUP_PARAM_NAME = "Group"          # used only when GROUP_METHOD == "parameter"

# --- Unit type ---
#
# "infer"     : classify from room name keywords (see SUITE/DELUXE keywords below)
#               Room names like "Suite Bedroom" → suite
#               Room names like "Executive Suite Bedroom" → suite
#               Room names containing only "Bedroom" / "Bathroom" / "Closet" → standard
# "parameter" : read a shared parameter named UNIT_TYPE_PARAM_NAME from each room
#
UNIT_TYPE_METHOD     = "infer"
UNIT_TYPE_PARAM_NAME = "Unit Type"  # used only when UNIT_TYPE_METHOD == "parameter"

# If any room name in the key contains one of these keywords → unit_type = "suite"
SUITE_NAME_KEYWORDS  = ["suite", "full suite"]

# If no suite keyword found but any room name contains one of these → "deluxe"
DELUXE_NAME_KEYWORDS = ["executive", "deluxe", "superior", "junior suite"]

# Fallback: a key with this many rooms or more is "suite" even without keywords
SUITE_ROOM_THRESHOLD = 5

# Coordinate output units for geometry: "mm" or "m"
COORD_UNIT = "mm"

# Points sampled per curved boundary segment (higher = smoother polygon)
CURVE_POINTS = 8

# =============================================================================
# END CONFIGURATION
# =============================================================================

import clr
import sys
import json
import os
import math
from collections import defaultdict

clr.AddReference('RevitAPI')
clr.AddReference('RevitAPIUI')

from Autodesk.Revit.DB import (
    FilteredElementCollector,
    BuiltInCategory,
    BuiltInParameter,
    SpatialElementBoundaryOptions,
    SpatialElementBoundaryLocation,
    Line,
)

doc = __revit__.ActiveUIDocument.Document

# ---------------------------------------------------------------------------
# Unit conversion
# ---------------------------------------------------------------------------

if COORD_UNIT == "mm":
    _XY_SCALE = 304.8    # 1 Revit internal foot → mm
elif COORD_UNIT == "m":
    _XY_SCALE = 0.3048   # 1 Revit internal foot → m
else:
    raise ValueError("COORD_UNIT must be 'mm' or 'm'")

_AREA_SCALE = 0.092903   # sq ft → m²


def _xy(pt):
    return [round(pt.X * _XY_SCALE, 4), round(pt.Y * _XY_SCALE, 4)]


def _area_m2(room):
    return round(room.Area * _AREA_SCALE, 3)


# ---------------------------------------------------------------------------
# Geometry — boundary polygon
# ---------------------------------------------------------------------------

def _tessellate(curve, n=CURVE_POINTS):
    """Sample n+1 points along a Revit curve (normalised parameter 0→1)."""
    pts = []
    for i in range(n + 1):
        t = float(i) / float(n)
        pts.append(_xy(curve.Evaluate(t, True)))
    return pts


def _loop_to_polygon(loop):
    """Convert a BoundarySegment loop to a list of [x, y] vertices."""
    pts = []
    for seg in loop:
        curve = seg.GetCurve()
        if isinstance(curve, Line):
            pts.append(_xy(curve.GetEndPoint(0)))
        else:
            seg_pts = _tessellate(curve)
            pts.extend(seg_pts[:-1])   # last point == first point of next seg
    return pts


# ---------------------------------------------------------------------------
# Room parameter helpers
# ---------------------------------------------------------------------------

def _param_str(elem, bip=None, name=None):
    """Read a parameter by BuiltInParameter enum or by name. Returns '' on miss."""
    if bip is not None:
        p = elem.get_Parameter(bip)
    else:
        p = elem.LookupParameter(name)
    if p and p.HasValue:
        return (p.AsString() or p.AsValueString() or "").strip()
    return ""


def _room_number(room):
    return _param_str(room, bip=BuiltInParameter.ROOM_NUMBER)


def _room_name(room):
    n = _param_str(room, bip=BuiltInParameter.ROOM_NAME)
    return n if n else "Unknown"


# ---------------------------------------------------------------------------
# Grid bay helpers  (used when KEY_METHOD == "grid_bay")
# ---------------------------------------------------------------------------

_h_grids = None   # [(gy, name)]  loaded once
_v_grids = None   # [(gx, name)]


def _load_grids():
    global _h_grids, _v_grids
    _h_grids = []
    _v_grids = []
    raw = (FilteredElementCollector(doc)
           .OfCategory(BuiltInCategory.OST_Grids)
           .WhereElementIsNotElementType()
           .ToElements())
    for grid in raw:
        try:
            curve = grid.Curve
            sp = curve.GetEndPoint(0)
            ep = curve.GetEndPoint(1)
            dx = abs(ep.X - sp.X)
            dy = abs(ep.Y - sp.Y)
            name = grid.Name
            if dx >= dy:
                _h_grids.append(((sp.Y + ep.Y) / 2.0, name))
            else:
                _v_grids.append(((sp.X + ep.X) / 2.0, name))
        except Exception:
            pass


def _bay_id_for_point(pt):
    """Return a stable key ID string from the four enclosing grid lines."""
    above = sorted([(gy, n) for gy, n in _h_grids if gy > pt.Y], key=lambda t: t[0])
    below = sorted([(gy, n) for gy, n in _h_grids if gy < pt.Y], key=lambda t: t[0], reverse=True)
    east  = sorted([(gx, n) for gx, n in _v_grids if gx > pt.X], key=lambda t: t[0])
    west  = sorted([(gx, n) for gx, n in _v_grids if gx < pt.X], key=lambda t: t[0], reverse=True)

    north = above[0][1] if above else "?"
    south = below[0][1] if below else "?"
    e     = east[0][1]  if east  else "?"
    w     = west[0][1]  if west  else "?"
    return "{0}-{1}/{2}-{3}".format(south, north, w, e)


# ---------------------------------------------------------------------------
# Key ID
# ---------------------------------------------------------------------------

def _key_id(room):
    if KEY_METHOD == "grid_bay":
        if room.Location is None:
            print("  [WARN] Room '{0}' has no location — key=UNKNOWN".format(
                _room_name(room)))
            return "UNKNOWN"
        return _bay_id_for_point(room.Location.Point)
    if KEY_METHOD == "parameter":
        val = _param_str(room, name=KEY_PARAM_NAME)
        if val:
            return val
        print("  [WARN] Room {0} (id={1}) missing parameter '{2}' — key=UNKNOWN".format(
            _room_name(room), room.Id.IntegerValue, KEY_PARAM_NAME))
        return "UNKNOWN"
    # number_prefix
    num = _room_number(room)
    return num[:KEY_PREFIX_CHARS] if len(num) >= KEY_PREFIX_CHARS else num or "UNKNOWN"


# ---------------------------------------------------------------------------
# Group (A / B)
# ---------------------------------------------------------------------------

def _group(room):
    """Return 'A', 'B', or None (spatial mode handled separately in extract())."""
    if GROUP_METHOD == "none":
        return None
    if GROUP_METHOD == "parameter":
        val = _param_str(room, name=GROUP_PARAM_NAME).upper()
        return val if val in ("A", "B") else None
    if GROUP_METHOD == "suffix":
        num = _room_number(room)
        suffix = num[KEY_PREFIX_CHARS:].upper() if len(num) > KEY_PREFIX_CHARS else ""
        if "A" in suffix:
            return "A"
        if "B" in suffix:
            return "B"
        return None
    # "spatial" — caller handles this after all rooms are collected
    return None


def _assign_groups_spatial(rooms):
    """
    Split rooms into groups A and B by X coordinate of room location.
    The left half (lower X) → A, right half (higher X) → B.
    Returns a dict local_id → 'A' | 'B'.
    """
    indexed = [(i, r) for i, r in enumerate(rooms)]
    indexed.sort(key=lambda ir: ir[1].Location.Point.X if ir[1].Location else 0)
    mid = len(indexed) // 2
    result = {}
    for i, (local_id, _) in enumerate(indexed):
        result[local_id] = 'A' if i < mid else 'B'
    return result


# ---------------------------------------------------------------------------
# Unit type inference
# ---------------------------------------------------------------------------

def _infer_unit_type(rooms):
    """
    Classify by looking for suite/deluxe keywords in any room name.
    Room names like "Suite Bedroom" or "Executive Suite Bedroom" carry the tier.
    """
    names = [_room_name(r).lower() for r in rooms]
    combined = " ".join(names)
    if len(rooms) >= SUITE_ROOM_THRESHOLD:
        return "suite"
    if any(kw in combined for kw in SUITE_NAME_KEYWORDS):
        return "suite"
    if any(kw in combined for kw in DELUXE_NAME_KEYWORDS):
        return "deluxe"
    return "standard"


# ---------------------------------------------------------------------------
# Door adjacency
# ---------------------------------------------------------------------------

def _collect_doors():
    """
    Return list of {from_id: int, to_id: int} using Revit room IDs (integers).
    to_id == -1 means the door leads outside the key (corridor / exterior).
    """
    phase = doc.Phases[doc.Phases.Size - 1]   # last/active phase
    doors = (FilteredElementCollector(doc)
             .OfCategory(BuiltInCategory.OST_Doors)
             .WhereElementIsNotElementType()
             .ToElements())
    result = []
    for door in doors:
        try:
            from_room = door.get_FromRoom(phase)
            to_room   = door.get_ToRoom(phase)
        except Exception:
            continue
        if from_room is None:
            continue
        result.append({
            'from_id': from_room.Id.IntegerValue,
            'to_id':   to_room.Id.IntegerValue if to_room else -1,
        })
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def extract():
    print("Hotel BIM export — Revit Python Shell")
    print("=" * 50)
    print("  Document : {0}".format(doc.Title))
    print("  Output   : {0}".format(OUTPUT_PATH))
    print("")

    # Load grids if using grid_bay key method
    if KEY_METHOD == "grid_bay":
        _load_grids()
        print("Grid lines: {0} horizontal, {1} vertical".format(
            len(_h_grids), len(_v_grids)))

    # Collect placed rooms
    all_rooms = (FilteredElementCollector(doc)
                 .OfCategory(BuiltInCategory.OST_Rooms)
                 .WhereElementIsNotElementType()
                 .ToElements())
    placed_rooms = [r for r in all_rooms if r.Area > 0]
    print("Placed rooms found: {0}".format(len(placed_rooms)))

    # Collect doors
    door_adj = _collect_doors()
    print("Door connections found: {0}".format(len(door_adj)))

    # Boundary options — use finish-face of walls
    bopts = SpatialElementBoundaryOptions()
    bopts.SpatialElementBoundaryLocation = SpatialElementBoundaryLocation.Finish

    # Group rooms by key
    key_buckets = defaultdict(list)
    for room in placed_rooms:
        key_buckets[_key_id(room)].append(room)

    print("Keys found: {0}".format(len(key_buckets)))
    print("")

    keys_out = []

    for kid in sorted(key_buckets.keys()):
        key_rooms = key_buckets[kid]
        print("  Key {0} — {1} room(s)".format(kid, len(key_rooms)))

        # Build per-room records, assign local IDs 0..N-1
        revit_to_local = {}
        room_records   = []

        for local_id, room in enumerate(key_rooms):
            rid = room.Id.IntegerValue
            revit_to_local[rid] = local_id

            loops = room.GetBoundarySegments(bopts)
            if loops and loops.Count > 0:
                polygon = _loop_to_polygon(list(loops[0]))
            else:
                print("    [WARN] Room '{0}' (id={1}) has no boundary".format(
                    _room_name(room), rid))
                polygon = []

            rec = {
                'id':        local_id,
                'name':      _room_name(room),
                'area_m2':   _area_m2(room),
                'geometry':  polygon,
                '_revit_id': rid,       # removed before output
            }
            room_records.append(rec)

        # Determine groups and key_configuration
        if GROUP_METHOD == "spatial":
            group_map = _assign_groups_spatial(key_rooms)
        else:
            group_map = {}
            for local_id, room in enumerate(key_rooms):
                group_map[local_id] = _group(room)

        has_a = any(g == 'A' for g in group_map.values())
        has_b = any(g == 'B' for g in group_map.values())
        key_cfg = 'standard' if (has_a and has_b) else 'suite'

        if key_cfg == 'standard':
            for rec in room_records:
                rec['group'] = group_map.get(rec['id']) or 'A'

        # Unit type
        if UNIT_TYPE_METHOD == "parameter":
            utype = "unknown"
            for room in key_rooms:
                val = _param_str(room, name=UNIT_TYPE_PARAM_NAME).lower()
                if val:
                    utype = val
                    break
        else:
            utype = _infer_unit_type(key_rooms)

        # Filter doors to those originating inside this key
        key_revit_ids = set(rec['_revit_id'] for rec in room_records)
        doors_out = []
        seen_doors = set()
        for d in door_adj:
            fid = d['from_id']
            tid = d['to_id']
            if fid not in key_revit_ids:
                continue
            f_local = revit_to_local[fid]
            t_local = revit_to_local.get(tid, -1) if tid != -1 else -1
            key = (f_local, t_local)
            if key not in seen_doors:
                seen_doors.add(key)
                doors_out.append({'from': f_local, 'to': t_local})

        # Strip internal _revit_id before writing
        for rec in room_records:
            del rec['_revit_id']

        keys_out.append({
            'key_id':            kid,
            'unit_type':         utype,
            'key_configuration': key_cfg,
            'rooms':             room_records,
            'doors':             doors_out,
        })

        cfg_str = "({0}) [{1}]".format(key_cfg, utype)
        print("    {0}  rooms: {1}  doors: {2}".format(
            cfg_str, len(room_records), len(doors_out)))

    output = {
        'star_rating': STAR_RATING,
        'floor':       FLOOR_LABEL,
        'keys':        keys_out,
    }

    # Write output
    out_dir = os.path.dirname(OUTPUT_PATH)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)

    with open(OUTPUT_PATH, 'w') as f:
        json.dump(output, f, indent=2)

    print("")
    print("=" * 50)
    print("Exported {0} key(s) to:".format(len(keys_out)))
    print("  {0}".format(OUTPUT_PATH))
    print("")
    print("Next step — run from DataPreparation/:")
    print('  python extract_hotel_keys.py "{0}" --out_dir extracted_keys/'.format(OUTPUT_PATH))


extract()

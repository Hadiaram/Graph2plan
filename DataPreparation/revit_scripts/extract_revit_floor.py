# extract_revit_floor.py
#
# Revit Python Shell — IronPython 3 compatible
#
# Extracts the entire hotel floor as a single flat JSON.
# Unlike extract_revit_hotel.py, rooms are NOT grouped by key —
# the whole floor is one training example for floor-level generation.
#
# Output schema
# -------------
# {
#   "star_rating": 5,
#   "floor": "L1",
#   "rooms": [
#     {"id": 0, "name": "BR", "area_m2": 34.25,
#      "geometry": [[x,y],...], "number": "20"}
#   ],
#   "doors": [{"from": 0, "to": 1}]   # to=-1 means corridor/outside
# }
#
# Usage
# -----
#   1. Open the floor plan view in Revit
#   2. Edit CONFIGURATION below
#   3. Run (F5)
#   4. Then run:  python extract_hotel_floor.py "<OUTPUT_PATH>"

# =============================================================================
# CONFIGURATION
# =============================================================================

OUTPUT_PATH = r"C:\Users\hmbashir\source\BIM Exports\floor_L1.json"

STAR_RATING = 4
FLOOR_LABEL = "L1"

# Filter to this level name only. Set "" to include all levels.
LEVEL_NAME = "01-F1-FFL"

# Coordinate output units: "mm" or "m"
COORD_UNIT = "mm"

# Points sampled per curved boundary segment
CURVE_POINTS = 8

# =============================================================================

import clr
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
import json
import os

doc = __revit__.ActiveUIDocument.Document

if COORD_UNIT == "mm":
    _XY_SCALE = 304.8
elif COORD_UNIT == "m":
    _XY_SCALE = 0.3048
else:
    raise ValueError("COORD_UNIT must be 'mm' or 'm'")

_AREA_SCALE = 0.092903

# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def _xy(pt):
    return [round(pt.X * _XY_SCALE, 4), round(pt.Y * _XY_SCALE, 4)]

def _tessellate(curve, n=CURVE_POINTS):
    pts = []
    for i in range(n + 1):
        t = float(i) / float(n)
        pts.append(_xy(curve.Evaluate(t, True)))
    return pts

def _loop_to_polygon(loop):
    pts = []
    for seg in loop:
        curve = seg.GetCurve()
        if isinstance(curve, Line):
            pts.append(_xy(curve.GetEndPoint(0)))
        else:
            pts.extend(_tessellate(curve)[:-1])
    return pts

# ---------------------------------------------------------------------------
# Parameter helpers
# ---------------------------------------------------------------------------

def _rname(room):
    p = room.get_Parameter(BuiltInParameter.ROOM_NAME)
    return (p.AsString() or "").strip() if (p and p.HasValue) else "Unknown"

def _rnum(room):
    p = room.get_Parameter(BuiltInParameter.ROOM_NUMBER)
    return (p.AsString() or "").strip() if (p and p.HasValue) else ""

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def extract():
    print("Floor-level BIM export")
    print("=" * 50)
    print("  Document : {0}".format(doc.Title))
    print("  Level    : {0}".format(LEVEL_NAME or "(all)"))
    print("  Output   : {0}".format(OUTPUT_PATH))
    print("")

    # Collect placed rooms
    all_rooms = list(FilteredElementCollector(doc)
                     .OfCategory(BuiltInCategory.OST_Rooms)
                     .WhereElementIsNotElementType()
                     .ToElements())
    placed = [r for r in all_rooms if r.Area > 0]

    if LEVEL_NAME:
        placed = [r for r in placed
                  if r.Level and r.Level.Name == LEVEL_NAME]

    print("Rooms on level: {0}".format(len(placed)))

    # Sort by room number for stable ordering
    placed.sort(key=lambda r: _rnum(r))

    # Build Revit element ID → local ID map
    revit_to_local = {}
    for local_id, room in enumerate(placed):
        revit_to_local[room.Id.IntegerValue] = local_id

    # Boundary options
    bopts = SpatialElementBoundaryOptions()
    bopts.SpatialElementBoundaryLocation = SpatialElementBoundaryLocation.Finish

    # Extract room records
    rooms_out = []
    for local_id, room in enumerate(placed):
        loops = room.GetBoundarySegments(bopts)
        if loops and loops.Count > 0:
            polygon = _loop_to_polygon(list(loops[0]))
        else:
            print("  [WARN] Room '{0}' #{1} has no boundary".format(
                _rname(room), _rnum(room)))
            polygon = []

        rooms_out.append({
            'id':       local_id,
            'name':     _rname(room),
            'area_m2':  round(room.Area * _AREA_SCALE, 3),
            'geometry': polygon,
            'number':   _rnum(room),
        })

        print("  [{0:>3}] #{1:<5} '{2}'  {3:.2f} m2  {4} pts".format(
            local_id,
            _rnum(room)[:4],
            _rname(room)[:25],
            room.Area * _AREA_SCALE,
            len(polygon)))

    # Extract doors
    phase = doc.Phases[doc.Phases.Size - 1]
    doors = list(FilteredElementCollector(doc)
                 .OfCategory(BuiltInCategory.OST_Doors)
                 .WhereElementIsNotElementType()
                 .ToElements())

    doors_out = []
    seen      = set()
    for door in doors:
        try:
            fr = door.get_FromRoom(phase)
            to = door.get_ToRoom(phase)
            if fr is None:
                continue
            fr_id = fr.Id.IntegerValue
            to_id = to.Id.IntegerValue if to else -1
            if fr_id not in revit_to_local:
                continue
            f_local = revit_to_local[fr_id]
            t_local = revit_to_local.get(to_id, -1) if to_id != -1 else -1
            key = (f_local, t_local)
            if key not in seen:
                seen.add(key)
                doors_out.append({'from': f_local, 'to': t_local})
        except Exception:
            pass

    print("")
    print("Doors extracted: {0}".format(len(doors_out)))

    output = {
        'star_rating': STAR_RATING,
        'floor':       FLOOR_LABEL,
        'rooms':       rooms_out,
        'doors':       doors_out,
    }

    out_dir = os.path.dirname(OUTPUT_PATH)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)

    with open(OUTPUT_PATH, 'w') as f:
        json.dump(output, f, indent=2)

    print("")
    print("=" * 50)
    print("Exported {0} rooms, {1} doors to:".format(
        len(rooms_out), len(doors_out)))
    print("  {0}".format(OUTPUT_PATH))
    print("")
    print("Next step:")
    print('  python extract_hotel_floor.py "{0}"'.format(OUTPUT_PATH))


extract()

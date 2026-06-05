# inspect_all_rooms_by_name.py
#
# Revit Python Shell diagnostic — IronPython 2.7 compatible
#
# Loops through every room matching SEARCH_NAME and prints a compact
# summary for each: parameters, connected doors, and grid bay.

# =============================================================================
# CONFIGURATION
# =============================================================================

SEARCH_NAME = "BR"      # Room name to inspect (case-insensitive substring)
EXACT_MATCH = False     # True = exact match only

# =============================================================================

import clr
clr.AddReference('RevitAPI')
clr.AddReference('RevitAPIUI')

from Autodesk.Revit.DB import (
    FilteredElementCollector,
    BuiltInCategory,
    BuiltInParameter,
    StorageType,
    SpatialElementBoundaryOptions,
    SpatialElementBoundaryLocation,
    ElementId,
)

doc = __revit__.ActiveUIDocument.Document

SEP  = "=" * 60
SEP2 = "-" * 40

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pstr(room, bip=None, name=None):
    p = room.get_Parameter(bip) if bip else room.LookupParameter(name)
    if p and p.HasValue:
        return (p.AsString() or p.AsValueString() or "").strip()
    return ""

def _rname(room):
    return _pstr(room, bip=BuiltInParameter.ROOM_NAME) or "Unknown"

def _rnum(room):
    return _pstr(room, bip=BuiltInParameter.ROOM_NUMBER)

def _load_grids():
    h, v = [], []
    for grid in (FilteredElementCollector(doc)
                 .OfCategory(BuiltInCategory.OST_Grids)
                 .WhereElementIsNotElementType()
                 .ToElements()):
        try:
            curve = grid.Curve
            sp = curve.GetEndPoint(0)
            ep = curve.GetEndPoint(1)
            dx = abs(ep.X - sp.X)
            dy = abs(ep.Y - sp.Y)
            name = grid.Name
            if dx >= dy:
                h.append(((sp.Y + ep.Y) / 2.0, name))
            else:
                v.append(((sp.X + ep.X) / 2.0, name))
        except Exception:
            pass
    return h, v

def _bay(pt, h_grids, v_grids):
    above = sorted([(gy, n) for gy, n in h_grids if gy > pt.Y], key=lambda t: t[0])
    below = sorted([(gy, n) for gy, n in h_grids if gy < pt.Y], key=lambda t: t[0], reverse=True)
    east  = sorted([(gx, n) for gx, n in v_grids if gx > pt.X], key=lambda t: t[0])
    west  = sorted([(gx, n) for gx, n in v_grids if gx < pt.X], key=lambda t: t[0], reverse=True)
    n = above[0][1] if above else "?"
    s = below[0][1] if below else "?"
    e = east[0][1]  if east  else "?"
    w = west[0][1]  if west  else "?"
    return "{0}-{1}/{2}-{3}".format(s, n, w, e)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

print(SEP)
print("Inspecting rooms matching: '{0}'".format(SEARCH_NAME))
print(SEP)

# All placed rooms
all_rooms = (FilteredElementCollector(doc)
             .OfCategory(BuiltInCategory.OST_Rooms)
             .WhereElementIsNotElementType()
             .ToElements())
placed = [r for r in all_rooms if r.Area > 0]

search_lower = SEARCH_NAME.lower()
if EXACT_MATCH:
    targets = [r for r in placed if _rname(r).lower() == search_lower]
else:
    targets = [r for r in placed if search_lower in _rname(r).lower()]

print("Found {0} matching room(s) out of {1} placed\n".format(len(targets), len(placed)))

if not targets:
    print("No rooms found. Check SEARCH_NAME.")
else:
    h_grids, v_grids = _load_grids()
    phase = doc.Phases[doc.Phases.Size - 1]

    # Collect all doors once
    all_doors = (FilteredElementCollector(doc)
                 .OfCategory(BuiltInCategory.OST_Doors)
                 .WhereElementIsNotElementType()
                 .ToElements())

    bopts = SpatialElementBoundaryOptions()
    bopts.SpatialElementBoundaryLocation = SpatialElementBoundaryLocation.Finish

    # Build a lookup: revit element id → room name/number for door labels
    room_lookup = {}
    for r in placed:
        room_lookup[r.Id.IntegerValue] = "{0} ({1})".format(_rname(r), _rnum(r))

    targets_sorted = sorted(targets, key=lambda r: _rnum(r))

    for room in targets_sorted:
        rid = room.Id.IntegerValue
        num = _rnum(room)
        name = _rname(room)
        area = round(room.Area * 0.092903, 2)

        print(SEP2)
        print("#{0}  '{1}'  id={2}".format(num, name, rid))
        print("  Area   : {0} m2".format(area))

        # Grid bay
        if room.Location:
            pt = room.Location.Point
            bay = _bay(pt, h_grids, v_grids)
            print("  Bay    : {0}".format(bay))
        else:
            print("  Bay    : (no location)")

        # All non-default parameters that have a value
        print("  Params :")
        skip_bips = {
            BuiltInParameter.ROOM_NAME,
            BuiltInParameter.ROOM_NUMBER,
            BuiltInParameter.ROOM_AREA,
            BuiltInParameter.ROOM_LEVEL_ID,
            BuiltInParameter.PHASE_CREATED,
            BuiltInParameter.PHASE_DEMOLISHED,
        }
        printed = 0
        for p in room.Parameters:
            try:
                if not p.HasValue:
                    continue
                defn = p.Definition
                if defn is None:
                    continue
                # Skip the built-ins we already show
                bip_param = room.get_Parameter(defn.BuiltInParameter) if hasattr(defn, 'BuiltInParameter') else None
                if p.StorageType == StorageType.String:
                    val = p.AsString()
                elif p.StorageType == StorageType.Integer:
                    val = str(p.AsInteger())
                elif p.StorageType == StorageType.Double:
                    val = str(round(p.AsDouble(), 4))
                elif p.StorageType == StorageType.ElementId:
                    val = "id:{0}".format(p.AsElementId().IntegerValue)
                else:
                    val = p.AsValueString()
                if not val or val in ("0", "-1", ""):
                    continue
                pname = defn.Name
                # Skip names we already print
                if pname in ("Name", "Number", "Area", "Level", "Phase Created",
                             "Phase Demolished", "Room: Name", "Room: Number"):
                    continue
                print("    {0:<35} = {1}".format(pname[:34], str(val)[:50]))
                printed += 1
            except Exception:
                pass
        if printed == 0:
            print("    (no extra parameters with values)")

        # Boundary segment count
        loops = room.GetBoundarySegments(bopts)
        seg_count = len(list(loops[0])) if (loops and loops.Count > 0) else 0
        print("  Boundary: {0} segment(s)".format(seg_count))

        # Connected doors
        connected = []
        for door in all_doors:
            try:
                fr = door.get_FromRoom(phase)
                to = door.get_ToRoom(phase)
                fr_id = fr.Id.IntegerValue if fr else None
                to_id = to.Id.IntegerValue if to else -1
                if fr_id != rid and to_id != rid:
                    continue
                other_id = to_id if fr_id == rid else fr_id
                other = room_lookup.get(other_id, "outside/corridor")
                connected.append("door id={0} → {1}".format(
                    door.Id.IntegerValue, other))
            except Exception:
                pass
        if connected:
            print("  Doors  :")
            for c in connected:
                print("    {0}".format(c))
        else:
            print("  Doors  : (none found)")

    print(SEP)
    print("Done. {0} room(s) inspected.".format(len(targets)))
    print(SEP)

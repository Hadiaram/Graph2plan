# diagnose_bathrooms.py
#
# Revit Python Shell — IronPython 3 compatible
#
# Picks one BR room, probes the surrounding bay, and reports exactly what
# the probe encounters at each grid point — helping identify why
# auto_place_bathrooms.py finds no bathroom spaces.

import clr
clr.AddReference('RevitAPI')
clr.AddReference('RevitAPIUI')

from Autodesk.Revit.DB import (
    FilteredElementCollector,
    BuiltInCategory,
    BuiltInParameter,
    Transaction,
    UV,
    XYZ,
)

doc = __revit__.ActiveUIDocument.Document

PROBE_STEP_FT = 1.5

_FT2_TO_M2 = 0.092903

SEP = "=" * 60

# ---------------------------------------------------------------------------
# Grab the first BR room
# ---------------------------------------------------------------------------

all_rooms = [r for r in
             FilteredElementCollector(doc)
             .OfCategory(BuiltInCategory.OST_Rooms)
             .WhereElementIsNotElementType()
             .ToElements()
             if r.Area > 0]

br_room = None
for r in all_rooms:
    p = r.get_Parameter(BuiltInParameter.ROOM_NAME)
    if p and p.HasValue and (p.AsString() or "").strip().upper() == "BR":
        br_room = r
        break

if br_room is None:
    print("No BR room found.")
    raise SystemExit(1)

num_p = br_room.get_Parameter(BuiltInParameter.ROOM_NUMBER)
num   = num_p.AsString() if (num_p and num_p.HasValue) else "?"
print(SEP)
print("Diagnosing bay for BR room #{0}  id={1}".format(num, br_room.Id.IntegerValue))
print("  Room area : {0:.2f} m2".format(br_room.Area * _FT2_TO_M2))
print("  Level     : {0}".format(br_room.Level.Name if br_room.Level else "(none)"))
if br_room.Level:
    print("  Level elev: {0:.3f} ft".format(br_room.Level.Elevation))
print(SEP)

# ---------------------------------------------------------------------------
# Find enclosing bay bounds
# ---------------------------------------------------------------------------

grids = list(FilteredElementCollector(doc)
             .OfCategory(BuiltInCategory.OST_Grids)
             .WhereElementIsNotElementType()
             .ToElements())

h_grids, v_grids = [], []
for grid in grids:
    try:
        curve = grid.Curve
        sp = curve.GetEndPoint(0)
        ep = curve.GetEndPoint(1)
        name = grid.Name
        if abs(ep.X - sp.X) >= abs(ep.Y - sp.Y):
            h_grids.append(((sp.Y + ep.Y) / 2.0, name))
        else:
            v_grids.append(((sp.X + ep.X) / 2.0, name))
    except Exception:
        pass

pt = br_room.Location.Point
above = sorted([(gy, n) for gy, n in h_grids if gy > pt.Y], key=lambda t: t[0])
below = sorted([(gy, n) for gy, n in h_grids if gy < pt.Y], key=lambda t: t[0], reverse=True)
east  = sorted([(gx, n) for gx, n in v_grids if gx > pt.X], key=lambda t: t[0])
west  = sorted([(gx, n) for gx, n in v_grids if gx < pt.X], key=lambda t: t[0], reverse=True)

if not (above and below and east and west):
    print("Could not determine bay bounds.")
    raise SystemExit(1)

x_min = west[0][0]
x_max = east[0][0]
y_min = below[0][0]
y_max = above[0][0]

print("Bay bounds (ft): X {0:.2f} → {1:.2f},  Y {2:.2f} → {3:.2f}".format(
    x_min, x_max, y_min, y_max))
print("Bay size (m):    W={0:.2f}  H={1:.2f}".format(
    (x_max - x_min) * 0.3048, (y_max - y_min) * 0.3048))
print("")

# ---------------------------------------------------------------------------
# Probe each grid point and report what's there
# ---------------------------------------------------------------------------

level = br_room.Level
elev  = level.Elevation + 1.0

print("Probe elevation: {0:.3f} ft  (level {1:.3f} + 1.0)".format(elev, level.Elevation))
print("")
print("Probe results:")
print("  {0:<22} {1:<25} {2}".format("Point (ft)", "GetRoomAtPoint", "NewRoom area"))
print("  {0:<22} {1:<25} {2}".format("-"*21, "-"*24, "-"*20))

n_occupied  = 0
n_empty     = 0
n_placed    = 0
n_zero_area = 0

t = Transaction(doc, "Diagnose probe")
t.Start()

try:
    x = x_min + PROBE_STEP_FT
    while x < x_max:
        y = y_min + PROBE_STEP_FT
        while y < y_max:
            pt3d     = XYZ(x, y, elev)
            existing = doc.GetRoomAtPoint(pt3d)

            if existing is not None:
                ex_name_p = existing.get_Parameter(BuiltInParameter.ROOM_NAME)
                ex_name   = ex_name_p.AsString() if (ex_name_p and ex_name_p.HasValue) else "?"
                ex_num_p  = existing.get_Parameter(BuiltInParameter.ROOM_NUMBER)
                ex_num    = ex_num_p.AsString() if (ex_num_p and ex_num_p.HasValue) else "?"
                room_str  = "#{0} '{1}'".format(ex_num, ex_name)
                new_str   = "(skipped)"
                n_occupied += 1
            else:
                room_str  = "(none)"
                n_empty   += 1
                try:
                    nr = doc.Create.NewRoom(level, UV(x, y))
                    if nr is not None:
                        area_m2  = nr.Area * _FT2_TO_M2
                        new_str  = "{0:.3f} m2".format(area_m2)
                        n_placed += 1
                        if nr.Area < 0.01:
                            n_zero_area += 1
                        doc.Delete(nr.Id)
                    else:
                        new_str = "(NewRoom returned None)"
                except Exception as e:
                    new_str = "ERROR: {0}".format(str(e)[:30])

            print("  ({0:6.2f}, {1:6.2f})         {2:<25} {3}".format(
                x, y, room_str[:24], new_str))

            y += PROBE_STEP_FT
        x += PROBE_STEP_FT

finally:
    t.RollBack()

print("")
print(SEP)
print("Summary:")
print("  Probe points tried  : {0}".format(n_occupied + n_empty))
print("  Already occupied    : {0}  (GetRoomAtPoint returned a room)".format(n_occupied))
print("  Unoccupied points   : {0}  (GetRoomAtPoint returned None)".format(n_empty))
print("  NewRoom placements  : {0}".format(n_placed))
print("  Zero-area placements: {0}  (not enclosed by walls)".format(n_zero_area))
print(SEP)
print("")
print("Interpretation:")
if n_empty == 0:
    print("  ALL points hit an existing room.")
    print("  The BR room boundary likely covers the bathroom area too.")
    print("  Check whether the BR room area seems larger than just the bedroom.")
elif n_zero_area == n_placed and n_placed > 0:
    print("  Unoccupied space found but no walls enclose it.")
    print("  Bathrooms may not have wall boundaries in this model.")
elif n_placed > 0:
    print("  Some enclosed spaces found — check areas above.")
    print("  Area filter in auto_place_bathrooms.py may need adjusting.")
else:
    print("  No enclosed spaces found in unoccupied areas.")
    print("  Bathrooms likely have no wall boundaries in this model.")

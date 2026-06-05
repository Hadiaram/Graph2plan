# inspect_room.py
#
# Revit Python Shell diagnostic — IronPython 2.7 compatible
#
# Prints every parameter on one room so you can decide how to configure
# extract_revit_hotel.py (key assignment, group assignment, unit type, etc.)
#
# Usage
# -----
#   Option A (recommended): select a room in the Revit canvas first,
#                           then run this script — it inspects that room.
#   Option B:               select nothing — it picks the first placed room.

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

doc   = __revit__.ActiveUIDocument.Document
uidoc = __revit__.ActiveUIDocument

# ---------------------------------------------------------------------------
# Pick a room — selection first, otherwise first placed room
# ---------------------------------------------------------------------------

room = None

sel_ids = uidoc.Selection.GetElementIds()
for eid in sel_ids:
    elem = doc.GetElement(eid)
    if elem is None:
        continue
    cat = elem.Category
    if cat is None:
        continue
    if cat.Id == ElementId(BuiltInCategory.OST_Rooms):
        room = elem
        break

if room is None:
    all_rooms = (FilteredElementCollector(doc)
                 .OfCategory(BuiltInCategory.OST_Rooms)
                 .WhereElementIsNotElementType()
                 .ToElements())
    placed = [r for r in all_rooms if r.Area > 0]
    if placed:
        room = placed[0]

if room is None:
    print("No placed rooms found in the document.")
else:
    SEP = "-" * 60

    print(SEP)
    print("ROOM INSPECTION")
    print(SEP)
    print("Element ID : {0}".format(room.Id.IntegerValue))

    # Built-in fields
    num_p  = room.get_Parameter(BuiltInParameter.ROOM_NUMBER)
    name_p = room.get_Parameter(BuiltInParameter.ROOM_NAME)
    num_str  = num_p.AsString()  if (num_p  and num_p.HasValue)  else "(none)"
    name_str = name_p.AsString() if (name_p and name_p.HasValue) else "(none)"
    print("Number     : {0}".format(num_str))
    print("Name       : {0}".format(name_str))
    print("Area (m2)  : {0:.3f}".format(room.Area * 0.092903))
    if room.Location:
        pt = room.Location.Point
        print("Location   : X={0:.2f}  Y={1:.2f}  (Revit feet)".format(pt.X, pt.Y))
    print("")

    # All parameters sorted by name
    print("ALL PARAMETERS:")
    print("{0:<45} {1:<12} {2}".format("Name", "Type", "Value"))
    print("{0:<45} {1:<12} {2}".format("-" * 44, "-" * 11, "-" * 30))

    params = list(room.Parameters)
    params.sort(key=lambda p: p.Definition.Name.lower() if p.Definition else "")

    for p in params:
        try:
            pname = p.Definition.Name if p.Definition else "(no definition)"
            if not p.HasValue:
                val = "(no value)"
            elif p.StorageType == StorageType.String:
                val = p.AsString() or ""
            elif p.StorageType == StorageType.Integer:
                val = str(p.AsInteger())
            elif p.StorageType == StorageType.Double:
                val = str(round(p.AsDouble(), 4))
            elif p.StorageType == StorageType.ElementId:
                val = "id:{0}".format(p.AsElementId().IntegerValue)
            else:
                val = p.AsValueString() or "(unknown type)"
            stype = p.StorageType.ToString().replace("StorageType.", "")
            print("{0:<45} {1:<12} {2}".format(
                pname[:44], stype[:11], str(val)[:60]))
        except Exception as e:
            print("  (error reading parameter: {0})".format(e))

    # Boundary
    print("")
    print("BOUNDARY:")
    bopts = SpatialElementBoundaryOptions()
    bopts.SpatialElementBoundaryLocation = SpatialElementBoundaryLocation.Finish
    loops = room.GetBoundarySegments(bopts)
    if loops and loops.Count > 0:
        for i in range(loops.Count):
            segs = list(loops[i])
            print("  Loop {0}: {1} segment(s)".format(i, len(segs)))
    else:
        print("  (no boundary segments found)")

    # Doors connected to this room
    print("")
    print("CONNECTED DOORS:")
    phase = doc.Phases[doc.Phases.Size - 1]
    doors = (FilteredElementCollector(doc)
             .OfCategory(BuiltInCategory.OST_Doors)
             .WhereElementIsNotElementType()
             .ToElements())
    found = 0
    rid = room.Id.IntegerValue
    for door in doors:
        try:
            fr = door.get_FromRoom(phase)
            to = door.get_ToRoom(phase)
            fr_id = fr.Id.IntegerValue if fr else None
            to_id = to.Id.IntegerValue if to else -1
            if fr_id != rid and to_id != rid:
                continue
            fr_p = fr.get_Parameter(BuiltInParameter.ROOM_NAME) if fr else None
            to_p = to.get_Parameter(BuiltInParameter.ROOM_NAME) if to else None
            fr_name = fr_p.AsString() if (fr_p and fr_p.HasValue) else "outside"
            to_name = to_p.AsString() if (to_p and to_p.HasValue) else "outside"
            print("  Door id={0}  from='{1}'  to='{2}'".format(
                door.Id.IntegerValue, fr_name, to_name))
            found += 1
        except Exception as e:
            print("  (error reading door {0}: {1})".format(door.Id.IntegerValue, e))
    if found == 0:
        print("  (none — check that doors are placed and phase is correct)")

    # Grid lines — find the enclosing bay and all rooms inside it
    print("")
    print("GRID BAY:")
    if room.Location is None:
        print("  (room has no location point)")
    else:
        pt = room.Location.Point

        grids = (FilteredElementCollector(doc)
                 .OfCategory(BuiltInCategory.OST_Grids)
                 .WhereElementIsNotElementType()
                 .ToElements())

        # Split grids into horizontal (run in X) and vertical (run in Y).
        # For each, record (coord, name) where coord is the fixed axis value.
        h_above = []   # horizontal grids with Y > pt.Y  (north of room)
        h_below = []   # horizontal grids with Y < pt.Y  (south of room)
        v_east  = []   # vertical grids with X > pt.X    (east of room)
        v_west  = []   # vertical grids with X < pt.X    (west of room)

        for grid in grids:
            try:
                curve = grid.Curve
                sp = curve.GetEndPoint(0)
                ep = curve.GetEndPoint(1)
                dx = abs(ep.X - sp.X)
                dy = abs(ep.Y - sp.Y)
                name = grid.Name
                if dx >= dy:
                    # Horizontal grid — representative Y is average of endpoints
                    gy = (sp.Y + ep.Y) / 2.0
                    if gy > pt.Y:
                        h_above.append((gy - pt.Y, name, gy))
                    elif gy < pt.Y:
                        h_below.append((pt.Y - gy, name, gy))
                else:
                    # Vertical grid — representative X is average of endpoints
                    gx = (sp.X + ep.X) / 2.0
                    if gx > pt.X:
                        v_east.append((gx - pt.X, name, gx))
                    elif gx < pt.X:
                        v_west.append((pt.X - gx, name, gx))
            except Exception as e:
                print("  (error reading grid: {0})".format(e))

        # Closest enclosing grid on each side
        h_above.sort(key=lambda t: t[0])
        h_below.sort(key=lambda t: t[0])
        v_east.sort(key=lambda t:  t[0])
        v_west.sort(key=lambda t:  t[0])

        enc_north = h_above[0] if h_above else None
        enc_south = h_below[0] if h_below else None
        enc_east  = v_east[0]  if v_east  else None
        enc_west  = v_west[0]  if v_west  else None

        print("  Room centre: X={0:.2f}, Y={1:.2f} (Revit ft)".format(pt.X, pt.Y))
        print("")
        print("  Enclosing grid lines:")
        print("    North : {0}".format("'{0}'  ({1:.2f} ft away)".format(
            enc_north[1], enc_north[0]) if enc_north else "(none found)"))
        print("    South : {0}".format("'{0}'  ({1:.2f} ft away)".format(
            enc_south[1], enc_south[0]) if enc_south else "(none found)"))
        print("    East  : {0}".format("'{0}'  ({1:.2f} ft away)".format(
            enc_east[1],  enc_east[0])  if enc_east  else "(none found)"))
        print("    West  : {0}".format("'{0}'  ({1:.2f} ft away)".format(
            enc_west[1],  enc_west[0])  if enc_west  else "(none found)"))

        # Suggested key ID
        h_name = "{0}-{1}".format(enc_south[1], enc_north[1]) if (enc_south and enc_north) else "?"
        v_name = "{0}-{1}".format(enc_west[1],  enc_east[1])  if (enc_west  and enc_east)  else "?"
        key_id = "{0} / {1}".format(h_name, v_name)
        print("")
        print("  Suggested Key ID: '{0}'".format(key_id))

        # Find all other rooms inside the same bay
        y_min = enc_south[2] if enc_south else None
        y_max = enc_north[2] if enc_north else None
        x_min = enc_west[2]  if enc_west  else None
        x_max = enc_east[2]  if enc_east  else None

        print("")
        print("  ROOMS IN THE SAME BAY:")
        if None in (y_min, y_max, x_min, x_max):
            print("    (cannot determine bay — one or more enclosing grids missing)")
        else:
            all_rooms = (FilteredElementCollector(doc)
                         .OfCategory(BuiltInCategory.OST_Rooms)
                         .WhereElementIsNotElementType()
                         .ToElements())
            bay_rooms = []
            for r in all_rooms:
                if r.Area <= 0 or r.Location is None:
                    continue
                rpt = r.Location.Point
                if x_min < rpt.X < x_max and y_min < rpt.Y < y_max:
                    rn_p = r.get_Parameter(BuiltInParameter.ROOM_NAME)
                    rnum_p = r.get_Parameter(BuiltInParameter.ROOM_NUMBER)
                    rn   = rn_p.AsString()   if (rn_p   and rn_p.HasValue)   else "Unknown"
                    rnum = rnum_p.AsString() if (rnum_p and rnum_p.HasValue) else ""
                    marker = " ← this room" if r.Id.IntegerValue == room.Id.IntegerValue else ""
                    bay_rooms.append((rnum, rn, r.Id.IntegerValue, marker))

            if not bay_rooms:
                print("    (no placed rooms found in this bay)")
            else:
                print("    {0:<12} {1:<30} {2}".format("Number", "Name", "Element ID"))
                print("    {0:<12} {1:<30} {2}".format("-" * 11, "-" * 29, "-" * 10))
                for rnum, rn, eid, marker in sorted(bay_rooms):
                    print("    {0:<12} {1:<30} {2}{3}".format(
                        rnum[:11], rn[:29], eid, marker))

    print(SEP)

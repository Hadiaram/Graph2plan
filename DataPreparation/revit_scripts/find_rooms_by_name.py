# find_rooms_by_name.py
#
# Revit Python Shell diagnostic — IronPython 2.7 compatible
#
# Finds all rooms matching a name, determines which grid bay each sits in,
# and groups them — confirming whether rooms share a structural key.

# =============================================================================
# CONFIGURATION
# =============================================================================

SEARCH_NAME   = ""     # Room name to search for (case-insensitive substring)
EXACT_MATCH   = False    # True = exact match only, False = substring match

# =============================================================================

import clr
clr.AddReference('RevitAPI')
clr.AddReference('RevitAPIUI')

from Autodesk.Revit.DB import (
    FilteredElementCollector,
    BuiltInCategory,
    BuiltInParameter,
)

doc = __revit__.ActiveUIDocument.Document

SEP  = "=" * 60
SEP2 = "-" * 60

# ---------------------------------------------------------------------------
# Grid helpers
# ---------------------------------------------------------------------------

def _load_grids():
    """Return two lists: h_grids [(gy, name)], v_grids [(gx, name)]."""
    raw = (FilteredElementCollector(doc)
           .OfCategory(BuiltInCategory.OST_Grids)
           .WhereElementIsNotElementType()
           .ToElements())
    h_grids = []
    v_grids = []
    for grid in raw:
        try:
            curve = grid.Curve
            sp = curve.GetEndPoint(0)
            ep = curve.GetEndPoint(1)
            dx = abs(ep.X - sp.X)
            dy = abs(ep.Y - sp.Y)
            name = grid.Name
            if dx >= dy:
                h_grids.append(((sp.Y + ep.Y) / 2.0, name))
            else:
                v_grids.append(((sp.X + ep.X) / 2.0, name))
        except Exception:
            pass
    return h_grids, v_grids


def _enclosing_bay(pt, h_grids, v_grids):
    """
    Return (south_name, north_name, west_name, east_name) for the grid bay
    that contains pt, or None for any side that has no enclosing grid.
    """
    above = [(gy, n) for gy, n in h_grids if gy > pt.Y]
    below = [(gy, n) for gy, n in h_grids if gy < pt.Y]
    east  = [(gx, n) for gx, n in v_grids if gx > pt.X]
    west  = [(gx, n) for gx, n in v_grids if gx < pt.X]

    above.sort(key=lambda t: t[0])
    below.sort(key=lambda t: t[0], reverse=True)
    east.sort(key=lambda t:  t[0])
    west.sort(key=lambda t:  t[0], reverse=True)

    north_name = above[0][1] if above else None
    south_name = below[0][1] if below else None
    east_name  = east[0][1]  if east  else None
    west_name  = west[0][1]  if west  else None

    return south_name, north_name, west_name, east_name


def _bay_key(south, north, west, east):
    """Stable string key for a bay — used to group rooms."""
    return "{0}|{1}|{2}|{3}".format(
        south or "?", north or "?", west or "?", east or "?")


def _bay_label(south, north, west, east):
    h = "{0}-{1}".format(south or "?", north or "?")
    v = "{0}-{1}".format(west  or "?", east  or "?")
    return "{0} / {1}".format(h, v)


# ---------------------------------------------------------------------------
# Room helpers
# ---------------------------------------------------------------------------

def _rname(room):
    p = room.get_Parameter(BuiltInParameter.ROOM_NAME)
    return (p.AsString() or "").strip() if (p and p.HasValue) else ""


def _rnum(room):
    p = room.get_Parameter(BuiltInParameter.ROOM_NUMBER)
    return (p.AsString() or "").strip() if (p and p.HasValue) else ""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

print(SEP)
print("Room search: '{0}' ({1})".format(
    SEARCH_NAME, "exact" if EXACT_MATCH else "substring"))
print(SEP)

# Collect all placed rooms
all_rooms = (FilteredElementCollector(doc)
             .OfCategory(BuiltInCategory.OST_Rooms)
             .WhereElementIsNotElementType()
             .ToElements())
placed = [r for r in all_rooms if r.Area > 0]
print("Total placed rooms in document: {0}".format(len(placed)))

# Filter by name
search_lower = SEARCH_NAME.lower()
if EXACT_MATCH:
    matches = [r for r in placed if _rname(r).lower() == search_lower]
else:
    matches = [r for r in placed if search_lower in _rname(r).lower()]

print("Rooms matching '{0}': {1}".format(SEARCH_NAME, len(matches)))

if not matches:
    print("\n  No rooms found. Check SEARCH_NAME spelling.")
else:
    h_grids, v_grids = _load_grids()
    print("Grid lines loaded: {0} horizontal, {1} vertical".format(
        len(h_grids), len(v_grids)))
    print("")

    # Assign each match to a bay
    from collections import defaultdict
    bay_groups = defaultdict(list)   # bay_key → list of (room, bay_names)

    for r in matches:
        if r.Location is None:
            print("  [WARN] Room '{0}' has no location — skipped".format(_rname(r)))
            continue
        pt = r.Location.Point
        south, north, west, east = _enclosing_bay(pt, h_grids, v_grids)
        bk = _bay_key(south, north, west, east)
        bay_groups[bk].append((r, south, north, west, east))

    # Report
    for bk, entries in sorted(bay_groups.items()):
        _, south, north, west, east = entries[0]
        label = _bay_label(south, north, west, east)
        print("BAY: {0}  ({1} matching room(s))".format(label, len(entries)))
        print(SEP2)
        print("  {0:<12} {1:<30} {2:<12} {3}".format(
            "Number", "Name", "Area (m2)", "Element ID"))
        print("  {0:<12} {1:<30} {2:<12} {3}".format(
            "-"*11, "-"*29, "-"*11, "-"*10))
        for r, s, n, w, e in sorted(entries, key=lambda t: _rnum(t[0])):
            print("  {0:<12} {1:<30} {2:<12} {3}".format(
                _rnum(r)[:11],
                _rname(r)[:29],
                str(round(r.Area * 0.092903, 2)),
                r.Id.IntegerValue))

        # Also list ALL rooms in this bay (not just matches)
        print("")
        print("  All rooms in this bay:")
        if None in (south, north, west, east):
            print("    (incomplete bay — cannot determine bounds)")
        else:
            # Find the Y and X bounds from the grid coords
            y_min = next((gy for gy, nm in h_grids if nm == south), None)
            y_max = next((gy for gy, nm in h_grids if nm == north), None)
            x_min = next((gx for gx, nm in v_grids if nm == west),  None)
            x_max = next((gx for gx, nm in v_grids if nm == east),  None)

            if None in (y_min, y_max, x_min, x_max):
                print("    (could not resolve grid coordinates)")
            else:
                bay_all = []
                for r2 in placed:
                    if r2.Location is None:
                        continue
                    rpt = r2.Location.Point
                    if x_min < rpt.X < x_max and y_min < rpt.Y < y_max:
                        bay_all.append(r2)

                if not bay_all:
                    print("    (none found)")
                else:
                    print("  {0:<12} {1:<30} {2}".format(
                        "Number", "Name", "Element ID"))
                    print("  {0:<12} {1:<30} {2}".format(
                        "-"*11, "-"*29, "-"*10))
                    for r2 in sorted(bay_all, key=lambda r: _rnum(r)):
                        marker = " *" if search_lower in _rname(r2).lower() else ""
                        print("  {0:<12} {1:<30} {2}{3}".format(
                            _rnum(r2)[:11],
                            _rname(r2)[:29],
                            r2.Id.IntegerValue,
                            marker))
                    print("  (* = matches search)")
        print("")

    print(SEP)
    print("Summary: {0} '{1}' room(s) across {2} bay(s)".format(
        len(matches), SEARCH_NAME, len(bay_groups)))
    print(SEP)

# auto_place_bathrooms.py
#
# Revit Python Shell — IronPython 3 compatible
#
# Automatically places Bathroom and WC room objects in each hotel key bay.
# Works key-by-key so counts are validated against the unit type.
#
# Rules (hardcoded from hotel program):
#   BR              → 1 Bathroom per bedroom (2 in a standard double-BR key)
#   SUITE           → 1 Bathroom + 1 WC
#   EXECUTIVE SUITE → 2 Bathroom + 1 WC
#
# Usage
# -----
#   1. Set DRY_RUN = True first — reviews what would be placed without saving
#   2. Check the output for warnings
#   3. Set DRY_RUN = False and re-run to commit

# =============================================================================
# CONFIGURATION
# =============================================================================

BATHROOM_NAME = "Bathroom"
WC_NAME       = "WC"

# Area ranges (m²) used to classify found enclosed spaces
BATHROOM_AREA_MIN = 4.0
BATHROOM_AREA_MAX = 10.0
WC_AREA_MIN       = 1.5
WC_AREA_MAX       = 4.0

# Probe grid spacing in Revit feet (~0.46 m) — smaller = more thorough, slower
PROBE_STEP_FT = 1.5

# True = probe + report but roll back all changes (safe preview)
# False = probe + place + commit
DRY_RUN = True

# =============================================================================

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
from collections import defaultdict

doc = __revit__.ActiveUIDocument.Document

SEP  = "=" * 60
SEP2 = "-" * 40

_FT2_TO_M2 = 0.092903
_M2_TO_FT2 = 10.7639

UNIT_NAMES = {'BR', 'SUITE', 'EXECUTIVE SUITE'}

# ---------------------------------------------------------------------------
# Grid helpers  (all coordinates in Revit feet)
# ---------------------------------------------------------------------------

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
            name = grid.Name
            if abs(ep.X - sp.X) >= abs(ep.Y - sp.Y):
                h.append(((sp.Y + ep.Y) / 2.0, name))
            else:
                v.append(((sp.X + ep.X) / 2.0, name))
        except Exception:
            pass
    return h, v


def _bay_bounds(pt, h_grids, v_grids):
    """Return (x_min, x_max, y_min, y_max) in feet, or None if incomplete."""
    above = sorted([(gy, n) for gy, n in h_grids if gy > pt.Y], key=lambda t: t[0])
    below = sorted([(gy, n) for gy, n in h_grids if gy < pt.Y], key=lambda t: t[0], reverse=True)
    east  = sorted([(gx, n) for gx, n in v_grids if gx > pt.X], key=lambda t: t[0])
    west  = sorted([(gx, n) for gx, n in v_grids if gx < pt.X], key=lambda t: t[0], reverse=True)
    if not (above and below and east and west):
        return None
    return (west[0][0], east[0][0], below[0][0], above[0][0])


def _bay_key(pt, h_grids, v_grids):
    above = sorted([(gy, n) for gy, n in h_grids if gy > pt.Y], key=lambda t: t[0])
    below = sorted([(gy, n) for gy, n in h_grids if gy < pt.Y], key=lambda t: t[0], reverse=True)
    east  = sorted([(gx, n) for gx, n in v_grids if gx > pt.X], key=lambda t: t[0])
    west  = sorted([(gx, n) for gx, n in v_grids if gx < pt.X], key=lambda t: t[0], reverse=True)
    s = below[0][1] if below else "?"
    n = above[0][1] if above else "?"
    w = west[0][1]  if west  else "?"
    e = east[0][1]  if east  else "?"
    return "{0}-{1}/{2}-{3}".format(s, n, w, e)


# ---------------------------------------------------------------------------
# Room helpers
# ---------------------------------------------------------------------------

def _rname(room):
    p = room.get_Parameter(BuiltInParameter.ROOM_NAME)
    return (p.AsString() or "").strip().upper() if (p and p.HasValue) else ""


def _set_name(room, name):
    p = room.get_Parameter(BuiltInParameter.ROOM_NAME)
    if p and not p.IsReadOnly:
        p.Set(name)


def _expected_counts(rooms_in_bay):
    """
    Return (n_bathrooms, n_wcs) for the bay based on its room names.
    BR: 1 bathroom per bedroom.
    SUITE: 1 bathroom + 1 WC.
    EXECUTIVE SUITE: 2 bathrooms + 1 WC.
    """
    names = [_rname(r) for r in rooms_in_bay]
    if 'EXECUTIVE SUITE' in names:
        return 2, 1
    if 'SUITE' in names:
        return 1, 1
    # BR — one bathroom per bedroom in the bay
    n_br = sum(1 for n in names if n == 'BR')
    return n_br, 0


# ---------------------------------------------------------------------------
# Probe bay for unoccupied enclosed spaces
# ---------------------------------------------------------------------------

def _probe_bay(bounds, level):
    """
    Scan the bay on a grid. Place a room at each unoccupied point and keep
    those whose area falls within the bathroom/WC size range.
    Returns list of (area_m2, room_element).
    """
    x_min, x_max, y_min, y_max = bounds
    elev        = level.Elevation + 1.0
    min_ft2     = WC_AREA_MIN       * _M2_TO_FT2
    max_ft2     = (BATHROOM_AREA_MAX + 2.0) * _M2_TO_FT2   # slight tolerance

    found     = []
    claimed   = set()   # grid cells already inside a room we placed

    x = x_min + PROBE_STEP_FT
    while x < x_max:
        y = y_min + PROBE_STEP_FT
        while y < y_max:
            cell = (int(x / PROBE_STEP_FT), int(y / PROBE_STEP_FT))
            if cell not in claimed:
                pt3d = XYZ(x, y, elev)
                if doc.GetRoomAtPoint(pt3d) is None:
                    try:
                        r = doc.Create.NewRoom(level, UV(x, y))
                        if r is not None:
                            a = r.Area
                            if min_ft2 <= a <= max_ft2:
                                found.append((a * _FT2_TO_M2, r))
                                # mark bounding box cells as claimed
                                try:
                                    bb = r.get_BoundingBox(None)
                                    if bb:
                                        cx = bb.Min.X
                                        while cx <= bb.Max.X + PROBE_STEP_FT:
                                            cy = bb.Min.Y
                                            while cy <= bb.Max.Y + PROBE_STEP_FT:
                                                claimed.add((
                                                    int(cx / PROBE_STEP_FT),
                                                    int(cy / PROBE_STEP_FT)))
                                                cy += PROBE_STEP_FT
                                            cx += PROBE_STEP_FT
                                except Exception:
                                    pass
                            else:
                                doc.Delete(r.Id)
                    except Exception:
                        pass
            y += PROBE_STEP_FT
        x += PROBE_STEP_FT

    return found


# ---------------------------------------------------------------------------
# Assign names to found spaces
# ---------------------------------------------------------------------------

def _assign(candidates, n_bathrooms, n_wcs):
    """
    Sort candidates by area descending.
    Largest n_bathrooms → Bathroom, next n_wcs → WC, rest deleted.
    Returns (assigned_list, warnings).
    assigned_list: [(name, area_m2, room)]
    """
    warnings = []
    expected = n_bathrooms + n_wcs
    ordered  = sorted(candidates, key=lambda t: t[0], reverse=True)

    if len(ordered) < expected:
        warnings.append("expected {0} wet room(s), found {1}".format(
            expected, len(ordered)))
    elif len(ordered) > expected:
        warnings.append("expected {0} wet room(s), found {1} — extras deleted".format(
            expected, len(ordered)))

    assigned = []
    for i, (area_m2, room) in enumerate(ordered):
        if i < n_bathrooms:
            _set_name(room, BATHROOM_NAME)
            assigned.append((BATHROOM_NAME, area_m2, room))
        elif i < expected:
            _set_name(room, WC_NAME)
            assigned.append((WC_NAME, area_m2, room))
        else:
            doc.Delete(room.Id)

    return assigned, warnings


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

print(SEP)
print("Hotel bathroom auto-placement")
print("DRY RUN: {0}".format(DRY_RUN))
print(SEP)

t = Transaction(doc, "Auto-place hotel bathrooms")
t.Start()

try:
    h_grids, v_grids = _load_grids()
    print("Grids: {0} horizontal, {1} vertical\n".format(len(h_grids), len(v_grids)))

    placed = [r for r in
              FilteredElementCollector(doc)
              .OfCategory(BuiltInCategory.OST_Rooms)
              .WhereElementIsNotElementType()
              .ToElements()
              if r.Area > 0]

    unit_rooms = [r for r in placed if _rname(r) in UNIT_NAMES]
    print("Hotel unit rooms: {0}\n".format(len(unit_rooms)))

    # Group by bay
    bays = defaultdict(list)
    for r in unit_rooms:
        if r.Location:
            bays[_bay_key(r.Location.Point, h_grids, v_grids)].append(r)

    total_placed   = 0
    all_warnings   = []

    for bk in sorted(bays.keys()):
        bay_rooms          = bays[bk]
        n_bath, n_wc       = _expected_counts(bay_rooms)
        names              = sorted(set(_rname(r) for r in bay_rooms))
        unit_label         = ', '.join(names)

        print(SEP2)
        print("Bay: {0}  [{1}]".format(bk, unit_label))
        print("  Expect: {0} x {1}, {2} x {3}".format(
            n_bath, BATHROOM_NAME, n_wc, WC_NAME))

        bounds = _bay_bounds(bay_rooms[0].Location.Point, h_grids, v_grids)
        if bounds is None:
            msg = "could not determine bay bounds — skipped"
            print("  [WARN] " + msg)
            all_warnings.append("{0}: {1}".format(bk, msg))
            continue

        level = bay_rooms[0].Level
        if level is None:
            msg = "no level found — skipped"
            print("  [WARN] " + msg)
            all_warnings.append("{0}: {1}".format(bk, msg))
            continue

        candidates = _probe_bay(bounds, level)

        if not candidates:
            msg = "no bathroom-sized enclosed spaces found"
            print("  [WARN] " + msg)
            all_warnings.append("{0}: {1}".format(bk, msg))
            continue

        assigned, warnings = _assign(candidates, n_bath, n_wc)

        for name, area_m2, room in assigned:
            print("  + '{0}'  {1:.2f} m2  id={2}".format(
                name, area_m2, room.Id.IntegerValue))
            total_placed += 1

        for w in warnings:
            print("  [WARN] " + w)
            all_warnings.append("{0}: {1}".format(bk, w))

    print("")
    print(SEP)

    if DRY_RUN:
        t.RollBack()
        print("DRY RUN — {0} room(s) would be placed. No changes saved.".format(total_placed))
        print("Set DRY_RUN = False and re-run to apply.")
    else:
        t.Commit()
        print("Committed. {0} room(s) placed.".format(total_placed))

    if all_warnings:
        print("\nWarnings ({0}):".format(len(all_warnings)))
        for w in all_warnings:
            print("  [WARN] {0}".format(w))
    else:
        print("No warnings.")

    print(SEP)

except Exception as e:
    t.RollBack()
    print("ERROR — transaction rolled back: {0}".format(e))
    raise

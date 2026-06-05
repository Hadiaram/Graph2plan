# inspect_blowup_view.py
#
# Revit Python Shell diagnostic — IronPython 3 compatible
#
# Inspects a specific named view and reports what element types are present:
# filled regions, rooms (including unlabelled), floors, and annotations.
#
# Usage
# -----
#   Option A: set VIEW_NAME to the exact name of the view you want.
#   Option B: set VIEW_NAME = "" and double-click INTO the viewport on the
#             sheet first — the active view becomes that specific viewport.

# =============================================================================
# CONFIGURATION
# =============================================================================

VIEW_NAME = ""   # e.g. "Blow Up - Key Type A L03".  Leave "" to use active view.

# =============================================================================

import clr
clr.AddReference('RevitAPI')
clr.AddReference('RevitAPIUI')

from Autodesk.Revit.DB import (
    FilteredElementCollector,
    BuiltInCategory,
    BuiltInParameter,
    StorageType,
    ElementId,
    ViewSheet,
    View,
)
from collections import defaultdict

doc   = __revit__.ActiveUIDocument.Document
uidoc = __revit__.ActiveUIDocument

SEP  = "=" * 60
SEP2 = "-" * 40

# ---------------------------------------------------------------------------
# Resolve which view to inspect
# ---------------------------------------------------------------------------

view_to_inspect = None

if VIEW_NAME:
    all_views = (FilteredElementCollector(doc)
                 .OfClass(View)
                 .ToElements())
    for v in all_views:
        if v.Name == VIEW_NAME:
            view_to_inspect = v
            break
    if view_to_inspect is None:
        print("ERROR: No view named '{0}' found.".format(VIEW_NAME))
        print("Views containing that text:")
        for v in all_views:
            if VIEW_NAME.lower() in v.Name.lower():
                print("  '{0}'".format(v.Name))
        raise SystemExit(1)
    print("Using named view: '{0}'".format(view_to_inspect.Name))

else:
    active_view = uidoc.ActiveView
    if isinstance(active_view, ViewSheet):
        print("A sheet is active: '{0}'".format(active_view.Name))
        print("")
        print("Set VIEW_NAME at the top of this script to one of these,")
        print("or double-click INTO the viewport you want before running:")
        print("")
        for vpid in active_view.GetAllViewports():
            vp = doc.GetElement(vpid)
            if vp is None:
                continue
            linked = doc.GetElement(vp.ViewId)
            if linked is None:
                continue
            print("  VIEW_NAME = \"{0}\"  (type: {1})".format(
                linked.Name, linked.ViewType.ToString()))
        raise SystemExit(1)
    else:
        view_to_inspect = active_view
        print("Using active view: '{0}'  ({1})".format(
            active_view.Name, active_view.ViewType.ToString()))

print("")

# ---------------------------------------------------------------------------
# All elements in the view — count by category
# ---------------------------------------------------------------------------

print(SEP)
print("ELEMENTS IN VIEW: '{0}'".format(view_to_inspect.Name))
print(SEP)

all_elems = list(FilteredElementCollector(doc, view_to_inspect.Id)
                 .WhereElementIsNotElementType()
                 .ToElements())

cat_counts = defaultdict(int)
for elem in all_elems:
    cat = elem.Category
    cat_counts[cat.Name if cat else "(no category)"] += 1

print("Total elements: {0}".format(len(all_elems)))
print("")
print("Count by category:")
for cat_name, count in sorted(cat_counts.items(), key=lambda t: -t[1]):
    print("  {0:<45} {1}".format(cat_name[:44], count))

# ---------------------------------------------------------------------------
# Filled Regions
# ---------------------------------------------------------------------------

print("")
print(SEP2)
print("FILLED REGIONS:")
filled_list = list(FilteredElementCollector(doc, view_to_inspect.Id)
                   .OfCategory(BuiltInCategory.OST_FilledRegion)
                   .WhereElementIsNotElementType()
                   .ToElements())
print("Count: {0}".format(len(filled_list)))

for i, fr in enumerate(filled_list[:15]):
    try:
        fr_type   = doc.GetElement(fr.GetTypeId())
        type_name = fr_type.Name if fr_type else "(unknown type)"
        mark_p    = fr.LookupParameter("Mark")
        comment_p = fr.LookupParameter("Comments")
        mark    = mark_p.AsString()    if (mark_p    and mark_p.HasValue)    else ""
        comment = comment_p.AsString() if (comment_p and comment_p.HasValue) else ""
        print("  [{0}] type='{1}'  mark='{2}'  comments='{3}'".format(
            i, type_name[:35], mark[:20], comment[:20]))
    except Exception as e:
        print("  [{0}] error: {1}".format(i, e))

if len(filled_list) > 15:
    print("  ... and {0} more".format(len(filled_list) - 15))

# ---------------------------------------------------------------------------
# Rooms visible in this view
# ---------------------------------------------------------------------------

print("")
print(SEP2)
print("ROOMS VISIBLE IN THIS VIEW:")
rooms_list = list(FilteredElementCollector(doc, view_to_inspect.Id)
                  .OfCategory(BuiltInCategory.OST_Rooms)
                  .WhereElementIsNotElementType()
                  .ToElements())
print("Count: {0}".format(len(rooms_list)))

for room in rooms_list:
    num_p  = room.get_Parameter(BuiltInParameter.ROOM_NUMBER)
    name_p = room.get_Parameter(BuiltInParameter.ROOM_NAME)
    num  = num_p.AsString()  if (num_p  and num_p.HasValue)  else "(no number)"
    name = name_p.AsString() if (name_p and name_p.HasValue) else "(no name)"
    area = round(room.Area * 0.092903, 2)
    print("  #{0:<10} '{1}'  {2} m2  id={3}".format(
        num[:9], name[:25], area, room.Id.IntegerValue))

# ---------------------------------------------------------------------------
# Room tags
# ---------------------------------------------------------------------------

print("")
print(SEP2)
print("ROOM TAGS IN THIS VIEW:")
tags_list = list(FilteredElementCollector(doc, view_to_inspect.Id)
                 .OfCategory(BuiltInCategory.OST_RoomTags)
                 .WhereElementIsNotElementType()
                 .ToElements())
print("Count: {0}".format(len(tags_list)))

for tag in tags_list[:20]:
    try:
        tagged_room = tag.Room
        if tagged_room:
            num_p  = tagged_room.get_Parameter(BuiltInParameter.ROOM_NUMBER)
            name_p = tagged_room.get_Parameter(BuiltInParameter.ROOM_NAME)
            num  = num_p.AsString()  if (num_p  and num_p.HasValue)  else ""
            name = name_p.AsString() if (name_p and name_p.HasValue) else "(unnamed)"
            print("  Tag id={0}  room #{1} '{2}'".format(
                tag.Id.IntegerValue, num, name))
        else:
            print("  Tag id={0}  (no linked room)".format(tag.Id.IntegerValue))
    except Exception as e:
        print("  Tag id={0}  error: {1}".format(tag.Id.IntegerValue, e))

# ---------------------------------------------------------------------------
# Floors
# ---------------------------------------------------------------------------

print("")
print(SEP2)
print("FLOORS IN THIS VIEW:")
floors_list = list(FilteredElementCollector(doc, view_to_inspect.Id)
                   .OfCategory(BuiltInCategory.OST_Floors)
                   .WhereElementIsNotElementType()
                   .ToElements())
print("Count: {0}".format(len(floors_list)))

for fl in floors_list[:10]:
    try:
        ftype     = doc.GetElement(fl.GetTypeId())
        type_name = ftype.Name if ftype else "(unknown)"
        mark_p    = fl.LookupParameter("Mark")
        mark      = mark_p.AsString() if (mark_p and mark_p.HasValue) else ""
        print("  Floor id={0}  type='{1}'  mark='{2}'".format(
            fl.Id.IntegerValue, type_name[:35], mark[:20]))
    except Exception as e:
        print("  Floor id={0}  error: {1}".format(fl.Id.IntegerValue, e))

print("")
print(SEP)
print("Done. Share the 'Count by category' section to identify")
print("which element type represents the colored room areas.")
print(SEP)

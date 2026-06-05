"""
Stage 1 of the hotel floor-level pipeline.

Reads the flat floor JSON exported from Revit by extract_revit_floor.py,
infers a vocabulary by matching the room names found in the file against
canonical hotel categories, normalises all geometry to 0-255 pixel space,
computes the outer floor boundary and room adjacency graph, then writes a
clean JSON ready for Stage 2 (convert_hotel_floor_to_mat.py).

The vocabulary is inferred automatically from the room names present in each
JSON file, so the script works across hotels that use different naming
conventions ("BEDROOM" vs "BR", "CORRIDOR" vs "CIRCULATION", etc.).
Supply --vocab to override with an explicit name→integer mapping.

Input schema (from extract_revit_floor.py)
------------------------------------------
{
  "star_rating": 4,
  "floor": "L1",
  "rooms": [
    {"id": 0, "name": "BR", "area_m2": 34.25,
     "geometry": [[x,y],...], "number": "20"}
  ],
  "doors": [{"from": 0, "to": 1}]  # to=-1 means corridor/outside
}

Output schema (written to out_dir/<floor_id>.json)
--------------------------------------------------
{
  "source_file":  "<path>",
  "floor_id":     "<str>",
  "star_rating":  4,
  "floor":        "L1",
  "n_rooms":      42,
  "canvas_size":  256,
  "vocab_used":   {"BR": 0, "CIRCULATION": 3, ...},
  "boundary":     [[x,y,dir,isNew], ...],
  "rooms": [
    {"id":0, "name":"BR", "rtype":0,
     "box":[x0,y0,x1,y1], "polygon":[[x,y],...]}
  ],
  "edges": [[from, to, etype], ...]
}

Usage
-----
  python extract_hotel_floor.py floor_L1.json
  python extract_hotel_floor.py bim_exports/ --out processed/
  python extract_hotel_floor.py floor_L1.json --vocab override.json

After processing, run:
  python convert_hotel_floor_to_mat.py processed/ --out data/data_hotel.mat
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

try:
    from shapely.geometry import Polygon, MultiPolygon
    from shapely.ops import unary_union
    HAS_SHAPELY = True
except ImportError:
    HAS_SHAPELY = False
    print("[WARNING] shapely not installed — outer_boundary will be a bounding box.\n"
          "          Install with:  pip install shapely")

# =============================================================================
# Canonical hotel room categories
# =============================================================================
# Each entry: (type_int, canonical_name, [match_keywords])
#
# Rules:
#   - Matching is case-insensitive substring: keyword must appear inside the
#     room name (e.g. "SUITE" matches "JUNIOR SUITE", "SUITE DELUXE", etc.)
#   - Entries with longer keywords are listed first so more specific names
#     (EXECUTIVE SUITE, LIFT LOBBY) win before shorter ones (SUITE, LIFT).
#   - First matching entry wins — order of the list matters.
#
# To add a new project-specific synonym, add it to the keywords list of the
# appropriate canonical type.

CANONICAL_TYPES = [
    (2, "EXECUTIVE SUITE", ["EXECUTIVE SUITE", "EXEC SUITE", "PRESIDENTIAL SUITE",
                             "PRESIDENTIAL"]),
    (5, "LIFT LOBBY",      ["LIFT LOBBY", "ELEVATOR LOBBY", "LIFT HALL",
                             "ELEVATOR HALL", "LIFT CORE LOBBY"]),
    (1, "SUITE",           ["SUITE", "JUNIOR SUITE", "DELUXE SUITE"]),
    (4, "LIFT",            ["LIFT", "ELEVATOR"]),
    (0, "BR",              ["BR", "BEDROOM", "BED ROOM", "GUEST ROOM",
                             "STANDARD ROOM", "DELUXE ROOM", "CLASSIC ROOM",
                             "CLASSIC"]),
    (3, "CIRCULATION",     ["CIRCULATION", "CORRIDOR", "HALLWAY", "PASSAGE",
                             "HALL", "LANDING", "LOBBY"]),
    (6, "STAIRS",          ["STAIRCASE", "STAIRWELL", "STAIRS", "STAIR"]),
    (7, "MEP",             ["MEP", "MECHANICAL", "ELECTRICAL", "PLANT ROOM",
                             "SERVICES", "SERVICE ROOM", "UTILITY"]),
]

# Canonical names whose room centroid is used to locate the floor "entrance"
# for boundary reordering.  Listed in priority order; first match found wins.
ENTRANCE_CANONICAL = ["LIFT LOBBY", "CIRCULATION", "LIFT"]

# Canvas size in pixels (coordinates range 0 … CANVAS_SIZE-1).
# Must match the model's expected image_size (128 px uses a 256-pixel canvas).
CANVAS_SIZE = 256
MARGIN = 8  # pixels of whitespace around the floor boundary


# =============================================================================
# Vocabulary inference
# =============================================================================

def _match_canonical(name):
    """
    Match a room name string against CANONICAL_TYPES.
    Returns (type_int, canonical_name) for the first match, or (None, None).
    """
    upper = name.upper().strip()
    for type_int, canonical, keywords in CANONICAL_TYPES:
        for kw in keywords:
            if kw in upper:
                return type_int, canonical
    return None, None


def _infer_vocab(raw_rooms):
    """
    Scan room names in raw_rooms and build a complete vocab — no room is ever
    skipped.

    1. Names that match a canonical keyword → assigned that canonical type int.
    2. Names with no keyword match → assigned a new type int (starting after
       the highest canonical int already used), in the order first seen.

    Returns:
      vocab    : dict  name_upper → type_int
      name_map : dict  name_upper → canonical_name (or original name if new)
      new_types: list of (name, type_int) for names that got auto-assigned IDs
    """
    vocab    = {}
    name_map = {}
    new_types = []

    # Collect all unique names in file order
    seen_order = []
    for r in raw_rooms:
        name  = (r.get("name") or "").strip()
        upper = name.upper()
        if upper and upper not in vocab and upper not in {n for n, _ in new_types}:
            seen_order.append((name, upper))

    # First pass: keyword matches
    for name, upper in seen_order:
        type_int, canonical = _match_canonical(name)
        if type_int is not None:
            vocab[upper]    = type_int
            name_map[upper] = canonical

    # Second pass: anything still unmatched gets the next available int
    next_id = max(vocab.values()) + 1 if vocab else 0
    for name, upper in seen_order:
        if upper not in vocab:
            vocab[upper]    = next_id
            name_map[upper] = name          # canonical name = whatever Revit used
            new_types.append((name, next_id))
            next_id += 1

    return vocab, name_map, new_types

# Two room polygons are considered spatially adjacent when their Shapely
# geometries are within this tolerance (in normalised pixel units).
ADJACENCY_TOL = 1.5

# Edge type codes (spatial relationship from room A to room B)
_ETYPE = {
    "left-above":  0,
    "left-below":  1,
    "left-of":     2,
    "above":       3,
    "inside":      4,
    "surrounding": 5,
    "below":       6,
    "right-of":    7,
    "right-above": 8,
    "right-below": 9,
}


# =============================================================================
# Geometry helpers
# =============================================================================

def _scale_transform(all_polys):
    """
    Return (offset_x, offset_y, scale) that maps real-world coordinates into
    the range [MARGIN, CANVAS_SIZE - MARGIN].
    """
    pts = [pt for poly in all_polys for pt in poly]
    if not pts:
        return 0.0, 0.0, 1.0
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    w = max(xs) - min(xs)
    h = max(ys) - min(ys)
    usable = CANVAS_SIZE - 2 * MARGIN
    scale = usable / max(w, h) if max(w, h) > 0 else 1.0
    # Centre on canvas
    cx = (min(xs) + max(xs)) / 2.0
    cy = (min(ys) + max(ys)) / 2.0
    offset_x = CANVAS_SIZE / 2.0 - cx * scale
    offset_y = CANVAS_SIZE / 2.0 - cy * scale
    return offset_x, offset_y, scale


def _apply(pts, offset_x, offset_y, scale):
    return [[round(x * scale + offset_x, 4),
             round(y * scale + offset_y, 4)] for x, y in pts]


def _box(poly):
    """Return [x0, y0, x1, y1] from a list of [x, y] points."""
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return [round(min(xs), 4), round(min(ys), 4),
            round(max(xs), 4), round(max(ys), 4)]


def _centroid(poly):
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return sum(xs) / len(xs), sum(ys) / len(ys)


# =============================================================================
# Boundary helpers
# =============================================================================

def _outer_boundary(polygons):
    """
    Return the outer boundary of the union of all room polygons as a list of
    [x, y] points, using Shapely when available.
    """
    if HAS_SHAPELY and polygons:
        polys = []
        for pts in polygons:
            if len(pts) >= 3:
                p = Polygon(pts)
                if not p.is_valid:
                    p = p.buffer(0)
                if p.is_valid and not p.is_empty:
                    polys.append(p)
        if polys:
            union = unary_union(polys)
            if isinstance(union, MultiPolygon):
                union = max(union.geoms, key=lambda g: g.area)
            coords = list(union.exterior.coords)[:-1]  # drop closing duplicate
            return [[round(x, 4), round(y, 4)] for x, y in coords]

    # Fallback: bounding box
    pts = [pt for poly in polygons for pt in poly]
    if not pts:
        return []
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def _direction_codes(coords):
    """
    Compute Graph2plan direction code for each boundary vertex.
      0 = right  (+X)
      1 = up     (+Y)
      2 = left   (-X)
      3 = down   (-Y)
    The direction is from vertex i toward vertex i+1 (cyclic).
    """
    n = len(coords)
    dirs = np.zeros(n, dtype=int)
    for i in range(n):
        j = (i + 1) % n
        dx = coords[j][0] - coords[i][0]
        dy = coords[j][1] - coords[i][1]
        if abs(dx) >= abs(dy):
            dirs[i] = 0 if dx >= 0 else 2
        else:
            dirs[i] = 1 if dy >= 0 else 3
    return dirs


def _reorder_boundary(coords, entrance_point):
    """
    Rotate the coords list so that the vertex closest to entrance_point
    becomes index 0, putting the entrance at the start of the boundary.
    """
    if entrance_point is None or not coords:
        return coords
    ex, ey = entrance_point
    dists = [(i, (c[0] - ex) ** 2 + (c[1] - ey) ** 2)
             for i, c in enumerate(coords)]
    start = min(dists, key=lambda t: t[1])[0]
    return coords[start:] + coords[:start]


def _build_boundary(polygons, entrance_pt):
    """
    Build the (N×4) boundary array: [x, y, dir, isNew].
    """
    coords = _outer_boundary(polygons)
    if not coords:
        return []
    coords = _reorder_boundary(coords, entrance_pt)
    dirs = _direction_codes(coords)
    boundary = []
    for i, (pt, d) in enumerate(zip(coords, dirs)):
        boundary.append([pt[0], pt[1], int(d), 0])
    return boundary


# =============================================================================
# Adjacency / edge helpers
# =============================================================================

def _spatial_etype(cx_a, cy_a, cx_b, cy_b):
    """
    Classify the spatial relationship of room B relative to room A.
    Returns one of the _ETYPE integer codes.
    """
    dx = cx_b - cx_a
    dy = cy_b - cy_a
    horiz = abs(dx)
    vert  = abs(dy)
    if horiz < vert * 0.5:        # mostly vertical
        return _ETYPE["above"] if dy > 0 else _ETYPE["below"]
    if vert < horiz * 0.5:        # mostly horizontal
        return _ETYPE["right-of"] if dx > 0 else _ETYPE["left-of"]
    # diagonal
    if dx > 0 and dy > 0:
        return _ETYPE["right-above"]
    if dx > 0 and dy < 0:
        return _ETYPE["right-below"]
    if dx < 0 and dy > 0:
        return _ETYPE["left-above"]
    return _ETYPE["left-below"]


def _build_edges(rooms, door_list):
    """
    Build the edge list as [[from, to, etype], ...].

    1. Door edges  → use the door connectivity from the JSON (etype=1).
    2. Spatial adjacency edges → add edges for rooms whose Shapely polygons
       touch (share a boundary), excluding pairs already connected by a door
       and pairs where to=-1 (exterior) (etype derived from spatial position).

    Door-to-exterior connections (to=-1) are included with etype=1.
    """
    n = len(rooms)
    door_pairs = set()
    edges = []

    # Door edges
    for d in door_list:
        f, t = d["from"], d["to"]
        if f < 0 or f >= n:
            continue
        key = (min(f, t), max(f, t)) if t >= 0 else (f, -1)
        if key not in door_pairs:
            door_pairs.add(key)
            etype = 1  # door connection
            edges.append([f, t, etype])

    if not HAS_SHAPELY:
        return edges

    # Build Shapely polygons for spatial adjacency
    shp = []
    for r in rooms:
        pts = r["polygon"]
        if len(pts) >= 3:
            p = Polygon(pts)
            shp.append(p.buffer(0) if not p.is_valid else p)
        else:
            shp.append(None)

    centroids = []
    for r in rooms:
        cx, cy = _centroid(r["polygon"]) if r["polygon"] else (0, 0)
        centroids.append((cx, cy))

    for i in range(n):
        for j in range(i + 1, n):
            if shp[i] is None or shp[j] is None:
                continue
            key = (i, j)
            if key in door_pairs:
                continue
            try:
                dist = shp[i].distance(shp[j])
            except Exception:
                continue
            if dist <= ADJACENCY_TOL:
                cx_a, cy_a = centroids[i]
                cx_b, cy_b = centroids[j]
                et = _spatial_etype(cx_a, cy_a, cx_b, cy_b)
                edges.append([i, j, et])

    return edges


# =============================================================================
# File processing
# =============================================================================

def _load_json(path):
    encodings = ["utf-8-sig", "utf-16", "cp1252", "latin-1"]
    for enc in encodings:
        try:
            with open(path, "r", encoding=enc) as f:
                return json.load(f)
        except (UnicodeDecodeError, UnicodeError):
            continue
        except json.JSONDecodeError as e:
            print(f"  [ERROR] Not valid JSON ({enc}): {e}")
            return None
        except Exception as e:
            print(f"  [ERROR] Could not read file: {e}")
            return None
    print(f"  [ERROR] Could not decode {path}")
    return None


def process_file(input_path, output_dir, vocab_override=None):
    """
    Process one BIM JSON file.

    vocab_override : dict (name_upper → type_int) or None.
        When None the vocabulary is inferred from the room names in this file.
        When provided it is used as-is (--vocab flag).
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  {input_path.name}")

    data = _load_json(input_path)
    if data is None:
        return None

    star_rating = data.get("star_rating", 0)
    floor_label = data.get("floor", "")
    raw_rooms   = data.get("rooms", [])
    raw_doors   = data.get("doors", [])

    if not raw_rooms:
        print("  [WARN] No rooms found in JSON")
        return None

    # -------------------------------------------------------------------------
    # 1. Build vocabulary for this file
    # -------------------------------------------------------------------------
    if vocab_override is not None:
        vocab    = vocab_override
        name_map = {k: k for k in vocab}   # canonical = whatever the key is
        unmatched_names = []
    else:
        vocab, name_map, new_types = _infer_vocab(raw_rooms)
        # Print the inferred mapping so the user can verify it
        print("  Vocab inferred from room names:")
        for upper, type_int in sorted(vocab.items(), key=lambda kv: kv[1]):
            canonical = name_map.get(upper, upper)
            alias = f"  → canonical '{canonical}'" if canonical != upper else ""
            print(f"    {type_int}  {upper}{alias}")
        if new_types:
            print(f"  [NOTE] {len(new_types)} name(s) had no keyword match — "
                  f"assigned new type IDs:")
            for name, tid in new_types:
                print(f"    {tid}  {name}  (auto-assigned)")
            print("         To group these under a canonical type, add their "
                  "names to CANONICAL_TYPES.")

    # -------------------------------------------------------------------------
    # 2. Apply vocabulary — collect rooms with a known type
    # -------------------------------------------------------------------------
    valid_rooms = []
    for r in raw_rooms:
        name  = (r.get("name") or "").strip()
        rtype = vocab.get(name.upper())
        if rtype is None:
            continue   # only possible when vocab_override is used and name absent
        valid_rooms.append({
            "orig_id": r["id"],
            "name":    name,
            "rtype":   rtype,
            "polygon": r.get("geometry", []),
            "area_m2": r.get("area_m2", 0.0),
            "number":  r.get("number", ""),
        })

    # Filter rooms without usable geometry
    valid_rooms = [r for r in valid_rooms if len(r["polygon"]) >= 3]

    if not valid_rooms:
        print("  [WARN] No rooms with valid geometry after filtering")
        return None

    n = len(valid_rooms)
    print(f"  Rooms extracted: {n}")

    # Build a map from original Revit ID to new local index
    orig_to_local = {r["orig_id"]: i for i, r in enumerate(valid_rooms)}

    # -------------------------------------------------------------------------
    # 3. Normalise coordinates to CANVAS_SIZE pixel space
    # -------------------------------------------------------------------------
    all_polys = [r["polygon"] for r in valid_rooms]
    ox, oy, scale = _scale_transform(all_polys)

    for r in valid_rooms:
        r["polygon"] = _apply(r["polygon"], ox, oy, scale)
        r["box"]     = _box(r["polygon"])

    # -------------------------------------------------------------------------
    # 4. Outer floor boundary
    # -------------------------------------------------------------------------
    # Find entrance point: look for the first room whose canonical name matches
    # an entrance category (LIFT LOBBY → CIRCULATION → LIFT).
    entrance_pt = None
    canonical_by_room = {r["name"].upper(): name_map.get(r["name"].upper(), "")
                         for r in valid_rooms}
    for target_canonical in ENTRANCE_CANONICAL:
        for r in valid_rooms:
            if canonical_by_room.get(r["name"].upper()) == target_canonical:
                entrance_pt = _centroid(r["polygon"])
                break
        if entrance_pt is not None:
            break

    boundary = _build_boundary([r["polygon"] for r in valid_rooms], entrance_pt)

    # -------------------------------------------------------------------------
    # 5. Room adjacency edges
    # -------------------------------------------------------------------------
    remapped_doors = []
    for d in raw_doors:
        f = orig_to_local.get(d.get("from"), -1)
        t_orig = d.get("to", -1)
        t = orig_to_local.get(t_orig, -1) if t_orig != -1 else -1
        if f < 0:
            continue
        remapped_doors.append({"from": f, "to": t})

    edges = _build_edges(valid_rooms, remapped_doors)

    # -------------------------------------------------------------------------
    # 6. Build output record
    # -------------------------------------------------------------------------
    floor_id = f"{star_rating}star_{floor_label}" if floor_label else f"{star_rating}star"

    # Compact vocab summary saved alongside results for traceability
    vocab_used = {name: int(tid) for name, tid in sorted(vocab.items(), key=lambda kv: kv[1])}

    out_rooms = []
    for r in valid_rooms:
        out_rooms.append({
            "id":      orig_to_local[r["orig_id"]],
            "name":    r["name"],
            "rtype":   r["rtype"],
            "number":  r["number"],
            "area_m2": r["area_m2"],
            "box":     r["box"],
            "polygon": r["polygon"],
        })

    record = {
        "source_file":  str(input_path),
        "floor_id":     floor_id,
        "star_rating":  star_rating,
        "floor":        floor_label,
        "n_rooms":      n,
        "canvas_size":  CANVAS_SIZE,
        "vocab_used":   vocab_used,
        "boundary":     boundary,
        "rooms":        out_rooms,
        "edges":        edges,
    }

    out_path = _unique_path(output_dir, floor_id)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)

    type_counts = {}
    for r in out_rooms:
        type_counts[r["name"]] = type_counts.get(r["name"], 0) + 1
    type_str = ", ".join(f"{k}×{v}" for k, v in sorted(type_counts.items()))
    print(f"  → {out_path.name}")
    print(f"    {n} rooms  [{type_str}]")
    print(f"    {len(boundary)} boundary pts  |  {len(edges)} edges")

    return record


def _unique_path(directory, stem):
    p = directory / f"{stem}.json"
    suffix = 1
    while p.exists():
        p = directory / f"{stem}_{suffix}.json"
        suffix += 1
    return p


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Stage 1 (floor level): Process hotel BIM JSON for Graph2plan training.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python extract_hotel_floor.py floor_L1.json\n"
            "  python extract_hotel_floor.py bim_exports/ --out processed/\n"
            "  python extract_hotel_floor.py floor_L1.json --vocab hotel_floor_vocab.json\n"
            "\nAfter processing, run:\n"
            "  python convert_hotel_floor_to_mat.py processed/ --out data/data_hotel.mat"
        ),
    )
    parser.add_argument(
        "input",
        help="BIM JSON file or directory of BIM JSON files from extract_revit_floor.py",
    )
    parser.add_argument(
        "--out", default="processed_floors",
        help="Output directory for processed floor JSONs (default: processed_floors/)",
    )
    parser.add_argument(
        "--vocab", default=None,
        help="Optional JSON file mapping room names to integer types. "
             "If omitted, the built-in hotel vocab is used.",
    )
    args = parser.parse_args()

    # Load explicit vocab override if provided
    vocab_override = None
    if args.vocab:
        try:
            with open(args.vocab, "r", encoding="utf-8") as f:
                vocab_data = json.load(f)
            # Accept either {"BR": 0, ...} or {"vocab": {"BR": 0, ...}}
            raw = vocab_data.get("vocab", vocab_data)
            vocab_override = {k.upper(): v for k, v in raw.items()
                              if isinstance(k, str) and isinstance(v, int)}
            print(f"Using explicit vocab from {args.vocab}: {len(vocab_override)} entries")
            for name, idx in sorted(vocab_override.items(), key=lambda kv: kv[1]):
                print(f"  {idx}  {name}")
            print()
        except Exception as e:
            print(f"[ERROR] Could not load vocab file: {e}")
            sys.exit(1)
    else:
        print("No --vocab supplied — vocabulary will be inferred from each file's room names.")
        print()

    print("Hotel floor extraction — Stage 1")
    print("=" * 40)

    input_path = Path(args.input)
    count = 0

    if input_path.is_dir():
        files = sorted(input_path.glob("*.json"))
        if not files:
            print(f"[ERROR] No JSON files found in {input_path}")
            sys.exit(1)
        print(f"Found {len(files)} file(s) in {input_path}")
        for f in files:
            r = process_file(f, args.out, vocab_override)
            if r is not None:
                count += 1
    elif input_path.is_file():
        r = process_file(input_path, args.out, vocab_override)
        if r is not None:
            count += 1
    else:
        print(f"[ERROR] {input_path} does not exist")
        sys.exit(1)

    print()
    print("=" * 40)
    print(f"Processed {count} floor(s) → {args.out}/")
    print()
    print("Next step:")
    print(f"  python convert_hotel_floor_to_mat.py {args.out}/ --out data/data_hotel.mat")


if __name__ == "__main__":
    main()

"""
Stage 1 of the hotel data pipeline — key extraction.

Reads BIM JSON files exported from Revit and writes one JSON file per
bookable hotel unit.  Room names are preserved as plain strings; no
integer vocabulary is applied at this stage.

Expected input schema
---------------------
{
  "star_rating": 4 | 5,
  "floor": "<label>",            # optional
  "keys": [
    {
      "key_id":            "<string>",
      "unit_type":         "standard" | "deluxe" | "suite",
      "key_configuration": "standard" | "suite",
        # "standard" — two independent bookable rooms sharing one structural module.
        #              Each room must have a "group" field set to "A" or "B".
        # "suite"    — one bookable unit occupying the whole module.
      "rooms": [
        {
          "id":       <int>,
          "name":     "<string>",    # room label exactly as it appears in Revit
          "area_m2":  <float>,
          "geometry": [[x, y], ...], # polygon vertices in real-world coordinates
          "group":    "A" | "B"      # required only when key_configuration == "standard"
        }
      ],
      "doors": [
        {"from": <room_id>, "to": <room_id>}
        # "to": -1  means a door to the corridor / outside the key
      ]
    }
  ]
}

Output per extracted unit  (written to out_dir/<unit_id>.json)
---------------------
{
  "source_file":       "<path>",
  "key_id":            "<string>",
  "unit_id":           "<string>",
  "unit_type":         "<string>",
  "key_configuration": "<string>",
  "star_rating":       <int>,
  "floor":             "<string>",
  "rooms": [
    {"id": <int>, "name": "<string>", "area_m2": <float>, "geometry": [[x,y],...]}
  ],
  "doors":          [{"from": <int>, "to": <int>}],
  "outer_boundary": [[x, y], ...]   # union outline of all room polygons
}

Usage
-----
  python extract_hotel_keys.py floor3.json --out_dir extracted_keys/
  python extract_hotel_keys.py bim_exports/ --out_dir extracted_keys/
"""

import argparse
import json
import sys
from pathlib import Path

try:
    from shapely.geometry import Polygon, MultiPolygon
    from shapely.ops import unary_union
    HAS_SHAPELY = True
except ImportError:
    HAS_SHAPELY = False
    print("[WARNING] shapely not installed — outer_boundary will be a bounding box.\n"
          "          Install with:  pip install shapely")


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def compute_outer_boundary(geometries):
    """
    Return the outer boundary vertices of the union of all room polygons.
    Uses Shapely when available; falls back to axis-aligned bounding box.

    Parameters
    ----------
    geometries : list of list of [x, y]

    Returns
    -------
    list of [x, y]  — vertices of the outer polygon (not closed)
    """
    if not geometries:
        return []

    if HAS_SHAPELY:
        try:
            polys = []
            for geom in geometries:
                if len(geom) >= 3:
                    p = Polygon(geom)
                    if not p.is_valid:
                        p = p.buffer(0)
                    if p.is_valid and not p.is_empty:
                        polys.append(p)
            if polys:
                union = unary_union(polys)
                if isinstance(union, MultiPolygon):
                    union = max(union.geoms, key=lambda g: g.area)
                coords = list(union.exterior.coords)[:-1]  # drop closing duplicate
                return [[round(x, 6), round(y, 6)] for x, y in coords]
        except Exception as e:
            print(f"    [WARN] Shapely union failed ({e}), using bounding box")

    return _bounding_box(geometries)


def _bounding_box(geometries):
    """Axis-aligned bounding box of all geometry points as a 4-vertex polygon."""
    pts = [pt for geom in geometries for pt in geom]
    if not pts:
        return []
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


# ---------------------------------------------------------------------------
# Door helpers
# ---------------------------------------------------------------------------

def remap_doors(doors, id_map):
    """
    Apply id_map to door endpoints.  Doors referencing ids not in the map
    (except -1 for external) are dropped.
    """
    result = []
    for d in doors:
        f = d.get('from')
        t = d.get('to')
        if f not in id_map:
            continue
        mapped_t = id_map.get(t, -1) if t != -1 else -1
        result.append({'from': id_map[f], 'to': mapped_t})
    return result


# ---------------------------------------------------------------------------
# Key extraction
# ---------------------------------------------------------------------------

def extract_key(key_data, star_rating, floor_label, source_file):
    """
    Extract one or two bookable units from a single key definition.

    Returns
    -------
    list of dict  — one output record per bookable unit
    """
    key_id   = str(key_data.get('key_id', 'unknown'))
    unit_type = key_data.get('unit_type', 'unknown')
    key_cfg  = key_data.get('key_configuration', 'suite')
    rooms    = key_data.get('rooms', [])
    doors    = key_data.get('doors', [])

    if not rooms:
        print(f"  [SKIP] key {key_id}: no rooms defined")
        return []

    results = []

    if key_cfg == 'standard':
        # Two independent bookable rooms sharing one structural module.
        # Each room must have a "group" field ("A" or "B").
        for group_label in ('A', 'B'):
            group_rooms = [r for r in rooms if r.get('group') == group_label]
            if not group_rooms:
                print(f"  [WARN] key {key_id}: no rooms for group {group_label} — skipping")
                continue

            id_map = {r['id']: i for i, r in enumerate(group_rooms)}
            norm_rooms = _normalise_rooms(group_rooms, id_map)
            norm_doors = remap_doors(doors, id_map)
            outer = compute_outer_boundary([r['geometry'] for r in norm_rooms])

            results.append(_build_record(
                source_file, key_id, f"{key_id}_{group_label}",
                unit_type, key_cfg, star_rating, floor_label,
                norm_rooms, norm_doors, outer,
            ))

    else:
        # Single bookable unit (deluxe, suite, or any non-split configuration).
        id_map = {r['id']: i for i, r in enumerate(rooms)}
        norm_rooms = _normalise_rooms(rooms, id_map)
        norm_doors = remap_doors(doors, id_map)
        outer = compute_outer_boundary([r['geometry'] for r in norm_rooms])

        results.append(_build_record(
            source_file, key_id, key_id,
            unit_type, key_cfg, star_rating, floor_label,
            norm_rooms, norm_doors, outer,
        ))

    return results


def _normalise_rooms(rooms, id_map):
    return [
        {
            'id':       id_map[r['id']],
            'name':     r.get('name', 'Unknown').strip(),
            'area_m2':  float(r.get('area_m2', 0.0)),
            'geometry': r.get('geometry', []),
        }
        for r in rooms
    ]


def _build_record(source_file, key_id, unit_id, unit_type, key_cfg,
                  star_rating, floor_label, rooms, doors, outer):
    return {
        'source_file':       str(source_file),
        'key_id':            key_id,
        'unit_id':           unit_id,
        'unit_type':         unit_type,
        'key_configuration': key_cfg,
        'star_rating':       star_rating,
        'floor':             floor_label,
        'rooms':             rooms,
        'doors':             doors,
        'outer_boundary':    outer,
    }


# ---------------------------------------------------------------------------
# File processing
# ---------------------------------------------------------------------------

def _load_json(path):
    """
    Try common encodings in order and return the parsed JSON, or None on failure.
    Handles UTF-8, UTF-16 (with/without BOM), and Windows-1252.
    """
    encodings = ['utf-8-sig', 'utf-16', 'cp1252', 'latin-1']
    for enc in encodings:
        try:
            with open(path, 'r', encoding=enc) as f:
                return json.load(f)
        except (UnicodeDecodeError, UnicodeError):
            continue
        except json.JSONDecodeError as e:
            print(f"  [ERROR] File decoded but is not valid JSON ({enc}): {e}")
            return None
        except Exception as e:
            print(f"  [ERROR] Could not read file: {e}")
            return None
    print(f"  [ERROR] Could not decode file — tried {encodings}")
    return None


def process_file(input_path, output_dir):
    """
    Process one BIM JSON file.  Returns the number of units written.
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  {input_path.name}")

    data = _load_json(input_path)
    if data is None:
        return 0

    star_rating = data.get('star_rating', 0)
    floor_label = data.get('floor', '')
    keys = data.get('keys', [])

    if not keys:
        print(f"  [WARN] No keys found — check that the JSON matches the expected schema")
        return 0

    count = 0
    for key_data in keys:
        units = extract_key(key_data, star_rating, floor_label, input_path)
        for unit in units:
            out_path = _unique_path(output_dir, unit['unit_id'])
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(unit, f, indent=2)
            n_rooms = len(unit['rooms'])
            n_doors = len(unit['doors'])
            print(f"    → {out_path.name}  "
                  f"[{unit['unit_type']}, {unit['star_rating']}-star, "
                  f"{n_rooms} room(s), {n_doors} door(s)]")
            count += 1

    return count


def _unique_path(directory, stem):
    """Return a path that does not already exist, appending _1, _2 … if needed."""
    p = directory / f"{stem}.json"
    suffix = 1
    while p.exists():
        p = directory / f"{stem}_{suffix}.json"
        suffix += 1
    return p


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Stage 1: Extract hotel key units from BIM JSON files.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python extract_hotel_keys.py floor3.json\n"
            "  python extract_hotel_keys.py bim_exports/ --out_dir extracted_keys/\n"
            "\nAfter extraction, run build_hotel_vocab.py to build the room vocabulary."
        )
    )
    parser.add_argument(
        'input',
        help='BIM JSON file or directory of BIM JSON files to process.'
    )
    parser.add_argument(
        '--out_dir', default='extracted_keys',
        help='Output directory for extracted unit JSON files (default: extracted_keys/)'
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    total = 0

    print("Hotel key extraction — Stage 1")
    print("=" * 40)

    if input_path.is_dir():
        files = sorted(input_path.glob('*.json'))
        if not files:
            print(f"[ERROR] No JSON files found in {input_path}")
            sys.exit(1)
        print(f"Found {len(files)} file(s) in {input_path}")
        for f in files:
            total += process_file(f, args.out_dir)
    elif input_path.is_file():
        total += process_file(input_path, args.out_dir)
    else:
        print(f"[ERROR] {input_path} does not exist")
        sys.exit(1)

    print(f"\n{'=' * 40}")
    print(f"Extracted {total} unit(s) → {args.out_dir}/")
    print(f"Next: python build_hotel_vocab.py {args.out_dir}/")


if __name__ == '__main__':
    main()

"""
Stage 2b of the hotel floor-level pipeline (HouseDiffusion variant).

Reads processed floor JSONs produced by extract_hotel_floor.py and writes
per-floor JSON files in HouseDiffusion training format.

Each output file contains:
  floor_id    — string identifier
  star_rating — int (4 or 5)
  room_type   — [int, ...]          room-type integer per room
  room_names  — [str, ...]          canonical room name per room
  boxes       — [[x0,y0,x1,y1]...] bounding boxes, coords in [0,255]
  polygons    — [[[x,y],...], ...]  polygon corner sequences, coords in [0,255]
  num_corners — [int, ...]          corner count per room
  edges       — [[i,j], ...]        door-connected room index pairs (0-indexed)

Coordinate space: polygons and boxes from extract_hotel_floor.py are already
in 0–255 pixel space; this script rounds to integers and clamps.

Usage
-----
  python convert_to_housediffusion.py processed_floors/ --out housediffusion_data/
  python convert_to_housediffusion.py processed_floors/ --stats
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Room types treated as residual circulation — excluded from generation.
# The corridor is recovered post-inference as: boundary − union(all rooms).
# rtype 3 covers LOBBY, BOH LOBBY, CORRIDOR, CIRCULATION variants.
CIRCULATION_RTYPES = {3}
CIRCULATION_NAMES  = {"CIRCULATION", "CORRIDOR", "LOBBY", "BOH LOBBY", "HALLWAY"}


# =============================================================================
# Helpers
# =============================================================================

def _load_json(path):
    for enc in ("utf-8-sig", "utf-8", "cp1252"):
        try:
            with open(path, "r", encoding=enc) as f:
                return json.load(f)
        except (UnicodeDecodeError, UnicodeError):
            continue
        except Exception as e:
            print(f"  [ERROR] {path.name}: {e}")
            return None
    return None


def _round_coords(poly):
    """Round and clamp polygon [[x,y],...] to integer coords in [0,255]."""
    out = []
    for x, y in poly:
        out.append([max(0, min(255, int(round(x)))),
                    max(0, min(255, int(round(y))))])
    return out


def _round_box(box):
    """Round and clamp [x0,y0,x1,y1] to integer coords in [0,255]."""
    return [max(0, min(255, int(round(v)))) for v in box]


# =============================================================================
# Floor conversion
# =============================================================================

def _is_circulation(room):
    """Return True if this room is residual circulation and should be excluded."""
    if int(room.get("rtype", -1)) in CIRCULATION_RTYPES:
        return True
    if room.get("name", "").upper() in CIRCULATION_NAMES:
        return True
    return False


def _convert_floor(data):
    """
    Convert one processed floor dict to HouseDiffusion format.

    Circulation rooms (rtype 3 / LOBBY / CORRIDOR etc.) are excluded —
    they are residual space derived post-inference from boundary − union(rooms).

    Returns a dict or None if the floor is missing required data.
    """
    rooms    = data.get("rooms", [])
    edges    = data.get("edges", [])
    floor_id = data.get("floor_id", "unknown")
    star     = data.get("star_rating", 0)

    if not rooms:
        return None

    # Filter out circulation rooms first
    kept_rooms      = [r for r in rooms if not _is_circulation(r)]
    excluded_ids    = {r["id"] for r in rooms if _is_circulation(r)}
    n_excluded      = len(rooms) - len(kept_rooms)

    if not kept_rooms:
        return None

    # Build room list — must have box and polygon
    room_types  = []
    room_names  = []
    boxes       = []
    polygons    = []
    num_corners = []

    for r in kept_rooms:
        box  = r.get("box")
        poly = r.get("polygon")
        if not box or len(box) != 4:
            return None
        if not poly or len(poly) < 3:
            return None

        room_types.append(int(r.get("rtype", 0)))
        room_names.append(str(r.get("name", "")))
        boxes.append(_round_box(box))
        rounded = _round_coords(poly)
        polygons.append(rounded)
        num_corners.append(len(rounded))

    # Build id→new-index map (only kept rooms)
    id_to_idx = {r["id"]: i for i, r in enumerate(kept_rooms)}

    # Keep door edges (etype==1) between two kept rooms only.
    # Edges to/from circulation are dropped — those openings become part of
    # the residual corridor geometry recovered in post-processing.
    door_edges = []
    for e in edges:
        if len(e) < 3:
            continue
        from_id, to_id, etype = e[0], e[1], e[2]
        if etype != 1:
            continue
        if to_id < 0:
            continue
        if from_id in excluded_ids or to_id in excluded_ids:
            continue
        fi = id_to_idx.get(from_id)
        ti = id_to_idx.get(to_id)
        if fi is None or ti is None:
            continue
        door_edges.append([fi, ti])

    return {
        "floor_id":         floor_id,
        "star_rating":      star,
        "room_type":        room_types,
        "room_names":       room_names,
        "boxes":            boxes,
        "polygons":         polygons,
        "num_corners":      num_corners,
        "edges":            door_edges,
        "_n_excluded_circ": n_excluded,
    }


# =============================================================================
# Stats report
# =============================================================================

def _print_stats(all_samples):
    print()
    print("=" * 55)
    print("STATS — HouseDiffusion training data")
    print("=" * 55)

    # Star rating distribution
    star_dist = Counter(s["star_rating"] for s in all_samples)
    print(f"\nFloors by star rating: {dict(sorted(star_dist.items()))}")

    # Room count distribution
    room_counts = [len(s["room_type"]) for s in all_samples]
    print(f"Rooms per floor: min={min(room_counts)}  "
          f"max={max(room_counts)}  avg={sum(room_counts)/len(room_counts):.1f}")

    # Corner count histogram (critical for HouseDiffusion's Ni sampling)
    corner_hist = Counter()
    max_corners = 0
    for s in all_samples:
        for nc in s["num_corners"]:
            corner_hist[nc] += 1
            if nc > max_corners:
                max_corners = nc
    print(f"\nCorner count histogram (max={max_corners}):")
    for nc in sorted(corner_hist.keys()):
        bar = "#" * min(corner_hist[nc], 40)
        print(f"  {nc:>4} corners: {corner_hist[nc]:>4}  {bar}")

    # Room type distribution
    type_counts = Counter()
    type_names  = defaultdict(set)
    for s in all_samples:
        for rtype, rname in zip(s["room_type"], s["room_names"]):
            type_counts[rtype] += 1
            type_names[rtype].add(rname)
    print("\nRoom type distribution:")
    for rtype in sorted(type_counts.keys()):
        names = ", ".join(sorted(type_names[rtype])[:3])
        print(f"  type {rtype:>3}: {type_counts[rtype]:>5}  ({names})")

    print("=" * 55)


# =============================================================================
# Main
# =============================================================================

def convert(input_dir, output_dir, stats_only=False):
    input_dir  = Path(input_dir)
    output_dir = Path(output_dir)

    files = sorted(input_dir.glob("*.json"))
    if not files:
        print(f"[ERROR] No JSON files found in {input_dir}")
        sys.exit(1)

    print(f"Converting {len(files)} floor(s) from {input_dir}")
    print()

    samples = []
    skipped = 0

    for fpath in files:
        data = _load_json(fpath)
        if data is None or "rooms" not in data:
            print(f"  [SKIP] {fpath.name} — not a processed floor JSON")
            skipped += 1
            continue

        sample = _convert_floor(data)
        if sample is None:
            print(f"  [SKIP] {fpath.name} — missing required fields")
            skipped += 1
            continue

        n_rooms   = len(sample["room_type"])
        n_edges   = len(sample["edges"])
        n_excl    = sample["_n_excluded_circ"]
        total_c   = sum(sample["num_corners"])
        max_corn  = max(sample["num_corners"]) if sample["num_corners"] else 0
        print(f"  {fpath.name:<30}  {n_rooms:>3} rooms  "
              f"{n_edges:>3} door edges  "
              f"{total_c:>4} total corners  max {max_corn}  "
              f"(excluded {n_excl} circulation)")
        samples.append(sample)

    print()
    print(f"Samples ready: {len(samples)}  (skipped: {skipped})")

    if not samples:
        print("[ERROR] No valid samples to write.")
        sys.exit(1)

    if stats_only:
        _print_stats(samples)
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    for s in samples:
        out_path = output_dir / f"{s['floor_id']}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(s, f, indent=2)
        print(f"  Wrote {out_path.name}")

    print(f"\nWrote {len(samples)} sample(s) → {output_dir}/")

    _print_stats(samples)

    print()
    print("Next steps:")
    print("  1. Review corner histogram — adjust HouseDiffusion's max_corners")
    print("     if many rooms exceed the model's sampling limit (~12 by default)")
    print("  2. Task 2: fix room index encoding in house_diffusion/unet.py")
    print("     (replace 32D one-hot with nn.Embedding)")
    print("  3. Implement training loop adaptor for this JSON format")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Convert processed floor JSONs to HouseDiffusion training format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python convert_to_housediffusion.py processed_floors/ --out housediffusion_data/\n"
            "  python convert_to_housediffusion.py processed_floors/ --stats\n"
        ),
    )
    parser.add_argument("input_dir",
                        help="Directory of processed floor JSONs from extract_hotel_floor.py")
    parser.add_argument("--out", default="housediffusion_data",
                        help="Output directory for per-floor JSON files (default: housediffusion_data/)")
    parser.add_argument("--stats", action="store_true",
                        help="Print stats only — do not write output files")
    args = parser.parse_args()

    convert(args.input_dir, args.out, stats_only=args.stats)


if __name__ == "__main__":
    main()

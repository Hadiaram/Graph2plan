"""
Stage 2 of the hotel floor-level pipeline.

Reads processed floor JSONs produced by extract_hotel_floor.py and writes
a single data.mat file compatible with Graph2plan's FloorPlanDataset.

Each floor becomes one training sample with the fields:
  name      — floor_id string
  boundary  — (N, 4) float array  [x, y, dir, isNew]
  box       — (M, 5) float array  [x0, y0, x1, y1, rtype]
  edge      — (E, 3) int array    [from, to, etype]
  order     — (M,)   float array  1-indexed room ordering (natural)

The FloorPlanDataset compatibility layer in floorplan.py automatically
splits 'box' into 'gtBoxNew' + 'rType' and renames 'edge' to 'rEdge'.

Usage
-----
  python convert_hotel_floor_to_mat.py processed_floors/
  python convert_hotel_floor_to_mat.py processed_floors/ --out data/data_hotel.mat
  python convert_hotel_floor_to_mat.py processed_floors/ --report

After conversion, update Network/model/utils.py (see --report output),
then run:
  cd ../Network
  python split.py --data data/data_hotel.mat
  python train.py --dataset_dir data/
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import scipy.io as sio


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


def _floor_to_arrays(floor):
    """
    Convert one processed floor dict to numpy arrays.

    Returns a dict with keys: name, boundary, box, edge, order.
    Returns None if the floor is missing required data.
    """
    rooms    = floor.get("rooms", [])
    edges    = floor.get("edges", [])
    boundary = floor.get("boundary", [])

    if not rooms:
        return None
    if not boundary:
        return None

    n = len(rooms)

    # boundary: (N, 4)  [x, y, dir, isNew]
    boundary_arr = np.array(boundary, dtype=np.float64)
    if boundary_arr.ndim != 2 or boundary_arr.shape[1] < 4:
        return None

    # box: (M, 5)  [x0, y0, x1, y1, rtype]
    box_rows = []
    for r in rooms:
        b = r.get("box", [])
        if len(b) != 4:
            return None
        box_rows.append([b[0], b[1], b[2], b[3], float(r["rtype"])])
    box_arr = np.array(box_rows, dtype=np.float64)

    # edge: (E, 3)  [from, to, etype]
    # Drop exterior-door edges (to=-1) — model expects room-to-room edges only
    room_edges = [e for e in edges if e[1] >= 0]
    if room_edges:
        edge_arr = np.array(room_edges, dtype=np.int64)
    else:
        edge_arr = np.zeros((0, 3), dtype=np.int64)

    # order: (M,)  1-indexed natural ordering
    # floorplan.py does  order = (data.order - 1).astype(int)  to get 0-indexed
    order_arr = np.arange(1, n + 1, dtype=np.float64)

    return {
        "name":     floor.get("floor_id", "unknown"),
        "boundary": boundary_arr,
        "box":      box_arr,
        "edge":     edge_arr,
        "order":    order_arr,
    }


# =============================================================================
# Vocab report
# =============================================================================

def _vocab_report(all_floors):
    """
    Print the room type integers found across all floors so the user knows
    exactly how to update Network/model/utils.py.
    """
    type_names = {}   # rtype → set of room names
    for floor in all_floors:
        vocab = floor.get("vocab_used", {})
        for name, tid in vocab.items():
            type_names.setdefault(tid, set()).add(name)

    print()
    print("=" * 55)
    print("VOCAB REPORT — update Network/model/utils.py")
    print("=" * 55)
    print()
    print("Replace the room_label list in get_vocab() with:")
    print()
    print("    room_label = [")
    for tid in sorted(type_names.keys()):
        names = sorted(type_names[tid])
        # Use the shortest name as the canonical label
        canonical = min(names, key=len)
        aliases   = [n for n in names if n != canonical]
        alias_str = f"  # also: {', '.join(aliases)}" if aliases else ""
        print(f"        ({tid}, '{canonical}', 1, 'HotelRoom'),{alias_str}")
    print("    ]")
    print()
    print("Also update get_color_map() to add colours for the new types.")
    print("=" * 55)


# =============================================================================
# Main conversion
# =============================================================================

def convert(input_dir, output_path, report_only=False):
    input_dir   = Path(input_dir)
    output_path = Path(output_path)

    files = sorted(input_dir.glob("*.json"))
    if not files:
        print(f"[ERROR] No JSON files found in {input_dir}")
        sys.exit(1)

    print(f"Converting {len(files)} floor(s) from {input_dir}")
    print()

    samples   = []
    all_floors = []
    skipped   = 0

    for fpath in files:
        data = _load_json(fpath)
        if data is None or "rooms" not in data:
            print(f"  [SKIP] {fpath.name} — not a processed floor JSON")
            skipped += 1
            continue

        arrays = _floor_to_arrays(data)
        if arrays is None:
            print(f"  [SKIP] {fpath.name} — missing required fields")
            skipped += 1
            continue

        n_rooms = len(data["rooms"])
        n_edges = len(arrays["edge"])
        n_bpts  = len(arrays["boundary"])
        print(f"  {fpath.name:<30}  {n_rooms:>3} rooms  "
              f"{n_bpts:>3} boundary pts  {n_edges:>3} edges")
        samples.append(arrays)
        all_floors.append(data)

    print()
    print(f"Samples ready: {len(samples)}  (skipped: {skipped})")

    if not samples:
        print("[ERROR] No valid samples to write.")
        sys.exit(1)

    if report_only:
        _vocab_report(all_floors)
        return

    # -------------------------------------------------------------------------
    # Build numpy structured array — scipy converts this to a MATLAB struct array
    # -------------------------------------------------------------------------
    dtype = np.dtype([
        ("name",     "O"),
        ("boundary", "O"),
        ("box",      "O"),
        ("order",    "O"),
        ("edge",     "O"),
    ])

    data_array = np.empty((len(samples),), dtype=dtype)
    for i, s in enumerate(samples):
        data_array[i]["name"]     = s["name"]
        data_array[i]["boundary"] = s["boundary"]
        data_array[i]["box"]      = s["box"]
        data_array[i]["order"]    = s["order"]
        data_array[i]["edge"]     = s["edge"]

    # -------------------------------------------------------------------------
    # Write .mat
    # -------------------------------------------------------------------------
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sio.savemat(str(output_path), {"data": data_array}, do_compression=True)

    print(f"\nWrote {len(samples)} sample(s) → {output_path}")

    # Stats
    room_counts = [len(f["rooms"]) for f in all_floors]
    print(f"  Rooms per floor: min={min(room_counts)}  "
          f"max={max(room_counts)}  avg={sum(room_counts)/len(room_counts):.1f}")

    # Vocab report always printed after writing
    _vocab_report(all_floors)

    print()
    print("Next steps:")
    print("  1. Update Network/model/utils.py  (see VOCAB REPORT above)")
    print(f"  2. cd ../Network && python split.py --data ../{output_path}")
    print("  3. python train.py --dataset_dir data/")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Stage 2 (floor level): Convert processed floor JSONs to .mat training data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python convert_hotel_floor_to_mat.py processed_floors/\n"
            "  python convert_hotel_floor_to_mat.py processed_floors/ --out data/data_hotel.mat\n"
            "  python convert_hotel_floor_to_mat.py processed_floors/ --report\n"
        ),
    )
    parser.add_argument("input_dir",
                        help="Directory of processed floor JSONs from extract_hotel_floor.py")
    parser.add_argument("--out", default="data/data_hotel.mat",
                        help="Output .mat file path (default: data/data_hotel.mat)")
    parser.add_argument("--report", action="store_true",
                        help="Print vocab report only — do not write .mat file")
    args = parser.parse_args()

    convert(args.input_dir, args.out, report_only=args.report)


if __name__ == "__main__":
    main()

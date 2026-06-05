"""
Visualise a processed hotel floor JSON or .mat training file.

JSON mode  (from extract_hotel_floor.py):
  Draws each room as a filled polygon with name, number, and area labels.

MAT mode  (from convert_hotel_floor_to_mat.py):
  Draws each room as a filled bounding-box rectangle — useful for verifying
  that nothing was lost during the JSON → .mat conversion.

Usage
-----
  python visualize_hotel_floor.py 4star_L1.json
  python visualize_hotel_floor.py processed_floors/
  python visualize_hotel_floor.py data/data_hotel.mat --out images/
  python visualize_hotel_floor.py data/data_hotel.mat --index 2
  python visualize_hotel_floor.py 4star_L1.json --no-labels
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon as MplPolygon, Rectangle as MplRect
from matplotlib.collections import PatchCollection
import numpy as np

# =============================================================================
# Colour palette  (type_int → fill colour)
# =============================================================================
# Colours are chosen to be distinguishable and roughly semantically meaningful.

TYPE_COLOURS = {
    0: "#AED6F1",   # BR               — light blue
    1: "#A9DFBF",   # SUITE            — light green
    2: "#27AE60",   # EXECUTIVE SUITE  — darker green
    3: "#D5D8DC",   # CIRCULATION      — light grey
    4: "#F9E79F",   # LIFT             — pale yellow
    5: "#F0B27A",   # LIFT LOBBY       — light orange
    6: "#F1948A",   # STAIRS           — salmon
    7: "#D2B4DE",   # MEP              — lavender
}
DEFAULT_COLOUR = "#FDFEFE"   # unknown type

BOUNDARY_COLOUR  = "#1A1A2E"  # outer floor outline
DOOR_EDGE_COLOUR = "#E74C3C"  # door connection markers
WALL_COLOUR      = "#5D6D7E"  # room boundary lines


# =============================================================================
# Helpers
# =============================================================================

def _poly_centroid(pts):
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return sum(xs) / len(xs), sum(ys) / len(ys)


def _midpoint(a, b):
    return ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)


def _label_size(poly, canvas):
    """Return a font size scaled to the room's bounding box."""
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    w = max(xs) - min(xs)
    h = max(ys) - min(ys)
    short = min(w, h)
    # Scale: 6 pt at 20 px wide, up to 9 pt at 60+ px
    return max(5, min(9, short * 0.15))


# =============================================================================
# Rendering
# =============================================================================

def render(data, out_path=None, show=False, draw_labels=True, draw_edges=True):
    canvas   = data.get("canvas_size", 256)
    rooms    = data.get("rooms", [])
    boundary = data.get("boundary", [])
    edges    = data.get("edges", [])
    vocab    = data.get("vocab_used", {})
    floor_id = data.get("floor_id", "floor")
    star     = data.get("star_rating", "?")

    # Invert Y so the floor reads top-to-bottom as drawn in Revit.
    # All Y coordinates are flipped: y_plot = canvas - y_data
    def flip(pts):
        return [[p[0], canvas - p[1]] for p in pts]

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_aspect("equal")
    ax.set_xlim(0, canvas)
    ax.set_ylim(0, canvas)
    ax.axis("off")
    ax.set_facecolor("#F8F9FA")
    fig.patch.set_facecolor("#F8F9FA")

    # ------------------------------------------------------------------
    # Room polygons — draw largest first so smaller rooms sit on top
    # ------------------------------------------------------------------
    seen_types = {}   # type_int → canonical name (for legend)

    def _poly_area(pts):
        # Shoelace formula
        n = len(pts)
        a = 0.0
        for i in range(n):
            j = (i + 1) % n
            a += pts[i][0] * pts[j][1]
            a -= pts[j][0] * pts[i][1]
        return abs(a) / 2.0

    rooms_sorted = sorted(rooms,
                          key=lambda r: _poly_area(r["polygon"]),
                          reverse=True)

    for r in rooms_sorted:
        poly   = flip(r["polygon"])
        rtype  = r["rtype"]
        name   = r["name"]
        colour = TYPE_COLOURS.get(rtype, DEFAULT_COLOUR)

        patch = MplPolygon(poly, closed=True,
                           facecolor=colour, edgecolor=WALL_COLOUR,
                           linewidth=0.8, zorder=2)
        ax.add_patch(patch)

        # Track unique types for legend
        if rtype not in seen_types:
            seen_types[rtype] = name

        if draw_labels:
            cx, cy = _poly_centroid(poly)
            fs = _label_size(poly, canvas)

            # Room name
            ax.text(cx, cy + fs * 0.5, name,
                    ha="center", va="center",
                    fontsize=fs, fontweight="bold",
                    color="#1A1A2E", zorder=4,
                    clip_on=True)

            # Room number and area on separate lines
            num  = r.get("number", "")
            area = r.get("area_m2", 0)
            if num:
                ax.text(cx, cy - fs * 0.1, f"#{num}",
                        ha="center", va="center",
                        fontsize=max(4, fs - 1.5), color="#5D6D7E",
                        zorder=4, clip_on=True)
                ax.text(cx, cy - fs * 0.7, f"{area:.1f} m²",
                        ha="center", va="center",
                        fontsize=max(4, fs - 2.0), color="#7F8C8D",
                        zorder=4, clip_on=True)
            else:
                ax.text(cx, cy - fs * 0.4, f"{area:.1f} m²",
                        ha="center", va="center",
                        fontsize=max(4, fs - 1.5), color="#5D6D7E",
                        zorder=4, clip_on=True)

    # ------------------------------------------------------------------
    # Door edges — draw a small marker at the midpoint between room centroids
    # ------------------------------------------------------------------
    if draw_edges:
        centroids = {}
        for r in rooms:
            cx, cy = _poly_centroid(flip(r["polygon"]))
            centroids[r["id"]] = (cx, cy)

        for edge in edges:
            f, t, etype = edge[0], edge[1], edge[2]
            if etype != 1:   # only door edges (etype=1)
                continue
            if f not in centroids or t not in centroids:
                continue
            cx_f, cy_f = centroids[f]
            cx_t, cy_t = centroids[t]
            mx, my = _midpoint((cx_f, cy_f), (cx_t, cy_t))
            ax.plot(mx, my, marker="D", markersize=4,
                    color=DOOR_EDGE_COLOUR, zorder=5, markeredgewidth=0)

    # ------------------------------------------------------------------
    # Outer floor boundary
    # ------------------------------------------------------------------
    if boundary:
        bpts = flip([[p[0], p[1]] for p in boundary])
        bpts_closed = bpts + [bpts[0]]
        bx = [p[0] for p in bpts_closed]
        by = [p[1] for p in bpts_closed]
        ax.plot(bx, by, color=BOUNDARY_COLOUR, linewidth=2.0, zorder=6)

    # ------------------------------------------------------------------
    # Legend
    # ------------------------------------------------------------------
    handles = []
    for type_int in sorted(seen_types.keys()):
        colour  = TYPE_COLOURS.get(type_int, DEFAULT_COLOUR)
        label   = seen_types[type_int]
        handles.append(mpatches.Patch(facecolor=colour,
                                      edgecolor=WALL_COLOUR,
                                      linewidth=0.6,
                                      label=f"[{type_int}] {label}"))
    if draw_edges:
        handles.append(mpatches.Patch(facecolor=DOOR_EDGE_COLOUR,
                                      label="Door connection"))

    ax.legend(handles=handles,
              loc="lower right", fontsize=7,
              framealpha=0.9, edgecolor="#CCCCCC")

    # Title
    ax.set_title(f"{floor_id}  ({star}-star)",
                 fontsize=11, fontweight="bold",
                 color="#1A1A2E", pad=8)

    plt.tight_layout(pad=1.0)

    if show:
        matplotlib.use("TkAgg")
        plt.show()
    else:
        fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
        print(f"  → {out_path}")

    plt.close(fig)


# =============================================================================
# .mat rendering  (bounding-box view)
# =============================================================================

# Integer type → name for .mat rendering (matches CANONICAL_TYPES in
# extract_hotel_floor.py).  Unknown integers are shown as "TYPE_<n>".
_MAT_TYPE_NAMES = {
    0: "BR",
    1: "SUITE",
    2: "EXECUTIVE SUITE",
    3: "CIRCULATION",
    4: "LIFT",
    5: "LIFT LOBBY",
    6: "STAIRS",
    7: "MEP",
}


def render_mat_sample(sample, out_path, show=False, draw_labels=True,
                      draw_edges=True):
    """Render one mat_struct sample as a bounding-box floor plan."""
    import scipy.io as sio   # local import — only needed for mat mode

    name     = str(sample.name)
    boundary = np.array(sample.boundary, dtype=float)   # (N, 4)
    box      = np.array(sample.box,      dtype=float)   # (M, 5)
    edge     = np.array(sample.edge,     dtype=int) if sample.edge.size else np.zeros((0,3),int)

    canvas = 256
    def flip(y): return canvas - y

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_aspect("equal")
    ax.set_xlim(0, canvas)
    ax.set_ylim(0, canvas)
    ax.axis("off")
    ax.set_facecolor("#F8F9FA")
    fig.patch.set_facecolor("#F8F9FA")

    # Draw boxes largest-first
    order = np.argsort((box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1]))[::-1]
    seen_types = {}

    for i in order:
        x0, y0, x1, y1, rtype = box[i]
        rtype = int(rtype)
        colour = TYPE_COLOURS.get(rtype, DEFAULT_COLOUR)
        w, h   = x1 - x0, y1 - y0

        # Y-flip: origin at top-left
        rect = MplRect((x0, flip(y1)), w, h,
                       facecolor=colour, edgecolor=WALL_COLOUR,
                       linewidth=0.8, zorder=2)
        ax.add_patch(rect)

        if rtype not in seen_types:
            seen_types[rtype] = _MAT_TYPE_NAMES.get(rtype, f"TYPE_{rtype}")

        if draw_labels:
            cx = (x0 + x1) / 2
            cy = flip((y0 + y1) / 2)
            short = min(w, h)
            fs = max(5, min(9, short * 0.15))
            label = _MAT_TYPE_NAMES.get(rtype, f"TYPE_{rtype}")
            ax.text(cx, cy + fs * 0.2, label,
                    ha="center", va="center",
                    fontsize=fs, fontweight="bold",
                    color="#1A1A2E", zorder=4, clip_on=True)
            ax.text(cx, cy - fs * 0.5, f"{w:.0f}×{h:.0f} px",
                    ha="center", va="center",
                    fontsize=max(4, fs - 2), color="#7F8C8D",
                    zorder=4, clip_on=True)

    # Door edges
    if draw_edges and edge.shape[0] > 0:
        centroids = {}
        for i in range(len(box)):
            cx = (box[i, 0] + box[i, 2]) / 2
            cy = flip((box[i, 1] + box[i, 3]) / 2)
            centroids[i] = (cx, cy)

        for f, t, etype in edge:
            if etype != 1 or t < 0:
                continue
            if f not in centroids or t not in centroids:
                continue
            mx = (centroids[f][0] + centroids[t][0]) / 2
            my = (centroids[f][1] + centroids[t][1]) / 2
            ax.plot(mx, my, marker="D", markersize=4,
                    color=DOOR_EDGE_COLOUR, zorder=5, markeredgewidth=0)

    # Boundary outline
    if boundary.shape[0] >= 2:
        bx = boundary[:, 0].tolist() + [boundary[0, 0]]
        by = [flip(y) for y in boundary[:, 1].tolist()] + [flip(boundary[0, 1])]
        ax.plot(bx, by, color=BOUNDARY_COLOUR, linewidth=2.0, zorder=6)

    # Legend
    handles = [
        mpatches.Patch(facecolor=TYPE_COLOURS.get(t, DEFAULT_COLOUR),
                       edgecolor=WALL_COLOUR, linewidth=0.6,
                       label=f"[{t}] {n}")
        for t, n in sorted(seen_types.items())
    ]
    if draw_edges:
        handles.append(mpatches.Patch(facecolor=DOOR_EDGE_COLOUR,
                                      label="Door connection"))
    ax.legend(handles=handles, loc="lower right", fontsize=7,
              framealpha=0.9, edgecolor="#CCCCCC")

    ax.set_title(f"{name}  (from .mat — bounding boxes)",
                 fontsize=11, fontweight="bold", color="#1A1A2E", pad=8)

    plt.tight_layout(pad=1.0)
    if show:
        plt.show()
    else:
        fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
        print(f"  → {out_path}")
    plt.close(fig)


def process_mat(mat_path, output_dir, show, draw_labels, draw_edges,
                index=None):
    """Load a .mat file and render all samples (or just one if index given)."""
    import scipy.io as sio

    try:
        raw = sio.loadmat(str(mat_path), squeeze_me=True,
                          struct_as_record=False)["data"]
    except Exception as e:
        print(f"  [ERROR] Could not load {mat_path.name}: {e}")
        return

    # squeeze_me collapses a single-element array to a scalar
    if not hasattr(raw, "__len__"):
        raw = np.array([raw])

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    targets = [index] if index is not None else range(len(raw))
    print(f"  {mat_path.name}  ({len(raw)} sample(s) total)")

    for i in targets:
        if i >= len(raw):
            print(f"  [WARN] Index {i} out of range (file has {len(raw)} samples)")
            continue
        sample = raw[i]
        name   = str(sample.name)
        n_rooms = len(sample.box)
        print(f"    [{i}] {name}  ({n_rooms} rooms)")
        out_png = out_dir / f"{mat_path.stem}_{i}_{name}.png"
        render_mat_sample(sample, out_png, show=show,
                          draw_labels=draw_labels, draw_edges=draw_edges)


# =============================================================================
# JSON file handling
# =============================================================================

def _load(path):
    for enc in ("utf-8-sig", "utf-8", "cp1252"):
        try:
            with open(path, "r", encoding=enc) as f:
                return json.load(f)
        except (UnicodeDecodeError, UnicodeError):
            continue
        except Exception as e:
            print(f"  [ERROR] {e}")
            return None
    return None


def process(input_path, output_dir, show, draw_labels, draw_edges):
    input_path = Path(input_path)
    data = _load(input_path)
    if data is None:
        print(f"  [SKIP] Could not read {input_path.name}")
        return

    if "rooms" not in data:
        print(f"  [SKIP] {input_path.name} — no 'rooms' key")
        return

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / f"{input_path.stem}.png"

    print(f"  {input_path.name}  ({data.get('n_rooms','?')} rooms)")
    render(data, out_path=out_png, show=show,
           draw_labels=draw_labels, draw_edges=draw_edges)


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Visualise a processed hotel floor JSON or .mat training file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python visualize_hotel_floor.py 4star_L1.json\n"
            "  python visualize_hotel_floor.py processed_floors/ --out images/\n"
            "  python visualize_hotel_floor.py data/data_hotel.mat --out images/\n"
            "  python visualize_hotel_floor.py data/data_hotel.mat --index 0\n"
            "  python visualize_hotel_floor.py 4star_L1.json --no-labels\n"
        ),
    )
    parser.add_argument("input",
                        help="JSON file, directory of JSONs, or .mat file.")
    parser.add_argument("--out", default="floor_images",
                        help="Output directory for PNG files (default: floor_images/)")
    parser.add_argument("--show", action="store_true",
                        help="Open an interactive window instead of saving.")
    parser.add_argument("--no-labels", dest="labels", action="store_false",
                        help="Omit room labels.")
    parser.add_argument("--no-edges", dest="edges", action="store_false",
                        help="Omit door edge markers.")
    parser.add_argument("--index", type=int, default=None,
                        help="For .mat files: render only this sample index.")
    parser.set_defaults(labels=True, edges=True)
    args = parser.parse_args()

    input_path = Path(args.input)

    if input_path.suffix.lower() == ".mat":
        process_mat(input_path, args.out, args.show,
                    args.labels, args.edges, args.index)
    elif input_path.is_dir():
        files = sorted(input_path.glob("*.json"))
        if not files:
            print(f"[ERROR] No JSON files found in {input_path}")
            sys.exit(1)
        print(f"Rendering {len(files)} floor(s) from {input_path}")
        for f in files:
            process(f, args.out, args.show, args.labels, args.edges)
    elif input_path.is_file():
        process(input_path, args.out, args.show, args.labels, args.edges)
    else:
        print(f"[ERROR] {input_path} does not exist")
        sys.exit(1)

    print(f"\nImages saved to: {args.out}/")


if __name__ == "__main__":
    main()

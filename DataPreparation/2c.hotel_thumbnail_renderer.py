"""
Hotel floor thumbnail renderer — hotel-specific data path.

Reads processed_floors/*.json and renders each floor plan as a PNG thumbnail
using actual polygon geometry.  Output goes to:

  Interface/static/Data/snapshot_train/<name>.png

Colors match the frontend roomcolor() palette where hotel types overlap with
residential types; hotel-specific types use a consistent hotel palette.

Hotel room type integers (from vocab_used in the JSON):
  0  Guest room (Classic / Deluxe)
  1  Premium room / Suite bedroom
  2  Meeting / Function room
  3  Lobby / BOH lobby
  4  Lift / Elevator
  5  Lift lobby / Elevator lobby
  6  Stairs
  7  Service room / Linen
  8  Technical / Plant

Run after 2b.hotel_pkl_builder.py whenever new JSON floors are added.
"""

from pathlib import Path

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')                   # no display needed
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection
from tqdm.auto import tqdm

# ── Paths ──────────────────────────────────────────────────────────────────────
HERE      = Path(__file__).parent
JSON_DIR  = HERE / 'processed_floors'
OUT_DIR   = HERE.parent / 'Interface' / 'static' / 'Data' / 'snapshot_train'

# ── Hotel room type → fill colour (RGB 0–255, matching frontend roomcolor()) ──
# Residential equivalents kept identical so the UI feels consistent.
_TYPE_COLOURS = {
    0: (253, 244, 171),   # guest room         — MasterRoom yellow
    1: (255, 230, 140),   # premium / suite     — deeper amber
    2: (220, 210, 240),   # meeting / function  — soft lavender
    3: (244, 242, 229),   # lobby / BOH         — LivingRoom beige
    4: (190, 215, 235),   # lift / elevator     — steel blue
    5: (215, 215, 215),   # lift lobby          — light grey
    6: (180, 180, 180),   # stairs              — medium grey
    7: (249, 222, 189),   # service / linen     — Storage orange
    8: (155, 155, 155),   # technical / plant   — dark grey
}
_DEFAULT_COLOUR = (240, 240, 240)   # unmapped types
_BOUNDARY_COLOUR = (79 / 255, 79 / 255, 79 / 255)   # Exterior wall dark grey
_ROOM_EDGE_COLOUR = (128 / 255, 128 / 255, 128 / 255)


def _to_mpl(rgb_255):
    return tuple(c / 255.0 for c in rgb_255)


def render_floor(floor: dict, out_path: Path, px: int = 512) -> None:
    canvas = floor.get('canvas_size', 256)

    fig, ax = plt.subplots(figsize=(px / 100, px / 100), dpi=100)
    ax.set_xlim(0, canvas)
    ax.set_ylim(0, canvas)
    ax.set_aspect('equal')
    ax.axis('off')
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    # ── Room polygons ──────────────────────────────────────────────────────────
    patches = []
    colours = []
    for room in floor.get('rooms', []):
        poly = np.array(room['polygon'], dtype=float)
        if len(poly) < 3:
            continue
        # Flip y so north-up matches the JS canvas (y increases downward there,
        # but matplotlib y increases upward — flip to keep the visual orientation)
        poly[:, 1] = canvas - poly[:, 1]
        rtype = int(room.get('rtype', -1))
        rgb   = _TYPE_COLOURS.get(rtype, _DEFAULT_COLOUR)
        patches.append(MplPolygon(poly, closed=True))
        colours.append(_to_mpl(rgb))

    if patches:
        pc = PatchCollection(
            patches,
            facecolors=colours,
            edgecolors=[_ROOM_EDGE_COLOUR] * len(patches),
            linewidths=0.5,
        )
        ax.add_collection(pc)

    # ── Boundary outline ───────────────────────────────────────────────────────
    boundary = np.array(floor.get('boundary', []), dtype=float)
    if len(boundary) >= 3:
        bxy = boundary[:, :2].copy()
        bxy[:, 1] = canvas - bxy[:, 1]
        bxy = np.vstack([bxy, bxy[:1]])           # close the polygon
        ax.plot(bxy[:, 0], bxy[:, 1],
                color=_BOUNDARY_COLOUR, linewidth=1.5, solid_capstyle='round')

    fig.savefig(out_path, bbox_inches='tight', pad_inches=0.05,
                facecolor='white', dpi=100)
    plt.close(fig)


# ── Main ───────────────────────────────────────────────────────────────────────
json_files = sorted(JSON_DIR.glob('*.json'))
if not json_files:
    raise FileNotFoundError(f'No JSON files found in {JSON_DIR}')

OUT_DIR.mkdir(parents=True, exist_ok=True)
print(f'Rendering {len(json_files)} hotel floor thumbnail(s) → {OUT_DIR}')

for jf in tqdm(json_files, desc='Rendering thumbnails'):
    with open(jf, encoding='utf-8') as f:
        floor = json.load(f)
    out_path = OUT_DIR / (jf.stem + '.png')
    render_floor(floor, out_path)

print('Done.')

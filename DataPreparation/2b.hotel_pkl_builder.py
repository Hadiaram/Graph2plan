"""
Hotel floor PKL builder — hotel-specific data path.

Reads processed_floors/*.json and produces the three files that the Django
interface expects when building_mode == 'hotel':

  Interface/static/Data/data_train_hotel.pkl    {'data', 'nameList', 'trainTF'}
  Interface/static/Data/rNum_train_hotel.npy    (N, 14) room-count matrix
  Interface/static/Data/data_train_eNum_hotel.pkl  {'eNum': (N, 25)}

Room polygon geometry from the JSON is preserved in each entry's .rBoundary
field so thumbnails and post-processing operate on actual shapes, not bounding
box rectangles.  The residential MAT pipeline is untouched.
"""

import json
import pickle
import types
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

# ── Paths ──────────────────────────────────────────────────────────────────────
HERE     = Path(__file__).parent
JSON_DIR = HERE / 'processed_floors'
OUT_DIR  = HERE.parent / 'Interface' / 'static' / 'Data'

OUT_PKL  = OUT_DIR / 'data_train_hotel.pkl'
OUT_RNUM = OUT_DIR / 'rNum_train_hotel.npy'
OUT_ENUM = OUT_DIR / 'data_train_eNum_hotel.pkl'


# ── Turn-function (from 1.tf_train.py, kept in sync) ──────────────────────────
def _compute_tf(boundary):
    b = np.array(boundary, dtype=float)
    if b.ndim == 2 and b.shape[1] > 2:
        b = b[:, :2]
    b = np.concatenate([b, b[:1]])
    n = len(b) - 1
    vec = b[1:] - b[:-1]
    lengths = np.linalg.norm(vec, axis=1)
    perimeter = lengths.sum()
    if perimeter == 0:
        return np.zeros(n + 1), np.zeros(n + 1)
    vec     = vec     / perimeter
    lengths = lengths / perimeter

    angles = np.zeros(n)
    for i in range(n):
        vi, vj = vec[i], vec[(i + 1) % n]
        z = vi[0] * vj[1] - vi[1] * vj[0]   # 2-D cross product → scalar
        dot = float(np.clip(np.dot(vec[i], vec[(i + 1) % n]), -1.0, 1.0))
        angles[i] = np.arccos(dot) * np.sign(float(z))

    x = np.zeros(n + 1)
    y = np.zeros(n + 1)
    s = 0.0
    for i in range(1, n + 1):
        x[i]     = lengths[i - 1] + x[i - 1]
        y[i - 1] = angles[i - 1] + s
        s         = y[i - 1]
    y[-1] = s
    return x, y


# ── eNum helper (mirrors 4.data_train_eNum.py) ────────────────────────────────
_R_MAP   = np.array([1, 2, 3, 4, 1, 2, 2, 2, 2, 5, 1, 6, 1, 10, 7, 8, 9, 10]) - 1
_REORDER = np.array([0, 1, 3, 2, 4, 5, 6, 7, 8, 9])


def _compute_enum(r_types, edges):
    """Return a (25,) eNum vector (flattened 5×5 edge-type co-occurrence)."""
    mat = np.zeros((5, 5), dtype='uint8')
    if not edges:
        return mat.reshape(-1)

    e = np.array(edges, dtype=int)
    if e.ndim == 1:
        e = e.reshape(1, -1)
    if e.shape[1] < 2:
        return mat.reshape(-1)

    r = np.array(r_types, dtype=int)
    src, dst = e[:, 0], e[:, 1]
    valid    = (src < len(r)) & (dst < len(r)) & (src >= 0) & (dst >= 0)
    src, dst = src[valid], dst[valid]
    if len(src) == 0:
        return mat.reshape(-1)

    et = np.stack([r[src], r[dst]], axis=1)
    et = np.clip(et, 0, len(_R_MAP) - 1)
    mapped    = _R_MAP[et]
    reordered = _REORDER[np.clip(mapped, 0, len(_REORDER) - 1)]
    keep = ((reordered[:, 0] >= 1) & (reordered[:, 0] <= 5) &
            (reordered[:, 1] >= 1) & (reordered[:, 1] <= 5))
    reordered = reordered[keep] - 1
    for row in reordered:
        mat[row[0], row[1]] += 1
        if row[0] != row[1]:
            mat[row[1], row[0]] += 1
    return mat.reshape(-1)


# ── Main build loop ────────────────────────────────────────────────────────────
json_files = sorted(JSON_DIR.glob('*.json'))
if not json_files:
    raise FileNotFoundError(f'No JSON files found in {JSON_DIR}')

print(f'Found {len(json_files)} hotel floor JSON file(s) in {JSON_DIR}')

data_entries = []
name_list    = []
train_tf     = []
r_num_rows   = []
e_num_rows   = []

for jf in tqdm(json_files, desc='Building PKL entries'):
    with open(jf, encoding='utf-8') as f:
        floor = json.load(f)

    name    = jf.stem                                          # e.g. '4star_L1_1'

    # Outer floor boundary — bounding rectangle of ALL room polygons.
    # The JSON 'boundary' field is the inner corridor polygon, which is too
    # narrow for the logic layer and UI.  We store the outer envelope instead.
    _all_pts = np.vstack([np.array(r['polygon'], dtype=np.float64) for r in floor['rooms']])
    _x0, _x1 = float(_all_pts[:, 0].min()), float(_all_pts[:, 0].max())
    _y0, _y1 = float(_all_pts[:, 1].min()), float(_all_pts[:, 1].max())
    _ym = (_y0 + _y1) / 2.0
    # 6-point boundary: starts/ends at centre-left (door side), goes clockwise.
    # dir: 0=right, 1=down(+y), 2=left, 3=up(-y)
    boundary = np.array([
        [_x0, _ym, 1, 0],   # centre-left  → heading down
        [_x0, _y1, 0, 0],   # bottom-left  → heading right
        [_x1, _y1, 3, 0],   # bottom-right → heading up
        [_x1, _y0, 2, 0],   # top-right    → heading left
        [_x0, _y0, 1, 0],   # top-left     → heading down
        [_x0, _ym, 0, 0],   # back to door → heading right
    ], dtype=np.float64)

    rooms   = floor['rooms']
    n_rooms = len(rooms)

    r_types = np.array([r['rtype'] for r in rooms], dtype=np.int32)

    # Derive bounding boxes from polygon vertices (Mx5: x1 y1 x2 y2 rtype)
    boxes        = []
    r_boundaries = []
    for r in rooms:
        poly = np.array(r['polygon'], dtype=np.float64)  # Px2
        r_boundaries.append(poly)
        x1, y1 = poly[:, 0].min(), poly[:, 1].min()
        x2, y2 = poly[:, 0].max(), poly[:, 1].max()
        boxes.append([x1, y1, x2, y2, float(r['rtype'])])

    box = (np.array(boxes, dtype=np.float64)
           if boxes else np.zeros((0, 5), dtype=np.float64))

    edges_raw = floor.get('edges', [])
    if edges_raw:
        edge = np.array(edges_raw, dtype=np.int32)
        if edge.ndim == 1:
            edge = edge.reshape(1, -1)
        # Pad to 3 columns if only room-pair columns present (no edge-type column)
        if edge.shape[1] == 2:
            edge = np.hstack([edge, np.ones((len(edge), 1), dtype=np.int32)])
    else:
        edge = np.zeros((0, 3), dtype=np.int32)

    entry = types.SimpleNamespace(
        name      = name,
        boundary  = boundary,
        box       = box,
        edge      = edge,
        rEdge     = edge,
        rType     = r_types,
        rBoundary = r_boundaries,
        order     = np.arange(n_rooms, dtype=np.int32),
    )
    data_entries.append(entry)
    name_list.append(name)

    x, y = _compute_tf(boundary)
    train_tf.append({'x': x, 'y': y})

    # rNum: cols 0-12 = per-type room count, col 13 = sum of bedroom-like types
    r_row = np.zeros(14, dtype='uint8')
    for j in range(13):
        r_row[j] = min(255, int((r_types == j).sum()))
    r_row[13] = min(255, int(r_row[[1, 5, 6, 7, 8]].sum()))
    r_num_rows.append(r_row)

    e_num_rows.append(_compute_enum(r_types.tolist(), edges_raw))

# ── Save ───────────────────────────────────────────────────────────────────────
OUT_DIR.mkdir(parents=True, exist_ok=True)

with open(OUT_PKL, 'wb') as f:
    pickle.dump({'data': data_entries, 'nameList': name_list, 'trainTF': train_tf}, f)
print(f'Saved {OUT_PKL}  ({len(data_entries)} entries)')

r_num_arr = np.stack(r_num_rows)
np.save(OUT_RNUM, r_num_arr)
print(f'Saved {OUT_RNUM}  shape={r_num_arr.shape}')

e_num_arr = np.stack(e_num_rows)
with open(OUT_ENUM, 'wb') as f:
    pickle.dump({'eNum': e_num_arr}, f)
print(f'Saved {OUT_ENUM}  shape={e_num_arr.shape}')

# HouseDiffusion — Progress & Handoff Notes

## Goal

Evaluate HouseDiffusion as a short-term replacement for Graph2plan's GNN model, while
GSDiff is evaluated as the longer-term solution. The plan is to integrate whichever
model is chosen via a user toggle or fallback inside the Graph2plan interface.

---

## Current Status

**Inference: working.**
A minimal test script was written and confirmed that the model loads, runs 1000-step
diffusion sampling, and produces a floor plan output. The result looked rough, which
is expected — the pre-trained model was trained on RPLAN (residential) data and was
given a basic test bubble diagram.

**Training: not yet started.** Several prerequisites are still outstanding (see below).

---

## Repository Locations

| Path | Description |
|---|---|
| `C:\Users\hmbashir\source\house_diffusion\` | Local HouseDiffusion repo |
| `scripts\test_minimal.py` | Minimal inference test script (no dataset needed) |
| `ckpts\exp\model250000.pt` | Pre-trained weights (downloaded from Google Drive) |
| `house_diffusion\rplanhg_datasets.py` | RPLAN data loader — reference for writing the ResPlan equivalent |
| `house_diffusion\transformer.py` | Model architecture |
| `house_diffusion\gaussian_diffusion.py` | Diffusion process |
| `C:\Users\hmbashir\source\ResPlan_Dataset\ResPlan.pkl` | ResPlan dataset (17,000 plans, augmented to ~136,000) |
| `C:\Users\hmbashir\source\GSDiff\ANALYSIS.md` | Full GSDiff architecture and integration analysis |

---

## Environment

- **Python**: 3.10 (Windows, venv at `house_diffusion\.venv`)
- **PyTorch**: 2.0.1 from PyTorch official wheel server (`--index-url https://download.pytorch.org/whl/cpu`)
- **NumPy**: 1.26.4 (must stay on 1.x — matplotlib 3.5.1 breaks on NumPy 2.x)
- **GPU torch**: not yet installed — switch `cu118` to match your CUDA version:
  ```
  pip install torch==2.0.1 --index-url https://download.pytorch.org/whl/cu118
  ```
  Check CUDA version with `nvidia-smi`.

### Known dependency issues
- `numpy==1.21.5` (pinned in requirements.txt) is too old — use `1.26.4` instead
- `torch==2.0.0.dev20221212` (pinned in requirements.txt) does not exist on PyPI — use `2.0.1`
- `matplotlib==3.5.1` breaks on NumPy 2.x — stay on NumPy 1.26.4
- `mpi4py` requires Microsoft MPI runtime on Windows for training (see below)

---

## Model Architecture (key facts)

- Single-stage Transformer, `d_model=512`, 4 attention heads, 4 encoder layers
- Three attention types per layer: CSA (per-room), GSA (global), RCA (room-to-door)
- Input: noisy 2D corner coordinates + room type (25D one-hot) + corner index (32D) + room index (32D)
- Output: denoised corner coordinates
- Coordinate range: integers [0, 255] mapped to [-1, 1]
- Max corners per floor plan: 100
- Diffusion steps: 1000 (cosine schedule)
- Trained for 250k steps, batch size 512, on single NVIDIA RTX 6000 (24GB)

---

## Data Format

### What the model expects (per floor plan sample)

```
house_layouts: (100, 94)
    cols 0-1:   x, y coordinates normalised to [-1, 1]
    cols 2-26:  room type one-hot (25D)
    cols 27-58: corner index one-hot (32D)
    cols 59-90: room index one-hot (32D)
    col  91:    padding mask (1 = valid, 0 = padding)
    cols 92-93: connections [this_corner_idx, next_corner_idx]

door_mask:  (100, 100) — RCA attention mask
self_mask:  (100, 100) — CSA attention mask
gen_mask:   (100, 100) — GSA attention mask
graph:      (200, 3)   — room adjacency triples [room_i, +1/-1, room_j]
```

Attention mask convention: **1 = blocked, 0 = attend**.

### ResPlan PKL format (already confirmed)

Each of the 17,000 entries is a dict with keys:
- Room geometry keys: `living`, `kitchen`, `bedroom`, `bathroom`, `balcony`,
  `inner`, `storage`, `front_door`, `door` — each a Shapely `MultiPolygon`
- `graph`: NetworkX Graph where each node has `geometry` (Shapely Polygon),
  `type` (string), `area`. Edge types: `direct`, `adjacency`, `via_door`.
- `unitType`, `id`, `net_area`, `area`, `wall_depth`

Coordinates are in ~0–256 pixel space (same scale as RPLAN).
The graph nodes already split multi-room types into individual rooms
(`bedroom_0`, `bedroom_1`, etc.) — use graph nodes, not dict keys directly.

---

## What Still Needs to Be Done for Training

### 1. Room type mapping decision (first step — blocks everything else)
Define which integer (0–24) each ResPlan room type maps to.
Since training from scratch, you choose the codes — just be consistent.

Suggested starting mapping:
| ResPlan type | Suggested integer | Frequency |
|---|---|---|
| `living`      | 1  | 500/500 |
| `kitchen`     | 2  | 498/500 |
| `bedroom`     | 3  | 500/500 |
| `bathroom`    | 4  | 500/500 |
| `inner`       | 7  | 500/500 — corridor/hallway |
| `balcony`     | 6  | 400/500 |
| `front_door`  | 12 | 500/500 |
| `door`        | 11 | 500/500 — interior door |
| `storage`     | 8  | 48/500  |
| `garden`, `stair`, `veranda`, `parking` | 9 | rare — map to Unknown |

### 2. Write `resplan_dataset.py`
A custom dataset class alongside `rplanhg_datasets.py` that:
- Reads ResPlan PKL
- Iterates graph nodes (each has a single Polygon + type string)
- Maps type string → integer code (from step 1)
- Gets corners from `node['geometry'].exterior.coords`
- Normalises from ~0–256 to [-1, 1]
- Builds `house_layouts`, `door_mask`, `self_mask`, `gen_mask`, `graph`
- Skips plans where total corners exceed 100
- Saves processed data as NPZ for faster subsequent runs

**Key advantage**: ResPlan already has polygon vertices — skip the OpenCV
mask rendering + contour detection that RPLAN requires.

### 3. Adapt `scripts/image_train.py`
Add a `resplan` branch (alongside the existing `rplan` branch) that loads the
new dataset class.

### 4. Fix mpi4py / dist_util for single-GPU Windows training
`dist_util.setup_dist()` uses MPI. For single-GPU training on Windows, either:
- Install Microsoft MPI runtime, OR
- Patch `dist_util.py` to bypass MPI when `WORLD_SIZE=1`

### 5. Install CUDA torch
Replace the CPU torch with a CUDA build matching the available GPU.

---

## Integration Plan with Graph2plan (longer term)

- HouseDiffusion outputs **polygonal loops** (ordered corner coordinates per room)
- Graph2plan expects **bounding boxes**
- An adapter converting polygons → bounding boxes is needed, OR update
  Graph2plan's post-processing to accept polygons directly
- Integration mode: user toggle or fallback (decision pending)

---

## GSDiff (longer-term replacement)

See `C:\Users\hmbashir\source\GSDiff\ANALYSIS.md` for full details.
GSDiff is a two-stage model (node generation + edge prediction) that outperforms
HouseDiffusion but is more complex to integrate. Same output format issue applies.
Sloping wall / non-Manhattan support added in GSDiff-main (not in local GSDiff copy).

---

## Licensing Note

**RPLAN dataset cannot be used** — non-commercial license only.
**ResPlan dataset**: verify its license permits commercial use before committing
to it as the training dataset.

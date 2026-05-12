# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the Application

```bash
cd Interface
python manage.py runserver 0.0.0.0:8888
# Access at http://127.0.0.1:8888/home
```

Ports 8000, 5000, 5174, and 8080 are typically in use in this environment. Use 8888 or another free port.

## Training

```bash
cd Network
python split.py          # generate train/test split
python train.py          # train model (101 epochs, batch 20, Adam)
# checkpoints saved every 5 epochs under experiment/
```

## Batch Post-Processing / Inference

```bash
cd PostProcess
python app.py                     # standalone CLI inference
python test.py                    # test on training data
python test_interface_data.py     # test on boundary images
```

## Architecture Overview

The system has three tiers that work together:

**1. Web Interface** (`Interface/`)  
Django app serving a single-page application. All application state lives as module-level globals in `Houseweb/views.py` (no database ORM used for layout data). Key globals: `last_fp_data`, `boxes_pred`, `user_edited_boundary`, `current_scale`, `steps_run`. Routes are defined in `House/urls.py`. The frontend is ~4400 lines of vanilla JS + D3.js across `static/js/buttonEvent.js` and `static/js/load.js`.

**2. Neural Network Model** (`Interface/model/`)  
Graph2Plan GNN: takes a building boundary image (3-channel 128×128) + room graph (node embeddings + adjacency edges) and outputs a floorplan image + bounding boxes per room. Weights are in `model/model.pth`. Only 4 room types are active in the current application: LivingRoom (0), MasterRoom (1), Kitchen (2), Bathroom (3). The full vocabulary has 18 classes but generating new room types requires retraining.

**3. Post-Processing** (`PostProcess/optimizer/`)  
Heuristic refinement passes run in sequence after generation: AlignWalls → FillWallGaps → SnapRooms → FillLivingRoom → FixRooms → EnforceRoomSizes. Tracked server-side via `steps_run` set (reset on each Generate call). `solver.py` contains the scale-aware room size enforcement logic.

## Key Data Structures

**Boundary array**: Nx4 — columns are `[x, y, dir, isNew]`. When passing to Shapely, always extract only the first two columns: `np.array(boundary)[:, :2]`. Passing the full 4-column array to `shapely.geometry.Polygon()` silently fails.

**Scale**: `current_scale` is stored as **mm per pixel** (mm/px). To convert pixel area to m²: `pixel_area × (scale_mm_per_px / 1000)²`. Set via the SetScale view using either `width_m` or `area_m2` input method.

**Retrieval**: `Interface/retrieval/retrieval.py` uses turning-function fingerprints + FAISS to find K nearest-neighbour training layouts for the given boundary. Results seed the GNN inference.

**Training data**: `Interface/static/Data/data_train_converted.pkl` — each entry has `boundary` (Nx4), `box` (Mx4 room boxes), `rType` (room class integers), `rEdge` (adjacency pairs).

## Graph2plan-dev

`source/Graph2plan-dev/` is a **folder copy** (not a git repo) kept in sync with `Graph2plan` manually. After making changes to `Graph2plan`, the changed files must be copied to the corresponding paths in `Graph2plan-dev`. There is no automated sync.

## Dependencies

- Python 3.9–3.10, PyTorch 2.5.1 + CUDA 12.1 (or CPU)
- Django 3.0, OpenCV, SciPy, Shapely, NumPy, ezdxf
- MATLAB Engine for Python (optional — improves room alignment)
- Full list: `Interface/requirements.txt`

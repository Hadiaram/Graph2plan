# ResPlan Dataset Integration Guide

**Continuation of CUSTOM_DATASET_GUIDE.md**

This guide covers the complete process of integrating your ResPlan dataset with Graph2plan, including handling Shapely geometries, graph structures, and the complete data preparation pipeline.

---

## Table of Contents

1. [ResPlan Dataset Structure](#resplan-dataset-structure)
2. [Enhanced Conversion Script](#enhanced-conversion-script)
3. [Step-by-Step Integration Process](#step-by-step-integration-process)
4. [Creating Train/Test Splits](#creating-traintest-splits)
5. [Running Data Preparation Scripts](#running-data-preparation-scripts)
6. [Output Files and Locations](#output-files-and-locations)
7. [Updating the Interface](#updating-the-interface)
8. [Troubleshooting ResPlan Issues](#troubleshooting-resplan-issues)
9. [Understanding the Conversion Process](#understanding-the-conversion-process)

---

## ResPlan Dataset Structure

### What Makes ResPlan Different?

ResPlan datasets have a unique structure compared to Graph2plan's original RPLAN format:

**ResPlan Structure:**
```python
item = {
    'id': 'floorplan_00123',
    'graph': <NetworkX graph>,      # Room adjacency graph
    'inner': <MultiPolygon>,        # Floorplan boundary
    'front_door': <Polygon>,        # Front door location
    'living': <MultiPolygon>,       # Living room geometries
    'kitchen': <MultiPolygon>,      # Kitchen geometries
    'bedroom': <MultiPolygon>,      # Bedroom geometries
    'bathroom': <MultiPolygon>,     # Bathroom geometries
    # ... other room types
    'wall': <MultiPolygon>,         # Wall elements (excluded)
    'window': <MultiPolygon>,       # Window elements (excluded)
    'door': <MultiPolygon>,         # Door elements (excluded)
}
```

**Graph Structure:**
- Nodes: `"living_0"`, `"bedroom_1"`, `"bathroom_0"`, etc.
- Edges: Adjacency relationships between rooms
- Node names: `{room_type}_{index}` format

**Key Differences from RPLAN:**
1. ❌ No pre-computed `boundary` field → Must extract from `inner`
2. ❌ No `room_types` array → Must parse from graph nodes
3. ❌ No `boxes` array → Must compute from Shapely geometries
4. ❌ No `rBoundary` list → Must extract from room polygons
5. ✅ Has graph structure → Can derive room adjacency
6. ✅ Has Shapely geometries → Can compute everything else

---

## Enhanced Conversion Script

The `convert_resplan_to_mat.py` script has been enhanced with **full ResPlan support** including:

### New Features

#### 1. Shapely Geometry Handling
```python
# Automatically extracts coordinates from Shapely objects
extract_coords_from_shapely(geometry)          # Polygon → numpy array
compute_boundary_from_rooms(room_geometries)   # Union of rooms → boundary
compute_boxes_from_geometries(room_geometries) # Polygons → bounding boxes
```

#### 2. ResPlan Graph Processing
```python
# Parses graph nodes and extracts room information
parse_resplan_node_name("living_0")           # → ("living", 0)
build_room_list_from_graph(item)              # Extracts all valid rooms
compute_edges_from_graph(graph, ...)          # Uses graph edges
```

#### 3. Boundary Alignment
```python
# Aligns boundary with front door
extract_boundary_from_inner(item)             # Extracts from 'inner' field
find_front_door_point(boundary)               # Reorders to start at door
```

#### 4. Room Type Filtering
```python
# Automatically excludes non-space elements
should_include_room(prefix)                   # Filters wall/window/door

# Included: living, kitchen, bedroom, bathroom, dining, balcony, entrance, storage, stair
# Excluded: wall, window, door, interior_door, exterior_wall
```

#### 5. Room Category Mapping
```python
RESPLAN_TO_GRAPH2PLAN = {
    'living': 0,        # LivingRoom
    'bedroom': 1,       # MasterRoom
    'kitchen': 2,       # Kitchen
    'bathroom': 3,      # Bathroom
    'dining': 4,        # DiningRoom
    'balcony': 9,       # Balcony
    'entrance': 10,     # Entrance
    'storage': 11,      # Storage
    'front_door': 15,   # FrontDoor
    'stair': 10,        # Treat as Entrance
}
```

---

## Step-by-Step Integration Process

### Prerequisites

```bash
pip install shapely networkx scipy numpy
```

### Complete Workflow

```
┌─────────────────────────────────────────────────────────────┐
│  ResPlan PKL (Shapely + Graph)                              │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   │ convert_resplan_to_mat.py
                   │ (Auto-detects ResPlan format)
                   ▼
┌─────────────────────────────────────────────────────────────┐
│  data_resplan.mat (Graph2plan format)                       │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   │ create_train_test_split.py
                   ▼
┌─────────────────────────────────────────────────────────────┐
│  data/train.txt + data/test.txt                             │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   │ Data Preparation Scripts 1-6
                   ▼
┌─────────────────────────────────────────────────────────────┐
│  data/ (tf_train.npy, centroids, clusters, etc.)            │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   │ Copy to Interface directories
                   ▼
┌─────────────────────────────────────────────────────────────┐
│  Interface ready to use!                                     │
└─────────────────────────────────────────────────────────────┘
```

---

## Step 1: Convert ResPlan PKL to MAT

### Input File Location
```
C:\Users\hmbashir\source\ResPlan_Dataset\cleaned_resplan.pkl
```

### Conversion Command

```bash
cd C:\Users\hmbashir\source\Graph2plan\DataPreparation

python convert_resplan_to_mat.py --convert "C:\Users\hmbashir\source\ResPlan_Dataset\cleaned_resplan.pkl" --output "C:\Users\hmbashir\source\Graph2plan\DataPreparation\data_resplan.mat"
```

### What Happens During Conversion

1. **Loads ResPlan PKL** - Reads ~17,000 floor plans
2. **Auto-detects ResPlan format** - Checks for `graph` and `inner` fields
3. **For each floor plan:**
   - Extracts boundary from `item['inner']`
   - Aligns boundary with `item['front_door']`
   - Parses graph nodes: `"living_0"` → living room #0
   - Filters out wall/window/door nodes
   - Extracts polygon for each room from MultiPolygons
   - Computes bounding boxes from `.bounds`
   - Extracts room boundaries from `.exterior.coords`
   - Computes edge relations from graph structure
   - Orders rooms by area (largest first)
4. **Saves to MAT format** - Creates Graph2plan-compatible `.mat` file

### Expected Output

```
Loading ResPlan data from: C:\Users\hmbashir\source\ResPlan_Dataset\cleaned_resplan.pkl
Found 16996 floorplans

Converting to Graph2plan format...
  Processed 1000/16996
  Processed 2000/16996
  ...
  Processed 16000/16996

Successfully converted: 16996/16996

Saving to: C:\Users\hmbashir\source\Graph2plan\DataPreparation\data_resplan.mat
✓ Conversion complete!

Next steps:
1. Update DataPreparation/config.py to point to data_resplan.mat
2. Run the data preparation scripts 1-6 in order
```

### Optional: Inspect Before Converting

```bash
python convert_resplan_to_mat.py --inspect "C:\Users\hmbashir\source\ResPlan_Dataset\cleaned_resplan.pkl" --samples 1
```

This shows:
- Data structure
- Available keys
- Graph node names
- Room types present

---

## Step 2: Update Config

The config file has been updated to point to your new dataset:

**File:** `DataPreparation/config.py`
```python
# Original RPLAN dataset path
# data_path = '../Network/data/data.mat'

# Custom ResPlan dataset path
data_path = r'C:\Users\hmbashir\source\Graph2plan\DataPreparation\data_resplan.mat'
```

---

## Creating Train/Test Splits

### What are train.txt and test.txt?

These files tell the data preparation scripts which floor plans to use for:
- **train.txt**: Training data (default: 85% of dataset)
- **test.txt**: Test/validation data (default: 15% of dataset)

Each file contains one floor plan name per line:

**train.txt:**
```
floorplan_00001
floorplan_00003
floorplan_00005
...
```

**test.txt:**
```
floorplan_00002
floorplan_00004
floorplan_00006
...
```

### Generate the Split

```bash
cd C:\Users\hmbashir\source\Graph2plan\DataPreparation

python create_train_test_split.py --train-ratio 0.85
```

**Options:**
- `--train-ratio 0.85`: Use 85% for training (default)
- `--train-ratio 0.90`: Use 90% for training
- `--seed 42`: Random seed for reproducibility (default: 42)

**Output:**
```
Loading data from: C:\Users\hmbashir\source\Graph2plan\DataPreparation\data_resplan.mat
Total floor plans: 16996

Training samples: 14447
Test samples: 2549

Created: ./data/train.txt
Created: ./data/test.txt

First 5 training names:
  floorplan_00123
  floorplan_00456
  floorplan_00789
  floorplan_01234
  floorplan_01567

First 5 test names:
  floorplan_00012
  floorplan_00345
  floorplan_00678
  floorplan_00901
  floorplan_01234

✓ Train/test split complete!
  Train: 14447 (85.0%)
  Test: 2549 (15.0%)
```

---

## Running Data Preparation Scripts

All scripts run from the `DataPreparation` directory and output to `./data/`.

```bash
cd C:\Users\hmbashir\source\Graph2plan\DataPreparation
```

### Script 1: Compute Turn Functions

**What it does:**
- Computes geometric Turn Functions (TF) for all floor plans
- Creates distance matrix between test and train sets
- This is the core of Graph2plan's geometric matching

**Command:**
```bash
python 1.tf_train.py
```

**Output files:**
- `data/trainTF.pkl` - Piecewise TF for training data
- `data/testTF.pkl` - Piecewise TF for test data
- `data/tf_train.npy` - Sampled TF matrix (ntrain × 1000)
- `data/D_test_train.npy` - Distance matrix (ntest × ntrain)

**Expected time:** 10-30 minutes for ~17K floor plans

**Progress output:**
```
Computing training turning functions...
100%|██████████| 14447/14447 [05:23<00:00, 44.67it/s]

Computing test turning functions...
100%|██████████| 2549/2549 [00:58<00:00, 43.89it/s]

Computing turning function distance ... it will take a long time.
100%|██████████| 2549/2549 [25:43<00:00, 1.65it/s]
```

---

### Script 2: Convert Data Format

**What it does:**
- Converts mat format to pickle for faster loading
- Restructures data for the neural network

**Command:**
```bash
python 2.data_train_converted.py
```

**Output files:**
- `data/data_train_converted.mat` - Training data (MATLAB format)
- `data/data_train_converted.pkl` - Training data (pickle format)

**Expected time:** 1-2 minutes

---

### Script 3: Compute Room Counts

**What it does:**
- Counts occurrences of each room type in training data
- Used for statistics and balancing

**Command:**
```bash
python 3.rNum_train.py
```

**Output files:**
- `data/rNum_train.npy` - Room type counts (shape: ntrain × 18)

**Expected time:** < 1 minute

---

### Script 4: Compute Edge Patterns

**What it does:**
- Analyzes edge adjacency patterns between coarse room types
- Creates 5×5 adjacency matrices for each floor plan

**Command:**
```bash
python 4.data_train_eNum.py
```

**Output files:**
- `data/data_train_eNum.pkl` - Edge pattern data

**Expected time:** < 1 minute

---

### Script 5: Convert Test Data (Optional)

**What it does:**
- Converts test data to pickle format
- Only needed if you want to evaluate on test set

**Command:**
```bash
python 5.data_test_converted.py
```

**Output files:**
- `data/data_test_converted.mat`
- `data/data_test_converted.pkl`

**Expected time:** 1-2 minutes

---

### Script 6: Create Clustering Index

**What it does:**
- Creates k-means clusters (1000 centroids) for fast retrieval
- Uses faiss library for efficient similarity search

**Prerequisites:**
```bash
pip install faiss-cpu
# OR for GPU support:
pip install faiss-gpu
```

**Command:**
```bash
python 6.cluster.py
```

**Output files:**
- `data/centroids_train.npy` - 1000 cluster centroids (1000-d TF vectors)
- `data/clusters_train.npy` - 1000 nearest neighbors per centroid

**Expected time:** 5-10 minutes

**Note:** If you don't have a GPU, edit line 27 in `6.cluster.py`:
```python
# Change this:
kmeans = faiss.Kmeans(d, ncentroids, niter=niter, verbose=verbose, gpu=True)

# To this:
kmeans = faiss.Kmeans(d, ncentroids, niter=niter, verbose=verbose, gpu=False)
```

---

## Output Files and Locations

### DataPreparation/data/ Directory

After running all scripts, you should have:

```
DataPreparation/data/
├── train.txt                      # Training floor plan names
├── test.txt                       # Test floor plan names
├── trainTF.pkl                    # Piecewise turn functions (train)
├── testTF.pkl                     # Piecewise turn functions (test)
├── tf_train.npy                   # Sampled TF matrix (ntrain × 1000)
├── D_test_train.npy               # Distance matrix (ntest × ntrain)
├── data_train_converted.mat       # Converted training data (MAT)
├── data_train_converted.pkl       # Converted training data (pickle)
├── rNum_train.npy                 # Room type counts
├── data_train_eNum.pkl            # Edge adjacency patterns
├── data_test_converted.mat        # Converted test data (MAT) [optional]
├── data_test_converted.pkl        # Converted test data (pickle) [optional]
├── centroids_train.npy            # Cluster centroids (1000 × 1000)
└── clusters_train.npy             # Cluster assignments (1000 × 1000)
```

### Files Needed by Interface

The Interface needs these specific files:

**Interface/retrieval/** (for geometric matching):
- `tf_train.npy`
- `centroids_train.npy`
- `clusters_train.npy`

**Interface/static/Data/** (for room generation):
- `data_train_converted.pkl`
- `data_train_eNum.pkl`
- `rNum_train.npy`

---

## Updating the Interface

### Step 1: Copy Generated Files

**Windows (PowerShell):**
```powershell
cd C:\Users\hmbashir\source\Graph2plan\DataPreparation

# Copy to retrieval directory
Copy-Item data\tf_train.npy ..\Interface\retrieval\
Copy-Item data\centroids_train.npy ..\Interface\retrieval\
Copy-Item data\clusters_train.npy ..\Interface\retrieval\

# Copy to static/Data directory
Copy-Item data\data_train_converted.pkl ..\Interface\static\Data\
Copy-Item data\data_train_eNum.pkl ..\Interface\static\Data\
Copy-Item data\rNum_train.npy ..\Interface\static\Data\
```

**Linux/WSL:**
```bash
cd /mnt/c/Users/hmbashir/source/Graph2plan/DataPreparation

# Copy to retrieval directory
cp data/tf_train.npy ../Interface/retrieval/
cp data/centroids_train.npy ../Interface/retrieval/
cp data/clusters_train.npy ../Interface/retrieval/

# Copy to static/Data directory
cp data/data_train_converted.pkl ../Interface/static/Data/
cp data/data_train_eNum.pkl ../Interface/static/Data/
cp data/rNum_train.npy ../Interface/static/Data/
```

### Step 2: Verify Paths in views.py

**File:** `Interface/Houseweb/views.py`

Check these lines (107-134):
```python
# Lines 107-109: Retrieval files
tf_train = np.load(r'C:\Users\hmbashir\source\Graph2plan\Interface\retrieval\tf_train.npy')
centroids = np.load(r'C:\Users\hmbashir\source\Graph2plan\Interface\retrieval\centroids_train.npy')
clusters = np.load(r'C:\Users\hmbashir\source\Graph2plan\Interface\retrieval\clusters_train.npy')

# Lines 129-134: Training data files
train_data = pickle.load(open(r'C:\Users\hmbashir\source\Graph2plan\Interface\static\Data\data_train_converted.pkl', 'rb'))
train_data_eNum = pickle.load(open(r'C:\Users\hmbashir\source\Graph2plan\Interface\static\Data\data_train_eNum.pkl', 'rb'))
train_data_rNum = np.load(r'C:\Users\hmbashir\source\Graph2plan\Interface\static\Data\rNum_train.npy')
```

If paths are incorrect, update them to match your actual file locations.

### Step 3: Test the Interface

```bash
cd C:\Users\hmbashir\source\Graph2plan\Interface

# Activate environment (if using conda)
conda activate g2p_app

# Run Django server
python manage.py runserver 0.0.0.0:8000
```

Open browser: `http://localhost:8000`

---

## Troubleshooting ResPlan Issues

### Issue 1: "No valid rooms found in graph"

**Error:**
```
ValueError: No valid rooms found in graph for floorplan_00123
```

**Cause:** All rooms in the graph are filtered out (e.g., only wall/window/door nodes)

**Solution:**
1. Check what room types exist in your ResPlan data:
   ```bash
   python convert_resplan_to_mat.py --inspect cleaned_resplan.pkl --samples 5
   ```

2. Look at the graph node names. Are they using unexpected prefixes?

3. Add missing room types to `RESPLAN_TO_GRAPH2PLAN` mapping (line 73-85 in `convert_resplan_to_mat.py`):
   ```python
   RESPLAN_TO_GRAPH2PLAN = {
       'living': 0,
       'bedroom': 1,
       'kitchen': 2,
       'bathroom': 3,
       'dining': 4,
       'balcony': 9,
       'study': 6,         # Add if you have study rooms
       'laundry': 11,      # Add if you have laundry rooms
       # ... add more as needed
   }
   ```

---

### Issue 2: "Shapely library required"

**Error:**
```
ImportError: Shapely library required. Install with: pip install shapely
```

**Solution:**
```bash
pip install shapely
```

---

### Issue 3: "NetworkX library required"

**Error:**
```
ImportError: NetworkX library required. Install with: pip install networkx
```

**Solution:**
```bash
pip install networkx
```

---

### Issue 4: Conversion is very slow

**Expected time:** 30-60 minutes for ~17K floor plans

**If taking > 2 hours:**

1. **Check disk I/O** - Saving to network drive? Use local drive
2. **Memory issues** - Close other programs
3. **Add progress tracking** - Edit `convert_resplan_to_mat.py` line 883:
   ```python
   # Change interval from 1000 to 100 for more frequent updates
   if (i + 1) % 100 == 0:
       print(f"  Processed {i + 1}/{len(floorplans)}")
   ```

---

### Issue 5: "room_multipolygon index out of range"

**Error:**
```
Warning: bedroom index 3 out of range
```

**Cause:** Graph node `"bedroom_3"` exists, but `item['bedroom']` only has 0-2 geometries

**Solution:** This is a warning, not an error. The script skips invalid rooms. If too many warnings:

1. Check ResPlan data consistency
2. Verify graph nodes match available geometries
3. Consider filtering the dataset before conversion

---

### Issue 6: faiss installation fails

**Error:**
```
ERROR: Could not find a version that satisfies the requirement faiss
```

**Solution:**

Try these in order:
```bash
# Option 1: CPU version
pip install faiss-cpu

# Option 2: Conda (recommended)
conda install -c pytorch faiss-cpu

# Option 3: GPU version (if you have CUDA)
conda install -c pytorch faiss-gpu
```

If still failing, skip script 6 for now and run it later when faiss is installed.

---

### Issue 7: Memory error during script 1

**Error:**
```
MemoryError: Unable to allocate array
```

**Cause:** Computing distance matrix for large datasets (ntest × ntrain) requires significant RAM

**Solution:**

1. **Reduce test set size** in `create_train_test_split.py`:
   ```bash
   python create_train_test_split.py --train-ratio 0.95  # Only 5% test
   ```

2. **Increase virtual memory** (Windows):
   - Settings → System → About → Advanced system settings
   - Advanced → Performance Settings → Advanced → Virtual memory
   - Set to 2x your RAM size

3. **Chunk processing** - Edit `1.tf_train.py` to process in batches

---

### Issue 8: Front door not aligned correctly

**Symptom:** Boundary starts at wrong point, not near the door

**Cause:** Front door location detection might fail if door is not a polygon

**Solution:**

Edit `extract_boundary_from_inner()` in `convert_resplan_to_mat.py` (line 600-653):

```python
# Add debug logging:
print(f"Front door type: {type(front_door)}")
print(f"Front door centroid: {front_door.centroid}")
```

If front_door is None or empty, boundary will use default ordering (bottom-left).

---

## Understanding the Conversion Process

### How ResPlan is Transformed

```
┌─────────────────────────────────────────────────────────────┐
│ ResPlan Item                                                 │
├─────────────────────────────────────────────────────────────┤
│ • id: "floorplan_00123"                                      │
│ • graph: NetworkX(living_0 ↔ kitchen_0 ↔ bedroom_0)         │
│ • inner: MultiPolygon([polygon1, polygon2])                  │
│ • front_door: Polygon([(x1,y1), (x2,y2), ...])              │
│ • living: MultiPolygon([living_polygon_0])                   │
│ • kitchen: MultiPolygon([kitchen_polygon_0])                 │
│ • bedroom: MultiPolygon([bedroom_polygon_0])                 │
└─────────────────────────────────────────────────────────────┘
                            │
                            │ convert_resplan_to_mat.py
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ Graph2plan Item                                              │
├─────────────────────────────────────────────────────────────┤
│ • name: "floorplan_00123"                                    │
│ • boundary: [[x, y, dir, isNew], ...] (N × 4)               │
│   ↳ Extracted from 'inner', aligned with front_door         │
│                                                              │
│ • rType: [0, 2, 1] (room category indices)                  │
│   ↳ Parsed from graph: living_0→0, kitchen_0→2, bedroom_0→1 │
│                                                              │
│ • rBoundary: [room1_coords, room2_coords, ...]              │
│   ↳ Extracted from polygon.exterior.coords                  │
│                                                              │
│ • gtBox: [[y0,x0,y1,x1], ...] (N × 4)                       │
│   ↳ Computed from polygon.bounds, format converted          │
│                                                              │
│ • gtBoxNew: [[x0,y0,x1,y1], ...] (N × 4)                    │
│   ↳ Same as gtBox, different coordinate order               │
│                                                              │
│ • rEdge: [[u, v, relation], ...] (M × 3)                    │
│   ↳ Extracted from graph.edges(), relation computed         │
│                                                              │
│ • order: [2, 0, 1] (rendering order by area)                │
│   ↳ Sorted by polygon.area (largest first)                  │
└─────────────────────────────────────────────────────────────┘
```

### Boundary Extraction Process

```
Step 1: Get largest polygon from 'inner'
┌─────────────────┐
│  MultiPolygon   │
│  ┌────┐  ┌───┐  │  →  Select largest  →  ┌────────────┐
│  │ P1 │  │P2 │  │                         │  Polygon   │
│  └────┘  └───┘  │                         │    P1      │
└─────────────────┘                         └────────────┘

Step 2: Extract exterior coordinates
┌────────────┐
│  Polygon   │
│   P1       │  →  polygon.exterior.coords  →  [(x0,y0), (x1,y1), ...]
└────────────┘

Step 3: Add direction codes
[(x0,y0), (x1,y1), (x2,y2), (x3,y3)]
    ↓         ↓         ↓         ↓
  →0=R      ↑1=U      ←2=L      ↓3=D

Result: [(x0,y0,0,0), (x1,y1,1,0), (x2,y2,2,0), (x3,y3,3,0)]

Step 4: Align with front door
Find closest point to front_door centroid
Rotate array to start at that point
Mark first 2 points as door: isNew=1

Final: [(x_door,y_door,dir,1), (x_next,y_next,dir,1), ...]
```

### Room Extraction Process

```
For each node in graph.nodes():

1. Parse node name: "living_0" → prefix="living", index=0

2. Check if should include:
   ✓ living → Include (space)
   ✗ wall → Exclude (non-space)

3. Get room type:
   RESPLAN_TO_GRAPH2PLAN["living"] → 0

4. Get polygon:
   item["living"].geoms[0] → Polygon object

5. Extract data:
   • rBoundary: polygon.exterior.coords → [(x,y), ...]
   • box: polygon.bounds → (minx, miny, maxx, maxy)
   • rType: 0

6. Add to room list
```

### Edge Computation Process

```
For each edge in graph.edges():

1. Get edge: ("living_0", "kitchen_0")

2. Map to indices:
   living_0 → room_list[0]
   kitchen_0 → room_list[1]

3. Get boxes:
   box_u = room_list[0]['box'] = [x0, y0, x1, y1]
   box_v = room_list[1]['box'] = [x0, y0, x1, y1]

4. Compute relation:
   Δx = center_v.x - center_u.x
   Δy = center_v.y - center_u.y

   If |Δx| > |Δy|:  # Horizontal
       If Δx > 0: relation = 7 (right-of)
       Else: relation = 2 (left-of)
   Else:  # Vertical
       If Δy > 0: relation = 6 (below)
       Else: relation = 3 (above)

5. Add edge: [u_idx, v_idx, relation]
```

---

## Summary Checklist

Use this checklist to verify each step:

### ☐ Conversion Phase
- [ ] ResPlan pkl file exists at correct path
- [ ] Shapely and networkx installed
- [ ] Ran `convert_resplan_to_mat.py --convert`
- [ ] Conversion completed without errors
- [ ] `data_resplan.mat` file created

### ☐ Train/Test Split
- [ ] Ran `create_train_test_split.py`
- [ ] `data/train.txt` created
- [ ] `data/test.txt` created
- [ ] Split ratio is correct (e.g., 85/15)

### ☐ Data Preparation
- [ ] Ran `1.tf_train.py` → trainTF.pkl, tf_train.npy created
- [ ] Ran `2.data_train_converted.py` → data_train_converted.pkl created
- [ ] Ran `3.rNum_train.py` → rNum_train.npy created
- [ ] Ran `4.data_train_eNum.py` → data_train_eNum.pkl created
- [ ] (Optional) Ran `5.data_test_converted.py`
- [ ] Ran `6.cluster.py` → centroids_train.npy, clusters_train.npy created

### ☐ Interface Update
- [ ] Copied 6 files to Interface directories
- [ ] Verified paths in `views.py`
- [ ] Django server starts without errors
- [ ] Interface loads in browser
- [ ] Can search for similar floor plans

---

## Quick Reference Commands

```bash
# Full workflow from scratch
cd C:\Users\hmbashir\source\Graph2plan\DataPreparation

# 1. Convert ResPlan to MAT
python convert_resplan_to_mat.py --convert "C:\Users\hmbashir\source\ResPlan_Dataset\cleaned_resplan.pkl" --output data_resplan.mat

# 2. Create train/test split
python create_train_test_split.py --train-ratio 0.85

# 3. Run data preparation
python 1.tf_train.py
python 2.data_train_converted.py
python 3.rNum_train.py
python 4.data_train_eNum.py
python 6.cluster.py

# 4. Copy files to Interface
Copy-Item data\tf_train.npy ..\Interface\retrieval\
Copy-Item data\centroids_train.npy ..\Interface\retrieval\
Copy-Item data\clusters_train.npy ..\Interface\retrieval\
Copy-Item data\data_train_converted.pkl ..\Interface\static\Data\
Copy-Item data\data_train_eNum.pkl ..\Interface\static\Data\
Copy-Item data\rNum_train.npy ..\Interface\static\Data\

# 5. Run interface
cd ..\Interface
python manage.py runserver 0.0.0.0:8000
```

---

## Additional Resources

- **Original Guide**: See CUSTOM_DATASET_GUIDE.md for general concepts
- **Script Source**: `convert_resplan_to_mat.py` (lines 411-823)
- **Shapely Docs**: https://shapely.readthedocs.io/
- **NetworkX Docs**: https://networkx.org/documentation/stable/
- **Graph2plan Paper**: SIGGRAPH 2020

---

**Generated**: 2025-12-16
**Author**: Claude Code
**Continuation of**: CUSTOM_DATASET_GUIDE.md
**For**: ResPlan Dataset Integration with Graph2plan

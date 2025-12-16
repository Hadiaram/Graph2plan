# Graph2Plan Custom Dataset Integration Guide

## Complete guide for replacing Graph2plan's RPLAN dataset with your custom dataset

---

## Table of Contents

1. [Overview](#overview)
2. [Do You Need to Retrain?](#do-you-need-to-retrain)
3. [Data Format Requirements](#data-format-requirements)
4. [Workflow Summary](#workflow-summary)
5. [Step-by-Step Instructions](#step-by-step-instructions)
6. [Script Reference](#script-reference)
7. [Troubleshooting](#troubleshooting)

---

## Overview

Graph2plan uses a **two-stage architecture**:

1. **Retrieval Stage** (Geometric)
   - Uses Turn Function (TF) algorithm
   - Matches boundary shapes geometrically
   - **NO neural network involved**
   - Works with any dataset

2. **Generation Stage** (Neural Network)
   - Pre-trained model converts layout graphs to floorplans
   - Works generically on any layout graph
   - **NO retraining needed**

**Conclusion**: You can use your custom dataset **without retraining** the model. You only need to prepare the dataset files for the retrieval system.

---

## Do You Need to Retrain?

**❌ NO** - You do **NOT** need to retrain the neural network to use a custom dataset.

### Why Not?

- **Retrieval** uses geometric Turn Function matching (no ML)
- **Generation** uses a pre-trained model that works on any valid layout graph
- You only need to:
  1. Format your dataset correctly
  2. Generate retrieval index files (TF features, clusters)
  3. Update file paths in the interface

### Key Files to Replace

Located in `Interface/Houseweb/views.py`:

```python
# Lines 107-109: Retrieval system files
tf_train = np.load('Interface/retrieval/tf_train.npy')
centroids = np.load('Interface/retrieval/centroids_train.npy')
clusters = np.load('Interface/retrieval/clusters_train.npy')

# Lines 129-134: Training dataset
train_data = pickle.load(open('Interface/static/Data/data_train_converted.pkl', 'rb'))
train_data_eNum = pickle.load(open('Interface/static/Data/data_train_eNum.pkl', 'rb'))
train_data_rNum = np.load('Interface/static/Data/rNum_train.npy')
```

---

## Data Format Requirements

Your dataset must follow Graph2plan's structure. Each floor plan needs:

### Required Fields

```python
data_item:
    - name: str          # Unique identifier (e.g., 'floorplan_00001')
    - boundary: array    # (N×4): x, y, direction, isNew
                        # First two points = front door
                        # direction: 0=right, 1=down, 2=left, 3=up
                        # isNew: 0=corner, 1=not corner (door point)

    - rType: array       # (M,): Room category indices [0-17]
    - rBoundary: list    # List of (P×2) arrays: boundary points per room
    - gtBox: array       # (M×4): Bounding boxes (y0, x0, y1, x1)
    - gtBoxNew: array    # (M×4): Bounding boxes (x0, y0, x1, y1)
    - rEdge: array       # (K×3): Edges (room_u, room_v, relation_type)
    - order: array       # (M,): Rendering order (larger drawn first)
```

### Room Categories (0-17)

```python
0: LivingRoom    6: StudyRoom      12: Wall-in
1: MasterRoom    7: SecondRoom     13: External
2: Kitchen       8: GuestRoom      14: ExteriorWall
3: Bathroom      9: Balcony        15: FrontDoor
4: DiningRoom   10: Entrance       16: InteriorWall
5: ChildRoom    11: Storage        17: InteriorDoor
```

### Edge Relation Types (0-9)

```python
0: left-above    3: above         6: below        9: right-below
1: left-below    4: inside        7: right-of
2: left-of       5: surrounding   8: right-above
```

---

## Workflow Summary

```text
┌─────────────────────────────────────────────────────────────┐
│ Step 1: Prepare Your Dataset                                │
│ • Recombine split files (if needed)                         │
│ • Convert to data.mat format                                │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 2: Generate Retrieval Files                            │
│ • Run DataPreparation scripts 1-6                           │
│ • Creates: tf_train, centroids, clusters, etc.              │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 3: Update Interface Paths                              │
│ • Edit Interface/Houseweb/views.py                          │
│ • Point to your new data files                              │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 4: Run the Interface                                   │
│ • python manage.py runserver                                │
│ • Your custom dataset is now live!                          │
└─────────────────────────────────────────────────────────────┘
```

---

## Step-by-Step Instructions

### Prerequisites

```bash
# Install required packages
pip install scipy numpy pandas pickle tqdm faiss-cpu

# Or use faiss-gpu for faster clustering
pip install faiss-gpu
```

---

### Step 1: Recombine Split Files (If Applicable)

If your ResPlan dataset is split into ~17,000 separate files:

```bash
cd /mnt/c/Users/hmbashir/source/Graph2plan/DataPreparation

# Preview what will be combined
python recombine_split_dataset.py \
    --input-dir /path/to/split_floorplans \
    --preview

# Recombine into single pickle file
python recombine_split_dataset.py \
    --input-dir /path/to/split_floorplans \
    --output resplan_combined.pkl
```

**Output**: `resplan_combined.pkl` containing all ~16,996 floor plans

---

### Step 2: Convert to data.mat Format

#### 2a. Inspect Your Data Structure

```bash
# If you have a .pkl file
python convert_mat_to_python.py resplan_combined.pkl --explore-pickle

# If you have a .mat file
python convert_mat_to_python.py data.mat --inspect
```

#### 2b. Customize the Conversion Script

Edit `convert_resplan_to_mat.py`, specifically the `convert_item()` function (lines 220-350):

```python
def convert_item(item: Any, index: int) -> Dict[str, Any]:
    """
    Customize this based on YOUR data structure!
    """
    converted = {}

    # Map your fields to Graph2plan format
    # Example for dict:
    converted['name'] = item['id']  # or item.name if object
    converted['boundary'] = item['boundary']
    converted['rType'] = item['room_types']
    # ... etc

    return converted
```

#### 2c. Inspect Your ResPlan Structure First

```bash
python convert_resplan_to_mat.py --inspect resplan_combined.pkl
```

This shows you the exact field names and structure.

#### 2d. Run the Conversion

```bash
python convert_resplan_to_mat.py \
    --convert resplan_combined.pkl \
    --output data.mat
```

**Output**: `data.mat` in Graph2plan format

---

### Step 3: Create Train/Test Split

Create text files listing which floor plans are for training vs testing:

```bash
cd DataPreparation

# Create data directory if it doesn't exist
mkdir -p data

# Example: Put first 15,000 for training, rest for testing
# You'll need to create these based on your data.mat
```

Create `data/train.txt`:

```text
floorplan_00000
floorplan_00001
floorplan_00002
...
floorplan_14999
```

Create `data/test.txt`:

```text
floorplan_15000
floorplan_15001
...
floorplan_16995
```

**Python helper to generate these**:

```python
import scipy.io as sio

# Load your data.mat
data = sio.loadmat('data.mat', squeeze_me=True, struct_as_record=False)['data']

# Get all names
names = [item.name for item in data.flat]

# Split: 90% train, 10% test
split_idx = int(len(names) * 0.9)
train_names = names[:split_idx]
test_names = names[split_idx:]

# Write files
with open('data/train.txt', 'w') as f:
    f.write('\n'.join(train_names))

with open('data/test.txt', 'w') as f:
    f.write('\n'.join(test_names))

print(f"Train: {len(train_names)}, Test: {len(test_names)}")
```

---

### Step 4: Update config.py

```bash
cd DataPreparation
```

Edit `config.py`:

```python
# Point to your data.mat file
data_path = './data.mat'  # Relative path
# or
data_path = 'C:/Users/hmbashir/datasets/data.mat'  # Absolute path
```

---

### Step 5: Run Data Preparation Scripts (1-6)

**IMPORTANT**: Run these **in order**:

#### Script 1: Compute Turn Functions

```bash
python 1.tf_train.py
```

**Creates**:

- `data/trainTF.pkl` - Piecewise turning functions (training)
- `data/testTF.pkl` - Piecewise turning functions (testing)
- `data/tf_train.npy` - Sampled TF (N×1000) for fast matching
- `data/D_test_train.npy` - Distance matrix (test vs train)

**Time**: ~5-30 minutes depending on dataset size

---

#### Script 2: Convert Training Data

```bash
python 2.data_train_converted.py
```

**Creates**:

- `data/data_train_converted.pkl` - Reformatted training data
- `data/data_train_converted.mat` - Same as .mat format

**Time**: ~1-5 minutes

---

#### Script 3: Count Room Types

```bash
python 3.rNum_train.py
```

**Creates**:

- `data/rNum_train.npy` - Room type counts per floor plan (N×14)

**Time**: <1 minute

---

#### Script 4: Compute Edge Numbers

```bash
python 4.data_train_eNum.py
```

**Creates**:

- `data/data_train_eNum.pkl` - Coarse room adjacency patterns (N×25)

**Time**: ~1 minute

---

#### Script 5: Convert Test Data (Optional)

```bash
python 5.data_test_converted.py
```

**Creates**:

- `data/data_test_converted.pkl` - Prepared test boundaries

**Time**: ~1-5 minutes

**Note**: Skip this if you only want to upload custom boundaries via the interface

---

#### Script 6: Create Clustering Index (Recommended)

```bash
# Requires: pip install faiss-cpu (or faiss-gpu)
python 6.cluster.py
```

**Creates**:

- `data/centroids_train.npy` - 1000 cluster centroids (1000×1000)
- `data/clusters_train.npy` - Nearest neighbors per centroid (1000×1000)

**Time**: ~2-10 minutes (faster with GPU)

**Why Important**: Enables fast retrieval. Without this, the system falls back to slow brute-force search.

---

### Step 6: Copy Files to Interface Directory

```bash
cd DataPreparation

# Copy retrieval files
cp data/tf_train.npy ../Interface/retrieval/
cp data/centroids_train.npy ../Interface/retrieval/
cp data/clusters_train.npy ../Interface/retrieval/

# Copy dataset files
cp data/data_train_converted.pkl ../Interface/static/Data/
cp data/data_train_eNum.pkl ../Interface/static/Data/
cp data/rNum_train.npy ../Interface/static/Data/

# Optional: Copy test data
cp data/data_test_converted.pkl ../Interface/static/Data/
```

---

### Step 7: Update Interface File Paths

Edit `Interface/Houseweb/views.py`:

**Lines 107-109**: Update retrieval paths (if different from default)

```python
tf_train = np.load(r'C:\Users\hmbashir\source\Graph2plan\Interface\retrieval\tf_train.npy')
centroids = np.load(r'C:\Users\hmbashir\source\Graph2plan\Interface\retrieval\centroids_train.npy')
clusters = np.load(r'C:\Users\hmbashir\source\Graph2plan\Interface\retrieval\clusters_train.npy')
```

**Lines 129-134**: Update dataset paths

```python
train_data = pickle.load(open(r'C:\Users\hmbashir\source\Graph2plan\Interface\static\Data\data_train_converted.pkl', 'rb'))
train_data_eNum = pickle.load(open(r'C:\Users\hmbashir\source\Graph2plan\Interface\static\Data\data_train_eNum.pkl', 'rb'))
train_data_rNum = np.load(r'C:\Users\hmbashir\source\Graph2plan\Interface\static\Data\rNum_train.npy')
```

**Lines 118-120**: Update test data paths (if applicable)

```python
test_data = pickle.load(open(r'C:\Users\hmbashir\source\Graph2plan\Interface\static\Data\data_test_converted.pkl', 'rb'))
```

---

### Step 8: Run the Interface

```bash
cd /mnt/c/Users/hmbashir/source/Graph2plan

# Activate environment
conda activate g2p_app

# Run Django server
python manage.py runserver 0.0.0.0:8000
```

**Open browser**: <http://127.0.0.1:8000/home>

Your custom dataset is now live! 🎉

---

## Script Reference

### 1. `recombine_split_dataset.py`

**Purpose**: Combine multiple split floor plan files into a single dataset

**Usage**:

```bash
# Preview
python recombine_split_dataset.py --input-dir ./split_files --preview

# Recombine
python recombine_split_dataset.py --input-dir ./split_files --output combined.pkl

# Multiple output formats
python recombine_split_dataset.py --input-dir ./split_files \
    --output combined.pkl --also-json --also-mat
```

**Supported Formats**: .pkl, .json, .npy, .npz, .mat

---

### 2. `convert_resplan_to_mat.py`

**Purpose**: Convert ResPlan dataset to Graph2plan data.mat format

**Usage**:

```bash
# Inspect structure
python convert_resplan_to_mat.py --inspect resplan.pkl

# Convert
python convert_resplan_to_mat.py --convert resplan.pkl --output data.mat
```

**Customize**: Edit `convert_item()` function to map your data fields

---

### 3. `convert_mat_to_python.py`

**Purpose**: Convert .mat files to Python-friendly formats

**Usage**:

```bash
# Inspect a .mat file
python convert_mat_to_python.py data.mat --inspect

# Convert to pickle
python convert_mat_to_python.py data.mat --to-pickle data.pkl

# Convert to JSON (human-readable)
python convert_mat_to_python.py data.mat --to-json data.json

# Convert to CSV (metadata table)
python convert_mat_to_python.py data.mat --to-csv metadata.csv

# Explore a pickle file
python convert_mat_to_python.py data.pkl --explore-pickle
```

**Output Formats**:

- `.pkl` - Python native, preserves all data types
- `.json` - Human-readable, universal
- `.npz` - Numpy archive, efficient
- `.csv` - Flat metadata table

---

### 4. Data Preparation Scripts (1-6)

**Purpose**: Generate retrieval index and dataset files

**Configuration**: Edit `config.py` to point to your `data.mat`

**Execution Order**:

```bash
python 1.tf_train.py        # Turn Functions (REQUIRED)
python 2.data_train_converted.py  # Convert data (REQUIRED)
python 3.rNum_train.py      # Room counts (REQUIRED)
python 4.data_train_eNum.py # Edge patterns (REQUIRED)
python 5.data_test_converted.py   # Test data (OPTIONAL)
python 6.cluster.py         # Fast search index (RECOMMENDED)
```

**Minimum Required**: Scripts 1, 2, 3, 4

**Full Functionality**: All scripts 1-6

---

## Troubleshooting

### Issue: "Turn function computation is slow"

**Solution**:

- Use a smaller dataset for testing first
- The TF computation is O(N²) for test-train distance matrix
- Consider using only `retrieve_cluster()` instead of `retrieve_bf()`

---

### Issue: "Clustering script fails"

**Problem**: `faiss` not installed or incompatible

**Solution**:

```bash
# CPU version
pip install faiss-cpu

# GPU version (faster)
pip install faiss-gpu

# If still fails, skip script 6 and use brute-force retrieval
```

---

### Issue: "Data structure doesn't match"

**Problem**: Your ResPlan format differs from expected

**Solution**:

1. Run inspect mode:

   ```bash
   python convert_resplan_to_mat.py --inspect your_data.pkl
   ```

2. Check the output carefully
3. Update the `convert_item()` function in `convert_resplan_to_mat.py`
4. Map your field names to Graph2plan's expected fields

---

### Issue: "Room boundaries are missing"

**Problem**: `rBoundary` field not in your data

**Solution**:

- If you have bounding boxes, you can create simple rectangular boundaries:

  ```python
  def box_to_boundary(box):
      x0, y0, x1, y1 = box
      return np.array([
          [x0, y0],
          [x1, y0],
          [x1, y1],
          [x0, y1]
      ])

  converted['rBoundary'] = [box_to_boundary(box) for box in boxes]
  ```

---

### Issue: "Edge relations are incorrect"

**Problem**: Edge computation produces wrong relationships

**Solution**:

- The script includes automatic edge computation (`compute_edge_relations()`)
- If your data has pre-computed edges, use those instead
- Verify edge types match Graph2plan's definitions (0-9)

---

### Issue: "Interface shows wrong floor plans"

**Problem**: File paths in views.py are incorrect

**Solution**:

1. Check paths in `Interface/Houseweb/views.py` lines 107-134
2. Use absolute paths for clarity:

   ```python
   train_data = pickle.load(open(r'C:\full\path\to\data_train_converted.pkl', 'rb'))
   ```

3. Verify files exist at those paths
4. Restart Django server after changes

---

## File Locations Reference

### Data Preparation Scripts

```text
Graph2plan/DataPreparation/
├── convert_resplan_to_mat.py      # ResPlan → data.mat
├── convert_mat_to_python.py       # .mat → .pkl/.json/.csv
├── recombine_split_dataset.py     # Combine split files
├── config.py                      # Data path configuration
├── 1.tf_train.py                  # Compute turn functions
├── 2.data_train_converted.py      # Convert training data
├── 3.rNum_train.py                # Count room types
├── 4.data_train_eNum.py           # Compute edge patterns
├── 5.data_test_converted.py       # Convert test data
└── 6.cluster.py                   # Create search index
```

### Generated Files

```text
Graph2plan/DataPreparation/data/
├── train.txt                      # Training split names
├── test.txt                       # Test split names
├── trainTF.pkl                    # Turn functions (train)
├── testTF.pkl                     # Turn functions (test)
├── tf_train.npy                   # Sampled TF (N×1000)
├── D_test_train.npy               # Distance matrix
├── data_train_converted.pkl       # Converted training data
├── data_test_converted.pkl        # Converted test data
├── rNum_train.npy                 # Room type counts
├── data_train_eNum.pkl            # Edge patterns
├── centroids_train.npy            # Cluster centroids
└── clusters_train.npy             # Cluster assignments
```

### Interface Files

```text
Graph2plan/Interface/
├── Houseweb/views.py              # Main backend (UPDATE PATHS HERE)
├── retrieval/
│   ├── tf_train.npy               # Copy here from DataPreparation/data/
│   ├── centroids_train.npy        # Copy here
│   └── clusters_train.npy         # Copy here
└── static/Data/
    ├── data_train_converted.pkl   # Copy here from DataPreparation/data/
    ├── data_test_converted.pkl    # Copy here
    ├── data_train_eNum.pkl        # Copy here
    └── rNum_train.npy             # Copy here
```

---

## Quick Reference Commands

```bash
# === STEP 1: Recombine split files ===
python recombine_split_dataset.py \
    --input-dir /path/to/split_files \
    --output resplan_combined.pkl

# === STEP 2: Inspect your data ===
python convert_mat_to_python.py resplan_combined.pkl --explore-pickle

# === STEP 3: Convert to data.mat ===
python convert_resplan_to_mat.py --inspect resplan_combined.pkl
# Edit convert_item() function as needed
python convert_resplan_to_mat.py --convert resplan_combined.pkl --output data.mat

# === STEP 4: Update config ===
echo "data_path = './data.mat'" > config.py

# === STEP 5: Create train/test split ===
# (Write Python script to generate train.txt and test.txt)

# === STEP 6: Run data preparation ===
python 1.tf_train.py
python 2.data_train_converted.py
python 3.rNum_train.py
python 4.data_train_eNum.py
python 5.data_test_converted.py  # Optional
python 6.cluster.py              # Requires faiss

# === STEP 7: Copy files to interface ===
cp data/tf_train.npy ../Interface/retrieval/
cp data/centroids_train.npy ../Interface/retrieval/
cp data/clusters_train.npy ../Interface/retrieval/
cp data/data_train_converted.pkl ../Interface/static/Data/
cp data/data_train_eNum.pkl ../Interface/static/Data/
cp data/rNum_train.npy ../Interface/static/Data/

# === STEP 8: Update paths in views.py ===
# Edit Interface/Houseweb/views.py lines 107-134

# === STEP 9: Run interface ===
cd ../..
conda activate g2p_app
python manage.py runserver 0.0.0.0:8000
```

---

## Summary

✅ **No retraining needed** - Graph2plan's architecture separates retrieval (geometric) from generation (pre-trained)

✅ **Replace dataset files** - Generate new TF features, clusters, and converted data

✅ **Update paths** - Point Interface/Houseweb/views.py to your new files

✅ **Use provided scripts** - Three custom scripts handle all conversions and recombination

✅ **Run data preparation** - Scripts 1-6 generate all required retrieval files

✅ **Test interface** - Your custom dataset is now searchable!

---

## Questions or Issues?

- Check the [Troubleshooting](#troubleshooting) section
- Review the [Script Reference](#script-reference)
- Verify all [file paths](#file-locations-reference) are correct
- Ensure all [data preparation scripts](#4-data-preparation-scripts-1-6) ran successfully

**Contact**:

- Ruizhen Hu: <ruizhen.hu@gmail.com>
- Zeyu Huang: <vcchzy@gmail.com>
- Yuhan Tang: <yuhantang55@gmail.com>

---

**Generated**: 2025-12-16
**Graph2plan**: <https://vcc.tech/research/2020/Graph2Plan>
**Paper**: SIGGRAPH 2020

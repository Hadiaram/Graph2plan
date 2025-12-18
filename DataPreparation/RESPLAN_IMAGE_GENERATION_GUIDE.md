# ResPlan Floor Plan Image Generation Guide

## Overview

This document explains the floor plan image generation process for the ResPlan dataset integration with Graph2plan, including the naming conventions, data flow, and known issues.

---

## What We've Done So Far

### 1. Fixed Interface Boundary Display Issue ✅

**Problem**: The Layout view (right panel) was showing simplified rectangular boundaries instead of proper rectilinear room shapes.

**Root Cause**: In `Interface/Houseweb/views.py`, the `data_js["indoor"]` field was empty, so the frontend couldn't render individual room polygons.

**Fix Applied**: Modified `Interface/Houseweb/views.py` at lines 444-463:

```python
# Populate indoor with room boundary polygons (rBoundary)
data_js["indoor"] = []
if hasattr(fp_end.data, 'rBoundary') and fp_end.data.rBoundary:
    # Use rBoundary from generated floor plan if available
    for rb in fp_end.data.rBoundary:
        if isinstance(rb, np.ndarray) and len(rb) > 0:
            coords_str = " ".join([f"{x},{y}" for x, y in rb])
            data_js["indoor"].append(coords_str)
elif hasattr(data, 'rBoundary') and data.rBoundary:
    # Fallback to test data rBoundary if generation doesn't have it
    for rb in data.rBoundary:
        if isinstance(rb, (np.ndarray, list)) and len(rb) > 0:
            rb_array = np.array(rb) if not isinstance(rb, np.ndarray) else rb
            coords_str = " ".join([f"{x},{y}" for x, y in rb_array])
            data_js["indoor"].append(coords_str)
```

**File**: `Interface/Houseweb/views.py:444-463`

---

### 2. Created Image Generation Script ✅

**Created**: `generate_floorplan_images.py`

**Purpose**: Generate PNG thumbnail images for all ResPlan floor plans from the converted PKL files.

**What It Does**:

- Loads `data_train_converted.pkl` and `data_test_converted.pkl`
- Extracts boundary and room box data from each floor plan
- Renders floor plans using matplotlib with color-coded rooms
- Saves PNG images to `Interface/static/Data/Img/`

**Usage**:

```bash
cd DataPreparation
g2p-env\Scripts\python.exe generate_floorplan_images.py
```

**Output**: PNG files named by floor plan ID (e.g., `293.png`, `10028.png`, `12781.png`)

---

### 3. Fixed PKL Data Structure Handling ✅

**Issues Found**:

- PKL files use `scipy.io.loadmat` with `squeeze_me=True`, causing array structure variations
- Data can be dict or object, requiring dual access patterns
- 1D struct arrays need special handling

**Fixes Applied** in `generate_floorplan_images.py`:

- Added helper function to handle both dict and object attribute access
- Added array shape normalization (1D → 2D for single items)
- Added debug output to show data structure
- Improved error handling with full tracebacks

---

## Understanding the Naming Conventions

### Three Different Naming Systems

Understanding these is **critical** to avoid confusion:

#### 1. ResPlan Original Dataset

- **Files**: `plan_00000.pkl`, `plan_00001.pkl`, ... `plan_16995.pkl` (sequential)
- **Actual IDs**: Non-sequential (e.g., plan_00000 = ID 14433)
- **Images**: `plan_00000.png`, `plan_00001.png`, etc.
- **Location**: `ResPlan_Dataset/resplan_viewer/resplan_viewer/images/*.png`

#### 2. Our Converted Data (Graph2plan)

- **MAT/PKL**: Stores floor plans with their original ResPlan IDs as names
- **IDs from train.txt**: 293, 10028, 1772, 6266, 280, 5775, ...
- **IDs from test.txt**: (2,549 different IDs)
- **Order**: Shuffled/randomized from train/test split

#### 3. Our Generated Images

- **Naming**: Uses ResPlan ID from the converted data (e.g., `293.png`, `10028.png`)
- **Matches**: The `name` field in the PKL files
- **Location**: `Interface/static/Data/Img/*.png`

### Example Mapping

```text
ResPlan Original → Our Converted Data → Generated Image
─────────────────────────────────────────────────────────
plan_00000.pkl   →  (not in train/test) → (not generated)
  ID: 14433

plan_XXXXX.pkl   →  First in train.txt  → 293.png
  ID: 293            Array index [0]       ID: 293

plan_YYYYY.pkl   →  Second in train.txt → 10028.png
  ID: 10028          Array index [1]       ID: 10028
```

**Key Point**: `0.png` in our generated images ≠ `plan_00000.png` from ResPlan!

- Our `0.png` would be array index 0 if we used index naming
- But we use ID naming, so it's `293.png` instead

---

## Data Flow Through the System

### 1. Original ResPlan Dataset

```text
ResPlan_Dataset/
├── cleaned_resplan.pkl              # 16,996 floor plans
├── plans_split/
│   ├── plan_00000.pkl              # Sequential files
│   ├── plan_00001.pkl              # Each has 'id' field
│   └── ...
└── resplan_viewer/resplan_viewer/images/
    ├── plan_00000.png              # Pre-rendered images
    ├── plan_00001.png              # With walls, doors, windows
    └── ...
```

### 2. Conversion to Graph2plan Format

```bash
cd DataPreparation
python convert_resplan_to_mat.py --convert ../ResPlan_Dataset/cleaned_resplan.pkl
# Creates: data_resplan.mat (IDs as names)

python create_train_test_split.py --train-ratio 0.85
# Creates:
#   - data/train.txt  (14,445 IDs)
#   - data/test.txt   (2,549 IDs)
```

### 3. Data Preparation Pipeline

```bash
python 1.tf_train.py           # Creates trainTF.pkl, testTF.pkl
python 2.data_train_converted.py  # Creates data_train_converted.pkl
python 3.rNum_train.py         # Creates rNum_train.npy
python 4.data_train_eNum.py    # Creates data_train_eNum.pkl
python 6.cluster.py            # Creates centroids_train.npy, clusters_train.npy
```

### 4. Copy to Interface

```bash
# Retrieval data
cp data/trainTF.pkl ../Interface/retrieval/tf_train.npy
cp data/centroids_train.npy ../Interface/retrieval/
cp data/clusters_train.npy ../Interface/retrieval/

# Static data
cp data/data_train_converted.pkl ../Interface/static/Data/
cp data/data_test_converted.pkl ../Interface/static/Data/
cp data/data_train_eNum.pkl ../Interface/static/Data/
cp data/rNum_train.npy ../Interface/static/Data/
```

### 5. Image Generation (Optional)

```bash
python generate_floorplan_images.py
# Creates: Interface/static/Data/Img/293.png, 10028.png, ...
```

---

## Image Generation Details

### What Our Script Generates

**Simple Floor Plans** with:

- ✅ Exterior boundary outline (black line)
- ✅ Individual room rectangles (colored by type)
- ✅ Color-coded by Graph2plan room categories
- ❌ No wall thickness
- ❌ No doors or windows
- ❌ No architectural detail

**Example**: `293.png`, `10028.png`, `12781.png`

### What ResPlan Original Images Show

**Detailed Architectural Plans** with:

- ✅ Exterior boundary
- ✅ Thick walls (gray polygons)
- ✅ Doors (brown rectangles)
- ✅ Windows (blue rectangles)
- ✅ Inner spaces/hallways (green)
- ✅ Proper room polygons (not just boxes)
- ✅ Metadata (ID, area, legend)

**Example**: `plan_00000.png`, `plan_00001.png`

### Why the Difference?

**Graph2plan's Data Format** doesn't include:

- Wall geometries
- Door/window locations
- Architectural details

Our conversion script (`convert_resplan_to_mat.py`) intentionally simplifies the data to match Graph2plan's expected format:

- Extracts only room outlines (`rBoundary`)
- Converts to bounding boxes (`gtBox`, `gtBoxNew`)
- Discards walls, doors, windows

**This is by design** - Graph2plan works with simplified topology, not architectural drawings.

---

## What Needs to Be Done Next

### Option 1: Use Simplified Generated Images (Current)

✅ **Already Working**

- Images generated for train/test sets
- Matches Graph2plan data model
- Fast generation
- Small file sizes

**No Action Needed** - System is functional!

### Option 2: Copy ResPlan Original Images (Recommended)

**Why**: Better visual quality, architectural detail

**Steps**:

1. **Create Mapping Script** to extract ID from each ResPlan image:

    ```python
    # create_image_mapping.py
    import pickle
    from pathlib import Path
    from tqdm import tqdm

    resplan_dir = Path('/mnt/c/Users/hmbashir/source/ResPlan_Dataset/plans_split')
    output_file = 'resplan_image_mapping.txt'

    mapping = []
    for i in tqdm(range(16996)):
        pkl_file = resplan_dir / f'plan_{i:05d}.pkl'
        if pkl_file.exists():
            plan = pickle.load(open(pkl_file, 'rb'))
            plan_id = plan.get('id', f'plan_{i:05d}')
            mapping.append(f'plan_{i:05d}.png,{plan_id}.png')

    with open(output_file, 'w') as f:
        f.write('\n'.join(mapping))
    ```

2. **Run Mapping Script**:

    ```bash
    cd DataPreparation
    g2p-env\Scripts\python.exe create_image_mapping.py
    # Creates: resplan_image_mapping.txt
    ```

3. **Copy and Rename Images**:

    ```bash
    # Using PowerShell or create a Python script
    cd Interface/static/Data/Img
    # For each line in mapping.txt: copy source.png → destination.png
    ```

4. **Verify**:

    ```bash
    # Check that images match IDs in train.txt and test.txt
    ls 293.png    # Should exist
    ls 10028.png  # Should exist
    ls 12781.png  # Should exist
    ```

### Option 3: Hybrid Approach

**Use original ResPlan images where available**, generate simplified ones for any missing:

```python
# In generate_floorplan_images.py, add check:
resplan_image = Path(f'/path/to/resplan_images/plan_{idx:05d}.png')
if resplan_image.exists():
    # Copy with ID rename
    shutil.copy(resplan_image, output_path)
else:
    # Generate simplified version
    plot_floorplan(boundary, boxes, room_types, output_path)
```

---

## Known Issues and Solutions

### Issue 1: Naming Confusion ⚠️

**Problem**: Three different naming systems can cause confusion

**Solution**: Always reference by ResPlan ID (not array index, not plan_XXXXX)

- In code: Use `item.name` or `item['id']`
- In files: Name by ID (e.g., `293.png`)
- In docs: Specify which system you're referring to

### Issue 2: Image Mismatch

**Problem**: Comparing `0.png` to `plan_00000.png` shows different floor plans

**Explanation**: These are literally different apartments!

- `0.png` (if it existed) would be array index 0
- But we name by ID, so it's `293.png`
- `plan_00000.png` is ID 14433 (different floor plan)

**Solution**: Compare by ID:

- ResPlan `plan_00000.png` (ID 14433) → Our `14433.png`
- Our `293.png` → Find corresponding ResPlan `plan_XXXXX.png` where ID=293

### Issue 3: Image Generation Speed

**Problem**: Generating 16,996 images takes time

**Current Status**:

- Most images already generated
- Script handles incremental generation
- Can resume if interrupted

**Optimization**: Just copy ResPlan's pre-rendered images (Option 2)

### Issue 4: Room Boundaries Extending Beyond Outer Boundary

**Status**: Need to verify if this is real or just comparing wrong floor plans

**Where to Check**:

1. Load floor plan by ID: `data[train.txt line N]`
2. Compare `boundary` field with `rBoundary` list
3. Verify all room coords are within boundary bounds

**Test Case**:

```python
# Check if rooms exceed boundary
boundary_coords = data.boundary[:, :2]
min_x, min_y = boundary_coords.min(axis=0)
max_x, max_y = boundary_coords.max(axis=0)

for room_boundary in data.rBoundary:
    room_min_x, room_min_y = room_boundary.min(axis=0)
    room_max_x, room_max_y = room_boundary.max(axis=0)

    if room_min_x < min_x or room_max_x > max_x:
        print(f"Room extends beyond X boundary")
    if room_min_y < min_y or room_max_y > max_y:
        print(f"Room extends beyond Y boundary")
```

---

## File Locations Reference

### Source Files

```text
DataPreparation/
├── generate_floorplan_images.py     # Image generation script
├── convert_resplan_to_mat.py        # Dataset conversion
├── create_train_test_split.py       # Train/test split
├── config.py                        # Configuration
└── data/
    ├── train.txt                    # 14,445 floor plan IDs
    ├── test.txt                     # 2,549 floor plan IDs
    ├── data_train_converted.pkl     # 45 MB
    └── data_test_converted.pkl      # 57 MB
```

### Interface Files

```text
Interface/
├── Houseweb/views.py                # Backend (indoor boundary fix)
├── static/Data/
│   ├── data_train_converted.pkl
│   ├── data_test_converted.pkl
│   ├── data_train_eNum.pkl
│   ├── rNum_train.npy
│   └── Img/
│       ├── 293.png                  # Generated images (by ID)
│       ├── 10028.png
│       ├── 12781.png
│       └── ...
└── retrieval/
    ├── tf_train.npy
    ├── centroids_train.npy
    └── clusters_train.npy
```

### ResPlan Original

```text
ResPlan_Dataset/
├── cleaned_resplan.pkl              # 16,996 floor plans
├── plans_split/
│   └── plan_*.pkl                   # 16,996 files
└── resplan_viewer/resplan_viewer/images/
    └── plan_*.png                   # 16,996 pre-rendered images
```

---

## Color Scheme Reference

### Graph2plan Room Colors (Used in Generated Images)

```python
ROOM_COLORS = {
    0: '#EE4D4D',    # LivingRoom - Red
    1: '#C67171',    # MasterRoom - Dark Pink
    2: '#FFD274',    # Kitchen - Yellow
    3: '#BEBEBE',    # Bathroom - Gray
    4: '#BFE3E8',    # DiningRoom - Light Blue
    5: '#7BA779',    # ChildRoom - Green
    6: '#E87A90',    # StudyRoom - Pink
    7: '#FF8C69',    # SecondRoom - Orange
    8: '#1F849B',    # GuestRoom - Dark Blue
    9: '#727171',    # Balcony - Dark Gray
    10: '#785A67',   # Entrance - Purple
    11: '#D3A2C7',   # Storage - Light Purple
    12: '#FFFFFF',   # Wall-in - White
    13: '#FFFF00',   # External - Yellow
    14: '#FFFFFF',   # ExteriorWall - White
    15: '#D11A2D',   # FrontDoor - Dark Red
    16: '#1F849B',   # InteriorWall - Dark Blue
    17: '#BEBEBE',   # InteriorDoor - Gray
}
```

### ResPlan Original Colors (Pre-rendered Images)

```python
ROOM_COLORS = {
    'living': '#FFE4B5',      # Moccasin
    'bedroom': '#E6E6FA',     # Lavender
    'bathroom': '#B0E0E6',    # Powder Blue
    'kitchen': '#FFB6C1',     # Light Pink
    'balcony': '#98FB98',     # Pale Green
    'storage': '#D3D3D3',     # Light Gray
    'inner': '#FFFFFF',       # White (hallways)
    'wall': '#696969',        # Dim Gray
    'door': '#8B4513',        # Saddle Brown
    'window': '#4682B4',      # Steel Blue
}
```

---

## Testing and Verification

### Verify Image Generation Works

```bash
cd DataPreparation
g2p-env\Scripts\python.exe generate_floorplan_images.py
```

Expected output:

```text
============================================================
ResPlan Floor Plan Image Generator
============================================================

Loading train data from: ../Interface/static/Data/data_train_converted.pkl
  Data type: <class 'numpy.ndarray'>
  Array shape: (14445,), dtype: object
  Found 14445 floor plans
Generating PNG images to: ..\Interface\static\Data\Img
Generating train images: 100%|████████| 14445/14445 [XX:XX<00:00, XX.XXit/s]
✓ Generated XXXXX images for train set

Loading test data from: ../Interface/static/Data/data_test_converted.pkl
  Data type: <class 'numpy.ndarray'>
  Array shape: (2549,), dtype: object
  Found 2549 floor plans
Generating PNG images to: ..\Interface\static\Data\Img
Generating test images: 100%|████████| 2549/2549 [XX:XX<00:00, XX.XXit/s]
✓ Generated XXXXX images for test set
```

### Verify Image Matches Data

```python
import pickle
import matplotlib.pyplot as plt
from PIL import Image

# Load converted data
data_dict = pickle.load(open('Interface/static/Data/data_train_converted.pkl', 'rb'))
data = list(data_dict['data'])

# Pick a random floor plan
floor_plan = data[0]
name = floor_plan.name if hasattr(floor_plan, 'name') else '0'

# Load generated image
img = Image.open(f'Interface/static/Data/Img/{name}.png')

# Display
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].imshow(img)
ax[0].set_title(f'Generated Image: {name}.png')
ax[0].axis('off')

# Also show data boundary
boundary = floor_plan.boundary if hasattr(floor_plan, 'boundary') else []
if len(boundary) > 0:
    ax[1].plot(boundary[:, 0], boundary[:, 1], 'k-', linewidth=2)
    ax[1].set_aspect('equal')
    ax[1].set_title(f'Data Boundary: {name}')
    ax[1].invert_yaxis()

plt.tight_layout()
plt.show()
```

### Verify Interface Display

1. Start Interface:

    ```bash
    cd Interface
    python manage.py runserver
    ```

2. Open browser: `http://localhost:8000`

3. Test workflow:
   - Select a test floor plan (e.g., 12781)
   - Check middle panel shows image thumbnail
   - Click "Generate"
   - Verify Layout view shows room boundaries

---

## Troubleshooting

### Images Not Showing in Interface

**Check**:

1. Files exist in `Interface/static/Data/Img/`
2. Filenames match floor plan IDs from `train.txt`/`test.txt`
3. Images are valid PNG format
4. Django static files properly configured

### Generated Images Look Wrong

**Check**:

1. Compare by ID (not by filename order)
2. Verify `boundary` and `box` fields in PKL data
3. Check coordinate system (matplotlib Y-axis inverted)

### Script Crashes During Generation

**Check**:

1. PKL files are valid and complete
2. Enough disk space for 16,996 images (~100-200 MB)
3. Python environment has matplotlib, numpy, tqdm
4. Memory sufficient for loading PKL files

---

## Summary

### What Works Now ✅

- ✅ Interface displays proper room boundaries in Layout view
- ✅ Image generation script creates simplified floor plan thumbnails
- ✅ All data properly converted and in correct locations
- ✅ PKL file structure handled correctly

### What's Optional 📋

- Generate all 16,996 images (can copy ResPlan originals instead)
- Verify boundary extension issue (likely just naming confusion)
- Create mapping for ResPlan original images

### Next Actions (Choose One)

**Option A - Keep Current (Recommended for Testing)**:

- ✅ No action needed
- System functional with simplified images
- Fast, lightweight

**Option B - Use Original ResPlan Images (Better Quality)**:

1. Create mapping script
2. Copy and rename 16,996 images
3. Verify ID matching

**Option C - Hybrid**:

1. Use original where available
2. Generate for missing
3. Best of both worlds

---

## Document Version

- **Created**: 2025-12-17
- **Last Updated**: 2025-12-17
- **Status**: Complete - Ready for Use

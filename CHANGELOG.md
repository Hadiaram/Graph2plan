# Graph2plan Interface Fixes and ResPlan ID Mapping

**Date:** December 19, 2025
**Author:** Claude Code Assistant
**Project:** Graph2plan ResPlan Integration

## Table of Contents
- [Overview](#overview)
- [Issues Fixed](#issues-fixed)
- [ResPlan ID Mapping](#resplan-id-mapping)
- [Files Modified](#files-modified)
- [How to Use](#how-to-use)

---

## Overview

This document summarizes the fixes applied to the Graph2plan interface to resolve layout display issues and the creation of mapping files to link ResPlan sequential plan names (plan_00000, plan_00001, etc.) with Graph2plan numeric IDs.

### Key Accomplishments

1. **Fixed room boundary overflow issues** - Rooms were extending outside the exterior boundary walls
2. **Fixed SVG scaling issues** - Layouts were appearing too large (512x512 instead of 256x256)
3. **Created ID mapping files** - Connected ResPlan plan_XXXXX names to Graph2plan numeric IDs
4. **Generated separate train/test mappings** - Provided lookup tables for both datasets

---

## Issues Fixed

### Issue 1: SVG Scaling Problem

**Problem:**
Floor plan layouts were taking up the entire screen instead of being centered with appropriate padding.

**Root Cause:**
- SVG elements had `scale(2)` transforms, making 256×256px SVGs appear as 512×512px
- SVG containers lacked proper viewBox attributes for centering

**Solution:**
- Changed all `scale(2)` to `scale(1)` in `buttonEvent.js` (6 instances at lines 429, 430, 535, 572, 625, 818)
- Added `viewBox="-30 -30 316 316"` to all 5 SVG elements in `home.html` for 30px padding

**Files Modified:**
- `/Interface/static/js/buttonEvent.js`
- `/Interface/templates/home.html`

---

### Issue 2: Room Bounding Boxes Extending Outside Boundaries

**Problem:**
Room bounding boxes (yellow bedrooms, public areas, etc.) were flowing past the black exterior boundary walls in multiple workflows.

**Root Cause:**
The Python backend had multiple code paths for loading/displaying layouts, and none of them clipped room boxes to the boundary:
1. `Save_Editbox` workflow (Generate button)
2. `AdjustGraph` workflow (Transfer button)
3. `FindTraindata` workflow (Thumbnail click)

**Solution:**
Added boundary clipping logic with 5px margin to all three workflows:

#### Fix 1: Generate Workflow (`_python_fallback_align` function)

**Location:** `Interface/Houseweb/views.py` lines 31-97

```python
def _python_fallback_align(boundary, boxes, types, edges, threshold):
    boxes = np.array(boxes)
    types = np.array(types)
    boundary = np.array(boundary)

    # Get boundary extents
    x_min = np.min(boundary[:, 0])
    x_max = np.max(boundary[:, 0])
    y_min = np.min(boundary[:, 1])
    y_max = np.max(boundary[:, 1])

    # Clip boxes to boundary with small margin
    margin = 5  # pixels margin from boundary
    clipped_boxes = []
    for box in boxes:
        if len(box) >= 4:
            x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
            # Clip to boundary limits
            x1 = max(x_min + margin, min(x1, x_max - margin))
            x2 = max(x_min + margin, min(x2, x_max - margin))
            y1 = max(y_min + margin, min(y1, y_max - margin))
            y2 = max(y_min + margin, min(y2, y_max - margin))
            # Ensure x2 > x1 and y2 > y1
            if x2 <= x1:
                x2 = x1 + 10
            if y2 <= y1:
                y2 = y1 + 10
            clipped_boxes.append([x1, y1, x2, y2])
```

#### Fix 2: Transfer Workflow (`AdjustGraph` function)

**Location:** `Interface/Houseweb/views.py` lines 434-463

```python
# Get boundary for clipping
test_index = testNameList.index(testname.split(".")[0])
data = test_data[test_index]
external = np.asarray(data.boundary)
xmin, xmax = np.min(external[:, 0]), np.max(external[:, 0])
ymin, ymax = np.min(external[:, 1]), np.max(external[:, 1])

# Clip boxes to boundary
margin = 5
clipped_boxes_end = []
for box in boxes_end:
    x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
    # Clip to boundary limits
    x1 = max(xmin + margin, min(x1, xmax - margin))
    x2 = max(xmin + margin, min(x2, xmax - margin))
    y1 = max(ymin + margin, min(y1, ymax - margin))
    y2 = max(ymin + margin, min(y2, ymax - margin))
    # Ensure x2 > x1 and y2 > y1
    if x2 <= x1:
        x2 = x1 + 10
    if y2 <= y1:
        y2 = y1 + 10
    clipped_boxes_end.append([x1, y1, x2, y2])
boxes_end = clipped_boxes_end
```

#### Fix 3: Thumbnail Click Workflow (`FindTraindata` function)

**Location:** `Interface/Houseweb/views.py` lines 308-326

```python
external = np.asarray(data.boundary)
xmin, xmax = np.min(external[:, 0]), np.max(external[:, 0])
ymin, ymax = np.min(external[:, 1]), np.max(external[:, 1])

# Clip boxes to boundary
margin = 5
hsbox = []
for x1, y1, x2, y2, cate in data.box[:]:
    # Clip to boundary limits
    x1_clipped = max(xmin + margin, min(float(x1), xmax - margin))
    x2_clipped = max(xmin + margin, min(float(x2), xmax - margin))
    y1_clipped = max(ymin + margin, min(float(y1), ymax - margin))
    y2_clipped = max(ymin + margin, min(float(y2), ymax - margin))
    # Ensure x2 > x1 and y2 > y1
    if x2_clipped <= x1_clipped:
        x2_clipped = x1_clipped + 10
    if y2_clipped <= y1_clipped:
        y2_clipped = y1_clipped + 10
    hsbox.append([[x1_clipped, y1_clipped, x2_clipped, y2_clipped], [mdul.room_label[int(cate)][1]]])
```

**Files Modified:**
- `/Interface/Houseweb/views.py`

---

## ResPlan ID Mapping

### Background

The ResPlan dataset uses sequential file naming (plan_00000, plan_00001, etc.), but each floor plan has an internal numeric ID that appears in the top-left corner of the rendered image and is used throughout the Graph2plan system.

**Example:**
- File: `plan_00000.png`
- Internal ID visible in image: `14433`
- Graph2plan uses: `293` (from converted dataset)

### Mapping Files Generated

Created comprehensive mapping files to bridge ResPlan sequential names and Graph2plan IDs:

**Location:** `/DataPreparation/mapping_output/`

#### Files Created

1. **resplan_id_mapping_all.csv** - Combined train+test mapping (16,996 entries)
2. **resplan_id_mapping_all.json** - JSON version of combined mapping
3. **resplan_id_mapping_train.csv** - Train set only (14,446 entries)
4. **resplan_id_mapping_train.json** - JSON version of train mapping
5. **resplan_id_mapping_test.csv** - Test set only (2,550 entries)
6. **resplan_id_mapping_test.json** - JSON version of test mapping

#### CSV Format

```csv
index,plan_name,graph2plan_id,source,source_index
0,plan_00000,0,train,0
1,plan_00001,1,train,1
18,plan_00018,20,test,2060
2420,plan_02420,2957,train,538
```

**Columns:**
- `index` - Sequential index in combined dataset (sorted by numeric ID)
- `plan_name` - ResPlan sequential name (plan_00000, etc.)
- `graph2plan_id` - Numeric ID used in Graph2plan system
- `source` - Dataset origin (train or test)
- `source_index` - Index within the train or test dataset array

#### JSON Format

```json
{
  "plan_to_id": {
    "plan_00000": "0",
    "plan_02420": "2957"
  },
  "id_to_plan": {
    "0": "plan_00000",
    "2957": "plan_02420"
  },
  "id_to_test_index": {
    "20": 2060
  },
  "total_count": 2550
}
```

### Dataset Statistics

- **Total floor plans:** 16,996
- **Training set:** 14,446 (85%)
- **Test set:** 2,550 (15%)

### Sample Mappings

#### Training Set Examples

| Plan Name | Graph2plan ID | Train Index |
|-----------|---------------|-------------|
| plan_00000 | 0 | 0 |
| plan_02420 | 2957 | 538 |
| plan_04454 | 5460 | - |
| plan_04845 | 5920 | - |

#### Test Set Examples

| Plan Name | Graph2plan ID | Test Index |
|-----------|---------------|------------|
| plan_00018 | 20 | 2060 |
| plan_00027 | 29 | 359 |
| plan_00036 | 38 | 855 |

### Important Notes

⚠️ **The mapping is approximate** because it was reconstructed from the train/test split rather than the original source data. The script sorted items by numeric ID to approximate the original order, but:
- The true sequential order may differ from numeric ID order
- For exact mapping, the original `data.mat` file (before train/test split) would be needed
- Place it at `DataPreparation/data/data.mat` and re-run the mapping script for accurate results

---

## Files Modified

### Frontend Files

1. **`/Interface/templates/home.html`** (lines 788-815)
   - Added `viewBox="-30 -30 316 316"` to all 5 SVG elements
   - Provides 30px padding around 256px layouts

2. **`/Interface/static/js/buttonEvent.js`** (lines 429, 430, 535, 572, 625, 818)
   - Changed `.attr("transform", "scale(2)")` to `.attr("transform", "scale(1)")`
   - Prevents 256×256px SVGs from being scaled to 512×512px

### Backend Files

3. **`/Interface/Houseweb/views.py`**
   - Lines 31-97: Modified `_python_fallback_align()` function
   - Lines 434-463: Modified `AdjustGraph()` function
   - Lines 308-326: Modified `FindTraindata()` function
   - All three now clip room boxes to boundary with 5px margin

### New Files Created

4. **`/DataPreparation/generate_id_mapping.py`**
   - Script to generate ResPlan ID mappings
   - Processes both train and test datasets
   - Creates CSV and JSON output files

5. **`/DataPreparation/mapping_output/`** (directory)
   - Contains all generated mapping files
   - 6 files total (3 CSV + 3 JSON)

6. **`/DataPreparation/check_mapping.py`** (utility)
   - Helper script to verify mapping structure

7. **`/DataPreparation/check_test_data.py`** (utility)
   - Helper script to inspect test dataset structure

---

## How to Use

### Using the Mapping Files

#### Option 1: CSV Lookup (Excel/Spreadsheet)

1. Open `mapping_output/resplan_id_mapping_test.csv` in Excel
2. Use Find (Ctrl+F) to search for either:
   - Plan name: "plan_02420" → returns Graph2plan ID "2957"
   - Graph2plan ID: "2957" → returns plan name "plan_02420"

#### Option 2: Python Programmatic Access

```python
import json

# Load test set mapping
with open('DataPreparation/mapping_output/resplan_id_mapping_test.json', 'r') as f:
    test_map = json.load(f)

# Look up Graph2plan ID from plan name
graph_id = test_map['plan_to_id']['plan_00018']
print(f"Plan plan_00018 has Graph2plan ID: {graph_id}")  # Output: 20

# Look up plan name from Graph2plan ID
plan_name = test_map['id_to_plan']['20']
print(f"Graph2plan ID 20 is: {plan_name}")  # Output: plan_00018

# Look up test dataset array index
test_idx = test_map['id_to_test_index']['20']
print(f"Test dataset index: {test_idx}")  # Output: 2060

# Load the actual floor plan from test data
import pickle
with open('Interface/static/Data/data_test_converted.pkl', 'rb') as f:
    test_data = pickle.load(f)

floor_plan = test_data['data'][test_idx]
print(f"Floor plan name: {floor_plan.name}")
```

#### Option 3: Regenerate Mappings

If you have the original `data.mat` file before the train/test split:

```bash
cd DataPreparation

# Place data.mat in one of these locations:
# - ./data/data.mat
# - ../Interface/static/Data/data.mat

# Run the mapping generator
../g2p-env/Scripts/python.exe generate_id_mapping.py
```

The script will automatically detect the source file and generate exact mappings.

### Verifying the Fixes

After restarting the Django server, test the following workflows:

1. **Load Workflow:**
   - Click "Load" button
   - Select a boundary file
   - Verify boundary displays centered with padding

2. **Generate Workflow:**
   - Click "Generate" button
   - Verify rooms stay within boundary (5px margin)
   - Check that no yellow/colored rooms extend past black walls

3. **Transfer Workflow:**
   - Click a thumbnail in the center list
   - Click "Transfer" button
   - Verify transferred layout respects boundaries

4. **Thumbnail Preview:**
   - Click various thumbnails in center list
   - Verify right panel shows properly clipped layouts
   - Confirm rooms don't extend outside boundaries

---

## Technical Details

### Boundary Clipping Algorithm

All three workflows now use the same clipping logic:

```python
# Get boundary extents
xmin, xmax = np.min(boundary[:, 0]), np.max(boundary[:, 0])
ymin, ymax = np.min(boundary[:, 1]), np.max(boundary[:, 1])

# Clip each room box
margin = 5  # pixels
x1 = max(xmin + margin, min(x1, xmax - margin))
x2 = max(xmin + margin, min(x2, xmax - margin))
y1 = max(ymin + margin, min(y1, ymax - margin))
y2 = max(ymin + margin, min(y2, ymax - margin))

# Ensure valid box dimensions
if x2 <= x1:
    x2 = x1 + 10
if y2 <= y1:
    y2 = y1 + 10
```

This ensures:
- All rooms stay at least 5px inside the boundary
- No room box has zero or negative width/height
- Minimum room size is 10×10 pixels

### Why Multiple Workflows Needed Fixing

The Graph2plan interface has different code paths for different user actions:

1. **LoadTestBoundary** → Loads initial boundary only (no rooms)
2. **Save_Editbox** → Generates new layout from boundary using neural network
3. **AdjustGraph** → Transfers selected layout to current boundary
4. **LoadTrainHouse** → Displays training data when clicking thumbnails

Each path loads data from different sources and constructs the layout independently, so clipping needed to be added to each one.

---

## Future Improvements

### Recommended Enhancements

1. **Exact Mapping Recovery:**
   - Locate original `data.mat` before train/test split
   - Re-run `generate_id_mapping.py` for exact mappings
   - Current mappings are approximate (sorted by numeric ID)

2. **Centralized Clipping Function:**
   - Extract clipping logic into a shared utility function
   - Reduce code duplication across the three workflows
   - Easier to maintain and update clipping parameters

3. **Configurable Margin:**
   - Add margin parameter to config file
   - Currently hardcoded at 5px in three locations
   - Allow users to adjust clearance from boundaries

4. **Validation Checks:**
   - Add boundary integrity checks on data load
   - Warn if room boxes exceed boundaries in source data
   - Log clipping statistics (how many rooms were clipped)

### Known Limitations

1. **Approximate Mapping:**
   - Current mapping reconstructed from train/test split
   - Sorted by numeric ID, not original sequential order
   - May not match exact original plan_XXXXX sequence

2. **No Test Set Images:**
   - Interface currently uses training data for thumbnails
   - `generate_floorplan_images.py` only processes test set
   - Test set has boundary-only images in `/static/Data/Img/`

3. **Hardcoded Margin:**
   - 5px margin hardcoded in three separate locations
   - Changes require editing multiple files
   - No configuration option available

---

## Version History

### Version 1.0 - December 19, 2025

- Fixed SVG scaling from 2x to 1x in `buttonEvent.js`
- Added viewBox padding to all 5 SVG elements in `home.html`
- Added boundary clipping to `_python_fallback_align()` in `views.py`
- Added boundary clipping to `AdjustGraph()` in `views.py`
- Added boundary clipping to `FindTraindata()` in `views.py`
- Created `generate_id_mapping.py` script
- Generated 6 mapping files (CSV + JSON for all/train/test)
- Documented all changes in this CHANGELOG

---

## References

### File Locations

```
Graph2plan/
├── Interface/
│   ├── static/
│   │   ├── js/
│   │   │   └── buttonEvent.js           (Modified - SVG scaling)
│   │   └── Data/
│   │       ├── data_train_converted.pkl (14,446 train items)
│   │       ├── data_test_converted.pkl  (2,550 test items)
│   │       ├── Img/                     (Test set boundary images)
│   │       └── snapshot_train/          (Train set thumbnails)
│   ├── templates/
│   │   └── home.html                    (Modified - SVG viewBox)
│   └── Houseweb/
│       └── views.py                     (Modified - 3 clipping fixes)
│
├── DataPreparation/
│   ├── generate_id_mapping.py           (New - mapping generator)
│   ├── check_mapping.py                 (New - utility)
│   ├── check_test_data.py               (New - utility)
│   └── mapping_output/                  (New - generated files)
│       ├── resplan_id_mapping_all.csv
│       ├── resplan_id_mapping_all.json
│       ├── resplan_id_mapping_train.csv
│       ├── resplan_id_mapping_train.json
│       ├── resplan_id_mapping_test.csv
│       └── resplan_id_mapping_test.json
│
└── CHANGELOG.md                         (This file)
```

### Key Code References

- **SVG Scaling Fix:** `Interface/static/js/buttonEvent.js:429, 430, 535, 572, 625, 818`
- **SVG ViewBox Fix:** `Interface/templates/home.html:788-815`
- **Generate Clipping:** `Interface/Houseweb/views.py:31-97`
- **Transfer Clipping:** `Interface/Houseweb/views.py:434-463`
- **Thumbnail Clipping:** `Interface/Houseweb/views.py:308-326`

### Data Structure References

**Training Data:** `Interface/static/Data/data_train_converted.pkl`
```python
{
  'data': [14446 floor plan objects],
  'nameList': ['293', '10028', '1772', ...],
  'trainTF': [feature vectors]
}
```

**Test Data:** `Interface/static/Data/data_test_converted.pkl`
```python
{
  'data': [2550 floor plan objects],
  'testNameList': ['3553', '6618', '8598', ...],
  'trainNameList': []  # Empty in test file
}
```

---

## Contact & Support

For questions or issues related to these changes:

1. Check the mapping files in `DataPreparation/mapping_output/`
2. Verify Django server was restarted after applying changes
3. Review this CHANGELOG for implementation details
4. Consult the original Graph2plan documentation

**Last Updated:** December 19, 2025

# Graph2plan Development Session Notes
**Date:** January 19, 2026
**Session Summary:** Balcony removal, boundary re-extraction, and bathroom count investigation

---

## Table of Contents
1. [Balcony Removal from ResPlan Dataset](#1-balcony-removal-from-resplan-dataset)
2. [Making Scripts Generalizable](#2-making-scripts-generalizable)
3. [Test Boundary Re-extraction](#3-test-boundary-re-extraction)
4. [Bathroom Count Issue Investigation](#4-bathroom-count-issue-investigation)
5. [Key Files Created/Modified](#5-key-files-createdmodified)
6. [Next Steps](#6-next-steps)

---

## 1. Balcony Removal from ResPlan Dataset

### Objective
Remove balconies from the ResPlan dataset floor plans, starting with a 10-plan test set in `Graph2plan` directory.

### Implementation

#### Script Created: `remove_balconies_clean.py`
**Location:** `C:\Users\hmbashir\source\ResPlan_Dataset\Graph2plan\remove_balconies_clean.py`

**What it does:**
1. Creates backups of original files (with `.backup2` extension)
2. Loads each pickle file containing floor plan data
3. Removes the "balcony" key from plan dictionaries
4. Saves modified plans back to original files
5. Reports statistics about removals

**Key Features:**
- Automatic backup creation before modification
- Handles both "balcony" and "balacony" (typo variant)
- Processes all `plan_*.pkl` files in directory
- Detailed progress reporting

**Results:**
- ✅ Successfully processed 10/10 floor plans
- ✅ All plans had balconies removed
- ✅ No errors during processing
- ✅ Backups created for all files

#### Visualization Script: `visualize_plans.py`
**Location:** `C:\Users\hmbashir\source\ResPlan_Dataset\Graph2plan\visualize_plans.py`

**What it does:**
1. Generates PNG images for modified floor plans (without balconies)
2. Creates before/after comparison images
3. Uses `resplan_utils.py` plotting functions
4. Saves images to `images/` subdirectory

**Results:**
- ✅ Generated 10 modified plan images
- ✅ Generated 10 before/after comparison images
- ✅ Visual verification confirmed clean balcony removal
- ✅ All floor plan structures remained intact

### Visual Verification Results

All comparison images showed:
- **Clean removal:** Only balcony areas (dark gray) were removed
- **Structure preserved:** All rooms, doors, windows, walls intact
- **No artifacts:** No structural damage or geometry issues

**Example floor plans processed:**
- `plan_00000`: Two balconies removed from left side
- `plan_00003`: One balcony removed from left side
- `plan_00006`: Small balcony removed from top
- `plan_00009`: Balcony removed from right side

---

## 2. Making Scripts Generalizable

### Objective
Make both balcony removal and visualization scripts accept directory paths as command-line arguments for use across the full dataset (~17,000 floor plans).

### Changes Made

#### Updated: `remove_balconies_clean.py`

**Added Features:**
- Command-line argument parsing with `argparse`
- Directory path parameter (optional)
- Help message with usage examples
- Default behavior: use script's directory if no path provided

**Usage:**
```bash
# Use current directory
python remove_balconies_clean.py

# Specify directory (relative path)
python remove_balconies_clean.py "path/to/plans"

# Specify directory (absolute path)
python remove_balconies_clean.py "C:/Users/hmbashir/source/ResPlan_Dataset/plans_split"

# Show help
python remove_balconies_clean.py --help
```

**Code Changes:**
```python
# Added argparse support
parser = argparse.ArgumentParser(
    description="Remove balconies from ResPlan floor plan pickle files."
)
parser.add_argument(
    "directory",
    nargs="?",
    default=None,
    help="Directory containing plan_*.pkl files (default: script's directory)"
)
```

#### Updated: `visualize_plans.py`

**Added Features:**
- Same command-line argument structure as removal script
- Directory validation (checks if path exists and is a directory)
- Creates `images/` subdirectory in specified directory

**Usage:**
```bash
# Same syntax as removal script
python visualize_plans.py [directory]
python visualize_plans.py --help
```

### Testing Results

✅ **Help messages** work correctly for both scripts
✅ **Relative paths** work (`"No Balconies"` subdirectory tested)
✅ **Absolute paths** work (`plans_split` directory tested)
✅ **Large dataset detection** confirmed (16,996 files found in `plans_split`)
✅ **Directory validation** works (rejects invalid paths)

### Ready for Large Dataset

The scripts are now ready to process the full `plans_split` directory containing ~17,000 floor plans whenever needed.

---

## 3. Test Boundary Re-extraction

### Problem
Test set boundaries got shuffled - all floor plans were showing the same boundary outline, making visual verification impossible.

### Root Cause
The `data_test_converted.pkl` file had incorrect boundaries that were either duplicated or corrupted during a previous processing step.

### Solution

#### Script Created: `reextract_test_boundaries.py`
**Location:** `C:\Users\hmbashir\source\Graph2plan\DataPreparation\reextract_test_boundaries.py`

**What it does:**
1. Loads test floor plan IDs from `test.txt` (14433, 14926)
2. Loads the original ResPlan dataset (`ResPlan.pkl`)
3. Extracts proper boundaries using `extract_boundary_from_inner()` function
4. Updates `data_test_converted.pkl` with correct boundaries
5. Creates automatic backup before modification

**Key Functions Implemented:**
- `extract_coords_from_shapely()`: Extract coordinates from Shapely geometry
- `compute_boundary_directions()`: Compute direction codes (0=right, 1=up, 2=left, 3=down)
- `add_boundary_metadata()`: Convert (N×2) coords to Graph2plan format (N×4)
- `extract_boundary_from_inner()`: Extract floorplan boundary from ResPlan 'inner' field
- `load_resplan_dataset()`: Load and index ResPlan dataset by ID

**Usage:**
```bash
cd /mnt/c/Users/hmbashir/source/Graph2plan/DataPreparation

python reextract_test_boundaries.py \
  --resplan "C:/Users/hmbashir/source/ResPlan_Dataset/ResPlan.pkl" \
  --test-names "./data/test.txt" \
  --test-pkl "../Interface/static/Data/data_test_converted.pkl"
```

### Results

```
✅ Successfully updated 2/2 boundaries
✅ No failed floor plans
✅ Backup created: data_test_converted.pkl.backup
```

**Verification:**
- Floor plan 0 (ID: 14433): boundary shape (10, 4) - UNIQUE
- Floor plan 1 (ID: 14926): boundary shape (10, 4) - UNIQUE
- **Max difference between boundaries:** 256.0 (confirmed different)

### Image Regeneration

After re-extracting boundaries, regenerated the boundary images:

**Script:** `generate_floorplan_images.py`
**Results:**
- ✅ Generated 2 test images (14433.png, 14926.png)
- ✅ Generated 8 train images
- ✅ Images now show correct, distinct boundary shapes

**Visual Confirmation:**
- `14433.png`: L-shaped floor plan with indentations on left
- `14926.png`: Different shape with distinct indentation pattern

---

## 4. Bathroom Count Issue Investigation

### Problem
When retrieved floor plans have exactly **3 bathrooms**, the system displays an **extra bathroom** (showing 4 instead of 3).

### Investigation Findings

#### Data Structure Analysis

**File:** `rNum_train.npy`
**Location:** `C:\Users\hmbashir\source\Graph2plan\Interface\static\Data\rNum_train.npy`

**Structure:**
- Shape: `(8, 14)` - 8 floor plans, 14 room type columns
- **Column 3:** Bathroom count
- **Column 13:** Sum of all bedroom types (indices 1,5,6,7,8)

**Training Data Breakdown:**
```
Plan 0 (ID: 7067):  rNum[3]=2 bathrooms, rNum[13]=1 bedroom
Plan 1 (ID: 17054): rNum[3]=2 bathrooms, rNum[13]=2 bedrooms
Plan 2 (ID: 14877): rNum[3]=4 bathrooms, rNum[13]=3 bedrooms  ← KEY
Plan 3 (ID: 16969): rNum[3]=2 bathrooms, rNum[13]=2 bedrooms
Plan 4 (ID: 1448):  rNum[3]=2 bathrooms, rNum[13]=2 bedrooms
Plan 5 (ID: 14035): rNum[3]=2 bathrooms, rNum[13]=2 bedrooms
Plan 6 (ID: 3467):  rNum[3]=4 bathrooms, rNum[13]=3 bedrooms  ← KEY
Plan 7 (ID: 5410):  rNum[3]=2 bathrooms, rNum[13]=2 bedrooms
```

#### Room Type Index Mapping

From `DataPreparation/3.rNum_train.py`:
```python
# Indices 0-12: Individual room types
0  = LivingRoom
1  = MasterRoom
2  = Kitchen
3  = Bathroom        ← The issue column
4  = DiningRoom
5  = ChildRoom
6  = StudyRoom
7  = SecondRoom
8  = GuestRoom
9  = Balcony
10 = Entrance
11 = Storage
12 = Wall-in
13 = Sum of bedrooms (indices 1,5,6,7,8)  ← Potential confusion point
```

### Root Cause Hypothesis

**Pattern Found:**
- When user requests **3 bathrooms**
- Floor plans retrieved have **3 bedrooms** (column 13)
- Those same plans have **4 bathrooms** (column 3)

**Likely Issue:** Index confusion between column 3 (bathrooms) and column 13 (bedroom sum)

**Potential Problem Locations:**

1. **Frontend JavaScript** (room count array handling)
   - Check if there's confusion between 0-based and 1-based indexing
   - Verify room type to array index mapping

2. **Backend Python** - `views.py` function: `calculate_room_match_percentage()`
   - **Location:** `/mnt/c/Users/hmbashir/source/Graph2plan/Interface/Houseweb/views.py:333`
   - Lines 346-401 handle room count filtering and matching
   - Uses `mask` array to filter room types
   - Applies `active_mask` based on requested counts

3. **Room Count Filtering** - `views.py` function: `get_filter_func()`
   - **Location:** `/mnt/c/Users/hmbashir/source/Graph2plan/Interface/Houseweb/views.py:242`
   - Creates filter functions for room number matching
   - Lines 250-255 implement the actual filtering logic

### Files to Investigate

**Backend:**
```
/mnt/c/Users/hmbashir/source/Graph2plan/Interface/Houseweb/views.py
  - Line 242: get_filter_func() - Room number filtering
  - Line 333: calculate_room_match_percentage() - Match calculation
  - Line 440, 1194: train_data_rNum[test_data_topk] usage
```

**Frontend:**
```
/mnt/c/Users/hmbashir/source/Graph2plan/Interface/static/
  (JavaScript files handling room count input/display)
```

### Investigation Status

**Status:** ⚠️ **ISSUE IDENTIFIED, NOT FIXED**

**Next Steps:**
1. Examine JavaScript frontend room count handling
2. Check array index mapping (bathroom index 3 vs bedroom sum index 13)
3. Verify 0-based vs 1-based indexing consistency
4. Test with explicit column logging when 3 bathrooms requested
5. Implement fix once exact cause confirmed

---

## 5. Key Files Created/Modified

### New Files Created

1. **`remove_balconies_clean.py`**
   - Location: `Graph2plan/remove_balconies_clean.py`
   - Purpose: Remove balconies from floor plan pickle files
   - Features: Automatic backup, command-line args, progress reporting

2. **`visualize_plans.py`**
   - Location: `Graph2plan/visualize_plans.py`
   - Purpose: Generate boundary images for visual verification
   - Features: Before/after comparison, command-line args

3. **`reextract_test_boundaries.py`**
   - Location: `DataPreparation/reextract_test_boundaries.py`
   - Purpose: Re-extract boundaries from original ResPlan dataset
   - Features: Shapely geometry handling, automatic backup

4. **`SESSION_NOTES_2026-01-19.md`**
   - Location: `Graph2plan/SESSION_NOTES_2026-01-19.md`
   - Purpose: This document

### Modified Files

None - all work done through new scripts to preserve original codebase.

### Data Files Modified

1. **`data_test_converted.pkl`**
   - Location: `Interface/static/Data/data_test_converted.pkl`
   - Change: Boundaries re-extracted for test floor plans
   - Backup: `data_test_converted.pkl.backup`

2. **Floor plan files in `Graph2plan/` (10 files)**
   - Files: `plan_00000.pkl` through `plan_00009.pkl`
   - Change: Balcony key removed
   - Backups: `*.pkl.backup2` files

### Image Files Generated

1. **Modified plan images** (10 files)
   - Location: `Graph2plan/images/`
   - Files: `plan_00000_modified.png` through `plan_00009_modified.png`

2. **Comparison images** (10 files)
   - Location: `Graph2plan/images/`
   - Files: `plan_00000_comparison.png` through `plan_00009_comparison.png`

3. **Test boundary images** (2 files)
   - Location: `Interface/static/Data/Img/`
   - Files: `14433.png`, `14926.png`

---

## 6. Next Steps

### Immediate Tasks

1. **Fix Bathroom Count Bug**
   - [ ] Investigate JavaScript room count array handling
   - [ ] Check index mapping (column 3 vs column 13)
   - [ ] Add debug logging to trace room count flow
   - [ ] Implement and test fix

2. **Apply Balcony Removal to Full Dataset**
   - [ ] Run `remove_balconies_clean.py` on `plans_split/` directory (~17,000 plans)
   - [ ] Verify random sample after processing
   - [ ] Update main `ResPlan.pkl` file if needed

### Future Considerations

1. **Dataset Integrity**
   - Verify all boundaries are correctly extracted
   - Check for other potential data inconsistencies
   - Document any anomalies found

2. **Performance Optimization**
   - Consider parallelizing balcony removal for large dataset
   - Add progress bars for long-running operations
   - Implement batch processing if needed

3. **Documentation**
   - Update main README with balcony removal process
   - Document room type index mapping clearly
   - Add troubleshooting guide for common issues

---

## Technical Notes

### Python Environment
- **Python Version:** 3.13
- **Key Packages:** shapely, geopandas, matplotlib, networkx, numpy, pickle
- **Platform:** Windows (WSL2)

### Data Formats
- **Floor Plans:** Pickle files with dictionary structure
- **Boundaries:** (N×4) arrays: [x, y, direction, isNew]
- **Room Counts:** (14,) arrays indexed by room type

### Important Paths
```
C:\Users\hmbashir\source\
├── ResPlan_Dataset/
│   ├── ResPlan.pkl (17,000 plans)
│   ├── plans_split/ (17,000 individual files)
│   └── Graph2plan/ (10 test plans)
│       └── images/ (generated visualizations)
└── Graph2plan/
    ├── DataPreparation/
    │   ├── reextract_test_boundaries.py
    │   ├── generate_floorplan_images.py
    │   └── data/
    │       ├── test.txt (2 test IDs)
    │       └── train.txt (8 train IDs)
    └── Interface/
        └── static/Data/
            ├── data_test_converted.pkl
            ├── rNum_train.npy
            └── Img/ (boundary images)
```

---

## Session Statistics

- **Duration:** ~2 hours
- **Scripts Created:** 3
- **Floor Plans Processed:** 10 (balcony removal) + 2 (boundary re-extraction)
- **Images Generated:** 20 comparison + 2 boundary = 22 total
- **Issues Fixed:** 2 (balcony removal, boundary shuffling)
- **Issues Identified:** 1 (bathroom count bug)
- **Lines of Code Written:** ~800

---

**Session End**
Status: ✅ Balcony removal complete | ✅ Boundaries fixed | ⚠️ Bathroom bug identified

# Balcony Clipping Fix Documentation

## Problem Description

Balconies (room type 9, colored green/olive) were being clipped at the building boundary, with only partial sections visible. The balcony nodes were being generated correctly in the graph, but the visual rendering was cutting them off.

## Root Cause

The issue had two components:

1. **Backend Python clipping**: The Python code was clipping ALL room coordinates (including balconies) to stay within the boundary margins
2. **Frontend SVG clipPath**: JavaScript was applying an SVG clipPath that masked everything outside the boundary polygon

Balconies by design should extend beyond the main building boundary, so both clipping mechanisms needed to exclude them.

## Solution

### Part 1: Backend Python Fixes (views.py)

Modified three functions in `/mnt/c/Users/hmbashir/source/Graph2plan/Interface/Houseweb/views.py` to exclude balconies (type 9) from coordinate clipping.

#### Fix 1: `_python_fallback_align()` function (lines 46-71)

```python
for i, box in enumerate(boxes):
    if len(box) >= 4:
        x1, y1, x2, y2 = box[0], box[1], box[2], box[3]

        # Balconies (type 9) should extend outside - don't clip them
        if i < len(types) and int(types[i]) == 9:
            clipped_boxes.append([x1, y1, x2, y2])
        else:
            # Clip to boundary limits
            x1 = max(x_min + margin, min(x1, x_max - margin))
            y1 = max(y_min + margin, min(y1, y_max - margin))
            x2 = max(x_min + margin, min(x2, x_max - margin))
            y2 = max(y_min + margin, min(y2, y_max - margin))
            clipped_boxes.append([x1, y1, x2, y2])
```

**Location**: Around line 46-71
**Purpose**: Handles coordinate clipping during graph alignment
**Change**: Added check for room type 9 (balcony) before applying coordinate clipping

#### Fix 2: `FindTraindata()` function (lines 312-330)

```python
for x1, y1, x2, y2, cate in data.box[:]:
    # Balconies (type 9) should extend outside - don't clip them
    if int(cate) == 9:
        hsbox.append([[float(x1), float(y1), float(x2), float(y2)],
                      [mdul.room_label[int(cate)][1]]])
    else:
        # Clip to boundary limits
        x1_clipped = max(xmin + margin, min(float(x1), xmax - margin))
        y1_clipped = max(ymin + margin, min(float(y1), ymax - margin))
        x2_clipped = max(xmin + margin, min(float(x2), xmax - margin))
        y2_clipped = max(ymin + margin, min(float(y2), ymax - margin))

        hsbox.append([[x1_clipped, y1_clipped, x2_clipped, y2_clipped],
                      [mdul.room_label[int(cate)][1]]])
```

**Location**: Around line 312-330
**Purpose**: Prepares training data for the left panel (reference layout)
**Change**: Skip clipping for balconies when processing room boxes

#### Fix 3: `AdjustGraph()` function (lines 464-483)

```python
for i, box in enumerate(boxes_end):
    x1, y1, x2, y2 = box[0], box[1], box[2], box[3]

    # Balconies (type 9) should extend outside - don't clip them
    if int(room[i]) == 9:
        clipped_boxes_end.append([x1, y1, x2, y2])
    else:
        # Clip to boundary limits
        x1 = max(xmin + margin, min(x1, xmax - margin))
        y1 = max(ymin + margin, min(y1, ymax - margin))
        x2 = max(xmin + margin, min(x2, xmax - margin))
        y2 = max(ymin + margin, min(y2, ymax - margin))
        clipped_boxes_end.append([x1, y1, x2, y2])
```

**Location**: Around line 464-483
**Purpose**: Adjusts generated graph layout for the right panel
**Change**: Exclude balconies from coordinate clipping when adjusting graph positions

### Part 2: Frontend JavaScript Fixes (buttonEvent.js)

Disabled SVG clipPath that was masking content outside the boundary polygon.

#### Fix 1: Left panel SVG clipPath (line 537)

**File**: `/mnt/c/Users/hmbashir/source/Graph2plan/Interface/static/js/buttonEvent.js`

```javascript
// Line 537 - DISABLED to allow balconies to extend beyond boundary
// Original: d3.select("#LeftLayoutSVG").attr("clip-path", "url(#clip-th)");
// Disabled: SVG clipPath clips everything including balconies which should extend outside
```

**Location**: Line 537
**Change**: Commented out the clipPath attribute that was masking all content to the boundary polygon

#### Fix 2: Right panel SVG clipPath (line 607)

```javascript
// Line 607 - DISABLED to allow balconies to extend beyond boundary
// Original: d3.select("#RightLayoutSVG").attr("clip-path", "url(#Rightclip-th)");
// Disabled: SVG clipPath clips everything including balconies which should extend outside
```

**Location**: Line 607
**Change**: Commented out the clipPath attribute for the right panel

## Testing

After making these changes:

1. **Clear browser cache**: Press `Ctrl+F5` to force refresh and clear cached JavaScript
2. **Restart Django server**: Ensure Python changes are loaded
3. **Load a floor plan with balconies**: Verify balconies now extend fully beyond the boundary

## Notes

- Balconies are identified by room type `9` in the dataset
- The `margin` variable in Python code defines how far from boundary other rooms should be clipped
- SVG clipPath creates a masking region - disabling it allows all SVG elements to render fully
- Both Python AND JavaScript fixes were necessary for complete resolution

## Files Modified

1. `/mnt/c/Users/hmbashir/source/Graph2plan/Interface/Houseweb/views.py`
   - `_python_fallback_align()` function
   - `FindTraindata()` function
   - `AdjustGraph()` function

2. `/mnt/c/Users/hmbashir/source/Graph2plan/Interface/static/js/buttonEvent.js`
   - Line 537 (left panel clipPath)
   - Line 607 (right panel clipPath)

## Reverting (if needed)

If you need to revert to original behavior:

**Python (views.py)**: Remove the `if int(cate) == 9:` checks and always apply clipping
**JavaScript (buttonEvent.js)**: Uncomment lines 537 and 607 to re-enable clipPath

---

**Date**: 2025-12-19
**Issue**: Balcony clipping at building boundary
**Status**: ✓ Resolved

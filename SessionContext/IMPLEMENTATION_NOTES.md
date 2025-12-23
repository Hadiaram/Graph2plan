# Graph2plan Enhancement Implementation Notes

## Session Date

December 22, 2025

## Overview

This document details the implementation of three major features for the Graph2plan floor plan generation system:

1. **Similarity-based fallback matching** when hard filters return no results
2. **Match percentage display** showing how well floor plans match user requirements
3. **Automatic room adjustment** to add/remove rooms based on user input

---

## Table of Contents

- [Background](#background)
- [Feature 1: Similarity-Based Fallback](#feature-1-similarity-based-fallback)
- [Feature 2: Match Percentage Display](#feature-2-match-percentage-display)
- [Feature 3: Auto-Adjustment](#feature-3-auto-adjustment)
- [Files Modified](#files-modified)
- [Testing Guide](#testing-guide)
- [Debugging Tips](#debugging-tips)
- [Known Issues](#known-issues)

---

## Background

### Original System Behavior

The Graph2plan system originally used **strict hard filtering**:

- Users specify room requirements (e.g., "3 bedrooms, 2 bathrooms")
- System searches ~75,000 floor plans for exact matches
- If no exact matches found → **returns empty results**
- No visibility into how close results are to requirements

### User Requirements

The user wanted:

1. Always show results even when no exact match exists
2. Show similarity scores/match percentages
3. Automatically adjust floor plans to match requirements

---

## Feature 1: Similarity-Based Fallback

### Purpose of Similarity-Based Fallback

When hard filters return no results, automatically fall back to similarity-based matching to always show the closest floor plans.

### Implementation Details

#### Backend: `views.py`

**Location:** Lines 256-318

**New Functions:**

1. **`compute_similarity_scores()`** (lines 256-318)
   - Combines three metrics:
     - **Room count similarity** (35-50% weight): L1 distance between room counts
     - **Edge structure similarity** (0-35% weight): L1 distance between adjacency graphs
     - **Boundary similarity** (30-50% weight): Turn Function (TF) distance
   - Returns normalized scores where **lower = better match**
   - Weights adjust automatically if edge data unavailable

2. **`calculate_room_match_percentage()`** (lines 321-406)
   - Calculates percentage of requested rooms that are present
   - Respects `exact_match` requirements:
     - **Exact match**: Only counts if candidate has exactly requested count
     - **At least (1+)**: Counts if candidate has >= requested count
   - Formula: `(matched_rooms / requested_rooms) × 100`

**Integration Points:**

1. **`NumSearch()`** (lines 409-519)
   - Lines 445-486: Fallback logic when `len(indices[0]) == 0`
   - Computes TF distances for all candidates
   - Calls `compute_similarity_scores()` with room-only weights (50% room, 50% boundary)
   - Returns top 20 results sorted by similarity

2. **`GraphSearch()`** (lines 1028-1249)
   - Lines 1100-1148: Same fallback pattern
   - Uses all three metrics (35% room, 35% edge, 30% boundary)
   - Searches across ~75k floor plans when graph filter fails

**Response Format:**

```json
{
  "floorPlans": ["12345.png", "67890.png", ...],  // Backward compatible
  "metadata": [
    {
      "name": "12345.png",
      "match": 85.5,
      "fallback": true
    }
  ]
}
```

#### Frontend: `buttonEvent.js`

**Location:** Lines 274-278, 720-725

**Changes:**

- Updated `NumSearch()` and `GraphSearch()` callbacks to extract both `floorPlans` and `metadata`
- Pass `metadata` to `ListBox()` for display

---

## Feature 2: Match Percentage Display

### Purpose of Match Percentage Display

Show users how well each floor plan matches their requirements as a percentage.

### Implementation of Match Display

#### Frontend (Match Display): `buttonEvent.js`

**Location:** Lines 179-220

**Modified `ListBox()` function:**

- Accepts new `metadata` parameter
- For each floor plan, displays colored badge with match percentage:
  - **Green badge**: 100% match (exact match from hard filter)
  - **Orange badge**: <100% match (similarity-based fallback)
- Badge positioned as `float: right` in floor plan title

**Visual Example:**

```text
Floor Plan Name    [95%]  ← Orange badge (fallback)
Floor Plan Name    [100%] ← Green badge (exact match)
```

#### CSS Styling

```javascript
matchBadge.style.cssText = 'float: right; background-color: ' +
    (isFallback ? '#FFA500' : '#4CAF50') + // Orange vs Green
    '; color: white; padding: 2px 6px; border-radius: 3px; font-size: 11px;';
```

### Exact Match vs. "At Least" Logic

**Critical Fix (lines 321-406):**

The original implementation incorrectly treated all comparisons as "at least" (>=), giving 100% for floor plans with **more** rooms than requested.

**Example of Bug:**

```text
Requested: [1 LivingRoom, 2 Bedrooms, 1 Bath, 1 Kitchen] = 5 total
Candidate: [1 LivingRoom, 4 Bedrooms, 2 Bath, 3 Kitchen] = 10 total
Old Result: min([1,4,2,3], [1,2,1,1]) = [1,2,1,1] = 5/5 = 100% ❌
```

**Fixed Logic:**

```python
if exact_match_array[i]:
    # Exact match required: only count if candidate == requested
    if candidate_counts[i] == requested_counts[i]:
        matched_counts[i] = requested_counts[i]
    else:
        matched_counts[i] = 0
else:
    # At least match: count min(candidate, requested)
    matched_counts[i] = min(candidate_counts[i], requested_counts[i])
```

**With Fix:**

```text
Requested: [1, 2, 1, 1] with exact_match: [True, True, True, True]
Candidate: [1, 4, 2, 3]
Result: [1, 0, 0, 0] = 1/5 = 20% ✓
```

---

## Feature 3: Auto-Adjustment

### Purpose of Auto-Adjustment

Automatically add missing rooms and remove excess rooms to match user requirements with one click.

### Implementation of Auto-Adjustment

#### HTML: `home.html`

**Location:** Lines 835-838

**Added:**

```html
<div id="AutoAdjust"
     style="cursor: pointer;background-color: #ff9800;color: #fff;
            width: 100px;border-radius: 30px;
            display: none;">
    Auto-Adjust
</div>
```

- Orange button positioned between "Generate" and "Save"
- Initially hidden (`display: none`)
- Shown only after floor plan is transferred to left side

**Layout Change:**

- Moved "Save" button from `margin-left: 180px` to `margin-left: 290px` to make room

#### Backend (Auto-Adjustment): `views.py`

**Location:** Lines 1252-1391

**New Function: `AutoAdjustGraph()`**

**Algorithm:**

1. **Count Current Rooms** (lines 1285-1296)
   - Iterate through graph nodes
   - Count each room type
   - Group bedroom variants (MasterRoom, ChildRoom, StudyRoom, etc.)

2. **Determine Adjustments** (lines 1298-1323)
   - For each active room type:
     - If `current < requested`: Add `(requested - current)` rooms
     - If `current > requested` AND `exact_match`: Remove `(current - requested)` rooms
   - Creates `rooms_to_add` list and `rooms_to_remove` list

3. **Add Rooms** (lines 1332-1341)
   - Calculate average position of existing nodes
   - Place new rooms at `avg_position + offset`
   - Offset spreads rooms out: `(index % 5) * 20 pixels`
   - Each room gets next available index number
   - Default scale = 1

4. **Remove Rooms** (lines 1343-1357)
   - Remove from end of node list first (LIFO)
   - Remove associated edges for deleted nodes
   - Handles bedroom grouping (removes any bedroom type if "Bedroom" requested)

5. **Recalculate Edges** (lines 1359-1380)
   - For each node without edges:
     - Find nearest neighbor by Euclidean distance
     - Create edge to nearest neighbor
   - Ensures graph remains connected

**Room Type Mapping:**

```python
room_idx_to_name = {
    0: 'LivingRoom', 1: 'MasterRoom', 2: 'Kitchen', 3: 'Bathroom',
    4: 'DiningRoom', 5: 'ChildRoom', 6: 'StudyRoom', 7: 'SecondRoom',
    8: 'GuestRoom', 9: 'Balcony', 10: 'Entrance', 11: 'Storage', 12: 'Wall-in'
}
```

**Response:**

```json
{
  "nodes": [
    [index, roomname, x, y, scalesize],
    ...
  ],
  "edges": [
    [u, v],
    ...
  ]
}
```

#### Frontend (Auto-Adjustment): `buttonEvent.js`

**Location:** Lines 756, 788-850

**Integration:**

1. **Show Button** (line 756)

   ```javascript
   document.getElementById("AutoAdjust").style.display = "block";
   ```

   - Triggered when `CreateLeftGraph()` is called (after Transfer)

2. **Click Handler** (lines 788-850)
   - Gets current graph state via `GetEditGraph(0)`
   - Gets room requirements via `Num()`
   - Calls `/index/AutoAdjustGraph/` endpoint
   - Receives adjusted nodes and edges
   - **Redraws graph:**
     - Clears existing circles and lines
     - Draws new edges using `CreateLine()`
     - Draws new nodes using `CreateCircle()`
   - Updates room count cookie
   - Shows alert confirmation

**Workflow:**

```text
User clicks Transfer
  → Floor plan appears on left side
  → Auto-Adjust button appears
  → User clicks Auto-Adjust
  → Backend adds/removes rooms
  → Frontend redraws graph
  → User clicks Generate to create layout
```

---

## Files Modified

### Backend Files

#### 1. `Houseweb/views.py`

- **Line 256-318**: Added `compute_similarity_scores()` function
- **Line 321-406**: Added `calculate_room_match_percentage()` function
- **Line 409-519**: Modified `NumSearch()` with fallback logic
- **Line 1028-1249**: Modified `GraphSearch()` with fallback logic
- **Line 1252-1391**: Added `AutoAdjustGraph()` function

#### 2. `model/floorplan.py`

- **Line 165-179**: Fixed float-to-int conversion for array slicing
- **Line 199-201**: Added `np.clip()` to prevent index out of bounds

#### 3. `model/test.py`

- **Line 118**: Added None check for `vw.boxes_pred`

### Frontend Files

#### 4. `templates/home.html`

- **Line 835-838**: Added Auto-Adjust button
- **Line 840**: Adjusted Save button position

#### 5. `static/js/buttonEvent.js`

- **Line 179-220**: Modified `ListBox()` to display match percentages
- **Line 274-278**: Updated `NumSearch()` to pass metadata
- **Line 720-725**: Updated `GraphSearch()` to pass metadata
- **Line 756**: Show Auto-Adjust button on transfer
- **Line 788-850**: Added Auto-Adjust click handler

---

## Testing Guide

### Test 1: Similarity Fallback

**Goal:** Verify fallback works when no exact matches exist

**Steps:**

1. Load a test boundary file
2. Set unrealistic requirements (e.g., 10 bedrooms, 10 bathrooms)
3. Click "Search"
4. **Expected:** See ~20 results with orange badges showing <100% match
5. **Check console:** Should see "NumSearch: Hard filter returned no results. Using similarity-based fallback."

### Test 2: Match Percentage Display

**Goal:** Verify percentages are accurate

**Steps:**

1. Set requirements: 2 Bedrooms (exact), 1 Bathroom (exact), 1 Kitchen (1+)
2. Click "Search"
3. **Expected:**
   - Results show match percentages
   - Green badges for 100% matches
   - Orange badges for partial matches
4. **Check console:** Look for debug output:

   ```text
   Candidate counts (filtered): [2 1 1]
   Requested counts (filtered): [2 1 1]
   Exact match required (filtered): [True True False]
   Match percentage: 100.0%
   ```

### Test 3: Exact Match Logic

**Goal:** Verify exact match correctly rejects floor plans with extra rooms

**Steps:**

1. Set requirements: 2 Bedrooms (exact match checked)
2. Click "Search"
3. Select a result with 4 bedrooms
4. **Expected:** Match percentage should be low (e.g., 40-60%), NOT 100%
5. **Check console:** Should see matched_counts with 0s for mismatched rooms

### Test 4: Auto-Adjustment

**Goal:** Verify rooms are added/removed correctly

**Steps:**

1. Set requirements: 3 Bedrooms (exact), 2 Bathrooms (exact)
2. Search and select a floor plan with 1 Bedroom, 1 Bathroom
3. Click "Transfer"
4. **Expected:** Auto-Adjust button appears (orange)
5. Click "Auto-Adjust"
6. **Check left graph:**
   - Should now have 3 bedroom nodes + 2 bathroom nodes
   - New nodes should be connected with edges
7. **Check console:**

   ```text
   Current nodes: 2
   Rooms to add: ['MasterRoom', 'MasterRoom', 'Bathroom']
   Adjusted nodes: 5
   ```

8. Click "Generate" to create floor plan layout

---

## Debugging Tips

### Backend Debugging

#### 1. Enable Debug Logging

All key functions have extensive `print()` statements. Check the Django console output:

```bash
# Run Django server
python manage.py runserver

# Watch console for:
NumSearch: Hard filter returned no results. Using similarity-based fallback.
Candidate counts (filtered): [1 4 2 3]
Requested counts (filtered): [1 2 1 1]
Match percentage: 60.0%

=== AutoAdjustGraph called ===
Current nodes: 5
Rooms to add: ['Bathroom', 'Kitchen']
Adjusted nodes: 7
```

#### 2. Common Backend Issues

**Issue:** Match percentage always 100%

- **Debug:** Check if `roomexaarr` is being passed correctly
- **Location:** `views.py:475, 499, 1195, 1225`
- **Fix:** Ensure all calls include 4th parameter: `calculate_room_match_percentage(candidate, requested, mask, exact_match)`

**Issue:** Similarity fallback not triggered

- **Debug:** Check `len(indices[0])` value
- **Location:** `views.py:445, 1100`
- **Verify:** Hard filter should return empty when no exact matches

**Issue:** Auto-adjust adds wrong room types

- **Debug:** Check `room_idx_to_name` mapping
- **Location:** `views.py:1275-1279`
- **Note:** Bedroom types (indices 1,5,6,7,8) are grouped together

#### 3. Database Issues

If retrieval data is corrupted:

```bash
# Check data files exist
ls -la model/dataset/

# Verify training data loaded
# In Django shell:
python manage.py shell
>>> from Houseweb.views import train_data, trainNameList
>>> len(train_data)  # Should be ~74995
>>> trainNameList[0]  # Should show first floor plan name
```

### Frontend Debugging

#### 1. Browser Console

Open browser DevTools (F12) and check console for:

```javascript
// NumSearch/GraphSearch response
{
  floorPlans: ["12345.png", ...],
  metadata: [{name: "12345.png", match: 85.5, fallback: true}, ...]
}

// Auto-Adjust
Auto-Adjust clicked!
Current graph: [[0, "LivingRoom", 128, 128, 1], ...]
Room requirements: [[1,1,1,...], [0,0,0,...], [1,2,1,...]]
Auto-adjust result: {nodes: [...], edges: [...]}
```

#### 2. Common Frontend Issues

**Issue:** Match percentages not displaying

- **Debug:** Check if `metadata` is null
- **Location:** `buttonEvent.js:195`
- **Fix:** Verify backend returns `metadata` field

**Issue:** Auto-Adjust button doesn't appear

- **Debug:** Check CSS display property
- **Verify:** `document.getElementById("AutoAdjust").style.display === "block"`
- **Location:** `buttonEvent.js:756`

**Issue:** Graph doesn't redraw after auto-adjust

- **Debug:** Check if `CreateCircle()` and `CreateLine()` functions exist
- **Verify:** D3.js is loaded
- **Check:** Browser console for JavaScript errors

#### 3. Network Debugging

Check AJAX requests in Network tab:

```text
Request: /index/NumSearch/?userInfo=[...]
Response: {floorPlans: [...], metadata: [...]}

Request: /index/AutoAdjustGraph/?NewGraph=[...]&Numrooms=[...]
Response: {nodes: [...], edges: [...]}
```

If 404 or 500 errors:

- Verify URL routing in Django `urls.py`
- Check Django server console for Python exceptions

### Room Counting Debug

If room counts seem wrong, add this debug code:

```python
# In calculate_room_match_percentage() after line 360:
print(f"Active mask: {active_mask}")
print(f"Filtered candidate: {candidate_counts}")
print(f"Filtered requested: {requested_counts}")
print(f"Exact match array: {exact_match_array}")
```

### Visual Debugging

To verify graph structure:

```javascript
// In browser console after Auto-Adjust:
d3.selectAll('.TransCircle').each(function() {
    console.log(this.id, this.cx.baseVal.value, this.cy.baseVal.value);
});

d3.selectAll('.TransLine').each(function() {
    console.log(this.id);
});
```

---

## Known Issues

### 1. Room Placement in Auto-Adjust

**Issue:** New rooms are placed at average position with simple offset, which may cause overlapping or placement outside boundary.

**Workaround:** User can manually drag nodes after auto-adjustment.

**Future Fix:** Implement smart placement algorithm that:

- Respects boundary constraints
- Avoids overlapping existing rooms
- Considers room adjacency relationships

**Code Location:** `views.py:1336-1341`

### 2. Edge Recalculation

**Issue:** Auto-adjust connects new rooms to nearest neighbor only, which may not reflect logical adjacencies (e.g., bathroom should connect to bedrooms, not kitchen).

**Workaround:** User can manually add/remove edges after auto-adjustment.

**Future Fix:** Implement rule-based edge creation:

- Bedrooms connect to bathrooms and living room
- Kitchen connects to dining room
- Storage connects to entrance

**Code Location:** `views.py:1359-1380`

### 3. Bedroom Grouping

**Issue:** When user requests "Bedroom" (index 1), system may add MasterRoom, ChildRoom, etc. inconsistently.

**Current Behavior:** Adds 'MasterRoom' by default for all bedroom requests.

**Future Fix:** Allow user to specify preferred bedroom types or distribute evenly (1 Master, rest Secondary).

**Code Location:** `views.py:1312, 1351`

### 4. Performance with Large Fallback

**Issue:** When hard filter fails, similarity calculation processes ~75,000 floor plans, which can take 5-10 seconds.

**Workaround:** Use clustering optimization (already implemented in `rt.retrieval()`).

**Future Fix:**

- Precompute similarity scores
- Use approximate nearest neighbor search (FAISS, Annoy)

**Code Location:** `views.py:1104` (GraphSearch processes full dataset)

### 5. Match Percentage Edge Cases

**Issue:** When user selects "Any" for a room type (mask = False), it's excluded from percentage calculation. This can show 100% even when room types differ.

**Example:**

```text
Requested: 2 Bedrooms (active), Kitchen = Any (inactive)
Candidate: 2 Bedrooms, 0 Kitchens
Result: 100% (Kitchen not counted)
```

**Expected Behavior:** This is actually correct per requirements.

**Note:** If this is confusing, could show separate metrics: "Required: 100%, Optional: N/A"

---

## Data Flow Diagrams

### Similarity Fallback Flow

```text
User clicks Search
  ↓
NumSearch/GraphSearch called
  ↓
Apply hard filters
  ↓
indices empty? ─NO→ Return top 20 exact matches ─┐
  │                                                │
  YES                                              │
  ↓                                                │
Compute similarity scores for all candidates       │
  ↓                                                │
Sort by score (lower = better)                     │
  ↓                                                │
Return top 20 similar matches ─────────────────────┤
  ↓                                                │
Calculate match % for each result ←────────────────┘
  ↓
Return {floorPlans: [...], metadata: [...]}
  ↓
Frontend displays with colored badges
```

### Auto-Adjustment Flow

```text
User clicks Transfer
  ↓
Floor plan appears on left (graph nodes + edges)
  ↓
Auto-Adjust button shown
  ↓
User clicks Auto-Adjust ─┐
  ↓                        │
Get current graph         │
Get room requirements     │
  ↓                        │
Send to backend ──────────┘
  ↓
AutoAdjustGraph():
  1. Count current rooms
  2. Compare to requirements
  3. Determine adds/removes
  4. Add new nodes at avg position
  5. Remove excess nodes (if exact match)
  6. Recalculate edges
  ↓
Return {nodes: [...], edges: [...]}
  ↓
Frontend redraws graph
  ↓
User clicks Generate → Create floor plan layout
```

---

## Future Enhancements

### 1. Smart Room Placement

Use graph neural network or heuristic placement:

- Bathrooms near bedrooms
- Kitchen near dining room
- Balconies on exterior
- Entrance near front door

### 2. Multi-Criteria Sorting

Allow user to sort results by:

- Match percentage
- Total area
- Number of rooms
- Compactness score

### 3. Interactive Adjustment

- Slider to adjust room sizes
- Drag-and-drop room placement
- Visual edge editing

### 4. Batch Auto-Adjust

- Automatically adjust all top results
- Compare multiple adjusted versions
- Allow A/B testing

### 5. Undo/Redo

- Save graph state history
- Allow reverting auto-adjustments
- Compare before/after

---

## Contact & Support

For issues or questions about this implementation:

1. Check [Debugging Tips](#debugging-tips) section
2. Review console logs (both Django server and browser)
3. Verify all modified files are saved
4. Check Django URL routing includes new endpoints

**Critical Files:**

- Backend: `Houseweb/views.py`, `model/floorplan.py`
- Frontend: `static/js/buttonEvent.js`, `templates/home.html`

**URLs to add to Django routing:**

```python
path('AutoAdjustGraph/', views.AutoAdjustGraph, name='AutoAdjustGraph'),
```

---

## Version History

- **v1.0** (2025-12-22): Initial implementation
  - Similarity-based fallback
  - Match percentage display
  - Auto-adjustment feature

---

## Last Updated: December 22, 2025

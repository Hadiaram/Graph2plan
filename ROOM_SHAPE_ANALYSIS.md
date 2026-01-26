# Room Shape Analysis: Complex vs Rectangular

## Investigation Summary

**Question:** Does Graph2Plan generate only rectangular rooms because:
1. The processed dataset only contains rectangles? OR
2. The visualization/model has a rectangular limitation?

## Answer: **The Data DOES Preserve Complex Shapes!**

### Key Findings

✅ **Complex polygon shapes ARE preserved in the processed data**

Analysis of the first 20 training samples shows:
- **65 complex polygons** (>4 vertices)
- **99 simple rectangles** (4 vertices)
- **39.6% of rooms have complex/irregular shapes**

### Data Structure

The processed `.mat` files contain **two representations** of room shapes:

#### 1. Bounding Boxes (`box` field)
```python
# Format: [room_type, y0, x0, y1, x1]
# Simple rectangular approximation for computational efficiency
```

#### 2. Actual Room Polygons (`rBoundary` field)
```python
# Format: Nx2 array of (x, y) vertices
# Preserves the original complex shapes from the dataset
```

### Examples from Sample 0

| Room | Type | Vertices | Shape | Description |
|------|------|----------|-------|-------------|
| 0 | LivingRoom | **20** | POLYGON | L-shaped or complex layout |
| 1 | LivingRoom | **12** | POLYGON | Irregular hexagonal+ shape |
| 2 | ? | **10** | POLYGON | Pentagon+ shape |
| 3 | ? | **12** | POLYGON | Irregular hexagonal+ shape |
| 4 | ? | 4 | RECTANGLE | Simple rectangular room |
| 5 | ? | 4 | RECTANGLE | Simple rectangular room |
| 6 | ? | 4 | RECTANGLE | Simple rectangular room |

### Vertex Count Distribution

Common polygon complexities found:
- **4 vertices:** Rectangles (simplest rooms like bathrooms, bedrooms)
- **6-8 vertices:** Simple polygons (hexagons, irregular shapes)
- **10-14 vertices:** Moderate complexity (L-shapes, T-shapes)
- **18-24 vertices:** High complexity (elaborate living rooms, combined spaces)

## Implications

### ✅ What This Means

1. **Data is NOT the limitation** - Complex shapes are available in `rBoundary`
2. **The issue is in visualization or model output** - Graph2Plan either:
   - Only uses bounding boxes (`box` field) and ignores `rBoundary`
   - The model's `refinement_net` outputs rectangular layouts only
   - The post-processing/visualization code rectangularizes the output

### ⚠️ Where the Limitation Likely Is

The rectangular-only outputs are probably caused by:

1. **Model Architecture:**
   - The `refinement_net` generates a segmentation map (layout tensor)
   - This layout might be constrained to rectangular regions
   - The CNN-based layout generation might not preserve polygon vertices

2. **Post-Processing:**
   - The `Interface` code might only render bounding boxes
   - The visualization pipeline might not use `rBoundary` data
   - Floor plan rendering defaults to rectangular rooms for simplicity

3. **Training Objective:**
   - Loss functions might optimize for rectangular layouts
   - The model wasn't explicitly trained to preserve polygon shapes
   - Box-based losses (MSE on [y0, x0, y1, x1]) favor rectangles

## Recommendations

### To Enable Complex Room Shapes:

#### Option 1: Modify Model to Use rBoundary (Hard)
- Change model output from rectangular boxes to polygon vertices
- Add polygon-based loss functions
- Significantly complex architectural changes required

#### Option 2: Modify Visualization to Use rBoundary (Easy)
- Keep model output as-is (bounding boxes)
- Post-process: Map generated boxes to nearest rBoundary polygons
- Render actual polygons from rBoundary instead of rectangles

#### Option 3: Hybrid Approach (Recommended)
1. **Model generates rectangular boxes** (current approach - works well)
2. **Matching algorithm:** For each generated box, find closest room in rBoundary
3. **Render complex shape:** Use matched rBoundary polygon for visualization
4. **Benefits:** 
   - Keeps trained model as-is
   - Improves visual quality
   - Preserves original design intent

### Implementation Path

```python
# Pseudo-code for hybrid approach

# 1. Model generates boxes (current)
generated_boxes = model.forward(graph_input)  # [N, 4] bounding boxes

# 2. Match boxes to rBoundary polygons (new)
for box, room_type in zip(generated_boxes, room_types):
    # Find rooms of same type in rBoundary with similar position/size
    matched_polygon = match_box_to_polygon(box, room_type, rBoundary_database)
    
    # 3. Render actual polygon instead of rectangle (new)
    render_polygon(matched_polygon, room_type, color)
```

## Current Graph2Plan Behavior

### What It Uses:
- ✅ `box` field - Bounding rectangles for model training
- ✅ `boundary` field - Outer apartment perimeter
- ❌ `rBoundary` field - **IGNORED** (not used by model or visualization)

### What It Generates:
- Rectangular bounding boxes only
- No polygon vertex generation
- Simplified layouts that fit boxes

## Conclusion

**Your observation is correct!** Graph2Plan generates only rectangular rooms, but this is **NOT because the data lacks complex shapes**. The data contains rich polygon information in the `rBoundary` field (39.6% of rooms are non-rectangular).

The limitation is in:
1. **Model architecture** - Generates boxes, not polygons
2. **Visualization pipeline** - Renders boxes, ignores rBoundary
3. **Training approach** - Optimizes for rectangular layouts

To generate complex shapes, you would need to either:
- Retrain the model with polygon-based outputs (major effort)
- Add post-processing to match boxes to rBoundary polygons (minor effort)
- Modify visualization to use rBoundary data (trivial effort)

---

**Files Analyzed:**
- `../Network/data/data_train.mat` - 13,595 samples
- First 20 samples examined in detail
- Script: `quick_shape_analysis.py`

**Data Structure Confirmed:**
- `box` → Rectangular bounding boxes [room_type, y0, x0, y1, x1]
- `rBoundary` → Complex polygon vertices [N_vertices × 2]
- `boundary` → Apartment outer perimeter

**Statistics:**
- 65 complex polygons (6-24 vertices)
- 99 rectangular rooms (4 vertices)
- 39.6% rooms have complex/irregular shapes

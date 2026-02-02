# Integration Guide: Connecting Refinement to Interface

This guide shows how to integrate the Python geometric refinement with the Graph2plan web interface.

---

## 📁 New Folder Structure

The folders have been reorganized:

```
PostProcess/                      # Main post-processing module
├── app.py                        # Model inference application
├── g2p/                          # Graph-to-Plan model code
│   ├── model.py                  # Neural network model
│   ├── align.py                  # Original MATLAB wrapper
│   └── ...
├── refinement/                   # ⭐ NEW: Geometric refinement (MATLAB replacement)
│   ├── __init__.py               # Package
│   ├── geometry_utils.py         # Geometric utilities
│   ├── boundary_align.py         # Step 1: Boundary snapping
│   ├── visualize.py              # Visualization tools
│   ├── integration.py            # ⭐ Interface integration
│   ├── test_boundary_align.py    # Tests
│   ├── README.md                 # Documentation
│   └── INTEGRATION_GUIDE.md      # This file
└── test.py
```

**Why this structure?**
- All post-processing code is in one place (`PostProcess/`)
- Geometric refinement is a submodule (`refinement/`)
- Clear separation between model code (`g2p/`) and refinement (`refinement/`)

---

## 🔌 How to Integrate with Interface

### Option 1: Update Interface/Houseweb/views.py (Recommended)

Replace the MATLAB call with the Python implementation.

**Current code (lines ~1091-1103):**

```python
if engview is not None and HAS_MATLAB:
    # Use MATLAB alignment
    boundary_mat = matlab.double(boundary)
    rType_mat = matlab.double(rType.tolist())
    Edge_mat = matlab.double(Edge)
    Box_mat = matlab.double(Box)
    box_refine = engview.align_fp(boundary_mat, Box_mat, rType_mat, Edge_mat, 18, False, nargout=3)
    box_out = box_refine[0]
    box_order = box_refine[1]
    rBoundary = box_refine[2]
else:
    # Use Python fallback
    box_out, box_order, rBoundary = _python_fallback_align(boundary, Box, rType.tolist(), Edge, 18)
```

**Updated code:**

```python
# Try Python refinement first, fall back to MATLAB if needed
try:
    from PostProcess.refinement.integration import align_fp_python

    # Use pure Python geometric refinement
    box_out, box_order, rBoundary = align_fp_python(
        boundary=boundary,
        boxes=Box,
        room_types=rType,
        edges=Edge,
        fp_id=str(fp_id),  # or appropriate floor plan identifier
        threshold=8.0,     # Adjust as needed (8-18 pixels)
        draw_result=False  # Set True for debugging
    )
    print("[Interface] Using Python geometric refinement")

except ImportError as e:
    print(f"[Interface] Python refinement not available: {e}")

    # Fall back to MATLAB if Python refinement fails
    if engview is not None and HAS_MATLAB:
        print("[Interface] Falling back to MATLAB refinement")
        boundary_mat = matlab.double(boundary)
        rType_mat = matlab.double(rType.tolist())
        Edge_mat = matlab.double(Edge)
        Box_mat = matlab.double(Box)
        box_refine = engview.align_fp(boundary_mat, Box_mat, rType_mat, Edge_mat, 18, False, nargout=3)
        box_out = box_refine[0]
        box_order = box_refine[1]
        rBoundary = box_refine[2]
    else:
        # Use old Python fallback as last resort
        print("[Interface] Using basic Python fallback")
        box_out, box_order, rBoundary = _python_fallback_align(boundary, Box, rType.tolist(), Edge, 18)
```

### Option 2: Update PostProcess/g2p/align.py

Replace the MATLAB call in the existing `align.py` module.

**Find this section in `PostProcess/g2p/align.py`:**

```python
def align_fp_refine(boundary, boxes, types, edges, image, threshold=REFINE_ThRESHOLD, dtype=int):
    if HAS_MATLAB and eng is not None:
        # MATLAB alignment code...
    else:
        # Python fallback
        return align_fp_python_fallback(boundary, boxes, types, edges, image, threshold, dtype)
```

**Update to:**

```python
def align_fp_refine(boundary, boxes, types, edges, image, threshold=REFINE_ThRESHOLD, dtype=int):
    # Try new Python refinement first
    try:
        from ..refinement.integration import align_fp_python

        new_boxes, order, room_boundaries = align_fp_python(
            boundary=boundary,
            boxes=boxes,
            room_types=types,
            edges=edges,
            fp_id="refine",
            threshold=threshold,
            draw_result=False
        )

        # Convert order from list of lists to flat array (MATLAB uses 1-indexed)
        order_flat = np.array([o[0] - 1 for o in order])  # Convert to 0-indexed

        return np.array(new_boxes), order_flat, room_boundaries

    except ImportError:
        # Fall back to MATLAB if new refinement not available
        if HAS_MATLAB and eng is not None:
            # MATLAB code...
        else:
            # Old Python fallback
            return align_fp_python_fallback(boundary, boxes, types, edges, image, threshold, dtype)
```

---

## 🧪 Testing the Integration

### Test 1: Standalone Test

```bash
cd /mnt/c/Users/hmbashir/source/Graph2plan
python -m PostProcess.refinement.integration
```

This runs the built-in test in `integration.py`.

### Test 2: Import Test

```python
# Test in Python console
import sys
sys.path.insert(0, '/mnt/c/Users/hmbashir/source/Graph2plan')

from PostProcess.refinement.integration import align_fp_python
import numpy as np

# Create test data
boundary = np.array([[0, 0], [100, 0], [100, 100], [0, 100]])
boxes = np.array([[5, 10, 45, 50], [55, 10, 95, 50]])
room_types = np.array([0, 1])
edges = np.array([[0, 1]])

# Run refinement
new_boxes, order, boundaries = align_fp_python(
    boundary, boxes, room_types, edges, "test", threshold=8.0
)

print("Success! Got", len(new_boxes), "refined boxes")
```

### Test 3: Integration Test with Interface

1. Start the Django server:
```bash
cd /mnt/c/Users/hmbashir/source/Graph2plan/Interface
python manage.py runserver
```

2. Create a floor plan in the web interface

3. Check the console output for:
```
[Python Refinement] Processing floor plan ...
  Step 1: Aligning boxes with boundary...
    ✓ Aligned X/Y boxes
  ...
```

---

## 🔧 Configuration

### Adjust Threshold

The threshold controls how aggressively boxes snap to boundaries:

```python
# In views.py or align.py
align_fp_python(
    ...,
    threshold=8.0,  # ← Adjust this
    ...
)
```

**Threshold guide:**
- **4-6 pixels**: Conservative, preserves AI predictions
- **8 pixels** (default): Balanced, good for most cases
- **10-15 pixels**: Aggressive, snaps more edges
- **18 pixels**: Very aggressive (MATLAB default)

### Enable Visualization

For debugging, enable visualization output:

```python
align_fp_python(
    ...,
    draw_result=True,  # ← Save visualizations
    ...
)
```

(Note: Visualization saving not yet implemented, will save to `/tmp/` when complete)

---

## 📊 Current Implementation Status

| Step | Status | Description |
|------|--------|-------------|
| **Step 1: Boundary Alignment** | ✅ Complete | Snap boxes to boundary walls |
| **Step 2: Neighbor Alignment** | ⬜ TODO | Close gaps between adjacent rooms |
| **Step 3: Gap Filling** | ⬜ TODO | Expand rooms to fill remaining gaps |
| **Step 4: Polygon Generation** | ⬜ TODO | Create final room polygons |

**Current behavior:**
- Step 1 works fully (boundary snapping)
- Steps 2-4 use placeholder implementations
- Output is valid but not fully refined yet

**Next steps:**
1. Implement neighbor alignment
2. Implement gap filling
3. Implement polygon generation with boundary cropping
4. Add visualization saving

---

## 🐛 Troubleshooting

### Import Error: No module named 'PostProcess'

**Solution:** Make sure you're running from the Graph2plan root directory:

```bash
cd /mnt/c/Users/hmbashir/source/Graph2plan
python -m PostProcess.refinement.integration
```

Or add the path in your code:
```python
import sys
sys.path.insert(0, '/mnt/c/Users/hmbashir/source/Graph2plan')
```

### Import Error: No module named 'numpy'

**Solution:** Install dependencies in your environment:

```bash
source g2p-env/Scripts/activate  # or g2p-train-env
pip install numpy matplotlib
```

### Function returns wrong format

**Check:** The `align_fp_python()` function returns:
- `new_boxes`: List of lists `[[x1,y1,x2,y2], ...]`
- `order`: List of lists `[[idx], ...]` (1-indexed)
- `room_boundaries`: List of lists of lists `[[[x,y], ...], ...]`

This matches MATLAB's output format exactly.

### Boxes don't snap to boundary

**Check:**
1. Is threshold too low? Try increasing it.
2. Are boxes actually close to boundary? Print distances to debug.
3. Enable `verbose=True` in `align_all_boxes_with_boundary()` to see details.

---

## 💡 Example: Complete Integration

Here's a complete example of integrating with views.py:

```python
# At the top of views.py, add import
from PostProcess.refinement.integration import align_fp_python

# In your Save_Editbox function (or similar), replace MATLAB call:
def Save_Editbox(request):
    # ... your existing code to get boundary, Box, rType, Edge ...

    # NEW: Use Python refinement
    try:
        print("[DEBUG] Attempting Python geometric refinement...")

        box_out, box_order, rBoundary = align_fp_python(
            boundary=boundary,
            boxes=Box,
            room_types=rType,
            edges=Edge,
            fp_id=str(fp_end.id) if hasattr(fp_end, 'id') else "unknown",
            threshold=8.0,  # Adjust based on your needs
            draw_result=False
        )

        print(f"[DEBUG] Python refinement successful! Got {len(box_out)} boxes")

    except Exception as e:
        print(f"[ERROR] Python refinement failed: {e}")
        print("[DEBUG] Falling back to MATLAB or simple Python fallback...")

        # Your existing MATLAB or fallback code here
        if engview is not None and HAS_MATLAB:
            # MATLAB code
            pass
        else:
            # Old Python fallback
            box_out, box_order, rBoundary = _python_fallback_align(...)

    # Continue with your existing code
    fp_end.data.newBox = np.array(box_out)
    fp_end.data.order = np.array(box_order)
    fp_end.data.rBoundary = [np.array(rb) for rb in rBoundary]
    # ...
```

---

## 📞 Need Help?

- **Issues with imports**: Check your Python path and virtual environment
- **Issues with data format**: See the type hints in `integration.py`
- **Issues with results**: Enable `verbose=True` for debugging output
- **Feature requests**: Steps 2-4 coming soon!

---

**Last Updated**: 2026-01-30
**Status**: Step 1 complete, Steps 2-4 in progress

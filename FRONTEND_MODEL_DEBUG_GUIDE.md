# Frontend-Model Connection Debug Guide

## Project Overview

**Graph2Plan** is a floor plan generation system that uses graph neural networks to create residential floor plans. The system has two main components:

1. **Neural Network Model** (`Network/` folder) - PyTorch-based GNN that generates floor plans from graph representations
2. **Web Interface** (`Interface/` folder) - Django web app with D3.js visualization for interactive floor plan editing

## System Architecture

```
User Browser (JavaScript/D3.js)
    ↓ AJAX requests
Django Web Server (views.py)
    ↓ Python function calls
Model Inference (test.py)
    ↓ PyTorch model
Trained Neural Network (model.pth)
    ↓ Results
Floor Plan Visualization (D3.js)
```

## Key Files and Their Roles

### Frontend Layer
- **`Interface/templates/home.html`** (lines 839-842)
  - Contains the "Show Plan" button HTML
  - Purple button that triggers AI model floor plan generation
  - Initially hidden, becomes visible when graph editing is active

- **`Interface/static/js/buttonEvent.js`**
  - **`CreateLeftFloorPlan()` function** (lines 752-813): Renders floor plan visualization using D3.js
  - **Show Plan button click handler** (lines 822-851): Sends AJAX request to backend with graph data
  - **Key AJAX call**: `$.get("/index/AdjustGraph/", {...})` sends current graph state to regenerate floor plan

### Backend Layer
- **`Interface/Houseweb/views.py`**
  - **`AdjustGraph()` function** (line ~658): Handles `/index/AdjustGraph/` endpoint
  - Parses incoming graph data (nodes, edges)
  - Calls `mltest.get_userinfo_adjust()` to run AI model
  - Returns JSON with generated floor plan data

- **`Interface/model/test.py`**
  - **`load_model()` function** (lines 64-79): Loads trained PyTorch model
  - **`get_data()` function** (lines 30-37): Prepares input tensors for model
  - **`test()` function** (lines 39-62): Runs model inference
  - **`get_userinfo_adjust()` function** (lines 89-246): Main pipeline for graph adjustment + model inference

### Model Layer
- **`Interface/model/model.pth`** (~30MB)
  - Trained neural network checkpoint
  - 50 epochs on 1000 ResPlan floor plans
  - Final validation loss: 0.0076
  - Trained on GPU but compatible with CPU

## Data Flow: Show Plan Button Click

1. **User clicks "Show Plan" button** → `buttonEvent.js:822-851`
2. **JavaScript collects graph state** → `GetEditGraph(ret['rmpos'])`
   - Node positions, room types, sizes
   - Edge connections between rooms
3. **AJAX GET request** → `/index/AdjustGraph/`
   ```javascript
   $.get("/index/AdjustGraph/", {
       'NewGraph': JSON.stringify(currentGraph),
       'userRoomID': rooms.toString().split(',')[0],
       'adptRoomID': roomID
   }, ...)
   ```
4. **Django view parses request** → `views.py:AdjustGraph()`
5. **Calls model inference** → `test.py:get_userinfo_adjust()`
   - Loads model if not already loaded
   - Processes graph data into tensors
   - Runs neural network inference
   - Post-processes output (alignment, refinement)
6. **Returns JSON to frontend**
   ```json
   {
       "roomret": [[boxes], [room_types]],
       "exterior": "polygon_points",
       "door": "x1,y1,x2,y2"
   }
   ```
7. **JavaScript renders floor plan** → `CreateLeftFloorPlan()`

## Recent Changes

### 1. Show Plan Button Addition (Latest)
**Branch:** AI-training branch
**Date:** January 2026

Added AI-powered floor plan generation to the frontend:
- Added Show Plan button to `home.html` (lines 839-842)
- Added `CreateLeftFloorPlan()` function to `buttonEvent.js` (lines 752-813)
- Added button click handler (lines 822-851)
- Connected to `/index/AdjustGraph/` endpoint

**Key Difference from Original Branch:**
- Original branch: Retrieved pre-computed floor plans from database
- AI-training branch: Generates floor plans in real-time using trained model

### 2. CPU Compatibility Fix (Latest)
**File:** `Interface/model/test.py`
**Lines modified:** 26-28, 30-37, 64-79

Fixed GPU dependency issue so Interface can run on machines without CUDA GPU:

```python
# Added device detection
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Changed .cuda() to .to(DEVICE)
batch[0] = batch[0].unsqueeze(0).to(DEVICE)

# Fixed model loading
if torch.cuda.is_available():
    model.load_state_dict(torch.load('./model/model.pth', map_location={'cuda:0': 'cuda:0'}))
else:
    model.load_state_dict(torch.load('./model/model.pth', map_location='cpu'))
```

**Previous Error:**
```
RuntimeError: Cannot access accelerator device when none is available
at test.py:63 - model.cuda(0)
```

## Current System State

### Working Components
✅ Model training pipeline (CPU/GPU compatible)
✅ Model checkpoint saved and accessible
✅ Show Plan button UI implemented
✅ AJAX request to backend endpoint
✅ Device detection (CPU/GPU)

### Recently Fixed
✅ GPU hardcoding issue in `test.py`
✅ Model loading now works on CPU-only machines

### Potential Issues to Debug

#### 1. Model Inference Performance on CPU
- **Expected:** 10-100x slower than GPU
- **Test:** Check browser console for timing logs
- **Location:** `test.py:207` prints "model test time"

#### 2. MATLAB Fallback
- **Issue:** MATLAB engine may not be available
- **Fallback:** Python implementation in `views._python_fallback_align()`
- **Check:** `test.py:221` - conditional on `vw.engview is not None and HAS_MATLAB`

#### 3. Model Loading Path
- **Expected path:** `./model/model.pth` (relative to working directory)
- **Common issue:** Working directory not set to `Interface/` folder
- **Debug:** Print `os.getcwd()` in `load_model()`

#### 4. Tensor Shape Mismatches
- **Common issue:** Graph with different number of rooms than training data
- **Location:** `test.py:191` - `fp_end.data.box = np.array(newbox)`
- **Check:** Print shapes of tensors before model inference

#### 5. AJAX Response Parsing
- **Issue:** Backend may return error but frontend doesn't handle it
- **Location:** `buttonEvent.js:845-851` - `.fail()` handler logs to console
- **Debug:** Check browser console for error messages

## Debugging Checklist

### Backend (Django/Model)
```bash
# 1. Check working directory
cd /mnt/c/Users/hmbashir/source/Graph2plan/Interface

# 2. Verify model file exists
ls -lh model/model.pth

# 3. Check device detection
python -c "import torch; print('CUDA:', torch.cuda.is_available())"

# 4. Run Django server with verbose logging
python manage.py runserver --verbosity 2

# 5. Monitor Django console for errors when clicking Show Plan
```

### Frontend (Browser)
```javascript
// 1. Open browser developer console (F12)

// 2. Check for JavaScript errors
// Look for errors in Console tab

// 3. Monitor network requests
// Network tab → Filter by XHR → Look for /index/AdjustGraph/

// 4. Check AJAX response
// Click on the request → Preview/Response tab

// 5. Verify graph data being sent
// Console should log: "Current graph to send:", currentGraph
```

### Model Inference
```python
# Add debug prints to test.py:get_userinfo_adjust()

# Line 203 (before model inference):
print("Running model inference...")
print("Number of rooms:", len(newbox))
print("Number of edges:", len(adjust_Edge))

# Line 204 (after model inference):
print("Model output shapes:")
print("boxes_pred:", boxes_pred.shape)
print("gene_layout:", gene_layout.shape)
```

## Common Error Patterns

### 1. "Cannot access accelerator device"
- **Status:** FIXED (see CPU Compatibility Fix above)
- **Cause:** Hardcoded `.cuda()` calls
- **Solution:** Use device detection

### 2. "Model file not found"
- **Cause:** Working directory incorrect
- **Solution:** Ensure Django runs from `Interface/` folder
- **Check:** Model path is relative: `./model/model.pth`

### 3. "Tensor shape mismatch"
- **Cause:** Graph structure incompatible with model
- **Debug:** Print tensor shapes before model call
- **Location:** `test.py:204` - before `test(vw.model, fp_end)`

### 4. "AJAX request returns 500 error"
- **Cause:** Backend exception during model inference
- **Debug:** Check Django console output
- **Fix:** Add try-catch in `views.py:AdjustGraph()`

### 5. "Floor plan doesn't render"
- **Cause:** JavaScript error or malformed response
- **Debug:** Check browser console
- **Verify:** Response has correct structure: `{roomret, exterior, door}`

## Testing Workflow

### 1. Basic Connectivity Test
```bash
# Start Django server
cd /mnt/c/Users/hmbashir/source/Graph2plan/Interface
python manage.py runserver

# Open browser to http://localhost:8000
# Check if home page loads
```

### 2. Model Loading Test
```python
# In Django shell
python manage.py shell

>>> from model import test
>>> print("Device:", test.DEVICE)
>>> model = test.load_model()
>>> print("Model loaded successfully!")
```

### 3. Show Plan Button Test
1. Load a floor plan in the web interface
2. Open browser console (F12)
3. Click "Show Plan" button
4. Check console logs:
   - "Showing current floor plan..."
   - "Current graph to send:", {...}
   - Should see AJAX request complete
5. Verify left panel shows generated floor plan

### 4. End-to-End Test
```javascript
// Browser console test
$.get("/index/AdjustGraph/", {
    'NewGraph': JSON.stringify([[],[],[]]),  // minimal graph
    'userRoomID': 'test_room',
    'adptRoomID': 'train_room'
}, function(data) {
    console.log("Success!", data);
}).fail(function(xhr) {
    console.error("Failed!", xhr.responseText);
});
```

## Environment Details

### Training Environment
- **Machine:** 80GB GPU machine
- **Device:** CUDA GPU
- **Training time:** 50 epochs in 1 hour 3 minutes
- **Dataset:** 800 train + 200 test ResPlan floor plans

### Interface Environment
- **Machine:** Windows PC (WSL2)
- **Device:** CPU only (no CUDA GPU)
- **Expected performance:** Slower but functional
- **Path:** `/mnt/c/Users/hmbashir/source/Graph2plan/Interface/`

## File Locations Reference

```
/mnt/c/Users/hmbashir/source/Graph2plan/
├── Interface/
│   ├── model/
│   │   ├── model.pth                    # Trained model (30MB)
│   │   ├── test.py                      # Model inference (CPU-compatible)
│   │   └── model.py                     # Model architecture
│   ├── Houseweb/
│   │   └── views.py                     # Django views (AdjustGraph endpoint)
│   ├── templates/
│   │   └── home.html                    # HTML with Show Plan button
│   ├── static/js/
│   │   └── buttonEvent.js               # Frontend logic
│   └── manage.py
├── Network/
│   └── train.py                         # Training script (CPU-compatible)
└── DataPreparation/
    └── data/
        └── data.mat                     # Training/test data
```

## Next Steps for Debugging

1. **Verify the Show Plan button appears** when graph editing is active
2. **Click the button and monitor browser console** for errors
3. **Check Django console output** for backend errors
4. **Verify AJAX request completes** in Network tab
5. **Check if floor plan renders** in left panel
6. **If errors occur:**
   - Note exact error message
   - Check which layer failed (frontend JS, Django view, model inference)
   - Review relevant section above

## Contact Points for Issues

### Model Loading Issues
- File: `Interface/model/test.py`
- Function: `load_model()` (lines 64-79)
- Key: Device detection and checkpoint loading

### AJAX Request Issues
- File: `Interface/static/js/buttonEvent.js`
- Function: Show Plan button handler (lines 822-851)
- Endpoint: `/index/AdjustGraph/`

### Backend Processing Issues
- File: `Interface/Houseweb/views.py`
- Function: `AdjustGraph()` (~line 658)
- Key: Graph parsing and model invocation

### Visualization Issues
- File: `Interface/static/js/buttonEvent.js`
- Function: `CreateLeftFloorPlan()` (lines 752-813)
- Key: D3.js rendering logic

---

**Document Created:** January 15, 2026
**Last Model Update:** January 8, 2026
**Current Branch:** AI-training branch
**Status:** CPU compatibility implemented, Show Plan button functional

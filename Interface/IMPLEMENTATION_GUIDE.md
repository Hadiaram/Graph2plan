# Graph2Plan Floor Plan Generation Interface - Implementation Guide

## Table of Contents
1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [How It Works](#how-it-works)
4. [Setup & Installation](#setup--installation)
5. [User Guide](#user-guide)
6. [Developer Guide](#developer-guide)
7. [Debugging Journey](#debugging-journey)
8. [Known Limitations](#known-limitations)
9. [Future Improvements](#future-improvements)

---

## Overview

### What is Graph2Plan?

Graph2Plan is a **graph-to-image deep learning system** that generates residential floor plan layouts from graph representations. The system allows users to:

1. **Define floor plans as graphs**: Nodes represent rooms (living room, kitchen, bedroom, etc.), and edges represent adjacency relationships
2. **Retrieve similar layouts**: Find existing floor plans from a database that match user requirements
3. **Generate new layouts**: Use a neural network to generate realistic floor plan layouts adapted to custom building boundaries
4. **Edit and refine**: Modify room positions, sizes, and relationships interactively

### Key Technologies

- **Backend**: Django 5.2.7 (Python 3.13)
- **Deep Learning**: PyTorch (Graph Neural Networks + CNNs)
- **Frontend**: D3.js for interactive graph visualization
- **Data Storage**: Pickle files (preprocessed RPLAN/ResPlan dataset)
- **Optional**: MATLAB Engine (with Python fallback)

---

## System Architecture

### High-Level Components

```
┌─────────────────────────────────────────────────────────────┐
│                        Web Interface                         │
│  (Django + D3.js - home.html, buttonEvent.js)               │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│                    Django Backend                            │
│              (Houseweb/views.py)                             │
│  • NumSearch: Room-based retrieval                           │
│  • TransGraph: Use pre-existing floor plan                   │
│  • AdjustGraph: Generate new floor plan from edited graph    │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│                   Model Layer                                │
│              (model/test.py, model.py)                       │
│  • FloorPlan: Data structure for floor plan representation   │
│  • Model: Graph2Plan neural network (GNN + CNN)             │
│  • Inference: Generate boxes from graph + boundary           │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│                   Data Storage                               │
│  • data_test_converted.pkl: Test floor plans (2 samples)    │
│  • data_train_converted.pkl: Training floor plans (8)       │
│  • model.pth: Trained Graph2Plan weights (30MB)             │
│  • snapshot_train/*.png: Floor plan thumbnails               │
└─────────────────────────────────────────────────────────────┘
```

### Directory Structure

```
Interface/
├── Houseweb/               # Django app
│   ├── views.py           # Main backend logic
│   └── urls.py            # URL routing
├── model/                  # Deep learning models
│   ├── model.py           # Graph2Plan architecture
│   ├── test.py            # Inference functions
│   ├── floorplan.py       # Floor plan data structure
│   ├── utils.py           # Vocabulary, room types
│   └── model.pth          # Trained weights (30MB)
├── static/
│   ├── Data/              # Preprocessed datasets
│   │   ├── data_test_converted.pkl
│   │   ├── data_train_converted.pkl
│   │   ├── data_train_eNum.pkl
│   │   ├── rNum_train.npy
│   │   └── snapshot_train/  # Thumbnails
│   ├── js/
│   │   └── buttonEvent.js  # Frontend interactions
│   └── *.mat              # Generated floor plan data
├── templates/
│   └── home.html          # Main interface
└── manage.py              # Django entry point
```

---

## How It Works

### 1. User Workflow

#### Step 1: Select Boundary
User selects a test boundary (building outline with door position) from 2 available samples.

#### Step 2: Specify Requirements
User specifies desired rooms:
- Room types (e.g., 2 bedrooms, 1 kitchen, 1 bathroom)
- Optionally: room count only (system retrieves best matches)

#### Step 3: Retrieval
**NumSearch** endpoint searches training database using:
- Room count similarity
- Boundary shape similarity (TF-IDF on room counts)
Returns top 20 matching floor plans as thumbnails.

#### Step 4: Adapt Layout
User selects a retrieved layout. **TransGraph** endpoint:
- Loads the selected training floor plan's graph structure
- Aligns it to the user's test boundary (rotation + scaling)
- Displays the adapted graph on the canvas

#### Step 5: Edit Graph (Optional)
User can:
- Move room nodes
- Add/remove rooms
- Add/remove connections (adjacency edges)
- Resize rooms (scale slider)

#### Step 6: Generate Floor Plan
**AdjustGraph** endpoint (triggered by "Show Plan" button):
1. Parses edited graph from frontend
2. Calls **Graph2Plan model** to generate room bounding boxes
3. Aligns boxes to boundary (MATLAB or Python fallback)
4. Renders final floor plan with rooms, walls, doors, windows

### 2. Technical Deep Dive

#### Graph2Plan Model Architecture

```
Input Graph                              Input Boundary Image
(nodes + edges)                          (128x128x3)
      │                                          │
      ▼                                          ▼
┌─────────────────┐                    ┌──────────────┐
│  Room Embeddings │                    │  CNN Encoder │
│  (ID 0-14)      │                    │  (Inside/    │
│                 │                    │   Boundary/  │
│  Edge Embeddings│                    │   Door)      │
│  (10 relations) │                    └──────┬───────┘
└────────┬────────┘                           │
         │                                     │
         ▼                                     │
┌─────────────────┐                           │
│  Graph Conv Net │                           │
│  (5 layers GNN) │                           │
└────────┬────────┘                           │
         │                                     │
         ├─────────────────────────────────────┤
         │                                     │
         ▼                                     ▼
┌──────────────────────────────────────────────┐
│         Box Prediction Head                   │
│  (Predict center + size for each room)       │
└──────────────────┬───────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────┐
│         Layout Refinement Network            │
│  (CNN-based refinement using boundary mask)  │
└──────────────────┬───────────────────────────┘
                   │
                   ▼
              Final Boxes
           (x1, y1, x2, y2)
```

**Key Model Details**:
- **Vocabulary Size**: 15 room types (0-14) due to trained model limitation
- **Room Type Mapping**: Types 15-17 (FrontDoor, InteriorWall, InteriorDoor) mapped to similar types 10/14
- **Input**:
  - Room types tensor (N rooms)
  - Edge triples tensor (E edges x 3: source, relation, target)
  - Boundary image (128x128x3)
  - Room attributes (position grid + area bins)
- **Output**: Bounding boxes in normalized coordinates [0, 1]

#### FloorPlan Data Structure

```python
class FloorPlan:
    data.boundary   # (N, 4): boundary points [x, y, type, flag]
    data.box        # (M, 5): room boxes [x1, y1, x2, y2, room_type]
    data.edge       # (E, 3): edges [u, v, relation_type]
    data.gene       # (128, 128): generated layout raster
    data.newBox     # Aligned boxes after MATLAB/Python processing
    data.order      # Room ordering for layout
    data.rBoundary  # Room-specific boundaries
    data.windows    # Window positions
    data.doors      # Door positions
```

---

## Setup & Installation

### Prerequisites

- **Python 3.13** (or 3.10+)
- **Windows** (WSL supported)
- **4GB+ RAM** (model inference)
- **Optional**: MATLAB R2020b+ with Python Engine

### Installation Steps

```bash
# 1. Navigate to Interface directory
cd /mnt/c/Users/hmbashir/source/Graph2plan/Interface

# 2. Install Python dependencies
pip install django==5.2.7 torch torchvision scipy numpy opencv-python pandas

# 3. Verify data files exist
ls static/Data/data_test_converted.pkl
ls static/Data/data_train_converted.pkl
ls model/model.pth

# 4. Run Django server
python manage.py runserver

# 5. Open browser
# Navigate to http://127.0.0.1:8000/home
```

### Verify Installation

After starting the server, you should see console output:
```
📊 Loaded 2 test floor plans
📊 testNameList has 2 names: ['17054', '14035']
📊 Loaded 8 training floor plans
📊 trainNameList has 8 names: ['16969', '14926', '1448', ...]
🖥️  Model will use device: cuda (or cpu)
MATLAB not available - using Python fallback for alignment
```

---

## User Guide

### Basic Usage

#### 1. Select Test Boundary
- Dropdown in top-left shows 2 available test floor plans
- Each has a different building shape and door position
- Select one to load its boundary outline

#### 2. Specify Room Requirements
Use the "Room Request" panel:
- **Add Rooms**: Click room type buttons (LivingRoom, Kitchen, Bedroom, etc.)
- **Set Quantities**: Each click increments the room count
- **Search**: Click "Search" to find matching floor plans

#### 3. Browse Retrieved Plans
- Top 20 matches displayed as thumbnails
- Click a thumbnail to load that floor plan's graph structure
- Graph appears on canvas with colored circles (rooms) and lines (connections)

#### 4. Edit Graph (Optional)
- **Move Rooms**: Drag room circles
- **Resize Room**: Select room, adjust "Room Size" slider
- **Add Room**: Click "Add Room" button, select type, click canvas to place
- **Add Connection**: Click "Add Edge", click two rooms to connect
- **Delete**: Select room/edge, press Delete key

#### 5. Generate Floor Plan
- Click "**Show Plan**" button
- System generates floor plan layout (takes ~15-20 seconds)
- Floor plan appears on right canvas with:
  - Room boxes (colored by type)
  - Walls (black lines)
  - Doors (yellow markers)
  - Room labels

#### 6. Save (Optional)
- Click "Save" to export .mat file to static/ directory

### Tips

- **MATLAB vs Python**: System auto-detects MATLAB. If unavailable, uses Python fallback (slightly less accurate alignment)
- **Room Types**: FrontDoor/Entrance, Balcony, Bathroom, Kitchen, LivingRoom, MasterRoom, etc.
- **Room Type Limitation**: Model supports 15 room types; FrontDoor (15), InteriorWall (16), InteriorDoor (17) mapped to similar types
- **Boundary Constraints**: Rooms clipped to fit within boundary (except Balconies which extend outside)

---

## Developer Guide

### Key Code Files

#### 1. `Houseweb/views.py` (2000+ lines)

**Initialization Functions**:
```python
def getTestData()      # Load test boundaries (2 samples)
def getTrainData()     # Load training floor plans (8 samples)
def loadModel()        # Load Graph2Plan neural network
def Init()             # Initialize all components on startup
```

**Main Endpoints**:
```python
def NumSearch(request)       # GET: Room-based retrieval
def TransGraph(request)      # GET: Load retrieved floor plan
def AdjustGraph(request)     # GET: Generate floor plan from edited graph
```

**Key Variables**:
```python
STATIC_DIR = r'C:\Users\hmbashir\source\Graph2plan\Interface\static'
DATA_DIR = os.path.join(STATIC_DIR, 'Data')
test_data, testNameList     # Test floor plans
train_data, trainNameList   # Training floor plans
model                       # Graph2Plan PyTorch model
```

#### 2. `model/test.py`

**Core Functions**:
```python
def load_model()                          # Load model.pth
def test(model, fp)                       # Run inference
def get_userinfo_adjust(userRoomID,      # Process graph edit
                        adptRoomID,
                        NewGraph)
```

**Key Variables**:
```python
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = os.path.join(MODEL_DIR, 'model.pth')
```

#### 3. `model/floorplan.py`

**FloorPlan Class**:
```python
def get_input_boundary()    # Create boundary image (128x128x3)
def get_rooms()            # Extract room types (with mapping 15-17→10/14)
def get_attributes()       # Compute position/area attributes
def get_triples()          # Create edge relations
def adapt_graph()          # Align graph to boundary
def adjust_graph()         # Move rooms inside boundary
```

#### 4. `model/utils.py`

**Vocabulary**:
```python
room_label = [
    (0, 'LivingRoom', ...),
    (1, 'MasterRoom', ...),
    ...
    (14, 'Internal', ...)
]

vocab = {
    'object_name_to_idx': {...},  # "LivingRoom" → 0
    'object_idx_to_name': [...],  # 0 → "LivingRoom"
    'pred_name_to_idx': {...},    # "left-of" → 2
    'pred_idx_to_name': [...]     # 2 → "left-of"
}
```

**Room Type Mapping**:
```python
def map_room_type_for_model(room_type):
    # Maps 15→10, 16→14, 17→10 for model compatibility
```

### Adding New Features

#### Add a New Room Type (Requires Retraining)
1. Update `room_label` in `model/utils.py`
2. Update `get_vocab()` to include new type
3. **Retrain model** with new vocabulary
4. Update frontend buttons in `templates/home.html`

#### Add New Endpoint
1. Define view function in `Houseweb/views.py`
2. Add URL mapping in `Houseweb/urls.py`
3. Add frontend button/handler in `static/js/buttonEvent.js`

#### Modify Model Architecture
1. Edit `model/model.py` (Model class)
2. **Retrain from scratch** (pretrained weights incompatible)
3. Update `load_model()` to handle new architecture

### Debugging Tips

**Enable Verbose Logging**:
```python
# In views.py, add at function start:
print(f"🔍 Function called with: {locals()}")
```

**Check Model Output Shapes**:
```python
# In model/test.py, test() function:
print(f"boxes_pred shape: {boxes_pred.shape}")
print(f"gene_layout shape: {gene_layout.shape}")
```

**Inspect Graph Structure**:
```python
# In AdjustGraph, views.py:
print(f"NewGraph nodes: {NewGraph[0]}")
print(f"NewGraph edges: {NewGraph[1]}")
```

**Common Issues**:
- **"index out of range"**: Room type > 14 (check mapping in get_rooms())
- **"tuple has no attribute 'boundary'"**: Variable naming collision (ensure `data` not reused)
- **"FileNotFoundError: ./static/"**: Use absolute paths (STATIC_DIR constant)
- **Model size mismatch**: Vocab changed but using old model.pth (retrain required)

---

## Debugging Journey

This section documents all bugs encountered and fixed during implementation.

### Bug 1: CUDA Device Incompatibility
**Error**: `Cannot access accelerator device when none is available`
**Cause**: Hardcoded `.cuda()` calls in test.py
**Fix**:
```python
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Changed all .cuda() → .to(DEVICE)
```
**Files**: `model/test.py:22-24`

### Bug 2: Trailing Spaces in Room Names
**Error**: `ValueError: '5410' is not in list` (actual list contains `'5410 '`)
**Cause**: Pickle files had trailing spaces in `trainNameList`
**Fix**:
```python
trainNameList = [str(n).strip() for n in train_data['nameList']]
```
**Files**: `Houseweb/views.py:191`, `model/test.py:87,109,273`

### Bug 3: Coordinate Indexing Bug
**Error**: `IndexError: index 256 is out of bounds for axis 0 with size 256`
**Cause**: Using coordinate values (0-256) as array indices (0-255)
**Fix**:
```python
# Before: X[x0] where x0 can be 256
# After: x0/256 (normalize directly)
box = np.array([[x0/256, y0/256, x1/256, y1/256]])
```
**Files**: `model/floorplan.py:63-72`

### Bug 4: Vocab Size Mismatch
**Error**: `RuntimeError: size mismatch for obj_embeddings.weight: copying a param with shape torch.Size([15, 128]) from checkpoint, the shape in current model is torch.Size([18, 128])`
**Cause**: Trained model created with buggy vocab (15 types), but code defined 18 types
**Fix**: Kept vocab at 15, added mapping function for unsupported types 15-17
```python
def map_room_type_for_model(room_type):
    if room_type == 15: return 10  # FrontDoor → Entrance
    elif room_type == 16: return 14  # InteriorWall → Internal
    elif room_type == 17: return 10  # InteriorDoor → Entrance
    return room_type
```
**Files**: `model/utils.py:375-408`, `model/floorplan.py:77`

### Bug 5: Variable Naming Collision
**Error**: `AttributeError: 'tuple' object has no attribute 'boundary'`
**Cause**: Variable `data` reused in loop, overwriting boundary object with tuple
**Fix**: Renamed loop variable from `data` to `room_data`
```python
# Before: data = boxes_end[k], [room_label], order_val
# After: room_data = boxes_end[k], [room_label], order_val
```
**Files**: `Houseweb/views.py:819`

### Bug 6: Relative Path Issues
**Error**: `FileNotFoundError: [Errno 2] No such file or directory: './static/17054.mat'`
**Cause**: Relative paths fail when Django working directory differs
**Fix**: Created absolute path constants
```python
STATIC_DIR = r'C:\Users\hmbashir\source\Graph2plan\Interface\static'
DATA_DIR = os.path.join(STATIC_DIR, 'Data')
MODEL_PATH = os.path.join(MODEL_DIR, 'model.pth')
```
**Files**: `Houseweb/views.py:39-40,166,187-201,929,1122`, `model/test.py:27-28,72`

### Bug 7: OpenCV Polygon Type Error
**Error**: `OpenCV(4.12.0) error: (-215:Assertion failed) p.checkVector(2, CV_32S) > 0`
**Cause**: OpenCV fillPoly requires int32 arrays, was passing float64
**Fix**:
```python
pts = pts.astype(np.int32)
pts_door = pts_door.astype(np.int32)
```
**Files**: `model/floorplan.py:42-43`

### Bug 8: NumPy Indexing Errors
**Error**: `"only integers, slices (:)... are valid indices"`
**Cause**: box_order values used as indices without int conversion/validation
**Fix**: Added comprehensive validation
```python
order_val = box_order[o][0] if len(box_order[o]) > 0 else box_order[o]
order_idx = int(float(order_val)) - 1
if 0 <= order_idx < len(rooms):
    room.append(float(rooms[order_idx]))
```
**Files**: `Houseweb/views.py:745-758,763-776`

---

## Known Limitations

### 1. Model Limitations
- **Vocab Size**: Only supports 15 room types (0-14) due to trained model
- **Room Types 15-17**: Mapped to similar types (FrontDoor→Entrance, etc.)
- **Retraining Required**: Cannot change vocab without full retraining

### 2. Dataset Limitations
- **Small Dataset**: Only 2 test boundaries, 8 training floor plans
- **Limited Diversity**: All samples are residential apartments
- **No Commercial**: No office, retail, or mixed-use layouts

### 3. Performance
- **Inference Time**: 15-20 seconds per generation (CPU)
- **MATLAB Dependency**: Optional but provides better alignment
- **Memory**: Requires 4GB+ RAM for model inference

### 4. UI/UX
- **No Undo**: Graph edits cannot be undone
- **Limited Validation**: No checks for invalid graphs (disconnected rooms, etc.)
- **No Export**: Cannot export to DXF, SVG, or other CAD formats

### 5. Boundary Constraints
- **Fixed Boundaries**: Cannot edit boundary shapes, only select from dataset
- **Door Position**: Fixed in test boundaries, cannot be moved

---

## Future Improvements

### Short-Term (1-2 weeks)

1. **Undo/Redo for Graph Edits**
   - Add history stack in JavaScript
   - Store graph state after each edit
   - Files: `static/js/buttonEvent.js`

2. **Export to DXF/SVG**
   - Add export button
   - Convert boxes to CAD format
   - Files: `Houseweb/views.py` (new endpoint)

3. **Validation & Error Messages**
   - Check graph connectivity
   - Warn about disconnected rooms
   - Files: `static/js/buttonEvent.js`

4. **Loading Indicators**
   - Show progress bar during inference
   - Display "Generating..." message
   - Files: `templates/home.html`, `static/js/buttonEvent.js`

### Medium-Term (1-2 months)

1. **Expand Dataset**
   - Add more test boundaries
   - Include commercial layouts
   - Retrain model on larger dataset

2. **Editable Boundaries**
   - Allow drawing custom boundaries
   - Support irregular shapes
   - Files: New boundary editor module

3. **Real-Time Preview**
   - Show estimated layout during editing
   - Update as user moves rooms
   - Files: `static/js/buttonEvent.js` (debounced inference)

4. **Collaborative Editing**
   - Multi-user support via WebSockets
   - Shared floor plan sessions
   - Files: New Django Channels integration

### Long-Term (3-6 months)

1. **Retrain with Full Vocabulary**
   - Support all 18 room types natively
   - Include FrontDoor, InteriorWall, InteriorDoor
   - Files: Retrain `Network/train.py`, update `model.pth`

2. **3D Visualization**
   - Extrude floor plans to 3D
   - Interactive 3D view
   - Files: New Three.js integration

3. **AI-Assisted Design**
   - Suggest optimal room arrangements
   - Auto-optimize for metrics (circulation, sunlight, etc.)
   - Files: New optimization module

4. **Mobile App**
   - React Native mobile client
   - Touch-based editing
   - Files: New mobile app repository

---

## Troubleshooting

### Issue: Server won't start
**Symptoms**: `ModuleNotFoundError`, `ImportError`
**Solution**:
```bash
pip install --upgrade django torch scipy numpy opencv-python pandas
python manage.py runserver
```

### Issue: "Model test failed"
**Symptoms**: Console shows `⚠️ Model test failed (non-fatal)`
**Solution**: Model expects specific data format. Check:
1. `model.pth` exists (30MB file)
2. Test data loaded: `📊 Loaded 2 test floor plans`
3. Device detected: `🖥️ Model will use device: cuda/cpu`

### Issue: "Show Plan" takes forever
**Symptoms**: Button clicked, no response after 60+ seconds
**Solution**:
1. Check console for errors
2. Reduce graph complexity (fewer rooms)
3. Verify MATLAB/Python fallback working
4. Restart server (memory leak possible)

### Issue: Floor plan looks wrong
**Symptoms**: Rooms overlap, extend outside boundary
**Solution**:
1. Check room types: FrontDoor (15) should map to Entrance (10)
2. Verify boundary loaded: `🔍 AdjustGraph: Loading boundary for testname=...`
3. Check box clipping logic in `views.py:790-811`

### Issue: Cannot save .mat file
**Symptoms**: `FileNotFoundError: ./static/...`
**Solution**: Verify absolute paths set correctly:
```python
# Check views.py line 39:
STATIC_DIR = r'C:\Users\hmbashir\source\Graph2plan\Interface\static'
```

---

## References

### Original Paper
- **Title**: Graph2Plan: Learning Floorplan Generation from Layout Graphs
- **Authors**: Ruizhen Hu, Zeyu Huang, Yuhan Tang, Oliver van Kaick, Hao Zhang, Hui Huang
- **Venue**: ACM Transactions on Graphics (SIGGRAPH 2020)
- **Link**: https://github.com/HanHan55/Graph2Plan

### Dependencies
- **PyTorch**: 2.0+ (https://pytorch.org)
- **Django**: 5.2.7 (https://djangoproject.com)
- **OpenCV**: 4.12.0 (https://opencv.org)
- **D3.js**: 3.5.17 (https://d3js.org)
- **MATLAB Engine**: Optional (https://mathworks.com/help/matlab/matlab-engine-for-python.html)

### Dataset
- **RPLAN**: 80,000+ residential floor plans (not included, requires download)
- **ResPlan**: 8 sample floor plans (included in `static/Data/`)

---

## Contact & Support

### Issues
Report bugs or request features at:
- GitHub Issues: https://github.com/HanHan55/Graph2Plan/issues

### Questions
For technical questions, refer to:
1. This guide (IMPLEMENTATION_GUIDE.md)
2. Original paper documentation
3. Code comments in `Houseweb/views.py`, `model/test.py`

---

## Version History

### v1.0 (Current - December 2025)
- ✅ Fixed all major bugs (8 bugs resolved)
- ✅ Converted to absolute paths
- ✅ Added room type mapping (15-17→10/14)
- ✅ Python fallback for MATLAB alignment
- ✅ Comprehensive error handling
- ✅ Documentation created

### v0.9 (Original Release)
- Graph-based floor plan editing interface
- Neural network inference integration
- RPLAN dataset support
- MATLAB-only alignment

---

**Document Author**: Claude (Anthropic)
**Date**: December 30, 2025
**Version**: 1.0
**Status**: Complete

*This guide is based on the actual debugging and implementation journey of getting the Graph2Plan interface working from a non-functional state to a fully operational system.*

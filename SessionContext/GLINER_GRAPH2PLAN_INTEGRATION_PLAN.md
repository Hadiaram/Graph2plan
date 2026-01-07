# GLiNER + Graph2Plan Integration Plan
**Prompt-to-Floor-Plan Pipeline**

**Date Created:** 2026-01-06
**Status:** Planning Complete - Ready for Implementation
**Goal:** Create an end-to-end system that generates floor plan layouts from natural language prompts

---

## Table of Contents
1. [Executive Summary](#executive-summary)
2. [Current System Architecture](#current-system-architecture)
3. [Integration Strategy](#integration-strategy)
4. [Technical Implementation Plan](#technical-implementation-plan)
5. [File Locations & Key References](#file-locations--key-references)
6. [Next Steps](#next-steps)

---

## Executive Summary

### What We're Building

An integrated pipeline that takes natural language descriptions (or form input) and generates complete floor plan layouts:

```
Text Prompt or Form Input
    ↓
[GLiNER System] - Extracts rooms & predicts adjacency
    ↓
Graph Structure (nodes + edges)
    ↓
[Graph2Plan Model] - Generates geometric layout
    ↓
Floor Plan Image with room positions
```

### Why This Integration Makes Sense

1. **GLiNER provides the graph** - Graph2Plan needs graph input, GLiNER creates it from text/forms
2. **Complementary systems** - GLiNER has no layout generation, Graph2Plan has no prompt interface
3. **Edge predictions add value** - GLiNER's adjacency predictions tell Graph2Plan which rooms should be neighbors
4. **Complete workflow** - Users go from idea to visual floor plan in one flow

### Key Decision: Direct Generation (Approach B)

We will **bypass Graph2Plan's retrieval database** and send GLiNER graphs directly to the Graph2Plan neural network.

**Rationale:**
- Graph2Plan's database only has 10 floor plans (too small for effective retrieval)
- GLiNER's edge predictions provide spatial information that retrieval doesn't use
- More flexible - works for any room combination
- The Graph2Plan model is trained and can generate from scratch

---

## Current System Architecture

### GLiNER System
**Location:** `/mnt/c/Users/hmbashir/source/GLINER/`

**Capabilities:**
1. **Sentence Input Tab** - Parses natural language using GLiNER NER model
2. **Form Input Tab** - Direct room type + area input via dropdown/fields
3. **Edge Prediction** - Trained model predicts room adjacency relationships
4. **Graph Output** - NetworkX graphs saved as `.gpickle` files

**Key Files:**
- `graph_ui.py` - Main GUI (622 lines, now with form input)
- `run_gliner.py` - Sentence parser (outputs `[{"type": "bedroom", "size": 5}, ...]`)
- `JSON_to_Node.py` - Builds NetworkX graphs from room list
- `predict_edges_semantic_only_f5_save_graphs.py` - Edge prediction model
- Edge model checkpoint: `semantic_edge_best_residential.pt`
- Vocab file: `room_type_vocab.json`

**Current Output Format:**
```python
# NetworkX Graph (.gpickle)
G.nodes(data=True) = [
    ('bedroom_1', {'type': 'bedroom', 'size': 15, 'coordinates': [x, y, 0.0]}),
    ('kitchen_1', {'type': 'kitchen', 'size': 10, 'coordinates': [x, y, 0.0]})
]
G.edges() = [
    ('bedroom_1', 'kitchen_1'),
    ('bedroom_1', 'living room_1')
]
```

### Graph2Plan System
**Location:** `/mnt/c/Users/hmbashir/source/Graph2plan/Interface/`

**Capabilities:**
1. **Django Web Interface** - D3.js graph editor
2. **Graph2Plan Neural Network** - GNN + CNN architecture
3. **Layout Generation** - Converts graph → room bounding boxes
4. **Boundary Alignment** - Fits layout to building outline

**Key Files:**
- `Houseweb/views.py` - Backend logic (NumSearch, TransGraph, AdjustGraph endpoints)
- `model/model.py` - Graph2Plan neural network architecture
- `model/test.py` - Inference functions
- `model/floorplan.py` - Floor plan data structure
- `model/model.pth` - Trained weights (30MB)
- `model/utils.py` - Vocabulary, room type mappings

**Current Input Format (AdjustGraph endpoint expects):**
```python
{
    'graph': {
        'nodes': [
            {'id': 0, 'type': 'living_room'},
            {'id': 1, 'type': 'bedroom'}
        ],
        'edges': [
            {'source': 0, 'target': 1}
        ]
    },
    'boundary_data': boundary_image  # 128x128x3 image
}
```

**Model Architecture:**
- Input: Graph (nodes + edges) + Boundary image (128x128)
- GNN: 5 layers of graph convolution
- CNN: Encodes boundary constraints
- Output: Bounding boxes [x, y, width, height] for each room

---

## Integration Strategy

### Chosen Approach: Direct Generation

**Flow:**
```
User Input (Text/Form)
    ↓
GLiNER GUI (existing)
    ↓
Graph Generation (existing)
    ↓
Edge Prediction (existing)
    ↓
Format Conversion (NEW - GLiNER → Graph2Plan format)
    ↓
Boundary Selection (NEW - UI or default)
    ↓
Graph2Plan AdjustGraph Endpoint (existing)
    ↓
Floor Plan Visualization (existing Graph2Plan UI)
```

### Why Not Retrieval?

Graph2Plan has a **NumSearch** endpoint that retrieves similar floor plans from a database, but:

1. **Database is tiny** - Only 8 training + 2 test floor plans
2. **Limited room combinations** - Won't have all possible room type combinations
3. **Doesn't use edge information** - GLiNER's adjacency predictions would be wasted
4. **Adds complexity** - Extra step that doesn't add value given small database

The retrieval was designed as a UI convenience for manual graph building, not for programmatic generation.

---

## Technical Implementation Plan

### Phase 1: Format Conversion (Critical Bridge)

**Task:** Convert GLiNER's NetworkX graph to Graph2Plan's expected format

**Input (GLiNER):**
```python
# NetworkX graph with:
- Nodes: {'type': 'bedroom', 'size': 15, 'coordinates': [x, y, z]}
- Edges: List of tuples (node1, node2)
```

**Output (Graph2Plan expects):**
```python
{
    'nodes': [
        {'id': 0, 'type': 'living_room'},  # Note: room type naming
        {'id': 1, 'type': 'bedroom'}
    ],
    'edges': [
        {'source': 0, 'target': 1}  # Indices, not names
    ]
}
```

**Key Challenges:**
1. **Room type vocabulary mapping**
   - GLiNER uses: "bedroom", "kitchen", "living room" (with spaces)
   - Graph2Plan might use: "bedroom", "kitchen", "living_room" (underscores)
   - Need to check `Graph2plan/Interface/model/utils.py` for exact vocabulary

2. **Node ID assignment**
   - GLiNER: Nodes have string IDs like "bedroom_1"
   - Graph2Plan: Expects integer IDs (0, 1, 2, ...)
   - Need to create mapping: node_name → integer_id

3. **Edge format**
   - GLiNER: `[('bedroom_1', 'kitchen_1'), ...]`
   - Graph2Plan: `[{'source': 0, 'target': 1}, ...]`
   - Convert node names to IDs

**Implementation Location:**
Create new file: `/mnt/c/Users/hmbashir/source/GLINER/gliner_to_graph2plan.py`

**Pseudocode:**
```python
def convert_gliner_to_graph2plan(networkx_graph):
    """
    Convert GLiNER NetworkX graph to Graph2Plan format

    Args:
        networkx_graph: NetworkX Graph with nodes/edges from GLiNER

    Returns:
        dict: Graph2Plan compatible format
    """
    # 1. Load Graph2Plan vocabulary
    vocab = load_graph2plan_vocab()  # From Graph2plan/Interface/model/utils.py

    # 2. Create node list with integer IDs
    nodes = []
    node_name_to_id = {}

    for idx, (node_name, data) in enumerate(networkx_graph.nodes(data=True)):
        room_type = data['type']

        # Map GLiNER room type to Graph2Plan vocab
        g2p_type = map_room_type(room_type, vocab)

        nodes.append({
            'id': idx,
            'type': g2p_type
        })
        node_name_to_id[node_name] = idx

    # 3. Create edge list with integer IDs
    edges = []
    for (u, v) in networkx_graph.edges():
        edges.append({
            'source': node_name_to_id[u],
            'target': node_name_to_id[v]
        })

    return {
        'nodes': nodes,
        'edges': edges
    }

def map_room_type(gliner_type, graph2plan_vocab):
    """
    Map GLiNER room types to Graph2Plan vocabulary

    Examples:
        "living room" → "living_room"
        "bedroom" → "bedroom"
        "maid's room" → "maid_room" (or closest match)
    """
    # Replace spaces with underscores
    normalized = gliner_type.replace(" ", "_").replace("'", "")

    # Check if exists in Graph2Plan vocab
    if normalized in graph2plan_vocab:
        return normalized

    # Handle special cases or find closest match
    # TODO: Define mapping for edge cases

    return normalized
```

### Phase 2: Boundary Handling

**Options:**

**Option A: Use Test Boundaries (Simplest)**
- Graph2Plan has 2 pre-defined test boundaries
- Just let user select which one (dropdown in GUI)
- No additional complexity

**Option B: Generate Default Boundary (Automatic)**
- Create a simple rectangular boundary based on total area
- Calculate: `total_area = sum(room['size'] for room in rooms)`
- Generate 128x128 image with rectangular outline
- Pros: Fully automatic, no user input needed
- Cons: Less realistic than actual building outlines

**Option C: Custom Boundary Upload (Advanced)**
- Allow users to upload/draw boundary shapes
- Most flexible but complex implementation
- Could be future enhancement

**Recommendation:** Start with **Option A** (use test boundaries) for MVP, add Option B later.

### Phase 3: API Integration

**Two Integration Approaches:**

#### Approach 3A: Django REST API (Clean Separation)

**Pros:**
- Clean separation of concerns
- Can call from any client (GLiNER GUI, web, mobile)
- Easier to debug and test

**Cons:**
- Requires running Django server
- Network overhead (minimal for local)

**Implementation:**
1. Start Graph2Plan Django server: `python manage.py runserver`
2. From GLiNER, make HTTP POST to `http://localhost:8000/AdjustGraph/`
3. Receive floor plan data back

**GLiNER Code Addition:**
```python
import requests
import json

def generate_floor_plan_via_api(gliner_graph, boundary_id=0):
    """
    Send graph to Graph2Plan API and get floor plan
    """
    # Convert format
    g2p_graph = convert_gliner_to_graph2plan(gliner_graph)

    # Prepare request
    payload = {
        'graph': json.dumps(g2p_graph),
        'boundary_id': boundary_id  # 0 or 1 for test boundaries
    }

    # Call API
    response = requests.post(
        'http://localhost:8000/AdjustGraph/',
        data=payload
    )

    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Graph2Plan API error: {response.text}")
```

#### Approach 3B: Direct Python Import (Tight Integration)

**Pros:**
- No server needed
- Faster (no network)
- Simpler deployment

**Cons:**
- Requires compatible Python environments
- GLiNER and Graph2Plan dependencies must coexist

**Implementation:**
```python
import sys
sys.path.append('/mnt/c/Users/hmbashir/source/Graph2plan/Interface')

from model.test import generate_layout  # Import Graph2Plan functions
from model.model import Model  # Import model class

def generate_floor_plan_direct(gliner_graph, boundary_id=0):
    """
    Call Graph2Plan model directly (no server)
    """
    # Convert format
    g2p_graph = convert_gliner_to_graph2plan(gliner_graph)

    # Load model
    model = Model()
    model.load_state_dict(torch.load('.../model.pth'))

    # Load boundary
    boundary = load_test_boundary(boundary_id)

    # Generate layout
    floor_plan = generate_layout(model, g2p_graph, boundary)

    return floor_plan
```

**Recommendation:** Start with **Approach 3A (API)** because:
- Graph2Plan already has Django setup
- Easier to debug
- Environments stay separate
- Can add direct integration later if needed

### Phase 4: UI Integration

**Add to GLiNER GUI (`graph_ui.py`):**

1. **Boundary Selection**
   ```python
   # Add after "Generate Graph from Form" button
   boundary_frame = tk.Frame(form_tab)
   boundary_frame.pack(pady=5)

   tk.Label(boundary_frame, text="Building Boundary:").pack(side="left")
   boundary_var = tk.IntVar(value=0)
   ttk.Radiobutton(boundary_frame, text="Test 1", variable=boundary_var, value=0).pack(side="left")
   ttk.Radiobutton(boundary_frame, text="Test 2", variable=boundary_var, value=1).pack(side="left")
   ```

2. **Generate Floor Plan Button**
   ```python
   def generate_complete_floor_plan():
       """
       Full pipeline: Current graph → Graph2Plan → Floor plan
       """
       global current_graph

       if current_graph is None:
           messagebox.showerror("Error", "Generate a graph first")
           return

       try:
           status_var.set("Generating floor plan layout...")

           # Get boundary selection
           boundary_id = boundary_var.get()

           # Call Graph2Plan
           floor_plan = generate_floor_plan_via_api(current_graph, boundary_id)

           # Display result (could show in new window or save image)
           display_floor_plan(floor_plan)

           status_var.set("Floor plan generated successfully!")

       except Exception as e:
           messagebox.showerror("Error", f"Floor plan generation failed: {str(e)}")
           status_var.set("Floor plan generation failed")

   # Add button
   tk.Button(shared_btn_frame, text="Generate Floor Plan",
             font=("Arial", 11), command=generate_complete_floor_plan,
             bg="#FF9800", fg="white").pack(side="left", padx=6)
   ```

3. **Display Result**
   ```python
   def display_floor_plan(floor_plan_data):
       """
       Show generated floor plan

       Options:
       - Open in new Tkinter window
       - Save as image and open with default viewer
       - Embed in GUI using PIL/ImageTk
       """
       # TODO: Implement based on floor_plan_data format
       pass
   ```

---

## File Locations & Key References

### GLiNER System
```
/mnt/c/Users/hmbashir/source/GLINER/
├── graph_ui.py                           # Main GUI (MODIFIED - has form input + scrolling)
├── run_gliner.py                         # Sentence parser
├── JSON_to_Node.py                       # Graph builder
├── predict_edges_semantic_only_f5_save_graphs.py  # Edge prediction
├── semantic_edge_best_residential.pt     # Edge model weights
├── room_type_vocab.json                  # Edge model vocabulary
└── gliner_to_graph2plan.py              # NEW - Format converter (to create)
```

**Room Types Supported (18 total):**
```python
labels = [
    "bedroom","kitchen","toilet","living room","sitting room","lobby","bathroom","hallway",
    "garage","foyer","dining room","office","study","laundry room","maid's room","guest room",
    "dressing room","pantry"
]
```

### Graph2Plan System
```
/mnt/c/Users/hmbashir/source/Graph2plan/
├── Interface/
│   ├── manage.py                         # Django entry point
│   ├── Houseweb/
│   │   ├── views.py                      # Backend (NumSearch, AdjustGraph endpoints)
│   │   └── urls.py                       # URL routing
│   ├── model/
│   │   ├── model.py                      # Graph2Plan neural network
│   │   ├── test.py                       # Inference functions
│   │   ├── floorplan.py                  # Floor plan data structure
│   │   ├── utils.py                      # Vocabulary (CHECK THIS for room types)
│   │   └── model.pth                     # Trained weights (30MB)
│   ├── static/Data/
│   │   ├── data_test_converted.pkl       # Test boundaries (2 samples)
│   │   └── data_train_converted.pkl      # Training data (8 samples)
│   └── templates/
│       └── home.html                     # Web interface
└── SessionContext/
    └── GLINER_GRAPH2PLAN_INTEGRATION_PLAN.md  # This file
```

**Critical Files to Examine:**
1. `Graph2plan/Interface/model/utils.py` - Room type vocabulary
2. `Graph2plan/Interface/Houseweb/views.py` - AdjustGraph endpoint implementation
3. `Graph2plan/Interface/model/test.py` - Inference API

---

## Next Steps

### Immediate Actions (Session Start)

1. **Verify Room Type Compatibility**
   ```bash
   # Read Graph2Plan vocabulary
   cd /mnt/c/Users/hmbashir/source/Graph2plan/Interface
   grep -r "room_types\|vocabulary\|ROOM_TYPES" model/utils.py model/floorplan.py
   ```

   Compare with GLiNER's 18 room types. Create mapping for mismatches.

2. **Test Graph2Plan Server**
   ```bash
   cd /mnt/c/Users/hmbashir/source/Graph2plan/Interface
   python manage.py runserver
   # Visit http://localhost:8000 to verify it works
   ```

3. **Create Format Converter**
   - Implement `gliner_to_graph2plan.py`
   - Test with sample GLiNER graph
   - Verify output matches Graph2Plan format

4. **Test End-to-End**
   - Generate graph in GLiNER (use form input)
   - Convert format
   - Call Graph2Plan API manually (curl/Postman)
   - Verify floor plan is generated

### Implementation Checklist

- [ ] Examine `Graph2plan/Interface/model/utils.py` for room vocabulary
- [ ] Create room type mapping (GLiNER ↔ Graph2Plan)
- [ ] Implement `gliner_to_graph2plan.py` converter
- [ ] Test converter with sample graphs
- [ ] Start Graph2Plan Django server
- [ ] Test AdjustGraph endpoint with converted graphs
- [ ] Add boundary selection UI to GLiNER
- [ ] Add "Generate Floor Plan" button to GLiNER
- [ ] Implement API call from GLiNER → Graph2Plan
- [ ] Implement floor plan result display
- [ ] Test complete pipeline end-to-end
- [ ] Handle error cases (invalid graphs, model failures)
- [ ] Add loading indicators/progress feedback
- [ ] Document user workflow

### Testing Strategy

1. **Unit Tests**
   - Test format converter with known graphs
   - Test room type mapping edge cases

2. **Integration Tests**
   - GLiNER graph → Converter → Graph2Plan format (verify JSON structure)
   - API call → Graph2Plan → Response (verify communication)

3. **End-to-End Tests**
   - Simple: "One bedroom 10 sqm, one kitchen 8 sqm"
   - Medium: "Two bedrooms 15 sqm each, living room 20 sqm, kitchen 10 sqm, bathroom 5 sqm"
   - Complex: All 18 room types with various sizes

4. **Edge Cases**
   - Empty graph (no rooms)
   - Single room
   - No edges (disconnected graph)
   - Rooms with no size information
   - Unknown room types

### Known Questions to Resolve

1. **Room Type Vocabulary**
   - Does Graph2Plan use underscores or spaces?
   - Are all 18 GLiNER room types supported?
   - How to handle unsupported types?

2. **Boundary Format**
   - Exact format of test boundaries in `data_test_converted.pkl`
   - How to load them?
   - Image dimensions and channels?

3. **Response Format**
   - What does AdjustGraph return?
   - Is it an image, JSON, or .mat file?
   - How to display it in Tkinter?

4. **Size Information**
   - Does Graph2Plan use the 'size' attribute from GLiNER?
   - Or does it infer sizes from the model?
   - Should we include size in the conversion?

### Future Enhancements

1. **Custom Boundaries**
   - Allow users to upload/draw building outlines
   - Auto-generate boundaries from total area

2. **Layout Iteration**
   - Generate multiple layout options
   - Let user select preferred one

3. **Interactive Refinement**
   - After generation, allow room repositioning
   - Re-run model with constraints

4. **Export Options**
   - Save as PNG, SVG, PDF
   - Export measurements/dimensions
   - Generate AutoCAD files

5. **Batch Processing**
   - Process multiple prompts at once
   - Compare different layouts

---

## Technical Notes

### Environment Requirements

**GLiNER Environment:**
- Python 3.x
- tkinter (GUI)
- NetworkX (graphs)
- PyTorch (edge prediction model)
- GLiNER library

**Graph2Plan Environment:**
- Python 3.13
- Django 5.2.7
- PyTorch (deep learning)
- NetworkX
- Optional: MATLAB Engine

**Potential Conflicts:**
- Both use PyTorch (should be compatible)
- Both use NetworkX (should be compatible)
- May need separate virtual environments if Django version conflicts

### Performance Considerations

1. **Model Loading**
   - Graph2Plan model is 30MB
   - Consider loading once and keeping in memory
   - Or start Django server once and keep running

2. **Generation Time**
   - Graph2Plan inference: ~1-5 seconds per layout
   - Add loading indicator in GUI

3. **Memory**
   - Both models in memory: ~100-200MB
   - Should be fine for modern systems

### Debugging Tips

1. **Check Graph Format**
   ```python
   # Print before conversion
   print("GLiNER nodes:", list(G.nodes(data=True)))
   print("GLiNER edges:", list(G.edges()))

   # Print after conversion
   print("Graph2Plan format:", json.dumps(converted, indent=2))
   ```

2. **Test Graph2Plan Independently**
   - Use web interface to create test graph
   - Inspect network requests in browser DevTools
   - See exact format being sent to AdjustGraph

3. **Gradual Integration**
   - Don't try to integrate everything at once
   - Test each component separately first
   - Combine only after each works

---

## Success Criteria

The integration is complete when a user can:

1. Enter a prompt: "Two bedrooms 15 sqm each, one living room 20 sqm, one kitchen 10 sqm, one bathroom 5 sqm"
2. Click "Generate Graph from Form" (or use sentence input)
3. See the graph with predicted edges in GLiNER
4. Select a building boundary (Test 1 or Test 2)
5. Click "Generate Floor Plan"
6. See a complete floor plan layout with rooms positioned and sized appropriately
7. Save or export the result

**Bonus:** The layout should respect the adjacency edges predicted by GLiNER (e.g., if kitchen-living room has an edge, they should be placed adjacent in the floor plan).

---

## Questions for Future Sessions

Before starting implementation, answer:

1. Have you verified Graph2Plan's room type vocabulary?
2. Have you successfully run Graph2Plan's web interface?
3. Have you examined the AdjustGraph endpoint code?
4. Do you have the Graph2Plan model weights (model.pth)?
5. Are both systems currently working independently?

---

## Contact & Resources

**GLiNER Implementation:**
- Session: 2026-01-06
- Form input + scrolling completed
- All code tested and working

**Graph2Plan Documentation:**
- Implementation guide: `Graph2plan/Interface/IMPLEMENTATION_GUIDE.md`
- Original paper: Graph2Plan research (2020)

**Next Session:**
- Start with "Immediate Actions" checklist
- Focus on format conversion first
- Test incrementally

---

**Document Version:** 1.0
**Last Updated:** 2026-01-06
**Status:** Ready for Implementation

# Edge Prediction Guide - Beginner Level

## What is Edge Prediction?

### The Problem

When you use Auto-Adjust to add missing rooms to your floor plan, the new rooms appear as **isolated nodes** with no connections (edges) to other rooms. You need to predict which rooms should be connected to these new nodes.

**Example**:

```text
Before Auto-Adjust:
  Kitchen ---- LivingRoom ---- Bedroom
                    |
                Bathroom

After Auto-Adjust (adds missing DiningRoom):
  Kitchen ---- LivingRoom ---- Bedroom
                    |
                Bathroom

  DiningRoom (isolated - no connections!)
```

**Goal**: Predict that DiningRoom should connect to Kitchen and LivingRoom:

```text
After Edge Prediction:
  Kitchen ---- LivingRoom ---- Bedroom
     |             |
  DiningRoom   Bathroom
```

---

## Why Do We Need Edge Prediction?

### Current Workflow Problem

1. User filters: "2 bedrooms, 2 bathrooms, 1 kitchen, 1 dining room"
2. System finds floor plan with: 2 bedrooms, 2 bathrooms, 1 kitchen (no dining room)
3. User clicks Transfer → floor plan loads
4. User clicks Auto-Adjust → dining room node added randomly
5. **Problem**: Dining room has no connections - it just floats there!

### What Edges Represent in Floor Plans

In architectural floor plans, an **edge** between two rooms means:

- The rooms are **adjacent** (share a wall or doorway)
- You can walk directly from one room to the other
- They're part of the same connected floor plan layout

**Not having edges** means:

- Rooms are disconnected/isolated
- Can't be used for layout generation
- Doesn't represent a realistic floor plan

---

## What is a Graph Neural Network (GNN)?

### Simple Explanation

Imagine you have a social network:

- **Nodes** = People
- **Edges** = Friendships
- You want to predict: "Who should be friends with whom?"

A GNN learns patterns like:

- People with similar interests tend to be friends
- People with mutual friends tend to become friends
- People in the same location tend to be friends

In our case:

- **Nodes** = Rooms (kitchen, bedroom, bathroom, etc.)
- **Edges** = Adjacencies (which rooms share walls)
- We want to predict: "Which rooms should be adjacent?"

The GNN learns patterns like:

- Kitchens are usually adjacent to dining rooms
- Bathrooms are rarely adjacent to living rooms
- Bedrooms cluster together

### How GNNs Work (Simplified)

1. **Start**: Each node has features (room type, size, position)
2. **Message Passing**: Nodes "talk" to their neighbors, sharing information
3. **Aggregation**: Each node combines messages from all neighbors
4. **Repeat**: Do this multiple times to spread information across the graph
5. **Prediction**: Use the final node states to predict missing edges

---

## Data You Already Have

### ResPlan Dataset

The ResPlan dataset contains ~75,000 floor plans with:

- **Nodes**: Room types and positions
- **Edges**: Which rooms are adjacent
- **Boundaries**: Outer wall shapes

**Location**: `C:\Users\hmbashir\source\Graph2plan\Interface\static\Data\`

**Files**:

```text
data_train_converted.pkl  - Training floor plans (nodes, edges, boundaries)
data_train_eNum.pkl       - Edge structures as vectors
rNum_train.npy            - Room counts as vectors
trainTF.pkl               - Turn Functions (boundary shapes)
```

### What's Already Loaded in Your Code

In `views.py`, these are already loaded:

```python
train_data        # Floor plan objects with .data.box, .data.edge, .data.boundary
train_data_rNum   # Room count vectors [num_floorplans, 14]
train_data_eNum   # Edge structure vectors [num_floorplans, edge_dim]
trainNameList     # Floor plan IDs
```

---

## What You Need for Edge Prediction

### Input Features (What the Model Needs)

For each node in your graph:

1. **Room Type**: Kitchen, Bedroom, Bathroom, etc. (categorical)
2. **Position**: (x, y) coordinates in the floor plan
3. **Existing Connections**: Which nodes already have edges
4. **Boundary Information**: Outer wall shape (from Turn Function)

For the entire graph:
5. **Graph Structure**: Current adjacency matrix
6. **Global Features**: Total rooms, boundary area, etc.

### Output (What the Model Predicts)

For each **pair of nodes** (i, j):

- **Probability of edge**: Should room i be connected to room j?
- Value between 0 (definitely not connected) and 1 (definitely connected)

**Threshold**: If probability > 0.5, add the edge

### Ground Truth (Training Labels)

Use the existing edges from ResPlan dataset:

- **Positive examples**: Pairs of rooms that ARE connected in real floor plans
- **Negative examples**: Pairs of rooms that are NOT connected

---

## Step-by-Step Learning Plan

### Phase 1: Understand the Data (1-2 days)

**Goal**: Load and visualize ResPlan data

**Tasks**:

1. Load a few floor plans from `train_data`
2. Print the node types (room types)
3. Print the edges (adjacencies)
4. Visualize one floor plan as a graph
5. Count how many floor plans have each room type

**Code Starting Point**:

```python
import pickle
train_data = pickle.load(open('static/Data/data_train_converted.pkl', 'rb'))
floor_plan = train_data['data'][0]  # First floor plan
print("Rooms:", floor_plan.get_rooms())
print("Edges:", floor_plan.get_triples())
```

### Phase 2: Prepare Training Data (2-3 days)

**Goal**: Convert floor plans into GNN-ready format

**What You Need**:

- **Node features matrix**: [num_nodes, feature_dim]
  - One-hot encode room types (14 types → 14 features)
  - Normalize x, y positions to [0, 1]

- **Edge index**: [2, num_edges]
  - Format: [[source_nodes], [target_nodes]]
  - Example: [[0, 1, 2], [1, 2, 0]] means edges 0→1, 1→2, 2→0

- **Edge labels**: [num_possible_edges]
  - For all pairs (i, j): label = 1 if edge exists, 0 otherwise
  - Total pairs = num_nodes × (num_nodes - 1) / 2

**Libraries to Use**:

- PyTorch Geometric (PyG) - makes GNN implementation easy
- NetworkX - for graph manipulation
- NumPy - for array operations

**Install**:

```bash
pip install torch-geometric networkx
```

### Phase 3: Build Simple GNN Model (3-5 days)

**Goal**: Create a basic edge prediction model

**Architecture** (start simple):

```text
Input: Node features [N, feature_dim]
       Edge index [2, E]

Layer 1: Graph Convolution (hidden_dim=64)
         ↓ (aggregate neighbor information)
         ReLU activation

Layer 2: Graph Convolution (hidden_dim=64)
         ↓
         ReLU activation

Layer 3: Graph Convolution (output_dim=32)
         ↓

Edge Prediction:
  For each pair (i, j):
    Concatenate: [node_i_embedding, node_j_embedding]
    ↓
    Linear layer → Sigmoid
    ↓
    Probability of edge
```

**PyTorch Geometric Example**:

```python
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

class EdgePredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, 32)

        # Edge prediction head
        self.edge_mlp = nn.Sequential(
            nn.Linear(32 * 2, 64),  # Concatenate two node embeddings
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x, edge_index):
        # Node embedding
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        x = self.conv3(x, edge_index)

        return x

    def predict_edges(self, node_embeddings, node_pairs):
        # Get embeddings for source and target nodes
        src_emb = node_embeddings[node_pairs[0]]
        dst_emb = node_embeddings[node_pairs[1]]

        # Concatenate and predict
        edge_features = torch.cat([src_emb, dst_emb], dim=-1)
        return self.edge_mlp(edge_features)
```

### Phase 4: Train the Model (2-3 days)

**Goal**: Train on ResPlan dataset and evaluate

**Training Loop**:

```python
for epoch in range(num_epochs):
    for floor_plan in train_data:
        # 1. Load graph
        node_features = get_node_features(floor_plan)
        edge_index = get_edge_index(floor_plan)

        # 2. Generate training pairs
        positive_pairs = get_existing_edges(floor_plan)  # Real edges
        negative_pairs = sample_non_edges(floor_plan)   # Non-existent edges

        # 3. Forward pass
        embeddings = model(node_features, edge_index)
        predictions = model.predict_edges(embeddings, all_pairs)

        # 4. Compute loss
        loss = binary_cross_entropy(predictions, labels)

        # 5. Backward pass
        loss.backward()
        optimizer.step()
```

**Evaluation Metrics**:

- **Accuracy**: % of correct predictions
- **Precision**: Of predicted edges, how many are correct?
- **Recall**: Of real edges, how many did we find?
- **F1 Score**: Balance of precision and recall

**Target Performance**:

- Start: Aim for 70%+ accuracy
- Good: 80%+ accuracy
- Excellent: 90%+ accuracy

### Phase 5: Integrate with Graph2plan (3-4 days)

**Goal**: Use trained model in AutoAdjustGraph

**Current Code** (`views.py:1407-1409`):

```python
# Note: New nodes are added without edges
# Edges will be added later through edge prediction model
# (Previously we auto-connected to nearest neighbor, but that's been removed)
```

**Replace with**:

```python
# Add edges using edge prediction model
if len(rooms_to_add) > 0:
    # Prepare graph for model
    node_features = prepare_node_features(newNode)
    edge_index = prepare_edge_index(newEdge)

    # Get node embeddings
    with torch.no_grad():
        embeddings = edge_model(node_features, edge_index)

    # For each new node
    for new_node_idx in new_node_indices:
        # Get all possible connections
        possible_targets = [i for i in range(len(newNode)) if i != new_node_idx]

        # Predict edge probabilities
        pairs = torch.tensor([[new_node_idx] * len(possible_targets),
                              possible_targets])
        probs = edge_model.predict_edges(embeddings, pairs)

        # Add edges where probability > threshold
        for target, prob in zip(possible_targets, probs):
            if prob > 0.5:
                newEdge.append([new_node_idx, target])
                print(f"Added edge {new_node_idx} -> {target} (confidence: {prob:.2f})")
```

**Integration Steps**:

1. Save trained model: `torch.save(model.state_dict(), 'edge_predictor.pth')`
2. Load in `views.py`: `model.load_state_dict(torch.load('edge_predictor.pth'))`
3. Add model as global variable in `views.py`
4. Call model in `AutoAdjustGraph` after adding new nodes

---

## Understanding Graph Convolution

### What is Convolution?

In images, convolution means:

- Look at a pixel and its neighbors
- Apply a filter to combine them
- Move the filter across the image

In graphs, convolution means:

- Look at a node and its neighbors
- Combine their features
- Do this for all nodes

### Graph Convolution Example

**Initial State**:

```text
Kitchen (features: [1, 0, 0, 100, 50])  ← room type, x, y
   |
LivingRoom (features: [0, 1, 0, 150, 75])
   |
Bedroom (features: [0, 0, 1, 200, 100])
```

**After One Convolution Layer**:

```text
Kitchen:
  - Takes its own features
  - Takes LivingRoom's features (its neighbor)
  - Combines them: new_features = W1 * kitchen + W2 * livingroom
  - Now knows about adjacent room

LivingRoom:
  - Takes kitchen, bedroom, and own features
  - Combines: new_features = W1 * kitchen + W2 * livingroom + W3 * bedroom
  - Now knows about both neighbors
```

**After Multiple Layers**:

- Kitchen knows about LivingRoom's neighbors (Bedroom)
- Information spreads across entire graph
- Each node has "context" of the whole floor plan

---

## Key Concepts to Learn

### 1. Message Passing

Nodes send information to neighbors:

```text
message = W * neighbor_features
aggregate = sum(all_messages)
new_features = old_features + aggregate
```

### 2. Node Embeddings

A **learned representation** of each node:

- Input: Room type, position (sparse, high-dimensional)
- Output: Dense vector (e.g., 32 dimensions) capturing room's "essence"
- Similar rooms get similar embeddings

### 3. Graph Pooling

Combine information from all nodes into one vector:

- Sum pooling: `global_features = sum(all_node_features)`
- Mean pooling: `global_features = mean(all_node_features)`
- Max pooling: `global_features = max(all_node_features)`

Use for predicting graph-level properties (e.g., total floor area)

### 4. Edge Prediction Strategies

#### Option 1: Pairwise Classification

- For each pair (i, j), predict edge independently
- Fast inference, but ignores dependencies

#### Option 2: Autoregressive

- Predict edges one at a time
- Each prediction conditions on previous edges
- Slower but more coherent

#### Option 3: Graph-to-Graph

- Predict entire adjacency matrix at once
- Ensures global consistency
- More complex training

**Recommendation**: Start with Option 1 (pairwise)

---

## Common Pitfalls & Solutions

### Pitfall 1: Class Imbalance

**Problem**: Most node pairs are NOT connected

- Positive examples (edges exist): ~5% of all pairs
- Negative examples (no edge): ~95% of all pairs
- Model learns to always predict "no edge"

**Solution**: Balanced sampling

```python
num_positive = len(real_edges)
num_negative = num_positive  # Sample equal number

negative_edges = random.sample(all_non_edges, num_negative)
training_pairs = positive_edges + negative_edges
```

### Pitfall 2: Over-smoothing

**Problem**: After many GNN layers, all nodes have same features

- Information spreads too much
- Nodes lose individual identity

**Solution**: Use 2-3 layers max, add skip connections

```python
x_residual = x
x = conv1(x, edge_index)
x = x + x_residual  # Skip connection
```

### Pitfall 3: Ignoring Spatial Information

**Problem**: GNN only uses graph structure, ignores room positions

- Distant rooms might get connected
- Violates physical constraints

**Solution**: Add position as features, use spatial attention

```python
# Include distance in edge weight
distances = compute_distances(node_positions)
edge_weights = 1.0 / (distances + 1e-6)
x = conv(x, edge_index, edge_weights)
```

### Pitfall 4: Train-Test Leakage

**Problem**: Testing on floor plans seen during training

**Solution**: Split dataset by floor plans, not by edges

```python
train_plans = train_data[:60000]  # First 60k floor plans
val_plans = train_data[60000:67500]  # Next 7.5k
test_plans = train_data[67500:]  # Last 7.5k
```

---

## Evaluation: How to Know If It Works

### Quantitative Metrics

1. **Accuracy**: Overall correct predictions
   - Target: >80%

2. **Precision**: Of predicted edges, % that are correct
   - Target: >85% (avoid false connections)

3. **Recall**: Of real edges, % that we found
   - Target: >75% (find most important connections)

4. **F1 Score**: Harmonic mean of precision and recall
   - Target: >80%

### Qualitative Evaluation

1. **Visual Inspection**: Does the graph look reasonable?
   - All nodes connected (no isolated rooms)?
   - Sensible adjacencies (kitchen near dining room)?
   - No weird connections (bedroom to garage)?

2. **Architect Review**: Show to domain expert
   - Would this floor plan work in real life?
   - Do room arrangements make sense?

3. **Comparison to Heuristics**: Better than simple rules?
   - Nearest neighbor: Connect to closest room
   - Room type rules: Always connect kitchen to dining room
   - Your GNN should outperform these

---

## Resources for Learning

### Tutorials

1. **PyTorch Geometric Tutorial**
   - <https://pytorch-geometric.readthedocs.io/>
   - Start with "Introduction by Example"
   - Do the "Node Classification" tutorial first

2. **Stanford CS224W: Machine Learning with Graphs**
   - <http://web.stanford.edu/class/cs224w/>
   - Free lectures on YouTube
   - Covers GNN fundamentals

3. **Distill.pub GNN Article**
   - <https://distill.pub/2021/gnn-intro/>
   - Visual, interactive explanations

### Papers (Beginner-Friendly)

1. **Graph Convolutional Networks (GCN)**
   - Kipf & Welling, 2017
   - Foundation of modern GNNs

2. **GraphSAGE**
   - Hamilton et al., 2017
   - How to handle large graphs

3. **Link Prediction Survey**
   - Zhang & Chen, 2018
   - Overview of edge prediction methods

---

## Example: Complete Minimal Pipeline

### Step 1: Load One Floor Plan

```python
import pickle
import numpy as np

# Load data
train_data = pickle.load(open('static/Data/data_train_converted.pkl', 'rb'))
fp = train_data['data'][0]

# Extract information
rooms = fp.get_rooms(tensor=False)  # [N] room types
edges = fp.get_triples(tensor=False)  # [E, 3] edges
boxes = fp.data.box  # [N, 5] room bounding boxes

print(f"Rooms: {len(rooms)}")
print(f"Edges: {len(edges)}")
```

### Step 2: Create Node Features

```python
# One-hot encode room types
num_room_types = 14
node_features = np.zeros((len(rooms), num_room_types + 2))

for i, room_type in enumerate(rooms):
    node_features[i, int(room_type)] = 1  # One-hot

    # Add position (normalized)
    x1, y1, x2, y2, _ = boxes[i]
    center_x = (x1 + x2) / 2 / 256  # Normalize to [0, 1]
    center_y = (y1 + y2) / 2 / 256

    node_features[i, -2] = center_x
    node_features[i, -1] = center_y

print(f"Node features shape: {node_features.shape}")
```

### Step 3: Create Edge Index

```python
# Convert to PyTorch Geometric format
edge_index = []
for edge in edges:
    src, edge_type, dst = edge
    edge_index.append([int(src), int(dst)])
    edge_index.append([int(dst), int(src)])  # Undirected

edge_index = np.array(edge_index).T  # [2, num_edges]
print(f"Edge index shape: {edge_index.shape}")
```

### Step 4: Create Labels for All Pairs

```python
num_nodes = len(rooms)

# Get all possible pairs
all_pairs = []
labels = []

for i in range(num_nodes):
    for j in range(i + 1, num_nodes):
        all_pairs.append([i, j])

        # Check if edge exists
        edge_exists = any((e[0] == i and e[2] == j) or
                         (e[0] == j and e[2] == i)
                         for e in edges)
        labels.append(1 if edge_exists else 0)

print(f"Total pairs: {len(all_pairs)}")
print(f"Positive pairs: {sum(labels)}")
print(f"Negative pairs: {len(labels) - sum(labels)}")
```

### Step 5: Train Simple Model

```python
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

# Convert to PyTorch
x = torch.FloatTensor(node_features)
edge_idx = torch.LongTensor(edge_index)
pair_idx = torch.LongTensor(all_pairs).T
y = torch.FloatTensor(labels)

# Create graph
graph = Data(x=x, edge_index=edge_idx)

# Define model (simplified)
class SimpleGNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(16, 32)
        self.conv2 = GCNConv(32, 16)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

# Train (pseudocode - needs loss, optimizer, etc.)
model = SimpleGNN()
embeddings = model(graph)

# Predict edges
src_emb = embeddings[pair_idx[0]]
dst_emb = embeddings[pair_idx[1]]
edge_score = (src_emb * dst_emb).sum(dim=-1).sigmoid()

print(f"Predictions shape: {edge_score.shape}")
```

---

## Timeline Estimate

| Phase | Duration | Effort Level |
| ------- | ---------- | ------------- |
| Data exploration | 1-2 days | Easy |
| Data preparation | 2-3 days | Medium |
| Model building | 3-5 days | Medium-Hard |
| Training & tuning | 2-3 days | Medium |
| Integration | 3-4 days | Medium |
| **Total** | **11-17 days** | **Full-time** |

**Part-time**: 3-5 weeks (2-3 hours per day)

---

## Success Criteria

You'll know edge prediction is working when:

1. ✅ Model trains without errors
2. ✅ Validation accuracy >80%
3. ✅ Predicted edges look sensible in visualization
4. ✅ New nodes get connected appropriately in AutoAdjustGraph
5. ✅ Generated floor plans pass visual inspection
6. ✅ Users can create complete, connected floor plans

---

## Getting Help

### When Stuck

1. **Check PyTorch Geometric docs**: Most common issues covered
2. **Stack Overflow**: Tag with `pytorch-geometric` and `graph-neural-networks`
3. **Reddit r/MachineLearning**: Weekly beginner thread
4. **PyTorch Geometric Slack**: Active community

### Debugging Tips

1. **Print shapes**: Always print tensor shapes at each step
2. **Visualize**: Plot a few floor plans with predicted edges
3. **Start simple**: Use tiny dataset (10 floor plans) for debugging
4. **Sanity checks**: Can model overfit one floor plan? (Should reach 100% accuracy)

---

## Next Steps

1. **Read PyTorch Geometric tutorials** (1 day)
2. **Explore ResPlan data** (1 day)
3. **Implement data pipeline** (2 days)
4. **Build simple GNN** (3 days)
5. **Come back for help with specific issues!**

Good luck! Edge prediction is a challenging but rewarding problem. Take it one step at a time, and don't hesitate to ask questions when stuck.

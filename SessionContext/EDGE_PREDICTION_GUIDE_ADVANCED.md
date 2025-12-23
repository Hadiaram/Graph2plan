# Edge Prediction Guide - Advanced Level (GIN/GINE Experience)

## Executive Summary

You need to predict edges for newly added nodes in partially observed floor plan graphs. Given your experience with GIN (Graph Isomorphism Network) and GINE (GIN with Edge features), you're well-positioned to tackle this. The key challenges are:

1. **Inductive learning**: Predicting on graphs with different sizes than training
2. **Partial observability**: Some edges exist, others need prediction
3. **Spatial constraints**: Adjacency must respect 2D layout geometry
4. **Architectural semantics**: Room type relationships matter (kitchen-dining adjacency preferred)

This document focuses on architecture decisions, training strategies, and integration with the existing Graph2plan pipeline.

---

## Problem Formulation

### Task Definition

**Input**:

- Partially observed graph `G = (V_obs ∪ V_new, E_obs)`
- Node features: `X ∈ ℝ^{|V| × d}` (room type, position)
- Existing edges: `E_obs ⊆ V_obs × V_obs`
- Boundary: `B ∈ ℝ^{1000}` (Turn Function representation)

**Output**:

- Edge predictions: `Ê ⊆ V × V` where `V = V_obs ∪ V_new`
- Specifically need: `E_new = {(u,v) | u ∈ V_new ∨ v ∈ V_new}`

**Objective**:

```text
Ê* = argmax_{Ê} P(Ê | G_partial, X, B)
```

### Why This is Different from Your Previous Work

Assuming your GIN/GINE experience was on standard benchmarks (molecular graphs, social networks), this problem has unique constraints:

1. **Planar graphs**: Floor plans are 2D embeddings, edges can't cross
2. **Degree constraints**: Nodes have physical size limits (max ~5-6 connections)
3. **Hierarchical structure**: Room clusters (bedroom wing, living area)
4. **Symmetry**: Many floor plans have architectural symmetry
5. **Boundary constraints**: Edges must respect outer walls

---

## Architecture Considerations

### Option 1: GIN-Based Encoder + Pairwise Decoder

**Architecture**:

```text
Encoder:
  x^(0) = X (node features)
  for layer ℓ = 1 to L:
    m_i^(ℓ) = Σ_{j∈N(i)} MLP_ℓ(x_j^(ℓ-1))
    x_i^(ℓ) = MLP_ℓ'((1 + ε) · x_i^(ℓ-1) + m_i^(ℓ))

Decoder (for each pair (i,j)):
  z_ij = [x_i^(L) || x_j^(L) || |x_i^(L) - x_j^(L)| || x_i^(L) ⊙ x_j^(L)]
  p(e_ij = 1) = σ(MLP_decoder(z_ij))
```

**Pros**:

- Permutation invariant
- Scalable (O(|V|²) for inference)
- Easy to implement

**Cons**:

- Ignores edge dependencies (predicts independently)
- No constraint enforcement (might predict crossing edges)

---

### Option 2: GINE + Variational Graph Auto-Encoder (VGAE)

**Architecture**:

```text
Encoder (GINE-based):
  for layer ℓ:
    m_ij^(ℓ) = MLP_ℓ([x_i^(ℓ-1) || x_j^(ℓ-1) || e_ij])  # Edge features
    x_i^(ℓ) = (1 + ε) · x_i^(ℓ-1) + Σ_{j∈N(i)} m_ij^(ℓ)

Variational Layer:
  μ = MLP_μ(x^(L))
  log σ² = MLP_σ(x^(L))
  z ~ N(μ, σ²)  # Latent node embeddings

Decoder:
  P(A | Z) = Π_{i<j} σ(z_i^T z_j)  # Inner product decoder
```

**Pros**:

- Learns latent graph structure
- Probabilistic (can sample multiple graphs)
- Good for uncertainty estimation

**Cons**:

- Inner product decoder is restrictive
- Training instability (KL collapse)
- Harder to incorporate spatial constraints

---

### Option 3: GIN + Attention-Based Edge Selection

**Architecture**:

```text
Node Encoder (GIN):
  h_i^(L) = GIN_encoder(X, E_obs)

Edge Attention (for new nodes only):
  For each v_new ∈ V_new:
    α_ij = softmax_j(MLP([h_i || h_j || spatial_features(i,j)]))
    candidates = top_k(α, k=5)  # Select top-k most likely connections

Edge Refinement:
  For each candidate edge (i,j):
    context = GIN_refine(G ∪ {(i,j)}, X)  # Test adding edge
    score = MLP_score(context)
    if score > threshold: add (i,j) to Ê
```

**Pros**:

- Can enforce degree constraints (top-k)
- Context-aware (considers effect of adding edge)
- Interpretable attention weights

**Cons**:

- Sequential (slower inference)
- Requires careful tuning of k

---

### Recommendation: Hybrid GIN + Spatial Attention

Given your experience and the domain constraints, I recommend:

**Base**: GIN encoder (you know it well)
**Addition**: Spatial attention with geometric constraints
**Decoder**: Pairwise classification with planarity penalty

```python
class SpatialGIN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super().__init__()
        self.gin_layers = nn.ModuleList([
            GINConv(nn.Sequential(
                nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )) for i in range(num_layers)
        ])

        self.spatial_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            batch_first=True
        )

        self.edge_decoder = nn.Sequential(
            nn.Linear(hidden_dim * 4 + 3, 128),  # 4 features + 3 spatial
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x, edge_index, positions):
        # GIN encoding
        for layer in self.gin_layers:
            x = layer(x, edge_index)
            x = F.relu(x)

        # Spatial attention
        x_attn, _ = self.spatial_attention(x, x, x)
        x = x + x_attn  # Residual

        return x

    def predict_edges(self, h, node_pairs, positions):
        src_h = h[node_pairs[0]]
        dst_h = h[node_pairs[1]]

        # Spatial features
        src_pos = positions[node_pairs[0]]
        dst_pos = positions[node_pairs[1]]
        dist = torch.norm(src_pos - dst_pos, dim=-1, keepdim=True)
        angle = torch.atan2(dst_pos[:, 1] - src_pos[:, 1],
                           dst_pos[:, 0] - src_pos[:, 0]).unsqueeze(-1)

        # Concatenate features
        edge_features = torch.cat([
            src_h,
            dst_h,
            torch.abs(src_h - dst_h),
            src_h * dst_h,
            dist,
            angle,
            (dist < 0.2).float()  # Proximity indicator
        ], dim=-1)

        return torch.sigmoid(self.edge_decoder(edge_features))
```

---

## Training Strategy

### Dataset Construction from ResPlan

**You have**: 75k floor plans with ground truth adjacency

**Split Strategy**:

```python
# Don't split by edges - split by floor plans!
train_fps = floor_plans[:60000]  # 80%
val_fps = floor_plans[60000:67500]  # 10%
test_fps = floor_plans[67500:]  # 10%
```

**Negative Sampling**: Critical for balance

```python
def sample_negative_edges(graph, num_nodes, ratio=1.0):
    """Sample non-existent edges as negative examples"""
    pos_edges = set(map(tuple, graph.edge_index.T.tolist()))
    num_pos = len(pos_edges)
    num_neg = int(num_pos * ratio)

    neg_edges = []
    while len(neg_edges) < num_neg:
        i, j = random.randint(0, num_nodes-1), random.randint(0, num_nodes-1)
        if i != j and (i,j) not in pos_edges and (j,i) not in pos_edges:
            # Spatial constraint: don't sample edges between very distant nodes
            if euclidean_distance(positions[i], positions[j]) < max_distance:
                neg_edges.append((i, j))

    return neg_edges
```

**Why spatial constraint in negative sampling?**

- Without it: Model learns "if nodes are far, no edge" (trivial)
- With it: Model must learn architectural semantics (room type relationships)

---

### Loss Function Design

**Baseline**: Binary Cross-Entropy

```python
loss_bce = F.binary_cross_entropy_with_logits(pred, target, pos_weight=pos_weight)
```

**Enhanced**: BCE + Planarity Penalty + Connectivity Regularization

```python
def compute_loss(pred_probs, target, edge_index, positions):
    # 1. BCE with class imbalance handling
    pos_weight = (target == 0).sum() / (target == 1).sum()
    loss_bce = F.binary_cross_entropy(pred_probs, target, pos_weight=pos_weight)

    # 2. Planarity penalty (penalize crossing edges)
    pred_edges = (pred_probs > 0.5).nonzero()
    loss_planarity = compute_crossing_penalty(pred_edges, positions)

    # 3. Connectivity regularization (ensure graph is connected)
    loss_connectivity = -torch.log(compute_connectivity_score(pred_edges))

    # 4. Degree regularization (avoid hubs)
    degrees = torch_geometric.utils.degree(pred_edges[0])
    loss_degree = ((degrees - target_avg_degree) ** 2).mean()

    return loss_bce + 0.1 * loss_planarity + 0.05 * loss_connectivity + 0.02 * loss_degree
```

**Planarity Penalty** (simple version):

```python
def compute_crossing_penalty(edge_index, positions):
    """Penalize edges that geometrically intersect"""
    penalty = 0
    for i in range(edge_index.size(1)):
        for j in range(i+1, edge_index.size(1)):
            e1 = edge_index[:, i]  # (u1, v1)
            e2 = edge_index[:, j]  # (u2, v2)

            if segments_intersect(positions[e1[0]], positions[e1[1]],
                                 positions[e2[0]], positions[e2[1]]):
                penalty += 1

    return penalty
```

---

### Inductive Learning Challenge

**Problem**: Training graphs have variable sizes (5-15 nodes), test graphs at inference time might have different sizes.

**GIN handles this well** (you know this), but be careful about:

1. **Positional embeddings**: Don't use absolute position encoding

```python
# Bad: Learnable positional embeddings
pos_emb = nn.Embedding(max_nodes, hidden_dim)

# Good: Relative spatial features
spatial_features = compute_relative_positions(positions)
```

1. **Global pooling**: Use permutation-invariant aggregation

```python
# Global graph representation
graph_emb = global_mean_pool(node_emb, batch)
```

1. **Batch construction**: Use PyG's DataLoader

```python
from torch_geometric.loader import DataLoader

loader = DataLoader(dataset, batch_size=32, shuffle=True)
# Automatically handles variable-size graphs
```

---

## Data Preparation Pipeline

### From ResPlan to PyTorch Geometric

```python
import torch
from torch_geometric.data import Data, DataLoader
import numpy as np

def floorplan_to_pyg(fp, room_label_to_idx):
    """Convert ResPlan floor plan to PyG Data object"""

    # 1. Node features
    rooms = fp.get_rooms(tensor=False)
    boxes = fp.data.box

    num_nodes = len(rooms)
    node_features = []

    for i in range(num_nodes):
        # Room type (one-hot)
        room_type = int(rooms[i])
        room_onehot = np.zeros(14)
        room_onehot[room_type] = 1

        # Position (normalized)
        x1, y1, x2, y2, _ = boxes[i]
        center_x = (x1 + x2) / 2 / 256.0
        center_y = (y1 + y2) / 2 / 256.0

        # Size (normalized area)
        area = (x2 - x1) * (y2 - y1) / (256 * 256)

        features = np.concatenate([room_onehot, [center_x, center_y, area]])
        node_features.append(features)

    x = torch.FloatTensor(node_features)

    # 2. Edge index
    edges = fp.get_triples(tensor=False)
    edge_list = []
    for edge in edges:
        src, _, dst = edge
        edge_list.append([int(src), int(dst)])
        edge_list.append([int(dst), int(src)])  # Undirected

    edge_index = torch.LongTensor(edge_list).T if edge_list else torch.zeros((2, 0), dtype=torch.long)

    # 3. Positions (for spatial constraints)
    positions = torch.FloatTensor([[x[14], x[15]] for x in node_features])

    # 4. Boundary features (optional, for global context)
    boundary = torch.FloatTensor(fp.data.boundary)

    return Data(x=x, edge_index=edge_index, pos=positions, boundary=boundary)

# Process all floor plans
dataset = [floorplan_to_pyg(fp, room_label_to_idx) for fp in train_data['data']]
```

### Data Augmentation

**Augmentations that preserve graph structure**:

1. **Rotation**: Rotate floor plan by 90°, 180°, 270°

```python
def rotate_graph(data, angle):
    theta = angle * np.pi / 180
    rot_matrix = torch.FloatTensor([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])
    data.pos = data.pos @ rot_matrix.T
    return data
```

1. **Horizontal/Vertical flip**: Mirror floor plan

```python
def flip_graph(data, axis='horizontal'):
    if axis == 'horizontal':
        data.pos[:, 0] = 1 - data.pos[:, 0]
    else:
        data.pos[:, 1] = 1 - data.pos[:, 1]
    return data
```

1. **Node dropout**: Remove random nodes during training

```python
def node_dropout(data, p=0.1):
    mask = torch.rand(data.num_nodes) > p
    return data.subgraph(mask)
```

**Don't use**: Random edge addition/removal (changes ground truth)

---

## Evaluation Metrics

### Beyond Accuracy

Given your experience, you know accuracy alone isn't enough. Use:

1. **Link Prediction Metrics**:

   ```python
   from sklearn.metrics import roc_auc_score, average_precision_score

   # AUC-ROC: Discrimination ability
   auc_roc = roc_auc_score(y_true, y_pred_probs)

   # Average Precision: Precision-recall trade-off
   avg_precision = average_precision_score(y_true, y_pred_probs)
   ```

2. **Graph-Level Metrics**:

   ```python
   # Is predicted graph connected?
   is_connected = torch_geometric.utils.is_undirected(edge_index) and \
                  torch_geometric.utils.contains_self_loops(edge_index) == False

   # Average node degree
   degrees = torch_geometric.utils.degree(edge_index[0])
   avg_degree = degrees.float().mean()

   # Graph edit distance to ground truth
   ged = graph_edit_distance(pred_graph, true_graph)
   ```

3. **Architectural Validity** (domain-specific):

   ```python
   def architectural_validity(pred_edges, room_types, positions):
       """Check if predicted graph follows architectural rules"""

       # Rule 1: Kitchen-Dining adjacency preferred
       kitchen_idx = (room_types == KITCHEN).nonzero()
       dining_idx = (room_types == DINING).nonzero()
       if len(kitchen_idx) > 0 and len(dining_idx) > 0:
           has_kitchen_dining_edge = check_edge_exists(pred_edges, kitchen_idx[0], dining_idx[0])
           score_1 = 1.0 if has_kitchen_dining_edge else 0.0
       else:
           score_1 = 1.0

       # Rule 2: Bedrooms cluster together
       bedroom_indices = (room_types == BEDROOM).nonzero()
       if len(bedroom_indices) > 1:
           bedroom_subgraph = subgraph(pred_edges, bedroom_indices)
           score_2 = is_connected(bedroom_subgraph).float()
       else:
           score_2 = 1.0

       # Rule 3: No crossing edges
       score_3 = 1.0 - count_crossings(pred_edges, positions) / max_possible_crossings

       return (score_1 + score_2 + score_3) / 3.0
   ```

---

## Inference Strategy for AutoAdjustGraph

### Challenge: Incremental Edge Prediction

When AutoAdjustGraph adds N_new nodes:

- Existing graph: G_obs = (V_obs, E_obs)
- New nodes: V_new (with no edges)
- Need to predict: E_new = edges involving V_new

### Option 1: One-Shot Prediction

```python
def predict_edges_oneshot(model, existing_graph, new_nodes):
    """Predict all edges at once"""

    # Combine existing and new nodes
    all_nodes = torch.cat([existing_graph.x, new_nodes])
    all_positions = torch.cat([existing_graph.pos, new_node_positions])

    # Get embeddings
    with torch.no_grad():
        h = model(all_nodes, existing_graph.edge_index)

    # Generate candidate pairs (only involving new nodes)
    candidates = []
    for new_idx in range(len(existing_graph.x), len(all_nodes)):
        for other_idx in range(len(all_nodes)):
            if new_idx != other_idx:
                candidates.append([new_idx, other_idx])

    if not candidates:
        return []

    candidate_tensor = torch.LongTensor(candidates).T

    # Predict
    probs = model.predict_edges(h, candidate_tensor, all_positions)

    # Threshold
    threshold = 0.5
    predicted_edges = candidate_tensor[:, probs.squeeze() > threshold]

    return predicted_edges.T.tolist()
```

### Option 2: Autoregressive Prediction (Better)

```python
def predict_edges_autoregressive(model, existing_graph, new_nodes, max_edges_per_node=5):
    """Predict edges one at a time, conditioning on previous predictions"""

    current_graph = existing_graph.clone()
    predicted_edges = []

    for new_node_idx in range(len(existing_graph.x), len(existing_graph.x) + len(new_nodes)):
        # Add new node to graph
        current_graph.x = torch.cat([current_graph.x, new_nodes[new_node_idx - len(existing_graph.x)].unsqueeze(0)])

        # Get embeddings with current graph structure
        with torch.no_grad():
            h = model(current_graph.x, current_graph.edge_index)

        # Generate candidates for this new node
        candidates = [[new_node_idx, other_idx] for other_idx in range(len(current_graph.x)) if other_idx != new_node_idx]

        if not candidates:
            continue

        candidate_tensor = torch.LongTensor(candidates).T

        # Predict probabilities
        probs = model.predict_edges(h, candidate_tensor, current_graph.pos).squeeze()

        # Select top-k edges
        top_k_indices = torch.topk(probs, min(max_edges_per_node, len(probs))).indices

        # Add selected edges to graph
        for idx in top_k_indices:
            if probs[idx] > 0.5:  # Threshold
                edge = candidates[idx]
                predicted_edges.append(edge)

                # Update graph for next iteration
                new_edge = torch.LongTensor([[edge[0], edge[1]], [edge[1], edge[0]]]).T
                current_graph.edge_index = torch.cat([current_graph.edge_index, new_edge], dim=1)

    return predicted_edges
```

**Recommendation**: Use autoregressive for better architectural coherence, even though it's slower.

---

## Integration with Graph2plan

### Where to Add Edge Prediction

**Current code** (`views.py:1407-1409`):

```python
# Note: New nodes are added without edges
# Edges will be added later through edge prediction model
```

**Modified**:

```python
# Add edges using edge prediction model
if len(rooms_to_add) > 0:
    # Load model (do this once at startup)
    if not hasattr(AutoAdjustGraph, 'edge_model'):
        AutoAdjustGraph.edge_model = load_edge_predictor_model()

    # Prepare graph for model
    existing_nodes = torch.FloatTensor([
        get_node_features(node) for node in newNode[:-len(rooms_to_add)]
    ])
    existing_edges = torch.LongTensor([[u, v] for u, v in newEdge]).T

    new_node_features = torch.FloatTensor([
        get_node_features(node) for node in newNode[-len(rooms_to_add):]
    ])

    # Predict edges
    predicted_edges = predict_edges_autoregressive(
        model=AutoAdjustGraph.edge_model,
        existing_graph=PyGData(x=existing_nodes, edge_index=existing_edges),
        new_nodes=new_node_features
    )

    # Add predicted edges to newEdge
    for u, v in predicted_edges:
        newEdge.append([int(u), int(v)])
        print(f"Predicted edge: {newNode[u][1]} <-> {newNode[v][1]}")
```

### Model Loading

**Option 1**: Load at Django startup

```python
# In views.py, at module level
edge_prediction_model = None

def load_edge_predictor():
    global edge_prediction_model
    if edge_prediction_model is None:
        model = SpatialGIN(input_dim=17, hidden_dim=64, num_layers=3)
        model.load_state_dict(torch.load('static/models/edge_predictor.pth', map_location='cpu'))
        model.eval()
        edge_prediction_model = model
    return edge_prediction_model
```

**Option 2**: Lazy loading (first call)

```python
def get_edge_predictor():
    if not hasattr(get_edge_predictor, 'model'):
        get_edge_predictor.model = load_model_from_disk()
    return get_edge_predictor.model
```

---

## Comparison: GIN vs GINE for This Task

### GIN Pros

- ✅ You know it well
- ✅ Theoretically as powerful as WL test
- ✅ Fewer parameters (faster training)
- ✅ Works well for node-level tasks

### GIN Cons

- ❌ Doesn't use edge features (room relationships)
- ❌ Limited expressiveness for certain graph structures

### GINE Pros

- ✅ Can encode edge types (doorway, wall, window)
- ✅ More expressive for multigraphs
- ✅ Better for heterogeneous relationships

### GINE Cons

- ❌ Requires edge features (may not have ground truth)
- ❌ More parameters (risk of overfitting on 75k samples)

### Recommendation

**Start with GIN**, add edge features later if needed:

```python
# GIN baseline
model = GIN(input_dim=17, hidden_dim=64, num_layers=3)

# If performance plateaus, try GINE with edge features:
# - Edge feature 1: Euclidean distance between rooms
# - Edge feature 2: Relative angle (North, South, East, West)
# - Edge feature 3: Estimated wall length

model_enhanced = GINE(input_dim=17, edge_dim=3, hidden_dim=64, num_layers=3)
```

---

## Advanced Techniques to Try

### 1. Graph Attention Networks (GAT)

Add attention to weight neighbor contributions:

```python
from torch_geometric.nn import GATConv

class GINWithAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.gin1 = GINConv(...)
        self.gat = GATConv(hidden_dim, hidden_dim, heads=4)
        self.gin2 = GINConv(...)

    def forward(self, x, edge_index):
        x = self.gin1(x, edge_index)
        x = self.gat(x, edge_index)  # Attention layer
        x = self.gin2(x, edge_index)
        return x
```

**When to use**: If you find certain room types need more attention (e.g., bathrooms are hard to place correctly).

---

### 2. Contrastive Learning

Learn embeddings such that similar floor plans are close in latent space:

```python
def contrastive_loss(h1, h2, temperature=0.5):
    """InfoNCE loss for graph pairs"""
    h1 = F.normalize(h1, dim=-1)
    h2 = F.normalize(h2, dim=-1)

    similarity = torch.mm(h1, h2.T) / temperature
    labels = torch.arange(h1.size(0)).to(h1.device)

    loss = F.cross_entropy(similarity, labels)
    return loss

# Training loop
for graph1, graph2 in pairs:  # graph2 is augmented version of graph1
    h1 = model(graph1.x, graph1.edge_index)
    h2 = model(graph2.x, graph2.edge_index)

    h1_global = global_mean_pool(h1, graph1.batch)
    h2_global = global_mean_pool(h2, graph2.batch)

    loss_contrastive = contrastive_loss(h1_global, h2_global)
```

**When to use**: As pre-training, then fine-tune on edge prediction.

---

### 3. Graph Diffusion Models

Generate edges through iterative denoising:

```python
def forward_diffusion(edge_index, t, noise_schedule):
    """Add noise to edge_index at timestep t"""
    noise = torch.rand_like(edge_index.float())
    noisy_edge_index = edge_index * (1 - noise_schedule[t]) + noise * noise_schedule[t]
    return noisy_edge_index

def reverse_diffusion(model, noisy_edges, t):
    """Denoise edges using model"""
    pred_noise = model(noisy_edges, t)
    denoised = noisy_edges - pred_noise
    return denoised
```

**When to use**: If you want to generate multiple plausible edge configurations (probabilistic output).

---

## Debugging & Troubleshooting

### Problem: Model predicts no edges (all 0s)

**Cause**: Class imbalance, model learned to predict negative class

**Solution**:

1. Check positive weight: `pos_weight = neg_count / pos_count`
2. Balance batch: Equal positive and negative samples
3. Try focal loss:

```python
def focal_loss(pred, target, alpha=0.25, gamma=2.0):
    bce = F.binary_cross_entropy(pred, target, reduction='none')
    pt = torch.exp(-bce)
    focal = alpha * (1 - pt) ** gamma * bce
    return focal.mean()
```

---

### Problem: Model predicts too many edges (dense graph)

**Cause**: Threshold too low, or model biased toward positive class

**Solution**:

1. Tune threshold on validation set:

```python
thresholds = np.linspace(0.1, 0.9, 50)
best_f1 = 0
best_threshold = 0.5

for thresh in thresholds:
    preds = (probs > thresh).float()
    f1 = f1_score(y_true, preds)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = thresh
```

1. Add degree regularization to loss (see earlier)

---

### Problem: Predicted edges cross each other

**Cause**: Model ignores spatial layout

**Solution**:

1. Add planarity penalty to loss
2. Use spatial features in decoder:

```python
# Check if edge would intersect existing edges
def is_planar(new_edge, existing_edges, positions):
    for exist_edge in existing_edges:
        if segments_intersect(positions[new_edge[0]], positions[new_edge[1]],
                            positions[exist_edge[0]], positions[exist_edge[1]]):
            return False
    return True

# During inference, only add planar edges
for candidate in candidates:
    if probs[candidate] > thresh and is_planar(candidate, predicted_edges, positions):
        predicted_edges.append(candidate)
```

---

### Problem: Isolated nodes remain after prediction

**Cause**: Model doesn't ensure connectivity

**Solution**:

1. Post-processing: Connect isolated nodes to nearest neighbor

```python
def connect_isolated_nodes(edge_index, num_nodes, positions):
    degrees = torch_geometric.utils.degree(edge_index[0], num_nodes=num_nodes)
    isolated = (degrees == 0).nonzero().squeeze()

    for node in isolated:
        # Find nearest connected node
        distances = torch.norm(positions[node] - positions, dim=-1)
        distances[node] = float('inf')  # Exclude self
        nearest = distances.argmin()

        # Add edge
        new_edge = torch.LongTensor([[node, nearest], [nearest, node]]).T
        edge_index = torch.cat([edge_index, new_edge], dim=1)

    return edge_index
```

---

## Performance Benchmarks

### Expected Results (on ResPlan test set)

| Metric | Random Baseline | Heuristic (Nearest Neighbor) | Simple GNN | Your Target |
| -------- | ---------------- | ------------------------------ | ------------ | ------------- |
| Accuracy | 50% | 65% | 78% | **85%+** |
| Precision | 5% | 55% | 82% | **90%+** |
| Recall | 50% | 60% | 75% | **80%+** |
| F1 Score | 9% | 57% | 78% | **85%+** |
| AUC-ROC | 50% | 70% | 85% | **90%+** |

### Training Time Estimates

| Setup | Time per Epoch | Total Training | GPU |
| ------- | --------------- | ---------------- | ----- |
| GIN (3 layers, 64 dim) | 5 min | 2-3 hours (30 epochs) | GTX 1080 Ti |
| GINE (3 layers, 64 dim) | 8 min | 4-5 hours | GTX 1080 Ti |
| GIN + Attention | 10 min | 5-6 hours | GTX 1080 Ti |

## Based on 60k training graphs, batch size 32

---

## Next Steps

### Week 1-2: Implementation

1. ✅ Set up PyTorch Geometric environment
2. ✅ Convert ResPlan data to PyG format
3. ✅ Implement GIN baseline
4. ✅ Train on subset (1000 graphs) for debugging

### Week 3-4: Experimentation

1. ✅ Train on full dataset
2. ✅ Tune hyperparameters (learning rate, hidden dim, num layers)
3. ✅ Add spatial attention
4. ✅ Evaluate on test set

### Week 5: Integration

1. ✅ Export model to production format
2. ✅ Integrate with AutoAdjustGraph
3. ✅ Test end-to-end workflow
4. ✅ Benchmark inference time

---

## Code Repository Structure (Recommended)

```text
Graph2plan/
├── edge_prediction/
│   ├── models/
│   │   ├── gin.py          # GIN implementation
│   │   ├── gine.py         # GINE implementation
│   │   └── spatial_gin.py  # Recommended hybrid model
│   ├── data/
│   │   ├── dataset.py      # PyG dataset class
│   │   ├── transforms.py   # Data augmentation
│   │   └── utils.py        # Data loading helpers
│   ├── training/
│   │   ├── train.py        # Training loop
│   │   ├── evaluate.py     # Evaluation metrics
│   │   └── losses.py       # Custom loss functions
│   ├── inference/
│   │   ├── predict.py      # Edge prediction inference
│   │   └── postprocess.py  # Planarity checks, etc.
│   └── experiments/
│       ├── config.yaml     # Hyperparameters
│       ├── run_baseline.py # Train baseline models
│       └── analysis.ipynb  # Results visualization
```

---

## References

Your prior work likely covered these, but for completeness:

### Foundational Papers

1. **GIN**: Xu et al. "How Powerful are Graph Neural Networks?" (ICLR 2019)
2. **GINE**: Hu et al. "Strategies for Pre-training Graph Neural Networks" (ICLR 2020)
3. **Link Prediction Survey**: Zhang & Chen "Link Prediction Based on Graph Neural Networks" (NeurIPS 2018)

### Relevant to This Domain

1. **Graph2Plan (this paper)**: Hu et al. "Graph2Plan: Learning Floorplan Generation from Layout Graphs" (SIGGRAPH 2020)
2. **Structured Set Prediction**: Zhang et al. "Neural Graph Matching Networks for Fewshot 3D Action Recognition" (ECCV 2018)

### Advanced Techniques

1. **Spatial GNNs**: Veličković et al. "Neural Execution of Graph Algorithms" (ICLR 2020)
2. **Graph Diffusion**: Hoogeboom et al. "Equivariant Diffusion for Molecule Generation" (ICML 2022)

---

## Final Thoughts

Given your GIN/GINE experience, you're well-equipped to tackle this. The key differences from standard benchmarks are:

1. **Spatial constraints matter** - add them explicitly
2. **Class imbalance is severe** - handle carefully in sampling
3. **Architectural semantics** - room type relationships are important
4. **Inference is incremental** - autoregressive approach recommended

Start with a GIN baseline (you know it works), add spatial features, tune carefully, then integrate. Expect 2-3 weeks for a working solution.

Good luck! Feel free to continue this in a new session when you're ready to start implementation.

"""
test_edge_model_resplan.py
Test the trained edge prediction model on ResPlan data
"""

import os
import json
import pickle
import torch
import torch.nn as nn
import numpy as np
from torch_geometric.data import Data
from load_resplan_graphs import load_resplan_from_converted_pkl


# =========================
# Model Definition (same as training)
# =========================
class TypeCompatModel(nn.Module):
    """
    Type-only edge prediction model.
    Learns P(edge | type_u, type_v) using room type embeddings.
    """
    def __init__(self, num_types, emb_dim=64, hidden=128, dropout=0.2):
        super().__init__()
        self.emb = nn.Embedding(num_types, emb_dim)
        self.ln = nn.LayerNorm(4 * emb_dim)
        self.edge_mlp = nn.Sequential(
            nn.Linear(4 * emb_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1)
        )

    @staticmethod
    def _norm_pairs(edge_pairs: torch.Tensor) -> torch.Tensor:
        ep = edge_pairs
        if isinstance(ep, (list, tuple)): ep = torch.as_tensor(ep)
        ep = ep.squeeze()
        if ep.dim()==2 and ep.size(0)==2: return ep.long()
        if ep.dim()==2 and ep.size(1)==2: return ep.t().long()
        if ep.dim()==1:
            assert ep.numel()%2==0, "Odd length for edge_pairs"
            return ep.view(2, -1).long()
        axes = [i for i in range(ep.dim()) if ep.size(i)==2]
        if axes: return ep.movedim(axes[0],0).reshape(2,-1).long()
        assert ep.numel()%2==0, "Odd numel"
        return ep.reshape(2,-1).long()

    def forward(self, type_ids: torch.Tensor, edge_pairs: torch.Tensor):
        # type_ids: [N] (long), edge_pairs: [2, K]
        pairs = self._norm_pairs(edge_pairs)
        if pairs.numel() == 0:
            return torch.empty(0, dtype=torch.float, device=type_ids.device)

        u, v = pairs  # [K]
        eu, ev = self.emb(type_ids[u]), self.emb(type_ids[v])  # [K, D]
        feats = torch.cat([eu, ev, torch.abs(eu-ev), eu*ev], dim=1)  # [K, 4D]
        feats = self.ln(feats)
        return self.edge_mlp(feats).squeeze(-1)  # logits


# =========================
# Testing Functions
# =========================
def predict_edges_for_graph(model, graph, threshold_logit, device, type_to_id):
    """
    Predict edges for a single floor plan graph.

    Args:
        model: Trained TypeCompatModel
        graph: PyG Data object with node features (x)
        threshold_logit: Logit threshold for classification
        device: torch device
        type_to_id: Room type vocabulary

    Returns:
        predicted_edges: List of (src, dst) tuples for predicted edges
        scores: Edge scores for all node pairs
    """
    model.eval()

    with torch.no_grad():
        # Extract room types from node features
        # Features are [type_id, cx, cy, 0.0]
        type_ids = graph.x[:, 0].long().to(device)
        num_nodes = graph.num_nodes

        # Generate all possible edges (excluding self-loops)
        all_edges = []
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                all_edges.append((i, j))

        if len(all_edges) == 0:
            return [], {}

        # Predict for all pairs
        # Convert all_edges to edge_index format [2, K]
        edge_pairs = torch.tensor(all_edges, dtype=torch.long, device=device).t()

        # Call model with full type_ids array and edge indices
        logits = model(type_ids, edge_pairs)

        # Apply threshold
        predicted_edges = []
        scores = {}

        for idx, (i, j) in enumerate(all_edges):
            logit = logits[idx].item()
            scores[(i, j)] = logit

            if logit >= threshold_logit:
                predicted_edges.append((i, j))
                predicted_edges.append((j, i))  # Undirected

        return predicted_edges, scores


def evaluate_test_set(model, test_graphs, threshold_logit, device, type_to_id):
    """
    Evaluate model on test set using BALANCED negative sampling (same as threshold tuning).

    Returns:
        metrics: Dict with precision, recall, F1
    """
    from torch_geometric.utils import negative_sampling
    import numpy as np

    model.eval()
    y_true = []
    y_pred = []

    # Use fixed random seed for reproducible negatives
    rng = np.random.RandomState(42)

    with torch.no_grad():
        for graph in test_graphs:
            graph = graph.to(device)
            type_ids = graph.x[:, 0].long()

            # Positive edges
            pos_edges = graph.edge_index
            num_pos = pos_edges.size(1)

            # Sample SAME NUMBER of negative edges (balanced evaluation)
            neg_edges = negative_sampling(
                edge_index=pos_edges,
                num_nodes=graph.num_nodes,
                num_neg_samples=num_pos,
                method='sparse'
            )

            # Shuffle negatives with fixed seed for reproducibility
            perm = rng.permutation(neg_edges.size(1))
            neg_edges = neg_edges[:, perm]

            # Get predictions
            pos_logits = model(type_ids, pos_edges)
            neg_logits = model(type_ids, neg_edges)

            # Apply threshold
            y_true += [1] * num_pos + [0] * num_pos
            y_pred += (pos_logits >= threshold_logit).int().tolist() + \
                      (neg_logits >= threshold_logit).int().tolist()

    # Compute metrics
    from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    tn = cm[0, 0] if cm.shape == (2, 2) else 0
    fp = cm[0, 1] if cm.shape == (2, 2) else 0
    fn = cm[1, 0] if cm.shape == (2, 2) else 0
    tp = cm[1, 1] if cm.shape == (2, 2) else 0

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': int(tp),
        'fp': int(fp),
        'fn': int(fn),
        'tn': int(tn)
    }


def visualize_predictions(graph, predicted_edges, gt_edges, type_to_id):
    """
    Print a comparison of predicted vs ground truth edges.

    Args:
        graph: PyG Data object
        predicted_edges: List of (src, dst) predicted edge tuples
        gt_edges: List of (src, dst) ground truth edge tuples
        type_to_id: Room type vocabulary
    """
    # Get inverse mapping
    id_to_type = {v: k for k, v in type_to_id.items()}

    # Get room types for each node
    node_types = []
    for i in range(graph.num_nodes):
        type_id = int(graph.x[i, 0].item())
        node_types.append(id_to_type.get(type_id, f"unknown_{type_id}"))

    # Convert to sets for comparison
    pred_set = set()
    for src, dst in predicted_edges:
        if src < dst:
            pred_set.add((src, dst))

    gt_set = set()
    for src, dst in gt_edges:
        if src < dst:
            gt_set.add((src, dst))

    # True positives, false positives, false negatives
    tp = pred_set & gt_set
    fp = pred_set - gt_set
    fn = gt_set - pred_set

    print("\n" + "="*60)
    print(f"Graph with {graph.num_nodes} nodes")
    print(f"Node types: {node_types}")
    print("="*60)

    print(f"\n✅ True Positives ({len(tp)}):")
    for src, dst in sorted(tp):
        print(f"  {src} ({node_types[src]}) <-> {dst} ({node_types[dst]})")

    print(f"\n❌ False Positives ({len(fp)}):")
    for src, dst in sorted(fp):
        print(f"  {src} ({node_types[src]}) <-> {dst} ({node_types[dst]})")

    print(f"\n⚠️  False Negatives ({len(fn)}):")
    for src, dst in sorted(fn):
        print(f"  {src} ({node_types[src]}) <-> {dst} ({node_types[dst]})")

    precision = len(tp) / len(pred_set) if len(pred_set) > 0 else 0.0
    recall = len(tp) / len(gt_set) if len(gt_set) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    print(f"\n📊 Metrics: P={precision:.3f} R={recall:.3f} F1={f1:.3f}")
    print("="*60)


# =========================
# Main Testing Script
# =========================
def main():
    print("="*60)
    print("ResPlan Edge Prediction Model - Testing")
    print("="*60)

    # Paths (adjust to your system)
    model_path = r"C:\Users\hmbashir\AI Training\Graph2Plan\semantic_edge_best_resplan.pt"
    threshold_path = r"C:\Users\hmbashir\AI Training\Graph2Plan\best_threshold_resplan.json"
    data_path = r"C:\Users\hmbashir\AI Training\Graph2Plan\data_train_converted.pkl"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    # Load data
    print("Loading data...")
    type_to_id = {}
    all_graphs = load_resplan_from_converted_pkl(data_path, type_to_id)
    print(f"✅ Loaded {len(all_graphs)} graphs")
    print(f"✅ Room types ({len(type_to_id)}): {list(type_to_id.keys())}\n")

    # Split into train/val/test (same as training: 70/15/15)
    n = len(all_graphs)
    train_end = int(0.70 * n)
    val_end = train_end + int(0.15 * n)

    train_graphs = all_graphs[:train_end]
    val_graphs = all_graphs[train_end:val_end]
    test_graphs = all_graphs[val_end:]

    print(f"Split: {len(train_graphs)} train / {len(val_graphs)} val / {len(test_graphs)} test\n")

    # Load model
    print(f"Loading model from: {model_path}")
    num_types = len(type_to_id)
    model = TypeCompatModel(num_types=num_types, emb_dim=64, hidden=128, dropout=0.2).to(device)

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("✅ Model loaded successfully")
    else:
        print(f"❌ Model not found at {model_path}")
        return

    # Load threshold
    threshold_logit = 0.0  # Default
    if os.path.exists(threshold_path):
        with open(threshold_path, 'r') as f:
            threshold_config = json.load(f)
            threshold_logit = threshold_config.get('threshold', 0.0)
            print(f"✅ Using threshold: {threshold_logit:.4f} (P={threshold_config.get('precision', 0):.3f}, R={threshold_config.get('recall', 0):.3f}, F1={threshold_config.get('f1', 0):.3f})")
    else:
        print(f"⚠️  Threshold config not found, using default: {threshold_logit}")

    print("\n" + "="*60)
    print("Evaluating on Test Set")
    print("="*60)

    # Evaluate on test set
    metrics = evaluate_test_set(model, test_graphs, threshold_logit, device, type_to_id)

    print(f"\n📊 Test Set Results:")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1']:.4f}")
    print(f"\n  True Positives:  {metrics['tp']}")
    print(f"  True Negatives:  {metrics['tn']}")
    print(f"  False Positives: {metrics['fp']}")
    print(f"  False Negatives: {metrics['fn']}")

    # Visualize predictions on a few test examples
    print("\n" + "="*60)
    print("Example Predictions (first 3 test graphs)")
    print("="*60)

    for i in range(min(3, len(test_graphs))):
        graph = test_graphs[i]

        # Get ground truth (move to CPU first if needed)
        gt_edges = []
        edge_index = graph.edge_index.cpu().numpy()
        for j in range(edge_index.shape[1]):
            gt_edges.append((edge_index[0, j], edge_index[1, j]))

        # Predict
        pred_edges, scores = predict_edges_for_graph(model, graph, threshold_logit, device, type_to_id)

        # Visualize
        visualize_predictions(graph, pred_edges, gt_edges, type_to_id)

    print("\n" + "="*60)
    print("Testing Complete!")
    print("="*60)


if __name__ == "__main__":
    main()

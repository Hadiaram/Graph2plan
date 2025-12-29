"""
predict_floating_edges.py
Predict edges for floating nodes using trained edge prediction model.
This can be integrated into AutoAdjustGraph to connect disconnected components.
"""

import json
import torch
import torch.nn as nn
import numpy as np
from torch_geometric.data import Data


# =========================
# Model Definition (same as training)
# =========================
class TypeCompatModel(nn.Module):
    """Type-only edge prediction model."""
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
# Edge Predictor Utility
# =========================
class FloatingEdgePredictor:
    """
    Utility class for predicting edges on floating nodes.
    """
    def __init__(self, model_path, threshold_path=None, device=None):
        """
        Args:
            model_path: Path to saved model weights
            threshold_path: Path to threshold config JSON (optional)
            device: torch device (default: auto-detect CUDA)
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load threshold
        self.threshold_logit = 0.0
        if threshold_path:
            with open(threshold_path, 'r') as f:
                config = json.load(f)
                self.threshold_logit = config.get('threshold', 0.0)

        # Load model (will set num_types when first used)
        self.model = None
        self.model_path = model_path

    def _ensure_model_loaded(self, num_types):
        """Load model if not already loaded."""
        if self.model is None:
            self.model = TypeCompatModel(num_types=num_types, emb_dim=64, hidden=128, dropout=0.2).to(self.device)
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            self.model.eval()

    def predict_edges_for_floating_nodes(self, node_types, existing_edges, floating_indices=None):
        """
        Predict edges to connect floating nodes.

        Args:
            node_types: List/array of room type IDs (length = num_nodes)
            existing_edges: List of (src, dst) tuples for existing edges
            floating_indices: List of node indices that are floating (no edges).
                             If None, will auto-detect from existing_edges.

        Returns:
            predicted_edges: List of (src, dst) tuples for new edges to add
            scores: Dict mapping (src, dst) to edge score (logit)
        """
        num_nodes = len(node_types)
        self._ensure_model_loaded(num_types=max(node_types) + 1)

        # Auto-detect floating nodes if not provided
        if floating_indices is None:
            connected = set()
            for src, dst in existing_edges:
                connected.add(src)
                connected.add(dst)
            floating_indices = [i for i in range(num_nodes) if i not in connected]

        if len(floating_indices) == 0:
            return [], {}

        # Convert to tensor
        type_tensor = torch.tensor(node_types, dtype=torch.long, device=self.device)

        predicted_edges = []
        scores = {}

        with torch.no_grad():
            for floating_idx in floating_indices:
                # Predict edges to all other nodes
                other_indices = [i for i in range(num_nodes) if i != floating_idx]
                if len(other_indices) == 0:
                    continue

                # Create edge pairs [2, K] format: [[floating_idx, floating_idx, ...], [other_0, other_1, ...]]
                edge_pairs = torch.stack([
                    torch.full((len(other_indices),), floating_idx, dtype=torch.long, device=self.device),
                    torch.tensor(other_indices, dtype=torch.long, device=self.device)
                ], dim=0)

                # Get predictions (pass full type_tensor and edge indices)
                logits = self.model(type_tensor, edge_pairs)

                # Find edges above threshold
                for i, other_idx in enumerate(other_indices):
                    logit = logits[i].item()
                    scores[(floating_idx, other_idx)] = logit

                    if logit >= self.threshold_logit:
                        predicted_edges.append((floating_idx, other_idx))
                        predicted_edges.append((other_idx, floating_idx))  # Undirected

        return predicted_edges, scores

    def predict_top_k_edges_for_node(self, node_idx, node_types, k=3):
        """
        Predict top-k most likely edges for a specific node.

        Args:
            node_idx: Index of the node to connect
            node_types: List/array of room type IDs
            k: Number of top edges to return

        Returns:
            top_edges: List of (other_idx, score) tuples, sorted by score descending
        """
        num_nodes = len(node_types)
        self._ensure_model_loaded(num_types=max(node_types) + 1)

        type_tensor = torch.tensor(node_types, dtype=torch.long, device=self.device)
        node_type = type_tensor[node_idx]

        other_indices = [i for i in range(num_nodes) if i != node_idx]
        if len(other_indices) == 0:
            return []

        with torch.no_grad():
            # Create edge pairs [2, K]
            edge_pairs = torch.stack([
                torch.full((len(other_indices),), node_idx, dtype=torch.long, device=self.device),
                torch.tensor(other_indices, dtype=torch.long, device=self.device)
            ], dim=0)

            logits = self.model(type_tensor, edge_pairs)

            # Sort by score
            scores = logits.cpu().numpy()
            sorted_indices = np.argsort(scores)[::-1]  # Descending

            top_edges = []
            for i in sorted_indices[:k]:
                other_idx = other_indices[i]
                score = scores[i]
                top_edges.append((other_idx, float(score)))

        return top_edges


# =========================
# Integration Example
# =========================
def example_usage():
    """
    Example of how to use FloatingEdgePredictor in AutoAdjustGraph.
    """
    print("="*60)
    print("Floating Edge Predictor - Example Usage")
    print("="*60)

    # Paths
    model_path = r"C:\Users\hmbashir\AI Training\Graph2Plan\semantic_edge_best_resplan.pt"
    threshold_path = r"C:\Users\hmbashir\AI Training\Graph2Plan\best_threshold_resplan.json"

    # Initialize predictor
    predictor = FloatingEdgePredictor(model_path, threshold_path)
    print(f"✅ Predictor initialized")
    print(f"   Threshold: {predictor.threshold_logit:.4f}\n")

    # Example floor plan with floating nodes
    # ResPlan types: 0=living, 1=bedroom, 2=kitchen, 3=bathroom, 9=balcony, 15=front_door
    node_types = [0, 1, 1, 2, 3, 9, 15]  # 7 nodes
    existing_edges = [
        (0, 2),  # living <-> kitchen
        (1, 3),  # bedroom <-> bathroom
        # Nodes 4 (bathroom), 5 (balcony), 6 (front_door) are floating
    ]

    print("Example floor plan:")
    print(f"  Nodes: {len(node_types)}")
    print(f"  Node types: {node_types}")
    print(f"  Existing edges: {existing_edges}")
    print(f"  Floating nodes: 4, 5, 6\n")

    # Predict edges for floating nodes
    predicted, scores = predictor.predict_edges_for_floating_nodes(
        node_types, existing_edges
    )

    print(f"Predicted edges ({len(predicted) // 2} undirected):")
    seen = set()
    for src, dst in predicted:
        if (src, dst) not in seen and (dst, src) not in seen:
            score = scores.get((src, dst)) or scores.get((dst, src))
            print(f"  {src} (type_{node_types[src]}) <-> {dst} (type_{node_types[dst]}) | score={score:.4f}")
            seen.add((src, dst))

    # Alternative: Get top-3 edges for a specific floating node
    print(f"\nTop-3 edges for node 5 (balcony):")
    top_edges = predictor.predict_top_k_edges_for_node(5, node_types, k=3)
    for other_idx, score in top_edges:
        print(f"  5 (balcony) <-> {other_idx} (type_{node_types[other_idx]}) | score={score:.4f}")

    print("\n" + "="*60)


# =========================
# AutoAdjustGraph Integration Snippet
# =========================
def integrate_into_autoadjustgraph():
    """
    Code snippet showing how to integrate into AutoAdjustGraph.py
    """
    snippet = '''
# In AutoAdjustGraph.py, add this to the initialization:

from predict_floating_edges import FloatingEdgePredictor

class AutoAdjustGraph:
    def __init__(self):
        # ... existing initialization ...

        # Initialize edge predictor
        model_path = "path/to/semantic_edge_best_resplan.pt"
        threshold_path = "path/to/best_threshold_resplan.json"
        self.edge_predictor = FloatingEdgePredictor(model_path, threshold_path)

    def connect_floating_nodes(self, node_types, existing_edges):
        """
        Use ML model to predict edges for floating nodes.

        Args:
            node_types: List of room type IDs
            existing_edges: List of (src, dst) current edges

        Returns:
            predicted_edges: List of (src, dst) new edges to add
        """
        predicted, scores = self.edge_predictor.predict_edges_for_floating_nodes(
            node_types, existing_edges
        )

        # Filter to only unique undirected edges
        unique_edges = set()
        for src, dst in predicted:
            if src < dst:
                unique_edges.add((src, dst))

        return list(unique_edges)

    # Usage in your graph adjustment logic:
    def adjust_graph(self):
        # ... existing code ...

        # After constructing graph, if there are floating nodes:
        floating_nodes = self.detect_floating_nodes()

        if len(floating_nodes) > 0:
            # Use ML model to predict edges
            node_types = [self.get_node_type(i) for i in range(self.num_nodes)]
            existing_edges = [(e.source, e.target) for e in self.edges]

            new_edges = self.connect_floating_nodes(node_types, existing_edges)

            # Add predicted edges to graph
            for src, dst in new_edges:
                self.add_edge(src, dst)
    '''

    print("="*60)
    print("AutoAdjustGraph Integration Snippet")
    print("="*60)
    print(snippet)
    print("="*60)


if __name__ == "__main__":
    example_usage()
    print("\n")
    integrate_into_autoadjustgraph()

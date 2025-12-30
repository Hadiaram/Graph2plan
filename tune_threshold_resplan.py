"""
tune_threshold_resplan.py
Standalone script to tune the decision threshold for edge prediction model.
Run this if training was stopped early and threshold tuning didn't complete.
"""

import os
import json
import torch
import torch.nn as nn
import numpy as np
from torch_geometric.data import DataLoader
from sklearn.metrics import precision_recall_curve, average_precision_score
from load_resplan_graphs import load_resplan_from_converted_pkl

# Same config as training
SEED = 42
TARGET_PRECISION = 0.80
BATCH_SIZE = 1


class TypeCompatModel(nn.Module):
    """Same model as training."""
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
    def _norm_pairs(edge_pairs):
        ep = edge_pairs
        if isinstance(ep, (list, tuple)):
            ep = torch.as_tensor(ep)
        ep = ep.squeeze()
        if ep.dim() == 2 and ep.size(0) == 2:
            return ep.long()
        if ep.dim() == 2 and ep.size(1) == 2:
            return ep.t().long()
        if ep.dim() == 1:
            assert ep.numel() % 2 == 0
            return ep.view(2, -1).long()
        axes = [i for i in range(ep.dim()) if ep.size(i) == 2]
        if axes:
            return ep.movedim(axes[0], 0).reshape(2, -1).long()
        assert ep.numel() % 2 == 0
        return ep.reshape(2, -1).long()

    def forward(self, type_ids, edge_pairs):
        pairs = self._norm_pairs(edge_pairs)
        if pairs.numel() == 0:
            return torch.empty(0, dtype=torch.float, device=type_ids.device)
        u, v = pairs
        eu, ev = self.emb(type_ids[u]), self.emb(type_ids[v])
        feats = torch.cat([eu, ev, torch.abs(eu - ev), eu * ev], dim=1)
        feats = self.ln(feats)
        return self.edge_mlp(feats).squeeze(-1)


class GraphEdgeDataset(torch.utils.data.Dataset):
    """Same dataset as training (eval mode)."""
    def __init__(self, graphs, seed=SEED):
        from torch_geometric.utils import negative_sampling
        self.graphs = graphs
        self.fixed_negs = []
        rng = np.random.RandomState(seed)
        for d in self.graphs:
            k = d.edge_index.size(1)
            neg = negative_sampling(
                edge_index=d.edge_index,
                num_nodes=d.num_nodes,
                num_neg_samples=k,
                method='sparse'
            )
            order = rng.permutation(neg.size(1))
            self.fixed_negs.append(neg[:, order])

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        d = self.graphs[idx]
        pos = d.edge_index
        neg = self.fixed_negs[idx]
        return d, pos, neg


def collect_scores(model, loader, device):
    """Collect all predictions and ground truth labels."""
    y_true, y_score = [], []
    model.eval()

    with torch.no_grad():
        for d, pos, neg in loader:
            d = d.to(device)
            type_ids = d.x[:, 0].long()
            pos = pos.to(device)
            neg = neg.to(device)

            pos_logits = model(type_ids, pos)
            neg_logits = model(type_ids, neg)

            y_true += [1] * pos_logits.size(0) + [0] * neg_logits.size(0)
            y_score += pos_logits.tolist() + neg_logits.tolist()

    return np.asarray(y_true), np.asarray(y_score)


def choose_threshold(y_true, y_score, target_p=TARGET_PRECISION):
    """
    Find best threshold targeting precision >= target_p.
    Falls back to best F1 if no threshold achieves target precision.
    """
    P, R, T = precision_recall_curve(y_true, y_score)
    ap = float(average_precision_score(y_true, y_score))

    # Try to find threshold with P >= target_p
    idx_valid = np.where((P >= target_p) & (np.arange(len(P)) > 0))[0]

    if len(idx_valid) > 0:
        # Pick highest recall among valid thresholds
        j = idx_valid[np.argmax(R[idx_valid])]
        thr = float(T[j - 1])
        how = "precision_target"
        P_sel, R_sel = float(P[j]), float(R[j])
        F1_sel = float(2 * P_sel * R_sel / (P_sel + R_sel + 1e-9))
    else:
        # Fallback: best F1
        if len(T) == 0:
            return 0.0, 1.0, 0.0, 0.0, ap, "degenerate_safe0"

        f1s = (2 * P[1:] * R[1:]) / (P[1:] + R[1:] + 1e-9)
        j = int(np.nanargmax(f1s))
        thr = float(T[j])
        how = "best_f1_fallback"
        P_sel, R_sel = float(P[j + 1]), float(R[j + 1])
        F1_sel = float(f1s[j])

    if not np.isfinite(thr):
        thr, how = 0.0, how + "_safe0"

    return thr, P_sel, R_sel, F1_sel, ap, how


def main():
    print("=" * 60)
    print("Threshold Tuning - ResPlan Edge Prediction")
    print("=" * 60)

    # Paths
    model_path = r"C:\Users\hmbashir\AI Training\Graph2Plan\semantic_edge_best_resplan.pt"
    data_path = r"C:\Users\hmbashir\AI Training\Graph2Plan\Interface\static\Data\data_train_converted.pkl"
    output_path = r"C:\Users\hmbashir\AI Training\Graph2Plan\best_threshold_resplan.json"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    # Check files exist
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return

    if not os.path.exists(data_path):
        print(f"❌ Data not found: {data_path}")
        return

    # Load data
    print(f"📂 Loading data from: {data_path}")
    type_to_id = {}
    all_graphs = load_resplan_from_converted_pkl(data_path, type_to_id)
    all_graphs = [g for g in all_graphs if g.edge_index.size(1) > 0]

    print(f"✅ Loaded {len(all_graphs)} graphs")
    print(f"✅ Found {len(type_to_id)} room types\n")

    # Split into train/val/test (same as training: 70/15/15)
    n = len(all_graphs)
    train_end = int(0.70 * n)
    val_end = train_end + int(0.15 * n)

    val_graphs = all_graphs[train_end:val_end]
    print(f"📊 Using {len(val_graphs)} validation graphs for threshold tuning\n")

    # Load model
    print(f"📥 Loading model from: {model_path}")
    model = TypeCompatModel(
        num_types=len(type_to_id),
        emb_dim=64,
        hidden=128,
        dropout=0.2
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    print("✅ Model loaded successfully\n")

    # Create validation dataset
    val_ds = GraphEdgeDataset(val_graphs, seed=SEED)
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=lambda batch: batch[0]
    )

    # Collect predictions
    print("🔍 Collecting predictions on validation set...")
    y_true, y_score = collect_scores(model, val_loader, device)
    print(f"✅ Collected {len(y_true)} edge predictions\n")

    # Tune threshold
    print(f"🎯 Tuning threshold (target precision: {TARGET_PRECISION:.2f})...")
    thr, P, R, F1, AP, how = choose_threshold(y_true, y_score, TARGET_PRECISION)

    print(f"\n{'=' * 60}")
    print("Threshold Tuning Results")
    print("=" * 60)
    print(f"  Method:       {how}")
    print(f"  Threshold:    {thr:.4f}")
    print(f"  Precision:    {P:.4f}")
    print(f"  Recall:       {R:.4f}")
    print(f"  F1 Score:     {F1:.4f}")
    print(f"  Avg Precision: {AP:.4f}")
    print("=" * 60)

    # Save threshold config
    config = {
        "threshold": float(thr),
        "precision": float(P),
        "recall": float(R),
        "f1": float(F1),
        "ap": float(AP),
        "how": how,
        "target_precision": TARGET_PRECISION
    }

    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"\n💾 Saved threshold config to: {output_path}")
    print("\n✅ Threshold tuning complete!")
    print(f"\nℹ️  Re-run test_edge_model_resplan.py to see improved results.")


if __name__ == "__main__":
    main()

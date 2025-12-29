"""
train_semantic_edge_model_resplan_batched.py
Batched version of edge prediction training for faster GPU utilization.
"""

import os, re, json, random, pickle, math, csv
from collections import Counter
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import WeightedRandomSampler
from torch_geometric.data import DataLoader, Batch
from torch_geometric.utils import from_networkx, to_undirected, remove_self_loops, negative_sampling

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score,
    precision_recall_curve, average_precision_score
)

import matplotlib.pyplot as plt
import seaborn as sns

from load_resplan_graphs import load_resplan_from_converted_pkl

# =========================
# Config / Reproducibility
# =========================
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED), torch.cuda.manual_seed(SEED), torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# --- Training mix ---
NEG_POS_RATIO   = 1
NEG_WEIGHT      = 1.0
POS_WEIGHT      = 1.0

# --- Threshold selection ---
TARGET_PRECISION = 0.80
BEST_THR_JSON    = "best_threshold_resplan.json"

# --- Dataloading ---
BATCH_SIZE = 32  # Process 32 graphs at once (adjust based on GPU memory)

# --- Paths ---
TRAIN_PKL = r"C:\Users\hmbashir\AI Training\Graph2Plan\Interface\static\Data\data_train_converted.pkl"
TEST_PKL = r"C:\Users\hmbashir\AI Training\Graph2Plan\Interface\static\Data\data_test_converted.pkl"

# =========================
# Model: Type-only compatibility
# =========================
class TypeCompatModel(nn.Module):
    def __init__(self, num_types, emb_dim=64, hidden=128, dropout=0.2):
        super().__init__()
        self.emb = nn.Embedding(num_types, emb_dim)
        self.ln  = nn.LayerNorm(4*emb_dim)
        self.edge_mlp = nn.Sequential(
            nn.Linear(4*emb_dim, hidden),
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
        pairs = self._norm_pairs(edge_pairs)
        if pairs.numel() == 0:
            return torch.empty(0, dtype=torch.float, device=type_ids.device)

        u, v = pairs
        eu, ev = self.emb(type_ids[u]), self.emb(type_ids[v])
        feats = torch.cat([eu, ev, torch.abs(eu-ev), eu*ev], dim=1)
        feats = self.ln(feats)
        return self.edge_mlp(feats).squeeze(-1)

# =========================
# Batched Training with Dynamic Negative Sampling
# =========================
def train_epoch_batched(model, loader, opt, scaler, device):
    """
    Train for one epoch using batched graphs.
    Negative sampling happens on-the-fly for the entire batch.
    """
    model.train()
    total_loss = 0.0

    for batch in loader:
        batch = batch.to(device)
        type_ids = batch.x[:, 0].long()

        # Positive edges from the batch
        pos_edges = batch.edge_index
        num_pos = pos_edges.size(1)

        # Generate negative edges for the batch
        # PyG's negative_sampling works on batched graphs
        num_neg = int(max(1, NEG_POS_RATIO) * num_pos)
        neg_edges = negative_sampling(
            edge_index=pos_edges,
            num_nodes=batch.num_nodes,
            num_neg_samples=num_neg,
            method='sparse'
        )

        # Get logits
        logits_pos = model(type_ids, pos_edges)
        logits_neg = model(type_ids, neg_edges)

        logits = torch.cat([logits_pos, logits_neg], dim=0)
        labels = torch.cat([
            torch.ones(logits_pos.size(0), device=device),
            torch.zeros(logits_neg.size(0), device=device)
        ], dim=0)

        # Weights
        w = torch.ones_like(labels)
        w[labels==0] = NEG_WEIGHT
        w[labels==1] = POS_WEIGHT

        # Backward pass
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
            loss = F.binary_cross_entropy_with_logits(logits, labels, weight=w)

        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

        total_loss += loss.item()

    return total_loss / max(1, len(loader))


# =========================
# Batched Evaluation
# =========================
def evaluate_batched(model, loader, device, threshold_logit=0.0):
    """Evaluate on batched graphs."""
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            type_ids = batch.x[:, 0].long()

            # Positive edges
            pos_edges = batch.edge_index
            num_pos = pos_edges.size(1)

            # Negative edges (same amount as positive for balanced eval)
            neg_edges = negative_sampling(
                edge_index=pos_edges,
                num_nodes=batch.num_nodes,
                num_neg_samples=num_pos,
                method='sparse'
            )

            # Predictions
            pos_logits = model(type_ids, pos_edges)
            neg_logits = model(type_ids, neg_edges)

            y_true += [1] * pos_logits.size(0) + [0] * neg_logits.size(0)
            y_pred += (pos_logits > threshold_logit).int().tolist() + \
                      (neg_logits > threshold_logit).int().tolist()

    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    return f1, cm


def collect_scores_batched(model, loader, device):
    """Collect scores for threshold tuning."""
    y_true, y_score = [], []
    model.eval()

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            type_ids = batch.x[:, 0].long()

            pos_edges = batch.edge_index
            num_pos = pos_edges.size(1)

            neg_edges = negative_sampling(
                edge_index=pos_edges,
                num_nodes=batch.num_nodes,
                num_neg_samples=num_pos,
                method='sparse'
            )

            pos_logits = model(type_ids, pos_edges)
            neg_logits = model(type_ids, neg_edges)

            y_true += [1] * pos_logits.size(0) + [0] * neg_logits.size(0)
            y_score += pos_logits.tolist() + neg_logits.tolist()

    return np.asarray(y_true), np.asarray(y_score)


def choose_threshold_from_scores(y_true, y_score, target_p=TARGET_PRECISION):
    P, R, T = precision_recall_curve(y_true, y_score)
    ap = float(average_precision_score(y_true, y_score))

    idx_valid = np.where((P >= target_p) & (np.arange(len(P)) > 0))[0]

    if len(idx_valid) > 0:
        j = idx_valid[np.argmax(R[idx_valid])]
        thr = float(T[j-1]); how = "precision_target"
        P_sel, R_sel = float(P[j]), float(R[j])
        F1_sel = float(2*P_sel*R_sel/(P_sel+R_sel+1e-9))
    else:
        if len(T)==0: return 0.0, 1.0, 0.0, 0.0, ap, "degenerate_safe0"
        f1s = (2*P[1:]*R[1:])/(P[1:]+R[1:]+1e-9)
        j = int(np.nanargmax(f1s))
        thr = float(T[j]); how = "best_f1_fallback"
        P_sel, R_sel = float(P[j+1]), float(R[j+1])
        F1_sel = float(f1s[j])

    if not np.isfinite(thr): thr, how = 0.0, how+"_safe0"
    return thr, P_sel, R_sel, F1_sel, ap, how


# =========================
# Main Training Loop
# =========================
def train(model, train_loader, val_loader, device, epochs=500, lr=1e-3,
          save_path="semantic_edge_best_resplan.pt"):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    best_f1 = -1.0
    best_loss = float('inf')

    for ep in range(epochs):
        # Train
        avg_loss = train_epoch_batched(model, train_loader, opt, scaler, device)

        # Evaluate
        f1_0, _ = evaluate_batched(model, val_loader, device, threshold_logit=0.0)
        print(f"📚 Epoch {ep+1}/{epochs} | Loss {avg_loss:.4f} | Val F1@0.0 {f1_0:.4f}")

        # Save if improved
        should_save = False
        reason = ""

        if f1_0 > best_f1:
            should_save = True
            reason = f"F1 improved: {best_f1:.4f} -> {f1_0:.4f}"
            best_f1 = f1_0
            best_loss = avg_loss
        elif f1_0 == best_f1 and avg_loss < best_loss:
            should_save = True
            reason = f"Same F1 ({f1_0:.4f}), loss improved: {best_loss:.4f} -> {avg_loss:.4f}"
            best_loss = avg_loss

        if should_save:
            torch.save(model.state_dict(), save_path)
            print(f"💾 Saved best @ epoch {ep+1} ({reason})")


# =========================
# Main
# =========================
if __name__ == "__main__":
    print("="*60)
    print("Training Semantic Edge Prediction Model (BATCHED)")
    print("="*60)
    print(f"Batch Size: {BATCH_SIZE}")
    print("="*60)

    if not os.path.isfile(TRAIN_PKL):
        raise FileNotFoundError(f"Training data not found: {TRAIN_PKL}")

    print(f"\n📂 Loading training data from: {TRAIN_PKL}")

    # Load data
    type_to_id = {}
    all_graphs = load_resplan_from_converted_pkl(TRAIN_PKL, type_to_id)

    # Save vocabulary
    with open("room_type_vocab_resplan.json","w") as f:
        json.dump(type_to_id, f, indent=2)
    print(f"\n💾 Saved type vocab with {len(type_to_id)} types.")
    print(f"   Types: {list(type_to_id.keys())}")

    # Keep graphs with edges
    all_graphs = [g for g in all_graphs if g.edge_index.size(1) > 0]
    print(f"\n📦 Total graphs with edges: {len(all_graphs)}")

    # Split into train/val/test (70/15/15)
    train_graphs, temp_graphs = train_test_split(
        all_graphs, test_size=0.3, random_state=SEED, shuffle=True
    )
    val_graphs, test_graphs = train_test_split(
        temp_graphs, test_size=0.5, random_state=SEED, shuffle=True
    )

    print(f"📊 Split: {len(train_graphs)} train / {len(val_graphs)} val / {len(test_graphs)} test")

    # Create DataLoaders (no custom collate_fn needed - PyG handles batching)
    train_loader = DataLoader(train_graphs, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_graphs,   batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(test_graphs,  batch_size=BATCH_SIZE, shuffle=False)

    print(f"\n🔢 Batches per epoch: {len(train_loader)} (batch_size={BATCH_SIZE})")

    # Model setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Using device: {device}")

    model = TypeCompatModel(num_types=len(type_to_id), emb_dim=64, hidden=128, dropout=0.2).to(device)
    print(f"🧠 Model created with {len(type_to_id)} room types")

    # ---- Train ----
    print(f"\n🚀 Starting training for 500 epochs...")
    train(model, train_loader, val_loader, device, epochs=500, lr=1e-3,
          save_path="semantic_edge_best_resplan.pt")

    # ---- Threshold tuning ----
    print(f"\n🎯 Tuning threshold on validation set...")
    model.load_state_dict(torch.load("semantic_edge_best_resplan.pt", map_location=device))
    y_true, y_score = collect_scores_batched(model, val_loader, device)
    thr, P, R, F1, AP, how = choose_threshold_from_scores(y_true, y_score, TARGET_PRECISION)

    with open(BEST_THR_JSON,"w") as f:
        json.dump({
            "threshold": float(thr),
            "precision": P,
            "recall": R,
            "f1": F1,
            "ap": AP,
            "how": how,
            "target_precision": TARGET_PRECISION
        }, f, indent=2)

    print(f"\n🎯 Tuned threshold t={thr:.3f} via {how}")
    print(f"   Precision: {P:.3f} | Recall: {R:.3f} | F1: {F1:.3f} | AP: {AP:.3f}")
    print(f"   Saved to: {BEST_THR_JSON}")

    # ---- Final test ----
    print(f"\n📊 Evaluating on test set...")
    f1_test, cm_test = evaluate_batched(model, test_loader, device, threshold_logit=thr)

    print(f"\n✅ Final Results:")
    print(f"   Test F1 Score: {f1_test:.4f}")
    print(f"   Confusion Matrix:\n{cm_test}")

    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"\n📁 Output files:")
    print(f"   - Model: semantic_edge_best_resplan.pt")
    print(f"   - Threshold: {BEST_THR_JSON}")
    print(f"   - Vocabulary: room_type_vocab_resplan.json")

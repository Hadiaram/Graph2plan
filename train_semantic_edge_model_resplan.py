# train_semantic_edge_model_resplan.py
# Train edge prediction model using converted ResPlan data from Graph2plan Interface
# Learns room-to-room connectivity from room FUNCTION (types) only.

import os, re, json, random, pickle, math, csv
from collections import Counter
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import WeightedRandomSampler
from torch_geometric.data import DataLoader
from torch_geometric.utils import from_networkx, to_undirected, remove_self_loops, negative_sampling

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score,
    precision_recall_curve, average_precision_score
)

import matplotlib.pyplot as plt
import seaborn as sns

# Import our custom loader
from load_resplan_graphs import load_resplan_from_converted_pkl

# =========================
# Config / Reproducibility
# =========================
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED), torch.cuda.manual_seed(SEED), torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.benchmark = False  # for stable performance
torch.backends.cudnn.deterministic = True  # for reproducibility

# --- Training mix ---
NEG_POS_RATIO   = 1          # negatives per positive (balanced per-batch)
NEG_WEIGHT      = 1.0        # keep symmetric; we want recall
POS_WEIGHT      = 1.0

# --- Threshold selection ---
TARGET_PRECISION = 0.80       # pick highest-recall point s.t. P >= target
BEST_THR_JSON    = "best_threshold_resplan.json"

# --- Dataloading ---
BATCH_SIZE = 1               # graphs are heterogeneous; keep 1 for stability

# --- Paths ---
TRAIN_PKL = r"C:\Users\hmbashir\AI Training\Graph2Plan\Interface\static\Data\data_train_converted.pkl"
TEST_PKL = r"C:\Users\hmbashir\AI Training\Graph2Plan\Interface\static\Data\data_test_converted.pkl"

# =========================
# Dataset (random negatives only)
# =========================
class GraphEdgeDataset(torch.utils.data.Dataset):
    """Yields (data, pos_edge_index, neg_edge_index). Eval uses fixed RNG for stability."""
    def __init__(self, graphs, mode="train", seed=SEED):
        assert mode in ("train","eval")
        self.graphs = graphs; self.mode = mode; self.seed = seed
        self.fixed_negs = []
        if mode == "eval":
            rng = np.random.RandomState(seed)
            for d in self.graphs:
                k = d.edge_index.size(1)
                neg = negative_sampling(edge_index=d.edge_index, num_nodes=d.num_nodes,
                                        num_neg_samples=k, method='sparse')
                order = rng.permutation(neg.size(1))
                self.fixed_negs.append(neg[:, order])

    def __len__(self): return len(self.graphs)

    def __getitem__(self, idx):
        d = self.graphs[idx]
        pos = d.edge_index
        if self.mode == "train":
            k = pos.size(1)
            num_negs = int(max(1, NEG_POS_RATIO) * k)
            neg = negative_sampling(edge_index=pos, num_nodes=d.num_nodes,
                                    num_neg_samples=num_negs, method='sparse')
        else:
            neg = self.fixed_negs[idx]
        return d, pos, neg

# =========================
# Model: Type-only compatibility
# =========================
class TypeCompatModel(nn.Module):
    """
    Learn P(edge | type_u, type_v).
    Inputs:
      - type_ids: [N] long (indices into embedding table)
      - edge_pairs: [2, K]
    """
    def __init__(self, num_types, emb_dim=64, hidden=128, dropout=0.2):
        super().__init__()
        self.emb = nn.Embedding(num_types, emb_dim)
        self.ln  = nn.LayerNorm(4*emb_dim)         # stable with bs=1
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
# Threshold tuning
# =========================
def collect_scores(model, loader, device):
    y_true, y_score = [], []
    model.eval()
    with torch.no_grad():
        for d, pos, neg in loader:
            d = d.to(device)
            type_ids = d.x[:,0].long()      # use ONLY type ids
            pos = pos.to(device); neg = neg.to(device)
            pos_logits = model(type_ids, pos)
            neg_logits = model(type_ids, neg)
            y_true += [1]*pos_logits.size(0) + [0]*neg_logits.size(0)
            y_score += pos_logits.tolist()  + neg_logits.tolist()
    return np.asarray(y_true), np.asarray(y_score)

def choose_threshold_from_scores(y_true, y_score, target_p=TARGET_PRECISION):
    P, R, T = precision_recall_curve(y_true, y_score)
    ap = float(average_precision_score(y_true, y_score))
    # thresholds align with P[1:],R[1:]
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
# Evaluation
# =========================
def evaluate(model, loader, device, threshold_logit=0.0, return_metrics=False, title="Confusion Matrix"):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for d, pos, neg in loader:
            d = d.to(device)
            type_ids = d.x[:,0].long()
            pos = pos.to(device); neg = neg.to(device)
            p = model(type_ids, pos); n = model(type_ids, neg)
            y_true += [1]*p.size(0) + [0]*n.size(0)
            y_pred += (p > threshold_logit).int().tolist() + (n > threshold_logit).int().tolist()

    cm = confusion_matrix(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    if return_metrics:
        return f1, cm

    print("\n📊 Classification report:\n", classification_report(y_true, y_pred, zero_division=0, digits=4))
    print("F1:", f1)
    print("Confusion matrix:\n", cm)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted'); plt.ylabel('Actual'); plt.title(title)
    plt.tight_layout(); plt.show()

# =========================
# Training
# =========================
def train(model, train_loader, val_loader, device, epochs=20000, lr=1e-3, save_path="semantic_edge_best_resplan.pt"):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    best_f1 = -1.0
    best_loss = float('inf')

    for ep in range(epochs):
        model.train()
        total = 0.0
        for d, pos, neg in train_loader:
            d = d.to(device)
            type_ids = d.x[:,0].long()

            pos = pos.to(device); neg = neg.to(device)
            logits_pos = model(type_ids, pos)
            logits_neg = model(type_ids, neg)
            logits = torch.cat([logits_pos, logits_neg], dim=0)
            labels = torch.cat([
                torch.ones(logits_pos.size(0), device=device),
                torch.zeros(logits_neg.size(0), device=device)
            ], dim=0)

            w = torch.ones_like(labels)
            w[labels==0] = NEG_WEIGHT; w[labels==1] = POS_WEIGHT

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                loss = F.binary_cross_entropy_with_logits(logits, labels, weight=w)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            total += loss.item()

        avg_loss = total / max(1,len(train_loader))
        f1_0, _ = evaluate(model, val_loader, device, threshold_logit=0.0, return_metrics=True)
        print(f"\n📚 Epoch {ep+1}/{epochs} | Loss {avg_loss:.4f} | Val F1@0.0 {f1_0:.4f}")

        # Save if: (1) F1 improved, OR (2) F1 same but loss improved
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
# Inference wiring (optional)
# =========================
def predict_edges_all_pairs(model, d, device, threshold, top_k_per_node=None):
    """Return list of (u,v,logit,pred) for ALL pairs (or restrict with top_k_per_node)."""
    d = d.to(device)
    type_ids = d.x[:,0].long()
    N = d.num_nodes
    pairs = []
    for u in range(N):
        for v in range(u+1, N):
            pairs.append((u,v))
    if not pairs: return []

    uv = torch.tensor(pairs, dtype=torch.long, device=device).T
    with torch.no_grad():
        logits = model(type_ids, uv)
    rows = []
    for (u,v), lg in zip(pairs, logits.tolist()):
        rows.append([u,v,lg, int(lg>threshold)])
    # optional pruning per-node
    if isinstance(top_k_per_node, int) and top_k_per_node > 0:
        by_u = {}
        for u,v,lg,p in rows:
            by_u.setdefault(u, []).append((v,lg,p))
        pruned = []
        for u, lst in by_u.items():
            lst.sort(key=lambda t: -t[1])
            for v,lg,p in lst[:top_k_per_node]:
                pruned.append([u,v,lg,p])
        rows = pruned
    return rows

# =========================
# Main
# =========================
if __name__ == "__main__":
    print("="*60)
    print("Training Semantic Edge Prediction Model - ResPlan Dataset")
    print("="*60)

    # Check files exist
    if not os.path.isfile(TRAIN_PKL):
        raise FileNotFoundError(f"Training data not found: {TRAIN_PKL}")

    # Note: TEST_PKL has different structure (for retrieval), not suitable for edge training
    # We'll split the training data into train/val/test instead

    print(f"\n📂 Loading training data from: {TRAIN_PKL}")

    # Load data
    type_to_id = {}
    all_graphs = load_resplan_from_converted_pkl(TRAIN_PKL, type_to_id)

    # Save vocabulary
    with open("room_type_vocab_resplan.json","w") as f:
        json.dump(type_to_id, f, indent=2)
    print(f"\n💾 Saved type vocab with {len(type_to_id)} types.")
    print(f"   Types: {list(type_to_id.keys())}")

    # Keep graphs with at least one edge
    all_graphs = [g for g in all_graphs if g.edge_index.size(1) > 0]

    print(f"\n📦 Total graphs with edges: {len(all_graphs)}")

    # Split into train (70%), val (15%), test (15%)
    train_graphs, temp_graphs = train_test_split(
        all_graphs, test_size=0.3, random_state=SEED, shuffle=True
    )
    val_graphs, test_graphs = train_test_split(
        temp_graphs, test_size=0.5, random_state=SEED, shuffle=True
    )

    print(f"📊 Split: {len(train_graphs)} train / {len(val_graphs)} val / {len(test_graphs)} test")

    # Create datasets and loaders
    collate_one = lambda batch: batch[0]
    train_ds = GraphEdgeDataset(train_graphs, mode="train", seed=SEED)
    val_ds   = GraphEdgeDataset(val_graphs,   mode="eval",  seed=SEED)
    test_ds  = GraphEdgeDataset(test_graphs,  mode="eval",  seed=SEED)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  collate_fn=collate_one)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_one)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_one)

    # Model setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🖥️  Using device: {device}")

    model = TypeCompatModel(num_types=len(type_to_id), emb_dim=64, hidden=128, dropout=0.2).to(device)
    print(f"🧠 Model created with {len(type_to_id)} room types")

    # ---- Train ----
    print(f"\n🚀 Starting training for 20000 epochs...")
    train(model, train_loader, val_loader, device, epochs=20000, lr=1e-3, save_path="semantic_edge_best_resplan.pt")

    # ---- Threshold tuning on VALIDATION ----
    print(f"\n🎯 Tuning threshold on validation set...")
    model.load_state_dict(torch.load("semantic_edge_best_resplan.pt", map_location=device))
    y_true, y_score = collect_scores(model, val_loader, device)
    thr, P, R, F1, AP, how = choose_threshold_from_scores(y_true, y_score, TARGET_PRECISION)

    with open(BEST_THR_JSON,"w") as f:
        json.dump({
            "threshold": float(thr),
            "precision": P,
            "recall": R,
            "f1": F1,
            "ap": AP,
            "how": how
        }, f, indent=2)

    print(f"\n🎯 Tuned threshold t={thr:.3f} via {how}")
    print(f"   Precision: {P:.3f} | Recall: {R:.3f} | F1: {F1:.3f} | AP: {AP:.3f}")
    print(f"   Saved to: {BEST_THR_JSON}")

    # ---- Final evaluation on TEST set ----
    print(f"\n📊 Evaluating on test set...")
    f1_test, cm_test = evaluate(model, test_loader, device, threshold_logit=thr, return_metrics=True,
                                title=f"Test Set Confusion Matrix (thr={thr:.3f})")

    print(f"\n✅ Final Results:")
    print(f"   Test F1 Score: {f1_test:.4f}")
    print(f"   Confusion Matrix:\n{cm_test}")

    # Optionally visualize confusion matrix
    try:
        plt.figure(figsize=(6,5))
        sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted'); plt.ylabel('Actual')
        plt.title(f'Test Set - ResPlan (thr={thr:.3f})')
        plt.tight_layout()
        plt.savefig('confusion_matrix_resplan.png', dpi=150, bbox_inches='tight')
        print(f"\n📊 Saved confusion matrix to: confusion_matrix_resplan.png")
        plt.show()
    except Exception as e:
        print(f"⚠️  Could not save confusion matrix plot: {e}")

    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"\n📁 Output files:")
    print(f"   - Model: semantic_edge_best_resplan.pt")
    print(f"   - Threshold: {BEST_THR_JSON}")
    print(f"   - Vocabulary: room_type_vocab_resplan.json")
    print(f"   - Confusion Matrix: confusion_matrix_resplan.png")

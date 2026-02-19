"""Train GNN models (GAT, GCN, GraphSAGE) for multi-label GO term prediction.

Uses mini-batch neighbor sampling for scalable training on limited GPU memory.
The manual sampler works without pyg-lib/torch-sparse dependencies.

Usage:
  python train_gnn.py --model gat                    # train GAT
  python train_gnn.py --model gcn                    # train GCN
  python train_gnn.py --model sage                   # train GraphSAGE
  python train_gnn.py --model all                    # train all three sequentially
  python train_gnn.py --model gat --num-layers 3     # depth ablation
  python train_gnn.py --model gat --gat-heads 4      # heads ablation
  python train_gnn.py --model all --seed 123          # different seed for CI
"""

import argparse
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.nn import GATv2Conv, GCNConv, SAGEConv
from torch_geometric.data import Data as PyGData
from sklearn.metrics import classification_report


# =============================================================================
# Manual Neighbor Sampler (no pyg-lib / torch-sparse needed)
# =============================================================================

class NeighborSampler:
    """Mini-batch neighbor sampler using CSR adjacency.

    Drop-in replacement for torch_geometric.loader.NeighborLoader.
    Samples fixed-size neighborhoods per hop to keep mini-batches small,
    enabling GAT (and other models) to train on GPUs with limited VRAM.

    Each yielded batch is a PyG Data object with:
      - batch.x:          [num_sampled_nodes, F]
      - batch.y:          [num_sampled_nodes, C]
      - batch.edge_index: [2, num_sampled_edges]
      - batch.batch_size: int (seed nodes are the first batch_size rows)
    """

    def __init__(self, data, num_neighbors, batch_size, input_nodes, shuffle=True):
        """
        Args:
            data:           PyG Data object (full graph).
            num_neighbors:  list of ints, neighbors to sample per hop (e.g. [15, 10]).
            batch_size:     number of seed (target) nodes per mini-batch.
            input_nodes:    bool mask or index tensor of seed-eligible nodes.
            shuffle:        whether to shuffle seed nodes each epoch.
        """
        self.data = data
        self.num_neighbors = num_neighbors
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Resolve mask → indices
        if input_nodes.dtype == torch.bool:
            self.input_nodes = input_nodes.nonzero(as_tuple=True)[0].numpy()
        else:
            self.input_nodes = input_nodes.numpy()

        # Build CSR from edge_index for O(1) neighbor lookup
        ei = data.edge_index.numpy()
        n = data.num_nodes
        src, dst = ei[0], ei[1]
        order = np.argsort(src)
        self.col = dst[order].astype(np.int64)
        self.rowptr = np.zeros(n + 1, dtype=np.int64)
        np.add.at(self.rowptr, src[order] + 1, 1)
        self.rowptr = np.cumsum(self.rowptr)

    def __len__(self):
        return (len(self.input_nodes) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        nodes = self.input_nodes.copy()
        if self.shuffle:
            np.random.shuffle(nodes)
        for start in range(0, len(nodes), self.batch_size):
            seed = nodes[start:start + self.batch_size]
            yield self._sample(seed)

    def _sample(self, seed_nodes):
        """Build a k-hop subgraph around seed_nodes with bounded edges."""
        # Track all unique nodes (seeds first for easy loss slicing)
        all_nodes = list(seed_nodes)
        node_set = set(seed_nodes.tolist())
        edge_src = []
        edge_dst = []

        frontier = seed_nodes

        for num_nb in self.num_neighbors:
            next_frontier = []
            for v in frontier:
                s, e = self.rowptr[v], self.rowptr[v + 1]
                nbrs = self.col[s:e]
                if len(nbrs) == 0:
                    continue
                if len(nbrs) > num_nb:
                    chosen = nbrs[np.random.choice(len(nbrs), num_nb, replace=False)]
                else:
                    chosen = nbrs

                v_int = int(v)
                for nb in chosen:
                    nb = int(nb)
                    # Both directions (undirected graph)
                    edge_src.append(nb);  edge_dst.append(v_int)
                    edge_src.append(v_int); edge_dst.append(nb)
                    if nb not in node_set:
                        node_set.add(nb)
                        all_nodes.append(nb)
                        next_frontier.append(nb)

            frontier = next_frontier

        # Map global → local indices
        n_id = np.array(all_nodes, dtype=np.int64)
        local_map = np.full(self.data.num_nodes, -1, dtype=np.int64)
        local_map[n_id] = np.arange(len(n_id))

        if edge_src:
            ei = torch.tensor(
                np.stack([local_map[edge_src], local_map[edge_dst]]),
                dtype=torch.long,
            )
        else:
            ei = torch.zeros((2, 0), dtype=torch.long)

        batch = PyGData(
            x=self.data.x[n_id],
            y=self.data.y[n_id],
            edge_index=ei,
        )
        batch.batch_size = len(seed_nodes)
        return batch


# =============================================================================
# Model Definitions (variable depth for R4.6/R5.7 ablation)
# =============================================================================

class GATMultiLabel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, heads=8, dropout=0.3):
        super().__init__()
        self.convs = nn.ModuleList()
        self.dropout = dropout

        if num_layers == 1:
            self.convs.append(GATv2Conv(input_dim, 64, heads=1, concat=False, dropout=dropout))
        else:
            self.convs.append(GATv2Conv(input_dim, hidden_dim, heads=heads, dropout=dropout))
            for _ in range(num_layers - 2):
                self.convs.append(GATv2Conv(hidden_dim * heads, hidden_dim, heads=heads, dropout=dropout))
            self.convs.append(GATv2Conv(hidden_dim * heads, 64, heads=1, concat=False, dropout=dropout))

        self.fc = nn.Linear(64, output_dim)

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index).relu()
            if i < len(self.convs) - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)
        return self.fc(x)


class GCNMultiLabel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.3):
        super().__init__()
        self.convs = nn.ModuleList()
        self.dropout = dropout

        if num_layers == 1:
            self.convs.append(GCNConv(input_dim, 64))
        else:
            self.convs.append(GCNConv(input_dim, hidden_dim))
            for _ in range(num_layers - 2):
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.convs.append(GCNConv(hidden_dim, 64))

        self.fc = nn.Linear(64, output_dim)

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index).relu()
            if i < len(self.convs) - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)
        return self.fc(x)


class SAGEMultiLabel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.3):
        super().__init__()
        self.convs = nn.ModuleList()
        self.dropout = dropout

        if num_layers == 1:
            self.convs.append(SAGEConv(input_dim, 64))
        else:
            self.convs.append(SAGEConv(input_dim, hidden_dim))
            for _ in range(num_layers - 2):
                self.convs.append(SAGEConv(hidden_dim, hidden_dim))
            self.convs.append(SAGEConv(hidden_dim, 64))

        self.fc = nn.Linear(64, output_dim)

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index).relu()
            if i < len(self.convs) - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)
        return self.fc(x)


MODEL_REGISTRY = {
    "gat": GATMultiLabel,
    "gcn": GCNMultiLabel,
    "sage": SAGEMultiLabel,
}


# =============================================================================
# Training & Evaluation
# =============================================================================

def compute_pos_weight(y_train, device):
    """Compute inverse class frequency weights for BCEWithLogitsLoss."""
    pos_counts = y_train.sum(dim=0).float()
    pos_counts = pos_counts.clamp(min=1.0)
    pos_weight = pos_counts.max() / pos_counts
    return pos_weight.to(device)


def train_one_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch over mini-batches."""
    model.train()
    total_loss = 0.0
    total_nodes = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index)
        loss = criterion(out[:batch.batch_size], batch.y[:batch.batch_size])
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.batch_size
        total_nodes += batch.batch_size

    return total_loss / total_nodes


@torch.no_grad()
def evaluate_loader(model, loader, criterion, device):
    """Evaluate model on a data loader, return average loss."""
    model.eval()
    total_loss = 0.0
    total_nodes = 0

    for batch in loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index)
        loss = criterion(out[:batch.batch_size], batch.y[:batch.batch_size])
        total_loss += loss.item() * batch.batch_size
        total_nodes += batch.batch_size

    return total_loss / total_nodes


@torch.no_grad()
def predict_loader(model, loader, device):
    """Collect predictions and labels from a data loader."""
    model.eval()
    all_preds, all_labels = [], []

    for batch in loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index)
        probs = torch.sigmoid(out[:batch.batch_size])
        all_preds.append((probs > 0.5).int().cpu())
        all_labels.append(batch.y[:batch.batch_size].int().cpu())

    return torch.cat(all_preds).numpy(), torch.cat(all_labels).numpy()


def train_one_model(model_name, data, device, args):
    """Train a single GNN model and return test metrics."""
    print(f"\n{'='*60}")
    print(f"Training: {model_name.upper()} (layers={args.num_layers}, hidden={args.hidden_dim}, seed={args.seed})")
    print(f"{'='*60}")

    input_dim = data.x.shape[1]
    output_dim = data.y.shape[1]

    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Initialize model
    ModelClass = MODEL_REGISTRY[model_name]
    if model_name == "gat":
        model = ModelClass(input_dim, args.hidden_dim, output_dim,
                           num_layers=args.num_layers, heads=args.gat_heads, dropout=args.dropout)
    else:
        model = ModelClass(input_dim, args.hidden_dim, output_dim,
                           num_layers=args.num_layers, dropout=args.dropout)
    model = model.to(device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")

    # ---- Mini-batch loaders ----
    num_neighbors = [args.num_neighbors] * args.num_layers

    train_loader = NeighborSampler(
        data, num_neighbors=num_neighbors,
        batch_size=args.batch_size, input_nodes=data.train_mask, shuffle=True,
    )
    val_loader = NeighborSampler(
        data, num_neighbors=num_neighbors,
        batch_size=args.batch_size * 2, input_nodes=data.val_mask, shuffle=False,
    )
    test_loader = NeighborSampler(
        data, num_neighbors=num_neighbors,
        batch_size=args.batch_size * 2, input_nodes=data.test_mask, shuffle=False,
    )

    print(f"Mini-batch: batch_size={args.batch_size}, neighbors={num_neighbors}")
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")

    # Loss, optimizer, scheduler
    y_train = data.y[data.train_mask]
    pos_weight = compute_pos_weight(y_train, device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.t_max)

    # Training loop
    best_val_loss = float("inf")
    patience_counter = 0
    save_path = f"{args.output_dir}/best_{model_name}_model.pth"
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        scheduler.step()
        val_loss = evaluate_loader(model, val_loader, criterion, device)

        if epoch % 10 == 0 or epoch == 1:
            marker = " *" if val_loss < best_val_loss else ""
            print(f"Epoch [{epoch}/{args.epochs}], Train: {train_loss:.4f}, Val: {val_loss:.4f}{marker}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch} (no improvement for {args.patience} epochs)")
                break

    train_time = time.time() - start_time
    print(f"\nTraining complete in {train_time:.1f}s. Best Val Loss: {best_val_loss:.4f}")

    # Evaluate on test set
    model.load_state_dict(torch.load(save_path, weights_only=True))
    test_preds, test_labels = predict_loader(model, test_loader, device)

    go_terms = load_go_terms(args.output_dir)

    print(f"\nClassification Report ({model_name.upper()}):")
    report = classification_report(
        test_labels, test_preds,
        target_names=go_terms if go_terms else None,
        zero_division=0,
    )
    print(report)

    summary = classification_report(
        test_labels, test_preds,
        target_names=go_terms if go_terms else None,
        zero_division=0,
        output_dict=True,
    )

    result = {
        "model": model_name,
        "num_layers": args.num_layers,
        "hidden_dim": args.hidden_dim,
        "seed": args.seed,
        "params": param_count,
        "batch_size": args.batch_size,
        "num_neighbors": args.num_neighbors,
        "train_time_s": round(train_time, 1),
        "best_val_loss": round(best_val_loss, 6),
        "micro_precision": round(summary.get("micro avg", {}).get("precision", 0), 4),
        "micro_recall": round(summary.get("micro avg", {}).get("recall", 0), 4),
        "micro_f1": round(summary.get("micro avg", {}).get("f1-score", 0), 4),
        "macro_f1": round(summary.get("macro avg", {}).get("f1-score", 0), 4),
        "weighted_f1": round(summary.get("weighted avg", {}).get("f1-score", 0), 4),
        "samples_f1": round(summary.get("samples avg", {}).get("f1-score", 0), 4),
    }
    if model_name == "gat":
        result["gat_heads"] = args.gat_heads
    result["dropout"] = args.dropout

    return result


def load_go_terms(output_dir):
    """Load GO term names from saved file."""
    try:
        with open(f"{output_dir}/go_terms.txt") as f:
            return [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        return None


def save_results(results, output_dir):
    """Append results to JSON for manuscript reference."""
    path = f"{output_dir}/gnn_results.json"
    existing = []
    try:
        with open(path) as f:
            existing = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    existing.extend(results)
    with open(path, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"\nResults saved to {path}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train GNN models for multi-label GO prediction")
    parser.add_argument("--model", type=str, default="all", choices=["gat", "gcn", "sage", "all"])
    parser.add_argument("--data-path", type=str, default="../data/pyg_data_cosine_0.99.pt")
    parser.add_argument("--output-dir", type=str, default="../data")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2, help="GNN layers (depth ablation)")
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--gat-heads", type=int, default=8, help="Attention heads (GAT only)")
    parser.add_argument("--t-max", type=int, default=10, help="CosineAnnealingLR T_max")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=512, help="Seed nodes per mini-batch")
    parser.add_argument("--num-neighbors", type=int, default=10, help="Neighbors sampled per layer")
    args = parser.parse_args()

    print(f"Loading data from {args.data_path}")
    data = torch.load(args.data_path, weights_only=False)
    print(f"Data: {data}")
    print(f"Masks: train={data.train_mask.sum()}, val={data.val_mask.sum()}, test={data.test_mask.sum()}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    models_to_train = list(MODEL_REGISTRY.keys()) if args.model == "all" else [args.model]
    results = []

    for model_name in models_to_train:
        result = train_one_model(model_name, data, device, args)
        results.append(result)
        # Save immediately so results survive crashes
        save_results([result], args.output_dir)
        # Free GPU cache between models
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Summary table
    if len(results) > 1:
        print(f"\n{'='*70}")
        print("Summary")
        print(f"{'='*70}")
        print(f"{'Model':>8} {'Layers':>7} {'Time':>8} {'ValLoss':>9} {'microF1':>9} {'macroF1':>9} {'weightF1':>9}")
        print("-" * 70)
        for r in results:
            print(f"{r['model']:>8} {r['num_layers']:>7} {r['train_time_s']:>7.1f}s {r['best_val_loss']:>9.4f} "
                  f"{r['micro_f1']:>9.4f} {r['macro_f1']:>9.4f} {r['weighted_f1']:>9.4f}")

    # Results already saved incrementally above


if __name__ == "__main__":
    main()

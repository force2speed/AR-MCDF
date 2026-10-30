"""
=============================================================================
AR-MCDF  |  Module 3 -- Spatio-Temporal Graph Convolutional Network (ST-GCN)
=============================================================================
Purpose:
    - WindowGraphBuilder    : Converts a (50, 27) time-window tensor into a
                              torch_geometric.data.Data graph via cosine-
                              similarity thresholding OR k-NN edges.
    - SpatioTemporalGCN     : Full ST-GCN model --
                                Spatial  : 2 x GCNConv + BatchNorm + ReLU
                                Temporal : Multi-Head Self-Attention (MHSA)
                                Classify : Global pool + FC head -> logits[2]
    - FocalLoss             : FL(pt) = -alpha_t * (1-pt)^gamma * log(pt)
                              with defaults alpha=0.25, gamma=2.0.

Standalone execution:
    py src/module3_stgcn.py

Output:
    Graph summary, forward pass output probabilities, focal loss value,
    and confirmation that backward() completes (gradient flow verified).
=============================================================================
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Literal, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# PyTorch Geometric
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool
from torch_geometric.utils import add_self_loops, remove_self_loops

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s | %(module)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Typing aliases
# ---------------------------------------------------------------------------
EdgeMethod = Literal["cosine", "knn"]


# ===========================================================================
# 1.  WindowGraphBuilder
# ===========================================================================

class WindowGraphBuilder:
    """
    Converts a (window_size, n_features) time-window tensor into a
    ``torch_geometric.data.Data`` graph object for the ST-GCN.

    Graph semantics
    ---------------
    * **Nodes**  : Each of the ``window_size`` time-steps becomes one node.
                   Its feature vector is the 27-dimensional telemetry row.
    * **Edges**  : Constructed dynamically to capture spatio-temporal
                   correlation between time-steps.  Two methods are available:

        ``"cosine"``
            An edge (i -> j) exists iff ``cosine_similarity(x_i, x_j) >= threshold``.
            Self-loops are added automatically.

        ``"knn"``
            Each node connects to its ``k`` nearest neighbours in feature
            space (L2 distance). Self-loops are added automatically.

    Args:
        method    : Edge construction strategy ("cosine" or "knn").
        threshold : Cosine similarity cut-off (used when method="cosine").
        k         : Number of neighbours per node (used when method="knn").
        add_self_loop : Whether to append self-loops (always recommended for GCN).
    """

    def __init__(
        self,
        method: EdgeMethod = "cosine",
        threshold: float = 0.6,
        k: int = 4,
        add_self_loop: bool = True,
    ) -> None:
        self.method        = method
        self.threshold     = threshold
        self.k             = k
        self.add_self_loop = add_self_loop

    # ------------------------------------------------------------------
    def _cosine_edges(self, x: Tensor) -> Tensor:
        """
        Build edge_index using cosine similarity >= self.threshold.

        Args:
            x: Tensor (N, F) -- node feature matrix.

        Returns:
            edge_index: LongTensor (2, E).
        """
        # Normalise rows; guard against zero-norm rows
        norms = x.norm(dim=1, keepdim=True).clamp(min=1e-8)
        x_norm = x / norms                              # (N, F)

        sim = x_norm @ x_norm.T                         # (N, N)  cosine sim

        # Mask lower triangle + diagonal for undirected, then add both dirs
        mask = sim >= self.threshold
        src, dst = mask.nonzero(as_tuple=True)

        # Keep both directions (undirected graph)
        edge_index = torch.stack([
            torch.cat([src, dst]),
            torch.cat([dst, src]),
        ], dim=0)                                        # (2, E_undirected)

        return edge_index

    # ------------------------------------------------------------------
    def _knn_edges(self, x: Tensor) -> Tensor:
        """
        Build edge_index via k-nearest-neighbours in L2 feature space.

        Args:
            x: Tensor (N, F).

        Returns:
            edge_index: LongTensor (2, E).
        """
        N = x.shape[0]
        k = min(self.k, N - 1)

        # Pairwise L2-squared distances  (N, N)
        dist = torch.cdist(x, x, p=2)
        # Zero out diagonal so a node is not its own neighbour
        dist.fill_diagonal_(float("inf"))

        # Top-k smallest distances -> indices
        _, knn_idx = dist.topk(k, dim=1, largest=False)   # (N, k)

        src = torch.arange(N, device=x.device).unsqueeze(1).expand(-1, k).reshape(-1)
        dst = knn_idx.reshape(-1)

        edge_index = torch.stack([
            torch.cat([src, dst]),
            torch.cat([dst, src]),
        ], dim=0)                                          # (2, 2*N*k)

        return edge_index

    # ------------------------------------------------------------------
    def build(
        self,
        window: Tensor | np.ndarray,
        label: Optional[int] = None,
    ) -> Data:
        """
        Convert a single time-window into a PyG Data object.

        Args:
            window : Float array/tensor of shape (window_size, n_features).
            label  : Optional integer graph-level label (0=normal, 1=attack).

        Returns:
            data : ``torch_geometric.data.Data`` with attributes:
                     ``x``          -- node features (N, F)
                     ``edge_index`` -- connectivity  (2, E)
                     ``y``          -- label scalar tensor  [optional]
                     ``num_nodes``  -- N
        """
        if isinstance(window, np.ndarray):
            x = torch.tensor(window, dtype=torch.float32)
        else:
            x = window.float()

        N = x.shape[0]

        # Build edges
        if self.method == "cosine":
            edge_index = self._cosine_edges(x)
        else:
            edge_index = self._knn_edges(x)

        # Add self-loops (required by GCNConv for proper normalisation)
        if self.add_self_loop:
            edge_index, _ = remove_self_loops(edge_index)
            edge_index, _ = add_self_loops(edge_index, num_nodes=N)

        # Deduplicate edges
        edge_index = torch.unique(edge_index, dim=1)

        data = Data(x=x, edge_index=edge_index, num_nodes=N)
        if label is not None:
            data.y = torch.tensor([label], dtype=torch.long)

        logger.debug(
            "Graph built | nodes=%d | edges=%d | method=%s",
            N, edge_index.shape[1], self.method,
        )
        return data

    # ------------------------------------------------------------------
    def build_batch(
        self,
        windows: list[Tensor | np.ndarray],
        labels: Optional[list[int]] = None,
    ) -> Batch:
        """
        Build a PyG mini-batch from a list of time-windows.

        Args:
            windows : List of arrays/tensors, each (window_size, n_features).
            labels  : Optional list of integer labels, one per window.

        Returns:
            batch : ``torch_geometric.data.Batch`` ready for the ST-GCN.
        """
        graphs = []
        for i, w in enumerate(windows):
            lbl = labels[i] if labels is not None else None
            graphs.append(self.build(w, label=lbl))
        return Batch.from_data_list(graphs)


# ===========================================================================
# 2.  Temporal Self-Attention Block
# ===========================================================================

class TemporalSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention applied along the node (temporal) dimension.

    After the spatial GCN layers produce node embeddings of shape
    (N_total, hidden_dim), we reshape back into (batch, N_per_graph, hidden_dim)
    so that attention operates across the N time-step nodes within each graph.

    Args:
        embed_dim  : Dimension of input node embeddings.
        num_heads  : Number of attention heads (default 4).
        dropout    : Attention dropout probability.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,   # expects (batch, seq, dim)
        )
        self.norm    = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,          # (batch, N, embed_dim)  -- dense per-graph
    ) -> Tensor:
        """
        Args:
            x: Tensor (B, N, D) -- batch of node embedding sequences.

        Returns:
            out: Tensor (B, N, D) -- attention-refined node embeddings.
        """
        residual = x
        attn_out, _ = self.attn(x, x, x)          # self-attention
        out = self.norm(residual + self.dropout(attn_out))
        return out


# ===========================================================================
# 3.  SpatioTemporalGCN
# ===========================================================================

class SpatioTemporalGCN(nn.Module):
    """
    Full Spatio-Temporal Graph Convolutional Network for anomaly classification.

    Architecture
    ------------
    Spatial block (2 x GCNConv):
        GCNConv(n_features -> gcn_hidden)  + BN + ReLU
        GCNConv(gcn_hidden  -> gcn_hidden)  + BN + ReLU

    Temporal attention block:
        TemporalSelfAttention(gcn_hidden, num_heads)

    Pooling + Classification head:
        GlobalMeanPool  [+ GlobalMaxPool concatenated]
        FC(pool_dim -> fc_hidden) + ReLU + Dropout
        FC(fc_hidden  -> n_classes)

    Args:
        n_features  : Input feature dimension per node (default 27).
        gcn_hidden  : GCN hidden + output channel width (default 128).
        attn_heads  : Multi-head attention heads (default 4).
        fc_hidden   : Fully-connected classifier hidden width (default 64).
        n_classes   : Output classes (default 2: normal / attack).
        dropout     : Dropout probability in FC head (default 0.4).
        pool        : "mean", "max", or "both" -- global pooling strategy.
    """

    def __init__(
        self,
        n_features: int  = 27,
        gcn_hidden: int  = 128,
        attn_heads: int  = 4,
        fc_hidden: int   = 64,
        n_classes: int   = 2,
        dropout: float   = 0.4,
        pool: Literal["mean", "max", "both"] = "both",
    ) -> None:
        super().__init__()
        self.pool_mode = pool

        # ---- Spatial: GCN layers ----------------------------------------
        self.gcn1    = GCNConv(n_features, gcn_hidden)
        self.bn1     = nn.BatchNorm1d(gcn_hidden)
        self.gcn2    = GCNConv(gcn_hidden, gcn_hidden)
        self.bn2     = nn.BatchNorm1d(gcn_hidden)

        # ---- Temporal: Multi-Head Self-Attention -------------------------
        self.attn_block = TemporalSelfAttention(
            embed_dim=gcn_hidden,
            num_heads=attn_heads,
            dropout=0.1,
        )

        # ---- Pooling dimension -------------------------------------------
        pool_out_dim = gcn_hidden * 2 if pool == "both" else gcn_hidden

        # ---- Classification Head -----------------------------------------
        self.classifier = nn.Sequential(
            nn.Linear(pool_out_dim, fc_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, n_classes),
        )

    # ------------------------------------------------------------------
    def forward(self, data: Data | Batch) -> Tensor:
        """
        Full forward pass.

        Args:
            data: PyG Data or Batch with attributes:
                  ``x``          (N_total, n_features)
                  ``edge_index`` (2, E)
                  ``batch``      (N_total,)  -- graph assignment vector
                                              (auto-created by Batch)

        Returns:
            logits: Tensor (batch_size, n_classes).
        """
        x, edge_index = data.x, data.edge_index

        # Batch vector: all zeros for a single (unbatched) graph
        batch_vec = data.batch if hasattr(data, "batch") and data.batch is not None \
                    else torch.zeros(x.shape[0], dtype=torch.long, device=x.device)

        # ---- Spatial GCN block ------------------------------------------
        x = self.gcn1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.gcn2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)                               # (N_total, gcn_hidden)

        # ---- Temporal Attention -----------------------------------------
        # Reshape sparse (N_total, D) -> dense (B, N_per_graph, D)
        # using to_dense_batch (handles variable-size graphs)
        from torch_geometric.utils import to_dense_batch
        x_dense, mask = to_dense_batch(x, batch_vec)   # (B, N_max, D)

        x_dense = self.attn_block(x_dense)              # (B, N_max, D)

        # Flatten back to sparse for global pooling
        # (only keep the valid (non-padded) node positions)
        x_sparse = x_dense[mask]                        # (N_total, D)

        # ---- Global Pooling ----------------------------------------------
        if self.pool_mode == "mean":
            pooled = global_mean_pool(x_sparse, batch_vec)
        elif self.pool_mode == "max":
            pooled = global_max_pool(x_sparse, batch_vec)
        else:  # "both"
            pooled = torch.cat([
                global_mean_pool(x_sparse, batch_vec),
                global_max_pool(x_sparse, batch_vec),
            ], dim=1)                                   # (B, 2*D)

        # ---- Classifier Head --------------------------------------------
        logits = self.classifier(pooled)                # (B, n_classes)
        return logits


# ===========================================================================
# 4.  Focal Loss
# ===========================================================================

class FocalLoss(nn.Module):
    """
    Sigmoid-based Focal Loss for binary or multi-class classification.

    Equation (per Lin et al., 2017):
        FL(pt) = -alpha_t * (1 - pt)^gamma * log(pt)

    where:
        pt      = softmax probability of the true class
        alpha_t = class-balancing weight for the true class
        gamma   = focusing exponent (down-weights easy examples)

    This implementation supports an ``alpha`` tensor (one weight per class)
    or a scalar (applied uniformly to all classes).

    Args:
        alpha     : Class weight(s).  Use a list/tensor of length ``n_classes``
                    for per-class alpha, or a float for uniform weighting.
                    Default [0.25, 0.75] weights minority (attack) class higher.
        gamma     : Focusing exponent >= 0 (default 2.0).
        reduction : "mean" | "sum" | "none".
    """

    def __init__(
        self,
        alpha: float | list[float] = 0.25,
        gamma: float = 2.0,
        reduction: Literal["mean", "sum", "none"] = "mean",
    ) -> None:
        super().__init__()
        self.gamma     = gamma
        self.reduction = reduction

        # Register alpha as a buffer so it moves with .to(device)
        if isinstance(alpha, (int, float)):
            self.register_buffer("alpha", torch.tensor([alpha, 1.0 - alpha]))
        else:
            self.register_buffer("alpha", torch.tensor(alpha, dtype=torch.float32))

    # ------------------------------------------------------------------
    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        """
        Compute Focal Loss.

        Args:
            logits  : Raw (un-normalised) model output, shape (B, C).
            targets : Ground-truth class indices, shape (B,).

        Returns:
            Scalar loss (if reduction != "none") or per-sample losses (B,).
        """
        # Cross-entropy gives log(pt) per sample for the true class
        # F.cross_entropy = -log(softmax(logits)[target])
        log_pt = -F.cross_entropy(logits, targets, reduction="none")  # (B,)
        pt     = log_pt.exp()                                          # (B,)

        # Gather alpha for each sample's true class
        alpha_t = self.alpha[targets]                                  # (B,)

        # Focal weight: (1 - pt)^gamma
        focal_weight = (1.0 - pt).pow(self.gamma)

        # Final focal loss
        loss = -alpha_t * focal_weight * log_pt                       # (B,)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


# ===========================================================================
# 5.  Public pipeline helper
# ===========================================================================

def run_stgcn(
    windows: list,
    labels: list,
    n_features: int   = 27,
    gcn_hidden: int   = 128,
    attn_heads: int   = 4,
    edge_method: EdgeMethod = "cosine",
    threshold: float  = 0.6,
    k_neighbors: int  = 4,
    gamma: float      = 2.0,
    alpha: float      = 0.25,
    random_seed: int  = 42,
) -> Tuple[SpatioTemporalGCN, WindowGraphBuilder, FocalLoss]:
    """
    Convenience initialiser called by Module 5's end-to-end pipeline.

    Instantiates and returns the three key objects; the pipeline loop
    handles the actual training iterations.

    Args:
        windows      : List of np.ndarray windows (window_size, n_features).
        labels       : List of per-window majority-vote integer labels.
        n_features   : Feature width.
        gcn_hidden   : GCN hidden channel count.
        attn_heads   : Number of MHSA heads.
        edge_method  : "cosine" or "knn".
        threshold    : Cosine similarity threshold.
        k_neighbors  : k for k-NN edges.
        gamma        : Focal loss gamma.
        alpha        : Focal loss alpha.
        random_seed  : RNG seed.

    Returns:
        Tuple of (SpatioTemporalGCN, WindowGraphBuilder, FocalLoss).
    """
    torch.manual_seed(random_seed)

    graph_builder = WindowGraphBuilder(
        method=edge_method, threshold=threshold, k=k_neighbors
    )
    model = SpatioTemporalGCN(
        n_features=n_features,
        gcn_hidden=gcn_hidden,
        attn_heads=attn_heads,
    )
    criterion = FocalLoss(alpha=alpha, gamma=gamma)

    return model, graph_builder, criterion


# ===========================================================================
# 6.  Standalone Demo / Test Block
# ===========================================================================

if __name__ == "__main__":
    # ------------------------------------------------------------------
    # Path setup: make src/ importable and CWD = project root
    # ------------------------------------------------------------------
    _src = Path(__file__).resolve().parent
    _root = _src.parent
    if str(_src) not in sys.path:
        sys.path.insert(0, str(_src))
    os.chdir(_root)

    from module1_ingestion import run_ingestion   # noqa: E402

    torch.manual_seed(42)

    SEP = "=" * 68
    logger.info(SEP)
    logger.info("AR-MCDF  |  Module 3 -- Standalone Demo")
    logger.info(SEP)

    # ------------------------------------------------------------------
    # A. Ingest -- grab one (50, 27) window
    # ------------------------------------------------------------------
    logger.info("Step A: Ingesting telemetry via Module 1...")
    windows, labels, _ = run_ingestion(
        csv_path="data/raw/UNSW_NB15_training-set.csv",
        window_size=50,
        stride=25,
    )



    WIN_IDX = 42
    x_np    = windows[WIN_IDX]                          # (50, 27) float32
    y_np    = labels[WIN_IDX]                           # (50,) int64
    majority_label = int(y_np.mean() >= 0.5)

    logger.info(
        "Window %d | shape: %s | majority label: %d (%s)",
        WIN_IDX, x_np.shape, majority_label,
        "attack" if majority_label else "normal",
    )

    # ------------------------------------------------------------------
    # B. Build graph with cosine similarity edges
    # ------------------------------------------------------------------
    logger.info("Step B: Building dynamic graph (cosine threshold=0.6)...")
    builder = WindowGraphBuilder(method="cosine", threshold=0.6)
    graph   = builder.build(x_np, label=majority_label)

    print(f"\n{SEP}")
    print("  GRAPH SUMMARY (cosine method, threshold=0.6)")
    print(SEP)
    print(f"  Nodes (time steps)  : {graph.num_nodes}")
    print(f"  Node features (dim) : {graph.x.shape[1]}")
    print(f"  Edges (after dedup) : {graph.edge_index.shape[1]}")
    print(f"  Edge density        : {graph.edge_index.shape[1] / (graph.num_nodes**2):.4f}")
    print(f"  Graph label         : {graph.y.item()} "
          f"({'attack' if graph.y.item() else 'normal'})")

    # Also show k-NN variant for comparison
    logger.info("Step B2: Building k-NN graph (k=4) for comparison...")
    builder_knn = WindowGraphBuilder(method="knn", k=4)
    graph_knn   = builder_knn.build(x_np, label=majority_label)
    print(f"\n  k-NN (k=4) edge count : {graph_knn.edge_index.shape[1]}")
    print(SEP)

    # ------------------------------------------------------------------
    # C. Instantiate ST-GCN and Focal Loss
    # ------------------------------------------------------------------
    logger.info("Step C: Initialising SpatioTemporalGCN and FocalLoss...")
    model     = SpatioTemporalGCN(
        n_features=27,
        gcn_hidden=128,
        attn_heads=4,
        fc_hidden=64,
        n_classes=2,
        dropout=0.4,
        pool="both",
    )
    criterion = FocalLoss(alpha=0.25, gamma=2.0, reduction="mean")

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        "SpatioTemporalGCN | total params: %d | trainable: %d",
        total_params, trainable_params,
    )

    # ------------------------------------------------------------------
    # D. Forward pass (single graph)
    # ------------------------------------------------------------------
    logger.info("Step D: Forward pass through ST-GCN...")
    model.train()   # keep BN/dropout in train mode for gradient check

    logits = model(graph)                                # (1, 2)
    probs  = F.softmax(logits, dim=1)                   # (1, 2)

    target = graph.y                                     # (1,)
    loss   = criterion(logits, target)

    logger.info("Forward pass complete | logits=%s", logits.detach().numpy().round(4))

    # ------------------------------------------------------------------
    # E. Backward pass (gradient flow check)
    # ------------------------------------------------------------------
    logger.info("Step E: Running loss.backward() to verify gradient flow...")
    loss.backward()

    # Collect gradient norms from all parameter groups
    grad_norms = {
        name: p.grad.norm().item()
        for name, p in model.named_parameters()
        if p.grad is not None
    }
    has_gradients = all(v > 0 for v in grad_norms.values())

    # ------------------------------------------------------------------
    # F. Final Report
    # ------------------------------------------------------------------
    pred_class = probs.argmax(dim=1).item()
    print(f"\n{SEP}")
    print("  MODULE 3 -- ST-GCN DEMO RESULTS")
    print(SEP)
    print(f"  Window index              : {WIN_IDX}")
    print(f"  Majority label            : {majority_label} "
          f"({'attack' if majority_label else 'normal'})")
    print(f"\n  --- Graph Statistics ---")
    print(f"  Nodes (cosine graph)      : {graph.num_nodes}")
    print(f"  Edges (cosine graph)      : {graph.edge_index.shape[1]}")
    print(f"  Nodes (knn graph)         : {graph_knn.num_nodes}")
    print(f"  Edges (knn graph)         : {graph_knn.edge_index.shape[1]}")
    print(f"\n  --- Model Architecture ---")
    print(f"  GCN hidden dim            : 128")
    print(f"  Attention heads           : 4")
    print(f"  Pooling strategy          : both (mean + max concat)")
    print(f"  Total parameters          : {total_params:,}")
    print(f"\n  --- Forward Pass Output ---")
    print(f"  Raw logits                : {logits.detach().numpy().round(4).tolist()}")
    print(f"  Softmax probs (normal/atk): {probs.detach().numpy().round(4).tolist()}")
    print(f"  Predicted class           : {pred_class} "
          f"({'attack' if pred_class else 'normal'})")
    print(f"\n  --- Focal Loss ---")
    print(f"  alpha=0.25  gamma=2.0")
    print(f"  FL value                  : {loss.item():.6f}")
    print(f"\n  --- Gradient Flow ---")
    print(f"  loss.backward() OK        : {has_gradients}")
    print(f"  Layers with gradients     : {len(grad_norms)}")
    # Show a few representative gradient norms
    sample_layers = list(grad_norms.items())[:6]
    for name, gnorm in sample_layers:
        print(f"    {name:40s} grad_norm = {gnorm:.6f}")
    print(SEP)
    print("[OK]  Module 3 completed successfully.")
    print(SEP + "\n")

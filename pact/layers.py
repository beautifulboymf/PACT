"""Hand-rolled GNN layers.

Why hand-rolled: empirically, torch_geometric.nn.GATConv / GCNConv underperform
on the WSDM2020 BlogCatalog/Flickr setup and on the PACT semi-synthetic
datasets. Implementing the layers as plain matmul against a precomputed
normalized sparse adjacency reproduces the WSDM reference behavior and gives us
control over initialization.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# GCN layer (pygcn-style)
# ---------------------------------------------------------------------------


class GraphConvolution(nn.Module):
    """A pygcn-style GCN layer: H' = sigma(A_hat @ H @ W).

    Expects ``adj`` to be a torch sparse tensor that is already
    symmetric-normalized with self-loops (D^-1/2 (A+I) D^-1/2).
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Glorot init — same as pygcn reference, which we know works well here.
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        support = x @ self.weight
        if adj.is_sparse:
            out = torch.sparse.mm(adj, support)
        else:
            out = adj @ support
        if self.bias is not None:
            out = out + self.bias
        return out


class GCN(nn.Module):
    """Stack of GCN layers with ReLU + dropout between layers."""

    def __init__(self, in_dim: int, hidden_dims: list[int], dropout: float = 0.0):
        super().__init__()
        dims = [in_dim] + list(hidden_dims)
        self.layers = nn.ModuleList(
            [GraphConvolution(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
        )
        self.dropout = dropout

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h, adj)
            if i < len(self.layers) - 1:
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
        return h


# ---------------------------------------------------------------------------
# GAT layer (Velickovic-style, hand-rolled)
# ---------------------------------------------------------------------------


class GATLayer(nn.Module):
    """Multi-head graph attention layer.

    Uses a dense edge enumeration (sparse adjacency -> edge index) so we can
    score every edge with a learnable LeakyReLU(a^T [Wh_v || Wh_u]) without
    relying on PyG. Works fine for graphs up to ~50k nodes; for larger graphs
    you should switch to PyG GATConv (and accept the perf hit) or batch.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        heads: int = 4,
        dropout: float = 0.0,
        concat: bool = True,
        leaky_slope: float = 0.2,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.heads = heads
        self.concat = concat
        self.dropout = dropout
        self.leaky_slope = leaky_slope

        self.W = nn.Parameter(torch.empty(in_dim, heads * out_dim))
        # a in the paper is split into a_src and a_dst (Velickovic et al.).
        self.a_src = nn.Parameter(torch.empty(1, heads, out_dim))
        self.a_dst = nn.Parameter(torch.empty(1, heads, out_dim))
        self.bias = nn.Parameter(torch.empty(heads * out_dim if concat else out_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.W, gain=1.414)
        nn.init.xavier_uniform_(self.a_src, gain=1.414)
        nn.init.xavier_uniform_(self.a_dst, gain=1.414)
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """x: [N, in_dim]; edge_index: LongTensor [2, E] (src, dst)."""
        N = x.size(0)
        H, C = self.heads, self.out_dim

        h = (x @ self.W).view(N, H, C)  # [N, H, C]

        src, dst = edge_index[0], edge_index[1]
        # alpha_e = LeakyReLU(a_src^T h_src + a_dst^T h_dst)  per head
        alpha_src = (h * self.a_src).sum(dim=-1)  # [N, H]
        alpha_dst = (h * self.a_dst).sum(dim=-1)  # [N, H]
        alpha = alpha_src[src] + alpha_dst[dst]   # [E, H]
        alpha = F.leaky_relu(alpha, negative_slope=self.leaky_slope)

        # softmax over incoming edges per dst node, per head
        alpha = _edge_softmax(alpha, dst, N)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        # weighted aggregation: out[dst] += alpha[e] * h[src[e]]
        msg = h[src] * alpha.unsqueeze(-1)  # [E, H, C]
        out = torch.zeros(N, H, C, device=x.device, dtype=x.dtype)
        out.index_add_(0, dst, msg)

        if self.concat:
            out = out.reshape(N, H * C)
        else:
            out = out.mean(dim=1)
        out = out + self.bias
        return out


def _edge_softmax(scores: torch.Tensor, dst: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Softmax of `scores` (shape [E, H]) grouped by `dst` (shape [E])."""
    # numerical stability: subtract per-dst max
    H = scores.size(1)
    max_per_dst = torch.full((num_nodes, H), float("-inf"), device=scores.device, dtype=scores.dtype)
    max_per_dst.scatter_reduce_(0, dst.unsqueeze(-1).expand(-1, H), scores, reduce="amax", include_self=True)
    max_per_dst = torch.where(torch.isinf(max_per_dst), torch.zeros_like(max_per_dst), max_per_dst)
    scores = scores - max_per_dst[dst]
    exp_scores = scores.exp()
    sum_per_dst = torch.zeros(num_nodes, H, device=scores.device, dtype=scores.dtype)
    sum_per_dst.index_add_(0, dst, exp_scores)
    sum_per_dst = sum_per_dst.clamp_min(1e-16)
    return exp_scores / sum_per_dst[dst]


class GAT(nn.Module):
    """Stack of GAT layers with ELU activation between layers, residual optional."""

    def __init__(
        self,
        in_dim: int,
        hidden_dims: list[int],
        heads: int = 4,
        dropout: float = 0.0,
        residual: bool = True,
    ):
        super().__init__()
        self.residual = residual
        self.layers = nn.ModuleList()
        self.res_proj = nn.ModuleList()
        prev = in_dim
        for h_dim in hidden_dims:
            self.layers.append(
                GATLayer(prev, h_dim, heads=heads, dropout=dropout, concat=True)
            )
            out = heads * h_dim
            if residual:
                self.res_proj.append(
                    nn.Identity() if prev == out else nn.Linear(prev, out, bias=False)
                )
            prev = out
        self.out_dim = prev
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = x
        for i, layer in enumerate(self.layers):
            new_h = layer(h, edge_index)
            if self.residual:
                new_h = new_h + self.res_proj[i](h)
            new_h = F.elu(new_h)
            new_h = F.dropout(new_h, p=self.dropout, training=self.training)
            h = new_h
        return h


def sparse_adj_to_edge_index(adj: torch.Tensor) -> torch.Tensor:
    """Convert a torch sparse adjacency to an edge_index LongTensor [2, E]."""
    if not adj.is_sparse:
        adj = adj.to_sparse()
    return adj.coalesce().indices()

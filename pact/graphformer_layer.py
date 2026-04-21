"""Sparse Graph Transformer layer with degree centrality encoding.

Implements a Graphormer-inspired backbone that uses sparse attention over
edge_index (not dense N×N), making it feasible for graphs up to ~50K nodes.

Key differences from GAT (in layers.py):
  - Scaled dot-product attention (not additive LeakyReLU)
  - Degree centrality bias in attention scores (Graphormer's positional signal)
  - Pre-norm transformer block: LayerNorm → Attention → Residual → LayerNorm → FFN → Residual

This backbone is position-aware by design, enabling a controlled test:
  Does GPE add value even when the backbone already has positional information?
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import _edge_softmax


class SparseGraphTransformerLayer(nn.Module):
    """Single transformer layer with sparse (edge-based) dot-product attention."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        heads: int = 4,
        dropout: float = 0.0,
        max_degree: int = 512,
    ):
        super().__init__()
        self.heads = heads
        self.out_dim = out_dim
        self.max_degree = max_degree
        self.dropout = dropout

        total = heads * out_dim

        self.W_Q = nn.Linear(in_dim, total, bias=False)
        self.W_K = nn.Linear(in_dim, total, bias=False)
        self.W_V = nn.Linear(in_dim, total, bias=False)
        self.W_O = nn.Linear(total, total)

        # Graphormer-style degree centrality encoding: learned bias per head
        self.degree_enc_src = nn.Embedding(max_degree, heads)
        self.degree_enc_dst = nn.Embedding(max_degree, heads)

        self.norm1 = nn.LayerNorm(in_dim)
        self.norm2 = nn.LayerNorm(total)

        self.ffn = nn.Sequential(
            nn.Linear(total, 4 * total),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * total, total),
            nn.Dropout(dropout),
        )

        # Residual projection when in_dim != total
        if in_dim != total:
            self.res_proj = nn.Linear(in_dim, total, bias=False)
        else:
            self.res_proj = nn.Identity()

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.W_Q.weight, gain=1.0)
        nn.init.xavier_uniform_(self.W_K.weight, gain=1.0)
        nn.init.xavier_uniform_(self.W_V.weight, gain=1.0)
        nn.init.xavier_uniform_(self.W_O.weight, gain=1.0)
        nn.init.zeros_(self.W_O.bias)
        nn.init.zeros_(self.degree_enc_src.weight)
        nn.init.zeros_(self.degree_enc_dst.weight)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        degree: torch.Tensor,
    ) -> torch.Tensor:
        """x: [N, in_dim]; edge_index: [2, E]; degree: [N] (long)."""
        N = x.size(0)
        H, C = self.heads, self.out_dim

        # Pre-norm
        x_norm = self.norm1(x)

        Q = self.W_Q(x_norm).view(N, H, C)
        K = self.W_K(x_norm).view(N, H, C)
        V = self.W_V(x_norm).view(N, H, C)

        src, dst = edge_index[0], edge_index[1]

        # Scaled dot-product attention over edges
        alpha = (Q[dst] * K[src]).sum(dim=-1) / math.sqrt(C)  # [E, H]

        # Degree centrality bias (Graphormer's key positional signal)
        deg = degree.clamp(max=self.max_degree - 1)
        alpha = alpha + self.degree_enc_src(deg[src]) + self.degree_enc_dst(deg[dst])

        # Edge softmax + dropout
        alpha = _edge_softmax(alpha, dst, N)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        # Weighted aggregation
        msg = V[src] * alpha.unsqueeze(-1)  # [E, H, C]
        out = torch.zeros(N, H, C, device=x.device, dtype=x.dtype)
        out.index_add_(0, dst, msg)
        out = self.W_O(out.reshape(N, H * C))

        # Residual connection
        out = out + self.res_proj(x)

        # FFN block with pre-norm + residual
        out = out + self.ffn(self.norm2(out))

        return out


class SparseGraphTransformer(nn.Module):
    """Stack of SparseGraphTransformerLayers.

    Interface matches GAT: __init__(in_dim, hidden_dims, heads, dropout)
    and forward(x, edge_index) -> [N, out_dim].
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dims: list[int],
        heads: int = 4,
        dropout: float = 0.0,
        max_degree: int = 512,
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        prev = in_dim
        for h_dim in hidden_dims:
            self.layers.append(
                SparseGraphTransformerLayer(
                    prev, h_dim, heads=heads, dropout=dropout, max_degree=max_degree,
                )
            )
            prev = heads * h_dim
        self.out_dim = prev

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # Compute node degree from edge_index
        N = x.size(0)
        degree = torch.zeros(N, dtype=torch.long, device=x.device)
        ones = torch.ones(edge_index.size(1), dtype=torch.long, device=x.device)
        degree.index_add_(0, edge_index[1], ones)

        h = x
        for layer in self.layers:
            h = layer(h, edge_index, degree)
        return h

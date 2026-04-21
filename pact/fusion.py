"""Graph Positional Encoding (GPE) fusion module — PACT paper §4.1.1.

Multi-head attention with Q drawn from node features `h` and K/V drawn from
positional encoding `p` (e.g. node2vec). Output is a fused representation that
is fed into the GNN backbone.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class GPEFusion(nn.Module):
    def __init__(
        self,
        feat_dim: int,
        pos_dim: int,
        embed_dim: int,
        heads: int = 4,
        dropout: float = 0.0,
        out_mode: str = "concat",  # 'concat' -> [h ; attn_out]; 'add' -> h + attn_out (requires equal dims)
    ):
        super().__init__()
        assert embed_dim % heads == 0, "embed_dim must be divisible by heads"
        self.heads = heads
        self.head_dim = embed_dim // heads
        self.embed_dim = embed_dim
        self.out_mode = out_mode

        self.q_proj = nn.Linear(feat_dim, embed_dim)
        self.k_proj = nn.Linear(pos_dim, embed_dim)
        self.v_proj = nn.Linear(pos_dim, embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

        if out_mode == "concat":
            self.out_dim = feat_dim + embed_dim
            self.norm = nn.LayerNorm(self.out_dim)
            # learnable residual that maps the raw [h ; p] -> output dim
            self.residual = nn.Linear(feat_dim + pos_dim, self.out_dim)
        elif out_mode == "add":
            assert feat_dim == embed_dim, "for add mode feat_dim must equal embed_dim"
            self.out_dim = feat_dim
            self.norm = nn.LayerNorm(self.out_dim)
            self.residual = nn.Identity()
        else:
            raise ValueError(out_mode)

    def forward(self, h: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        N = h.size(0)
        Q = self.q_proj(h).view(N, self.heads, self.head_dim)
        K = self.k_proj(p).view(N, self.heads, self.head_dim)
        V = self.v_proj(p).view(N, self.heads, self.head_dim)

        # Self-attention across all N nodes (paper does this on the full node set
        # of the subgraph). For very large graphs the user is expected to pass a
        # subgraph batch — keep N modest here.
        scores = torch.einsum("nhd,mhd->nhm", Q, K) / math.sqrt(self.head_dim)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        ctx = torch.einsum("nhm,mhd->nhd", attn, V)
        ctx = ctx.reshape(N, self.embed_dim)
        ctx = self.o_proj(ctx)

        if self.out_mode == "concat":
            fused = torch.cat([h, ctx], dim=-1)
            res = self.residual(torch.cat([h, p], dim=-1))
            out = self.norm(F.elu(fused + res))
        else:
            out = self.norm(F.elu(h + ctx))
        return out

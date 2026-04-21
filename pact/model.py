"""PACT model assembly: GPE fusion -> GAT/GCN backbone -> X-learner heads."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from .fusion import GPEFusion
from .heads import XLearnerHeads, XLearnerOutput
from .graphformer_layer import SparseGraphTransformer
from .layers import GAT, GCN


@dataclass
class PACTConfig:
    in_dim: int
    pos_dim: int = 128                # node2vec embedding dim
    fusion_embed_dim: int = 256       # multi-head attention embed dim
    fusion_heads: int = 4
    use_gpe: bool = True              # ablation switch: drop the fusion entirely
    backbone: str = "gat"             # 'gat', 'gcn', or 'graphformer'
    gnn_hidden: tuple[int, ...] = (256, 128, 128, 128, 128)
    gnn_heads: int = 4                # only used if backbone == 'gat'
    mlp_hidden: tuple[int, ...] = (128, 128, 128, 128)
    dropout: float = 0.0
    use_variance: bool = True         # PACT if True, GAT+X if False

    def __post_init__(self):
        self.gnn_hidden = tuple(self.gnn_hidden)
        self.mlp_hidden = tuple(self.mlp_hidden)


class PACT(nn.Module):
    def __init__(self, cfg: PACTConfig):
        super().__init__()
        self.cfg = cfg

        # Fusion module (skipped if use_gpe=False; backbone takes raw features).
        if cfg.use_gpe:
            self.fusion: nn.Module = GPEFusion(
                feat_dim=cfg.in_dim,
                pos_dim=cfg.pos_dim,
                embed_dim=cfg.fusion_embed_dim,
                heads=cfg.fusion_heads,
                dropout=cfg.dropout,
                out_mode="concat",
            )
            backbone_in = self.fusion.out_dim
        else:
            self.fusion = nn.Identity()
            backbone_in = cfg.in_dim

        # Backbone: GAT (default) or GCN.
        if cfg.backbone == "gat":
            self.backbone: nn.Module = GAT(
                in_dim=backbone_in,
                hidden_dims=list(cfg.gnn_hidden),
                heads=cfg.gnn_heads,
                dropout=cfg.dropout,
                residual=True,
            )
            rep_dim = self.backbone.out_dim
        elif cfg.backbone == "gcn":
            self.backbone = GCN(
                in_dim=backbone_in,
                hidden_dims=list(cfg.gnn_hidden),
                dropout=cfg.dropout,
            )
            rep_dim = cfg.gnn_hidden[-1]
        elif cfg.backbone == "graphformer":
            self.backbone = SparseGraphTransformer(
                in_dim=backbone_in,
                hidden_dims=list(cfg.gnn_hidden),
                heads=cfg.gnn_heads,
                dropout=cfg.dropout,
            )
            rep_dim = self.backbone.out_dim
        else:
            raise ValueError(cfg.backbone)

        self.heads = XLearnerHeads(
            rep_dim=rep_dim,
            hidden_dims=list(cfg.mlp_hidden),
            dropout=cfg.dropout,
            use_variance=cfg.use_variance,
        )

    def representation(
        self,
        x: torch.Tensor,
        graph: torch.Tensor,
        pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.cfg.use_gpe:
            assert pos is not None, "use_gpe=True but no positional encoding given"
            h = self.fusion(x, pos)
        else:
            h = x
        # GAT expects edge_index [2, E]; GCN expects normalized sparse adjacency.
        return self.backbone(h, graph)

    def forward(
        self,
        x: torch.Tensor,
        graph: torch.Tensor,
        pos: Optional[torch.Tensor] = None,
    ) -> XLearnerOutput:
        rep = self.representation(x, graph, pos)
        return self.heads(rep)


# ---------------------------------------------------------------------------
# Graph baselines with optional GPE plug-in.
# BNN (S-learner) and TARNet (T-learner) share the same GPE+GAT backbone
# as PACT but use simpler heads. This proves GPE is a universal enhancer.
# ---------------------------------------------------------------------------


class GraphBaseline(nn.Module):
    """Shared GPE+GAT backbone for BNN / TARNet graph baselines.

    Subclass sets ``self.heads_forward(rep) -> XLearnerOutput``.
    """

    def __init__(self, cfg: PACTConfig):
        super().__init__()
        self.cfg = cfg
        if cfg.use_gpe:
            self.fusion: nn.Module = GPEFusion(
                feat_dim=cfg.in_dim, pos_dim=cfg.pos_dim,
                embed_dim=cfg.fusion_embed_dim, heads=cfg.fusion_heads,
                dropout=cfg.dropout, out_mode="concat",
            )
            backbone_in = self.fusion.out_dim
        else:
            self.fusion = nn.Identity()
            backbone_in = cfg.in_dim

        if cfg.backbone == "gat":
            self.backbone: nn.Module = GAT(
                in_dim=backbone_in, hidden_dims=list(cfg.gnn_hidden),
                heads=cfg.gnn_heads, dropout=cfg.dropout, residual=True,
            )
            self.rep_dim = self.backbone.out_dim
        elif cfg.backbone == "graphformer":
            self.backbone = SparseGraphTransformer(
                in_dim=backbone_in, hidden_dims=list(cfg.gnn_hidden),
                heads=cfg.gnn_heads, dropout=cfg.dropout,
            )
            self.rep_dim = self.backbone.out_dim
        else:
            self.backbone = GCN(
                in_dim=backbone_in, hidden_dims=list(cfg.gnn_hidden),
                dropout=cfg.dropout,
            )
            self.rep_dim = cfg.gnn_hidden[-1]

    def representation(self, x, graph, pos=None):
        if self.cfg.use_gpe:
            assert pos is not None
            h = self.fusion(x, pos)
        else:
            h = x
        return self.backbone(h, graph)

    def forward(self, x, graph, pos=None):
        rep = self.representation(x, graph, pos)
        return self.heads_forward(rep)


class BNNGraph(GraphBaseline):
    """BNN / S-learner on graph: single mu(x,t) head, ITE = mu(x,1) - mu(x,0)."""

    def __init__(self, cfg: PACTConfig):
        super().__init__(cfg)
        # mu takes [rep ; t_indicator] -> scalar
        mlp_in = self.rep_dim + 1
        layers: list[nn.Module] = []
        prev = mlp_in
        for h in cfg.mlp_hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(inplace=True)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.mu = nn.Sequential(*layers)

        prop_layers: list[nn.Module] = []
        prev = self.rep_dim
        for h in cfg.mlp_hidden:
            prop_layers += [nn.Linear(prev, h), nn.ReLU(inplace=True)]
            prev = h
        prop_layers.append(nn.Linear(prev, 1))
        self.prop = nn.Sequential(*prop_layers)

    def heads_forward(self, rep):
        N = rep.size(0)
        t0 = torch.zeros(N, 1, device=rep.device)
        t1 = torch.ones(N, 1, device=rep.device)
        mu0 = self.mu(torch.cat([rep, t0], dim=-1)).squeeze(-1)
        mu1 = self.mu(torch.cat([rep, t1], dim=-1)).squeeze(-1)
        e = torch.sigmoid(self.prop(rep).squeeze(-1))
        tau = mu1 - mu0
        zeros = torch.zeros_like(mu0)
        return XLearnerOutput(e=e, mu0=mu0, mu1=mu1, tau0=tau, tau1=tau, ls20=zeros, ls21=zeros)


class TARNetGraph(GraphBaseline):
    """TARNet / T-learner on graph: separate mu_0, mu_1 heads."""

    def __init__(self, cfg: PACTConfig):
        super().__init__(cfg)
        def _head():
            layers: list[nn.Module] = []
            prev = self.rep_dim
            for h in cfg.mlp_hidden:
                layers += [nn.Linear(prev, h), nn.ReLU(inplace=True)]
                prev = h
            layers.append(nn.Linear(prev, 1))
            return nn.Sequential(*layers)

        self.mu0 = _head()
        self.mu1 = _head()
        self.prop = _head()

    def heads_forward(self, rep):
        mu0 = self.mu0(rep).squeeze(-1)
        mu1 = self.mu1(rep).squeeze(-1)
        e = torch.sigmoid(self.prop(rep).squeeze(-1))
        tau = mu1 - mu0
        zeros = torch.zeros_like(mu0)
        return XLearnerOutput(e=e, mu0=mu0, mu1=mu1, tau0=tau, tau1=tau, ls20=zeros, ls21=zeros)


# ---------------------------------------------------------------------------
# Real-world (no-graph) variant: skip the GNN, run an MLP encoder instead.
# Used for Criteo / Lenta where there is no graph structure but we still want
# to compare meta-learners on uplift ranking metrics.
# ---------------------------------------------------------------------------


class PACTNoGraph(nn.Module):
    def __init__(
        self,
        in_dim: int,
        encoder_hidden: tuple[int, ...] = (256, 128),
        mlp_hidden: tuple[int, ...] = (128, 128),
        dropout: float = 0.0,
        use_variance: bool = True,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        for h in encoder_hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(inplace=True)]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        self.encoder = nn.Sequential(*layers)
        self.heads = XLearnerHeads(prev, list(mlp_hidden), dropout=dropout, use_variance=use_variance)

    def forward(self, x: torch.Tensor) -> XLearnerOutput:
        rep = self.encoder(x)
        return self.heads(rep)


class TLearnerNoGraph(nn.Module):
    """T-learner: shared encoder → separate mu_0, mu_1 heads.

    ITE = mu_1 - mu_0 (no cross-fitting tau heads). Compatible with
    variance-weighted loss via Bernoulli σ²=μ(1−μ) on the mu heads.
    """

    def __init__(
        self,
        in_dim: int,
        encoder_hidden: tuple[int, ...] = (256, 128),
        mlp_hidden: tuple[int, ...] = (128, 128),
        dropout: float = 0.0,
        use_variance: bool = False,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        for h in encoder_hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(inplace=True)]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        self.encoder = nn.Sequential(*layers)
        self.mu0 = _build_head(prev, mlp_hidden, dropout)
        self.mu1 = _build_head(prev, mlp_hidden, dropout)
        self.prop = _build_head(prev, mlp_hidden, dropout)

    def forward(self, x: torch.Tensor) -> XLearnerOutput:
        rep = self.encoder(x)
        mu0 = self.mu0(rep).squeeze(-1)
        mu1 = self.mu1(rep).squeeze(-1)
        e = torch.sigmoid(self.prop(rep).squeeze(-1))
        tau = mu1 - mu0
        zeros = torch.zeros_like(mu0)
        return XLearnerOutput(
            e=e, mu0=mu0, mu1=mu1,
            tau0=tau, tau1=tau,
            ls20=zeros, ls21=zeros,
        )


def _build_head(in_dim: int, hidden: tuple[int, ...], dropout: float) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev = in_dim
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.ReLU(inplace=True)]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        prev = h
    layers.append(nn.Linear(prev, 1))
    return nn.Sequential(*layers)

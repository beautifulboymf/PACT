"""SOTA graph ITE baselines with optional GPE plug-in.

Each model follows the pattern:
    [optional GPE fusion] → GNN backbone → model-specific heads → XLearnerOutput

This enables "control variable" evaluation: apply GPE to any baseline and
measure the improvement, proving GPE is a universal graph-uplift enhancer.

Models:
  NetDeconfGraph  — WSDM 2020, Guo et al.  (GCN + Wasserstein balance)
  GIALGraph       — KDD 2021, Chu et al.   (GNN + MI maximization + adversarial)
  GNUMGraph       — WWW 2023, Zhu et al.   (transformed target estimator)
  GDCGraph        — WSDM 2025, Hu et al.   (feature disentangle + 3 aggregators)
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .fusion import GPEFusion
from .heads import XLearnerOutput
from .layers import GAT, GCN, GraphConvolution, sparse_adj_to_edge_index


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def _pdist(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Pairwise L2 distance matrix [nx, ny]."""
    n1 = torch.sum(x ** 2, dim=1, keepdim=True)
    n2 = torch.sum(y ** 2, dim=1, keepdim=True)
    D = n1 + n2.T - 2.0 * x @ y.T
    return torch.sqrt(D.abs() + eps)


def wasserstein_distance(x: torch.Tensor, y: torch.Tensor, p: float = 0.5, lam: float = 10, its: int = 10) -> torch.Tensor:
    """Sinkhorn-based Wasserstein distance (from NetDeconf WSDM 2020)."""
    nx, ny = x.size(0), y.size(0)
    if nx == 0 or ny == 0:
        return x.new_zeros(())

    M = _pdist(x.squeeze(), y.squeeze())
    M_mean = M.mean().detach()
    delta = F.dropout(M, 10.0 / (nx * ny)).max().detach()
    eff_lam = (lam / (M_mean + 1e-8)).detach()

    row = delta * torch.ones(1, ny, device=x.device)
    col = torch.cat([delta * torch.ones(nx + 1, 1, device=x.device)], 0)
    col[-1, 0] = 0.0
    Mt = torch.cat([torch.cat([M, row], 0), col], 1)

    a = torch.cat([p * torch.ones(nx, 1, device=x.device) / nx, (1 - p) * torch.ones(1, 1, device=x.device)], 0)
    b = torch.cat([(1 - p) * torch.ones(ny, 1, device=x.device) / ny, p * torch.ones(1, 1, device=x.device)], 0)

    K = torch.exp(-eff_lam * Mt) + 1e-6
    ainvK = K / a
    u = a.clone()
    for _ in range(its):
        u = 1.0 / (ainvK @ (b / (u.T @ K).T))
    v = b / (u.T @ K).T
    E = (u * (v.T * K).detach()) * Mt
    return 2.0 * E.sum()


def _build_mlp(dims: list[int], act: str = "relu", final_act: bool = False) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2 or final_act:
            layers.append(nn.ReLU(inplace=True) if act == "relu" else nn.ELU())
    return nn.Sequential(*layers)


class _GPEMixin:
    """Adds optional GPE fusion to any baseline."""

    def _init_gpe(self, cfg, in_dim: int) -> int:
        if cfg.use_gpe:
            self.fusion = GPEFusion(
                feat_dim=in_dim, pos_dim=cfg.pos_dim,
                embed_dim=cfg.fusion_embed_dim, heads=cfg.fusion_heads,
                dropout=cfg.dropout, out_mode="concat",
            )
            return self.fusion.out_dim
        else:
            self.fusion = nn.Identity()
            return in_dim

    def _fuse(self, x: torch.Tensor, pos: Optional[torch.Tensor]) -> torch.Tensor:
        if isinstance(self.fusion, nn.Identity):
            return x
        assert pos is not None
        return self.fusion(x, pos)


# ---------------------------------------------------------------------------
# 1. NetDeconf (WSDM 2020)
# ---------------------------------------------------------------------------

class NetDeconfGraph(nn.Module, _GPEMixin):
    """Network Deconfounder (Guo et al. WSDM 2020).

    Architecture: GCN → rep → T-learner heads (mu_0, mu_1) + propensity.
    Loss adds Wasserstein(rep_T=1, rep_T=0) for representation balancing.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        backbone_in = self._init_gpe(cfg, cfg.in_dim)

        # GCN backbone (matching original: list of GraphConvolution layers)
        self.gc_layers = nn.ModuleList()
        prev = backbone_in
        for h in cfg.gnn_hidden:
            self.gc_layers.append(GraphConvolution(prev, h))
            prev = h
        self.rep_dim = prev

        # T-learner outcome heads
        self.out_t0 = nn.ModuleList()
        self.out_t1 = nn.ModuleList()
        prev = self.rep_dim
        for h in cfg.mlp_hidden:
            self.out_t0.append(nn.Linear(prev, h))
            self.out_t1.append(nn.Linear(prev, h))
            prev = h
        self.out_t0_final = nn.Linear(prev, 1)
        self.out_t1_final = nn.Linear(prev, 1)

        self.prop = nn.Sequential(nn.Linear(self.rep_dim, 1), nn.Sigmoid())
        self.dropout = cfg.dropout

    def forward(self, x, adj, pos=None, t=None):
        h = self._fuse(x, pos)
        # GCN forward
        for gc in self.gc_layers:
            h = F.relu(gc(h, adj))
            h = F.dropout(h, self.dropout, training=self.training)
        rep = h

        # Outcome heads
        h0, h1 = rep, rep
        for l0, l1 in zip(self.out_t0, self.out_t1):
            h0 = F.relu(l0(h0))
            h0 = F.dropout(h0, self.dropout, training=self.training)
            h1 = F.relu(l1(h1))
            h1 = F.dropout(h1, self.dropout, training=self.training)
        mu0 = self.out_t0_final(h0).squeeze(-1)
        mu1 = self.out_t1_final(h1).squeeze(-1)

        if t is not None:
            y_pred = torch.where(t > 0, mu1, mu0)
        else:
            y_pred = mu1 - mu0

        e = self.prop(rep).squeeze(-1)
        tau = mu1 - mu0
        z = torch.zeros_like(mu0)
        return XLearnerOutput(e=e, mu0=mu0, mu1=mu1, tau0=tau, tau1=tau, ls20=z, ls21=z), rep


# ---------------------------------------------------------------------------
# 2. GIAL (KDD 2021)
# ---------------------------------------------------------------------------

class GIALGraph(nn.Module, _GPEMixin):
    """Graph Infomax Adversarial Learning (Chu et al. KDD 2021).

    Architecture: GCN/GAT encoder → confounder representation →
      - MI maximization (DGI-style: maximize MI between node repr and graph summary)
      - Outcome generator (T-learner)
      - Adversarial discriminator (balances treatment/control representations)
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        backbone_in = self._init_gpe(cfg, cfg.in_dim)

        # GCN encoder
        self.gc_layers = nn.ModuleList()
        prev = backbone_in
        for h in cfg.gnn_hidden:
            self.gc_layers.append(GraphConvolution(prev, h))
            prev = h
        self.rep_dim = prev

        # MI discriminator (DGI-style bilinear scoring)
        self.mi_disc = nn.Bilinear(self.rep_dim, self.rep_dim, 1)

        # Outcome generator (T-learner)
        self.mu0 = _build_mlp([self.rep_dim] + list(cfg.mlp_hidden) + [1])
        self.mu1 = _build_mlp([self.rep_dim] + list(cfg.mlp_hidden) + [1])

        # Adversarial treatment discriminator
        self.disc = _build_mlp([self.rep_dim] + list(cfg.mlp_hidden) + [1])

        self.prop = nn.Sequential(nn.Linear(self.rep_dim, 1), nn.Sigmoid())
        self.dropout = cfg.dropout

    def _encode(self, x, adj):
        h = x
        for gc in self.gc_layers:
            h = F.relu(gc(h, adj))
            h = F.dropout(h, self.dropout, training=self.training)
        return h

    def forward(self, x, adj, pos=None, t=None):
        h = self._fuse(x, pos)
        rep = self._encode(h, adj)

        mu0 = self.mu0(rep).squeeze(-1)
        mu1 = self.mu1(rep).squeeze(-1)
        e = self.prop(rep).squeeze(-1)
        tau = mu1 - mu0
        z = torch.zeros_like(mu0)
        out = XLearnerOutput(e=e, mu0=mu0, mu1=mu1, tau0=tau, tau1=tau, ls20=z, ls21=z)

        # MI loss: positive = real repr vs graph summary; negative = shuffled repr vs summary
        summary = torch.sigmoid(rep.mean(dim=0, keepdim=True))  # [1, D]
        pos_mi = self.mi_disc(rep, summary.expand_as(rep))  # [N, 1]
        # Negative: shuffle node order
        perm = torch.randperm(rep.size(0), device=rep.device)
        neg_rep = self._encode(h[perm], adj)
        neg_mi = self.mi_disc(neg_rep, summary.expand_as(neg_rep))
        mi_loss = -torch.mean(F.logsigmoid(pos_mi)) - torch.mean(F.logsigmoid(-neg_mi))

        # Adversarial loss: discriminator tries to predict treatment from repr
        disc_pred = torch.sigmoid(self.disc(rep.detach()).squeeze(-1))
        if t is not None:
            disc_loss = F.binary_cross_entropy(disc_pred, t)
            # Generator loss: fool the discriminator
            gen_pred = torch.sigmoid(self.disc(rep).squeeze(-1))
            gen_loss = -F.binary_cross_entropy(gen_pred, t)
        else:
            disc_loss = torch.tensor(0.0, device=x.device)
            gen_loss = torch.tensor(0.0, device=x.device)

        return out, rep, {"mi_loss": mi_loss, "disc_loss": disc_loss, "gen_loss": gen_loss}


# ---------------------------------------------------------------------------
# 3. GNUM-CT (WWW 2023)
# ---------------------------------------------------------------------------

class GNUMGraph(nn.Module, _GPEMixin):
    """Graph Neural Uplift Model — Continuous Target variant (Zhu et al. 2023).

    Uses the transformed target: z_i = y_i * (t_i - p) / (p * (1 - p))
    where p is the marginal treatment probability. A single GNN predicts z
    directly, which equals E[tau] in expectation.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        backbone_in = self._init_gpe(cfg, cfg.in_dim)

        # GNN encoder (GCN backbone)
        self.gc_layers = nn.ModuleList()
        prev = backbone_in
        for h in cfg.gnn_hidden:
            self.gc_layers.append(GraphConvolution(prev, h))
            prev = h
        self.rep_dim = prev

        # Uplift predictor: rep → scalar (predicted tau directly)
        self.tau_head = _build_mlp([self.rep_dim] + list(cfg.mlp_hidden) + [1])
        # Propensity head for computing transformed target
        self.prop = nn.Sequential(nn.Linear(self.rep_dim, 1), nn.Sigmoid())
        self.dropout = cfg.dropout

    def forward(self, x, adj, pos=None, t=None):
        h = self._fuse(x, pos)
        for gc in self.gc_layers:
            h = F.relu(gc(h, adj))
            h = F.dropout(h, self.dropout, training=self.training)
        rep = h

        tau_pred = self.tau_head(rep).squeeze(-1)
        e = self.prop(rep).squeeze(-1)

        # Package as XLearnerOutput (mu0/mu1 are not separately estimated)
        z = torch.zeros_like(tau_pred)
        return XLearnerOutput(
            e=e, mu0=z, mu1=tau_pred,  # mu1 = tau so ITE via "s" mode = mu1 - mu0 = tau
            tau0=tau_pred, tau1=tau_pred,
            ls20=z, ls21=z,
        ), rep


# ---------------------------------------------------------------------------
# 4. GDC (WSDM 2025)
# ---------------------------------------------------------------------------

class _DisentangleMask(nn.Module):
    """Feature-wise sigmoid mask to split X → X_adj, X_conf."""

    def __init__(self, in_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, in_dim), nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mask = self.net(x)
        return x * mask, x * (1 - mask)  # X_adj, X_conf


class GDCGraph(nn.Module, _GPEMixin):
    """Graph Disentangle Causal Model (Hu et al. WSDM 2025).

    Architecture:
      1. GPE fusion (optional plug-in) on raw features
      2. Causal disentangle: feature-wise mask → X_adj, X_conf
      3. Three GAT aggregators:
         - Adjustment: aggregate X_adj over all neighbors
         - Confounder: aggregate X_conf over same-treatment neighbors
         - Counterfactual: aggregate X_conf over opposite-treatment neighbors
      4. Outcome heads from [adj_rep; conf_rep]
      5. Causal constraints: balance adj, predict T from conf, mapping loss
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        backbone_in = self._init_gpe(cfg, cfg.in_dim)

        # Disentangle
        self.disentangle = _DisentangleMask(backbone_in, hidden=256)

        # Three GCN aggregators (simpler than GAT for stability; GDC paper uses GAT)
        gnn_h = cfg.gnn_hidden[0] if cfg.gnn_hidden else 256
        n_layers = min(len(cfg.gnn_hidden), 2)  # GDC uses 2 layers
        self.adj_gc = nn.ModuleList([GraphConvolution(backbone_in if i == 0 else gnn_h, gnn_h) for i in range(n_layers)])
        self.conf_gc = nn.ModuleList([GraphConvolution(backbone_in if i == 0 else gnn_h, gnn_h) for i in range(n_layers)])
        self.cf_gc = nn.ModuleList([GraphConvolution(backbone_in if i == 0 else gnn_h, gnn_h) for i in range(n_layers)])
        self.rep_dim = gnn_h

        # Outcome heads from [adj_rep; conf_rep]
        cat_dim = self.rep_dim * 2
        self.mu0 = _build_mlp([cat_dim] + list(cfg.mlp_hidden) + [1])
        self.mu1 = _build_mlp([cat_dim] + list(cfg.mlp_hidden) + [1])

        # Treatment predictor from conf_rep
        self.treat_pred = nn.Sequential(nn.Linear(self.rep_dim, 1), nn.Sigmoid())

        # Mapping: conf_rep ↔ cf_conf_rep alignment
        self.mapper = nn.Linear(self.rep_dim, self.rep_dim)

        self.prop = nn.Sequential(nn.Linear(cat_dim, 1), nn.Sigmoid())
        self.dropout = cfg.dropout

    def _gcn_forward(self, gc_layers, x, adj):
        h = x
        for gc in gc_layers:
            h = F.relu(gc(h, adj))
            h = F.dropout(h, self.dropout, training=self.training)
        return h

    def _mask_adj(self, adj, t, same: bool):
        """Create adjacency subgraph: only same-treatment (or opposite) edges."""
        if not adj.is_sparse:
            adj = adj.to_sparse()
        idx = adj.coalesce().indices()
        vals = adj.coalesce().values()
        src, dst = idx[0], idx[1]
        if same:
            mask = (t[src] == t[dst])
        else:
            mask = (t[src] != t[dst])
        new_idx = idx[:, mask]
        new_vals = vals[mask]
        return torch.sparse_coo_tensor(new_idx, new_vals, adj.shape).coalesce()

    def forward(self, x, adj, pos=None, t=None):
        h = self._fuse(x, pos)

        # Disentangle
        x_adj, x_conf = self.disentangle(h)

        # Adjustment aggregation (full graph)
        adj_rep = self._gcn_forward(self.adj_gc, x_adj, adj)

        if t is not None:
            # Confounder: same-treatment neighbors
            adj_same = self._mask_adj(adj, t, same=True)
            conf_rep = self._gcn_forward(self.conf_gc, x_conf, adj_same)

            # Counterfactual confounder: opposite-treatment neighbors
            adj_opp = self._mask_adj(adj, t, same=False)
            cf_rep = self._gcn_forward(self.cf_gc, x_conf, adj_opp)
        else:
            # Inference without T: use full graph for both
            conf_rep = self._gcn_forward(self.conf_gc, x_conf, adj)
            cf_rep = self._gcn_forward(self.cf_gc, x_conf, adj)

        # Outcome prediction
        cat_rep = torch.cat([adj_rep, conf_rep], dim=-1)
        mu0 = self.mu0(cat_rep).squeeze(-1)
        mu1 = self.mu1(cat_rep).squeeze(-1)
        e = self.prop(cat_rep).squeeze(-1)
        tau = mu1 - mu0
        z = torch.zeros_like(mu0)
        out = XLearnerOutput(e=e, mu0=mu0, mu1=mu1, tau0=tau, tau1=tau, ls20=z, ls21=z)

        # Auxiliary losses for causal constraints
        treat_loss = torch.tensor(0.0, device=x.device)
        map_loss = torch.tensor(0.0, device=x.device)
        if t is not None:
            # Treatment prediction from conf_rep
            t_pred = self.treat_pred(conf_rep).squeeze(-1)
            treat_loss = F.binary_cross_entropy(t_pred, t)
            # Mapping loss: align conf_rep and cf_rep
            mapped = self.mapper(conf_rep)
            map_loss = F.mse_loss(mapped, cf_rep.detach())

        return out, adj_rep, {"treat_loss": treat_loss, "map_loss": map_loss}

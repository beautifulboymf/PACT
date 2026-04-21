"""GPE Mechanism Analysis: 3 experiments testing why GPE improves ITE estimation.

Experiment 1 — Propensity AUC (H1: confounding debiasing)
  Train models with/without GPE, measure propensity head AUC vs true T.

Experiment 2 — Representation Balance (H2: balanced representations)
  Compute MMD between T=1 and T=0 representations with/without GPE.

Experiment 3 — Concat vs Attention Ablation (H3: attention mechanism matters)
  Compare vanilla (no pos), concat pos, and full GPE cross-attention on DBLP.

Run on server:
    source conda activate pact
    cd /path/to/PACT
    python supplementary/run_gpe_mechanism.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Project imports
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pact.data import GraphUpliftSample
from pact.fusion import GPEFusion
from pact.heads import XLearnerOutput
from pact.layers import GAT, GCN, GraphConvolution
from pact.losses import LossWeights, x_learner_loss
from pact.metrics import evaluate_all
from pact.model import PACTConfig, PACT, TARNetGraph, BNNGraph, GraphBaseline
from pact.baselines import GDCGraph, NetDeconfGraph
from pact.train import TrainConfig, make_splits, _normalize_outcome, _slice, _ite_from_output

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CACHE_DIR = "runs/data_cache"
OUT_DIR = "runs/gpe_mechanism"
os.makedirs(OUT_DIR, exist_ok=True)

# Server configs use lr=0.001
SERVER_LR = 0.001
EPOCHS = 200
SEEDS = [0, 1, 2]


# ───────────────────────────────────────────────────────────────────────
# Data loading
# ───────────────────────────────────────────────────────────────────────

def load_cached_sample(dataset: str, rho: int, seed: int) -> GraphUpliftSample:
    cache_path = os.path.join(CACHE_DIR, f"{dataset}_rho{rho}_seed{seed}.pt")
    d = torch.load(cache_path, map_location=DEVICE, weights_only=True)
    A_norm = torch.sparse_coo_tensor(
        d["A_norm_indices"], d["A_norm_values"], d["A_norm_shape"]
    ).coalesce().to(DEVICE)
    return GraphUpliftSample(
        X=d["X"].to(DEVICE), pos=d["pos"].to(DEVICE),
        edge_index=d["edge_index"].to(DEVICE), A_norm=A_norm,
        T=d["T"].to(DEVICE), Y=d["Y"].to(DEVICE),
        Y0=d["Y0"].to(DEVICE), Y1=d["Y1"].to(DEVICE),
        true_tau=d["true_tau"].to(DEVICE),
    )


# ───────────────────────────────────────────────────────────────────────
# Concat fusion variant for Experiment 3
# ───────────────────────────────────────────────────────────────────────

class ConcatFusion(nn.Module):
    """Simple concatenation of features and positional encoding.

    Replaces GPEFusion: instead of cross-attention, just [h; p] -> Linear -> out.
    Keeps parameter count comparable by adding a projection layer.
    """
    def __init__(self, feat_dim: int, pos_dim: int, embed_dim: int):
        super().__init__()
        self.proj = nn.Linear(feat_dim + pos_dim, feat_dim + embed_dim)
        self.norm = nn.LayerNorm(feat_dim + embed_dim)
        self.out_dim = feat_dim + embed_dim

    def forward(self, h: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        cat = torch.cat([h, p], dim=-1)
        out = self.norm(F.elu(self.proj(cat)))
        return out


class TARNetConcat(GraphBaseline):
    """TARNet with concatenation fusion (no attention over positional encoding)."""

    def __init__(self, cfg: PACTConfig):
        # Override: use ConcatFusion instead of GPEFusion
        nn.Module.__init__(self)
        self.cfg = cfg

        # ConcatFusion instead of GPEFusion
        self.fusion = ConcatFusion(
            feat_dim=cfg.in_dim,
            pos_dim=cfg.pos_dim,
            embed_dim=cfg.fusion_embed_dim,
        )
        backbone_in = self.fusion.out_dim

        if cfg.backbone == "gat":
            self.backbone = GAT(
                in_dim=backbone_in, hidden_dims=list(cfg.gnn_hidden),
                heads=cfg.gnn_heads, dropout=cfg.dropout, residual=True,
            )
            self.rep_dim = self.backbone.out_dim
        else:
            self.backbone = GCN(
                in_dim=backbone_in, hidden_dims=list(cfg.gnn_hidden),
                dropout=cfg.dropout,
            )
            self.rep_dim = cfg.gnn_hidden[-1]

        # T-learner heads (same as TARNetGraph)
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

    def representation(self, x, graph, pos=None):
        assert pos is not None
        h = self.fusion(x, pos)
        return self.backbone(h, graph)

    def heads_forward(self, rep):
        mu0 = self.mu0(rep).squeeze(-1)
        mu1 = self.mu1(rep).squeeze(-1)
        e = torch.sigmoid(self.prop(rep).squeeze(-1))
        tau = mu1 - mu0
        zeros = torch.zeros_like(mu0)
        return XLearnerOutput(e=e, mu0=mu0, mu1=mu1, tau0=tau, tau1=tau, ls20=zeros, ls21=zeros)


# ───────────────────────────────────────────────────────────────────────
# Training helpers (reuse existing logic, but capture extra outputs)
# ───────────────────────────────────────────────────────────────────────

def train_and_extract(
    sample: GraphUpliftSample,
    model: nn.Module,
    model_cfg: PACTConfig,
    train_cfg: TrainConfig,
    is_baseline: bool = False,
    baseline_type: str = "",
) -> dict:
    """Train model and extract propensity predictions + representations at best epoch.

    Returns dict with keys: pehe, qini, propensity_auc, mmd, representations, etc.
    """
    from sklearn.metrics import roc_auc_score
    from pact.baselines import wasserstein_distance

    torch.manual_seed(train_cfg.seed)
    np.random.seed(train_cfg.seed)

    n = sample.X.size(0)
    train_idx, val_idx, test_idx = make_splits(
        n, train_cfg.train_frac, train_cfg.val_frac, seed=train_cfg.seed
    )
    train_t = torch.as_tensor(train_idx, device=DEVICE, dtype=torch.long)
    val_t = torch.as_tensor(val_idx, device=DEVICE, dtype=torch.long)
    test_t = torch.as_tensor(test_idx, device=DEVICE, dtype=torch.long)

    Xm = sample.X.mean(dim=0, keepdim=True)
    Xs = sample.X.std(dim=0, keepdim=True) + 1e-6
    X = (sample.X - Xm) / Xs

    if train_cfg.normalize_y:
        Y_n, ym, ys = _normalize_outcome(sample.Y, train_idx)
    else:
        Y_n, ym, ys = sample.Y, 0.0, 1.0

    if model_cfg.backbone == "gat":
        graph = sample.edge_index
    else:
        graph = sample.A_norm

    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay
    )

    use_variance = train_cfg.use_variance or (train_cfg.learner == "pact")
    ite_mode = "s" if train_cfg.learner in ("s", "t", "bnn", "tarnet") else "x"

    best = {
        "epoch": -1,
        "val_metric": float("-inf") if train_cfg.select_metric == "qini" else float("inf"),
    }

    for epoch in range(train_cfg.epochs):
        model.train()
        optimizer.zero_grad()

        if is_baseline:
            if baseline_type == "gdc":
                out_full, adj_rep, extras = model(X, sample.A_norm, pos=sample.pos, t=sample.T)
                YF_n = (sample.Y - ym) / ys if train_cfg.normalize_y else sample.Y
                y_pred = torch.where(sample.T > 0, out_full.mu1, out_full.mu0)
                mse = nn.MSELoss()
                loss_outcome = mse(y_pred[train_t], YF_n[train_t])
                ar_t1 = adj_rep[train_t][(sample.T[train_t] > 0).nonzero(as_tuple=True)[0]]
                ar_t0 = adj_rep[train_t][(sample.T[train_t] < 1).nonzero(as_tuple=True)[0]]
                if ar_t1.size(0) > 0 and ar_t0.size(0) > 0:
                    w_dist = wasserstein_distance(ar_t1, ar_t0)
                else:
                    w_dist = torch.tensor(0.0, device=DEVICE)
                loss = loss_outcome + 0.0001 * w_dist + 0.01 * extras["treat_loss"] + extras["map_loss"]
            else:
                raise ValueError(f"Unsupported baseline: {baseline_type}")
        else:
            out_full = model(X, graph, sample.pos)
            out_train = _slice(out_full, train_t)
            loss, _ = x_learner_loss(
                out_train, Y=Y_n[train_t], T=sample.T[train_t],
                weights=train_cfg.loss_weights,
                delta=train_cfg.delta,
                use_variance=use_variance,
                binary_outcome=train_cfg.binary_outcome,
            )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        if epoch % train_cfg.log_every == 0 or epoch == train_cfg.epochs - 1:
            model.eval()
            with torch.no_grad():
                if is_baseline:
                    if baseline_type == "gdc":
                        out_eval, _, _ = model(X, sample.A_norm, pos=sample.pos, t=sample.T)
                    ite = (out_eval.mu1 - out_eval.mu0) * ys
                else:
                    out_eval = model(X, graph, sample.pos)
                    ite = _ite_from_output(out_eval, ite_mode) * ys

                val_eval = evaluate_all(
                    ite[val_t], sample.T[val_t], sample.Y[val_t], sample.true_tau[val_t]
                )

            cur = val_eval.get(train_cfg.select_metric, val_eval["qini"])
            improved = (
                cur > best["val_metric"]
                if train_cfg.select_metric == "qini"
                else cur < best["val_metric"]
            )
            if improved:
                best = {
                    "epoch": epoch,
                    "val_metric": cur,
                    "val": val_eval,
                    "state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                }

    # Restore best model and extract outputs
    model.load_state_dict(best["state_dict"])
    model.eval()
    results = {}

    with torch.no_grad():
        # Get representations
        if is_baseline:
            if baseline_type == "gdc":
                out_final, adj_rep_final, _ = model(X, sample.A_norm, pos=sample.pos, t=sample.T)
                # For GDC, use the concatenated representation
                rep = torch.cat([adj_rep_final, model._gcn_forward(model.conf_gc, model.disentangle(model._fuse(X, sample.pos))[1], sample.A_norm)], dim=-1)
        else:
            rep = model.representation(X, graph, sample.pos)
            out_final = model(X, graph, sample.pos)

        # 1. Test-set metrics
        ite = (out_final.mu1 - out_final.mu0) * ys if is_baseline else _ite_from_output(out_final, ite_mode) * ys
        test_eval = evaluate_all(
            ite[test_t], sample.T[test_t], sample.Y[test_t], sample.true_tau[test_t]
        )
        results["pehe"] = test_eval.get("pehe", float("nan"))
        results["qini"] = test_eval["qini"]
        results["epoch"] = best["epoch"]

        # 2. Propensity AUC on test set
        e_test = out_final.e[test_t].cpu().numpy()
        T_test = sample.T[test_t].cpu().numpy()
        try:
            results["propensity_auc"] = float(roc_auc_score(T_test, e_test))
        except Exception:
            results["propensity_auc"] = float("nan")

        # 3. MMD between T=1 and T=0 representations on test set
        if not is_baseline:
            rep_test = rep[test_t]
        else:
            rep_test = rep[test_t] if 'rep' in dir() else adj_rep_final[test_t]
        T_test_t = sample.T[test_t]
        rep_t1 = rep_test[(T_test_t > 0.5).nonzero(as_tuple=True)[0]]
        rep_t0 = rep_test[(T_test_t < 0.5).nonzero(as_tuple=True)[0]]
        results["mmd"] = float(compute_mmd(rep_t1, rep_t0))

    return results


def compute_mmd(x: torch.Tensor, y: torch.Tensor, bandwidth: float = 1.0) -> torch.Tensor:
    """Compute Maximum Mean Discrepancy with RBF kernel.

    Uses the median heuristic for bandwidth selection.
    """
    if x.size(0) == 0 or y.size(0) == 0:
        return torch.tensor(0.0)

    # Median heuristic for bandwidth
    all_pts = torch.cat([x, y], dim=0)
    pdist = torch.cdist(all_pts, all_pts)
    median_dist = pdist[pdist > 0].median().item()
    sigma = median_dist if median_dist > 0 else 1.0

    def rbf(a, b):
        dist = torch.cdist(a, b)
        return torch.exp(-dist ** 2 / (2 * sigma ** 2))

    xx = rbf(x, x).mean()
    yy = rbf(y, y).mean()
    xy = rbf(x, y).mean()
    return (xx + yy - 2 * xy).clamp_min(0.0)


# ───────────────────────────────────────────────────────────────────────
# Experiment runners
# ───────────────────────────────────────────────────────────────────────

def get_model_cfg(sample: GraphUpliftSample, use_gpe: bool) -> PACTConfig:
    return PACTConfig(
        in_dim=sample.X.size(1),
        pos_dim=128,
        fusion_embed_dim=256,
        fusion_heads=4,
        use_gpe=use_gpe,
        backbone="gat",
        gnn_hidden=(256, 128, 128, 128, 128),
        gnn_heads=4,
        mlp_hidden=(128, 128, 128, 128),
        dropout=0.0,
        use_variance=False,  # default: no variance for baselines
    )


def get_train_cfg(seed: int, learner: str = "tarnet") -> TrainConfig:
    return TrainConfig(
        epochs=EPOCHS,
        lr=SERVER_LR,
        weight_decay=1e-4,
        train_frac=0.6,
        val_frac=0.2,
        learner=learner,
        loss_weights=LossWeights(mu=1.0, tau=1.0, sigma=0.25, prop=1.0),
        use_uncertainty_weighting=False,
        delta=0.01,
        log_every=10,
        select_metric="qini",
        normalize_y=True,
        seed=seed,
    )


def run_experiment_1_and_2():
    """Propensity AUC + MMD for TARNet, GDC, X-learner (PACT) with/without GPE."""
    print("=" * 70)
    print("EXPERIMENT 1 & 2: Propensity AUC + Representation Balance (MMD)")
    print("=" * 70)

    datasets = ["dblp", "cora_full"]
    # (name, learner_key, is_baseline, baseline_type, use_variance)
    model_specs = [
        ("TARNet",   "tarnet", False, "", False),
        ("GDC",      "gdc",    True,  "gdc", False),
        ("X-Learner","pact",  False, "", True),
    ]

    all_results = {}

    for ds_name in datasets:
        print(f"\n--- Dataset: {ds_name} ---")
        all_results[ds_name] = {}

        for model_name, learner, is_baseline, bl_type, use_var in model_specs:
            for use_gpe in [False, True]:
                gpe_tag = "GPE" if use_gpe else "noGPE"
                key = f"{model_name}_{gpe_tag}"
                print(f"\n  [{ds_name}] {key}:")

                seed_results = []
                for seed in SEEDS:
                    sample = load_cached_sample(ds_name, rho=10, seed=seed)
                    model_cfg = get_model_cfg(sample, use_gpe=use_gpe)
                    model_cfg.use_variance = use_var
                    train_cfg = get_train_cfg(seed=seed, learner=learner)
                    train_cfg.use_variance = use_var

                    if is_baseline and bl_type == "gdc":
                        model = GDCGraph(model_cfg)
                    elif learner == "tarnet":
                        model = TARNetGraph(model_cfg)
                    elif learner == "pact":
                        model = PACT(model_cfg)
                    else:
                        raise ValueError(learner)

                    r = train_and_extract(
                        sample, model, model_cfg, train_cfg,
                        is_baseline=is_baseline, baseline_type=bl_type,
                    )
                    seed_results.append(r)
                    print(f"    seed={seed}  PEHE={r['pehe']:.4f}  AUC={r['propensity_auc']:.4f}  MMD={r['mmd']:.6f}")

                # Aggregate across seeds
                agg = {
                    "propensity_auc_mean": float(np.mean([r["propensity_auc"] for r in seed_results])),
                    "propensity_auc_std": float(np.std([r["propensity_auc"] for r in seed_results])),
                    "mmd_mean": float(np.mean([r["mmd"] for r in seed_results])),
                    "mmd_std": float(np.std([r["mmd"] for r in seed_results])),
                    "pehe_mean": float(np.mean([r["pehe"] for r in seed_results])),
                    "pehe_std": float(np.std([r["pehe"] for r in seed_results])),
                    "qini_mean": float(np.mean([r["qini"] for r in seed_results])),
                    "qini_std": float(np.std([r["qini"] for r in seed_results])),
                    "per_seed": seed_results,
                }
                all_results[ds_name][key] = agg
                print(f"    => AUC={agg['propensity_auc_mean']:.4f}+/-{agg['propensity_auc_std']:.4f}  "
                      f"MMD={agg['mmd_mean']:.6f}+/-{agg['mmd_std']:.6f}  "
                      f"PEHE={agg['pehe_mean']:.4f}+/-{agg['pehe_std']:.4f}")

    return all_results


def run_experiment_3():
    """Concatenation vs Attention Ablation on DBLP rho=10 with TARNet."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Concat vs Attention Ablation (DBLP, TARNet)")
    print("=" * 70)

    results = {}

    for variant_name, variant_key in [
        ("Vanilla (no pos)", "vanilla"),
        ("Concat pos", "concat"),
        ("GPE attention", "gpe"),
    ]:
        print(f"\n  Variant: {variant_name}")
        seed_results = []

        for seed in SEEDS:
            sample = load_cached_sample("dblp", rho=10, seed=seed)
            model_cfg = get_model_cfg(sample, use_gpe=(variant_key != "vanilla"))
            model_cfg.use_variance = False
            train_cfg = get_train_cfg(seed=seed, learner="tarnet")
            train_cfg.use_variance = False

            if variant_key == "vanilla":
                model_cfg.use_gpe = False
                model = TARNetGraph(model_cfg)
            elif variant_key == "concat":
                model_cfg.use_gpe = True  # We handle this differently
                model = TARNetConcat(model_cfg)
            elif variant_key == "gpe":
                model_cfg.use_gpe = True
                model = TARNetGraph(model_cfg)

            r = train_and_extract(
                sample, model, model_cfg, train_cfg,
                is_baseline=False, baseline_type="",
            )
            seed_results.append(r)
            print(f"    seed={seed}  PEHE={r['pehe']:.4f}  Qini={r['qini']:.4f}")

        agg = {
            "pehe_mean": float(np.mean([r["pehe"] for r in seed_results])),
            "pehe_std": float(np.std([r["pehe"] for r in seed_results])),
            "qini_mean": float(np.mean([r["qini"] for r in seed_results])),
            "qini_std": float(np.std([r["qini"] for r in seed_results])),
            "propensity_auc_mean": float(np.mean([r["propensity_auc"] for r in seed_results])),
            "propensity_auc_std": float(np.std([r["propensity_auc"] for r in seed_results])),
            "mmd_mean": float(np.mean([r["mmd"] for r in seed_results])),
            "mmd_std": float(np.std([r["mmd"] for r in seed_results])),
            "per_seed": seed_results,
        }
        results[variant_key] = agg
        print(f"    => PEHE={agg['pehe_mean']:.4f}+/-{agg['pehe_std']:.4f}  "
              f"Qini={agg['qini_mean']:.4f}+/-{agg['qini_std']:.4f}")

    return results


# ───────────────────────────────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()

    # Run Experiments 1 & 2 (joint: same training, extract both propensity AUC and MMD)
    results_12 = run_experiment_1_and_2()

    # Run Experiment 3
    results_3 = run_experiment_3()

    elapsed = time.time() - t0
    print(f"\n\nTotal wall time: {elapsed:.0f}s ({elapsed/60:.1f}min)")

    # Save raw JSON
    payload = {
        "experiment_1_2": results_12,
        "experiment_3": results_3,
        "wall_time_seconds": elapsed,
    }
    json_path = os.path.join(OUT_DIR, "gpe_mechanism_data.json")
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"\nRaw data saved to: {json_path}")

    # Print summary tables
    print_summary(results_12, results_3)


def print_summary(results_12, results_3):
    """Print formatted tables for inclusion in the paper."""

    print("\n\n" + "=" * 80)
    print("TABLE 1: Propensity AUC (H1 — confounding debiasing)")
    print("=" * 80)
    print(f"{'Model':<20} {'Dataset':<12} {'No GPE':<20} {'With GPE':<20} {'Delta':<10}")
    print("-" * 80)
    for ds in ["dblp", "cora_full"]:
        for model_name in ["TARNet", "GDC", "X-Learner"]:
            nogpe = results_12[ds].get(f"{model_name}_noGPE", {})
            gpe = results_12[ds].get(f"{model_name}_GPE", {})
            if nogpe and gpe:
                no_str = f"{nogpe['propensity_auc_mean']:.4f}+/-{nogpe['propensity_auc_std']:.4f}"
                gpe_str = f"{gpe['propensity_auc_mean']:.4f}+/-{gpe['propensity_auc_std']:.4f}"
                delta = gpe['propensity_auc_mean'] - nogpe['propensity_auc_mean']
                print(f"{model_name:<20} {ds:<12} {no_str:<20} {gpe_str:<20} {delta:+.4f}")

    print("\n\n" + "=" * 80)
    print("TABLE 2: Representation Balance — MMD(T=1, T=0) (H2)")
    print("=" * 80)
    print(f"{'Model':<20} {'Dataset':<12} {'No GPE':<24} {'With GPE':<24} {'Ratio':<10}")
    print("-" * 80)
    for ds in ["dblp", "cora_full"]:
        for model_name in ["TARNet", "GDC", "X-Learner"]:
            nogpe = results_12[ds].get(f"{model_name}_noGPE", {})
            gpe = results_12[ds].get(f"{model_name}_GPE", {})
            if nogpe and gpe:
                no_str = f"{nogpe['mmd_mean']:.6f}+/-{nogpe['mmd_std']:.6f}"
                gpe_str = f"{gpe['mmd_mean']:.6f}+/-{gpe['mmd_std']:.6f}"
                ratio = gpe['mmd_mean'] / max(nogpe['mmd_mean'], 1e-10)
                print(f"{model_name:<20} {ds:<12} {no_str:<24} {gpe_str:<24} {ratio:.3f}x")

    print("\n\n" + "=" * 80)
    print("TABLE 3: Concat vs Attention Ablation — DBLP rho=10 (H3)")
    print("=" * 80)
    print(f"{'Variant':<25} {'PEHE (lower=better)':<25} {'Qini (higher=better)':<25}")
    print("-" * 80)
    for name, key in [
        ("(a) Vanilla (no pos)", "vanilla"),
        ("(b) Concat pos", "concat"),
        ("(c) GPE attention", "gpe"),
    ]:
        r = results_3[key]
        pehe_str = f"{r['pehe_mean']:.4f} +/- {r['pehe_std']:.4f}"
        qini_str = f"{r['qini_mean']:.4f} +/- {r['qini_std']:.4f}"
        print(f"{name:<25} {pehe_str:<25} {qini_str:<25}")


if __name__ == "__main__":
    main()

"""Unified training loop for PACT and the GAT+S/T/X ablation baselines.

Design notes:

- Feature normalization happens once before the loop, not every epoch,
  so train/val/test splits share a consistent input scale.
- Per-treatment outcome standardization stats are computed on train_idx
  only — never on test indices, to avoid leakage.
- The graph forward pass is run on the full graph each step (transductive),
  but losses and metrics are evaluated only on the masked split. This matches
  the WSDM2020 / PACT setup.
- `--learner` switch chooses the ablation:
    s     -> S-learner: only mu_0 / mu_1, ITE = mu_1 - mu_0     (use_variance=False, only mu loss)
    x     -> vanilla X-learner (no variance head, no GPE optional)
    pact -> full PACT (variance head + GPE)
- Best-model selection uses validation Qini by default. PEHE
  is also tracked for ablation tables.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .data import GraphUpliftSample
from .heads import XLearnerOutput
from .losses import LossWeights, UncertaintyWeighting, x_learner_loss
from .metrics import evaluate_all
from .model import PACT, PACTConfig


# ---------------------------------------------------------------------------
# Index utilities
# ---------------------------------------------------------------------------


def make_splits(n: int, train_frac: float = 0.6, val_frac: float = 0.2, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_tr = int(round(n * train_frac))
    n_va = int(round(n * val_frac))
    return idx[:n_tr], idx[n_tr : n_tr + n_va], idx[n_tr + n_va :]


def _normalize_outcome(Y: torch.Tensor, idx: np.ndarray) -> tuple[torch.Tensor, float, float]:
    sub = Y[idx]
    m, s = float(sub.mean()), float(sub.std() + 1e-6)
    return (Y - m) / s, m, s


def _slice(out: XLearnerOutput, idx) -> XLearnerOutput:
    return XLearnerOutput(
        e=out.e[idx],
        mu0=out.mu0[idx],
        mu1=out.mu1[idx],
        tau0=out.tau0[idx],
        tau1=out.tau1[idx],
        ls20=out.ls20[idx],
        ls21=out.ls21[idx],
    )


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


@dataclass
class TrainConfig:
    epochs: int = 200
    lr: float = 5e-2
    weight_decay: float = 1e-4
    train_frac: float = 0.6
    val_frac: float = 0.2
    learner: str = "pact"           # 's' | 'x' | 'pact'
    loss_weights: LossWeights = field(default_factory=LossWeights)
    use_uncertainty_weighting: bool = False
    delta: float = 1e-2
    log_every: int = 10
    select_metric: str = "qini"      # 'qini' or 'pehe'
    normalize_y: bool = True
    binary_outcome: bool = False     # BCE for mu, Bernoulli variance for binary Y
    use_variance: bool = False       # explicit variance plug-in toggle (overrides learner default)
    batch_size: int = 4096           # used by train_tabular only
    seed: int = 0


def _ite_from_output(out: XLearnerOutput, mode: str) -> torch.Tensor:
    if mode == "s":
        return out.mu1 - out.mu0
    return out.ite("x")


def train_graph(
    sample: GraphUpliftSample,
    model_cfg: PACTConfig,
    train_cfg: TrainConfig,
    device: str = "cpu",
    model: nn.Module | None = None,
) -> dict:
    torch.manual_seed(train_cfg.seed)
    np.random.seed(train_cfg.seed)

    n = sample.X.size(0)
    train_idx, val_idx, test_idx = make_splits(
        n, train_cfg.train_frac, train_cfg.val_frac, seed=train_cfg.seed
    )
    train_t = torch.as_tensor(train_idx, device=device, dtype=torch.long)
    val_t = torch.as_tensor(val_idx, device=device, dtype=torch.long)
    test_t = torch.as_tensor(test_idx, device=device, dtype=torch.long)

    # 1. Feature standardization, once.
    Xm = sample.X.mean(dim=0, keepdim=True)
    Xs = sample.X.std(dim=0, keepdim=True) + 1e-6
    X = (sample.X - Xm) / Xs

    # 2. Outcome standardization on train idx only.
    if train_cfg.normalize_y:
        Y_n, ym, ys = _normalize_outcome(sample.Y, train_idx)
    else:
        Y_n, ym, ys = sample.Y, 0.0, 1.0

    # 3. Choose graph object that the backbone expects.
    if model_cfg.backbone in ("gat", "graphformer"):
        graph = sample.edge_index
    else:
        graph = sample.A_norm

    if model is None:
        from .model import PACT as _PACT
        model = _PACT(model_cfg)
    model = model.to(device)
    uw = UncertaintyWeighting().to(device) if train_cfg.use_uncertainty_weighting else None

    params = list(model.parameters()) + (list(uw.parameters()) if uw else [])
    optimizer = optim.Adam(params, lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)

    use_variance = train_cfg.use_variance or (train_cfg.learner == "pact")
    ite_mode = "s" if train_cfg.learner in ("s", "t", "bnn", "tarnet") else "x"

    best = {"epoch": -1, "val_metric": float("-inf") if train_cfg.select_metric == "qini" else float("inf")}
    history: list[dict] = []

    for epoch in range(train_cfg.epochs):
        model.train()
        optimizer.zero_grad()
        out_full = model(X, graph, sample.pos)
        out_train = _slice(out_full, train_t)
        loss, log = x_learner_loss(
            out_train,
            Y=Y_n[train_t],
            T=sample.T[train_t],
            weights=train_cfg.loss_weights,
            delta=train_cfg.delta,
            use_variance=use_variance,
            binary_outcome=train_cfg.binary_outcome,
            uncertainty_weighting=uw,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        if epoch % train_cfg.log_every == 0 or epoch == train_cfg.epochs - 1:
            model.eval()
            with torch.no_grad():
                out_full = model(X, graph, sample.pos)
                ite = _ite_from_output(out_full, ite_mode) * ys  # de-standardize ITE scale
                val_eval = evaluate_all(
                    pred_tau=ite[val_t],
                    T=sample.T[val_t],
                    Y=sample.Y[val_t],
                    true_tau=sample.true_tau[val_t],
                )
                test_eval = evaluate_all(
                    pred_tau=ite[test_t],
                    T=sample.T[test_t],
                    Y=sample.Y[test_t],
                    true_tau=sample.true_tau[test_t],
                )
            row = {
                "epoch": epoch,
                "train_loss": log["loss/total"],
                **{f"val_{k}": v for k, v in val_eval.items()},
                **{f"test_{k}": v for k, v in test_eval.items()},
            }
            history.append(row)
            print(
                f"[ep {epoch:3d}] loss={log['loss/total']:.4f}  "
                f"val_pehe={val_eval.get('pehe', float('nan')):.4f}  "
                f"val_qini={val_eval['qini']:.4f}  "
                f"test_pehe={test_eval.get('pehe', float('nan')):.4f}  "
                f"test_qini={test_eval['qini']:.4f}"
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
                    "test": test_eval,
                    "state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                }

    return {"best": best, "history": history}


# ---------------------------------------------------------------------------
# SOTA baseline training loop (NetDeconf, GIAL, GNUM, GDC)
# ---------------------------------------------------------------------------


def train_graph_baseline(
    sample: GraphUpliftSample,
    model: nn.Module,
    model_cfg,
    train_cfg: TrainConfig,
    device: str = "cpu",
    baseline_type: str = "netdeconf",
    alpha: float = 1e-4,
) -> dict:
    """Train a SOTA graph baseline with optional GPE and variance weighting.

    Handles the extra losses that each baseline introduces (Wasserstein,
    MI, adversarial, disentangle constraints).
    """
    from .baselines import wasserstein_distance

    torch.manual_seed(train_cfg.seed)
    np.random.seed(train_cfg.seed)

    n = sample.X.size(0)
    train_idx, val_idx, test_idx = make_splits(
        n, train_cfg.train_frac, train_cfg.val_frac, seed=train_cfg.seed
    )
    train_t = torch.as_tensor(train_idx, device=device, dtype=torch.long)
    val_t = torch.as_tensor(val_idx, device=device, dtype=torch.long)
    test_t = torch.as_tensor(test_idx, device=device, dtype=torch.long)

    Xm = sample.X.mean(dim=0, keepdim=True)
    Xs = sample.X.std(dim=0, keepdim=True) + 1e-6
    X = (sample.X - Xm) / Xs

    if train_cfg.normalize_y:
        Y_n, ym, ys = _normalize_outcome(sample.Y, train_idx)
    else:
        Y_n, ym, ys = sample.Y, 0.0, 1.0

    adj = sample.A_norm  # all baselines use normalized sparse adj
    T = sample.T
    pos = sample.pos

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
    mse = nn.MSELoss()

    best = {"epoch": -1, "val_metric": float("-inf") if train_cfg.select_metric == "qini" else float("inf")}
    history: list[dict] = []

    for epoch in range(train_cfg.epochs):
        model.train()
        optimizer.zero_grad()

        use_var = train_cfg.use_variance
        delta = train_cfg.delta

        def _var_weighted_mse(pred, target, mask_t1, mask_t0):
            """Variance-weighted MSE: downweight high-residual samples."""
            residuals = (pred - target).detach() ** 2
            # Per-treatment-group variance estimate (rolling EMA across epochs
            # is ideal but per-batch is simpler and works well enough)
            if use_var and residuals.numel() > 1:
                sigma2 = residuals.clamp_min(delta)
                w = 1.0 / sigma2
                w = w / (w.mean().detach() + 1e-8)
                return (w * (pred - target) ** 2).mean()
            return mse(pred, target)

        # Forward — each baseline returns (XLearnerOutput, rep, [extras])
        if baseline_type == "netdeconf":
            out, rep = model(X, adj, pos=pos, t=T)
            YF_n = (sample.Y - ym) / ys if train_cfg.normalize_y else sample.Y
            y_pred = torch.where(T > 0, out.mu1, out.mu0)
            mask_t1 = (T[train_t] > 0)
            mask_t0 = (T[train_t] < 1)
            loss_outcome = _var_weighted_mse(y_pred[train_t], YF_n[train_t], mask_t1, mask_t0)
            rep_t1 = rep[train_t][(T[train_t] > 0).nonzero(as_tuple=True)[0]]
            rep_t0 = rep[train_t][(T[train_t] < 1).nonzero(as_tuple=True)[0]]
            if rep_t1.size(0) > 0 and rep_t0.size(0) > 0:
                w_dist = wasserstein_distance(rep_t1, rep_t0)
            else:
                w_dist = torch.tensor(0.0, device=device)
            loss = loss_outcome + alpha * w_dist

        elif baseline_type == "gial":
            out, rep, extras = model(X, adj, pos=pos, t=T)
            YF_n = (sample.Y - ym) / ys if train_cfg.normalize_y else sample.Y
            y_pred = torch.where(T > 0, out.mu1, out.mu0)
            loss_outcome = _var_weighted_mse(y_pred[train_t], YF_n[train_t], None, None)
            loss = loss_outcome + 0.1 * extras["mi_loss"] + 0.1 * extras["disc_loss"] + 0.1 * extras["gen_loss"]

        elif baseline_type == "gnum":
            out, rep = model(X, adj, pos=pos, t=T)
            p_marginal = T[train_t].mean().detach().clamp(0.1, 0.9)
            z_target = sample.Y[train_t] * (T[train_t] - p_marginal) / (p_marginal * (1 - p_marginal))
            if train_cfg.normalize_y:
                z_target = (z_target - z_target.mean()) / (z_target.std() + 1e-6)
            loss = _var_weighted_mse(out.tau0[train_t], z_target, None, None)

        elif baseline_type == "gdc":
            out, adj_rep, extras = model(X, adj, pos=pos, t=T)
            YF_n = (sample.Y - ym) / ys if train_cfg.normalize_y else sample.Y
            y_pred = torch.where(T > 0, out.mu1, out.mu0)
            loss_outcome = _var_weighted_mse(y_pred[train_t], YF_n[train_t], None, None)
            ar_t1 = adj_rep[train_t][(T[train_t] > 0).nonzero(as_tuple=True)[0]]
            ar_t0 = adj_rep[train_t][(T[train_t] < 1).nonzero(as_tuple=True)[0]]
            if ar_t1.size(0) > 0 and ar_t0.size(0) > 0:
                w_dist = wasserstein_distance(ar_t1, ar_t0)
            else:
                w_dist = torch.tensor(0.0, device=device)
            loss = loss_outcome + 0.0001 * w_dist + 0.01 * extras["treat_loss"] + extras["map_loss"]
        else:
            raise ValueError(baseline_type)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        if epoch % train_cfg.log_every == 0 or epoch == train_cfg.epochs - 1:
            model.eval()
            with torch.no_grad():
                if baseline_type in ("netdeconf", "gnum"):
                    out_eval, _ = model(X, adj, pos=pos, t=T)
                elif baseline_type == "gial":
                    out_eval, _, _ = model(X, adj, pos=pos, t=T)
                elif baseline_type == "gdc":
                    out_eval, _, _ = model(X, adj, pos=pos, t=T)

                ite = (out_eval.mu1 - out_eval.mu0) * ys
                val_eval = evaluate_all(ite[val_t], T[val_t], sample.Y[val_t], sample.true_tau[val_t])
                test_eval = evaluate_all(ite[test_t], T[test_t], sample.Y[test_t], sample.true_tau[test_t])

            row = {"epoch": epoch, "train_loss": float(loss.detach()), **{f"val_{k}": v for k, v in val_eval.items()}, **{f"test_{k}": v for k, v in test_eval.items()}}
            history.append(row)
            print(f"[ep {epoch:3d}] loss={float(loss):.4f}  val_pehe={val_eval.get('pehe', float('nan')):.4f}  val_qini={val_eval['qini']:.4f}  test_pehe={test_eval.get('pehe', float('nan')):.4f}  test_qini={test_eval['qini']:.4f}")

            cur = val_eval.get(train_cfg.select_metric, val_eval["qini"])
            improved = cur > best["val_metric"] if train_cfg.select_metric == "qini" else cur < best["val_metric"]
            if improved:
                best = {"epoch": epoch, "val_metric": cur, "val": val_eval, "test": test_eval,
                        "state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}

    return {"best": best, "history": history}


# ---------------------------------------------------------------------------
# Tabular training loop (no graph — Criteo / Hillstrom)
# ---------------------------------------------------------------------------


def _batched_predict(
    model: nn.Module,
    X: torch.Tensor,
    idx: np.ndarray,
    batch_size: int,
    ite_mode: str,
    ys: float,
    binary_outcome: bool = False,
) -> torch.Tensor:
    """Run inference in chunks, return de-standardized ITE predictions."""
    preds = []
    for start in range(0, len(idx), batch_size):
        batch_idx = idx[start : start + batch_size]
        out = model(X[batch_idx])
        if binary_outcome and ite_mode == "s":
            # mu outputs are logits → convert to probabilities for ITE
            ite = torch.sigmoid(out.mu1) - torch.sigmoid(out.mu0)
        else:
            ite = _ite_from_output(out, ite_mode) * ys
        preds.append(ite.detach().cpu())
    return torch.cat(preds)


def train_tabular(
    sample,   # TabularUpliftSample
    model: nn.Module,
    train_cfg: TrainConfig,
    device: str = "cpu",
    ite_mode: str = "x",
) -> dict:
    """Train a tabular (no-graph) uplift model with mini-batching.

    Observational regime: no oracle true_tau — only Qini / AUUC / Lift@k.
    """
    from .data import TabularUpliftSample

    torch.manual_seed(train_cfg.seed)
    np.random.seed(train_cfg.seed)

    n = sample.X.size(0)
    X = sample.X.to(device)
    T = sample.T.to(device)
    Y_raw = sample.Y.to(device)

    train_idx, val_idx, test_idx = make_splits(
        n, train_cfg.train_frac, train_cfg.val_frac, seed=train_cfg.seed
    )

    # Normalize features once.
    Xm = X.mean(dim=0, keepdim=True)
    Xs = X.std(dim=0, keepdim=True) + 1e-6
    X = (X - Xm) / Xs

    # Normalize outcomes on train split.
    if train_cfg.normalize_y:
        sub = Y_raw[train_idx]
        ym, ys = float(sub.mean()), float(sub.std() + 1e-6)
        Y = (Y_raw - ym) / ys
    else:
        Y, ym, ys = Y_raw, 0.0, 1.0

    model = model.to(device)
    use_variance = train_cfg.use_variance
    uw = UncertaintyWeighting().to(device) if train_cfg.use_uncertainty_weighting else None
    params = list(model.parameters()) + (list(uw.parameters()) if uw else [])
    optimizer = optim.Adam(params, lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)

    bs = train_cfg.batch_size
    best = {"epoch": -1, "val_metric": float("-inf")}
    history: list[dict] = []

    for epoch in range(train_cfg.epochs):
        model.train()
        rng = np.random.default_rng(train_cfg.seed + epoch)
        perm = rng.permutation(len(train_idx))
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, len(perm), bs):
            batch_perm = perm[start : start + bs]
            bi = train_idx[batch_perm]
            optimizer.zero_grad()
            out = model(X[bi])
            loss, log = x_learner_loss(
                out, Y=Y[bi], T=T[bi],
                weights=train_cfg.loss_weights,
                delta=train_cfg.delta,
                use_variance=use_variance,
                binary_outcome=train_cfg.binary_outcome,
                uncertainty_weighting=uw,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            epoch_loss += float(loss.detach())
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)

        if epoch % train_cfg.log_every == 0 or epoch == train_cfg.epochs - 1:
            model.eval()
            with torch.no_grad():
                ite_val = _batched_predict(model, X, val_idx, bs, ite_mode, ys, train_cfg.binary_outcome)
                ite_test = _batched_predict(model, X, test_idx, bs, ite_mode, ys, train_cfg.binary_outcome)
                val_eval = evaluate_all(ite_val, T[val_idx].cpu(), Y_raw[val_idx].cpu(), true_tau=None)
                test_eval = evaluate_all(ite_test, T[test_idx].cpu(), Y_raw[test_idx].cpu(), true_tau=None)

            row = {
                "epoch": epoch,
                "train_loss": avg_loss,
                **{f"val_{k}": v for k, v in val_eval.items()},
                **{f"test_{k}": v for k, v in test_eval.items()},
            }
            history.append(row)
            print(
                f"[ep {epoch:3d}] loss={avg_loss:.4f}  "
                f"val_qini={val_eval['qini']:.4f}  "
                f"val_auuc={val_eval['auuc']:.4f}  "
                f"test_qini={test_eval['qini']:.4f}  "
                f"test_auuc={test_eval['auuc']:.4f}"
            )

            cur = val_eval.get(train_cfg.select_metric, val_eval["qini"])
            improved = cur > best["val_metric"]
            if improved:
                best = {
                    "epoch": epoch,
                    "val_metric": cur,
                    "val": val_eval,
                    "test": test_eval,
                    "state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                }

    return {"best": best, "history": history}

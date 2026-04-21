"""Evaluation metrics for ITE estimation and uplift ranking.

We support two regimes:

  - Oracle (semi-synthetic): the true ITE `tau_i = Y1_i - Y0_i` is known
    for every node. We can compute PEHE, |ATE|, and oracle-Qini directly.

  - Observational (real-world: Criteo, Lenta): only (Y, T, predicted ITE)
    are available. Qini, AUUC, and Lift@k are computed from the cumulative
    treated/control outcome difference along the predicted-ITE ranking. These
    are the standard uplift-modeling definitions used by causalML / scikit-
    uplift, not the "predicted vs true" variant that treats the predicted
    ITE as ground truth (which silently breaks on real observational data).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch


def _to_np(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


# ---------------------------------------------------------------------------
# Oracle metrics (require true_tau)
# ---------------------------------------------------------------------------


def pehe(pred_tau, true_tau) -> float:
    pt = _to_np(pred_tau).reshape(-1)
    tt = _to_np(true_tau).reshape(-1)
    return float(np.sqrt(np.mean((pt - tt) ** 2)))


def abs_ate(pred_tau, true_tau) -> float:
    pt = _to_np(pred_tau).reshape(-1)
    tt = _to_np(true_tau).reshape(-1)
    return float(abs(pt.mean() - tt.mean()))


# ---------------------------------------------------------------------------
# Observational uplift-ranking metrics
# ---------------------------------------------------------------------------


def _ranked_cum_uplift(pred: np.ndarray, T: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (k, cum_uplift) where cum_uplift[i] is the cumulative incremental
    response after targeting the top-(i+1) units sorted by `pred`."""
    order = np.argsort(-pred, kind="stable")
    T_s = T[order]
    Y_s = Y[order]

    cum_T = np.cumsum(T_s)
    cum_C = np.cumsum(1 - T_s)
    cum_YT = np.cumsum(Y_s * T_s)
    cum_YC = np.cumsum(Y_s * (1 - T_s))

    # incremental response (Radcliffe Qini): YT_k - YC_k * (n_T_k / n_C_k)
    safe_C = np.where(cum_C > 0, cum_C, 1)
    cum_uplift = cum_YT - cum_YC * (cum_T / safe_C)
    k = np.arange(1, len(pred) + 1)
    return k, cum_uplift


def qini_curve(pred, T, Y) -> tuple[np.ndarray, np.ndarray]:
    pred = _to_np(pred).reshape(-1)
    T_a = _to_np(T).reshape(-1)
    Y_a = _to_np(Y).reshape(-1)
    return _ranked_cum_uplift(pred, T_a, Y_a)


def qini_coefficient(pred, T, Y) -> float:
    """Normalized Qini coefficient: area between model curve and random baseline,
    normalized by the area between the perfect-oracle curve and random."""
    pred = _to_np(pred).reshape(-1)
    T_a = _to_np(T).reshape(-1)
    Y_a = _to_np(Y).reshape(-1)
    n = len(pred)
    if n == 0:
        return 0.0

    _, cum = _ranked_cum_uplift(pred, T_a, Y_a)
    # random baseline: linear from 0 to cum[-1]
    base = np.linspace(cum[-1] / n, cum[-1], n)
    model_area = np.trapz(cum - base, dx=1.0)

    # perfect-model curve = sort by true response gap; we don't know it, so use
    # the strongest achievable proxy: rank by Y for treated, -Y for control,
    # which is the standard sklift "perfect_uplift_curve" trick.
    perfect_score = np.where(T_a == 1, Y_a, -Y_a)
    _, cum_perf = _ranked_cum_uplift(perfect_score, T_a, Y_a)
    base_perf = np.linspace(cum_perf[-1] / n, cum_perf[-1], n)
    perfect_area = np.trapz(cum_perf - base_perf, dx=1.0)

    if perfect_area <= 0:
        return 0.0
    return float(model_area / perfect_area)


def auuc(pred, T, Y, normalize: bool = True) -> float:
    """Area Under the Uplift Curve."""
    pred = _to_np(pred).reshape(-1)
    T_a = _to_np(T).reshape(-1)
    Y_a = _to_np(Y).reshape(-1)
    _, cum = _ranked_cum_uplift(pred, T_a, Y_a)
    area = float(np.trapz(cum, dx=1.0))
    if normalize:
        area /= max(len(pred), 1)
    return area


def lift_at_k(pred, T, Y, k_frac: float = 0.3) -> float:
    """Incremental response in the top `k_frac` of the predicted-ITE ranking."""
    pred = _to_np(pred).reshape(-1)
    T_a = _to_np(T).reshape(-1)
    Y_a = _to_np(Y).reshape(-1)
    n = len(pred)
    if n == 0:
        return 0.0
    k = max(1, int(round(k_frac * n)))
    _, cum = _ranked_cum_uplift(pred, T_a, Y_a)
    return float(cum[k - 1])


# ---------------------------------------------------------------------------
# Convenience: full eval bundle
# ---------------------------------------------------------------------------


def evaluate_all(
    pred_tau,
    T,
    Y,
    true_tau: Optional = None,
) -> dict[str, float]:
    out: dict[str, float] = {
        "qini": qini_coefficient(pred_tau, T, Y),
        "auuc": auuc(pred_tau, T, Y),
        "lift@30": lift_at_k(pred_tau, T, Y, 0.3),
    }
    if true_tau is not None:
        out["pehe"] = pehe(pred_tau, true_tau)
        out["ate"] = abs_ate(pred_tau, true_tau)
    return out

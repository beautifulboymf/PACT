"""Loss functions for the variance-weighted X-learner.

Key correctness notes:

1. mu_t loss is masked: mu_0 is fitted only on T=0 samples (where Y is the
   factual outcome of the control group), mu_1 only on T=1 samples.
2. The cross-residual D_t is computed with detached mu values. Otherwise
   gradients flow through mu via the tau_t loss path and contaminate mu's
   own training signal — and tau ends up chasing a moving target.
3. Pseudo-outcomes:
       D_1 = Y - mu_0(x)   defined on treatment samples (T=1)
       D_0 = mu_1(x) - Y   defined on control   samples (T=0)
   tau_1 is fitted on D_1 over T=1, tau_0 is fitted on D_0 over T=0.
4. Variance-weighted MSE for tau_t (weight = 1 / (sigma^2_t + delta)).
5. log-sigma^2 head is regressed on log(D_t^2 - tau_t^2 + eps) — i.e. the
   log target — so the head doesn't need to fit a possibly enormous raw
   variance scale. Equivalent to the paper's regression of sigma^2 on D^2 -
   tau^2 modulo the log link.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .heads import XLearnerOutput


@dataclass
class LossWeights:
    mu: float = 1.0
    tau: float = 1.0
    sigma: float = 0.25      # paper-style: ~0.2-0.3 of mu/tau
    prop: float = 1.0


def _safe_div(num: torch.Tensor, den: torch.Tensor) -> torch.Tensor:
    return num / den.clamp_min(1.0)


def x_learner_loss(
    out: XLearnerOutput,
    Y: torch.Tensor,
    T: torch.Tensor,
    weights: LossWeights = LossWeights(),
    delta: float = 1e-2,
    use_variance: bool = True,
    binary_outcome: bool = False,
    uncertainty_weighting: Optional["UncertaintyWeighting"] = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute the joint X-learner loss for a single mini-batch.

    Args:
        out: head outputs (already restricted to the train indices upstream).
        Y:   observed factual outcome [N].
        T:   treatment indicator in {0,1} [N].
        weights: fixed loss weights (used unless uncertainty_weighting is given).
        delta: floor on sigma^2 used in variance weights.
        use_variance: if False, behaves like a vanilla X-learner (no sigma head,
            uniform weights for tau).
        binary_outcome: if True, use BCE for mu_t and derive variance
            analytically as mu*(1-mu) (Bernoulli) instead of a learned sigma
            head.  This is critical for binary Y (e.g. Criteo conversion).
        uncertainty_weighting: optional Kendall-style learnable task weighting.

    Returns:
        (total_loss, log_dict)
    """
    mask0 = (T < 0.5)
    mask1 = (T >= 0.5)
    n0 = mask0.sum().clamp_min(1)
    n1 = mask1.sum().clamp_min(1)

    # 1. mu_t losses (masked), optionally variance-weighted.
    if binary_outcome:
        # mu outputs are logits; use BCE.
        bce = F.binary_cross_entropy_with_logits
        mu0_d = torch.sigmoid(out.mu0).detach()
        mu1_d = torch.sigmoid(out.mu1).detach()
        bce0 = bce(out.mu0, Y, reduction="none")
        bce1 = bce(out.mu1, Y, reduction="none")

        if use_variance:
            # Variance-weighted mu: downweight samples near decision boundary
            # (high aleatoric noise) → reduces Var[tau = mu1 - mu0].
            wmu0 = (1.0 / (mu0_d * (1.0 - mu0_d)).clamp_min(delta))
            wmu1 = (1.0 / (mu1_d * (1.0 - mu1_d)).clamp_min(delta))
            wmu0 = wmu0 / (wmu0[mask0].mean().detach() + 1e-8)
            wmu1 = wmu1 / (wmu1[mask1].mean().detach() + 1e-8)
            loss_mu0 = _safe_div((wmu0 * bce0 * mask0).sum(), n0)
            loss_mu1 = _safe_div((wmu1 * bce1 * mask1).sum(), n1)
        else:
            loss_mu0 = _safe_div((bce0 * mask0).sum(), n0)
            loss_mu1 = _safe_div((bce1 * mask1).sum(), n1)
    else:
        loss_mu0 = _safe_div(((out.mu0 - Y) ** 2 * mask0).sum(), n0)
        loss_mu1 = _safe_div(((out.mu1 - Y) ** 2 * mask1).sum(), n1)
        mu0_d = out.mu0.detach()
        mu1_d = out.mu1.detach()
    loss_mu = 0.5 * (loss_mu0 + loss_mu1)

    # 2. Cross residuals using detached mu.
    D1 = Y - mu0_d  # imputed effect on treated
    D0 = mu1_d - Y  # imputed effect on control

    # 3. tau_t losses, optionally variance-weighted.
    if use_variance:
        if binary_outcome:
            # Bernoulli variance: sigma^2 = mu*(1-mu).  For D_1 (treated),
            # the noise comes from Y|T=1 ~ Bernoulli(mu_1), so use mu_1.
            # For D_0 (control), noise from Y|T=0 ~ Bernoulli(mu_0), use mu_0.
            sigma2_1 = (mu1_d * (1.0 - mu1_d)).clamp_min(delta)
            sigma2_0 = (mu0_d * (1.0 - mu0_d)).clamp_min(delta)
        else:
            sigma2_0 = torch.exp(out.ls20.detach()).clamp_min(delta)
            sigma2_1 = torch.exp(out.ls21.detach()).clamp_min(delta)
        w0 = 1.0 / sigma2_0
        w1 = 1.0 / sigma2_1
        # normalize weights so the loss magnitude doesn't drift with delta.
        w0 = w0 / (w0[mask0].mean().detach() + 1e-8)
        w1 = w1 / (w1[mask1].mean().detach() + 1e-8)
    else:
        w0 = torch.ones_like(out.tau0)
        w1 = torch.ones_like(out.tau1)

    loss_tau0 = _safe_div((w0 * (out.tau0 - D0) ** 2 * mask0).sum(), n0)
    loss_tau1 = _safe_div((w1 * (out.tau1 - D1) ** 2 * mask1).sum(), n1)
    loss_tau = 0.5 * (loss_tau0 + loss_tau1)

    # 4. sigma^2 regression — skip for binary outcomes (variance is analytic).
    loss_sigma = torch.tensor(0.0, device=Y.device)
    if use_variance and not binary_outcome:
        eps = 1e-3
        with torch.no_grad():
            tgt0 = torch.log(((D0 - out.tau0.detach()) ** 2).clamp_min(eps))
            tgt1 = torch.log(((D1 - out.tau1.detach()) ** 2).clamp_min(eps))
        loss_s0 = _safe_div(((out.ls20 - tgt0) ** 2 * mask0).sum(), n0)
        loss_s1 = _safe_div(((out.ls21 - tgt1) ** 2 * mask1).sum(), n1)
        loss_sigma = 0.5 * (loss_s0 + loss_s1)

    # 5. propensity (BCE).
    eps = 1e-7
    loss_prop = -(
        T * torch.log(out.e.clamp(eps, 1 - eps))
        + (1 - T) * torch.log((1 - out.e).clamp(eps, 1 - eps))
    ).mean()

    if uncertainty_weighting is not None:
        total = uncertainty_weighting(
            mu=loss_mu, tau=loss_tau, sigma=loss_sigma, prop=loss_prop
        )
    else:
        total = (
            weights.mu * loss_mu
            + weights.tau * loss_tau
            + (weights.sigma * loss_sigma if use_variance else 0.0)
            + weights.prop * loss_prop
        )

    log = {
        "loss/total": float(total.detach()),
        "loss/mu": float(loss_mu.detach()),
        "loss/tau": float(loss_tau.detach()),
        "loss/sigma": float(loss_sigma.detach()) if use_variance else 0.0,
        "loss/prop": float(loss_prop.detach()),
    }
    return total, log


class UncertaintyWeighting(nn.Module):
    """Kendall & Gal (2018) homoscedastic-uncertainty multi-task weighting.

    Replaces the hand-tuned (mu, tau, sigma, prop) loss weights with learned
    log-variances. Each task loss becomes loss_i / (2 sigma_i^2) + log sigma_i.
    Use as: ``total = uw(mu=l_mu, tau=l_tau, sigma=l_sigma, prop=l_prop)``.
    """

    def __init__(self, tasks: tuple[str, ...] = ("mu", "tau", "sigma", "prop")):
        super().__init__()
        self.tasks = tasks
        self.log_var = nn.ParameterDict(
            {t: nn.Parameter(torch.zeros(())) for t in tasks}
        )

    def forward(self, **task_losses: torch.Tensor) -> torch.Tensor:
        total = task_losses[self.tasks[0]].new_zeros(())
        for name in self.tasks:
            l = task_losses.get(name)
            if l is None:
                continue
            lv = self.log_var[name]
            total = total + 0.5 * torch.exp(-lv) * l + 0.5 * lv
        return total

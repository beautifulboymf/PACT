"""X-learner heads for PACT.

Five heads, all conditioned on the GNN representation:

  e(x)        propensity (sigmoid)
  mu_t(x)     factual outcome regression, t in {0,1}
  tau_t(x)    pseudo-outcome regression for the cross-residual D_t
  ls2_t(x)    log-variance head, log sigma^2_t(x), t in {0,1}

We deliberately predict log-sigma^2 (linear output), not raw sigma with a ReLU,
because the latter has zero gradient on the negative side and explodes the
variance weight w_t = 1 / (sigma^2 + delta) when the head outputs ~0. Using a
log-variance head + exp() makes everything well-behaved.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


def _build_mlp(in_dim: int, hidden_dims: list[int], out_dim: int, dropout: float = 0.0) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev = in_dim
    for h in hidden_dims:
        layers.append(nn.Linear(prev, h))
        layers.append(nn.ReLU(inplace=True))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


@dataclass
class XLearnerOutput:
    e: torch.Tensor          # [N] propensity
    mu0: torch.Tensor        # [N] mu_0(x)
    mu1: torch.Tensor        # [N] mu_1(x)
    tau0: torch.Tensor       # [N] tau_0(x)
    tau1: torch.Tensor       # [N] tau_1(x)
    ls20: torch.Tensor       # [N] log sigma^2_0(x)
    ls21: torch.Tensor       # [N] log sigma^2_1(x)

    def sigma2_0(self) -> torch.Tensor:
        return torch.exp(self.ls20)

    def sigma2_1(self) -> torch.Tensor:
        return torch.exp(self.ls21)

    def ite(self, mode: str = "x") -> torch.Tensor:
        """Final ITE estimate.

        mode='x'     standard X-learner combination tau = e * tau_0 + (1-e) * tau_1
        mode='s'     S-learner-style: mu_1 - mu_0
        """
        if mode == "x":
            e = self.e.detach()
            return e * self.tau0 + (1 - e) * self.tau1
        elif mode == "s":
            return self.mu1 - self.mu0
        else:
            raise ValueError(mode)


class XLearnerHeads(nn.Module):
    def __init__(
        self,
        rep_dim: int,
        hidden_dims: list[int],
        dropout: float = 0.0,
        use_variance: bool = True,
    ):
        super().__init__()
        self.use_variance = use_variance
        self.mu0 = _build_mlp(rep_dim, hidden_dims, 1, dropout)
        self.mu1 = _build_mlp(rep_dim, hidden_dims, 1, dropout)
        self.tau0 = _build_mlp(rep_dim, hidden_dims, 1, dropout)
        self.tau1 = _build_mlp(rep_dim, hidden_dims, 1, dropout)
        self.prop = _build_mlp(rep_dim, hidden_dims, 1, dropout)
        if use_variance:
            self.ls20 = _build_mlp(rep_dim, hidden_dims, 1, dropout)
            self.ls21 = _build_mlp(rep_dim, hidden_dims, 1, dropout)

    def forward(self, rep: torch.Tensor) -> XLearnerOutput:
        e = torch.sigmoid(self.prop(rep)).squeeze(-1)
        mu0 = self.mu0(rep).squeeze(-1)
        mu1 = self.mu1(rep).squeeze(-1)
        tau0 = self.tau0(rep).squeeze(-1)
        tau1 = self.tau1(rep).squeeze(-1)
        if self.use_variance:
            ls20 = self.ls20(rep).squeeze(-1)
            ls21 = self.ls21(rep).squeeze(-1)
        else:
            ls20 = torch.zeros_like(mu0)
            ls21 = torch.zeros_like(mu0)
        return XLearnerOutput(e=e, mu0=mu0, mu1=mu1, tau0=tau0, tau1=tau1, ls20=ls20, ls21=ls21)

#!/usr/bin/env python3
"""Propensity AUC vs PEHE improvement correlation plot (W4).

Uses data from the mechanism analysis (propensity AUC table) and main results
(PEHE table) to show that GPE's confounding debiasing directly translates
to ITE estimation improvement.

Run: python -m pact.experiments.plot_auc_pehe_correlation
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


# Points are loaded from results/mechanism/auc_pehe_points.json, which is
# produced by running the mechanism analysis (see supplementary/run_gpe_mechanism.py).
# Each point = (model, dataset, rho=10, 5 seeds mean) of propensity AUC and
# test PEHE with and without GPE. Override the path via the POINTS_JSON env var
# if you want to plot a different aggregation.
import json as _json
import os as _os

_POINTS_JSON = _os.environ.get(
    "POINTS_JSON",
    _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))),
                  "results/mechanism/auc_pehe_points.json"),
)
with open(_POINTS_JSON) as _fh:
    _payload = _json.load(_fh)
DATA = [
    (p["model"], p["dataset"], p["auc_nogpe"], p["auc_gpe"], p["pehe_nogpe"], p["pehe_gpe"])
    for p in _payload["points"]
]


def main():
    delta_auc = []
    delta_pehe = []
    labels = []
    colors_map = {"TARNet": "#2196F3", "GDC": "#F44336", "X-learner": "#4CAF50"}
    marker_map = {"DBLP": "o", "CoraFull": "s"}

    for model, ds, auc_no, auc_gpe, pehe_no, pehe_gpe in DATA:
        da = auc_gpe - auc_no
        dp = pehe_no - pehe_gpe  # positive = GPE reduces PEHE (good)
        delta_auc.append(da)
        delta_pehe.append(dp)
        labels.append(f"{model}/{ds}")

    delta_auc = np.array(delta_auc)
    delta_pehe = np.array(delta_pehe)

    # Correlations
    r_pearson, p_pearson = stats.pearsonr(delta_auc, delta_pehe)
    r_spearman, p_spearman = stats.spearmanr(delta_auc, delta_pehe)

    print(f"Pearson:  r={r_pearson:.3f}, p={p_pearson:.4f}")
    print(f"Spearman: r={r_spearman:.3f}, p={p_spearman:.4f}")

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(5, 4))

    for i, (model, ds, _, _, _, _) in enumerate(DATA):
        ax.scatter(delta_auc[i], delta_pehe[i],
                   c=colors_map[model], marker=marker_map[ds],
                   s=80, zorder=3, edgecolors="white", linewidth=0.5)

    # Regression line
    slope, intercept = np.polyfit(delta_auc, delta_pehe, 1)
    x_fit = np.linspace(delta_auc.min() - 0.01, delta_auc.max() + 0.01, 100)
    ax.plot(x_fit, slope * x_fit + intercept, "k--", alpha=0.5, linewidth=1)

    ax.set_xlabel(r"$\Delta$ Propensity AUC (GPE $-$ No GPE)", fontsize=10)
    ax.set_ylabel(r"$\Delta$ PEHE (No GPE $-$ GPE)", fontsize=10)
    ax.set_title(f"Spearman $r_s$={r_spearman:.2f} ($p$={p_spearman:.3f})", fontsize=10)

    # Legend for models
    for model, color in colors_map.items():
        ax.scatter([], [], c=color, marker="o", s=60, label=model)
    # Legend for datasets
    for ds, marker in marker_map.items():
        ax.scatter([], [], c="gray", marker=marker, s=60, label=ds)
    ax.legend(fontsize=8, loc="lower right", framealpha=0.9)

    ax.axhline(0, color="gray", linewidth=0.5, alpha=0.5)
    ax.axvline(0, color="gray", linewidth=0.5, alpha=0.5)

    ax.grid(True, alpha=0.2)
    plt.tight_layout()

    out_path = "images/auc_pehe_scatter.pdf"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved to {out_path}")

    # Also save PNG for preview
    fig.savefig(out_path.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    print(f"Saved PNG preview")


if __name__ == "__main__":
    main()

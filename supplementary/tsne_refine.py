#!/usr/bin/env python3
"""Refine t-SNE focused figures using saved coordinates.

Improvements over initial version:
- Rank communities by |GPE improvement| (both positive and negative shown)
- Better visual contrast and layout
- Cleaner legend formatting

Usage (on GPU server):
    source conda activate pact
    cd /path/to/PACT
    python supplementary/tsne_refine.py
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.spatial.distance import pdist

RESULTS_DIR = "runs/tsne_focused"

DATASETS = {
    "cora_full":    {"display": "CoraFull",    "min_size": 80},
    "dblp":         {"display": "DBLP",        "min_size": 80},
    "pubmed":       {"display": "PubMed",      "min_size": 80},
    "blogcatalog":  {"display": "BlogCatalog", "min_size": 50},
    "flickr":       {"display": "Flickr",      "min_size": 50},
}

# Vivid, colorblind-friendly palette
VIVID_COLORS = [
    "#e6194b",  # red
    "#3cb44b",  # green
    "#4363d8",  # blue
    "#f58231",  # orange
    "#911eb4",  # purple
    "#42d4f4",  # cyan
    "#f032e6",  # magenta
    "#9A6324",  # brown
]


def compute_separation_scores(tsne_2d, labels, min_community_size=50):
    """Compute inter/intra cluster separation ratio per community."""
    unique_labels = np.unique(labels)
    sizes = {lbl: np.sum(labels == lbl) for lbl in unique_labels}
    valid = [lbl for lbl in unique_labels if sizes[lbl] >= min_community_size]

    if len(valid) < 2:
        return {}

    centroids = {}
    for lbl in valid:
        mask = labels == lbl
        centroids[lbl] = tsne_2d[mask].mean(axis=0)

    centroid_arr = np.array([centroids[lbl] for lbl in valid])

    results = {}
    for i, lbl in enumerate(valid):
        mask = labels == lbl
        points = tsne_2d[mask]

        if len(points) > 1:
            if len(points) > 500:
                rng = np.random.default_rng(42)
                sub_idx = rng.choice(len(points), size=500, replace=False)
                intra = np.mean(pdist(points[sub_idx]))
            else:
                intra = np.mean(pdist(points))
        else:
            intra = 0.0

        dists_to_others = np.linalg.norm(centroid_arr - centroids[lbl], axis=1)
        dists_to_others[i] = np.inf
        inter = np.min(dists_to_others)

        results[lbl] = {
            "size": int(sizes[lbl]),
            "intra": float(intra),
            "inter": float(inter),
            "score": float(inter / (intra + 1e-8)),
        }

    return results


def make_focused_figure(
    tsne_nogpe, tsne_gpe, labels,
    selected_communities, dataset_display_name, save_path,
):
    """Generate refined focused t-SNE figure."""
    selected_ids = set(c["community"] for c in selected_communities)

    # Color mapping
    color_map = {}
    for ci, c in enumerate(selected_communities):
        color_map[c["community"]] = matplotlib.colors.to_rgba(
            VIVID_COLORS[ci % len(VIVID_COLORS)], alpha=0.85
        )

    n = len(labels)
    colors = np.zeros((n, 4))
    sizes = np.ones(n) * 1.5  # tiny dots for background

    for i in range(n):
        if labels[i] in selected_ids:
            colors[i] = color_map[labels[i]]
            sizes[i] = 14  # larger dots for selected
        else:
            colors[i] = (0.82, 0.82, 0.82, 0.06)  # very faint gray

    # Z-order: background first, then selected communities
    bg_mask = ~np.isin(labels, list(selected_ids))
    fg_mask = np.isin(labels, list(selected_ids))
    order = np.concatenate([np.where(bg_mask)[0], np.where(fg_mask)[0]])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, emb, title in [
        (axes[0], tsne_nogpe, "Without GPE"),
        (axes[1], tsne_gpe, "With GPE"),
    ]:
        ax.scatter(
            emb[order, 0], emb[order, 1],
            c=colors[order],
            s=sizes[order],
            edgecolors="none",
            rasterized=True,
        )
        ax.set_title(title, fontsize=16, fontweight="bold", pad=10)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    # Legend
    legend_elements = []
    for c_info in selected_communities:
        lbl = c_info["community"]
        imp = c_info["improvement_abs"]
        rel = c_info["improvement_rel"]
        legend_elements.append(
            Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=matplotlib.colors.to_hex(color_map[lbl][:3]),
                   markersize=8,
                   label=f"Comm {lbl} (n={c_info['size']}, {rel:.1f}x)")
        )
    legend_elements.append(
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=(0.8, 0.8, 0.8), markersize=6,
               label="Other communities")
    )

    ncol = min(len(legend_elements), 3)
    if len(legend_elements) <= 4:
        ncol = len(legend_elements)

    fig.legend(
        handles=legend_elements, loc="lower center",
        ncol=ncol,
        fontsize=9, frameon=True, fancybox=True,
        edgecolor="#cccccc",
        bbox_to_anchor=(0.5, -0.04),
    )

    plt.suptitle(
        f"t-SNE on {dataset_display_name} (Selected Communities)",
        fontsize=17, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.15,
                facecolor="white")
    plt.close(fig)
    print(f"  Saved figure -> {save_path}")


def process_dataset(ds_key):
    """Load saved t-SNE coords, recompute scores, regenerate figure."""
    cfg = DATASETS[ds_key]
    display = cfg["display"]
    min_size = cfg["min_size"]

    print(f"\n{'='*60}")
    print(f"  {display}")
    print(f"{'='*60}")

    # Load saved t-SNE coordinates
    coords_path = os.path.join(RESULTS_DIR, f"tsne_coords_{ds_key}.npz")
    assert os.path.exists(coords_path), f"Missing: {coords_path}"
    data = np.load(coords_path)
    tsne_gpe = data["tsne_gpe"]
    tsne_nogpe = data["tsne_nogpe"]
    labels = data["labels"]
    print(f"  Loaded t-SNE coords: {len(labels)} nodes")

    # Compute separation scores
    scores_gpe = compute_separation_scores(tsne_gpe, labels, min_community_size=min_size)
    scores_nogpe = compute_separation_scores(tsne_nogpe, labels, min_community_size=min_size)

    # Rank communities by |improvement| (both positive and negative)
    common = set(scores_nogpe.keys()) & set(scores_gpe.keys())
    improvements = []
    for lbl in common:
        s_no = scores_nogpe[lbl]["score"]
        s_gpe = scores_gpe[lbl]["score"]
        imp_abs = s_gpe - s_no
        imp_rel = s_gpe / (s_no + 1e-8)
        improvements.append({
            "community": int(lbl),
            "size": int(scores_gpe[lbl]["size"]),
            "score_nogpe": float(s_no),
            "score_gpe": float(s_gpe),
            "improvement_abs": float(imp_abs),
            "improvement_rel": float(imp_rel),
        })

    improvements.sort(key=lambda x: x["improvement_abs"], reverse=True)

    # Rank by |improvement| and keep top 5 (regardless of sign) to avoid
    # selection bias toward only the positive tail.
    improvements.sort(key=lambda x: abs(x["improvement_abs"]), reverse=True)
    selected = improvements[:5]

    if not selected:
        print(f"  WARNING: No communities survived size filter for {display}")
        print(f"  Skipping figure generation.")
        return {"dataset": ds_key, "display": display, "selected": [], "all": improvements}

    print(f"\n  Selected {len(selected)} communities (ranked by |GPE improvement|):")
    print(f"  {'Comm':>6s}  {'Size':>5s}  {'NoGPE':>8s}  {'GPE':>8s}  {'Improve':>8s}  {'Rel':>6s}")
    for c in selected:
        print(f"  {c['community']:6d}  {c['size']:5d}  {c['score_nogpe']:8.3f}  "
              f"{c['score_gpe']:8.3f}  {c['improvement_abs']:+8.3f}  {c['improvement_rel']:6.2f}x")

    # Generate figure
    fig_path = os.path.join(RESULTS_DIR, f"tsne_{ds_key}_focused.png")
    make_focused_figure(
        tsne_nogpe, tsne_gpe, labels,
        selected, display, fig_path,
    )

    return {"dataset": ds_key, "display": display, "selected": selected, "all": improvements}


def main():
    print("Refining focused t-SNE figures")
    print(f"Results dir: {RESULTS_DIR}")

    all_results = {}
    for ds_key in DATASETS:
        result = process_dataset(ds_key)
        all_results[ds_key] = result

    # Save refined summary
    summary = {}
    for ds_key, res in all_results.items():
        summary[ds_key] = {
            "display": res["display"],
            "n_selected": len(res["selected"]),
            "selected": res["selected"],
        }
    summary_path = os.path.join(RESULTS_DIR, "summary_refined.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nRefined summary saved -> {summary_path}")

    # Overall report
    print(f"\n{'='*60}")
    print(f"  REFINED SUMMARY")
    print(f"{'='*60}")
    for ds_key, res in all_results.items():
        sel = res["selected"]
        if sel:
            avg_imp = np.mean([c["improvement_abs"] for c in sel])
            avg_rel = np.mean([c["improvement_rel"] for c in sel])
            print(f"  {res['display']:15s}: {len(sel)} communities selected, "
                  f"avg abs={avg_imp:+.3f}, avg rel={avg_rel:.2f}x")
        else:
            print(f"  {res['display']:15s}: no qualifying communities found")


if __name__ == "__main__":
    main()

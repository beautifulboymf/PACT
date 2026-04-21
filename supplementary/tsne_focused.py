#!/usr/bin/env python3
"""Focused t-SNE visualization: select top communities by GPE improvement.

Loads saved 512-dim embeddings from runs/tsne_all/, runs t-SNE, computes
per-community separation metrics, selects communities where GPE most
improves cluster quality, and generates focused figures.

Usage (on GPU server):
    source conda activate pact
    cd /path/to/PACT
    python supplementary/tsne_focused.py
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.manifold import TSNE
from scipy.spatial.distance import pdist, cdist

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
EMB_DIR = "runs/tsne_all"
RESULTS_DIR = "runs/tsne_focused"

DATASETS = {
    "cora_full":    {"display": "CoraFull",    "min_size": 80,  "top_k": 5},
    "dblp":         {"display": "DBLP",        "min_size": 80,  "top_k": 5},
    "pubmed":       {"display": "PubMed",      "min_size": 80,  "top_k": 5},
    "blogcatalog":  {"display": "BlogCatalog", "min_size": 50,  "top_k": 5},
    "flickr":       {"display": "Flickr",      "min_size": 50,  "top_k": 5},
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
    "#fabed4",  # pink
    "#9A6324",  # brown
    "#dcbeff",  # lavender
]


def compute_separation_scores(tsne_2d: np.ndarray, labels: np.ndarray,
                               min_community_size: int = 50):
    """Compute inter/intra cluster separation ratio per community.

    For each community c:
        intra_c = mean pairwise distance within c
        inter_c = distance from centroid of c to nearest OTHER centroid
        score_c = inter_c / (intra_c + 1e-8)

    Higher score = better separated cluster.
    """
    unique_labels = np.unique(labels)
    # Filter to communities above minimum size
    sizes = {lbl: np.sum(labels == lbl) for lbl in unique_labels}
    valid = [lbl for lbl in unique_labels if sizes[lbl] >= min_community_size]

    if len(valid) < 2:
        return {}

    # Compute centroids for all valid communities
    centroids = {}
    for lbl in valid:
        mask = labels == lbl
        centroids[lbl] = tsne_2d[mask].mean(axis=0)

    centroid_arr = np.array([centroids[lbl] for lbl in valid])

    results = {}
    for i, lbl in enumerate(valid):
        mask = labels == lbl
        points = tsne_2d[mask]

        # Intra-cluster: mean pairwise distance within community
        if len(points) > 1:
            # For large communities, subsample to keep computation fast
            if len(points) > 500:
                rng = np.random.default_rng(42)
                sub_idx = rng.choice(len(points), size=500, replace=False)
                intra = np.mean(pdist(points[sub_idx]))
            else:
                intra = np.mean(pdist(points))
        else:
            intra = 0.0

        # Inter-cluster: min distance from this centroid to any other centroid
        dists_to_others = np.linalg.norm(centroid_arr - centroids[lbl], axis=1)
        dists_to_others[i] = np.inf  # exclude self
        inter = np.min(dists_to_others)

        results[lbl] = {
            "size": int(sizes[lbl]),
            "intra": float(intra),
            "inter": float(inter),
            "score": float(inter / (intra + 1e-8)),
        }

    return results


def select_top_communities(scores_nogpe, scores_gpe, top_k=5):
    """Select communities with largest GPE improvement in separation score.

    improvement = score_gpe - score_nogpe  (absolute improvement)
    Also compute relative improvement = score_gpe / score_nogpe
    """
    common = set(scores_nogpe.keys()) & set(scores_gpe.keys())
    improvements = []
    for lbl in common:
        s_no = scores_nogpe[lbl]["score"]
        s_gpe = scores_gpe[lbl]["score"]
        imp_abs = s_gpe - s_no
        imp_rel = s_gpe / (s_no + 1e-8)
        improvements.append({
            "community": lbl,
            "size": scores_gpe[lbl]["size"],
            "score_nogpe": s_no,
            "score_gpe": s_gpe,
            "improvement_abs": imp_abs,
            "improvement_rel": imp_rel,
        })

    # Sort by absolute improvement (descending)
    improvements.sort(key=lambda x: x["improvement_abs"], reverse=True)
    selected = improvements[:top_k]
    return selected, improvements


def make_focused_figure(
    tsne_nogpe, tsne_gpe, labels,
    selected_communities, dataset_display_name, save_path,
):
    """Generate focused t-SNE figure showing only selected communities vividly."""
    selected_ids = set(c["community"] for c in selected_communities)

    # Color mapping: selected communities get vivid colors, rest gray
    color_map = {}
    ci = 0
    for c in selected_communities:
        color_map[c["community"]] = matplotlib.colors.to_rgba(VIVID_COLORS[ci % len(VIVID_COLORS)], alpha=0.85)
        ci += 1

    # Prepare colors and sizes arrays
    n = len(labels)
    colors_nogpe = np.zeros((n, 4))
    colors_gpe = np.zeros((n, 4))
    sizes = np.ones(n) * 2  # small dots for background

    for i in range(n):
        if labels[i] in selected_ids:
            c = color_map[labels[i]]
            colors_nogpe[i] = c
            colors_gpe[i] = c
            sizes[i] = 12  # larger dots for selected
        else:
            colors_nogpe[i] = (0.85, 0.85, 0.85, 0.08)  # very faint gray
            colors_gpe[i] = (0.85, 0.85, 0.85, 0.08)

    # Z-order: background first, then selected communities
    bg_mask = ~np.isin(labels, list(selected_ids))
    fg_mask = np.isin(labels, list(selected_ids))
    order = np.concatenate([np.where(bg_mask)[0], np.where(fg_mask)[0]])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, emb, colors_arr, title in [
        (axes[0], tsne_nogpe, colors_nogpe, "Without GPE"),
        (axes[1], tsne_gpe, colors_gpe, "With GPE"),
    ]:
        ax.scatter(
            emb[order, 0], emb[order, 1],
            c=colors_arr[order],
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
        legend_elements.append(
            Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=matplotlib.colors.to_hex(color_map[lbl][:3]),
                   markersize=8,
                   label=f"Comm {lbl} (n={c_info['size']}, +{c_info['improvement_abs']:.2f})")
        )
    legend_elements.append(
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=(0.8, 0.8, 0.8), markersize=6,
               label="Other communities")
    )

    fig.legend(
        handles=legend_elements, loc="lower center",
        ncol=min(len(legend_elements), 3),
        fontsize=9, frameon=True, fancybox=True,
        edgecolor="#cccccc",
        bbox_to_anchor=(0.5, -0.04),
    )

    plt.suptitle(
        f"t-SNE on {dataset_display_name} (Selected Communities)",
        fontsize=17, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.15,
                facecolor="white")
    plt.close(fig)
    print(f"  Saved figure -> {save_path}")


def process_dataset(ds_key):
    """Full pipeline for one dataset."""
    cfg = DATASETS[ds_key]
    display = cfg["display"]
    min_size = cfg["min_size"]
    top_k = cfg["top_k"]

    print(f"\n{'='*60}")
    print(f"  {display}")
    print(f"{'='*60}")

    # Load saved embeddings
    emb_path = os.path.join(EMB_DIR, f"embeddings_{ds_key}.npz")
    assert os.path.exists(emb_path), f"Missing: {emb_path}"
    data = np.load(emb_path)
    rep_gpe = data["rep_gpe"]
    rep_nogpe = data["rep_nogpe"]
    labels = data["labels"]
    print(f"  Loaded embeddings: {rep_gpe.shape[0]} nodes, {rep_gpe.shape[1]}-dim")
    print(f"  Communities: {len(np.unique(labels))} total")

    # Subsample for t-SNE if > 10k nodes
    n = rep_gpe.shape[0]
    if n > 10000:
        rng = np.random.default_rng(42)
        idx = rng.choice(n, size=10000, replace=False)
        rep_gpe_sub = rep_gpe[idx]
        rep_nogpe_sub = rep_nogpe[idx]
        labels_sub = labels[idx]
        print(f"  Subsampled to 10000 nodes for t-SNE")
    else:
        rep_gpe_sub = rep_gpe
        rep_nogpe_sub = rep_nogpe
        labels_sub = labels
        idx = np.arange(n)

    # Run t-SNE
    t0 = time.time()
    print(f"  Running t-SNE (GPE)...")
    tsne_gpe = TSNE(
        n_components=2, perplexity=30, random_state=42,
        max_iter=1000, learning_rate="auto", init="pca",
    ).fit_transform(rep_gpe_sub)
    print(f"  Running t-SNE (no GPE)...")
    tsne_nogpe = TSNE(
        n_components=2, perplexity=30, random_state=42,
        max_iter=1000, learning_rate="auto", init="pca",
    ).fit_transform(rep_nogpe_sub)
    print(f"  t-SNE done in {time.time()-t0:.0f}s")

    # Compute separation scores
    print(f"  Computing separation scores (min_size={min_size})...")
    scores_gpe = compute_separation_scores(tsne_gpe, labels_sub, min_community_size=min_size)
    scores_nogpe = compute_separation_scores(tsne_nogpe, labels_sub, min_community_size=min_size)
    print(f"  Valid communities (>={min_size} nodes): {len(scores_gpe)} with GPE, {len(scores_nogpe)} without")

    # Select top communities by GPE improvement
    selected, all_improvements = select_top_communities(scores_nogpe, scores_gpe, top_k=top_k)

    print(f"\n  Top {len(selected)} communities by GPE improvement:")
    print(f"  {'Comm':>6s}  {'Size':>5s}  {'NoGPE':>8s}  {'GPE':>8s}  {'Improve':>8s}  {'Rel':>6s}")
    for c in selected:
        print(f"  {c['community']:6d}  {c['size']:5d}  {c['score_nogpe']:8.3f}  "
              f"{c['score_gpe']:8.3f}  {c['improvement_abs']:+8.3f}  {c['improvement_rel']:6.2f}x")

    # Save analysis CSV
    csv_path = os.path.join(RESULTS_DIR, f"analysis_{ds_key}.csv")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(csv_path, "w", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=[
            "community", "size", "score_nogpe", "score_gpe",
            "improvement_abs", "improvement_rel",
        ])
        writer.writeheader()
        for row in all_improvements:
            writer.writerow({
                "community": row["community"],
                "size": row["size"],
                "score_nogpe": f"{row['score_nogpe']:.4f}",
                "score_gpe": f"{row['score_gpe']:.4f}",
                "improvement_abs": f"{row['improvement_abs']:.4f}",
                "improvement_rel": f"{row['improvement_rel']:.4f}",
            })
    print(f"  Saved analysis -> {csv_path}")

    # Generate focused figure
    fig_path = os.path.join(RESULTS_DIR, f"tsne_{ds_key}_focused.png")
    make_focused_figure(
        tsne_nogpe, tsne_gpe, labels_sub,
        selected, display, fig_path,
    )

    # Save t-SNE coordinates for reuse
    coords_path = os.path.join(RESULTS_DIR, f"tsne_coords_{ds_key}.npz")
    np.savez_compressed(
        coords_path,
        tsne_gpe=tsne_gpe, tsne_nogpe=tsne_nogpe,
        labels=labels_sub, idx=idx,
    )
    print(f"  Saved t-SNE coords -> {coords_path}")

    return {
        "dataset": ds_key,
        "display": display,
        "selected": selected,
        "all_improvements": all_improvements,
        "n_valid_communities": len(scores_gpe),
    }


def main():
    print("Focused t-SNE visualization pipeline")
    print(f"Loading embeddings from: {EMB_DIR}")
    print(f"Saving results to: {RESULTS_DIR}")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    all_results = {}
    total_t0 = time.time()

    for ds_key in DATASETS:
        result = process_dataset(ds_key)
        all_results[ds_key] = result

    # Save summary JSON
    summary_path = os.path.join(RESULTS_DIR, "summary.json")
    # Convert to serializable
    summary = {}
    for ds_key, res in all_results.items():
        summary[ds_key] = {
            "display": res["display"],
            "n_valid_communities": res["n_valid_communities"],
            "selected": res["selected"],
        }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved -> {summary_path}")

    # Print overall summary
    print(f"\n{'='*60}")
    print(f"  OVERALL SUMMARY")
    print(f"{'='*60}")
    for ds_key, res in all_results.items():
        sel = res["selected"]
        avg_imp = np.mean([c["improvement_abs"] for c in sel]) if sel else 0
        print(f"  {res['display']:15s}: selected {len(sel)} communities, "
              f"avg improvement = {avg_imp:+.3f}")

    print(f"\nTotal time: {time.time()-total_t0:.0f}s")
    print(f"Results in: {RESULTS_DIR}")


if __name__ == "__main__":
    main()

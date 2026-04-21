#!/usr/bin/env python3
"""t-SNE visualization: node embeddings with and without GPE on DBLP.

Trains two PACT models (use_gpe=True vs use_gpe=False) on DBLP rho=10,
extracts the GNN backbone output representations, and generates a side-by-side
t-SNE plot colored by Louvain community. This demonstrates that GPE improves
community separation in the learned embedding space.

Usage (on GPU server):
    conda activate pact
    cd /path/to/PACT
    python -m supplementary.run_tsne --device cuda

Output: tsne_gpe.png saved to RESULTS_DIR
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pact.main import _load_cached_sample
from pact.model import PACTConfig, PACT
from pact.train import TrainConfig, train_graph
from pact.losses import LossWeights
from pact.dgp import detect_communities


CACHE_DIR = "runs/data_cache"
RESULTS_DIR = "runs/tsne"
SEED = 0
RHO = 10


def train_and_extract(use_gpe: bool, device: str) -> tuple[np.ndarray, np.ndarray]:
    """Train PACT with/without GPE on DBLP, return (representations, community_labels)."""
    cache_path = os.path.join(CACHE_DIR, f"dblp_rho{RHO}_seed{SEED}.pt")
    assert os.path.exists(cache_path), f"Missing cache: {cache_path}"
    sample = _load_cached_sample(cache_path, device)

    model_cfg = PACTConfig(
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
        use_variance=True,
    )
    train_cfg = TrainConfig(
        epochs=200,
        lr=0.001,
        weight_decay=1e-4,
        train_frac=0.6,
        val_frac=0.2,
        learner="pact",
        loss_weights=LossWeights(mu=1.0, tau=1.0, sigma=0.25, prop=1.0),
        use_uncertainty_weighting=False,
        delta=0.01,
        select_metric="qini",
        normalize_y=True,
        log_every=10,
        seed=SEED,
    )

    model = PACT(model_cfg)
    result = train_graph(sample, model_cfg, train_cfg, device=device, model=model)

    # Reload best state dict
    best_state = result["best"]["state_dict"]
    model.load_state_dict(best_state)
    model = model.to(device)
    model.eval()

    # Extract representations from the backbone output (before X-learner heads)
    Xm = sample.X.mean(dim=0, keepdim=True)
    Xs = sample.X.std(dim=0, keepdim=True) + 1e-6
    X_normed = (sample.X - Xm) / Xs

    graph = sample.edge_index  # GAT backbone
    with torch.no_grad():
        rep = model.representation(X_normed, graph, sample.pos if use_gpe else None)

    rep_np = rep.detach().cpu().numpy()

    # Recover community labels from adjacency structure.
    # Reconstruct scipy sparse adjacency from edge_index.
    import scipy.sparse as sp
    ei = sample.edge_index.cpu().numpy()
    N = sample.X.size(0)
    A = sp.coo_matrix(
        (np.ones(ei.shape[1], dtype="float32"), (ei[0], ei[1])),
        shape=(N, N),
    ).tocsr()
    # Symmetrize
    A = ((A + A.T) > 0).astype("float32")
    node_class, _ = detect_communities(A)

    return rep_np, node_class


def make_tsne_figure(
    rep_gpe: np.ndarray,
    rep_nogpe: np.ndarray,
    labels: np.ndarray,
    save_path: str,
):
    """Generate a side-by-side t-SNE figure."""
    # Keep only communities with enough nodes for visibility
    unique, counts = np.unique(labels, return_counts=True)
    # Keep top-8 largest communities, merge rest into "other"
    top_k = 8
    top_communities = unique[np.argsort(-counts)[:top_k]]
    labels_vis = labels.copy()
    labels_vis[~np.isin(labels, top_communities)] = -1

    # Subsample for t-SNE speed if needed (17k nodes is fine)
    n = rep_gpe.shape[0]
    if n > 10000:
        rng = np.random.default_rng(42)
        idx = rng.choice(n, size=10000, replace=False)
        rep_gpe_sub = rep_gpe[idx]
        rep_nogpe_sub = rep_nogpe[idx]
        labels_sub = labels_vis[idx]
    else:
        rep_gpe_sub = rep_gpe
        rep_nogpe_sub = rep_nogpe
        labels_sub = labels_vis

    # Run t-SNE
    print("Running t-SNE for GPE model...")
    tsne_gpe = TSNE(
        n_components=2, perplexity=30, random_state=42,
        n_iter=1000, learning_rate="auto", init="pca",
    ).fit_transform(rep_gpe_sub)

    print("Running t-SNE for No-GPE model...")
    tsne_nogpe = TSNE(
        n_components=2, perplexity=30, random_state=42,
        n_iter=1000, learning_rate="auto", init="pca",
    ).fit_transform(rep_nogpe_sub)

    # Build color map
    unique_labels = np.unique(labels_sub)
    # Use a qualitative colormap
    cmap = plt.cm.get_cmap("Set1", len(unique_labels))
    color_map = {}
    ci = 0
    for lbl in sorted(unique_labels):
        if lbl == -1:
            color_map[lbl] = (0.8, 0.8, 0.8, 0.3)  # light gray for "other"
        else:
            color_map[lbl] = (*cmap(ci)[:3], 0.6)
            ci += 1
    colors = np.array([color_map[l] for l in labels_sub])

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Sort so "other" is drawn first (behind)
    order = np.argsort(labels_sub == -1)  # -1 last => reversed
    order_bg_first = np.argsort(-(labels_sub == -1).astype(int))

    for ax, emb, title in [
        (axes[0], tsne_nogpe, "Without GPE"),
        (axes[1], tsne_gpe, "With GPE"),
    ]:
        ax.scatter(
            emb[order_bg_first, 0],
            emb[order_bg_first, 1],
            c=colors[order_bg_first],
            s=4,
            edgecolors="none",
            rasterized=True,
        )
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)

    # Add a legend for top communities
    from matplotlib.lines import Line2D
    legend_elements = []
    for lbl in sorted(unique_labels):
        if lbl == -1:
            legend_elements.append(
                Line2D([0], [0], marker="o", color="w",
                       markerfacecolor=(0.8, 0.8, 0.8), markersize=6, label="Other")
            )
        else:
            legend_elements.append(
                Line2D([0], [0], marker="o", color="w",
                       markerfacecolor=color_map[lbl][:3], markersize=6,
                       label=f"Community {lbl}")
            )
    fig.legend(
        handles=legend_elements, loc="lower center", ncol=min(len(legend_elements), 5),
        fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.02),
    )

    plt.suptitle("t-SNE of Node Representations on DBLP", fontsize=15, y=1.02)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.1)
    print(f"Saved t-SNE figure to {save_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Train with GPE
    print("\n" + "="*60)
    print("  Training PACT WITH GPE on DBLP (rho=10, seed=0)")
    print("="*60)
    t0 = time.time()
    rep_gpe, labels = train_and_extract(use_gpe=True, device=args.device)
    print(f"  GPE model done in {time.time()-t0:.0f}s  rep shape={rep_gpe.shape}")

    # Train without GPE
    print("\n" + "="*60)
    print("  Training PACT WITHOUT GPE on DBLP (rho=10, seed=0)")
    print("="*60)
    t0 = time.time()
    rep_nogpe, labels2 = train_and_extract(use_gpe=False, device=args.device)
    print(f"  No-GPE model done in {time.time()-t0:.0f}s  rep shape={rep_nogpe.shape}")

    # Save raw embeddings for possible reuse
    np.savez_compressed(
        os.path.join(RESULTS_DIR, "embeddings.npz"),
        rep_gpe=rep_gpe, rep_nogpe=rep_nogpe, labels=labels,
    )

    # Generate t-SNE figure
    save_path = os.path.join(RESULTS_DIR, "tsne_gpe.png")
    make_tsne_figure(rep_gpe, rep_nogpe, labels, save_path)


if __name__ == "__main__":
    main()

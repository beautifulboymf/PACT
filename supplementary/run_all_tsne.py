#!/usr/bin/env python3
"""t-SNE visualization of node representations for ALL 5 graph datasets.

Generates side-by-side t-SNE plots (Without GPE vs With GPE) colored by
Louvain community for: CoraFull, DBLP, PubMed, BlogCatalog, Flickr.

Usage (on GPU server):
    source conda activate pact
    cd /path/to/PACT
    python supplementary/run_all_tsne.py --device cuda

Output: tsne_<dataset>.png files in RESULTS_DIR
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch
import scipy.sparse as sp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.manifold import TSNE

import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pact.main import _load_cached_sample
from pact.model import PACTConfig, PACT
from pact.train import TrainConfig, train_graph
from pact.losses import LossWeights
from pact.dgp import detect_communities
from pact.data import load_wsdm_dataset

CACHE_DIR = "runs/data_cache"
WSDM_DIR = os.environ.get("WSDM_DATA_DIR", "data/wsdm_datasets")
RESULTS_DIR = "runs/tsne_all"
SEED = 0
RHO = 10


# --------------------------------------------------------------------------- #
# Dataset configurations: model/training configs matching server YAML files
# --------------------------------------------------------------------------- #
DATASET_CONFIGS = {
    "cora_full": {
        "display_name": "CoraFull",
        "type": "dgp",
        "cache_file": f"cora_full_rho{RHO}_seed{SEED}.pt",
        "in_dim": 256,
        "model": dict(
            pos_dim=128, fusion_embed_dim=256, fusion_heads=4,
            backbone="gat",
            gnn_hidden=(256, 128, 128, 128, 128), gnn_heads=4,
            mlp_hidden=(128, 128, 128, 128),
            dropout=0.0, use_variance=True,
        ),
        "train": dict(
            epochs=200, lr=0.001, weight_decay=1e-4,
            train_frac=0.6, val_frac=0.2, learner="pact",
            loss_weights=LossWeights(mu=1.0, tau=1.0, sigma=0.25, prop=1.0),
            use_uncertainty_weighting=False,
            delta=0.01, select_metric="qini",
            normalize_y=True, log_every=20, seed=SEED,
        ),
    },
    "dblp": {
        "display_name": "DBLP",
        "type": "dgp",
        "cache_file": f"dblp_rho{RHO}_seed{SEED}.pt",
        "in_dim": 256,
        "model": dict(
            pos_dim=128, fusion_embed_dim=256, fusion_heads=4,
            backbone="gat",
            gnn_hidden=(256, 128, 128, 128, 128), gnn_heads=4,
            mlp_hidden=(128, 128, 128, 128),
            dropout=0.0, use_variance=True,
        ),
        "train": dict(
            epochs=200, lr=0.001, weight_decay=1e-4,
            train_frac=0.6, val_frac=0.2, learner="pact",
            loss_weights=LossWeights(mu=1.0, tau=1.0, sigma=0.25, prop=1.0),
            use_uncertainty_weighting=False,
            delta=0.01, select_metric="qini",
            normalize_y=True, log_every=20, seed=SEED,
        ),
    },
    "pubmed": {
        "display_name": "PubMed",
        "type": "dgp",
        "cache_file": f"pubmed_rho{RHO}_seed{SEED}.pt",
        "in_dim": 256,
        "model": dict(
            pos_dim=128, fusion_embed_dim=256, fusion_heads=4,
            backbone="gat",
            gnn_hidden=(256, 128, 128, 128, 128), gnn_heads=4,
            mlp_hidden=(128, 128, 128, 128),
            dropout=0.0, use_variance=True,
        ),
        "train": dict(
            epochs=200, lr=0.001, weight_decay=1e-4,
            train_frac=0.6, val_frac=0.2, learner="pact",
            loss_weights=LossWeights(mu=1.0, tau=1.0, sigma=0.25, prop=1.0),
            use_uncertainty_weighting=False,
            delta=0.01, select_metric="qini",
            normalize_y=True, log_every=20, seed=SEED,
        ),
    },
    "blogcatalog": {
        "display_name": "BlogCatalog",
        "type": "wsdm",
        "wsdm_name": "BlogCatalog",
        "extra_str": "1",
        "in_dim": 2160,  # X_100 dim for BlogCatalog
        "model": dict(
            pos_dim=128, fusion_embed_dim=256, fusion_heads=4,
            backbone="gat",
            gnn_hidden=(256, 128), gnn_heads=4,
            mlp_hidden=(128, 128),
            dropout=0.1, use_variance=True,
        ),
        "train": dict(
            epochs=200, lr=0.001, weight_decay=1e-4,
            train_frac=0.6, val_frac=0.2, learner="pact",
            loss_weights=LossWeights(mu=1.0, tau=1.0, sigma=0.25, prop=1.0),
            use_uncertainty_weighting=False,
            delta=0.01, select_metric="pehe",
            normalize_y=True, log_every=20, seed=SEED,
        ),
    },
    "flickr": {
        "display_name": "Flickr",
        "type": "wsdm",
        "wsdm_name": "Flickr",
        "extra_str": "1",
        "in_dim": 1205,  # X_100 dim for Flickr
        "model": dict(
            pos_dim=128, fusion_embed_dim=256, fusion_heads=4,
            backbone="gat",
            gnn_hidden=(256, 128), gnn_heads=4,
            mlp_hidden=(128, 128),
            dropout=0.1, use_variance=True,
        ),
        "train": dict(
            epochs=200, lr=0.001, weight_decay=1e-4,
            train_frac=0.6, val_frac=0.2, learner="pact",
            loss_weights=LossWeights(mu=1.0, tau=1.0, sigma=0.25, prop=1.0),
            use_uncertainty_weighting=False,
            delta=0.01, select_metric="pehe",
            normalize_y=True, log_every=20, seed=SEED,
        ),
    },
}


def load_data(ds_key: str, device: str):
    """Load graph data and return (sample, adjacency_scipy)."""
    cfg = DATASET_CONFIGS[ds_key]

    if cfg["type"] == "dgp":
        cache_path = os.path.join(CACHE_DIR, cfg["cache_file"])
        assert os.path.exists(cache_path), f"Missing cache: {cache_path}"
        sample = _load_cached_sample(cache_path, device)
        # Reconstruct scipy adjacency from edge_index
        ei = sample.edge_index.cpu().numpy()
        N = sample.X.size(0)
        A = sp.coo_matrix(
            (np.ones(ei.shape[1], dtype="float32"), (ei[0], ei[1])),
            shape=(N, N),
        ).tocsr()
        A = ((A + A.T) > 0).astype("float32")
    else:
        # WSDM dataset
        sample = load_wsdm_dataset(
            data_dir=WSDM_DIR,
            name=cfg["wsdm_name"],
            extra_str=cfg["extra_str"],
            exp_id=0,
            pos_dim=128,
            device=device,
        )
        from scipy.io import loadmat
        mat_dir = os.path.join(WSDM_DIR, f"{cfg['wsdm_name']}{cfg['extra_str']}")
        mat_path = os.path.join(mat_dir, f"{cfg['wsdm_name']}0.mat")
        data = loadmat(mat_path)
        A = data["Network"]
        if not sp.issparse(A):
            A = sp.csr_matrix(A)
        A = ((A + A.T) > 0).astype("float32")

    return sample, A


def train_and_extract(ds_key: str, use_gpe: bool, sample, device: str) -> np.ndarray:
    """Train PACT with/without GPE, return backbone representations."""
    cfg = DATASET_CONFIGS[ds_key]

    model_cfg = PACTConfig(
        in_dim=cfg["in_dim"],
        use_gpe=use_gpe,
        **cfg["model"],
    )
    train_cfg = TrainConfig(**cfg["train"])

    model = PACT(model_cfg)
    result = train_graph(sample, model_cfg, train_cfg, device=device, model=model)

    # Reload best state dict
    best_state = result["best"]["state_dict"]
    model.load_state_dict(best_state)
    model = model.to(device)
    model.eval()

    # Extract backbone representations
    Xm = sample.X.mean(dim=0, keepdim=True)
    Xs = sample.X.std(dim=0, keepdim=True) + 1e-6
    X_normed = (sample.X - Xm) / Xs

    with torch.no_grad():
        rep = model.representation(
            X_normed, sample.edge_index,
            sample.pos if use_gpe else None,
        )

    return rep.detach().cpu().numpy()


def make_tsne_figure(
    rep_gpe: np.ndarray,
    rep_nogpe: np.ndarray,
    labels: np.ndarray,
    dataset_name: str,
    save_path: str,
):
    """Generate a side-by-side t-SNE figure matching the DBLP style."""
    # Keep top-8 largest communities, merge rest into "other"
    unique, counts = np.unique(labels, return_counts=True)
    top_k = 8
    top_communities = unique[np.argsort(-counts)[:top_k]]
    labels_vis = labels.copy()
    labels_vis[~np.isin(labels, top_communities)] = -1

    # Subsample for t-SNE speed if > 10k nodes
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
    print(f"  [{dataset_name}] Running t-SNE for GPE model...")
    tsne_gpe = TSNE(
        n_components=2, perplexity=30, random_state=42,
        max_iter=1000, learning_rate="auto", init="pca",
    ).fit_transform(rep_gpe_sub)

    print(f"  [{dataset_name}] Running t-SNE for No-GPE model...")
    tsne_nogpe = TSNE(
        n_components=2, perplexity=30, random_state=42,
        max_iter=1000, learning_rate="auto", init="pca",
    ).fit_transform(rep_nogpe_sub)

    # Build color map (Set1 qualitative palette)
    unique_labels = np.unique(labels_sub)
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

    # Plot -- draw "other" points first (background)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    order_bg_first = np.argsort(-(labels_sub == -1).astype(int))

    for ax, emb, title in [
        (axes[0], tsne_nogpe, "Without GPE"),
        (axes[1], tsne_gpe, "With GPE"),
    ]:
        ax.scatter(
            emb[order_bg_first, 0], emb[order_bg_first, 1],
            c=colors[order_bg_first], s=4, edgecolors="none", rasterized=True,
        )
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    # Legend
    legend_elements = []
    for lbl in sorted(unique_labels):
        if lbl == -1:
            legend_elements.append(
                Line2D([0], [0], marker="o", color="w",
                       markerfacecolor=(0.8, 0.8, 0.8), markersize=6,
                       label="Other")
            )
        else:
            legend_elements.append(
                Line2D([0], [0], marker="o", color="w",
                       markerfacecolor=color_map[lbl][:3], markersize=6,
                       label=f"Community {lbl}")
            )
    fig.legend(
        handles=legend_elements, loc="lower center",
        ncol=min(len(legend_elements), 5),
        fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.02),
    )

    plt.suptitle(
        f"t-SNE of Node Representations on {dataset_name}",
        fontsize=15, y=1.02,
    )
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.1,
                facecolor="white")
    print(f"  [{dataset_name}] Saved figure -> {save_path}")
    plt.close(fig)


def process_dataset(ds_key: str, device: str, skip_existing: bool = False):
    """Full pipeline for one dataset: load -> train -> extract -> t-SNE."""
    cfg = DATASET_CONFIGS[ds_key]
    display_name = cfg["display_name"]
    save_path = os.path.join(RESULTS_DIR, f"tsne_{ds_key}.png")

    if skip_existing and os.path.exists(save_path):
        print(f"\n[SKIP] {display_name} -- figure already exists: {save_path}")
        return

    print(f"\n{'='*60}")
    print(f"  Processing {display_name}")
    print(f"{'='*60}")

    # Load data
    t0 = time.time()
    sample, A_scipy = load_data(ds_key, device)
    print(f"  Data loaded: N={sample.X.size(0)}, "
          f"X_dim={sample.X.size(1)}, edges={sample.edge_index.size(1)}")

    # Detect communities
    node_class, _ = detect_communities(A_scipy)
    unique_comms = np.unique(node_class)
    print(f"  Communities detected: {len(unique_comms)} Louvain communities")

    # Train WITH GPE
    print(f"\n  Training PACT WITH GPE on {display_name}...")
    t1 = time.time()
    rep_gpe = train_and_extract(ds_key, use_gpe=True, sample=sample, device=device)
    print(f"  GPE model done in {time.time()-t1:.0f}s  rep shape={rep_gpe.shape}")

    # Train WITHOUT GPE
    print(f"\n  Training PACT WITHOUT GPE on {display_name}...")
    t1 = time.time()
    rep_nogpe = train_and_extract(ds_key, use_gpe=False, sample=sample, device=device)
    print(f"  No-GPE model done in {time.time()-t1:.0f}s  rep shape={rep_nogpe.shape}")

    # Save raw embeddings
    emb_path = os.path.join(RESULTS_DIR, f"embeddings_{ds_key}.npz")
    np.savez_compressed(
        emb_path,
        rep_gpe=rep_gpe, rep_nogpe=rep_nogpe, labels=node_class,
    )
    print(f"  Saved embeddings -> {emb_path}")

    # Generate t-SNE figure
    make_tsne_figure(rep_gpe, rep_nogpe, node_class, display_name, save_path)

    print(f"  Total time for {display_name}: {time.time()-t0:.0f}s")


def main():
    parser = argparse.ArgumentParser(
        description="Generate t-SNE visualizations for all 5 graph datasets",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--datasets", nargs="+", default=None,
        help="Subset of datasets to process (default: all 5). "
             "Choices: cora_full dblp pubmed blogcatalog flickr",
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Skip datasets whose figures already exist",
    )
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    datasets = args.datasets or list(DATASET_CONFIGS.keys())
    print(f"Will process {len(datasets)} dataset(s): {datasets}")
    print(f"Device: {args.device}")
    print(f"Results dir: {RESULTS_DIR}")

    total_t0 = time.time()
    for ds_key in datasets:
        assert ds_key in DATASET_CONFIGS, \
            f"Unknown dataset: {ds_key}. Choose from {list(DATASET_CONFIGS.keys())}"
        process_dataset(ds_key, args.device, skip_existing=args.skip_existing)

    print(f"\nAll done! Total time: {time.time()-total_t0:.0f}s")
    print(f"Figures saved in: {RESULTS_DIR}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""PE method comparison: node2vec vs degree+PageRank vs Laplacian eigenvectors.

Tests whether the specific PE method matters, or any positional signal helps.
Runs TARNet with GPE using different positional inputs on DBLP, rho=10, 5 seeds.

Run: python -m pact.experiments.run_pe_comparison --device cuda
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import scipy.sparse as sp



# Module-level flag set by compute_laplacian_pe when scipy eigsh fails
# and the function falls back to a random embedding. Callers read this
# flag to record the fallback in their result JSON (transparency).
_LAPLACIAN_FALLBACK_USED = False

def compute_degree_pagerank(edge_index, num_nodes, dim=128):
    """Compute degree + PageRank as a simple positional encoding."""
    # Degree
    degree = torch.zeros(num_nodes)
    degree.index_add_(0, edge_index[1], torch.ones(edge_index.size(1)))

    # PageRank via power iteration
    row, col = edge_index[0].numpy(), edge_index[1].numpy()
    A = sp.csr_matrix((np.ones(len(row)), (row, col)), shape=(num_nodes, num_nodes))
    d = np.array(A.sum(axis=1)).flatten()
    d[d == 0] = 1
    D_inv = sp.diags(1.0 / d)
    M = D_inv @ A

    alpha = 0.85
    pr = np.ones(num_nodes) / num_nodes
    for _ in range(50):
        pr = alpha * (M.T @ pr) + (1 - alpha) / num_nodes

    # Expand to dim dimensions via random projection for fair comparison
    rng = np.random.default_rng(42)
    base = np.stack([degree.numpy(), pr, np.log1p(degree.numpy()), pr ** 2], axis=1)  # [N, 4]
    proj = rng.standard_normal((4, dim)).astype(np.float32)
    pos = (base @ proj).astype(np.float32)
    # Normalize
    pos = pos / (np.linalg.norm(pos, axis=1, keepdims=True) + 1e-8)
    return torch.from_numpy(pos)


def compute_laplacian_pe(edge_index, num_nodes, dim=128):
    """Compute Laplacian eigenvector positional encoding."""
    row, col = edge_index[0].numpy(), edge_index[1].numpy()
    A = sp.csr_matrix((np.ones(len(row)), (row, col)), shape=(num_nodes, num_nodes))
    A = A + A.T  # symmetrize
    A[A > 1] = 1

    d = np.array(A.sum(axis=1)).flatten()
    d[d == 0] = 1
    D_inv_sqrt = sp.diags(1.0 / np.sqrt(d))
    L_norm = sp.eye(num_nodes) - D_inv_sqrt @ A @ D_inv_sqrt

    # Compute smallest eigenvectors (skip the trivial one)
    from scipy.sparse.linalg import eigsh
    k = min(dim + 1, num_nodes - 1)
    try:
        eigenvalues, eigenvectors = eigsh(L_norm, k=k, which='SM', maxiter=5000)
        # Skip first (trivial) eigenvector
        pos = eigenvectors[:, 1:dim+1].astype(np.float32)
        # Pad if fewer than dim eigenvectors
        if pos.shape[1] < dim:
            pad = np.zeros((num_nodes, dim - pos.shape[1]), dtype=np.float32)
            pos = np.concatenate([pos, pad], axis=1)
    except Exception as e:
        print(f"  Laplacian eigsh failed: {e}, falling back to random")
        global _LAPLACIAN_FALLBACK_USED
        _LAPLACIAN_FALLBACK_USED = True
        pos = np.random.randn(num_nodes, dim).astype(np.float32)

    return torch.from_numpy(pos)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out-dir", default="results/pe_comparison")
    parser.add_argument("--cache-dir", default=None)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    proj_root = Path(__file__).resolve().parents[2]
    os.chdir(proj_root)

    # Import after chdir
    sys.path.insert(0, str(proj_root))
    from pact.data import build_graph_uplift_sample
    from pact.model import PACTConfig, TARNetGraph
    from pact.train import TrainConfig, LossWeights, train_graph

    import yaml
    config_path = "pact/configs/server/dblp.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    SEEDS = [0, 1, 2, 3, 4]
    RHO = 10.0
    PE_METHODS = ["node2vec", "degree_pagerank", "laplacian"]

    results = {m: [] for m in PE_METHODS}

    for pe_method in PE_METHODS:
        for seed in SEEDS:
            out_file = f"{args.out_dir}/dblp_{pe_method}_s{seed}.json"
            if os.path.exists(out_file):
                print(f"  Skipping (exists): {out_file}")
                with open(out_file) as f:
                    d = json.load(f)
                results[pe_method].append(d)
                continue

            print(f"\n=== {pe_method} seed={seed} ===")

            # Build sample
            dgp_cfg = cfg.get("dgp", {})
            dgp_cfg["rho"] = RHO
            ds_cfg = cfg["dataset"]

            sample = build_graph_uplift_sample(
                name=ds_cfg["name"],
                root=ds_cfg["root"],
                dgp_cfg=dgp_cfg,
                pos_path=ds_cfg.get("pos_emb_path"),
                seed=seed,
                device=args.device,
            )

            # Replace positional encoding based on method
            if pe_method == "node2vec":
                pass  # Use default node2vec from sample.pos
            elif pe_method == "degree_pagerank":
                sample.pos = compute_degree_pagerank(
                    sample.edge_index.cpu(), sample.X.size(0), dim=128
                ).to(args.device)
            elif pe_method == "laplacian":
                import pact.experiments.run_pe_comparison as _self_mod
                _self_mod._LAPLACIAN_FALLBACK_USED = False
                sample.pos = compute_laplacian_pe(
                    sample.edge_index.cpu(), sample.X.size(0), dim=128
                ).to(args.device)

            # Build model
            mdl_cfg = cfg.get("model", {})
            model_cfg = PACTConfig(in_dim=sample.X.size(1), **mdl_cfg)
            model = TARNetGraph(model_cfg)

            tr_cfg = cfg.get("train", {})
            tr_cfg["seed"] = seed
            tr_cfg["learner"] = "tarnet"
            train_cfg = TrainConfig(
                loss_weights=LossWeights(**tr_cfg.get("loss_weights", {})),
                **{k: v for k, v in tr_cfg.items() if k != "loss_weights"},
            )

            result = train_graph(sample, model_cfg, train_cfg, device=args.device, model=model)

            # Save — result structure is {"best": {"test": {"pehe": ..., "ate": ..., "qini": ...}}}
            best = result.get("best", {})
            test_metrics = best.get("test", {})
            # Record whether Laplacian eigsh fell back to random (transparency).
            _fallback = None
            if pe_method == "laplacian":
                import pact.experiments.run_pe_comparison as _self_mod
                _fallback = "random" if getattr(_self_mod, "_LAPLACIAN_FALLBACK_USED", False) else None
            out_data = {
                "pe_method": pe_method,
                "seed": seed,
                "pe_fallback": _fallback,
                "test": {
                    "pehe": float(test_metrics.get("pehe", 0)),
                    "ate": float(test_metrics.get("ate", 0)),
                    "qini": float(test_metrics.get("qini", 0)),
                }
            }
            with open(out_file, "w") as f:
                json.dump(out_data, f, indent=2)
            results[pe_method].append(out_data)
            print(f"  PEHE={out_data['test']['pehe']:.3f}")

    # Summary
    print("\n" + "=" * 60)
    print("PE COMPARISON: TARNet+GPE on DBLP, rho=10, 5 seeds")
    print("=" * 60)
    for method in PE_METHODS:
        if results[method]:
            pehes = [r["test"]["pehe"] for r in results[method]]
            m, s = np.mean(pehes), np.std(pehes)
            print(f"  {method:>20}: PEHE = {m:.3f} +/- {s:.3f} ({len(pehes)} seeds)")


if __name__ == "__main__":
    main()

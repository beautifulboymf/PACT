#!/usr/bin/env python3
"""Wall-clock benchmark: vanilla TARNet vs TARNet+GPE on CoraFull (rho=10).

5-layer GAT backbone matching the paper default; 200 epochs; 3 seeds.
Reports per-seed and mean elapsed seconds for each variant.

Usage (server): python -m supplementary.bench_gpe_walltime --device cuda
"""
from __future__ import annotations

import json
import os
import sys
import time

import torch

import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pact.main import _load_cached_sample
from pact.model import PACTConfig, TARNetGraph
from pact.train import TrainConfig, train_graph
from pact.losses import LossWeights

CACHE_DIR = "runs/data_cache"
RESULTS_DIR = "runs/bench_walltime"
DATASET = "cora_full"
RHO = 10
SEEDS = [0, 1, 2]
GNN_HIDDEN = (256, 128, 128, 128, 128)


def run_one(use_gpe: bool, seed: int, device: str) -> dict:
    cache_path = os.path.join(CACHE_DIR, f"{DATASET}_rho{RHO}_seed{seed}.pt")
    assert os.path.exists(cache_path), cache_path
    sample = _load_cached_sample(cache_path, device)

    model_cfg = PACTConfig(
        in_dim=sample.X.size(1),
        pos_dim=128,
        fusion_embed_dim=256,
        fusion_heads=4,
        use_gpe=use_gpe,
        backbone="gat",
        gnn_hidden=GNN_HIDDEN,
        gnn_heads=4,
        mlp_hidden=(128, 128, 128, 128),
        dropout=0.0,
        use_variance=False,
    )
    train_cfg = TrainConfig(
        epochs=200,
        lr=0.001,
        weight_decay=1e-4,
        train_frac=0.6,
        val_frac=0.2,
        learner="tarnet",
        loss_weights=LossWeights(mu=1.0, tau=1.0, sigma=0.25, prop=1.0),
        use_uncertainty_weighting=False,
        delta=0.01,
        select_metric="pehe",
        normalize_y=True,
        log_every=10000,
        seed=seed,
    )

    model = TARNetGraph(model_cfg)
    # GPU warmup + sync for accurate timing
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    t0 = time.time()
    result = train_graph(sample, model_cfg, train_cfg, device=device, model=model)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = time.time() - t0
    best = result["best"]
    return {
        "use_gpe": use_gpe,
        "seed": seed,
        "elapsed_s": elapsed,
        "test_pehe": best["test"]["pehe"],
    }


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} dataset={DATASET} rho={RHO} seeds={SEEDS}")

    all_res = []
    for use_gpe in [False, True]:
        label = "+GPE" if use_gpe else "vanilla"
        for seed in SEEDS:
            print(f"\n--- {label} seed={seed} ---")
            res = run_one(use_gpe, seed, device)
            print(f"  elapsed={res['elapsed_s']:.1f}s  pehe={res['test_pehe']:.4f}")
            all_res.append(res)

    with open(os.path.join(RESULTS_DIR, "bench.json"), "w") as f:
        json.dump(all_res, f, indent=2)

    # Summary
    van = [r["elapsed_s"] for r in all_res if not r["use_gpe"]]
    gpe = [r["elapsed_s"] for r in all_res if r["use_gpe"]]
    mean_v = sum(van) / len(van)
    mean_g = sum(gpe) / len(gpe)
    overhead = (mean_g - mean_v) / mean_v * 100
    print(f"\nvanilla mean: {mean_v:.1f}s ({van})")
    print(f"+GPE    mean: {mean_g:.1f}s ({gpe})")
    print(f"overhead: {overhead:+.1f}%")


if __name__ == "__main__":
    main()

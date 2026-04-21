"""Pre-generate and cache all (dataset, rho, seed) DGP instances as .pt files.

Usage::

    python -m pact.pregenerate --out-dir runs/data_cache

Generates 27 files: {dataset}_rho{rho}_seed{seed}.pt, each containing a
serialized GraphUpliftSample. The training script can then load these
directly instead of re-running the expensive DGP (Louvain + SVD + multi-hop).
"""

from __future__ import annotations

import argparse
import os
import time

import torch
import yaml

from .data import build_graph_uplift_sample


DATASETS = {
    "cora_full": "pact/configs/server/cora_full.yaml",
    "dblp": "pact/configs/server/dblp.yaml",
    "pubmed": "pact/configs/server/pubmed.yaml",
}
RHOS = [5, 10, 15, 20, 30]
SEEDS = [0, 1, 2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=str, default="runs/data_cache")
    parser.add_argument("--datasets", type=str, default="all")
    parser.add_argument("--rhos", type=str, default="5,10,15,20,30")
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--community-detection", type=str, default="louvain",
                        choices=["louvain", "leiden", "infomap"],
                        help="community-detection algorithm (louvain = paper main results)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    datasets = list(DATASETS.keys()) if args.datasets == "all" else args.datasets.split(",")
    rhos = [float(x) for x in args.rhos.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]

    comm = args.community_detection.lower()
    comm_suffix = "" if comm == "louvain" else f"_comm{comm}"

    total = len(datasets) * len(rhos) * len(seeds)
    done = 0
    for ds in datasets:
        cfg_path = DATASETS[ds]
        cfg = yaml.safe_load(open(cfg_path))
        ds_cfg = cfg["dataset"]
        dgp_cfg = cfg["dgp"]

        for rho in rhos:
            for seed in seeds:
                out_path = os.path.join(
                    args.out_dir,
                    f"{ds}_rho{int(rho)}{comm_suffix}_seed{seed}.pt",
                )
                if os.path.exists(out_path):
                    done += 1
                    print(f"[skip] {out_path} exists ({done}/{total})")
                    continue

                dgp_cfg_copy = dict(dgp_cfg)
                dgp_cfg_copy["rho"] = rho
                dgp_cfg_copy["community_detection"] = comm

                t0 = time.time()
                sample = build_graph_uplift_sample(
                    name=ds_cfg["name"],
                    root=ds_cfg["root"],
                    dgp_cfg=dgp_cfg_copy,
                    pos_path=ds_cfg.get("pos_emb_path"),
                    seed=seed,
                    device="cpu",  # save on CPU, move to GPU at train time
                )
                elapsed = time.time() - t0

                # Save as dict of tensors
                payload = {
                    "X": sample.X,
                    "pos": sample.pos,
                    "edge_index": sample.edge_index,
                    "A_norm_indices": sample.A_norm.coalesce().indices(),
                    "A_norm_values": sample.A_norm.coalesce().values(),
                    "A_norm_shape": list(sample.A_norm.shape),
                    "T": sample.T,
                    "Y": sample.Y,
                    "Y0": sample.Y0,
                    "Y1": sample.Y1,
                    "true_tau": sample.true_tau,
                }
                torch.save(payload, out_path)
                done += 1
                print(f"[saved] {out_path} ({elapsed:.1f}s, {done}/{total})")

    print(f"\ndone — {done} files in {args.out_dir}")


if __name__ == "__main__":
    main()

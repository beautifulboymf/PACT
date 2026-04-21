#!/usr/bin/env python3
"""Graphformer comparison experiment (W3).

Tests whether GPE adds value on top of a position-aware backbone.
Runs TARNet with Graphformer backbone ± GPE on DBLP, rho=10, 5 seeds.

Run: python -m pact.experiments.run_graphformer_comparison --device cuda
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np


DATASET = "dblp"
SEEDS = [0, 1, 2, 3, 4]
RHO = 10.0
EPOCHS = 200
CONFIG_DIR = "pact/configs/server"


def run_one(config: str, seed: int, backbone: str, no_gpe: bool,
            out_json: str, device: str, cache_dir: str | None) -> None:
    cmd = [
        sys.executable, "-m", "pact.main",
        "--config", config,
        "--learner", "tarnet",
        "--backbone", backbone,
        "--rho", str(RHO),
        "--seed", str(seed),
        "--epochs", str(EPOCHS),
        "--device", device,
        "--out-json", out_json,
    ]
    if no_gpe:
        cmd.append("--no-gpe")
    if cache_dir:
        cmd.extend(["--cache-dir", cache_dir])

    tag = f"{backbone}_{'nogpe' if no_gpe else 'gpe'}"
    print(f"  Running: TARNet-{tag} seed={seed}")
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out-dir", default="results/graphformer_comparison")
    parser.add_argument("--cache-dir", default=None)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    proj_root = Path(__file__).resolve().parents[2]
    os.chdir(proj_root)

    config_path = f"{CONFIG_DIR}/{DATASET}.yaml"

    # 4 configurations: GAT ± GPE, Graphformer ± GPE
    configs = [
        {"name": "gat_nogpe",          "backbone": "gat",          "no_gpe": True},
        {"name": "gat_gpe",            "backbone": "gat",          "no_gpe": False},
        {"name": "graphformer_nogpe",  "backbone": "graphformer",  "no_gpe": True},
        {"name": "graphformer_gpe",    "backbone": "graphformer",  "no_gpe": False},
    ]

    total = len(configs) * len(SEEDS)
    done = 0

    for cfg in configs:
        for seed in SEEDS:
            out_json = f"{args.out_dir}/{DATASET}_{cfg['name']}_s{seed}.json"
            if os.path.exists(out_json):
                print(f"  Skipping (exists): {out_json}")
                done += 1
                continue
            try:
                run_one(config_path, seed, cfg["backbone"], cfg["no_gpe"],
                        out_json, args.device, args.cache_dir)
                done += 1
                print(f"  [{done}/{total}] done")
            except subprocess.CalledProcessError as e:
                print(f"  FAILED: {e}")

    # Print comparison table
    print("\n" + "=" * 80)
    print("GRAPHFORMER COMPARISON: TARNet on DBLP, rho=10, 5 seeds")
    print("=" * 80)

    labels = {
        "gat_nogpe": "TARNet (GAT)",
        "gat_gpe": "TARNet+GPE (GAT)",
        "graphformer_nogpe": "TARNet (Graphformer)",
        "graphformer_gpe": "TARNet+GPE (Graphformer)",
    }

    print(f"{'Configuration':<30} {'PEHE':<18} {'ATE':<18}")
    print("-" * 66)

    for cfg in configs:
        pehes, ates = [], []
        for seed in SEEDS:
            fpath = f"{args.out_dir}/{DATASET}_{cfg['name']}_s{seed}.json"
            if os.path.exists(fpath):
                with open(fpath) as fh:
                    data = json.load(fh)
                    pehes.append(data.get("test_pehe", data.get("best_pehe", float("nan"))))
                    ates.append(data.get("test_ate", data.get("best_ate", float("nan"))))
        if pehes:
            mp, sp = np.mean(pehes), np.std(pehes)
            ma, sa = np.mean(ates), np.std(ates)
            print(f"{labels[cfg['name']]:<30} {mp:.2f} +/- {sp:.2f}    {ma:.2f} +/- {sa:.2f}")
        else:
            print(f"{labels[cfg['name']]:<30} N/A")

    print("\nKey question: Does GPE improve Graphformer backbone?")
    print("If yes -> GPE provides value orthogonal to position-aware attention")


if __name__ == "__main__":
    main()

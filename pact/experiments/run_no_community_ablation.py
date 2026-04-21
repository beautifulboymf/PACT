#!/usr/bin/env python3
"""No-Community DGP Ablation (W1 defense).

Tests whether GPE still helps when community structure is removed from the DGP
(kappa4=0, intra_w=1.0). This addresses the DGP circularity concern.

Run: python -m pact.experiments.run_no_community_ablation --device cuda
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


DATASETS = ["cora_full", "dblp", "pubmed"]
SEEDS = [0, 1, 2, 3, 4]
RHO = 10.0
LEARNERS = ["tarnet", "gdc"]
EPOCHS = 200

# Config paths relative to project root
CONFIG_DIR = "pact/configs/server"


def run_one(config: str, learner: str, seed: int, no_gpe: bool,
            kappa4: float, intra_w: float, out_json: str, device: str) -> None:
    cmd = [
        sys.executable, "-m", "pact.main",
        "--config", config,
        "--learner", learner,
        "--rho", str(RHO),
        "--seed", str(seed),
        "--epochs", str(EPOCHS),
        "--kappa4", str(kappa4),
        "--intra-w", str(intra_w),
        "--device", device,
        "--out-json", out_json,
    ]
    if no_gpe:
        cmd.append("--no-gpe")
    # Don't use cache — different DGP params
    print(f"  Running: {learner} seed={seed} no_gpe={no_gpe} -> {out_json}")
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out-dir", default="results/no_community_ablation")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    proj_root = Path(__file__).resolve().parents[2]
    os.chdir(proj_root)

    results = []
    total = len(DATASETS) * len(LEARNERS) * 2 * len(SEEDS)
    done = 0

    for ds in DATASETS:
        config = f"{CONFIG_DIR}/{ds}.yaml"
        for learner in LEARNERS:
            for no_gpe in [True, False]:
                gpe_tag = "nogpe" if no_gpe else "gpe"
                for seed in SEEDS:
                    out_json = f"{args.out_dir}/{ds}_{learner}_{gpe_tag}_s{seed}.json"
                    if os.path.exists(out_json):
                        print(f"  Skipping (exists): {out_json}")
                        done += 1
                        continue
                    try:
                        run_one(config, learner, seed, no_gpe,
                                kappa4=0.0, intra_w=1.0, out_json=out_json,
                                device=args.device)
                        done += 1
                        print(f"  [{done}/{total}] done")
                    except subprocess.CalledProcessError as e:
                        print(f"  FAILED: {e}")

    # Aggregate results
    print("\nAggregating results...")
    all_results = {}
    for f in Path(args.out_dir).glob("*.json"):
        with open(f) as fh:
            data = json.load(fh)
            parts = f.stem.split("_")
            # e.g. cora_full_tarnet_gpe_s0 -> ds=cora_full, learner=tarnet, gpe=gpe, seed=0
            seed_part = parts[-1]  # s0
            gpe_part = parts[-2]   # gpe or nogpe
            learner_part = parts[-3]  # tarnet or gdc
            ds_part = "_".join(parts[:-3])  # cora_full

            key = f"{ds_part}_{learner_part}_{gpe_part}"
            if key not in all_results:
                all_results[key] = []
            all_results[key].append(data)

    # Print summary table
    print("\n" + "=" * 80)
    print("NO-COMMUNITY DGP ABLATION RESULTS (kappa4=0, intra_w=1.0, rho=10)")
    print("=" * 80)
    print(f"{'Dataset':<12} {'Model':<10} {'Vanilla PEHE':<18} {'+GPE PEHE':<18} {'GPE helps?':<10}")
    print("-" * 68)

    import numpy as np
    for ds in DATASETS:
        for learner in LEARNERS:
            key_nogpe = f"{ds}_{learner}_nogpe"
            key_gpe = f"{ds}_{learner}_gpe"
            if key_nogpe in all_results and key_gpe in all_results:
                pehe_nogpe = [r.get("test_pehe", r.get("best_pehe", float("nan"))) for r in all_results[key_nogpe]]
                pehe_gpe = [r.get("test_pehe", r.get("best_pehe", float("nan"))) for r in all_results[key_gpe]]
                m_nogpe, s_nogpe = np.mean(pehe_nogpe), np.std(pehe_nogpe)
                m_gpe, s_gpe = np.mean(pehe_gpe), np.std(pehe_gpe)
                helps = "YES" if m_gpe < m_nogpe else "NO"
                print(f"{ds:<12} {learner:<10} {m_nogpe:.2f} +/- {s_nogpe:.2f}    {m_gpe:.2f} +/- {s_gpe:.2f}    {helps}")

    # Save aggregated
    summary_path = f"{args.out_dir}/summary.json"
    with open(summary_path, "w") as fh:
        json.dump(all_results, fh, indent=2)
    print(f"\nSaved summary to {summary_path}")


if __name__ == "__main__":
    main()

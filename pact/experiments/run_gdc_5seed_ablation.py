#!/usr/bin/env python3
"""Re-run Table 5 (GDC ablation at rho=30) with 5 seeds (W7 fix).

Run: python -m pact.experiments.run_gdc_5seed_ablation --device cuda
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np


DATASETS = ["cora_full", "dblp", "pubmed"]
SEEDS = [0, 1, 2, 3, 4]
RHO = 30.0
EPOCHS = 200

CONFIG_DIR = "pact/configs/server"


def run_one(config: str, seed: int, no_gpe: bool, use_var: bool,
            out_json: str, device: str, cache_dir: str | None) -> None:
    cmd = [
        sys.executable, "-m", "pact.main",
        "--config", config,
        "--learner", "gdc",
        "--rho", str(RHO),
        "--seed", str(seed),
        "--epochs", str(EPOCHS),
        "--device", device,
        "--out-json", out_json,
    ]
    if no_gpe:
        cmd.append("--no-gpe")
    if use_var:
        cmd.append("--use-variance")
    else:
        cmd.append("--no-variance")
    if cache_dir:
        cmd.extend(["--cache-dir", cache_dir])

    tag = f"{'nogpe' if no_gpe else 'gpe'}_{'var' if use_var else 'novar'}"
    print(f"  Running: GDC {tag} seed={seed}")
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out-dir", default="results/gdc_ablation_5seed")
    parser.add_argument("--cache-dir", default=None)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    proj_root = Path(__file__).resolve().parents[2]
    os.chdir(proj_root)

    # 4 configurations: GDC, +GPE, +Var, +GPE+Var
    configs = [
        {"name": "gdc",         "no_gpe": True,  "use_var": False},
        {"name": "gdc_gpe",     "no_gpe": False, "use_var": False},
        {"name": "gdc_var",     "no_gpe": True,  "use_var": True},
        {"name": "gdc_gpe_var", "no_gpe": False, "use_var": True},
    ]

    total = len(DATASETS) * len(configs) * len(SEEDS)
    done = 0

    for ds in DATASETS:
        config_path = f"{CONFIG_DIR}/{ds}.yaml"
        for cfg in configs:
            for seed in SEEDS:
                out_json = f"{args.out_dir}/{ds}_{cfg['name']}_s{seed}.json"
                if os.path.exists(out_json):
                    print(f"  Skipping (exists): {out_json}")
                    done += 1
                    continue
                try:
                    run_one(config_path, seed, cfg["no_gpe"], cfg["use_var"],
                            out_json, args.device, args.cache_dir)
                    done += 1
                    print(f"  [{done}/{total}] done")
                except subprocess.CalledProcessError as e:
                    print(f"  FAILED: {e}")

    # Aggregate and print LaTeX table
    print("\nAggregating results...")
    print("\n" + "=" * 80)
    print("TABLE 5: GDC ABLATION AT rho=30 (5 seeds)")
    print("=" * 80)

    labels = {"gdc": "GDC", "gdc_gpe": "+GPE", "gdc_var": "+Var", "gdc_gpe_var": "+GPE+Var"}
    print(f"{'Dataset':<12}", end="")
    for cfg in configs:
        print(f" {labels[cfg['name']]:<18}", end="")
    print()
    print("-" * 84)

    latex_rows = []
    for ds in DATASETS:
        row = [ds.replace("_", " ").title()]
        best_mean = float("inf")
        means = []
        for cfg in configs:
            pehes = []
            for seed in SEEDS:
                fpath = f"{args.out_dir}/{ds}_{cfg['name']}_s{seed}.json"
                if os.path.exists(fpath):
                    with open(fpath) as fh:
                        data = json.load(fh)
                        pehes.append(data.get("test_pehe", data.get("best_pehe", float("nan"))))
            if pehes:
                m, s = np.mean(pehes), np.std(pehes)
                means.append(m)
                if m < best_mean:
                    best_mean = m
            else:
                means.append(float("nan"))

        for i, cfg in enumerate(configs):
            pehes = []
            for seed in SEEDS:
                fpath = f"{args.out_dir}/{ds}_{cfg['name']}_s{seed}.json"
                if os.path.exists(fpath):
                    with open(fpath) as fh:
                        data = json.load(fh)
                        pehes.append(data.get("test_pehe", data.get("best_pehe", float("nan"))))
            if pehes:
                m, s = np.mean(pehes), np.std(pehes)
                bold = "**" if m == best_mean else ""
                print(f" {bold}{m:.2f} +/- {s:.2f}{bold}", end="   ")
                # LaTeX
                if m == best_mean:
                    row.append(f"\\textbf{{{m:.2f}}}$^{{\\pm{s:.2f}}}$")
                else:
                    row.append(f"${m:.2f}^{{\\pm{s:.2f}}}$")
            else:
                print(" N/A              ", end="   ")
                row.append("N/A")
        print()
        latex_rows.append(row)

    print("\n\nLaTeX rows:")
    for row in latex_rows:
        print(" & ".join(row) + " \\\\")

    # Save all raw results
    summary = {}
    for f in Path(args.out_dir).glob("*.json"):
        if f.name == "summary.json":
            continue
        with open(f) as fh:
            summary[f.stem] = json.load(fh)
    with open(f"{args.out_dir}/summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)


if __name__ == "__main__":
    main()

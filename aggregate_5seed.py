#!/usr/bin/env python
"""
Aggregate all graph experiment results across 5 seeds (0-4).

Seeds 0-2 come from:
  - runs/gpe_20260410_002659/{ds}/   (bnn, bnn_gpe, x, x_gpe, tarnet, tarnet_gpe, pact, pact_nogpe)
  - runs/sota_20260410_102248/{ds}/  (gdc, gdc_gpe, netdeconf, netdeconf_gpe, gial, gial_gpe, gnum, gnum_gpe)
  - runs/wsdm_all/{ds}/             (blogcatalog/flickr, all models including _var variants)

Seeds 3-4 come from:
  - runs/5seed/{ds}/                (tarnet, tarnet_gpe, gdc, gdc_gpe on cora_full/dblp only)
  - runs/5seed_full/{ds}/           (everything else)

Output:
  - 5seed_all_results.json          (machine-readable: all metrics per model/dataset/rho/seed)
"""

import json
import os
import glob
import sys
import numpy as np
from collections import defaultdict
from pathlib import Path

BASE = os.path.dirname(os.path.abspath(__file__))

# ---- Source directories ----
SOURCE_DIRS = [
    # Main {rho=5, 10, 30} sweep (3 seeds each)
    os.path.join(BASE, "runs/gpe_20260410_002659"),
    os.path.join(BASE, "runs/sota_20260410_102248"),
    os.path.join(BASE, "runs/wsdm_all"),
    # 5-seed extensions (seeds 3, 4 on top of the 3-seed base)
    os.path.join(BASE, "runs/5seed"),
    os.path.join(BASE, "runs/5seed_full"),
    # Supplementary rho={15, 20} extension (5 seeds each). meta/ and sota/
    # sit one level deeper in that tree because the extension was launched
    # as two passes (meta-learners first, SOTA baselines second).
    os.path.join(BASE, "runs/rho_ext_20260417_152312/meta"),
    os.path.join(BASE, "runs/sota_20260417_171739"),
]

# The 14 model tags we care about for the main paper table
MAIN_TAGS = [
    "bnn", "bnn_gpe",
    "x", "x_gpe",
    "tarnet", "tarnet_gpe",
    "gdc", "gdc_gpe",
    "netdeconf", "netdeconf_gpe",
    "gial", "gial_gpe",
    "gnum", "gnum_gpe",
    "pact_nogpe", "pact",
]

# DGP datasets have rho; WSDM datasets do not
DGP_DATASETS = ["cora_full", "dblp", "pubmed"]
WSDM_DATASETS = ["blogcatalog", "flickr"]
ALL_DATASETS = DGP_DATASETS + WSDM_DATASETS
# -----------------------------------------------------------------------------
# Aggregation policy (disclosed for reviewers):
#   * RHOS covers the five noise scales reported in Table 1.
#     rho=15 and rho=20 were run as a supplementary extension; their raw
#     JSONs live under runs/rho_ext_* and are merged into
#     runs/rho_ext_aggregated.json by a separate pass. Paper tables 1-2
#     combine both sweeps.
#   * SOURCE_DIRS is deterministic: when the same (dataset, tag, rho, seed)
#     key appears in more than one source, the first-encountered file is
#     retained and a [CONFLICT] line is emitted to stderr if the second
#     file differs on test PEHE. Duplicates remain auditable.
#   * MAIN_TAGS restricts aggregation to the learner labels reported in
#     the paper. Variants outside this set (exploratory runs, earlier
#     iterations) are counted in skipped_files and do not enter the
#     merged JSON.
# -----------------------------------------------------------------------------
RHOS = [5, 10, 15, 20, 30]
METRICS = ["pehe", "ate", "qini", "auuc"]


def parse_filename(fname):
    """Extract (tag, rho, seed) from filename like 'bnn__rho5__seed3.json' or 'bnn__seed0.json'."""
    name = fname.replace(".json", "")
    parts = name.split("__")

    seed = None
    rho = None
    tag_parts = []

    for p in parts:
        if p.startswith("seed"):
            seed = int(p.replace("seed", ""))
        elif p.startswith("rho"):
            rho = int(p.replace("rho", ""))
        else:
            tag_parts.append(p)

    tag = "__".join(tag_parts)
    return tag, rho, seed


def load_all_results():
    """Load all JSON result files, returning a dict keyed by (dataset, tag, rho, seed)."""
    results = {}
    loaded_files = 0
    skipped_files = 0

    for src_dir in SOURCE_DIRS:
        if not os.path.exists(src_dir):
            print(f"  [WARN] {src_dir} does not exist, skipping", file=sys.stderr)
            continue

        for ds in ALL_DATASETS:
            ds_path = os.path.join(src_dir, ds)
            if not os.path.isdir(ds_path):
                continue

            for jf in sorted(glob.glob(os.path.join(ds_path, "*.json"))):
                fname = os.path.basename(jf)
                tag, rho, seed = parse_filename(fname)

                # Only keep main paper tags
                if tag not in MAIN_TAGS:
                    skipped_files += 1
                    continue

                key = (ds, tag, rho, seed)

                try:
                    with open(jf) as f:
                        data = json.load(f)
                    if key in results:
                        # Explicit conflict reporting (previously silent).
                        existing = results[key]
                        existing_pehe = (existing.get("best", {}).get("test") or {}).get("pehe")
                        new_pehe = (data.get("best", {}).get("test") or {}).get("pehe")
                        if existing_pehe != new_pehe:
                            print(
                                f"  [CONFLICT] {key}: keeping first source (pehe={existing_pehe}), "
                                f"ignored duplicate (pehe={new_pehe}) from {jf}",
                                file=sys.stderr,
                            )
                        continue
                    results[key] = data
                    loaded_files += 1
                except Exception as e:
                    print(f"  [ERR] {jf}: {e}", file=sys.stderr)

    print(f"Loaded {loaded_files} result files, skipped {skipped_files} non-main tags", file=sys.stderr)
    return results


def compute_stats(results):
    """Compute mean/std for each (dataset, tag, rho) across seeds."""
    # Group by (dataset, tag, rho)
    groups = defaultdict(list)
    for (ds, tag, rho, seed), data in results.items():
        groups[(ds, tag, rho)].append((seed, data))

    stats = {}
    for (ds, tag, rho), entries in groups.items():
        seeds_found = sorted([s for s, _ in entries])
        n_seeds = len(seeds_found)

        metric_vals = defaultdict(list)
        for seed, data in entries:
            test = data.get("test", {})
            for m in METRICS:
                if m in test:
                    metric_vals[m].append(test[m])

        stat_entry = {
            "dataset": ds,
            "tag": tag,
            "rho": rho,
            "n_seeds": n_seeds,
            "seeds": seeds_found,
            "metrics": {},
        }

        for m in METRICS:
            vals = metric_vals[m]
            if vals:
                stat_entry["metrics"][m] = {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                    "values": [float(v) for v in vals],
                }

        stats[(ds, tag, rho)] = stat_entry

    return stats


def compute_3seed_stats(results):
    """Compute stats using only seeds 0,1,2 for comparison."""
    filtered = {k: v for k, v in results.items() if k[3] in [0, 1, 2]}
    return compute_stats(filtered)


def build_comparison(stats_5, stats_3):
    """Build comparison between 3-seed and 5-seed stats."""
    comparisons = []

    for key in sorted(stats_5.keys()):
        ds, tag, rho = key
        s5 = stats_5[key]
        s3 = stats_3.get(key)

        entry = {
            "dataset": ds,
            "model": tag,
            "rho": rho,
            "n_seeds_5": s5["n_seeds"],
            "seeds_5": s5["seeds"],
        }

        if s3:
            entry["n_seeds_3"] = s3["n_seeds"]

        entry["metrics"] = {}
        for m in METRICS:
            m_data = {}
            if m in s5["metrics"]:
                m5 = s5["metrics"][m]
                m_data["mean_5seed"] = m5["mean"]
                m_data["std_5seed"] = m5["std"]

            if s3 and m in s3["metrics"]:
                m3 = s3["metrics"][m]
                m_data["mean_3seed"] = m3["mean"]
                m_data["std_3seed"] = m3["std"]

                # Flag significant differences (>1 std separation)
                if m5["std"] > 0 and m3["std"] > 0:
                    diff = abs(m5["mean"] - m3["mean"])
                    max_std = max(m5["std"], m3["std"])
                    m_data["diff_over_std"] = diff / max_std
                    m_data["significant"] = diff > max_std

            entry["metrics"][m] = m_data

        comparisons.append(entry)

    return comparisons


def format_val(mean, std, is_pehe_or_ate=False):
    """Format a mean +/- std value for display."""
    if is_pehe_or_ate:
        return f"{mean:.3f}+/-{std:.3f}"
    else:
        return f"{mean:.4f}+/-{std:.4f}"


def missing_report(results):
    """Report what's missing for the full 5-seed set."""
    missing = []

    for ds in DGP_DATASETS:
        for tag in MAIN_TAGS:
            for rho in RHOS:
                for seed in range(5):
                    key = (ds, tag, rho, seed)
                    if key not in results:
                        missing.append(f"  {ds}/{tag}__rho{rho}__seed{seed}")

    for ds in WSDM_DATASETS:
        for tag in MAIN_TAGS:
            for seed in range(5):
                key = (ds, tag, None, seed)
                if key not in results:
                    missing.append(f"  {ds}/{tag}__seed{seed}")

    return missing


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "runs/5seed_full"

    print("Loading all results...", file=sys.stderr)
    results = load_all_results()

    # Check what's missing
    missing = missing_report(results)
    if missing:
        print(f"\n[WARN] Missing {len(missing)} result files:", file=sys.stderr)
        for m in missing[:20]:
            print(m, file=sys.stderr)
        if len(missing) > 20:
            print(f"  ... and {len(missing)-20} more", file=sys.stderr)

    # Compute stats
    print("\nComputing 5-seed stats...", file=sys.stderr)
    stats_5 = compute_stats(results)

    print("Computing 3-seed stats...", file=sys.stderr)
    stats_3 = compute_3seed_stats(results)

    print("Building comparison...", file=sys.stderr)
    comparisons = build_comparison(stats_5, stats_3)

    # Save machine-readable results
    json_out = os.path.join(out_dir, "5seed_all_results.json")
    with open(json_out, "w") as f:
        json.dump({
            "comparisons": comparisons,
            "stats_5seed": {f"{k[0]}_{k[1]}_rho{k[2]}": v for k, v in stats_5.items()},
            "stats_3seed": {f"{k[0]}_{k[1]}_rho{k[2]}": v for k, v in stats_3.items()},
            "n_missing": len(missing),
            "missing": missing,
        }, f, indent=2)
    print(f"\nSaved JSON: {json_out}", file=sys.stderr)

    # Print summary to stdout (will be captured as the summary)
    print("# 5-Seed Full Results Summary")
    print()
    print(f"Total result files loaded: {len(results)}")
    print(f"Missing files: {len(missing)}")
    print()

    # DGP datasets table
    for ds in DGP_DATASETS:
        print(f"## {ds}")
        print()
        for rho in RHOS:
            print(f"### rho={rho}")
            print()
            header = f"{'Model':<16} | {'PEHE (3s)':<16} | {'PEHE (5s)':<16} | {'ATE (3s)':<14} | {'ATE (5s)':<14} | {'Qini (3s)':<16} | {'Qini (5s)':<16} | n"
            print(header)
            print("-" * len(header))

            for tag in MAIN_TAGS:
                key = (ds, tag, rho)
                s5 = stats_5.get(key)
                s3 = stats_3.get(key)

                if not s5:
                    continue

                n5 = s5["n_seeds"]
                n3 = s3["n_seeds"] if s3 else 0

                pehe3 = format_val(s3["metrics"]["pehe"]["mean"], s3["metrics"]["pehe"]["std"], True) if s3 and "pehe" in s3["metrics"] else "N/A"
                pehe5 = format_val(s5["metrics"]["pehe"]["mean"], s5["metrics"]["pehe"]["std"], True) if "pehe" in s5["metrics"] else "N/A"
                ate3 = format_val(s3["metrics"]["ate"]["mean"], s3["metrics"]["ate"]["std"], True) if s3 and "ate" in s3["metrics"] else "N/A"
                ate5 = format_val(s5["metrics"]["ate"]["mean"], s5["metrics"]["ate"]["std"], True) if "ate" in s5["metrics"] else "N/A"
                qini3 = format_val(s3["metrics"]["qini"]["mean"], s3["metrics"]["qini"]["std"]) if s3 and "qini" in s3["metrics"] else "N/A"
                qini5 = format_val(s5["metrics"]["qini"]["mean"], s5["metrics"]["qini"]["std"]) if "qini" in s5["metrics"] else "N/A"

                sig = ""
                if s3 and s5 and "pehe" in s5["metrics"] and "pehe" in s3["metrics"]:
                    diff = abs(s5["metrics"]["pehe"]["mean"] - s3["metrics"]["pehe"]["mean"])
                    maxstd = max(s5["metrics"]["pehe"]["std"], s3["metrics"]["pehe"]["std"]) if s5["metrics"]["pehe"]["std"] > 0 and s3["metrics"]["pehe"]["std"] > 0 else 999
                    if maxstd < 999 and diff > maxstd:
                        sig = " *"

                print(f"{tag:<16} | {pehe3:<16} | {pehe5:<16} | {ate3:<14} | {ate5:<14} | {qini3:<16} | {qini5:<16} | {n5}{sig}")
            print()

    # WSDM datasets table
    for ds in WSDM_DATASETS:
        print(f"## {ds}")
        print()
        header = f"{'Model':<16} | {'PEHE (3s)':<16} | {'PEHE (5s)':<16} | {'ATE (3s)':<14} | {'ATE (5s)':<14} | {'Qini (3s)':<16} | {'Qini (5s)':<16} | n"
        print(header)
        print("-" * len(header))

        for tag in MAIN_TAGS:
            key = (ds, tag, None)
            s5 = stats_5.get(key)
            s3 = stats_3.get(key)

            if not s5:
                continue

            n5 = s5["n_seeds"]
            n3 = s3["n_seeds"] if s3 else 0

            pehe3 = format_val(s3["metrics"]["pehe"]["mean"], s3["metrics"]["pehe"]["std"], True) if s3 and "pehe" in s3["metrics"] else "N/A"
            pehe5 = format_val(s5["metrics"]["pehe"]["mean"], s5["metrics"]["pehe"]["std"], True) if "pehe" in s5["metrics"] else "N/A"
            ate3 = format_val(s3["metrics"]["ate"]["mean"], s3["metrics"]["ate"]["std"], True) if s3 and "ate" in s3["metrics"] else "N/A"
            ate5 = format_val(s5["metrics"]["ate"]["mean"], s5["metrics"]["ate"]["std"], True) if "ate" in s5["metrics"] else "N/A"
            qini3 = format_val(s3["metrics"]["qini"]["mean"], s3["metrics"]["qini"]["std"]) if s3 and "qini" in s3["metrics"] else "N/A"
            qini5 = format_val(s5["metrics"]["qini"]["mean"], s5["metrics"]["qini"]["std"]) if "qini" in s5["metrics"] else "N/A"

            print(f"{tag:<16} | {pehe3:<16} | {pehe5:<16} | {ate3:<14} | {ate5:<14} | {qini3:<16} | {qini5:<16} | {n5}")
        print()


if __name__ == "__main__":
    main()

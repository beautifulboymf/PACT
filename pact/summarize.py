"""Aggregate per-run JSON files into a tidy summary table.

Usage::

    python -m pact.summarize runs/20260409_135000

Prints a markdown table grouped by (dataset, learner, rho), averaging across
seeds. Also writes ``summary.json`` and ``summary.csv`` next to the runs.
"""

from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from statistics import mean, pstdev


def main(root: str) -> None:
    rows: list[dict] = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if not fn.endswith(".json") or fn == "summary.json":
                continue
            path = os.path.join(dirpath, fn)
            try:
                data = json.load(open(path))
            except Exception as e:
                print(f"[skip] {path}: {e}")
                continue
            row = {
                "dataset": os.path.basename(dirpath),
                "tag": data.get("tag", ""),
                "learner": data.get("learner") or "?",
                "no_gpe": bool(data.get("no_gpe")),
                "rho": data.get("rho"),
                "seed": data.get("seed"),
                "epoch": data.get("epoch"),
                **{f"val_{k}": v for k, v in (data.get("val") or {}).items()},
                **{f"test_{k}": v for k, v in (data.get("test") or {}).items()},
            }
            rows.append(row)

    if not rows:
        print(f"no run JSONs found under {root}")
        sys.exit(1)

    # Aggregate by (dataset, variant, rho)
    def variant(r: dict) -> str:
        # Derive variant name from the tag if available (more reliable
        # than reconstructing from learner+no_gpe flags).
        tag = r.get("tag", "")
        # tag format: "dataset/variant__rhoX__seedY" or "dataset/variant__seedY"
        if "/" in tag:
            v = tag.split("/")[-1].split("__")[0]
            return v

        # Fallback: reconstruct from flags.
        l = r["learner"]
        no_gpe = r.get("no_gpe", False)
        if l == "pact":
            return "PACT" if not no_gpe else "PACT_noGPE"
        if l == "x":
            return "X" if no_gpe else "X+GPE"
        if l == "s":
            return "S" if no_gpe else "S+GPE"
        return l

    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        groups[(r["dataset"], variant(r), r["rho"])].append(r)

    metrics_to_show = ["test_pehe", "test_ate", "test_qini", "test_auuc", "test_lift@30"]

    print("\n## Aggregated results (mean ± std over seeds)\n")
    print("| dataset | variant | rho | n_seeds | " + " | ".join(metrics_to_show) + " |")
    print("|---|---|---|---|" + "|".join("---" for _ in metrics_to_show) + "|")
    summary_rows = []
    for (ds, v, rho), grp in sorted(groups.items()):
        cells = []
        agg = {"dataset": ds, "variant": v, "rho": rho, "n_seeds": len(grp)}
        for m in metrics_to_show:
            vals = [g[m] for g in grp if m in g and g[m] is not None]
            if not vals:
                cells.append("—")
                continue
            mu = mean(vals)
            sd = pstdev(vals) if len(vals) > 1 else 0.0
            cells.append(f"{mu:.3f}±{sd:.3f}")
            agg[m + "_mean"] = mu
            agg[m + "_std"] = sd
        print(f"| {ds} | {v} | {rho} | {len(grp)} | " + " | ".join(cells) + " |")
        summary_rows.append(agg)

    out_json = os.path.join(root, "summary.json")
    with open(out_json, "w") as f:
        json.dump({"per_run": rows, "aggregated": summary_rows}, f, indent=2)
    print(f"\nwrote {out_json}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python -m pact.summarize <runs_root>")
        sys.exit(2)
    main(sys.argv[1])

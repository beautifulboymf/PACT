"""Generate complete experiment tables from all merged results."""
import json
import sys
import os

def fmt(r, key):
    v = r.get(key + "_mean")
    s = r.get(key + "_std")
    if v is None: return "—"
    if s is not None: return f"{v:.3f}±{s:.3f}"
    return f"{v:.3f}"

def main(merged_path, out_path):
    data = json.load(open(merged_path))["aggregated"]
    lookup = {}
    for r in data:
        lookup[(r["dataset"], r["variant"], r.get("rho"))] = r

    lines = []
    def p(s=""): lines.append(s)

    # === GRAPH: CoraFull/DBLP/PubMed ===
    graph_ds = ["cora_full", "dblp", "pubmed"]
    rhos = [5.0, 10.0, 15.0, 20.0, 30.0]
    groups = [
        ("BNN", ["bnn", "bnn_gpe", "bnn_var", "bnn_gpe_var"]),
        ("TARNet", ["tarnet", "tarnet_gpe", "tarnet_var", "tarnet_gpe_var"]),
        ("X-learner", ["x", "x_gpe", "x_var", "x_gpe_var"]),
        ("PACT", ["pact_nogpe", "pact"]),
        ("NetDeconf", ["netdeconf", "netdeconf_gpe", "netdeconf_var", "netdeconf_gpe_var"]),
        ("GIAL", ["gial", "gial_gpe", "gial_var", "gial_gpe_var"]),
        ("GNUM", ["gnum", "gnum_gpe", "gnum_var", "gnum_gpe_var"]),
        ("GDC", ["gdc", "gdc_gpe", "gdc_var", "gdc_gpe_var"]),
    ]

    for metric, metric_key, direction in [("PEHE", "test_pehe", "↓"), ("Qini", "test_qini", "↑")]:
        p(f"### Graph Datasets — {metric} {direction}")
        p()
        for ds in graph_ds:
            p(f"**{ds}**")
            p()
            p(f"| Base Model | Variant | ρ=5 | ρ=10 | ρ=30 |")
            p("|---|---|---|---|---|")
            for base, variants in groups:
                for v in variants:
                    # Label
                    if v.endswith("_gpe_var"): label = "+GPE+Var"
                    elif v.endswith("_gpe"): label = "+GPE"
                    elif v.endswith("_var") and "nogpe" not in v: label = "+Var"
                    elif v == "pact_nogpe": label = "noGPE (Var only)"
                    elif v == "pact": label = "full (GPE+Var)"
                    else: label = "vanilla"
                    cells = []
                    for rho in rhos:
                        r = lookup.get((ds, v, rho))
                        cells.append(fmt(r, metric_key) if r else "—")
                    p(f"| {base} | {label} | {' | '.join(cells)} |")
            p()

    # === WSDM: BlogCatalog/Flickr ===
    p("### BlogCatalog / Flickr — PEHE ↓ (κ=1, 3 seeds)")
    p()
    wsdm_groups = [
        ("NetDeconf", ["netdeconf", "netdeconf_gpe", "netdeconf_var", "netdeconf_gpe_var"]),
        ("GIAL", ["gial", "gial_gpe", "gial_var", "gial_gpe_var"]),
        ("GNUM", ["gnum", "gnum_gpe", "gnum_var", "gnum_gpe_var"]),
        ("GDC", ["gdc", "gdc_gpe", "gdc_var", "gdc_gpe_var"]),
        ("BNN", ["bnn", "bnn_gpe", "bnn_var", "bnn_gpe_var"]),
        ("TARNet", ["tarnet", "tarnet_gpe", "tarnet_var", "tarnet_gpe_var"]),
        ("X-learner", ["x", "x_gpe", "x_var", "x_gpe_var"]),
        ("PACT", ["pact_nogpe", "pact"]),
    ]
    p("| Base | Variant | BlogCatalog | Flickr |")
    p("|---|---|---|---|")
    for base, variants in wsdm_groups:
        for v in variants:
            if v.endswith("_gpe_var"): label = "+GPE+Var"
            elif v.endswith("_gpe"): label = "+GPE"
            elif v.endswith("_var") and "nogpe" not in v: label = "+Var"
            elif v == "pact_nogpe": label = "noGPE"
            elif v == "pact": label = "full"
            else: label = "vanilla"
            bc = lookup.get(("blogcatalog", v, None))
            fl = lookup.get(("flickr", v, None))
            p(f"| {base} | {label} | {fmt(bc, 'test_pehe') if bc else '—'} | {fmt(fl, 'test_pehe') if fl else '—'} |")
    p()

    # === TABULAR ===
    p("### Tabular — Qini ↑ (5 seeds)")
    p()
    tab_ds = ["criteo", "hillstrom", "hillstrom_spend", "retailhero", "lenta", "x5"]
    tab_vars = ["s", "s_var", "t", "t_var", "x", "x_var"]
    labels = {"s": "S", "s_var": "S+Var", "t": "T", "t_var": "T+Var", "x": "X", "x_var": "X+Var"}
    p("| Variant | Criteo | Hillstrom | Hillstrom(spend) | RetailHero | Lenta | X5 |")
    p("|---|---|---|---|---|---|---|")
    for v in tab_vars:
        cells = []
        for ds in tab_ds:
            r = lookup.get((ds, v, None))
            cells.append(fmt(r, "test_qini") if r else "—")
        p(f"| {labels[v]} | {' | '.join(cells)} |")
    p()

    # === SUMMARY: GPE improvement count ===
    p("### GPE Improvement Summary (PEHE, across all graph dataset×ρ cells)")
    p()
    p("| Baseline → +GPE | CoraFull/DBLP/PubMed (9 cells) | BlogCatalog/Flickr (2 cells) |")
    p("|---|---|---|")
    for base_v, gpe_v, name in [
        ("bnn", "bnn_gpe", "BNN"), ("tarnet", "tarnet_gpe", "TARNet"),
        ("x", "x_gpe", "X-learner"), ("pact_nogpe", "pact", "PACT"),
        ("netdeconf", "netdeconf_gpe", "NetDeconf"), ("gial", "gial_gpe", "GIAL"),
        ("gnum", "gnum_gpe", "GNUM"), ("gdc", "gdc_gpe", "GDC"),
    ]:
        wins_our = 0
        for ds in graph_ds:
            for rho in rhos:
                b = lookup.get((ds, base_v, rho))
                g = lookup.get((ds, gpe_v, rho))
                if b and g and g.get("test_pehe_mean") and b.get("test_pehe_mean"):
                    if g["test_pehe_mean"] < b["test_pehe_mean"]:
                        wins_our += 1
        wins_wsdm = 0
        for ds in ["blogcatalog", "flickr"]:
            b = lookup.get((ds, base_v, None))
            g = lookup.get((ds, gpe_v, None))
            if b and g and g.get("test_pehe_mean") and b.get("test_pehe_mean"):
                if g["test_pehe_mean"] < b["test_pehe_mean"]:
                    wins_wsdm += 1
        p(f"| {name} | {wins_our}/9 | {wins_wsdm}/2 |")
    p()

    # === SUMMARY: Variance improvement count ===
    p("### Variance Improvement Summary (PEHE, across all graph dataset×ρ cells)")
    p()
    p("| Baseline → +Var | CoraFull/DBLP/PubMed (9 cells) | BlogCatalog/Flickr (2 cells) |")
    p("|---|---|---|")
    for base_v, var_v, name in [
        ("bnn", "bnn_var", "BNN"), ("tarnet", "tarnet_var", "TARNet"),
        ("x", "x_var", "X-learner"),
        ("netdeconf", "netdeconf_var", "NetDeconf"), ("gial", "gial_var", "GIAL"),
        ("gnum", "gnum_var", "GNUM"), ("gdc", "gdc_var", "GDC"),
    ]:
        wins_our = 0
        for ds in graph_ds:
            for rho in rhos:
                b = lookup.get((ds, base_v, rho))
                g = lookup.get((ds, var_v, rho))
                if b and g and g.get("test_pehe_mean") and b.get("test_pehe_mean"):
                    if g["test_pehe_mean"] < b["test_pehe_mean"]:
                        wins_our += 1
        wins_wsdm = 0
        for ds in ["blogcatalog", "flickr"]:
            b = lookup.get((ds, base_v, None))
            g = lookup.get((ds, var_v, None))
            if b and g and g.get("test_pehe_mean") and b.get("test_pehe_mean"):
                if g["test_pehe_mean"] < b["test_pehe_mean"]:
                    wins_wsdm += 1
        p(f"| {name} | {wins_our}/9 | {wins_wsdm}/2 |")

    result = "\n".join(lines)
    with open(out_path, "w") as f:
        f.write(result)
    print(result)
    print(f"\nSaved to {out_path}")

if __name__ == "__main__":
    merged = sys.argv[1] if len(sys.argv) > 1 else "runs/all_merged.json"
    out = sys.argv[2] if len(sys.argv) > 2 else "results/complete_tables.md"
    main(merged, out)

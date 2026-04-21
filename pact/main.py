"""CLI entry point for PACT graph-uplift training.

Usage::

    python -m pact.main --config pact/configs/cora_full.yaml
    python -m pact.main --config pact/configs/cora_full.yaml --learner x
    python -m pact.main --smoke   # tiny synthetic graph, ~5s, no data needed

The smoke flag is what you should run first when picking up this repo on a new
machine — it exercises every code path without touching CoraFull/DBLP/PubMed.
"""

from __future__ import annotations

import argparse
import os

import yaml

import torch as _torch

from .data import build_graph_uplift_sample, tiny_synthetic_graph, GraphUpliftSample
from .losses import LossWeights
from .model import PACTConfig, PACTNoGraph, TLearnerNoGraph, PACT, BNNGraph, TARNetGraph
from .baselines import NetDeconfGraph, GIALGraph, GNUMGraph, GDCGraph
from .train import TrainConfig, train_graph, train_tabular


def _load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _load_cached_sample(cache_path: str, device: str) -> GraphUpliftSample:
    """Load a pre-generated .pt file into a GraphUpliftSample."""
    d = _torch.load(cache_path, map_location=device, weights_only=True)
    A_norm = _torch.sparse_coo_tensor(d["A_norm_indices"], d["A_norm_values"], d["A_norm_shape"]).coalesce().to(device)
    return GraphUpliftSample(
        X=d["X"].to(device), pos=d["pos"].to(device),
        edge_index=d["edge_index"].to(device), A_norm=A_norm,
        T=d["T"].to(device), Y=d["Y"].to(device),
        Y0=d["Y0"].to(device), Y1=d["Y1"].to(device),
        true_tau=d["true_tau"].to(device),
    )


def run_from_config(cfg_path: str, overrides: dict | None = None, device: str = "cpu", cache_dir: str | None = None) -> dict:
    cfg = _load_yaml(cfg_path)
    if overrides:
        for k, v in overrides.items():
            section, key = k.split(".", 1) if "." in k else (None, k)
            if section is None:
                cfg[key] = v
            else:
                cfg.setdefault(section, {})[key] = v

    ds_cfg = cfg["dataset"]
    dgp_cfg = cfg["dgp"]
    mdl_cfg = cfg["model"]
    tr_cfg = cfg["train"]

    # Try loading from pre-generated cache first.
    sample = None
    if cache_dir:
        ds_name = ds_cfg["name"].lower().replace(" ", "_")
        # Map dataset names to config-file-style names
        name_map = {"corafull": "cora_full"}
        ds_key = name_map.get(ds_name, ds_name)
        rho = int(dgp_cfg.get("rho", 10))
        seed = tr_cfg.get("seed", 0)
        # Only include community-detection suffix when non-default, so that
        # existing cached .pt files from before this flag existed remain valid.
        comm = dgp_cfg.get("community_detection", "louvain").lower()
        comm_suffix = "" if comm == "louvain" else f"_comm{comm}"
        cache_path = os.path.join(
            cache_dir, f"{ds_key}_rho{rho}{comm_suffix}_seed{seed}.pt"
        )
        if os.path.exists(cache_path):
            print(f"[cache] loading {cache_path}")
            sample = _load_cached_sample(cache_path, device)

    if sample is None:
        ds_name = ds_cfg["name"]
        # WSDM datasets (BlogCatalog, Flickr) use .mat files, no DGP
        if ds_name in ("BlogCatalog", "Flickr"):
            from .data import load_wsdm_dataset
            sample = load_wsdm_dataset(
                data_dir=ds_cfg["root"],
                name=ds_name,
                extra_str=str(ds_cfg.get("extra_str", "1")),
                exp_id=tr_cfg.get("seed", 0) % 10,
                pos_dim=mdl_cfg.get("pos_dim", 128),
                device=device,
            )
        else:
            pos_path = ds_cfg.get("pos_emb_path")
            if pos_path and not os.path.isabs(pos_path):
                proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
                pos_path = os.path.join(proj_root, pos_path)
            sample = build_graph_uplift_sample(
                name=ds_name,
                root=ds_cfg["root"],
                dgp_cfg=dgp_cfg,
                pos_path=pos_path,
                seed=tr_cfg.get("seed", 0),
                device=device,
            )
    model_cfg = PACTConfig(in_dim=sample.X.size(1), **mdl_cfg)
    train_cfg = TrainConfig(
        loss_weights=LossWeights(**tr_cfg.get("loss_weights", {})),
        **{k: v for k, v in tr_cfg.items() if k != "loss_weights"},
    )

    # Model dispatch
    learner = tr_cfg.get("learner", "pact")
    sota_baselines = {"netdeconf": NetDeconfGraph, "gial": GIALGraph, "gnum": GNUMGraph, "gdc": GDCGraph}
    if learner in sota_baselines:
        model = sota_baselines[learner](model_cfg)
        from .train import train_graph_baseline
        return train_graph_baseline(sample, model, model_cfg, train_cfg, device=device, baseline_type=learner)
    elif learner == "bnn":
        model = BNNGraph(model_cfg)
    elif learner == "tarnet":
        model = TARNetGraph(model_cfg)
    else:
        model = PACT(model_cfg)

    return train_graph(sample, model_cfg, train_cfg, device=device, model=model)


def run_smoke(device: str = "cpu") -> dict:
    sample = tiny_synthetic_graph(n=400, feat_dim=32, device=device, seed=0)
    model_cfg = PACTConfig(
        in_dim=sample.X.size(1),
        pos_dim=16,
        fusion_embed_dim=32,
        fusion_heads=4,
        use_gpe=True,
        backbone="gat",
        gnn_hidden=(32, 32),
        gnn_heads=2,
        mlp_hidden=(32,),
        dropout=0.0,
        use_variance=True,
    )
    train_cfg = TrainConfig(
        epochs=30,
        lr=5e-3,
        weight_decay=1e-5,
        train_frac=0.6,
        val_frac=0.2,
        learner="pact",
        log_every=5,
        select_metric="qini",
        normalize_y=True,
    )
    return train_graph(sample, model_cfg, train_cfg, device=device)


def run_tabular(cfg_path: str, overrides: dict | None = None, device: str = "cpu") -> dict:
    """Run tabular (no-graph) uplift training on Criteo / Hillstrom."""
    from .data import load_criteo_uplift, load_hillstrom, load_lenta

    cfg = _load_yaml(cfg_path)
    if overrides:
        for k, v in overrides.items():
            section, key = k.split(".", 1) if "." in k else (None, k)
            if section is None:
                cfg[key] = v
            else:
                cfg.setdefault(section, {})[key] = v

    ds_cfg = cfg["dataset"]
    mdl_cfg = cfg["model"]
    tr_cfg = cfg["train"]

    name = ds_cfg["name"].lower()
    csv_path = ds_cfg["csv_path"]
    max_rows = ds_cfg.get("max_rows")

    if name == "criteo":
        sample = load_criteo_uplift(csv_path, max_rows=max_rows, device=device)
    elif name == "hillstrom":
        sample = load_hillstrom(
            csv_path,
            treatment_val=ds_cfg.get("treatment_val", "Mens E-Mail"),
            outcome=ds_cfg.get("outcome", "visit"),
            device=device,
        )
    elif name == "retailhero":
        from .data import load_retailhero
        sample = load_retailhero(csv_path, device=device)
    elif name in ("x5", "x5_retailhero"):
        from .data import load_x5
        sample = load_x5(csv_path, device=device)
    elif name == "lenta":
        sample = load_lenta(csv_path, device=device)
    else:
        raise ValueError(f"unknown tabular dataset: {name}")

    learner = tr_cfg.get("learner", "pact")
    use_variance = tr_cfg.get("use_variance", learner == "pact")
    tr_cfg["use_variance"] = use_variance
    binary = tr_cfg.get("binary_outcome", False)

    # Model selection: T-learner vs X-learner (PACTNoGraph)
    model_kwargs = dict(
        in_dim=sample.X.size(1),
        encoder_hidden=tuple(mdl_cfg.get("encoder_hidden", [256, 128])),
        mlp_hidden=tuple(mdl_cfg.get("mlp_hidden", [128, 128])),
        dropout=mdl_cfg.get("dropout", 0.0),
    )
    if learner == "t":
        model = TLearnerNoGraph(**model_kwargs, use_variance=False)
    else:
        mdl_cfg["use_variance"] = use_variance and not binary
        model = PACTNoGraph(**model_kwargs, use_variance=use_variance and not binary)
    train_cfg = TrainConfig(
        loss_weights=LossWeights(**tr_cfg.get("loss_weights", {})),
        **{k: v for k, v in tr_cfg.items() if k != "loss_weights"},
    )

    ite_mode = "s" if learner == "s" else "x"
    return train_tabular(sample, model, train_cfg, device=device, ite_mode=ite_mode)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--mode", type=str, default="graph", choices=["graph", "tabular"])
    parser.add_argument("--learner", type=str, default=None, choices=["s", "t", "x", "pact", "bnn", "tarnet", "netdeconf", "gial", "gnum", "gdc"])
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--rho", type=float, default=None, help="heteroscedasticity scale")
    parser.add_argument("--kappa4", type=float, default=None, help="community bias weight (0 disables community structure)")
    parser.add_argument("--intra-w", type=float, default=None, help="intra-community edge weight (1.0 disables boosting)")
    parser.add_argument("--community-detection", type=str, default=None,
                        choices=["louvain", "leiden", "infomap"],
                        help="community-detection algorithm for DGP (default from config; louvain for paper main results)")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--backbone", type=str, default=None, choices=["gat", "gcn", "graphformer"], help="GNN backbone type")
    parser.add_argument("--no-gpe", action="store_true", help="disable graph positional encoding (ablation)")
    parser.add_argument("--use-variance", action="store_true", default=None, help="enable variance weighting (plug-in)")
    parser.add_argument("--no-variance", action="store_true", help="disable variance weighting")
    parser.add_argument("--out-json", type=str, default=None, help="dump best metrics as JSON to this path")
    parser.add_argument("--tag", type=str, default=None, help="optional tag printed with results")
    parser.add_argument("--cache-dir", type=str, default=None, help="load pre-generated DGP .pt files from this dir")
    args = parser.parse_args()

    # Resolve variance flag: --use-variance / --no-variance override config.
    var_override = None
    if args.use_variance:
        var_override = True
    elif args.no_variance:
        var_override = False

    if args.smoke:
        result = run_smoke(device=args.device)
    elif args.mode == "tabular":
        if args.config is None:
            parser.error("--config required for tabular mode")
        overrides: dict = {}
        if args.learner is not None:
            overrides["train.learner"] = args.learner
        if var_override is not None:
            overrides["train.use_variance"] = var_override
        if args.seed is not None:
            overrides["train.seed"] = int(args.seed)
        if args.epochs is not None:
            overrides["train.epochs"] = int(args.epochs)
        result = run_tabular(args.config, overrides=overrides, device=args.device)
    else:
        if args.config is None:
            parser.error("--config required unless --smoke is set")
        overrides = {}
        if args.learner is not None:
            overrides["train.learner"] = args.learner
            overrides["model.use_variance"] = (args.learner == "pact")
        if var_override is not None:
            overrides["train.use_variance"] = var_override
        if args.rho is not None:
            overrides["dgp.rho"] = float(args.rho)
        if args.kappa4 is not None:
            overrides["dgp.kappa4"] = float(args.kappa4)
        if args.intra_w is not None:
            overrides["dgp.intra_w"] = float(args.intra_w)
        if args.community_detection is not None:
            overrides["dgp.community_detection"] = args.community_detection
        if args.seed is not None:
            overrides["train.seed"] = int(args.seed)
        if args.epochs is not None:
            overrides["train.epochs"] = int(args.epochs)
        if args.backbone is not None:
            overrides["model.backbone"] = args.backbone
        if args.no_gpe:
            overrides["model.use_gpe"] = False
        result = run_from_config(args.config, overrides=overrides, device=args.device, cache_dir=args.cache_dir)

    best = result["best"]
    print("\n=== best ===")
    if args.tag:
        print(f"  tag={args.tag}")
    print(f"  epoch={best['epoch']}")
    for k, v in best.get("val", {}).items():
        print(f"  val_{k}={v:.4f}")
    for k, v in best.get("test", {}).items():
        print(f"  test_{k}={v:.4f}")

    if args.out_json:
        import json
        os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
        payload = {
            "tag": args.tag,
            "config": args.config,
            "learner": args.learner,
            "rho": args.rho,
            "seed": args.seed,
            "no_gpe": args.no_gpe,
            "community_detection": args.community_detection,
            "epoch": best["epoch"],
            "val": best.get("val", {}),
            "test": best.get("test", {}),
        }
        with open(args.out_json, "w") as f:
            json.dump(payload, f, indent=2)


if __name__ == "__main__":
    main()

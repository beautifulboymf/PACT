"""Dataset adapters for PACT.

Two regimes:

1. Graph datasets (CoraFull / DBLP / PubMed): we load the raw PyG dataset,
   apply the PACT semi-synthetic DGP (`dgp.semi_synthetic`), and load a
   precomputed node2vec embedding for the GPE module.

2. Real-world uplift (Criteo / Lenta): tabular CSV with columns
   {features..., treatment, outcome}. No graph; we still run the meta-learner
   ablation through `model.PACTNoGraph`. The user must provide a path; we
   don't auto-download.

Also exposes a tiny synthetic generator (`tiny_synthetic_graph`) used by the
smoke test so the harness can run without any large files on disk.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import scipy.sparse as sp
import torch


@dataclass
class GraphUpliftSample:
    X: torch.Tensor          # [N, D] node features (float32)
    pos: torch.Tensor        # [N, P] positional encoding (node2vec) or zeros
    edge_index: torch.LongTensor  # [2, E]
    A_norm: torch.Tensor     # sparse [N, N] symmetric-normalized
    T: torch.Tensor          # [N] float in {0,1}
    Y: torch.Tensor          # [N] observed factual outcome
    Y0: torch.Tensor         # [N] potential outcome under control (oracle)
    Y1: torch.Tensor         # [N] potential outcome under treatment (oracle)
    true_tau: torch.Tensor   # [N] = Y1 - Y0


def _scipy_to_torch_sparse(A: sp.spmatrix) -> torch.Tensor:
    A = A.tocoo().astype("float32")
    indices = torch.from_numpy(np.vstack([A.row, A.col]).astype(np.int64))
    values = torch.from_numpy(A.data)
    return torch.sparse_coo_tensor(indices, values, A.shape).coalesce()


def _scipy_to_edge_index(A: sp.spmatrix) -> torch.LongTensor:
    A = A.tocoo()
    return torch.from_numpy(np.vstack([A.row, A.col]).astype(np.int64))


def load_graph_dataset(name: str, root: str):
    """Load (X, A) for one of CoraFull / DBLP / PubMed via torch_geometric.

    PyG is used only for the dataset wrapper (data downloading + tensor
    layout). All downstream computation uses our own layers.
    """
    from torch_geometric.datasets import CitationFull, CoraFull

    name_l = name.lower()
    if name_l == "corafull":
        ds = CoraFull(root=root)
    elif name_l == "dblp":
        ds = CitationFull(root=root, name="DBLP")
    elif name_l == "pubmed":
        ds = CitationFull(root=root, name="PubMed")
    else:
        raise ValueError(f"unknown graph dataset: {name}")
    data = ds[0]
    X = data.x.numpy().astype("float32")
    src, dst = data.edge_index[0].numpy(), data.edge_index[1].numpy()
    A = sp.coo_matrix(
        (np.ones(len(src), dtype="float32"), (src, dst)),
        shape=(data.num_nodes, data.num_nodes),
    ).tocsr()
    # symmetrize
    A = ((A + A.T) > 0).astype("float32")
    return X, A


def load_pos_embedding(path: Optional[str], n: int, dim: int = 128) -> np.ndarray:
    """Load a precomputed node2vec embedding from .npy. If missing, returns zeros
    (which effectively disables GPE — useful for ablation runs)."""
    if path and os.path.exists(path):
        emb = np.load(path).astype("float32")
        if emb.shape[0] != n:
            raise ValueError(f"pos embedding rows {emb.shape[0]} != n_nodes {n}")
        return emb
    return np.zeros((n, dim), dtype="float32")


def build_graph_uplift_sample(
    name: str,
    root: str,
    dgp_cfg: dict,
    pos_path: Optional[str] = None,
    seed: int = 0,
    device: str = "cpu",
) -> GraphUpliftSample:
    from .dgp import semi_synthetic, normalize_adjacency

    X, A = load_graph_dataset(name, root)
    rng = np.random.default_rng(seed)
    out = semi_synthetic(X, A, dgp_cfg, rng=rng)
    A_boost = out["A"]
    A_norm = normalize_adjacency(A_boost, self_loop=True)

    pos = load_pos_embedding(pos_path, X.shape[0], dim=dgp_cfg.get("pos_dim", 128))

    s = GraphUpliftSample(
        X=torch.tensor(out["Z"], dtype=torch.float32, device=device),
        pos=torch.tensor(pos, dtype=torch.float32, device=device),
        edge_index=_scipy_to_edge_index(A_boost).to(device),
        A_norm=_scipy_to_torch_sparse(A_norm).to(device),
        T=torch.tensor(out["T"], dtype=torch.float32, device=device),
        Y=torch.tensor(out["Y"], dtype=torch.float32, device=device),
        Y0=torch.tensor(out["Y0"], dtype=torch.float32, device=device),
        Y1=torch.tensor(out["Y1"], dtype=torch.float32, device=device),
        true_tau=torch.tensor(out["true_tau"], dtype=torch.float32, device=device),
    )
    return s


# ---------------------------------------------------------------------------
# Real-world uplift adapter (no graph)
# ---------------------------------------------------------------------------


@dataclass
class TabularUpliftSample:
    X: torch.Tensor           # [N, D]
    T: torch.Tensor           # [N]
    Y: torch.Tensor           # [N]


def load_criteo_uplift(csv_path: str, max_rows: Optional[int] = None, device: str = "cpu") -> TabularUpliftSample:
    """Load Criteo Uplift v2.1.

    Supports both CSV files and HuggingFace Arrow directories. For Arrow,
    pass the directory containing ``*.arrow`` files (the loader will glob
    them and concatenate).
    """
    import pandas as pd

    if os.path.isdir(csv_path):
        # HuggingFace Arrow format: find .arrow files recursively
        import glob
        arrow_files = sorted(glob.glob(os.path.join(csv_path, "**/*.arrow"), recursive=True))
        if not arrow_files:
            raise FileNotFoundError(f"no .arrow files under {csv_path}")
        import pyarrow as pa
        tables = []
        for f in arrow_files:
            try:
                tables.append(pa.ipc.open_file(f).read_all())
            except pa.lib.ArrowInvalid:
                tables.append(pa.ipc.open_stream(f).read_all())
        table = pa.concat_tables(tables)
        df = table.to_pandas()
        # Shuffle before truncating — Arrow shards are sorted by treatment,
        # so naive head-slicing gives all T=1.
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        if max_rows:
            df = df.iloc[:max_rows]
    else:
        df = pd.read_csv(csv_path, nrows=max_rows)

    feat_cols = [c for c in df.columns if c.startswith("f")]
    X = df[feat_cols].values.astype("float32")
    T = df["treatment"].values.astype("float32")
    Y = df["conversion"].values.astype("float32") if "conversion" in df.columns else df["visit"].values.astype("float32")
    # standardize features
    X = (X - X.mean(0, keepdims=True)) / (X.std(0, keepdims=True) + 1e-6)
    return TabularUpliftSample(
        X=torch.tensor(X, device=device),
        T=torch.tensor(T, device=device),
        Y=torch.tensor(Y, device=device),
    )


def load_retailhero(
    data_dir: str,
    device: str = "cpu",
) -> TabularUpliftSample:
    """Load RetailHero X5 uplift dataset.

    Expects ``data_dir`` to contain ``uplift_train.csv`` and ``clients.csv``
    (downloaded from HuggingFace ``pytorch-lifestream/retailhero-uplift``).
    """
    import pandas as pd

    train = pd.read_csv(os.path.join(data_dir, "uplift_train.csv"))
    clients = pd.read_csv(os.path.join(data_dir, "clients.csv"))
    df = train.merge(clients, on="client_id", how="left")

    T = df["treatment_flg"].values.astype("float32")
    Y = df["target"].values.astype("float32")

    # Numeric features
    ref_date = pd.Timestamp("2019-03-01")
    df["days_since_issue"] = (ref_date - pd.to_datetime(df["first_issue_date"], errors="coerce")).dt.days.fillna(0).astype("float32")
    df["days_since_redeem"] = (ref_date - pd.to_datetime(df["first_redeem_date"], errors="coerce")).dt.days.fillna(0).astype("float32")

    num_cols = ["age", "days_since_issue", "days_since_redeem"]
    X_num = df[num_cols].fillna(0).values.astype("float32")

    gender_dummies = pd.get_dummies(df["gender"], prefix="g", dtype="float32").values
    X = np.hstack([X_num, gender_dummies])
    X = (X - X.mean(0, keepdims=True)) / (X.std(0, keepdims=True) + 1e-6)

    return TabularUpliftSample(
        X=torch.tensor(X, device=device),
        T=torch.tensor(T, device=device),
        Y=torch.tensor(Y, device=device),
    )


def load_lenta(csv_path: str, device: str = "cpu") -> TabularUpliftSample:
    """Load the Lenta uplift dataset. Handles both .csv and .csv.gz."""
    import pandas as pd
    df = pd.read_csv(csv_path)
    drop = {"group", "response_att", "response_sms", "response_viber", "gender"}
    feat_cols = [c for c in df.columns if c not in drop]
    X = df[feat_cols].select_dtypes(include=[np.number]).values.astype("float32")
    # One-hot encode gender
    gender_dummies = pd.get_dummies(df["gender"], prefix="g", dtype="float32").values
    X = np.hstack([X, gender_dummies])
    X = np.nan_to_num(X, nan=0.0)
    T = (df["group"] == "test").astype("float32").values
    Y = df["response_att"].astype("float32").values
    X = (X - X.mean(0, keepdims=True)) / (X.std(0, keepdims=True) + 1e-6)
    return TabularUpliftSample(
        X=torch.tensor(X, device=device),
        T=torch.tensor(T, device=device),
        Y=torch.tensor(Y, device=device),
    )


def load_x5(data_dir: str, device: str = "cpu") -> TabularUpliftSample:
    """Load X5 RetailHero with aggregated purchase features.

    Aggregates the purchases table per client, merges with client demographics
    and uplift labels. Caches the result as ``x5_processed.csv`` for fast reload.
    """
    import pandas as pd

    cache_path = os.path.join(data_dir, "x5_processed.csv")
    if os.path.exists(cache_path):
        df = pd.read_csv(cache_path)
    else:
        # Load raw parts (handles both .csv and .csv.gz)
        def _find(name):
            for ext in [".csv", ".csv.gz"]:  # prefer uncompressed
                p = os.path.join(data_dir, name + ext)
                if os.path.exists(p):
                    return p
            raise FileNotFoundError(f"{name} not found in {data_dir}")

        train = pd.read_csv(_find("uplift_train"))
        clients = pd.read_csv(_find("clients"))

        # Read purchases in chunks to handle large/partially-corrupt .gz files
        purch_path = _find("purchases")
        agg_parts = []
        for chunk in pd.read_csv(purch_path, chunksize=2_000_000):
            part = chunk.groupby("client_id").agg(
                total_spend=("purchase_sum", "sum"),
                num_products=("product_id", "count"),
                num_transactions=("transaction_id", "nunique"),
                total_pts_recv=("regular_points_received", "sum"),
                total_pts_spent=("regular_points_spent", "sum"),
            )
            agg_parts.append(part)
        trn_agg = pd.concat(agg_parts).groupby("client_id").sum().reset_index()
        trn_agg["avg_spend_per_trn"] = trn_agg["total_spend"] / trn_agg["num_transactions"].clip(lower=1)

        # Merge
        df = train.merge(clients, on="client_id", how="left")
        df = df.merge(trn_agg, on="client_id", how="left")

        # Date features
        ref = pd.Timestamp("2019-03-01")
        df["days_issue"] = (ref - pd.to_datetime(df["first_issue_date"], errors="coerce")).dt.days.fillna(0)
        df["days_redeem"] = (ref - pd.to_datetime(df["first_redeem_date"], errors="coerce")).dt.days.fillna(0)

        df.to_csv(cache_path, index=False)

    T = df["treatment_flg"].values.astype("float32")
    Y = df["target"].values.astype("float32")

    num_cols = ["age", "days_issue", "days_redeem", "total_spend", "num_products",
                "num_transactions", "total_pts_recv", "total_pts_spent", "avg_spend_per_trn"]
    X_num = df[num_cols].fillna(0).values.astype("float32")
    gender_dummies = pd.get_dummies(df["gender"], prefix="g", dtype="float32").values
    X = np.hstack([X_num, gender_dummies])
    X = (X - X.mean(0, keepdims=True)) / (X.std(0, keepdims=True) + 1e-6)

    return TabularUpliftSample(
        X=torch.tensor(X, device=device),
        T=torch.tensor(T, device=device),
        Y=torch.tensor(Y, device=device),
    )


def load_hillstrom(
    csv_path: str,
    treatment_val: str = "Mens E-Mail",
    control_val: str = "No E-Mail",
    outcome: str = "visit",
    device: str = "cpu",
) -> TabularUpliftSample:
    """Load the Hillstrom email marketing uplift dataset.

    Filters to binary treatment (treatment_val vs control_val from the
    ``segment`` column). Default: "Mens E-Mail" vs "No E-Mail".
    """
    import pandas as pd
    df = pd.read_csv(csv_path)
    df = df[df["segment"].isin([treatment_val, control_val])].reset_index(drop=True)
    T = (df["segment"] == treatment_val).astype("float32").values
    Y = df[outcome].astype("float32").values

    # Numeric features
    num_cols = ["recency", "history", "mens", "womens", "newbie"]
    X_num = df[num_cols].values.astype("float32")

    # One-hot encode categoricals
    zip_dummies = pd.get_dummies(df["zip_code"], prefix="zip", dtype="float32").values
    chan_dummies = pd.get_dummies(df["channel"], prefix="chan", dtype="float32").values
    X = np.hstack([X_num, zip_dummies, chan_dummies])

    # Standardize
    X = (X - X.mean(0, keepdims=True)) / (X.std(0, keepdims=True) + 1e-6)
    return TabularUpliftSample(
        X=torch.tensor(X, device=device),
        T=torch.tensor(T, device=device),
        Y=torch.tensor(Y, device=device),
    )


# ---------------------------------------------------------------------------
# BlogCatalog / Flickr (WSDM 2020 standard benchmarks)
# ---------------------------------------------------------------------------


def load_wsdm_dataset(
    data_dir: str,
    name: str = "BlogCatalog",
    extra_str: str = "1",
    exp_id: int = 0,
    pos_dim: int = 128,
    device: str = "cpu",
) -> GraphUpliftSample:
    """Load BlogCatalog or Flickr semi-synthetic data from .mat files.

    These are the standard benchmarks from Guo et al. WSDM 2020, used by
    all subsequent graph ITE papers (GIAL, GNUM, IGL, GDC).

    Args:
        data_dir: directory containing e.g. BlogCatalog1/BlogCatalog0.mat
        name: "BlogCatalog" or "Flickr"
        extra_str: confounding bias level ("0.5", "1", "2")
        exp_id: experiment replicate (0-9)
        pos_dim: dimension for node2vec positional encoding (generated on first call)
    """
    from scipy.io import loadmat

    mat_dir = os.path.join(data_dir, f"{name}{extra_str}")
    mat_path = os.path.join(mat_dir, f"{name}{exp_id}.mat")
    data = loadmat(mat_path)

    X = data["X_100"]  # precomputed 100-dim features
    if sp.issparse(X):
        X = np.asarray(X.todense())
    X = X.astype("float32")

    A = data["Network"]
    if not sp.issparse(A):
        A = sp.csr_matrix(A)
    A = A.astype("float32")

    T = np.squeeze(data["T"]).astype("float32")
    Y1 = np.squeeze(data["Y1"]).astype("float32")
    Y0 = np.squeeze(data["Y0"]).astype("float32")
    Y = T * Y1 + (1 - T) * Y0
    true_tau = Y1 - Y0

    # Normalize adjacency
    from .dgp import normalize_adjacency
    A_norm = normalize_adjacency(A, self_loop=True)

    # Generate or load node2vec positional encoding
    pos_path = os.path.join(mat_dir, f"gpe_{pos_dim}_exp{exp_id}.npy")
    if os.path.exists(pos_path):
        pos = np.load(pos_path).astype("float32")
    else:
        pos = _generate_node2vec(A, pos_dim, pos_path)

    return GraphUpliftSample(
        X=torch.tensor(X, dtype=torch.float32, device=device),
        pos=torch.tensor(pos, dtype=torch.float32, device=device),
        edge_index=_scipy_to_edge_index(A).to(device),
        A_norm=_scipy_to_torch_sparse(A_norm).to(device),
        T=torch.tensor(T, dtype=torch.float32, device=device),
        Y=torch.tensor(Y, dtype=torch.float32, device=device),
        Y0=torch.tensor(Y0, dtype=torch.float32, device=device),
        Y1=torch.tensor(Y1, dtype=torch.float32, device=device),
        true_tau=torch.tensor(true_tau, dtype=torch.float32, device=device),
    )


def _generate_node2vec(A: sp.spmatrix, dim: int, save_path: str) -> np.ndarray:
    """Generate node2vec embeddings using random walks + Word2Vec."""
    import networkx as nx
    import random

    G = nx.from_scipy_sparse_array(A)
    n = G.number_of_nodes()

    # Random walks
    walks = []
    for _ in range(100):
        nodes = list(G.nodes())
        random.shuffle(nodes)
        for node in nodes:
            walk = [str(node)]
            cur = node
            for _ in range(29):
                nbrs = list(G.neighbors(cur))
                if nbrs:
                    cur = random.choice(nbrs)
                    walk.append(str(cur))
                else:
                    break
            walks.append(walk)

    # Word2Vec
    from gensim.models import Word2Vec
    model = Word2Vec(walks, vector_size=dim, window=10, min_count=1, sg=1, workers=4, epochs=5)
    emb = np.array([model.wv[str(i)] for i in range(n)], dtype="float32")

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    np.save(save_path, emb)
    print(f"[GPE] generated {save_path} shape={emb.shape}")
    return emb


# ---------------------------------------------------------------------------
# Tiny synthetic graph for smoke testing — no external data needed.
# ---------------------------------------------------------------------------


def tiny_synthetic_graph(
    n: int = 400,
    feat_dim: int = 32,
    avg_degree: float = 6.0,
    n_communities: int = 4,
    rho: float = 10.0,
    seed: int = 0,
    device: str = "cpu",
) -> GraphUpliftSample:
    """Build a small SBM-style graph with known communities, run our DGP on it.

    Used by the smoke test so we can validate the whole training loop end-to-end
    without depending on CoraFull download / node2vec embeddings.
    """
    from .dgp import semi_synthetic, normalize_adjacency

    rng = np.random.default_rng(seed)
    # Stochastic Block Model: high intra-community probability, low inter.
    sizes = [n // n_communities] * n_communities
    sizes[-1] += n - sum(sizes)
    p_in = avg_degree / max(sizes[0] - 1, 1) * 0.8
    p_out = avg_degree / n * 0.2
    block_idx = np.repeat(np.arange(n_communities), sizes)
    rand = rng.random((n, n))
    P = np.where(block_idx[:, None] == block_idx[None, :], p_in, p_out)
    A_dense = (rand < P).astype("float32")
    np.fill_diagonal(A_dense, 0.0)
    A_dense = np.maximum(A_dense, A_dense.T)
    A = sp.csr_matrix(A_dense)

    # Random features with a small per-community offset.
    base = rng.standard_normal((n, feat_dim)).astype("float32")
    offset = rng.standard_normal((n_communities, feat_dim)).astype("float32") * 0.5
    X = base + offset[block_idx]

    cfg = dict(
        emb_dim=min(16, feat_dim),
        C=5.0,
        beta_0=-0.3,
        beta_1=0.3,
        gamma=0.6,
        kappa1=0.2,
        kappa2=0.1,
        kappa3=0.05,
        kappa4=0.05,
        hops=3,
        rho=rho,
        intra_w=3.0,
    )
    out = semi_synthetic(X, A, cfg, rng=rng)
    A_boost = out["A"]
    A_norm = normalize_adjacency(A_boost, self_loop=True)
    pos = rng.standard_normal((n, 16)).astype("float32") * 0.1  # fake "node2vec"

    return GraphUpliftSample(
        X=torch.tensor(out["Z"], dtype=torch.float32, device=device),
        pos=torch.tensor(pos, dtype=torch.float32, device=device),
        edge_index=_scipy_to_edge_index(A_boost).to(device),
        A_norm=_scipy_to_torch_sparse(A_norm).to(device),
        T=torch.tensor(out["T"], dtype=torch.float32, device=device),
        Y=torch.tensor(out["Y"], dtype=torch.float32, device=device),
        Y0=torch.tensor(out["Y0"], dtype=torch.float32, device=device),
        Y1=torch.tensor(out["Y1"], dtype=torch.float32, device=device),
        true_tau=torch.tensor(out["true_tau"], dtype=torch.float32, device=device),
    )

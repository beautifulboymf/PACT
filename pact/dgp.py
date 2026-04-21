"""Semi-synthetic data generation following PACT paper §5.1.

Inputs:
    X: (N, D) raw node features
    A: (N, N) scipy.sparse adjacency
    cfg: dict-like with kappa1..kappa4, beta_0, beta_1, gamma, hops, C, rho

Outputs:
    dict with Z, T, Y, Y0, Y1, true_tau, prob, A_norm, communities

Heteroscedasticity: epsilon ~ N(0, sigma(Z)) where sigma(Z) is computed by
projecting Z onto a fixed random direction omega and linearly mapping the
result to [-rho, rho].

"""

from __future__ import annotations

from typing import Any

import community as community_louvain  # python-louvain
import networkx as nx
import numpy as np
import scipy.sparse as sp
from scipy.linalg import svd


def normalize_adjacency(A: sp.spmatrix, self_loop: bool = True) -> sp.csr_matrix:
    """Symmetric normalization: D^-1/2 (A + I) D^-1/2."""
    A = A.astype("float32")
    if self_loop:
        A = A + sp.eye(A.shape[0], dtype="float32")
    deg = np.array(A.sum(axis=1)).reshape(-1)
    d_inv_sqrt = np.power(deg, -0.5, where=deg > 0)
    d_inv_sqrt[~np.isfinite(d_inv_sqrt)] = 0.0
    D_inv = sp.diags(d_inv_sqrt)
    return (D_inv @ A @ D_inv).tocsr()


def detect_communities(A: sp.spmatrix, resolution: float = 1.0) -> tuple[np.ndarray, dict[int, list[int]]]:
    """Run Louvain on the (un-normalized) adjacency. Returns (node_class, communities)."""
    G = nx.from_scipy_sparse_array(A)
    partition = community_louvain.best_partition(G, resolution=resolution)
    n = A.shape[0]
    node_class = np.zeros(n, dtype=np.int32)
    communities: dict[int, list[int]] = {}
    for node, c in partition.items():
        node_class[node] = c
        communities.setdefault(c, []).append(node)
    return node_class, communities


def detect_communities_leiden(A: sp.spmatrix, resolution: float = 1.0) -> tuple[np.ndarray, dict[int, list[int]]]:
    """Run the Leiden algorithm. Same return shape as detect_communities().

    Requires `leidenalg` and `igraph`. Used for reviewer-sensitivity experiments;
    not a substitute for Louvain in the main DGP.
    """
    import igraph as ig              # noqa: import-at-use — optional dep
    import leidenalg                 # noqa: import-at-use — optional dep

    A_coo = A.tocoo()
    n = A.shape[0]
    edges = list(zip(A_coo.row.tolist(), A_coo.col.tolist()))
    G = ig.Graph(n=n, edges=edges, directed=False)
    G = G.simplify(multiple=True, loops=True)
    part = leidenalg.find_partition(
        G,
        leidenalg.RBConfigurationVertexPartition,
        resolution_parameter=resolution,
    )
    membership = np.asarray(part.membership, dtype=np.int32)
    node_class = membership
    communities: dict[int, list[int]] = {}
    for node, c in enumerate(membership):
        communities.setdefault(int(c), []).append(node)
    return node_class, communities


def detect_communities_infomap(A: sp.spmatrix) -> tuple[np.ndarray, dict[int, list[int]]]:
    """Run the Infomap algorithm. Same return shape as detect_communities().

    Requires the `infomap` package. Used for reviewer-sensitivity experiments.
    """
    from infomap import Infomap      # noqa: import-at-use — optional dep

    n = A.shape[0]
    # Treat the graph as undirected; let infomap do its own clustering.
    # `--flow-model undirected` is critical: by default infomap assumes a
    # directed flow, which can collapse an undirected graph onto one module.
    im = Infomap("--flow-model undirected --silent")
    # Pre-register every node so isolated nodes also get their own module.
    for i in range(n):
        im.add_node(int(i))
    A_coo = A.tocoo()
    for u, v, w in zip(A_coo.row.tolist(), A_coo.col.tolist(), A_coo.data.tolist()):
        if u < v:                    # undirected, add each edge once
            im.add_link(int(u), int(v), float(w))
    im.run()
    node_class = np.zeros(n, dtype=np.int32)
    communities: dict[int, list[int]] = {}
    for node in im.tree:
        if node.is_leaf:
            nid = node.node_id
            c = node.module_id
            node_class[nid] = c
            communities.setdefault(int(c), []).append(int(nid))
    return node_class, communities


_COMMUNITY_DETECTORS = {
    "louvain": detect_communities,
    "leiden": detect_communities_leiden,
    "infomap": detect_communities_infomap,
}


def detect_communities_dispatch(A: sp.spmatrix, method: str = "louvain") -> tuple[np.ndarray, dict[int, list[int]]]:
    """Dispatch to the requested community-detection algorithm."""
    method = method.lower()
    if method not in _COMMUNITY_DETECTORS:
        raise ValueError(
            f"unknown community_detection={method!r}; choose from {list(_COMMUNITY_DETECTORS)}"
        )
    fn = _COMMUNITY_DETECTORS[method]
    # Infomap doesn't take a resolution parameter.
    if method == "infomap":
        return fn(A)
    return fn(A)


def boost_intra_community_edges(A: sp.spmatrix, node_class: np.ndarray, intra_w: float = 3.0) -> sp.csr_matrix:
    """Reweight intra-community edges to `intra_w`, leave inter-community at 1."""
    A = A.tocoo()
    same = node_class[A.row] == node_class[A.col]
    new_data = np.where(same, intra_w, 1.0).astype("float32")
    return sp.csr_matrix((new_data, (A.row, A.col)), shape=A.shape)


def semi_synthetic(
    X: np.ndarray,
    A: sp.spmatrix,
    cfg: dict[str, Any],
    rng: np.random.Generator | None = None,
) -> dict[str, Any]:
    """Generate semi-synthetic uplift data on a graph.

    cfg keys (with defaults):
        emb_dim    : int     SVD truncation dimension                          (128)
        C          : float   global outcome scale                                (5)
        beta_0     : float   control bias                                      (-0.3)
        beta_1     : float   treatment bias                                     (0.3)
        gamma      : float   per-hop decay                                      (0.6)
        kappa1..4  : float   self / k-hop / treatment-spillover / community     (varies)
        hops       : int     number of hops to aggregate                         (5)
        rho        : float   heteroscedastic noise scale (paper rho)             (10)
        intra_w    : float   intra-community edge boost                          (3.0)
    """
    rng = rng or np.random.default_rng(0)
    emb_dim = cfg.get("emb_dim", 128)
    C = cfg.get("C", 5.0)
    beta_0 = cfg.get("beta_0", -0.3)
    beta_1 = cfg.get("beta_1", 0.3)
    gamma = cfg.get("gamma", 0.6)
    k1, k2, k3, k4 = cfg.get("kappa1", 0.2), cfg.get("kappa2", 0.1), cfg.get("kappa3", 0.03), cfg.get("kappa4", 0.05)
    hops = cfg.get("hops", 5)
    rho = cfg.get("rho", 10.0)
    intra_w = cfg.get("intra_w", 3.0)
    community_detection = cfg.get("community_detection", "louvain")

    # 1. Community detection + intra-community edge boost (paper §5.1).
    #    Default is Louvain (paper main results); leiden / infomap are
    #    alternative algorithms used in the reviewer-sensitivity experiment.
    node_class, communities = detect_communities_dispatch(
        A.tocsr(), method=community_detection
    )
    A_boosted = boost_intra_community_edges(A.tocsr(), node_class, intra_w=intra_w)

    # 2. SVD-based feature compression Z = X V_k.
    X = np.asarray(X, dtype=np.float32)
    X_centered = X - X.mean(axis=0, keepdims=True)
    _, _, VT = svd(X_centered, full_matrices=False)
    k = min(emb_dim, VT.shape[0])
    Z = X_centered @ VT[:k, :].T  # [N, k]

    # 3. Treatment representative vectors c1, c0.
    norms = np.linalg.norm(Z, axis=1)
    c1 = Z[int(np.argmax(norms))]
    c0 = Z.mean(axis=0)

    # 4. Multi-hop aggregation of self/neighbor influences on probability.
    A_norm = normalize_adjacency(A_boosted, self_loop=False)
    Z_c1 = Z @ c1
    Z_c0 = Z @ c0
    AhZ = A_norm @ Z
    AhZ_c1 = AhZ @ c1
    AhZ_c0 = AhZ @ c0

    cur_adj = A_boosted
    decay = 1.0
    for _ in range(hops - 1):
        cur_adj = cur_adj @ A_boosted
        decay *= gamma
        cur_norm = normalize_adjacency(cur_adj, self_loop=False)
        AhZ_c1 = AhZ_c1 + decay * (cur_norm @ Z @ c1)
        AhZ_c0 = AhZ_c0 + decay * (cur_norm @ Z @ c0)

    # 5. Community bias: each community gets a random direction projected on c1.
    c_bias = np.zeros(Z.shape[0], dtype=np.float32)
    for cid, members in communities.items():
        d = rng.standard_normal(Z.shape[1]).astype(np.float32)
        d /= np.linalg.norm(d) + 1e-12
        c_bias[members] = float(d @ c1)

    p1 = k1 * Z_c1 + k2 * AhZ_c1 + k4 * c_bias
    p0 = k1 * Z_c0 + k2 * AhZ_c0
    prob = 1.0 / (1.0 + np.exp(p0 - p1))
    T = rng.binomial(1, prob).astype(np.float32)

    # 6. Treatment-spillover via centered T.
    T_centered = (2 * T - 1.0) / max(T.shape[0], 1)
    AT = A_norm @ T_centered
    cur_adj = A_boosted
    decay = 1.0
    for _ in range(hops - 1):
        cur_adj = cur_adj @ A_boosted
        decay *= gamma
        cur_norm = normalize_adjacency(cur_adj, self_loop=False)
        AT = AT + decay * (cur_norm @ T_centered)

    # 7. Potential outcomes f0, f1.
    f1 = p1 + beta_1 + k3 * AT + k4 * c_bias
    f0 = p0 + beta_0 + k3 * AT + k4 * c_bias

    # 8. Heteroscedastic noise sigma(Z).
    omega = rng.standard_normal(Z.shape[1]).astype(np.float32)
    omega /= np.linalg.norm(omega) + 1e-12
    raw = Z @ omega
    rmin, rmax = float(raw.min()), float(raw.max())
    if rmax > rmin:
        scale = -rho + 2 * rho * (raw - rmin) / (rmax - rmin)
    else:
        scale = np.zeros_like(raw)
    sigma = np.abs(scale).astype(np.float32) + 1e-3
    eps = rng.standard_normal(Z.shape[0]).astype(np.float32) * sigma

    Y0 = (C * f0 + eps).astype(np.float32)
    Y1 = (C * (f0 + f1) + eps).astype(np.float32)
    Y = T * Y1 + (1.0 - T) * Y0
    true_tau = (Y1 - Y0).astype(np.float32)

    return {
        "Z": Z,
        "T": T,
        "Y": Y.astype(np.float32),
        "Y0": Y0,
        "Y1": Y1,
        "true_tau": true_tau,
        "prob": prob.astype(np.float32),
        "A": A_boosted,
        "A_norm": A_norm,
        "node_class": node_class,
        "sigma": sigma,
    }

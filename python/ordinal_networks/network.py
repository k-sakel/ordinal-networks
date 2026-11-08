"""Ordinal partition networks as Markov chains on the ordinal partition.

The network is stored as a weighted edge list (``src``, ``dst``, ``weight`` =
transition counts, self-loops included). Every matrix is derived from it:

* ``W``  weighted adjacency (counts), ``W[i, j]`` = #(pattern i -> pattern j)
* ``A``  binary connectivity matrix
* ``T``  row-stochastic transition matrix, ``T = diag(1/W 1) W``
  (MATLAB ``RightStochasticNetwork``)
* ``Q``  column-normalised matrix, ``Q[i, j] = W[i, j] / sum_k W[k, j]``
  (MATLAB ``W./vecnorm(W,1)`` in ``LyapunovSum_OrdNet``)
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
from scipy.sparse.csgraph import connected_components

from . import _core
from . import patterns as _pat

DENSE_LIMIT = 1500  # use dense eigensolvers below this many nodes


def _leading_eig(M: sp.spmatrix, which: str = "LR"):
    """Leading eigenpair of a (sparse) nonnegative matrix."""
    n = M.shape[0]
    if n == 0:
        raise ValueError("empty network")
    if n <= DENSE_LIMIT:
        vals, vecs = np.linalg.eig(M.toarray())
        k = int(np.argmax(vals.real if which == "LR" else np.abs(vals)))
        return vals[k], vecs[:, k]
    try:
        vals, vecs = sla.eigs(M.astype(np.float64), k=1, which=which, maxiter=max(10000, 10 * n), tol=1e-12)
        return vals[0], vecs[:, 0]
    except sla.ArpackNoConvergence:
        vals, vecs = sla.eigs(M.astype(np.float64), k=2, which=which, maxiter=50 * n, tol=1e-10)
        k = int(np.argmax(vals.real if which == "LR" else np.abs(vals)))
        return vals[k], vecs[:, k]


@dataclass
class OrdinalNetwork:
    src: np.ndarray
    dst: np.ndarray
    weight: np.ndarray
    n_nodes: int
    node_counts: np.ndarray | None = None
    id_hi: np.ndarray | None = None
    id_lo: np.ndarray | None = None
    m: int | None = None
    ranking: str = "chronological"
    _cache: dict = field(default_factory=dict, repr=False, compare=False)

    # -- construction -------------------------------------------------------
    @classmethod
    def from_statistics(cls, st) -> "OrdinalNetwork":
        return cls(src=np.asarray(st.src, dtype=np.int64), dst=np.asarray(st.dst, dtype=np.int64),
                   weight=np.asarray(st.weight, dtype=np.float64), n_nodes=st.n_patterns,
                   node_counts=np.asarray(st.pattern_counts), id_hi=st.id_hi, id_lo=st.id_lo,
                   m=st.m, ranking=st.ranking)

    @classmethod
    def from_edges(cls, src, dst, weight=None, n_nodes=None) -> "OrdinalNetwork":
        src = np.asarray(src, dtype=np.int64)
        dst = np.asarray(dst, dtype=np.int64)
        weight = np.ones(len(src)) if weight is None else np.asarray(weight, dtype=np.float64)
        n = int(max(src.max(initial=-1), dst.max(initial=-1)) + 1) if n_nodes is None else int(n_nodes)
        return cls(src=src, dst=dst, weight=weight, n_nodes=n)

    @classmethod
    def from_matrix(cls, W) -> "OrdinalNetwork":
        W = sp.coo_matrix(W)
        return cls.from_edges(W.row, W.col, W.data, W.shape[0])

    def __repr__(self):
        return f"OrdinalNetwork(m={self.m}, nodes={self.n_nodes}, edges={self.n_edges}, self_loops={self.n_self_loops})"

    @property
    def n_edges(self) -> int:
        return len(self.weight)

    @property
    def n_self_loops(self) -> int:
        return int(np.count_nonzero(self.src == self.dst))

    @property
    def pattern_ids(self) -> np.ndarray:
        if self.id_hi is None:
            raise ValueError("network has no pattern IDs attached")
        return _pat.join(self.id_hi, self.id_lo, self.m)

    def permutations(self) -> np.ndarray:
        return _pat.decode(self.pattern_ids, self.m, self.ranking)

    # -- matrices -----------------------------------------------------------
    @property
    def W(self) -> sp.csr_matrix:
        if "W" not in self._cache:
            self._cache["W"] = sp.csr_matrix((self.weight, (self.src, self.dst)), shape=(self.n_nodes, self.n_nodes))
        return self._cache["W"]

    @property
    def A(self) -> sp.csr_matrix:
        if "A" not in self._cache:
            A = self.W.copy()
            A.data[:] = 1.0
            self._cache["A"] = A
        return self._cache["A"]

    @property
    def T(self) -> sp.csr_matrix:
        """Row-stochastic transition matrix (rows without out-links stay zero)."""
        if "T" not in self._cache:
            s = np.asarray(self.W.sum(axis=1)).ravel()
            inv = np.divide(1.0, s, out=np.zeros_like(s), where=s > 0)
            self._cache["T"] = sp.diags(inv) @ self.W
        return self._cache["T"]

    @property
    def Q(self) -> sp.csr_matrix:
        """Column-normalised matrix (columns without in-links stay zero)."""
        if "Q" not in self._cache:
            s = np.asarray(self.W.sum(axis=0)).ravel()
            inv = np.divide(1.0, s, out=np.zeros_like(s), where=s > 0)
            self._cache["Q"] = (self.W @ sp.diags(inv)).tocsr()
        return self._cache["Q"]

    def edge_probabilities(self) -> np.ndarray:
        """Empirical probability of each edge (transition)."""
        return self.weight / self.weight.sum()

    # -- editing ------------------------------------------------------------
    def subgraph(self, nodes, relabel: bool = True) -> "OrdinalNetwork":
        nodes = np.unique(np.asarray(nodes, dtype=np.int64))
        keep = np.zeros(self.n_nodes, dtype=bool)
        keep[nodes] = True
        e = keep[self.src] & keep[self.dst]
        new_index = -np.ones(self.n_nodes, dtype=np.int64)
        new_index[nodes] = np.arange(len(nodes))
        sub = lambda a: None if a is None else np.asarray(a)[nodes]
        return OrdinalNetwork(src=new_index[self.src[e]], dst=new_index[self.dst[e]], weight=self.weight[e],
                              n_nodes=len(nodes), node_counts=sub(self.node_counts), id_hi=sub(self.id_hi),
                              id_lo=sub(self.id_lo), m=self.m, ranking=self.ranking)

    def prune(self, min_rel_freq: float | None = None, min_count: float | None = None,
              drop_isolated: bool = True) -> "OrdinalNetwork":
        """Remove low-frequency transitions.

        min_rel_freq : drop edges whose relative frequency (weight / total weight)
            is <= this value (MATLAB ``SpectralRadius_OrdNet`` used 1e-4).
        min_count : drop edges with weight < this count.
        drop_isolated : remove nodes left without any edge.
        """
        keep = np.ones(self.n_edges, dtype=bool)
        if min_rel_freq is not None:
            keep &= self.edge_probabilities() > min_rel_freq
        if min_count is not None:
            keep &= self.weight >= min_count
        net = OrdinalNetwork(src=self.src[keep], dst=self.dst[keep], weight=self.weight[keep], n_nodes=self.n_nodes,
                             node_counts=self.node_counts, id_hi=self.id_hi, id_lo=self.id_lo, m=self.m,
                             ranking=self.ranking)
        if drop_isolated:
            used = np.unique(np.concatenate([net.src, net.dst]))
            if len(used) < net.n_nodes:
                net = net.subgraph(used)
        return net

    def without_self_loops(self) -> "OrdinalNetwork":
        keep = self.src != self.dst
        return OrdinalNetwork(src=self.src[keep], dst=self.dst[keep], weight=self.weight[keep], n_nodes=self.n_nodes,
                              node_counts=self.node_counts, id_hi=self.id_hi, id_lo=self.id_lo, m=self.m,
                              ranking=self.ranking)

    # -- connectivity -------------------------------------------------------
    def components(self, connection: str = "strong") -> tuple[int, np.ndarray]:
        """(number of components, label of each node); ``connection`` = 'strong' or 'weak'."""
        return connected_components(self.A, directed=True, connection=connection)

    def is_irreducible(self) -> bool:
        """True if the network is strongly connected (an irreducible Markov chain)."""
        return self.components("strong")[0] == 1

    def largest_scc(self) -> "OrdinalNetwork":
        """Restriction to the largest strongly connected component."""
        n, lab = self.components("strong")
        if n == 1:
            return self
        big = np.argmax(np.bincount(lab))
        return self.subgraph(np.flatnonzero(lab == big))

    # -- spectral / ergodic measures ------------------------------------------
    def spectral_radius(self, log: bool = False) -> float:
        """Spectral radius of the binary connectivity matrix A (Method IV).

        ``log=True`` returns ln(rho(A)), the topological-entropy estimate of
        Sakellariou, Stemler & Small.
        """
        val, _ = _leading_eig(self.A, which="LM")
        rho = float(abs(val))
        return math.log(rho) if log else rho

    def stationary_distribution(self) -> np.ndarray:
        """Left Perron eigenvector of T normalised to sum 1 (invariant measure of the chain)."""
        _, v = _leading_eig(self.T.T.tocsr(), which="LR")
        v = np.abs(v.real)
        return v / v.sum()

    def lyapunov_sum(self, method: str = "Q", return_details: bool = False):
        """Ordinal estimate of the sum of positive Lyapunov exponents.

        ``method="Q"`` reproduces MATLAB ``LyapunovSum_OrdNet``::

            Q = W ./ colsum(W);  u = leading right eigenvector of Q (sum 1)
            P = diag(1/u) Q diag(u);  v = leading left eigenvector of P (sum 1)
            lambda = -sum_ij v_i P_ij log Q_ij

        ``method="T"`` is the commented-out alternative using the row-stochastic T.
        Nodes outside the support of the leading eigenvector are excluded.
        """
        if method.upper() == "Q":
            M = self.Q
            _, u = _leading_eig(M, which="LR")
        elif method.upper() == "T":
            M = self.T
            _, u = _leading_eig(M.T.tocsr(), which="LR")
        else:
            raise ValueError("method must be 'Q' or 'T'")
        u = np.real(u)
        u = u / u.sum()
        supp = np.abs(u) > 1e-14 * np.abs(u).max()
        if not supp.all():
            idx = np.flatnonzero(supp)
            M = M[idx][:, idx]
            u = u[idx]
        M = M.tocoo()
        if method.upper() == "Q":
            pdata = M.data * u[M.col] / u[M.row]          # P = diag(1/u) Q diag(u)
        else:
            pdata = M.data * u[M.row] / u[M.col]          # P = diag(u) T diag(1/u)
        P = sp.csr_matrix((pdata, (M.row, M.col)), shape=M.shape)
        if method.upper() == "Q":
            _, v = _leading_eig(P.T.tocsr(), which="LR")      # left eigenvector of P
        else:
            _, v = _leading_eig(P, which="LR")                # right eigenvector of P
        v = np.real(v)
        v = v / v.sum()
        # use the aligned COO arrays of M (P has the same sparsity pattern)
        if method.upper() == "Q":
            lam = -np.sum(v[M.row] * pdata * np.log(M.data))
        else:
            lam = -np.sum(pdata * np.log(M.data) * v[M.col])
        if return_details:
            return float(lam), {"u": u, "v": v, "support": supp}
        return float(lam)

    def metric_entropy_rate(self, base: float = math.e) -> float:
        """Entropy rate of the Markov chain, -sum_i pi_i sum_j T_ij log T_ij."""
        pi = self.stationary_distribution()
        T = self.T.tocoo()
        return float(-np.sum(pi[T.row] * T.data * np.log(T.data)) / math.log(base))

    # -- local topology -------------------------------------------------------
    def in_degree(self) -> np.ndarray:
        return np.asarray(self.A.sum(axis=0)).ravel().astype(np.int64)

    def out_degree(self) -> np.ndarray:
        return np.asarray(self.A.sum(axis=1)).ravel().astype(np.int64)

    def in_strength(self) -> np.ndarray:
        return np.asarray(self.W.sum(axis=0)).ravel()

    def out_strength(self) -> np.ndarray:
        return np.asarray(self.W.sum(axis=1)).ravel()

    def avg_neighbor_degree(self, weighted: bool = False) -> np.ndarray:
        """Average nearest-neighbour degree (or strength), shape (n, 2): [in, out].

        Column 0: mean in-degree of each node's predecessors; column 1: mean
        out-degree of its successors (MATLAB ``AvgNearestNeighboursDegree``).
        """
        k_in = self.in_strength() if weighted else self.in_degree().astype(float)
        k_out = self.out_strength() if weighted else self.out_degree().astype(float)
        A = self.A
        n_pred = np.asarray(A.sum(axis=0)).ravel()
        n_succ = np.asarray(A.sum(axis=1)).ravel()
        with np.errstate(invalid="ignore", divide="ignore"):
            a_in = (A.T @ k_in) / n_pred
            a_out = (A @ k_out) / n_succ
        return np.column_stack([a_in, a_out])

    def node_link_entropy(self, normalized: bool = False, base: float = 2.0) -> np.ndarray:
        """Shannon entropy of each row of T (MATLAB ``nodeLinkEntropy``)."""
        T = self.T.tocsr()
        logp = np.log(T.data) / math.log(base)
        H = -np.add.reduceat(T.data * logp, T.indptr[:-1]) if T.nnz else np.zeros(self.n_nodes)
        H[np.diff(T.indptr) == 0] = 0.0
        if normalized:
            H = H / (math.log(self.n_nodes) / math.log(base))
        return H

    def hierarchical_degree(self, h: int = 10, exclude_self: bool = False) -> np.ndarray:
        """exp(node-link entropy of T^j), j = 1..h (shape n x h)."""
        out = np.zeros((self.n_nodes, h))
        P = self.T.tocsr()
        Tj = P.copy()
        for j in range(h):
            M = Tj.copy()
            if exclude_self:
                M.setdiag(0)
                M.eliminate_zeros()
                s = np.asarray(M.sum(axis=1)).ravel()
                M = sp.diags(np.divide(1.0, s, out=np.zeros_like(s), where=s > 0)) @ M
            M = M.tocsr()
            logp = np.log(M.data)
            H = -np.add.reduceat(M.data * logp, M.indptr[:-1]) if M.nnz else np.zeros(self.n_nodes)
            H[np.diff(M.indptr) == 0] = 0.0
            out[:, j] = np.exp(H)
            Tj = (Tj @ P).tocsr()
        return out

    def clustering(self) -> np.ndarray:
        """Local clustering of the undirected, unweighted, loop-free version (MATLAB ``clusteringCoeff``)."""
        U = ((self.A + self.A.T) > 0).astype(np.float64).tolil()
        U.setdiag(0)
        U = U.tocsr()
        U.eliminate_zeros()
        k = np.asarray(U.sum(axis=1)).ravel()
        tri = np.asarray((U @ U).multiply(U).sum(axis=1)).ravel()
        with np.errstate(invalid="ignore", divide="ignore"):
            c = np.where(k >= 2, tri / (k * (k - 1)), 0.0)
        return c

    def asymmetry(self) -> float:
        """Asymmetry coefficient of the weighted adjacency (MATLAB ``asymmetryCoeff``)."""
        W = self.without_self_loops().W
        S = (W + W.T).tocoo()
        D = abs(W - W.T).tocsr()
        iu = S.row < S.col
        rows, cols, s = S.row[iu], S.col[iu], S.data[iu]
        z = np.asarray(D[rows, cols]).ravel() / s
        return float(1.0 - z.sum() / len(s)) if len(s) else float("nan")

    def n_cycles(self, k: int) -> int:
        """Number of closed walks of length k, trace(A^k) (MATLAB ``numberOfCycles``)."""
        return int(round(self.cycle_diagonal(k).sum()))

    def cycle_diagonal(self, k: int) -> np.ndarray:
        """diag(A^k): number of closed k-walks through each node."""
        A = self.A.tocsr()
        M = A.copy()
        for _ in range(k - 1):
            M = (M @ A).tocsr()
        return M.diagonal()

    def cycle_nodes(self, k: int) -> np.ndarray:
        """Nodes lying on at least one closed walk of length k (UPO detection)."""
        return np.flatnonzero(self.cycle_diagonal(k) > 0)

    # -- global measures via igraph / networkx ------------------------------
    def to_igraph(self):
        import igraph as ig
        g = ig.Graph(n=self.n_nodes, edges=np.column_stack([self.src, self.dst]).tolist(), directed=True)
        g.es["weight"] = self.weight.tolist()
        return g

    def to_networkx(self):
        import networkx as nx
        g = nx.DiGraph()
        g.add_nodes_from(range(self.n_nodes))
        g.add_weighted_edges_from(zip(self.src.tolist(), self.dst.tolist(), self.weight.tolist()))
        return g

    def shortest_paths(self, weighted: bool = False) -> np.ndarray:
        """All-pairs shortest path lengths (dense n x n). Weighted uses 1/frequency as length."""
        from scipy.sparse.csgraph import shortest_path
        if weighted:
            L = self.T.copy()
            L.data = 1.0 / L.data
        else:
            L = self.A
        return shortest_path(L, directed=True, unweighted=not weighted)

    def characteristic_path_length(self) -> float:
        D = self.shortest_paths()
        n = self.n_nodes
        np.fill_diagonal(D, 0)
        D[~np.isfinite(D)] = 0
        return float(D.sum() / (n * (n - 1)))

    def centrality(self, kind: str, **kw) -> np.ndarray:
        """Node centrality: 'betweenness', 'closeness_in', 'closeness_out', 'harmonic',
        'eigenvector', 'katz', 'pagerank'."""
        kind = kind.lower()
        if kind in ("closeness_in", "closeness_out"):
            D = self.shortest_paths()
            D[~np.isfinite(D)] = 0
            S = D.sum(axis=0) if kind == "closeness_in" else D.sum(axis=1)
            with np.errstate(divide="ignore"):
                return 1.0 / S
        if kind == "harmonic":
            D = self.shortest_paths()
            with np.errstate(divide="ignore"):
                inv = 1.0 / D
            np.fill_diagonal(inv, 0)
            return (self.n_nodes - 1) / inv.sum(axis=0)
        if kind == "eigenvector":
            _, v = _leading_eig(self.A, which="LM")
            v = np.abs(v.real)
            return v / v.sum()
        if kind == "katz":
            alpha = kw.get("alpha")
            if alpha is None:
                alpha = 0.9 / self.spectral_radius()
            M = sp.identity(self.n_nodes, format="csc") - alpha * self.A.tocsc()
            return sla.spsolve(M, np.ones(self.n_nodes))
        if kind == "pagerank":
            g = self.to_igraph()
            return np.asarray(g.pagerank(weights="weight", damping=kw.get("damping", 0.85)))
        if kind == "betweenness":
            g = self.to_igraph()
            return np.asarray(g.betweenness(directed=True))
        raise ValueError(f"unknown centrality {kind!r}")

    def assortativity(self) -> float:
        """Degree assortativity (out-degree of source vs in-degree of target), loops excluded."""
        net = self.without_self_loops()
        ko, ki = net.out_degree(), net.in_degree()
        return float(np.corrcoef(ko[net.src], ki[net.dst])[0, 1])

    def communities(self, method: str = "leiden", **kw):
        """Community detection on the undirected weighted graph (igraph). Returns (membership, modularity)."""
        g = self.to_igraph().as_undirected(combine_edges="sum")
        if method == "leiden":
            cl = g.community_leiden(objective_function="modularity", weights="weight", **kw)
        elif method == "leading_eigenvector":
            cl = g.community_leading_eigenvector(weights="weight", **kw)
        elif method == "multilevel":
            cl = g.community_multilevel(weights="weight", **kw)
        else:
            raise ValueError(method)
        return np.asarray(cl.membership), float(cl.modularity)

    def degree_distribution(self, direction: str = "out") -> tuple[np.ndarray, np.ndarray]:
        """(k, P[k]) for the in- or out-degree."""
        k = self.out_degree() if direction == "out" else self.in_degree()
        vals, cnt = np.unique(k, return_counts=True)
        return vals, cnt / cnt.sum()

    # -- random walks ---------------------------------------------------------
    def random_walk(self, steps: int, start: int | None = None, weighted: bool = True, seed: int | None = None) -> np.ndarray:
        """Random walk of ``steps`` transitions (node indices). ``weighted=False``
        moves uniformly over out-links (MATLAB ``SimpleRandomWalk``), ``True``
        follows the transition probabilities (``TransitionsRandomWalk``)."""
        rng = np.random.default_rng(seed)
        W = self.W.tocsr()
        w = W.data if weighted else np.ones_like(W.data)
        if start is None:
            start = int(rng.integers(self.n_nodes))
        s = int(rng.integers(2 ** 63)) if seed is None else int(seed)
        return _core.random_walk(W.indptr.astype(np.int64), W.indices.astype(np.int64),
                                 np.ascontiguousarray(w, dtype=np.float64), int(start), int(steps), s)

    # -- I/O ------------------------------------------------------------------
    def save(self, path):
        from .io import save_network
        return save_network(self, path)

    @classmethod
    def load(cls, path) -> "OrdinalNetwork":
        from .io import load_network
        return load_network(path)

"""Order-based recurrence tools (Groth 2005) and order structure matrices."""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def order_recurrence_plot(sequence, sparse: bool | None = None):
    """R[i, j] = 1 if windows i and j have the same ordinal pattern.

    ``sequence`` is a symbol/node sequence (e.g. ``stats.sequence``); negative
    entries (skipped windows) never recur. Returns a dense boolean matrix for
    short sequences and a sparse matrix otherwise (or as requested).
    """
    s = np.asarray(sequence)
    n = len(s)
    if sparse is None:
        sparse = n > 5000
    if not sparse:
        R = s[:, None] == s[None, :]
        bad = s < 0 if np.issubdtype(s.dtype, np.signedinteger) else np.zeros(n, bool)
        R[bad, :] = False
        R[:, bad] = False
        return R
    order = np.argsort(s, kind="stable")
    ss = s[order]
    bounds = np.flatnonzero(np.diff(ss)) + 1
    rows, cols = [], []
    for grp, val in zip(np.split(order, bounds), ss[np.r_[0, bounds]]):
        if np.issubdtype(s.dtype, np.signedinteger) and val < 0:
            continue
        rows.append(np.repeat(grp, len(grp)))
        cols.append(np.tile(grp, len(grp)))
    r = np.concatenate(rows) if rows else np.zeros(0, int)
    c = np.concatenate(cols) if cols else np.zeros(0, int)
    return sp.csr_matrix((np.ones(len(r), dtype=bool), (r, c)), shape=(n, n))


def recurrence_rate(sequence) -> float:
    """Fraction of recurrent pairs, sum_pi p(pi)^2 (computed without the matrix)."""
    s = np.asarray(sequence)
    s = s[s >= 0] if np.issubdtype(s.dtype, np.signedinteger) else s
    _, c = np.unique(s, return_counts=True)
    return float(np.sum(c.astype(float) ** 2) / len(s) ** 2)


def order_structure_matrix(x) -> np.ndarray:
    """B[s, t] = 1 if x_s >= x_t for s < t, B[t, s] = not B[s, t], diag = 1
    (MATLAB ``OrderStructureMatrix``)."""
    x = np.asarray(x, dtype=float)
    ge = x[:, None] >= x[None, :]
    B = np.triu(ge, 1)
    B = B | np.tril(~B.T, -1)
    np.fill_diagonal(B, True)
    return B

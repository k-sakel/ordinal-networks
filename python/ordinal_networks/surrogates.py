"""Surrogate time series from random walks on an ordinal network.

A walk on the network is a sequence of overlapping ordinal patterns. Each new
pattern fixes the rank of one new point relative to the previous m-1 points;
the new value is drawn from the observed last values of windows with that
pattern that satisfy the constraint (MATLAB ``GetTimeSeriesReconstruction``
and ``ConstrainedReconstruction``). Requires ``tau = w = 1``.
"""
from __future__ import annotations

import numpy as np

from . import patterns as _pat


def reconstruct(walk, stats, x, method: str = "random", seed: int | None = None) -> tuple[np.ndarray, int]:
    """Turn a walk (node indices) into a surrogate series.

    method : ``"random"`` picks a uniformly random admissible candidate value;
        ``"constrained"`` picks the candidate closest to the previous value.
    Returns (series of length len(walk) + m - 1, number of steps where no
    observed value satisfied the constraint and a fallback was used).
    """
    if stats.tau != 1 or stats.w != 1:
        raise ValueError("reconstruction requires tau = 1 and w = 1")
    if stats.sequence is None:
        raise ValueError("fit with keep_sequence=True")
    rng = np.random.default_rng(seed)
    m = stats.m
    x = np.asarray(x, dtype=float)
    seq = np.asarray(stats.sequence)
    walk = np.asarray(walk, dtype=np.int64)
    # candidate last values for every node, sorted
    ok = seq >= 0
    idx = np.flatnonzero(ok)
    nodes = seq[ok]
    order = np.argsort(nodes, kind="stable")
    nodes_s = nodes[order]
    starts = np.searchsorted(nodes_s, np.arange(stats.n_patterns))
    ends = np.searchsorted(nodes_s, np.arange(stats.n_patterns), side="right")
    last_vals = x[idx[order] + m - 1]
    first_win = idx[order]
    cand = [np.sort(last_vals[a:b]) for a, b in zip(starts, ends)]
    ranks = _pat.ascending_ranks(stats.pattern_ids[np.unique(walk)], m, stats.ranking)
    rank_of = dict(zip(np.unique(walk).tolist(), ranks))

    out = np.empty(len(walk) + m - 1)
    n0 = walk[0]
    j0 = first_win[starts[n0] + rng.integers(ends[n0] - starts[n0])]
    out[:m] = x[j0:j0 + m]
    fallback = 0
    for i in range(1, len(walk)):
        r = rank_of[int(walk[i])]
        prev = out[i:i + m - 1]                 # the m-1 retained points
        rl = r[m - 1]
        lo = prev[np.flatnonzero(r[:m - 1] == rl - 1)[0]] if rl > 1 else -np.inf
        hi = prev[np.flatnonzero(r[:m - 1] == rl + 1)[0]] if rl < m else np.inf
        c = cand[walk[i]]
        a = np.searchsorted(c, lo, side="left")
        b = np.searchsorted(c, hi, side="left")
        if b > a:
            if method == "random":
                v = c[a + rng.integers(b - a)]
            elif method == "constrained":
                seg = c[a:b]
                v = seg[np.argmin(np.abs(seg - out[i + m - 2]))]
            else:
                raise ValueError("method must be 'random' or 'constrained'")
        else:
            fallback += 1
            if rl == m:
                v = prev.max() + 1e-3
            elif rl == 1:
                v = prev.min() - 1e-3
            else:
                v = 0.5 * (lo + hi)
                if not (lo <= v < hi):
                    v = lo          # tie with an earlier point counts as larger

        out[i + m - 1] = v
    return out, fallback


def network_surrogate(stats, x, length: int, method: str = "random", weighted: bool = True,
                      seed: int | None = None) -> tuple[np.ndarray, int]:
    """Random walk on ``stats.network()`` followed by :func:`reconstruct`."""
    rng = np.random.default_rng(seed)
    net = stats.network()
    walk = net.random_walk(length - stats.m, weighted=weighted, seed=int(rng.integers(2 ** 62)))
    return reconstruct(walk, stats, x, method=method, seed=int(rng.integers(2 ** 62)))


def shuffle_surrogate(x, seed: int | None = None) -> np.ndarray:
    """Random permutation of the series (i.i.d. null hypothesis)."""
    return np.random.default_rng(seed).permutation(np.asarray(x))

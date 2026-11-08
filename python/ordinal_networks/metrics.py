"""Distances between permutations and rank-based helpers.

All permutations are 1-based sequences of the same length ``n``. Distances are
normalised to [0, 1] by their maximum value, as in the MATLAB toolbox.
"""
from __future__ import annotations

import numpy as np


def _pair(s1, s2):
    a, b = np.asarray(s1, dtype=np.int64), np.asarray(s2, dtype=np.int64)
    if a.shape != b.shape:
        raise ValueError("The two permutations should be of the same order.")
    return a, b


def hamming(s1, s2) -> float:
    """Fraction of positions where the permutations differ.

    (Fixes the MATLAB ``HammingMetric`` typo ``simg2``.)
    """
    a, b = _pair(s1, s2)
    return float(np.count_nonzero(a != b) / a.size)


def kendall_tau_distance(s1, s2) -> float:
    """Number of discordant pairs (= bubble-sort swaps turning one into the
    other), normalised by n(n-1)/2.

    (The MATLAB ``KendallTauMetric`` actually computed a lagged Kendall
    correlation of a series; that is available as :func:`kendall_autocorrelation`.)
    """
    a, b = _pair(s1, s2)
    n = a.size
    if n < 2:
        return 0.0
    da = np.sign(a[:, None] - a[None, :])
    db = np.sign(b[:, None] - b[None, :])
    disc = np.count_nonzero(np.triu(da * db < 0, 1))
    return float(disc / (n * (n - 1) / 2))


def spearman_footrule(s1, s2) -> float:
    """L1 distance, normalised by its maximum (n^2/2 for even n, (n^2-1)/2 for odd n)."""
    a, b = _pair(s1, s2)
    n = a.size
    fmax = n * n / 2 if n % 2 == 0 else (n - 1) * (n + 1) / 2
    return float(np.abs(a - b).sum() / fmax) if fmax else 0.0


def spearman_rho_distance(s1, s2) -> float:
    """L2 distance, normalised by its maximum sqrt(n(n-1)(n+1)/3)."""
    a, b = _pair(s1, s2)
    n = a.size
    rmax = np.sqrt(n * (n - 1) * (n + 1) / 3)
    return float(np.linalg.norm(a - b) / rmax) if rmax else 0.0


def kendall_autocorrelation(x, lag: int = 1) -> float:
    """Kendall rank correlation between x_t and x_{t+lag}."""
    from scipy.stats import kendalltau
    x = np.asarray(x, dtype=float)
    return float(kendalltau(x[:-lag], x[lag:]).statistic)


def rank_numbers(x) -> np.ndarray:
    """Global rank number of every point: #{j : x_j <= x_i} (MATLAB ``RankNumber``)."""
    x = np.asarray(x, dtype=float)
    s = np.sort(x)
    return np.searchsorted(s, x, side="right")


def local_rank_numbers(x, L: int) -> np.ndarray:
    """#{j = 1..L : x_{s+j} <= x_s} for s = 1..N-L (MATLAB ``LocalRankNumber``)."""
    x = np.asarray(x, dtype=float)
    N = len(x)
    out = np.zeros(N - L, dtype=np.int64)
    for j in range(1, L + 1):
        out += x[j:N - L + j] <= x[:N - L]
    return out

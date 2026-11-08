"""Entropy and complexity estimators from ordinal statistics.

The four topological-entropy estimators of Sakellariou, Stemler & Small:

* Method I   ``growth_rate``               slope of ln V(m) against m
* Method II  ``topological_permutation_entropy``  log2 V(m) / (m-1)
* Method III ``log_mean_outgrowth_ratio``  ln G(m)
* Method IV  ``OrdinalNetwork.spectral_radius(log=True)``  ln rho(A)
"""
from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np

from . import patterns as _pat


def permutation_entropy(stats, normalized: bool = False, base: float = 2.0) -> float:
    """Shannon entropy of the pattern distribution (Bandt & Pompe).

    ``normalized=True`` divides by log(m!).
    """
    p = stats.pattern_counts / stats.pattern_counts.sum()
    H = float(-np.sum(p * np.log(p)) / math.log(base))
    if normalized:
        H /= math.log(_pat.factorial(stats.m)) / math.log(base)
    return H


def conditional_entropy(stats, base: float = 2.0) -> float:
    """Conditional entropy of successive patterns (MATLAB ``ConditionalEntropy``):
    ``-sum_ij p(i,j) log( p(i,j) / p(i) )``."""
    p = stats.pattern_counts / stats.pattern_counts.sum()
    t = stats.weight / stats.weight.sum()
    src = np.asarray(stats.src, dtype=np.int64)
    return float(-np.sum(t * np.log(t / p[src])) / math.log(base))


def topological_permutation_entropy(stats) -> float:
    """Method II: log2(V) / (m - 1)."""
    return math.log2(stats.n_patterns) / (stats.m - 1)


def forbidden_fraction(stats) -> float:
    """1 - V/m!, the relative count of forbidden patterns."""
    return 1.0 - stats.n_patterns / _pat.factorial(stats.m)


def growth_rate(sweep: Mapping[int, object], m_range: Sequence[int] | None = None) -> float:
    """Method I: least-squares slope of ln V(m) over ``m_range`` (default: all m)."""
    ms = sorted(sweep) if m_range is None else [m for m in m_range if m in sweep]
    if len(ms) < 2:
        raise ValueError("need at least two pattern lengths")
    V = np.array([sweep[m].n_patterns for m in ms], dtype=float)
    return float(np.polyfit(ms, np.log(V), 1)[0])


def gradient_log_patterns(sweep: Mapping[int, object]) -> tuple[np.ndarray, np.ndarray]:
    """Finite-difference d ln V / dm for consecutive m (MATLAB ``DiscreteMapGradLogV``)."""
    ms = np.array(sorted(sweep))
    lv = np.log([sweep[m].n_patterns for m in ms])
    return ms[1:], np.diff(lv) / np.diff(ms)


def outgrowth_ratio(stats_m, stats_m1) -> np.ndarray:
    """g(pi) for every admissible pattern pi of length m: the number of admissible
    patterns of length m+1 whose first m points form pi (MATLAB ``OutgrowthRatio``).

    Returned in the node order of ``stats_m``.
    """
    if stats_m1.m != stats_m.m + 1:
        raise ValueError("second argument must have pattern length m+1")
    if stats_m.ranking != stats_m1.ranking or stats_m.tau != stats_m1.tau:
        raise ValueError("statistics must share ranking and tau")
    pre = _pat.prefix(stats_m1.pattern_ids, stats_m1.m, stats_m1.ranking)
    nodes = stats_m.nodes_of(pre)
    nodes = nodes[nodes >= 0]
    return np.bincount(nodes, minlength=stats_m.n_patterns)


def log_mean_outgrowth_ratio(stats_m, stats_m1) -> float:
    """Method III: ln G(m), G = mean outgrowth ratio over admissible m-patterns."""
    return float(math.log(outgrowth_ratio(stats_m, stats_m1).mean()))


def entropy_table(sweep: Mapping[int, object], spectral: bool = True, lyapunov: bool = False) -> dict:
    """Convenience: per-m arrays of the main estimators for a sweep (natural log units)."""
    ms = sorted(sweep)
    out = {"m": np.array(ms), "V": np.array([sweep[m].n_patterns for m in ms]),
           "E": np.array([sweep[m].n_transitions for m in ms]),
           "PE": np.array([permutation_entropy(sweep[m], base=math.e) for m in ms]),
           "TPE": np.array([topological_permutation_entropy(sweep[m]) * math.log(2) for m in ms]),
           "lnG": np.array([log_mean_outgrowth_ratio(sweep[m], sweep[m + 1]) if m + 1 in sweep else np.nan for m in ms])}
    if spectral:
        out["ln_rho"] = np.array([sweep[m].network().spectral_radius(log=True) for m in ms])
    if lyapunov:
        out["lyap"] = np.array([sweep[m].network().lyapunov_sum() for m in ms])
    return out

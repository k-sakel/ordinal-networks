"""Entropy and complexity estimators from ordinal statistics.

The four topological-entropy estimators of Sakellariou, Stemler & Small:

* Method I   ``growth_rate``               slope of ln V(m) against m
* Method II  ``topological_permutation_entropy``  log2 V(m) / (m-1)
* Method III ``log_mean_outgrowth_ratio``  ln G(m)
* Method IV  ``OrdinalNetwork.spectral_radius(log=True)``  ln rho(A)

Bandt-Pompe quantifiers computed from the pattern frequencies alone:
``permutation_entropy``, ``renyi_entropy``, ``tsallis_entropy``,
``statistical_complexity`` / ``complexity_entropy`` (the complexity-entropy
plane), ``forbidden_fraction``; and from the transitions:
``conditional_entropy``, ``global_node_entropy``, ``missing_transition_fraction``.
"""
from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np

from . import patterns as _pat


def _require_windows(stats) -> None:
    if stats.n_patterns == 0:
        raise ValueError("no valid windows: the statistics are empty (series too short or all NaN/Inf)")


def permutation_entropy(stats, normalized: bool = False, base: float = 2.0) -> float:
    """Shannon entropy of the pattern distribution (Bandt & Pompe).

    ``normalized=True`` divides by log(m!). ``base`` is the logarithm base (default 2, i.e. bits).
    """
    _require_windows(stats)
    p = stats.pattern_counts / stats.pattern_counts.sum()
    H = float(-np.sum(p * np.log(p)) / math.log(base))
    if normalized:
        H = _normalise(H, stats.m, base)
    return H


def _normalise(H: float, m: int, base: float) -> float:
    """H / log(m!); 0 for m = 1, where there is a single pattern and log(m!) = 0."""
    n = _pat.factorial(m)
    return H / (math.log(n) / math.log(base)) if n > 1 else 0.0


def _transitions(stats):
    """Transition statistics shared by the transition-based entropies: weights w,
    source and target node of every edge, the out-strength of every node (the
    source marginal, which unlike the pattern counts excludes the last window of
    every run) and the transition probability t = w / out[src] of every edge."""
    w = np.asarray(stats.weight, dtype=np.float64)
    src = np.asarray(stats.src, dtype=np.int64)
    dst = np.asarray(stats.dst, dtype=np.int64)
    out = np.bincount(src, weights=w, minlength=stats.n_patterns)
    return w, src, dst, out, w / out[src]


def conditional_entropy(stats, base: float = 2.0) -> float:
    """Conditional entropy of successive patterns (MATLAB ``ConditionalEntropy``):
    ``-sum_ij p(i,j) log( p(i,j) / p(i) )``. ``base`` defaults to 2 (bits).

    ``p(i,j)`` is the relative frequency of the transition i -> j and ``p(i)`` its
    source marginal ``sum_j p(i,j)``, so the value is a proper conditional entropy
    (never negative), also for short series and series with gaps, where the
    pattern frequencies differ from the source marginal (the last window of every
    run has no successor).
    """
    _require_windows(stats)
    if stats.n_transitions == 0:
        return 0.0
    w, _, _, _, t = _transitions(stats)
    h = -np.sum(w * np.log(t)) / w.sum()
    return float(h / math.log(base)) if h > 0 else 0.0


def renyi_entropy(stats, alpha: float = 2.0, normalized: bool = False, base: float = 2.0) -> float:
    """Renyi entropy of order ``alpha`` of the pattern distribution,
    ``log(sum_i p_i^alpha) / (1 - alpha)``; ``alpha = 1`` is the permutation entropy
    and ``alpha = inf`` the min-entropy ``-log(max_i p_i)``.

    ``normalized=True`` divides by log(m!).
    """
    _require_windows(stats)
    if not alpha >= 0:
        raise ValueError("alpha must be >= 0")
    if alpha == 1:
        return permutation_entropy(stats, normalized=normalized, base=base)
    p = stats.pattern_counts / stats.pattern_counts.sum()
    if math.isinf(alpha):
        H = float(-np.log(p.max()) / math.log(base))
    else:
        # log sum_i p_i^alpha with the largest term factored out: p^alpha underflows for large alpha
        pmax = p.max()
        log_sum = alpha * math.log(pmax) + math.log(np.sum((p / pmax) ** alpha))
        H = float(log_sum / (1.0 - alpha) / math.log(base))
    if normalized:
        H = _normalise(H, stats.m, base)
    return H


def _logq(x, q: float):
    return np.log(x) if q == 1 else (np.asarray(x, dtype=float) ** (1.0 - q) - 1.0) / (1.0 - q)


def tsallis_entropy(stats, q: float = 2.0, normalized: bool = False) -> float:
    """Tsallis entropy of order ``q`` of the pattern distribution,
    ``sum_i p_i ln_q(1 / p_i)`` with ``ln_q(x) = (x^(1-q) - 1) / (1 - q)``
    (``q = 1`` is the permutation entropy in nats).

    ``normalized=True`` divides by ``ln_q(m!)``.
    """
    _require_windows(stats)
    p = stats.pattern_counts / stats.pattern_counts.sum()
    S = float(np.sum(p * _logq(1.0 / p, q)))
    if normalized:
        smax = float(_logq(float(_pat.factorial(stats.m)), q))
        S = S / smax if smax > 0 else 0.0
    return S


def statistical_complexity(stats) -> float:
    """Jensen-Shannon statistical complexity ``C = Q_J[P, U] * H[P]`` (Lamberti et al.
    2004; Rosso et al. 2007), where ``H`` is the normalised permutation entropy,
    ``U`` the uniform distribution over all m! patterns and ``Q_J`` the
    Jensen-Shannon divergence normalised to [0, 1].

    The m! - V unobserved patterns enter in closed form, so this works for any m <= 34.
    """
    return complexity_entropy(stats)[1]


def complexity_entropy(stats) -> tuple[float, float]:
    """Point ``(H, C)`` of the series in the complexity-entropy causality plane
    (Rosso et al. 2007): normalised permutation entropy and Jensen-Shannon
    statistical complexity."""
    _require_windows(stats)
    n = float(_pat.factorial(stats.m))
    if n < 2:
        return 0.0, 0.0
    p = stats.pattern_counts / stats.pattern_counts.sum()
    s_p = float(-np.sum(p * np.log(p)))
    h = s_p / math.log(n)
    u = 1.0 / n
    mix = 0.5 * (p + u)
    s_mix = float(-np.sum(mix * np.log(mix))) - (n - len(p)) * (0.5 * u) * math.log(0.5 * u)
    js = s_mix - 0.5 * s_p - 0.5 * math.log(n)
    # log1p keeps the maximum accurate when 1/n is below double precision (m > 18)
    js_max = -0.5 * ((n + 1.0) / n * (math.log(n) + math.log1p(u)) + math.log(n) - 2.0 * math.log(2.0 * n))
    return h, max(h * js / js_max, 0.0)


def global_node_entropy(stats, base: float = math.e) -> float:
    """Global node entropy of the ordinal network (McCullough et al. 2017; Pessa &
    Ribeiro 2019): ``sum_i p_i H_i`` with ``H_i`` the entropy of the out-transition
    probabilities of node i and ``p_i`` its share of the total *incoming* weight
    (the definition used by ``ordpy.global_node_entropy``). In nats by default.

    :func:`conditional_entropy` is the same sum weighted by the *outgoing* share;
    the two agree up to edge effects of order 1/N.
    """
    _require_windows(stats)
    if stats.n_transitions == 0:
        return 0.0
    w, src, dst, _, t = _transitions(stats)
    V = stats.n_patterns
    p_in = np.bincount(dst, weights=w, minlength=V) / w.sum()
    h_i = -np.bincount(src, weights=t * np.log(t), minlength=V)
    return float(np.sum(p_in * h_i) / math.log(base))


def possible_transitions(m: int, tau: int = 1, w: int = 1) -> int:
    """Number of transitions that are combinatorially possible between consecutive
    windows: ``m! * m!/(m-s)!`` when the windows share ``m - s`` points
    (``w = s * tau``, ``s < m``; ``m! * m`` for ``tau = w = 1``), else ``(m!)^2``."""
    f = _pat.factorial(m)
    if w % tau == 0 and w // tau < m:
        return f * (f // _pat.factorial(m - w // tau))
    return f * f


def missing_transition_fraction(stats) -> float:
    """Fraction of the combinatorially possible transitions that never occur
    (the "missing links" of Pessa & Ribeiro 2019); see :func:`possible_transitions`."""
    return 1.0 - stats.n_transitions / possible_transitions(stats.m, stats.tau, stats.w)


def topological_permutation_entropy(stats) -> float:
    """Method II: log2(V) / (m - 1), in bits."""
    _require_windows(stats)
    if stats.m < 2:
        raise ValueError("topological permutation entropy needs m >= 2")
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
    if stats_m.ranking != stats_m1.ranking or stats_m.tau != stats_m1.tau or stats_m.w != stats_m1.w:
        raise ValueError("statistics must share ranking, tau and w")
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
           "TPE": np.array([topological_permutation_entropy(sweep[m]) * math.log(2) if m >= 2 else np.nan for m in ms]),
           "lnG": np.array([log_mean_outgrowth_ratio(sweep[m], sweep[m + 1]) if m + 1 in sweep else np.nan for m in ms])}
    if spectral:
        out["ln_rho"] = np.array([sweep[m].network().spectral_radius(log=True) for m in ms])
    if lyapunov:
        out["lyap"] = np.array([sweep[m].network().lyapunov_sum() for m in ms])
    return out

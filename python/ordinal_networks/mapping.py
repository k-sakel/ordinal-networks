"""Mapping network properties back onto the time series / attractor."""
from __future__ import annotations

import numpy as np


def property_per_window(stats, node_values) -> np.ndarray:
    """Value of a node property for every window (NaN for skipped windows).

    Requires a statistics object fitted with ``keep_sequence=True``.
    """
    if stats.sequence is None:
        raise ValueError("fit with keep_sequence=True to map properties back")
    seq = np.asarray(stats.sequence)
    vals = np.asarray(node_values, dtype=float)
    out = np.full(len(seq), np.nan)
    ok = seq >= 0
    out[ok] = vals[seq[ok]]
    return out


def property_per_point(stats, node_values) -> np.ndarray:
    """Spread window values over time points as MATLAB ``MapNetworkPropertyTo*``:
    window k's value is assigned to points k*w ... k*w + w - 1. Points not
    covered are NaN. Returns an array of length ``stats.n_points``.
    """
    pw = property_per_window(stats, node_values)
    out = np.full(stats.n_points, np.nan)
    rep = np.repeat(pw, stats.w)
    out[: min(len(rep), stats.n_points)] = rep[: stats.n_points]
    return out


def windows_on_cycles(stats, network, k: int) -> np.ndarray:
    """Indices of windows whose pattern lies on a closed k-walk of the network
    (MATLAB ``DiscreteOrdinalNetworkCyclesLocation``). Use with the series to
    locate the neighbourhoods of unstable periodic orbits of order k."""
    if stats.sequence is None:
        raise ValueError("fit with keep_sequence=True")
    on = np.zeros(network.n_nodes, dtype=bool)
    on[network.cycle_nodes(k)] = True
    seq = np.asarray(stats.sequence)
    return np.flatnonzero((seq >= 0) & on[np.maximum(seq, 0)])


def pattern_value_ranges(stats, x) -> np.ndarray:
    """For every node, [min, max] of the first point of the windows carrying that
    pattern (the 1-D projection of the ordinal partition, MATLAB ``Ordinal1D``)."""
    if stats.sequence is None:
        raise ValueError("fit with keep_sequence=True")
    seq = np.asarray(stats.sequence)
    first = np.asarray(x, dtype=float)[np.arange(len(seq)) * stats.w]
    ok = seq >= 0
    lo = np.full(stats.n_patterns, np.inf)
    hi = np.full(stats.n_patterns, -np.inf)
    np.minimum.at(lo, seq[ok], first[ok])
    np.maximum.at(hi, seq[ok], first[ok])
    return np.column_stack([lo, hi])


def ordinal_partition_boundaries(stats, x, decimals: int = 4) -> np.ndarray:
    """Sorted unique boundaries of the 1-D projected ordinal partition."""
    return np.unique(np.round(pattern_value_ranges(stats, x).ravel(), decimals))

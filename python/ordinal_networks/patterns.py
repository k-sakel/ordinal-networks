"""Pattern IDs <-> permutations.

Every ordinal pattern of length ``m`` has a fixed ID in ``{1, ..., m!}``: its
1-based lexicographic rank among all permutations of ``(1, ..., m)`` written in
the chosen ranking convention. ID ``0`` means "no pattern" (a window that
contained NaN/Inf). IDs are exact up to ``m = 34``.

Internally IDs are 128-bit numbers carried as two ``uint64`` arrays
(``hi``, ``lo``). For ``m <= 20`` every ID fits in ``lo`` alone and the public
API returns plain ``uint64`` arrays; for ``m > 20`` it returns ``object``
arrays of Python ints.
"""
from __future__ import annotations

import numpy as np

from . import _core

MAX_M: int = _core.MAX_M
RANKINGS = ("chronological", "amplitude")


def check_ranking(ranking: str) -> str:
    r = ranking.lower()
    aliases = {"bp": "chronological", "bandt-pompe": "chronological", "index": "chronological",
               "small": "amplitude", "rank": "amplitude"}
    r = aliases.get(r, r)
    if r not in RANKINGS:
        raise ValueError(f"ranking must be one of {RANKINGS}, got {ranking!r}")
    return r


def factorial(m: int) -> int:
    """m! as an exact Python int (m <= 34)."""
    return int(_core.factorial(m))


def fits_uint64(m: int) -> bool:
    return m <= 20


def join(hi: np.ndarray, lo: np.ndarray, m: int) -> np.ndarray:
    """Combine (hi, lo) into the public representation."""
    if fits_uint64(m):
        return np.asarray(lo, dtype=np.uint64)
    out = np.empty(len(lo), dtype=object)
    for i, (h, l) in enumerate(zip(hi.tolist(), lo.tolist())):
        out[i] = (h << 64) | l
    return out


def split(ids) -> tuple[np.ndarray, np.ndarray]:
    """Public IDs (ints, uint64 array or object array) -> (hi, lo) uint64 arrays."""
    a = np.asarray(ids)
    if a.dtype != object and np.issubdtype(a.dtype, np.integer):
        if a.size and a.min() < 0:
            raise ValueError("pattern IDs must be positive")
        lo = a.astype(np.uint64).ravel()
        return np.zeros_like(lo), lo
    flat = [int(v) for v in a.ravel().tolist()]
    hi = np.fromiter(((v >> 64) for v in flat), dtype=np.uint64, count=len(flat))
    lo = np.fromiter(((v & 0xFFFFFFFFFFFFFFFF) for v in flat), dtype=np.uint64, count=len(flat))
    return hi, lo


def decode(ids, m: int, ranking: str = "chronological") -> np.ndarray:
    """Pattern ID(s) -> permutation(s), 1-based, shape (n, m) (or (m,) for a scalar).

    * ``chronological`` (Bandt & Pompe): ``pi = (p1..pm)`` with
      ``x[p1] < x[p2] < ... < x[pm]`` (MATLAB ``[~,s] = sort(window)``).
    * ``amplitude`` (Small): entry ``j`` is the rank of the ``j``-th point of
      the window in descending amplitude order (1 = largest).
    """
    scalar = np.ndim(ids) == 0
    hi, lo = split(np.atleast_1d(ids))
    out = _core.decode(hi, lo, m, check_ranking(ranking))
    return out[0] if scalar else out


def encode(perms, ranking: str = "chronological") -> np.ndarray:
    """Permutation(s) (1-based, in the convention of ``ranking``) -> pattern ID(s)."""
    check_ranking(ranking)
    a = np.asarray(perms, dtype=np.int64)
    scalar = a.ndim == 1
    a = np.atleast_2d(a)
    hi, lo = _core.encode_permutations(np.ascontiguousarray(a))
    out = join(hi, lo, a.shape[1])
    return out[0] if scalar else out


def convert(ids, m: int, from_ranking: str, to_ranking: str) -> np.ndarray:
    """Re-express pattern IDs in another ranking convention."""
    if check_ranking(from_ranking) == check_ranking(to_ranking):
        return ids
    p = np.atleast_2d(decode(ids, m, from_ranking)).astype(np.int64)
    # chronological pi <-> ascending ranks r: r[pi[j]-1] = j+1 ; amplitude = m+1-r
    n = p.shape[0]
    r = np.empty_like(p)
    if from_ranking == "chronological":
        rows = np.repeat(np.arange(n), m)
        r[rows, (p - 1).ravel()] = np.tile(np.arange(1, m + 1), n)
        q = m + 1 - r
    else:
        asc = m + 1 - p
        q = np.empty_like(p)
        rows = np.repeat(np.arange(n), m)
        q[rows, (asc - 1).ravel()] = np.tile(np.arange(1, m + 1), n)
    out = encode(q, to_ranking)
    return out if np.ndim(ids) else out[0]


def ascending_ranks(ids, m: int, ranking: str = "chronological") -> np.ndarray:
    """Ascending rank (1 = smallest) of each point of the window, in time order."""
    p = np.atleast_2d(decode(ids, m, ranking)).astype(np.int64)
    if check_ranking(ranking) == "amplitude":
        return m + 1 - p
    n = p.shape[0]
    r = np.empty_like(p)
    rows = np.repeat(np.arange(n), m)
    r[rows, (p - 1).ravel()] = np.tile(np.arange(1, m + 1), n)
    return r


def prefix(ids, m: int, ranking: str = "chronological") -> np.ndarray:
    """IDs (length m-1) of the patterns of the first m-1 points of windows with IDs of length m."""
    hi, lo = split(np.atleast_1d(ids))
    h2, l2 = _core.prefix_ids(hi, lo, m, check_ranking(ranking))
    return join(h2, l2, m - 1)


def pattern_of(window, ranking: str = "chronological") -> int:
    """ID of a single window of values (ties: earlier point is smaller)."""
    x = np.ascontiguousarray(window, dtype=np.float64)
    hi, lo, _ = _core.encode(x, len(x), 1, 1, check_ranking(ranking), 1)
    if lo[0] == 0 and hi[0] == 0:
        raise ValueError("window contains non-finite values")
    return int(join(hi, lo, len(x))[0])

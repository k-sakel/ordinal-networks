"""Ordinal symbolisation of (very long) time series.

>>> sym = OrdinalSymbolizer(m=6)              # tau = 1, w = 1 by default
>>> stats = sym.fit(x)                         # x: array, or a path to .npy/.csv/...
>>> stats.pattern_ids, stats.pattern_counts    # observed patterns & frequencies
>>> net = stats.network()                      # ordinal partition network
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from . import _core
from . import patterns as _pat

DEFAULT_CHUNK = 4_000_000           # windows per chunk
DEFAULT_MEMORY = 3 * 1024 ** 3      # bytes for the in-RAM hash tables (shared by all m of a sweep)


# ----------------------------------------------------------------- input ---

def load_series(source, column: int | str | None = None) -> np.ndarray:
    """Return a 1-D float array (memory-mapped for .npy) from an array or a file.

    Supported files: ``.npy`` (memory-mapped, recommended for N ~ 1e9),
    ``.parquet``, ``.csv``/``.txt``/``.dat`` (whitespace or comma separated),
    ``.json`` (a list). For 2-D data pass ``column``.
    """
    if isinstance(source, (str, os.PathLike)):
        p = Path(source)
        ext = p.suffix.lower()
        if ext == ".npy":
            a = np.load(p, mmap_mode="r")
        elif ext == ".parquet":
            import pyarrow.parquet as pq
            t = pq.read_table(p)
            col = t.column_names[0] if column is None else (column if isinstance(column, str) else t.column_names[column])
            return t.column(col).to_numpy(zero_copy_only=False).astype(np.float64)
        elif ext in (".csv", ".txt", ".dat"):
            delim = "," if ext == ".csv" else None
            a = np.loadtxt(p, delimiter=delim, dtype=np.float64, ndmin=1)
        elif ext == ".json":
            with open(p) as f:
                a = np.asarray(json.load(f), dtype=np.float64)
        else:
            raise ValueError(f"unsupported file type {ext!r}")
    else:
        a = np.asarray(source)
    if a.ndim == 2:
        if column is None:
            if a.shape[1] != 1:
                raise ValueError("2-D input: specify `column`")
            column = 0
        a = a[:, int(column)]
    elif a.ndim != 1:
        raise ValueError("time series must be 1-D")
    return a


# ---------------------------------------------------------------- result ---

@dataclass
class OrdinalStatistics:
    """Pattern and transition frequencies of one symbolised series.

    Nodes are the observed (admissible) patterns, sorted by pattern ID; edges
    are observed transitions between consecutive windows, as node indices.
    """
    m: int
    tau: int
    w: int
    ranking: str
    n_points: int
    n_windows: int
    n_valid: int
    n_nonfinite_points: int
    n_nonfinite_windows: int
    n_tied_windows: int
    n_tie_pairs: int
    id_hi: np.ndarray
    id_lo: np.ndarray
    pattern_counts: np.ndarray
    src: np.ndarray
    dst: np.ndarray
    weight: np.ndarray
    sequence: np.ndarray | None = field(default=None, repr=False)
    """Optional node index of every window (-1 = skipped window)."""

    # -- patterns ---------------------------------------------------------
    @property
    def pattern_ids(self) -> np.ndarray:
        """Fixed IDs (1..m!) of the observed patterns (uint64, or Python ints if m > 20)."""
        return _pat.join(self.id_hi, self.id_lo, self.m)

    @property
    def n_patterns(self) -> int:
        return len(self.pattern_counts)

    @property
    def n_transitions(self) -> int:
        """Number of distinct observed transitions (edges, incl. self-loops)."""
        return len(self.weight)

    @property
    def pattern_probabilities(self) -> np.ndarray:
        return self.pattern_counts / self.pattern_counts.sum()

    def permutations(self, ranking: str | None = None) -> np.ndarray:
        """Observed patterns as permutations (n_patterns x m, 1-based)."""
        ids = self.pattern_ids
        r = self.ranking if ranking is None else _pat.check_ranking(ranking)
        if r != self.ranking:
            ids = _pat.convert(ids, self.m, self.ranking, r)
        return _pat.decode(ids, self.m, r)

    def node_of(self, pattern_id) -> int:
        """Node index of a pattern ID (-1 if not observed)."""
        return int(self.nodes_of([pattern_id])[0])

    def nodes_of(self, pattern_ids) -> np.ndarray:
        """Node indices of pattern IDs (-1 where not observed)."""
        hi, lo = _pat.split(np.atleast_1d(pattern_ids))
        return _ids_to_nodes(hi, lo, _node_keys(self.id_hi, self.id_lo, self.m))

    @property
    def symbol_sequence(self) -> np.ndarray:
        """Pattern ID of every window (0 = skipped). Requires ``keep_sequence=True``."""
        if self.sequence is None:
            raise ValueError("symbol sequence not stored: fit with keep_sequence=True")
        seq = np.asarray(self.sequence)
        ok = seq >= 0
        if _pat.fits_uint64(self.m):
            out = np.zeros(len(seq), dtype=np.uint64)
            out[ok] = self.id_lo[seq[ok]]
            return out
        out = np.zeros(len(seq), dtype=object)
        ids = self.pattern_ids
        out[ok] = ids[seq[ok]]
        return out

    # -- derived quantities -------------------------------------------------
    def network(self):
        """The ordinal partition network (weighted, directed, with self-loops)."""
        from .network import OrdinalNetwork
        return OrdinalNetwork.from_statistics(self)

    def permutation_entropy(self, normalized: bool = False, base: float = 2.0) -> float:
        from .entropy import permutation_entropy
        return permutation_entropy(self, normalized=normalized, base=base)

    def conditional_entropy(self, base: float = 2.0) -> float:
        from .entropy import conditional_entropy
        return conditional_entropy(self, base=base)

    def forbidden_fraction(self) -> float:
        """Relative count of forbidden (unobserved) patterns, 1 - V/m!."""
        return 1.0 - self.n_patterns / _pat.factorial(self.m)

    def topological_permutation_entropy(self) -> float:
        """log2(V) / (m - 1) (Bandt, Keller & Pompe 2002)."""
        return math.log2(self.n_patterns) / (self.m - 1)

    @property
    def metadata(self) -> dict:
        return {k: getattr(self, k) for k in (
            "m", "tau", "w", "ranking", "n_points", "n_windows", "n_valid", "n_nonfinite_points",
            "n_nonfinite_windows", "n_tied_windows", "n_tie_pairs")}

    def summary(self) -> str:
        md = self.metadata
        return (f"OrdinalStatistics(m={md['m']}, tau={md['tau']}, w={md['w']}, ranking={md['ranking']!r}: "
                f"{self.n_patterns} patterns of {_pat.factorial(self.m)}, {self.n_transitions} transitions, "
                f"{md['n_valid']}/{md['n_windows']} windows valid, {md['n_tied_windows']} with ties, "
                f"{md['n_nonfinite_points']} non-finite points)")

    def __repr__(self) -> str:  # pragma: no cover
        return self.summary()

    # -- I/O ------------------------------------------------------------------
    def save(self, path) -> tuple[Path, Path]:
        """Write ``<path>.nodes.parquet`` and ``<path>.edges.parquet`` (+ ``<path>.sequence.npy``)."""
        from .io import save_statistics
        return save_statistics(self, path)

    @classmethod
    def load(cls, path) -> "OrdinalStatistics":
        from .io import load_statistics
        return load_statistics(path)


# ------------------------------------------------------------ symbolizer ---

class OrdinalSymbolizer:
    """Segment a series into ordinal patterns and count patterns & transitions.

    Parameters
    ----------
    m : pattern length (1 <= m <= 34)
    tau : embedding delay between points inside a window (default 1)
    w : slide lag between the starts of consecutive windows (default 1 =
        maximal overlap; w = m*tau = no overlap; any w >= 1 allowed)
    ranking : ``"chronological"`` (Bandt-Pompe, default) or ``"amplitude"`` (Small)
    threads : worker threads (0 = all cores)
    chunk_size : windows processed per chunk (bounds RAM use for long series)
    memory_budget : bytes for in-RAM hash tables; when exceeded and
        ``spill_dir`` is given, sorted runs are spilled to disk and merged.
    spill_dir : directory for temporary spill files (default: the system temp
        directory; ``False`` = never spill, keep everything in RAM)

    Ties inside a window: the earlier point counts as smaller (Bandt & Pompe).
    Windows containing NaN/Inf are skipped and no transition is recorded into
    or out of them.
    """

    def __init__(self, m: int, tau: int = 1, w: int = 1, ranking: str = "chronological", *,
                 threads: int = 0, chunk_size: int = DEFAULT_CHUNK, memory_budget: int = DEFAULT_MEMORY,
                 spill_dir: str | os.PathLike | None | bool = None):
        if not (1 <= m <= _pat.MAX_M):
            raise ValueError(f"m must be in 1..{_pat.MAX_M}")
        if tau < 1 or w < 1:
            raise ValueError("tau and w must be >= 1")
        self.m, self.tau, self.w = int(m), int(tau), int(w)
        self.ranking = _pat.check_ranking(ranking)
        self.threads = int(threads)
        self.chunk_size = int(chunk_size)
        self.memory_budget = int(memory_budget)
        if spill_dir is None or spill_dir is True:
            import tempfile
            spill_dir = tempfile.gettempdir()
        self.spill_dir = None if spill_dir is False else str(spill_dir)

    def __repr__(self):
        return f"OrdinalSymbolizer(m={self.m}, tau={self.tau}, w={self.w}, ranking={self.ranking!r})"

    @property
    def span(self) -> int:
        return (self.m - 1) * self.tau + 1

    def n_windows(self, n: int) -> int:
        return 0 if n < self.span else (n - self.span) // self.w + 1

    def encode(self, x, column=None) -> np.ndarray:
        """Pattern ID of every window of ``x`` (0 = skipped). Loads ``x`` into RAM."""
        a = np.ascontiguousarray(load_series(x, column), dtype=np.float64)
        hi, lo, _ = _core.encode(a, self.m, self.tau, self.w, self.ranking, self.threads)
        return _pat.join(hi, lo, self.m)

    def fit(self, x, column=None, *, keep_sequence: bool = False, sequence_path=None, output=None,
            progress: bool = False) -> OrdinalStatistics:
        """Symbolise ``x`` (array or file path) and count patterns and transitions.

        keep_sequence : also keep the node index of every window
            (``stats.sequence``; -1 = skipped). With ``sequence_path`` it is
            written to that ``.npy`` file (memory-mapped) instead of RAM.
        output : directory for out-of-core results. Pattern and transition
            tables are streamed to memory-mapped ``.npy`` files there instead of
            being assembled in RAM. Use it when the number of distinct transitions
            is very large (large m and N). Reload later with ``load_statistics(output)``.
        """
        outs = None if output is None else {self.m: Path(output)}
        return self._run(load_series(x, column), [self.m], keep_sequence, sequence_path, progress, outs)[self.m]

    def sweep(self, x, m: Iterable[int], column=None, *, output_dir=None,
              progress: bool = False) -> dict[int, OrdinalStatistics]:
        """Fit several pattern lengths in a single pass over the data.

        output_dir : if given, results for each m go out-of-core to ``output_dir/m{m}``.
        """
        ms = sorted(set(int(v) for v in m))
        outs = None if output_dir is None else {mm: Path(output_dir) / f"m{mm}" for mm in ms}
        return self._run(load_series(x, column), ms, False, None, progress, outs)

    # -- engine -------------------------------------------------------------
    def _run(self, x, ms: Sequence[int], keep_sequence, sequence_path, progress, outputs=None):
        import tempfile
        n = len(x)
        tau, w = self.tau, self.w
        spans = {mm: (mm - 1) * tau + 1 for mm in ms}
        nws = {mm: (0 if n < spans[mm] else (n - spans[mm]) // w + 1) for mm in ms}
        budget = max(self.memory_budget // max(len(ms), 1), 64 * 1024 ** 2)
        counters = {mm: _core.Counter(mm, tau, w, self.ranking, self.threads, budget, self.spill_dir) for mm in ms}
        maxspan = max(spans.values())
        nw_max = max(nws.values())
        want_ids = keep_sequence and len(ms) == 1
        tmp = None
        if self.spill_dir is not None:
            os.makedirs(self.spill_dir, exist_ok=True)
        if want_ids:
            m0 = ms[0]
            nw0 = nws[m0]
            if sequence_path is not None or (outputs is not None):
                tmp = tempfile.TemporaryDirectory(dir=self.spill_dir, ignore_cleanup_errors=True)
                raw_lo = np.lib.format.open_memmap(os.path.join(tmp.name, "lo.npy"), mode="w+", dtype=np.uint64, shape=(nw0,))
                raw_hi = (np.lib.format.open_memmap(os.path.join(tmp.name, "hi.npy"), mode="w+", dtype=np.uint64, shape=(nw0,))
                          if not _pat.fits_uint64(m0) else None)
            else:
                raw_lo = np.empty(nw0, dtype=np.uint64)
                raw_hi = None if _pat.fits_uint64(m0) else np.empty(nw0, dtype=np.uint64)
        nonfinite = 0
        counted_upto = 0
        cs = max(self.chunk_size, 1)
        nchunks = max(1, math.ceil(nw_max / cs))
        for ci, k0 in enumerate(range(0, max(nw_max, 1), cs)):
            k1 = min(k0 + cs, nw_max)
            a = k0 * w
            b = min(n, (k1 - 1) * w + maxspan) if k1 > k0 else a
            seg = np.ascontiguousarray(x[a:b], dtype=np.float64)
            # non-finite points, counted once (non-overlapping ranges)
            hi_pt = b if k1 >= nw_max else min(b, k1 * w)
            if hi_pt > counted_upto:
                part = seg[counted_upto - a: hi_pt - a]
                nonfinite += int(part.size - np.count_nonzero(np.isfinite(part)))
                counted_upto = hi_pt
            for mm in ms:
                kk1 = min(k1, nws[mm])
                if kk1 <= k0:
                    continue
                r = counters[mm].feed(seg, kk1 - k0, want_ids)
                if want_ids:
                    raw_lo[k0:kk1] = r[1]
                    if raw_hi is not None:
                        raw_hi[k0:kk1] = r[0]
            if progress:
                print(f"\rchunk {ci + 1}/{nchunks}", end="", flush=True)
        if counted_upto < n:
            tail = np.asarray(x[counted_upto:n], dtype=np.float64)
            nonfinite += int(tail.size - np.count_nonzero(np.isfinite(tail)))
        if progress:
            print()
        out = {}
        for mm in ms:
            common = dict(m=mm, tau=tau, w=w, ranking=self.ranking, n_points=n, n_nonfinite_points=nonfinite)
            if outputs is not None:
                out[mm] = _finalize_to_disk(counters[mm], outputs[mm], common, self.chunk_size)
                continue
            d = counters[mm].finalize()
            V = len(d["count"])
            idx_t = np.uint32 if V < 2 ** 32 else np.uint64
            out[mm] = OrdinalStatistics(
                **common, n_windows=int(d["n_windows"]), n_valid=int(d["n_valid"]),
                n_nonfinite_windows=int(d["n_nonfinite_windows"]), n_tied_windows=int(d["n_tied_windows"]),
                n_tie_pairs=int(d["n_tie_pairs"]), id_hi=d["id_hi"], id_lo=d["id_lo"],
                pattern_counts=d["count"], src=d["src"].astype(idx_t), dst=d["dst"].astype(idx_t),
                weight=d["weight"])
        if want_ids:
            st = out[ms[0]]
            nw0 = len(raw_lo)
            if sequence_path is not None:
                seq = np.lib.format.open_memmap(str(sequence_path), mode="w+", dtype=np.int64, shape=(nw0,))
            elif outputs is not None:
                seq = np.lib.format.open_memmap(str(outputs[ms[0]] / "sequence.npy"), mode="w+", dtype=np.int64, shape=(nw0,))
            else:
                seq = np.empty(nw0, dtype=np.int64)
            keys = _node_keys(st.id_hi, st.id_lo, st.m)
            step = max(self.chunk_size, 1)
            for p0 in range(0, nw0, step):
                p1 = min(p0 + step, nw0)
                h = np.zeros(p1 - p0, np.uint64) if raw_hi is None else np.array(raw_hi[p0:p1])
                seq[p0:p1] = _ids_to_nodes(h, np.array(raw_lo[p0:p1]), keys)
            if isinstance(seq, np.memmap):
                seq.flush()
            st.sequence = seq
            _close(raw_lo, raw_hi)
            del raw_lo, raw_hi
            if tmp is not None:
                import gc
                gc.collect()
                tmp.cleanup()
        return out


def _close(*arrays):
    """Release memory maps now (Windows cannot delete files that are still mapped)."""
    for a in arrays:
        mm = getattr(a, "_mmap", None)
        if mm is not None:
            try:
                mm.close()
            except (BufferError, ValueError):
                pass


def _node_keys(node_hi, node_lo, m):
    """Search keys for node lookup: the lo array (m <= 20) or a structured (hi, lo) array."""
    if _pat.fits_uint64(m):
        return node_lo
    keys = np.empty(len(node_lo), dtype=[("h", "<u8"), ("l", "<u8")])
    keys["h"], keys["l"] = node_hi, node_lo
    return keys


def _ids_to_nodes(h, l, keys) -> np.ndarray:
    """Node index (position in the sorted ID list) of each ID; -1 for ID 0 or unknown IDs."""
    l = np.asarray(l, dtype=np.uint64)
    h = np.asarray(h, dtype=np.uint64)
    if len(keys) == 0:
        return -np.ones(len(l), dtype=np.int64)
    if keys.dtype.names is None:
        idx = np.searchsorted(keys, l)
        idx = np.minimum(idx, len(keys) - 1)
        out = idx.astype(np.int64)
        out[(l == 0) | (h != 0) | (keys[idx] != l)] = -1
        return out
    q = np.empty(len(h), dtype=keys.dtype)
    q["h"], q["l"] = h, l
    idx = np.searchsorted(keys, q)
    idx = np.minimum(idx, len(keys) - 1)
    out = idx.astype(np.int64)
    found = keys[idx]
    bad = ((h == 0) & (l == 0)) | (found["h"] != h) | (found["l"] != l)
    out[bad] = -1
    return out


def _finalize_to_disk(counter, out_dir: Path, common: dict, chunk: int) -> OrdinalStatistics:
    """Stream a counter's results to ``out_dir`` as .npy files (constant memory)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    d = counter.finalize_to_disk(str(out_dir))
    V, E = int(d["n_patterns"]), int(d["n_edges"])
    opened = []

    def raw(name, k):
        if not k:
            return np.zeros(0, "<u8")
        mm = np.memmap(out_dir / f"{name}.bin", dtype="<u8", mode="r", shape=(k,))
        opened.append(mm)
        return mm
    m = common["m"]
    idx_t = np.uint32 if V < 2 ** 32 else np.uint64

    def to_npy(name, src, dtype, k):
        dst = np.lib.format.open_memmap(out_dir / f"{name}.npy", mode="w+", dtype=dtype, shape=(k,))
        for a in range(0, k, chunk):
            dst[a:a + chunk] = src[a:a + chunk]
        dst.flush()
        _close(dst)
        del dst

    to_npy("id_lo", raw("nodes_lo", V), np.uint64, V)
    if not _pat.fits_uint64(m):
        to_npy("id_hi", raw("nodes_hi", V), np.uint64, V)
        keys = np.lib.format.open_memmap(out_dir / "keys.npy", mode="w+", dtype=[("h", "<u8"), ("l", "<u8")], shape=(V,))
        hi_r, lo_r = raw("nodes_hi", V), raw("nodes_lo", V)
        for a in range(0, V, chunk):
            keys["h"][a:a + chunk] = hi_r[a:a + chunk]
            keys["l"][a:a + chunk] = lo_r[a:a + chunk]
        keys.flush()
    else:
        keys = np.load(out_dir / "id_lo.npy", mmap_mode="r")
    opened.append(keys)
    to_npy("pattern_counts", raw("counts", V), np.uint64, V)
    to_npy("src", raw("edge_src", E), idx_t, E)
    to_npy("weight", raw("edge_weight", E), np.uint64, E)
    dst = np.lib.format.open_memmap(out_dir / "dst.npy", mode="w+", dtype=idx_t, shape=(E,))
    dh, dl = raw("edge_dst_hi", E), raw("edge_dst_lo", E)
    for a in range(0, E, chunk):
        dst[a:a + chunk] = _ids_to_nodes(np.array(dh[a:a + chunk]), np.array(dl[a:a + chunk]), keys)
    dst.flush()
    _close(dst, *opened)
    del dst, dh, dl, keys, opened
    import gc
    gc.collect()
    for f in ("nodes_hi", "nodes_lo", "counts", "edge_src", "edge_dst_hi", "edge_dst_lo", "edge_weight"):
        try:
            (out_dir / f"{f}.bin").unlink()
        except OSError:
            pass
    (out_dir / "keys.npy").unlink(missing_ok=True)
    meta = dict(common, n_windows=int(d["n_windows"]), n_valid=int(d["n_valid"]),
                n_nonfinite_windows=int(d["n_nonfinite_windows"]), n_tied_windows=int(d["n_tied_windows"]),
                n_tie_pairs=int(d["n_tie_pairs"]), format=1, kind="statistics")
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=1)
    from .io import load_statistics
    return load_statistics(out_dir)


def symbolize(x, m: int, tau: int = 1, w: int = 1, ranking: str = "chronological", **kw) -> OrdinalStatistics:
    """Shortcut for ``OrdinalSymbolizer(m, tau, w, ranking).fit(x, **kw)``."""
    fit_kw = {k: kw.pop(k) for k in ("column", "keep_sequence", "sequence_path", "output", "progress") if k in kw}
    return OrdinalSymbolizer(m, tau, w, ranking, **kw).fit(x, **fit_kw)

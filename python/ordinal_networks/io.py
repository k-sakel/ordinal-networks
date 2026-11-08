"""Compact on-disk storage of ordinal statistics and networks.

Two formats:

* **Parquet** (``stats.save(stem)``): compressed, portable; described below.
* **Out-of-core directory** (``fit(..., output=dir)``): ``meta.json`` plus
  memory-mapped ``.npy`` arrays (``id_lo``, ``id_hi`` for m > 20,
  ``pattern_counts``, ``src``, ``dst``, ``weight``, optional ``sequence``),
  for results too large for RAM. ``load_statistics(dir)`` memory-maps them.

Parquet layout:

``stats.save("LOG_ChaosR4_m25")`` writes

* ``LOG_ChaosR4_m25.nodes.parquet``  pattern ID(s) and frequency, sorted by ID
  (``id`` uint64 for m <= 20, ``id_hi``/``id_lo`` for m > 20)
* ``LOG_ChaosR4_m25.edges.parquet``  ``src``, ``dst`` (node indices), ``weight``
* ``LOG_ChaosR4_m25.sequence.npy``   node index per window, only if kept

All parameters and counters are stored as JSON in the Parquet schema metadata.
Every matrix (W, A, T, Q) is rebuilt from the edge list on load.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

FORMAT_VERSION = 1


def _stem(path) -> Path:
    p = Path(path)
    for suf in (".nodes.parquet", ".edges.parquet", ".sequence.npy"):
        if p.name.endswith(suf):
            return p.with_name(p.name[: -len(suf)])
    return p


def _smallest_uint(a: np.ndarray):
    mx = int(a.max()) if a.size else 0
    for t in (np.uint8, np.uint16, np.uint32, np.uint64):
        if mx <= np.iinfo(t).max:
            return a.astype(t)
    return a


ROW_GROUP = 8_000_000


def _write(table: pa.Table, path: Path, meta: dict):
    table = table.replace_schema_metadata({b"ordinal_networks": json.dumps(meta).encode()})
    pq.write_table(table, path, compression="zstd", row_group_size=ROW_GROUP)


def _write_columns(cols: dict, path: Path, meta: dict):
    """Write (possibly memory-mapped) columns in row groups without loading them whole."""
    n = len(next(iter(cols.values())))
    first = {k: np.asarray(v[:min(n, 1)]) for k, v in cols.items()}
    schema = pa.table(first).schema.with_metadata({b"ordinal_networks": json.dumps(meta).encode()})
    with pq.ParquetWriter(path, schema, compression="zstd") as wr:
        for a in range(0, max(n, 1), ROW_GROUP):
            batch = {k: np.ascontiguousarray(v[a:a + ROW_GROUP]) for k, v in cols.items()}
            wr.write_table(pa.table(batch, schema=schema))


def _cast_uint(a: np.ndarray, mx: int):
    for t in (np.uint8, np.uint16, np.uint32, np.uint64):
        if mx <= np.iinfo(t).max:
            return t
    return np.uint64


class _Cast:
    """Lazy dtype cast of a (memory-mapped) array, sliced chunk by chunk."""
    def __init__(self, a, dtype):
        self.a, self.dtype = a, dtype

    def __len__(self):
        return len(self.a)

    def __getitem__(self, sl):
        return np.asarray(self.a[sl]).astype(self.dtype)


def _chunked_max(a, chunk=ROW_GROUP) -> int:
    mx = 0
    for i in range(0, len(a), chunk):
        mx = max(mx, int(np.max(a[i:i + chunk])))
    return mx


def save_statistics(stats, path):
    stem = _stem(path)
    stem.parent.mkdir(parents=True, exist_ok=True)
    meta = dict(stats.metadata, format=FORMAT_VERSION, kind="statistics")
    c = _Cast(stats.pattern_counts, _cast_uint(None, _chunked_max(stats.pattern_counts) if len(stats.pattern_counts) else 0))
    if stats.m <= 20:
        nodes = {"id": stats.id_lo, "count": c}
    else:
        nodes = {"id_hi": stats.id_hi, "id_lo": stats.id_lo, "count": c}
    V = len(stats.pattern_counts)
    it = _cast_uint(None, max(V - 1, 0))
    edges = {"src": _Cast(stats.src, it), "dst": _Cast(stats.dst, it),
             "weight": _Cast(stats.weight, _cast_uint(None, _chunked_max(stats.weight) if len(stats.weight) else 0))}
    pn, pe = stem.with_name(stem.name + ".nodes.parquet"), stem.with_name(stem.name + ".edges.parquet")
    _write_columns(nodes, pn, meta)
    _write_columns(edges, pe, meta)
    if stats.sequence is not None:
        seq = stats.sequence
        out = np.lib.format.open_memmap(stem.with_name(stem.name + ".sequence.npy"), mode="w+", dtype=np.int64, shape=(len(seq),))
        for a in range(0, len(seq), ROW_GROUP):
            out[a:a + ROW_GROUP] = seq[a:a + ROW_GROUP]
        out.flush()
    return pn, pe


def _load_dir(d: Path):
    from .symbolize import OrdinalStatistics
    with open(d / "meta.json") as f:
        meta = json.load(f)
    ld = lambda name: np.load(d / f"{name}.npy", mmap_mode="r")
    lo = ld("id_lo")
    hi = ld("id_hi") if (d / "id_hi.npy").exists() else np.zeros(len(lo), np.uint64)
    seq = ld("sequence") if (d / "sequence.npy").exists() else None
    kw = {k: meta[k] for k in ("m", "tau", "w", "ranking", "n_points", "n_windows", "n_valid", "n_nonfinite_points",
                               "n_nonfinite_windows", "n_tied_windows", "n_tie_pairs")}
    return OrdinalStatistics(**kw, id_hi=hi, id_lo=lo, pattern_counts=ld("pattern_counts"), src=ld("src"),
                             dst=ld("dst"), weight=ld("weight"), sequence=seq)


def load_statistics(path, mmap_sequence: bool = True):
    """Load statistics saved as Parquet (``stem``) or an out-of-core directory."""
    from .symbolize import OrdinalStatistics
    if Path(path).is_dir() and (Path(path) / "meta.json").exists():
        return _load_dir(Path(path))
    stem = _stem(path)
    tn = pq.read_table(stem.with_name(stem.name + ".nodes.parquet"))
    te = pq.read_table(stem.with_name(stem.name + ".edges.parquet"))
    meta = json.loads(tn.schema.metadata[b"ordinal_networks"])
    if "id" in tn.column_names:
        lo = tn.column("id").to_numpy().astype(np.uint64)
        hi = np.zeros_like(lo)
    else:
        hi = tn.column("id_hi").to_numpy().astype(np.uint64)
        lo = tn.column("id_lo").to_numpy().astype(np.uint64)
    seq_path = stem.with_name(stem.name + ".sequence.npy")
    seq = np.load(seq_path, mmap_mode="r" if mmap_sequence else None) if seq_path.exists() else None
    kw = {k: meta[k] for k in ("m", "tau", "w", "ranking", "n_points", "n_windows", "n_valid", "n_nonfinite_points",
                               "n_nonfinite_windows", "n_tied_windows", "n_tie_pairs")}
    return OrdinalStatistics(**kw, id_hi=hi, id_lo=lo, pattern_counts=tn.column("count").to_numpy().astype(np.uint64),
                             src=te.column("src").to_numpy(), dst=te.column("dst").to_numpy(),
                             weight=te.column("weight").to_numpy().astype(np.uint64), sequence=seq)


def save_network(net, path):
    """Save an OrdinalNetwork (e.g. after pruning) as ``<path>.edges.parquet`` (+ nodes)."""
    stem = _stem(path)
    stem.parent.mkdir(parents=True, exist_ok=True)
    meta = {"format": FORMAT_VERSION, "kind": "network", "m": net.m, "ranking": net.ranking, "n_nodes": net.n_nodes}
    w = np.asarray(net.weight)
    wcol = _smallest_uint(w.astype(np.uint64)) if np.all(w == np.round(w)) else w
    edges = pa.table({"src": _smallest_uint(net.src), "dst": _smallest_uint(net.dst), "weight": wcol})
    pe = stem.with_name(stem.name + ".edges.parquet")
    _write(edges, pe, meta)
    if net.id_lo is not None:
        cols = {"id": net.id_lo.astype(np.uint64)} if (net.m or 0) <= 20 else {"id_hi": net.id_hi, "id_lo": net.id_lo}
        if net.node_counts is not None:
            cols["count"] = _smallest_uint(np.asarray(net.node_counts))
        _write(pa.table(cols), stem.with_name(stem.name + ".nodes.parquet"), meta)
    return pe


def load_network(path):
    from .network import OrdinalNetwork
    stem = _stem(path)
    pe = stem.with_name(stem.name + ".edges.parquet")
    te = pq.read_table(pe)
    meta = json.loads(te.schema.metadata[b"ordinal_networks"])
    if meta.get("kind") == "statistics":
        from .symbolize import OrdinalStatistics  # noqa: F401
        return load_statistics(stem).network()
    net = OrdinalNetwork.from_edges(te.column("src").to_numpy(), te.column("dst").to_numpy(),
                                    te.column("weight").to_numpy().astype(np.float64), meta["n_nodes"])
    pn = stem.with_name(stem.name + ".nodes.parquet")
    if pn.exists():
        tn = pq.read_table(pn)
        if "id" in tn.column_names:
            net.id_lo = tn.column("id").to_numpy().astype(np.uint64)
            net.id_hi = np.zeros_like(net.id_lo)
        else:
            net.id_hi = tn.column("id_hi").to_numpy().astype(np.uint64)
            net.id_lo = tn.column("id_lo").to_numpy().astype(np.uint64)
        if "count" in tn.column_names:
            net.node_counts = tn.column("count").to_numpy()
    net.m, net.ranking = meta.get("m"), meta.get("ranking", "chronological")
    return net

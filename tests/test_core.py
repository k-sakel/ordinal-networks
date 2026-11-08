"""Correctness tests against straightforward reference implementations
(the MATLAB algorithm: stable sort of each window + unique rows)."""
import itertools
import math

import numpy as np
import pytest

import ordinal_networks as on
from ordinal_networks import patterns as P


def ref_patterns(x, m, tau=1, w=1, ranking="chronological"):
    """Per-window permutation (MATLAB sort, stable) or None for non-finite windows."""
    span = (m - 1) * tau + 1
    nw = 0 if len(x) < span else (len(x) - span) // w + 1
    out = []
    for k in range(nw):
        v = np.array([x[k * w + j * tau] for j in range(m)])
        if not np.all(np.isfinite(v)):
            out.append(None)
            continue
        s = np.argsort(v, kind="stable") + 1
        if ranking == "chronological":
            out.append(tuple(s))
        else:
            r = np.empty(m, int)
            r[s - 1] = np.arange(1, m + 1)          # ascending ranks
            out.append(tuple(m + 1 - r))            # descending = amplitude
    return out


def lex_rank(p):
    m = len(p)
    rank = 0
    rest = sorted(p)
    for k, v in enumerate(p):
        i = rest.index(v)
        rank += i * math.factorial(m - 1 - k)
        rest.pop(i)
    return rank + 1


def ref_stats(x, m, tau=1, w=1, ranking="chronological"):
    pats = ref_patterns(x, m, tau, w, ranking)
    ids = [None if p is None else lex_rank(p) for p in pats]
    nodes = sorted({i for i in ids if i is not None})
    counts = {i: 0 for i in nodes}
    trans = {}
    for i in ids:
        if i is not None:
            counts[i] += 1
    for a, b in zip(ids[:-1], ids[1:]):
        if a is not None and b is not None:
            trans[(a, b)] = trans.get((a, b), 0) + 1
    return nodes, counts, trans


def as_dicts(st):
    ids = [int(v) for v in st.pattern_ids]
    counts = dict(zip(ids, st.pattern_counts.tolist()))
    trans = {(ids[s], ids[d]): int(c) for s, d, c in zip(st.src, st.dst, st.weight)}
    return ids, counts, trans


@pytest.fixture(scope="module")
def noisy():
    rng = np.random.default_rng(1)
    x = np.round(rng.standard_normal(2500), 1)    # many ties
    x[[50, 51, 900]] = [np.nan, np.inf, -np.inf]
    return x


@pytest.mark.parametrize("m", [1, 2, 3, 4, 6])
@pytest.mark.parametrize("tau", [1, 2, 3])
@pytest.mark.parametrize("w", [1, 2, 5])
@pytest.mark.parametrize("ranking", ["chronological", "amplitude"])
def test_against_reference(noisy, m, tau, w, ranking):
    st = on.OrdinalSymbolizer(m, tau, w, ranking).fit(noisy)
    assert as_dicts(st) == ref_stats(noisy, m, tau, w, ranking)
    assert st.n_windows == len(ref_patterns(noisy, m, tau, w))
    assert st.n_nonfinite_points == 3


def test_ties_counted():
    x = np.array([1.0, 1.0, 2.0, 0.0, 0.0, 5.0])
    st = on.symbolize(x, 3)
    # windows: (1,1,2) tie, (1,2,0) none, (2,0,0) tie, (0,0,5) tie
    assert st.n_tied_windows == 3 and st.n_tie_pairs == 3
    # earlier point is smaller: (1,1,2) -> sort -> (1,2,3)
    assert tuple(on.decode(on.pattern_of([1.0, 1.0, 2.0]), 3)) == (1, 2, 3)


def test_nonfinite_breaks_transitions():
    x = np.array([1.0, 2.0, 3.0, np.nan, 3.0, 2.0, 1.0])
    st = on.symbolize(x, 2)
    ids, counts, trans = as_dicts(st)
    # windows: (1,2)(2,3)[nan][nan](3,2)(2,1): transitions only within each run
    assert sum(trans.values()) == 2
    assert st.n_nonfinite_windows == 2 and st.n_valid == 4


@pytest.mark.parametrize("m", [3, 7])
def test_chunking_and_sweep_consistent(m):
    x = on.systems.generate("henon", n=50_000)[:, 0]
    a = on.OrdinalSymbolizer(m, chunk_size=10_000_000).fit(x)
    b = on.OrdinalSymbolizer(m, chunk_size=777).fit(x)
    c = on.OrdinalSymbolizer(m).sweep(x, [2, m, 9])[m]
    for s in (b, c):
        assert as_dicts(s) == as_dicts(a)


def test_large_m_u128_and_spill(tmp_path):
    rng = np.random.default_rng(0)
    x = rng.standard_normal(20_000)
    m = 25
    st = on.OrdinalSymbolizer(m).fit(x, keep_sequence=True)
    ref = ref_stats(x, m)
    assert as_dicts(st) == ref
    assert max(ref[0]) > 2 ** 64          # genuinely needs 128 bits
    # spilling to disk gives identical results
    sp = on.OrdinalSymbolizer(m, chunk_size=1000, memory_budget=1, spill_dir=tmp_path).fit(x)
    assert as_dicts(sp) == ref
    # symbol sequence
    seq = st.symbol_sequence
    assert [int(v) for v in seq[:50]] == [lex_rank(p) for p in ref_patterns(x, m)[:50]]


def test_decode_encode_roundtrip():
    for m in (1, 2, 3, 4, 5):
        perms = np.array(list(itertools.permutations(range(1, m + 1))))
        ids = P.encode(perms)
        assert list(map(int, ids)) == list(range(1, math.factorial(m) + 1))
        assert (P.decode(ids, m) == perms).all()
    m = 34
    p = np.arange(m, 0, -1)
    assert int(P.encode(p)) == math.factorial(34)
    assert (P.decode(math.factorial(34), 34) == p).all()


def test_convert_and_ranks():
    x = np.array([0.3, 0.9, 0.1, 0.5])
    idc = P.pattern_of(x, "chronological")
    ida = P.pattern_of(x, "amplitude")
    assert int(P.convert(idc, 4, "chronological", "amplitude")) == ida
    assert int(P.convert(ida, 4, "amplitude", "chronological")) == idc
    assert tuple(P.ascending_ranks(idc, 4)[0]) == (2, 4, 1, 3)
    # paper example: x10 < x8 < x7 < x9 -> chronological (4,2,1,3), amplitude (2,3,1,4)
    w = [3.0, 2.0, 4.0, 1.0]
    assert tuple(P.decode(P.pattern_of(w), 4)) == (4, 2, 1, 3)
    assert tuple(P.decode(P.pattern_of(w, "amplitude"), 4, "amplitude")) == (2, 3, 1, 4)


def test_prefix_and_outgrowth():
    x = on.systems.generate("logistic", n=200_000)
    sw = on.OrdinalSymbolizer(2).sweep(x, [5, 6])
    g = on.entropy.outgrowth_ratio(sw[5], sw[6])
    # brute force: drop the value m+1 from each (m+1)-pattern (MATLAB OutgrowthRatio)
    p6 = sw[6].permutations()
    pref = [tuple(v for v in row if v != 6) for row in p6]
    p5 = [tuple(r) for r in sw[5].permutations()]
    brute = np.array([pref.count(q) for q in p5])
    assert (g == brute).all()


def test_save_load(tmp_path):
    x = on.systems.generate("logistic", n=10_000)
    for m in (5, 22):
        st = on.OrdinalSymbolizer(m).fit(x, keep_sequence=True)
        st.save(tmp_path / f"s{m}")
        st2 = on.load_statistics(tmp_path / f"s{m}")
        assert as_dicts(st2) == as_dicts(st)
        assert st2.metadata == st.metadata
        assert (np.asarray(st2.sequence) == st.sequence).all()
        net = st.network().prune(min_count=2)
        net.save(tmp_path / f"n{m}")
        net2 = on.load_network(tmp_path / f"n{m}")
        assert (net2.W != net.W).nnz == 0


@pytest.mark.parametrize("m", [6, 23])
def test_out_of_core_identical(tmp_path, m):
    x = np.random.default_rng(3).standard_normal(30_000)
    x[[10, 500]] = np.nan
    ref = on.OrdinalSymbolizer(m).fit(x, keep_sequence=True)
    sym = on.OrdinalSymbolizer(m, chunk_size=2000, memory_budget=1, spill_dir=tmp_path / "spill")
    st = sym.fit(x, keep_sequence=True, output=tmp_path / "out")
    assert as_dicts(st) == as_dicts(ref)
    assert st.metadata == ref.metadata
    assert (np.asarray(st.sequence) == ref.sequence).all()
    st2 = on.load_statistics(tmp_path / "out")
    assert as_dicts(st2) == as_dicts(ref)
    st2.save(tmp_path / "pq")                      # disk-backed -> Parquet
    assert as_dicts(on.load_statistics(tmp_path / "pq")) == as_dicts(ref)
    assert st2.network().n_edges == ref.n_transitions
    sw = on.OrdinalSymbolizer(2, chunk_size=3000).sweep(x, [4, m], output_dir=tmp_path / "sw")
    assert as_dicts(sw[m]) == as_dicts(ref)
    assert not list((tmp_path / "spill").glob("*.run"))   # spill files cleaned up

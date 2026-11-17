"""Input validation, silent-failure guards and centrality conventions."""
import math
import sys

import numpy as np
import pytest

import ordinal_networks as on


@pytest.fixture(scope="module")
def x():
    return np.random.default_rng(0).standard_normal(5000)


# ---------------------------------------------------------------- short / empty / NaN input --

@pytest.mark.parametrize("series", [[], [1.0, 2.0], np.zeros(5)])
def test_series_shorter_than_window_raises(series):
    with pytest.raises(ValueError, match="at least"):
        on.OrdinalSymbolizer(10).fit(series)
    with pytest.raises(ValueError, match="at least"):
        on.OrdinalSymbolizer(10).encode(series)


def test_tau_counts_towards_minimum_length():
    on.OrdinalSymbolizer(3, tau=2).fit(np.arange(5.0))            # span 5: exactly one window
    with pytest.raises(ValueError, match="at least 5"):
        on.OrdinalSymbolizer(3, tau=2).fit(np.arange(4.0))


def test_sweep_raises_if_any_m_is_too_long(x):
    with pytest.raises(ValueError, match="m=40|m must be"):
        on.OrdinalSymbolizer(2).sweep(x[:30], [3, 40])


def test_all_nan_warns_and_entropies_refuse():
    with pytest.warns(RuntimeWarning, match="no valid windows"):
        st = on.OrdinalSymbolizer(3).fit(np.full(100, np.nan))
    assert st.n_patterns == 0 and st.n_nonfinite_points == 100
    for f in (st.permutation_entropy, st.conditional_entropy, st.topological_permutation_entropy):
        with pytest.raises(ValueError, match="no valid windows"):
            f()


def test_single_window_conditional_entropy_is_zero():
    st = on.OrdinalSymbolizer(3).fit([1.0, 2.0, 3.0])
    assert st.n_patterns == 1 and st.conditional_entropy() == 0.0


def test_topological_permutation_entropy_needs_m2(x):
    with pytest.raises(ValueError, match="m >= 2"):
        on.OrdinalSymbolizer(1).fit(x).topological_permutation_entropy()


# ------------------------------------------------------------------------ input types --

def test_complex_input_rejected(x):
    with pytest.raises(TypeError, match="complex"):
        on.OrdinalSymbolizer(4).fit(x + 1j * x)


def test_dataframe_columns(x):
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame({"a": x, "b": x[::-1].copy()})
    ref = on.OrdinalSymbolizer(4).fit(x[::-1].copy())
    for col in ("b", 1):
        st = on.OrdinalSymbolizer(4).fit(df, column=col)
        assert (st.pattern_counts == ref.pattern_counts).all()
    with pytest.raises(ValueError, match=r"specify `column`.*\['a', 'b'\]"):
        on.OrdinalSymbolizer(4).fit(df)
    with pytest.raises(ValueError, match="not found"):
        on.OrdinalSymbolizer(4).fit(df, column="zzz")
    assert on.OrdinalSymbolizer(4).fit(df[["a"]]).n_windows == len(x) - 3


def test_polars_dataframe_columns(x):
    pl = pytest.importorskip("polars")
    df = pl.DataFrame({"a": x, "b": x[::-1].copy()})
    st = on.OrdinalSymbolizer(4).fit(df, column="b")
    assert (st.pattern_counts == on.OrdinalSymbolizer(4).fit(x[::-1].copy()).pattern_counts).all()


def test_string_column_on_ndarray_gives_clear_error(x):
    with pytest.raises(ValueError, match="integer index"):
        on.OrdinalSymbolizer(4).fit(np.c_[x, x], column="b")


# ------------------------------------------------------------------------ network layer --

def _path_network():
    # 0 -> 1 -> 2 -> 0, 2 -> 3, 3 -> 4  (not strongly connected)
    return on.OrdinalNetwork.from_edges([0, 1, 2, 2, 3], [1, 2, 0, 3, 4])


def test_harmonic_centrality_is_the_standard_definition():
    nx = pytest.importorskip("networkx")
    net = _path_network()
    G = net.to_networkx()
    ref = nx.harmonic_centrality(G)
    expect = np.array([ref[i] / (net.n_nodes - 1) for i in range(net.n_nodes)])
    assert np.allclose(net.centrality("harmonic"), expect)
    # downstream nodes are more central by incoming paths: higher value = more central
    assert net.centrality("harmonic")[4] > net.centrality("harmonic")[0]


def test_characteristic_path_length_ignores_unreachable_pairs():
    nx = pytest.importorskip("networkx")
    net = _path_network()
    G = net.to_networkx()
    ref = np.mean([d for s, dd in nx.all_pairs_shortest_path_length(G) for t, d in dd.items() if s != t])
    assert net.characteristic_path_length() == pytest.approx(ref)
    # unchanged for a strongly connected network (3-cycle: distances 1, 2 -> mean 1.5)
    cyc = on.OrdinalNetwork.from_edges([0, 1, 2], [1, 2, 0])
    assert cyc.characteristic_path_length() == pytest.approx(1.5)


def test_missing_igraph_gives_install_hint(monkeypatch):
    monkeypatch.setitem(sys.modules, "igraph", None)
    with pytest.raises(ImportError, match=r"ordinal-networks\[graph\]"):
        _path_network().centrality("betweenness")
    monkeypatch.setitem(sys.modules, "networkx", None)
    with pytest.raises(ImportError, match=r"ordinal-networks\[graph\]"):
        _path_network().to_networkx()


# ------------------------------------------------------------------------------ systems --

@pytest.mark.parametrize("name", ["bakers", "piecewise_linear"])
def test_constant_default_series_warn_explicitly(name):
    with pytest.warns(RuntimeWarning, match="constant"):
        a = on.systems.generate(name, n=500, squeeze=False)
    assert np.all(a == a[0])
    assert on.systems.get_system(name).note       # the limitation is documented on the system


# ------------------------------------------------------------------ regression tests --

def test_nonfinite_points_counted_once_when_windows_do_not_touch():
    # w > span leaves gaps between windows; the count must not depend on the chunking
    x = np.random.default_rng(0).standard_normal(1000)
    x[::7] = np.nan
    expect = int(np.isnan(x).sum())
    for w, cs in [(10, 7), (10, 1), (4, 3), (1, 5), (3, 10**6)]:
        assert on.OrdinalSymbolizer(3, w=w, chunk_size=cs).fit(x).n_nonfinite_points == expect


def test_row_entropies_with_sink_node_in_last_row():
    net = on.OrdinalNetwork.from_edges([0, 1, 0], [1, 2, 2])       # node 2 has no out-links
    assert np.allclose(net.node_link_entropy(), [1.0, 0.0, 0.0])
    # row 0 of T^2 is (0, 0, 1/2): half of the mass has been absorbed by the sink
    assert np.allclose(net.hierarchical_degree(2), [[2.0, math.sqrt(2)], [1.0, 1.0], [1.0, 1.0]])
    assert net.spectral_radius(log=True) == -math.inf               # no cycles: rho(A) = 0


def test_convert_accepts_ranking_aliases():
    ids = np.arange(1, 25, dtype=np.uint64)
    ref = on.patterns.convert(ids, 4, "chronological", "amplitude")
    assert (on.patterns.convert(ids, 4, "bp", "small") == ref).all()
    assert (on.patterns.convert(ref, 4, "small", "bp") == ids).all()


def test_sequence_path_implies_keep_sequence(tmp_path, x):
    st = on.OrdinalSymbolizer(3).fit(x, sequence_path=tmp_path / "seq.npy")
    ref = on.OrdinalSymbolizer(3).fit(x, keep_sequence=True)
    assert (np.asarray(st.sequence) == ref.sequence).all()
    assert (np.load(tmp_path / "seq.npy") == ref.sequence).all()
    del st


def test_empty_sweep_and_out_of_core_edge_cases(tmp_path, x):
    with pytest.raises(ValueError, match="no pattern lengths"):
        on.OrdinalSymbolizer(3).sweep(x, [])
    with pytest.warns(RuntimeWarning, match="no valid windows"):
        st = on.OrdinalSymbolizer(3).fit(np.full(50, np.nan), output=tmp_path / "nan")
    assert st.n_patterns == 0
    st = on.OrdinalSymbolizer(3, chunk_size=0).fit(x, output=tmp_path / "c0")
    assert st.n_patterns == 6
    del st


def test_save_network_does_not_leave_stale_node_ids(tmp_path, x):
    stem = tmp_path / "net"
    on.OrdinalSymbolizer(3).fit(x).network().save(stem)
    on.OrdinalNetwork.from_edges([0, 1], [1, 0]).save(stem)
    net = on.load_network(stem)
    assert net.n_nodes == 2 and net.id_lo is None


def test_poincare_gives_up_instead_of_hanging():
    with pytest.raises(ValueError, match="crossings"):
        on.systems.poincare("lorenz", n=5, normal=(0.0, 0.0, 1.0), offset=1e6, transient=0, max_steps=2000)


@pytest.mark.parametrize("kw", [dict(dt=0.0), dict(dt=-0.01), dict(dt=float("nan")), dict(every=0)])
def test_bad_step_arguments_rejected(kw):
    for name in ("lorenz", "mackey_glass"):
        with pytest.raises(ValueError, match="dt must be|every must be"):
            on.systems.generate(name, n=10, transient=0, **kw)


def test_small_helpers_edge_cases(x):
    assert on.metrics.kendall_autocorrelation(x, 0) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="lag"):
        on.metrics.kendall_autocorrelation(x, len(x))
    assert on.recurrence.order_recurrence_plot(np.zeros(0, int), sparse=True).shape == (0, 0)
    with pytest.raises(ValueError, match="empty network"):
        on.OrdinalNetwork.from_edges([], []).random_walk(3)


def test_conditional_entropy_uses_the_source_marginal():
    # windows up, down: one transition. p(up) as a pattern frequency is 1/2, as a source it is 1,
    # so the old pattern-frequency form gave -1 bit here
    assert on.symbolize([1.0, 2.0, 1.0], 2).conditional_entropy() == 0.0
    # windows up, up, down: from "up" the next pattern is up or down with probability 1/2 each
    assert on.symbolize([1.0, 2.0, 3.0, 1.0], 2).conditional_entropy() == pytest.approx(1.0)
    # gaps: every run loses its last window as a source; the value stays within [0, log2 m]
    y = np.random.default_rng(1).standard_normal(3000)
    y[::9] = np.nan
    st = on.symbolize(y, 3)
    T = st.network().T.toarray()
    out = st.network().out_strength()
    rows = -np.sum(np.where(T > 0, T * np.log2(np.where(T > 0, T, 1.0)), 0.0), axis=1)
    assert st.conditional_entropy() == pytest.approx(float(np.sum(out / out.sum() * rows)))
    assert 0.0 <= st.conditional_entropy() <= math.log2(3)


def test_outgrowth_ratio_requires_matching_slide(x):
    a = on.OrdinalSymbolizer(3, w=1).fit(x)
    b = on.OrdinalSymbolizer(4, w=2).fit(x)
    with pytest.raises(ValueError, match="share ranking, tau and w"):
        on.entropy.outgrowth_ratio(a, b)


def test_entropy_table_accepts_m1(x):
    tab = on.entropy.entropy_table(on.OrdinalSymbolizer(1).sweep(x, [1, 2, 3]))
    assert np.isnan(tab["TPE"][0]) and np.isfinite(tab["TPE"][1:]).all()


def test_surrogate_input_validation_and_scale_free_fallback():
    x = on.systems.generate("logistic", n=5000)
    st = on.OrdinalSymbolizer(4).fit(x, keep_sequence=True)
    walk = st.network().random_walk(500, seed=1)
    with pytest.raises(ValueError, match="method must be"):
        on.surrogates.reconstruct(walk, st, x, method="nearest")
    with pytest.raises(ValueError, match="fitted on"):
        on.surrogates.reconstruct(walk, st, x[:-1])
    with pytest.raises(ValueError, match="outside the network"):
        on.surrogates.reconstruct([0, st.n_patterns], st, x)
    with pytest.raises(ValueError, match="at least m"):
        on.surrogates.network_surrogate(st, x, 2)
    # the surrogate of a rescaled series is the rescaled surrogate (fallback steps included)
    for scale in (1e-9, 1e9):
        xs = x * scale
        sts = on.OrdinalSymbolizer(4).fit(xs, keep_sequence=True)
        s, _ = on.surrogates.reconstruct(walk, sts, xs, seed=3)
        assert (sts.nodes_of(on.OrdinalSymbolizer(4).encode(s)) == walk).mean() > 0.98


def test_small_network_edge_cases():
    loop = on.OrdinalNetwork.from_edges([0], [0])
    assert loop.node_link_entropy(normalized=True).tolist() == [0.0]
    with pytest.raises(ValueError, match="k must be"):
        loop.n_cycles(0)
    with pytest.raises(ValueError, match="unknown method"):
        pytest.importorskip("igraph") and loop.communities("spectral")
    dag = on.OrdinalNetwork.from_edges([0, 1], [1, 2])               # no cycles: rho(A) = 0
    assert np.allclose(dag.centrality("katz"), [3.0, 2.0, 1.0])       # 1 + (number of out-paths)
    with pytest.raises(ValueError, match="malformed CSR"):
        on._core.random_walk(np.array([0, 2, 1]), np.array([0, 5]), np.array([1.0, 1.0]), 0, 5, 1)
    with pytest.raises(ValueError, match="L must satisfy"):
        on.metrics.local_rank_numbers([1.0, 2.0], 5)


def test_mackey_glass_long_run_is_cheap_and_reproducible():
    a = on.systems.generate("mackey_glass", n=4000, every=10, transient=2000)
    b = on.systems.generate("mackey_glass", n=2000, every=10, transient=2000)
    assert np.array_equal(a[:2000], b) and np.isfinite(a).all() and a.std() > 0.05


def test_eigen_quantities_are_reproducible_for_sparse_solver():
    x = on.systems.generate("henon", n=300_000)[:, 0]
    net = on.OrdinalSymbolizer(14).fit(x).network()
    assert net.n_nodes > 1500                                         # ARPACK path
    assert net.spectral_radius() == net.spectral_radius.__func__(on.OrdinalSymbolizer(14).fit(x).network())


# ------------------------------------------------------------- PR #2 review comments --

def test_spectral_radius_is_exactly_zero_without_cycles():
    # a path of 12 nodes: A is one nilpotent Jordan block, for which dense eigensolvers
    # return |lambda| ~ eps**(1/12) ~ 0.05 instead of 0
    k = 12
    path = on.OrdinalNetwork.from_edges(np.arange(k - 1), np.arange(1, k))
    assert path.is_acyclic() and path.spectral_radius() == 0.0
    assert path.spectral_radius(log=True) == -math.inf
    # default Katz alpha: the finite series 1 + (number of nodes reachable along the path)
    assert np.allclose(path.centrality("katz"), np.arange(k, 0, -1))
    assert not on.OrdinalNetwork.from_edges([0, 1], [1, 0]).is_acyclic()
    assert not on.OrdinalNetwork.from_edges([0, 0], [0, 1]).is_acyclic()        # self-loop


def test_entropies_for_m1_and_min_entropy(x):
    st1 = on.OrdinalSymbolizer(1).fit(x)
    assert st1.permutation_entropy(normalized=True) == 0.0
    for a in (0.5, 1.0, 2.0, math.inf):
        assert st1.renyi_entropy(a, normalized=True) == 0.0
    assert st1.tsallis_entropy(2.0, normalized=True) == 0.0
    st = on.OrdinalSymbolizer(3).fit(x)
    p = st.pattern_probabilities
    assert st.renyi_entropy(math.inf) == pytest.approx(-math.log2(p.max()))
    # Renyi entropy is non-increasing in alpha and tends to the min-entropy
    hs = [st.renyi_entropy(a) for a in (0.5, 1.0, 2.0, 50.0, math.inf)]
    assert all(u >= v - 1e-12 for u, v in zip(hs, hs[1:]))
    assert st.renyi_entropy(1e4) == pytest.approx(st.renyi_entropy(math.inf), abs=1e-3)
    for bad in (-1.0, math.nan):
        with pytest.raises(ValueError, match="alpha must be"):
            st.renyi_entropy(bad)


def test_from_edges_rejects_n_nodes_smaller_than_the_edge_list():
    with pytest.raises(ValueError, match="refers to node 5"):
        on.OrdinalNetwork.from_edges([0, 5], [1, 2], n_nodes=3)
    assert on.OrdinalNetwork.from_edges([0, 5], [1, 2], n_nodes=9).n_nodes == 9


def test_random_walk_validates_the_graph_once_and_stays_reproducible():
    net = on.OrdinalSymbolizer(4).fit(np.random.default_rng(2).standard_normal(3000)).network()
    a = net.random_walk(500, start=0, seed=7)
    assert ("walk", True) in net._cache
    assert np.array_equal(net.random_walk(500, start=0, seed=7), a)             # cached arrays, same walk
    assert np.array_equal(net.random_walk(500, start=0, seed=7, weighted=False),
                          on.OrdinalNetwork.from_matrix(net.A).random_walk(500, start=0, seed=7))
    bad = on.OrdinalNetwork.from_edges([0, 1], [1, 0], [1.0, np.nan])
    with pytest.raises(ValueError, match="finite and non-negative"):
        bad.random_walk(5, start=0, seed=1)
    # the cheap consistency checks run even when the full scan is skipped
    with pytest.raises(ValueError, match="malformed CSR"):
        on._core.random_walk(np.array([0, 1, 3]), np.array([1, 0]), np.array([1.0, 1.0]), 0, 5, 1, False)

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

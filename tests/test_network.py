import math

import numpy as np
import pytest
import scipy.sparse as sp

import ordinal_networks as on


@pytest.fixture(scope="module")
def logistic():
    x = on.systems.generate("logistic", n=2_000_000)
    return x, on.OrdinalSymbolizer(1).sweep(x, range(2, 13))


def test_matrices(logistic):
    _, sw = logistic
    net = sw[6].network()
    W = net.W.toarray()
    assert np.allclose(net.T.toarray().sum(1)[W.sum(1) > 0], 1)
    assert np.allclose(net.Q.toarray().sum(0)[W.sum(0) > 0], 1)
    assert set(np.unique(net.A.data)) == {1.0}
    assert net.n_edges == sw[6].n_transitions


def test_topological_entropy_logistic(logistic):
    _, sw = logistic
    est = [sw[m].network().spectral_radius(log=True) for m in range(8, 13)]
    assert all(abs(e - math.log(2)) < 0.01 for e in est)
    assert abs(on.entropy.growth_rate(sw, range(4, 12)) - 0.8232) < 0.03


def test_lyapunov_logistic(logistic):
    _, sw = logistic
    lam = sw[10].network().lyapunov_sum()
    assert abs(lam - math.log(2)) < 0.02


def test_lyapunov_matches_matlab_formula(logistic):
    """Dense re-implementation of LyapunovSum_OrdNet."""
    _, sw = logistic
    net = sw[5].network()
    W = net.W.toarray()
    Q = W / W.sum(0)
    vals, vecs = np.linalg.eig(Q)
    u = np.real(vecs[:, np.argmax(vals.real)]); u /= u.sum()
    Pm = np.diag(1 / u) @ Q @ np.diag(u)
    vals, vecs = np.linalg.eig(Pm.T)
    v = np.real(vecs[:, np.argmax(vals.real)]); v /= v.sum()
    LgQ = np.zeros_like(Q); LgQ[Q > 0] = np.log(Q[Q > 0])
    lam = -np.sum(np.diag(v) @ Pm * LgQ)
    assert abs(lam - net.lyapunov_sum()) < 1e-9


def test_prune(logistic):
    _, sw = logistic
    net = sw[10].network()
    pr = net.prune(min_rel_freq=1e-4)
    assert pr.n_edges < net.n_edges
    assert np.all(pr.edge_probabilities() > 0) and pr.n_nodes <= net.n_nodes


def test_topology_measures():
    # 3-cycle with a self loop on node 0
    net = on.OrdinalNetwork.from_edges([0, 1, 2, 0], [1, 2, 0, 0], [2, 2, 2, 1])
    assert net.is_irreducible()
    assert list(net.out_degree()) == [2, 1, 1]
    assert net.n_cycles(3) == 4    # trace(A^3): three rotations of the cycle + loop^3
    assert set(net.cycle_nodes(1)) == {0}
    H = net.node_link_entropy()
    assert abs(H[0] - (-(2 / 3) * math.log2(2 / 3) - (1 / 3) * math.log2(1 / 3))) < 1e-12
    assert H[1] == 0
    pi = net.stationary_distribution()
    assert np.allclose(pi @ net.T.toarray(), pi)
    ann = net.avg_neighbor_degree()
    assert ann.shape == (3, 2)
    assert np.isfinite(net.centrality("betweenness")).all()
    walk = net.random_walk(1000, start=0, seed=3)
    assert len(walk) == 1001 and set(np.unique(walk)) <= {0, 1, 2}


def test_entropies(logistic):
    _, sw = logistic
    st = sw[4]
    p = st.pattern_probabilities
    assert abs(st.permutation_entropy() - (-np.sum(p * np.log2(p)))) < 1e-12
    assert 0 < st.permutation_entropy(normalized=True) <= 1
    assert abs(st.forbidden_fraction() - (1 - st.n_patterns / 24)) < 1e-12
    assert st.conditional_entropy() >= 0


def test_metrics():
    m = on.metrics
    assert m.hamming([1, 2, 3], [1, 3, 2]) == pytest.approx(2 / 3)
    assert m.kendall_tau_distance([1, 2, 3, 4], [4, 3, 2, 1]) == 1.0
    assert m.kendall_tau_distance([1, 2, 3], [1, 3, 2]) == pytest.approx(1 / 3)
    assert m.spearman_footrule([1, 2, 3, 4], [4, 3, 2, 1]) == 1.0
    assert m.spearman_rho_distance([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(1.0)
    x = np.array([3.0, 1.0, 2.0, 2.0])
    assert list(m.rank_numbers(x)) == [4, 1, 3, 3]
    assert list(m.local_rank_numbers(x, 2)) == [2, 0]


def test_mapping_and_surrogates():
    x = on.systems.generate("logistic", n=20_000)
    st = on.OrdinalSymbolizer(5).fit(x, keep_sequence=True)
    net = st.network()
    vals = on.mapping.property_per_point(st, net.out_degree())
    assert len(vals) == len(x) and np.isfinite(vals[: st.n_windows]).all()
    idx = on.mapping.windows_on_cycles(st, net, 1)
    assert len(idx) > 0
    rng = on.mapping.pattern_value_ranges(st, x)
    assert (rng[:, 0] <= rng[:, 1]).all()
    walk = net.random_walk(3000, seed=1)
    s, fb = on.surrogates.reconstruct(walk, st, x, method="random", seed=2)
    # the surrogate realises exactly the walk's pattern sequence
    s_ids = on.OrdinalSymbolizer(5).encode(s)
    assert np.mean(st.nodes_of(s_ids) == walk) > 0.98
    s2, _ = on.surrogates.reconstruct(walk, st, x, method="constrained", seed=2)
    assert np.mean(st.nodes_of(on.OrdinalSymbolizer(5).encode(s2)) == walk) > 0.85
    # on a well-spread series every window is realised exactly
    y = np.random.default_rng(0).standard_normal(20_000)
    sy = on.OrdinalSymbolizer(4).fit(y, keep_sequence=True)
    wy = sy.network().random_walk(2000, seed=5)
    r, _ = on.surrogates.reconstruct(wy, sy, y, seed=1)
    assert (sy.nodes_of(on.OrdinalSymbolizer(4).encode(r)) == wy).all()


def test_order_recurrence():
    seq = np.array([0, 1, 0, 2, 1])
    R = on.recurrence.order_recurrence_plot(seq)
    assert R[0, 2] and not R[0, 1] and R.sum() == 9
    Rs = on.recurrence.order_recurrence_plot(seq, sparse=True)
    assert (Rs.toarray() == R).all()
    assert on.recurrence.recurrence_rate(seq) == pytest.approx(9 / 25)
    B = on.recurrence.order_structure_matrix([3.0, 1.0, 2.0])
    assert B[0, 1] and not B[1, 0] and B.diagonal().all()


def test_systems():
    for name in on.systems.MAPS:
        s = on.systems.MAPS[name]
        if not s.regimes:
            continue
        x = on.systems.generate(name, n=1000, squeeze=False)
        assert x.shape == (1000, s.dim) and np.isfinite(x).all(), name
    for name in ("lorenz", "rossler", "chua"):
        P_ = on.systems.poincare(name, n=200)
        assert P_.shape == (200, 3) and np.isfinite(P_).all()
    P_ = on.systems.poincare("lorenz", n=500)
    assert np.allclose(P_[:, 2], 27.0)
    mg = on.systems.generate("mackey_glass", n=2000, every=20)
    assert np.isfinite(mg).all() and mg.std() > 0.05
    # Henon: Lyapunov spectrum ~ (0.42, -1.62), sum = ln 0.3
    xy = on.systems.generate("henon", n=100_000)
    spec = on.systems.lyapunov_spectrum("henon", xy)
    assert abs(spec[0] - 0.419) < 0.01 and abs(spec.sum() - math.log(0.3)) < 1e-6
    assert abs(on.systems.lyapunov_sum_jacobian("logistic", on.systems.generate("logistic", n=100_000)) - math.log(2)) < 0.01


def test_generate_to_file(tmp_path):
    p = tmp_path / "log.npy"
    on.systems.generate_to_file("logistic", p, n=25_000, chunk=7_000)
    a = np.load(p)
    b = on.systems.generate("logistic", n=25_000)
    assert np.array_equal(a, b)
    st = on.OrdinalSymbolizer(6, chunk_size=3000).fit(p)
    assert st.n_points == 25_000

"""External validation: agreement with ordpy (Pessa & Ribeiro 2021) wherever both
packages implement the same quantity, and with analytically known values."""
import math

import numpy as np
import pytest

import ordinal_networks as on

ordpy = pytest.importorskip("ordpy")


@pytest.fixture(scope="module")
def series():
    rng = np.random.default_rng(7)
    return {
        "logistic": on.systems.generate("logistic", n=20_000),
        "henon": on.systems.generate("henon", n=20_000)[:, 0],
        "noise": rng.standard_normal(20_000),
        "ar1": np.cumsum(rng.standard_normal(20_000)) * 0.1 + rng.standard_normal(20_000),
    }


CASES = [(name, m, tau) for name in ("logistic", "henon", "noise", "ar1") for m, tau in ((3, 1), (4, 1), (5, 2), (6, 1))]


@pytest.mark.parametrize("name,m,tau", CASES)
def test_pattern_distribution_matches_ordpy(series, name, m, tau):
    x = series[name]
    st = on.OrdinalSymbolizer(m, tau).fit(x)
    symbols, probs = ordpy.ordinal_distribution(x, dx=m, taux=tau)
    # ordpy symbols are 0-based argsort permutations = the chronological ranking
    ids = on.encode(np.asarray(symbols) + 1)
    order = np.argsort(ids)
    assert np.array_equal(ids[order], st.pattern_ids)
    assert np.allclose(probs[order], st.pattern_probabilities, rtol=0, atol=1e-15)


@pytest.mark.parametrize("name,m,tau", CASES)
def test_entropies_match_ordpy(series, name, m, tau):
    x = series[name]
    st = on.OrdinalSymbolizer(m, tau).fit(x)
    assert st.permutation_entropy(normalized=True) == pytest.approx(ordpy.permutation_entropy(x, dx=m, taux=tau), abs=1e-12)
    assert st.permutation_entropy(base=2) == pytest.approx(
        ordpy.permutation_entropy(x, dx=m, taux=tau, base=2, normalized=False), abs=1e-12)
    h, c = ordpy.complexity_entropy(x, dx=m, taux=tau)
    assert st.complexity_entropy() == pytest.approx((h, c), abs=1e-12)
    assert st.statistical_complexity() == pytest.approx(c, abs=1e-12)
    for a in (0.5, 2.0, 3.0):
        assert st.renyi_entropy(a, normalized=True) == pytest.approx(ordpy.renyi_entropy(x, alpha=a, dx=m, taux=tau), abs=1e-12)
        assert st.tsallis_entropy(a, normalized=True) == pytest.approx(ordpy.tsallis_entropy(x, q=a, dx=m, taux=tau), abs=1e-12)
    assert st.forbidden_fraction() == pytest.approx(
        ordpy.missing_patterns(x, dx=m, taux=tau, return_missing=False), abs=1e-15)


@pytest.mark.parametrize("name,m,tau", CASES)
def test_network_matches_ordpy(series, name, m, tau):
    x = series[name]
    st = on.OrdinalSymbolizer(m, tau).fit(x)
    nodes, links, weights = ordpy.ordinal_network(x, dx=m, taux=tau, normalized=False)

    def to_id(label):
        return int(on.encode(np.array([int(ch) for ch in str(label).split("|")]) + 1))

    theirs = {(to_id(a), to_id(b)): int(c) for (a, b), c in zip(links, weights)}
    ids = [int(v) for v in st.pattern_ids]
    ours = {(ids[s], ids[d]): int(c) for s, d, c in zip(st.src, st.dst, st.weight)}
    assert ours == theirs
    assert sorted(to_id(v) for v in nodes) == ids
    assert st.global_node_entropy() == pytest.approx(ordpy.global_node_entropy(x, dx=m, taux=tau), abs=1e-12)
    if tau == 1:
        assert st.missing_transition_fraction() == pytest.approx(
            ordpy.missing_links(x, dx=m, return_missing=False), abs=1e-15)


def test_complexity_is_stable_for_large_m():
    # m! - V unobserved patterns enter in closed form; white noise sits near (1, 0) only when N >> m!
    x = np.random.default_rng(0).standard_normal(50_000)
    for m in (12, 20, 30):
        h, c = on.OrdinalSymbolizer(m).fit(x).complexity_entropy()
        assert 0 < h < 1 and 0 <= c < 1 and math.isfinite(c)


def test_possible_transitions():
    f = math.factorial
    assert on.entropy.possible_transitions(4) == f(4) * 4
    assert on.entropy.possible_transitions(4, tau=1, w=2) == f(4) * 4 * 3
    assert on.entropy.possible_transitions(4, tau=2, w=2) == f(4) * 4
    assert on.entropy.possible_transitions(4, tau=1, w=4) == f(4) ** 2      # no shared points
    assert on.entropy.possible_transitions(4, tau=2, w=3) == f(4) ** 2      # windows do not align
    # every transition of overlapping windows is observed in long white noise at m = 3
    x = np.random.default_rng(1).standard_normal(100_000)
    for w in (1, 2):
        assert on.OrdinalSymbolizer(3, w=w).fit(x).missing_transition_fraction() == 0.0


# ------------------------------------------------------------- analytic values --

def test_logistic_r4_analytic_values():
    """Fully chaotic logistic map: h_top = h_KS = lambda = ln 2."""
    x = on.systems.generate("logistic", n=2_000_000)
    sw = on.OrdinalSymbolizer(2).sweep(x, [9, 10])
    net = sw[10].network()
    assert net.spectral_radius(log=True) == pytest.approx(math.log(2), abs=5e-3)
    assert net.lyapunov_sum() == pytest.approx(math.log(2), abs=2e-2)
    assert net.metric_entropy_rate() == pytest.approx(math.log(2), abs=2e-2)
    # Methods II and III converge from above and slowly in m; they bound ln 2 at m = 9
    assert math.log(2) < on.entropy.log_mean_outgrowth_ratio(sw[9], sw[10]) < sw[9].topological_permutation_entropy() * math.log(2)
    assert on.systems.lyapunov_sum_jacobian("logistic", x[:200_000]) == pytest.approx(math.log(2), abs=5e-3)


def test_white_noise_limits():
    x = np.random.default_rng(3).standard_normal(500_000)
    st = on.OrdinalSymbolizer(4).fit(x)
    h, c = st.complexity_entropy()
    assert h == pytest.approx(1.0, abs=1e-3) and c == pytest.approx(0.0, abs=1e-3)
    assert st.forbidden_fraction() == 0.0
    # m = 2, i.i.d. data: an "up" is followed by a "down" with probability 2/3 (and vice versa),
    # so the conditional entropy is the binary entropy of 1/3
    h2 = -(1 / 3) * math.log2(1 / 3) - (2 / 3) * math.log2(2 / 3)
    assert on.OrdinalSymbolizer(2).fit(x).conditional_entropy() == pytest.approx(h2, abs=1e-3)


def test_monotone_series_limits():
    st = on.OrdinalSymbolizer(5).fit(np.arange(1000.0))
    assert st.n_patterns == 1
    assert st.complexity_entropy() == (0.0, 0.0)
    assert st.permutation_entropy() == 0.0 and st.conditional_entropy() == 0.0
    assert st.renyi_entropy(2.0) == 0.0 and st.tsallis_entropy(2.0) == 0.0

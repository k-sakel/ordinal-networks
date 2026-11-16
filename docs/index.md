# ordinal-networks

Fast ordinal symbolic analysis of very long time series, and **ordinal partition
networks** as Markov models of the underlying dynamics. A Python library with a
multi-core Rust core.

## Quick start

```python
import ordinal_networks as on

x = on.systems.generate("logistic", "ChaosR4", n=10**7)   # or any 1-D array / .npy file
stats = on.OrdinalSymbolizer(m=12).fit(x)                  # tau = 1, w = 1 (defaults)
stats.n_patterns, stats.n_transitions                      # (10525, 21001)

net = stats.network()
net.spectral_radius(log=True)    # 0.692  topological entropy estimate (ln 2 = 0.693)
net.lyapunov_sum()               # 0.687  ordinal Lyapunov estimate
```

Ten million points at pattern length 12 take about a quarter of a second on a
16-core laptop.

## Why this library

1. **Scale.** Series of length N ~ 10⁹ and pattern lengths up to m = 34, streamed
   from disk, counted on all cores, with results written out-of-core when they do
   not fit in memory. Entropy estimators based on ordinal patterns converge slowly
   in m, and m! patterns need data: long series and long patterns are where the
   methods become informative.
2. **Ordinal partition networks as Markov chains.** Topological entropy (four
   estimators), the ordinal Lyapunov sum, entropy rate, stationary distribution,
   cycles and unstable periodic orbits, network measures, surrogates by random walk.
3. **Fixed, decodable pattern IDs.** Every pattern has the same integer ID in
   every series, chunk and run: its lexicographic rank among all m! permutations.
   IDs decode back to permutations, so no pattern dictionary is ever stored.

## Installation

```
pip install ordinal-networks
```

Wheels are provided for Windows, macOS and Linux (CPython ≥ 3.10); no Rust toolchain
is needed. Optional extras: `ordinal-networks[graph]` adds python-igraph and networkx
(betweenness, PageRank, community detection, graph export).

!!! note "Before the first PyPI release"
    Until version 0.2.0 is published on PyPI, install the wheel for your platform from
    the [GitHub releases](https://github.com/k-sakel/ordinal-networks/releases), or
    build from source (see [Contributing](contributing.md)).

## Where next

- [Concepts and conventions](concepts.md): what `m`, `tau`, `w` and `ranking` mean,
  how ties and missing values are treated, and how pattern IDs work.
- [Large data](large-data.md): memory-mapped input, sweeps over m, out-of-core results.
- [Ordinal statistics](statistics.md) and [networks](networks.md): every quantity,
  with units and references.
- Coming from [ordpy](from-ordpy.md) or the [MATLAB toolbox](from-matlab.md):
  function-by-function maps.

## Citing

If you use the library, please cite the software (see `CITATION.cff` in the
repository) and the methods papers:

- K. Sakellariou, T. Stemler, M. Small, *Markov modeling via ordinal partitions: An
  alternative paradigm for network-based time-series analysis*, Phys. Rev. E **100**,
  062307 (2019).
- K. Sakellariou, T. Stemler, M. Small, *Estimating topological entropy using ordinal
  partition networks*, Phys. Rev. E **103**, 022214 (2021).

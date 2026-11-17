# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[semantic versioning](https://semver.org/). While the version is below 1.0, minor
releases may change the public API; such changes are listed under **Changed**.

## [0.2.0] - unreleased

### Added
- `entropy.statistical_complexity` and `entropy.complexity_entropy` (Jensen–Shannon
  statistical complexity and the complexity–entropy plane), exact for any m ≤ 34
  because the unobserved patterns enter in closed form.
- `entropy.renyi_entropy`, `entropy.tsallis_entropy`.
- `entropy.global_node_entropy` (the definition used by ordpy) and
  `entropy.missing_transition_fraction` / `entropy.possible_transitions`.
- The same quantities as methods of `OrdinalStatistics`.
- Cross-validation tests against ordpy (pattern distributions, entropies, complexity,
  network edge weights) and against analytic values.
- Documentation site (MkDocs), `CITATION.cff`, contribution guide, this changelog.
- `paper/`: manuscript and the scripts that regenerate every figure and table.
- `benchmarks/compare.py`: head-to-head timing against ordpy, with thread scaling.
- Type marker (`py.typed`).

### Changed
- `conditional_entropy` now uses the source marginal of the transition counts instead
  of the pattern frequencies. The two differ by edge effects (the last window of every
  run has no successor); the old form could return negative values for short series
  or series with gaps. For long gap-free series the change is of order 1/N.
- Wheels are built against the stable ABI (`abi3`): one wheel per platform serves every
  CPython ≥ 3.10.
- `OrdinalNetwork.subgraph` no longer accepts the `relabel` argument, which was ignored.
- `surrogates.reconstruct` validates `method`, the series length and the walk before
  starting, and its fallback step for a new extreme value scales with the data range
  (it was the absolute constant 1e-3).
- `__version__` is read from the installed package metadata.

### Fixed
- Sparse eigen-computations (`spectral_radius`, `stationary_distribution`,
  `lyapunov_sum` above 1500 nodes) start ARPACK from a fixed vector, so results are
  reproducible from run to run.
- `systems.generate("mackey_glass", ...)` keeps a ring buffer of one delay instead of
  the whole trajectory: memory no longer grows with the number of steps. Output is
  bit-identical.
- `_core.random_walk` rejects a malformed CSR graph with `ValueError` instead of
  panicking.
- Spill-file names are unique per process through a serial number (they used the
  address of the counter, which changes when the counter is moved).
- `Counter.feed` after `finalize` fails before doing any work.
- `spectral_radius` returns exactly 0 for a network without cycles (new
  `OrdinalNetwork.is_acyclic`); eigensolvers return noise for nilpotent matrices, which
  made the default Katz `alpha` blow up.
- On a network without cycles, eigenvector centrality is 0 for every node and
  `stationary_distribution`, `lyapunov_sum` and `metric_entropy_rate` raise `ValueError`
  instead of returning normalised solver noise.
- `permutation_entropy(normalized=True)` and `renyi_entropy(normalized=True)` return 0
  for m = 1 instead of dividing by zero; `renyi_entropy(alpha=inf)` is the min-entropy
  instead of NaN.
- `OrdinalNetwork.from_edges` rejects an `n_nodes` smaller than the edge list requires.
- `random_walk` converts and validates the graph once per network instead of on every
  walk.
- `entropy.outgrowth_ratio` rejects statistics with different `w`;
  `entropy.entropy_table` accepts sweeps that include m = 1;
  `OrdinalNetwork.n_cycles(k)` rejects k < 1; `node_link_entropy(normalized=True)` on a
  one-node network returns 0 instead of dividing by zero; default Katz `alpha` on a
  network without cycles.

## [0.1.0] - 2026-11-08

First release: Rust core and Python API for ordinal patterns and ordinal partition
networks.

[0.2.0]: https://github.com/k-sakel/ordinal-networks/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/k-sakel/ordinal-networks/releases/tag/v0.1.0

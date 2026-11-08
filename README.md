# ordinal-networks

Fast ordinal symbolic analysis of very long time series, and **ordinal partition
networks** as Markov models of the underlying dynamics: a Python library with a
multi-core Rust core.

It ports and extends the MATLAB toolbox behind

* K. Sakellariou, T. Stemler, M. Small, *Markov modeling via ordinal partitions:
  An alternative paradigm for network-based time-series analysis*, Phys. Rev. E
  **100**, 062307 (2019)
* K. Sakellariou, T. Stemler, M. Small, *Estimating topological entropy using
  ordinal partition networks* (2021)

and handles series of length **N ~ 10⁹** and pattern lengths up to **m = 34** on a
laptop.

```python
import ordinal_networks as on

x = on.systems.generate("logistic", "ChaosR4", n=10**7)   # or any 1-D array / .npy file
stats = on.OrdinalSymbolizer(m=12).fit(x)                  # tau = 1, w = 1 (defaults)
stats.n_patterns, stats.n_transitions                      # 10525 patterns, 21001 transitions

net = stats.network()
net.spectral_radius(log=True)    # 0.692  topological entropy estimate (ln 2 = 0.693)
net.lyapunov_sum()               # 0.697  ordinal Lyapunov estimate (LyapunovSum_OrdNet)
```

## Installation

**Windows / macOS / Linux, prebuilt wheel (no Rust needed).** Every push to `main`
builds wheels for Python 3.10–3.13 in GitHub Actions (*Actions → latest run →
Artifacts → `wheels-windows-latest`*); tagged versions (`v0.1.0`, …) attach them
to a GitHub Release. Install the wheel that matches your Python:

```
pip install ordinal_networks-0.1.0-cp312-cp312-win_amd64.whl
```

**From source** (needs a Rust toolchain from <https://rustup.rs>, plus on Windows
the "Desktop development with C++" workload of Visual Studio Build Tools):

```
git clone https://github.com/k-sakel/ordinal-networks
cd ordinal-networks
pip install maturin
pip install -e ".[test,graph]"     # builds the Rust core in release mode
pytest                             # 113 tests
```

Dependencies: `numpy`, `scipy`, `pyarrow`; optional `python-igraph`/`networkx`
(betweenness, PageRank, community detection, graph export).

## Concepts and conventions

| | |
|---|---|
| `m` | pattern length, 1 ≤ m ≤ 34 |
| `tau` | embedding delay between the points of a window (default 1) |
| `w` | slide lag between the starts of consecutive windows (default 1 = maximal overlap; `w = m*tau` = no overlap; any w ≥ 1) |
| `ranking` | `"chronological"` (Bandt–Pompe, default: π = (p₁…p_m) with x_{p₁} < … < x_{p_m}, i.e. MATLAB `[~,s] = sort(window)`) or `"amplitude"` (Small: entry j = descending-amplitude rank of the j-th point) |

Window k contains `x[k*w + j*tau]`, j = 0…m−1; there are ⌊(N − (m−1)τ − 1)/w⌋ + 1 windows.
Transitions link consecutive windows k → k+1 of the ordinal representation.

**Pattern IDs.** Each pattern has a fixed ID in {1, …, m!}: its lexicographic rank
among all permutations (in the chosen ranking). IDs are computed directly from the
data in O(m) per window and are exact up to m = 34 (34! < 2¹²⁸). They
are comparable across series, chunks and parameter values, and decode back to the
permutation, so no pattern dictionary is ever stored. With the default ranking,
the ID order equals MATLAB's `unique(s','rows')` order:

```python
on.decode(stats.pattern_ids[:3], 12)          # 3 x 12 array of permutations (1-based)
on.encode([3, 1, 2])                          # -> 5
on.pattern_of([0.3, 0.9, 0.1])                # ID of one window
on.patterns.convert(ids, m, "chronological", "amplitude")
```

For m ≤ 20 IDs are `uint64` arrays; for m > 20 they are arrays of Python ints.

**Ties** inside a window: the earlier point counts as smaller (Bandt & Pompe). Ties
are counted: `stats.n_tied_windows`, `stats.n_tie_pairs`.

**NaN / ±Inf** are counted (`stats.n_nonfinite_points`). Windows containing them
are skipped (`n_nonfinite_windows`), and no transition is recorded into or out of
a skipped window, so the chain is never joined across a gap.

## Large data

```python
on.systems.generate_to_file("logistic", "log_r4.npy", n=10**9)   # chunked, constant memory

sym = on.OrdinalSymbolizer(m=25, threads=0,          # all cores
                           chunk_size=4_000_000,      # windows per chunk
                           memory_budget=3 * 2**30,   # RAM for the counting hash tables
                           spill_dir=None)            # None = system temp dir; False = never spill
stats = sym.fit("log_r4.npy", progress=True)          # .npy is memory-mapped
sweep = sym.sweep("log_r4.npy", m=range(2, 31))       # many m in one pass

# results too large for RAM (large m and N: nearly every window is a new pattern)
stats = sym.fit("log_r4.npy", output="LOG_m30")       # streamed to memory-mapped .npy files
stats = on.load_statistics("LOG_m30")                 # reload (memory-mapped)
sweep = sym.sweep("log_r4.npy", m=range(2, 31), output_dir="LOG_sweep")
```

Choose the out-of-core `output=` mode when the result tables approach your RAM.
At ~24 bytes per transition and ~16 bytes per pattern, 10⁸ distinct transitions
need about 4 GB.

The series is streamed from a memory-mapped `.npy` in chunks. Each chunk is
split across cores and encoded with incremental rank updates, and patterns and
transitions are counted in per-thread hash tables. When the tables outgrow
`memory_budget`, sorted runs are spilled to disk and merged at the end. Results
are identical whatever the chunk size, thread count or spilling.

Benchmarks on the x-component of the Hénon map (2-core cloud VM with 7 GB RAM;
an 8-core laptop is correspondingly faster). Series are read from memory-mapped
`.npy` files. Runs marked * used the out-of-core `output=` mode.

| N | m | time | distinct patterns | distinct transitions | peak RSS |
|---|---|---|---|---|---|
| 10⁸ | 5 | 3.0 s | 29 | 57 | 0.9 GB |
| 10⁸ | 10 | 4.0 s | 1,396 | 2,663 | 0.9 GB |
| 10⁸ | 15 | 5.2 s | 36,276 | 62,335 | 0.9 GB |
| 10⁸ | 20 | 10.0 s | 575,761 | 935,979 | 1.2 GB |
| 10⁸ | 25* | 29.9 s | 6,039,105 | 8,764,510 | 2.4 GB |
| 10⁸ | 30* | 99.0 s | 29,570,374 | 36,233,596 | 2.7 GB |
| 10⁹ | 12 | 43.1 s | 5,683 | 10,354 | 5.4 GB† |
| 10⁹ | 25* | 241.7 s | 8,267,290 | 13,113,093 | 5.4 GB† |

† mostly the operating system's page cache of the 8 GB memory-mapped input. This
memory is reclaimable and is not held by the library.

Other input formats: any NumPy array, `.parquet`, `.csv`/`.txt`/`.dat`, `.json`
(`column=` selects a column of 2-D data).

## Storage

Only what is needed downstream is kept: pattern frequencies and the transition
edge list. Together they define the Markov chain.

```python
stats.save("LOG_ChaosR4_m25")      # LOG_ChaosR4_m25.nodes.parquet + .edges.parquet
stats = on.load_statistics("LOG_ChaosR4_m25")
```

* `*.nodes.parquet`: `id` (or `id_hi`/`id_lo` for m > 20) and `count`, sorted by ID
* `*.edges.parquet`: `src`, `dst` (node indices), `weight` (transition count),
  using the smallest sufficient integer types, zstd-compressed
* parameters and counters (m, τ, w, ranking, N, ties, non-finite, …) as JSON in
  the Parquet metadata
* optional `*.sequence.npy`: node index of every window. Request it with
  `fit(..., keep_sequence=True)`, or pass `sequence_path=` to write it to disk directly.

## Networks

```python
net = stats.network()          # OrdinalNetwork: weighted, directed, with self-loops
net.W, net.A, net.T, net.Q     # counts, binary, row-stochastic, column-normalised (scipy.sparse)
net = net.prune(min_rel_freq=1e-4)     # optional low-frequency transition pruning
net.largest_scc(); net.is_irreducible(); net.components("strong")
```

| Quantity | Python | MATLAB origin |
|---|---|---|
| ln ρ(A), topological entropy (Method IV) | `net.spectral_radius(log=True)` | `SpectralRadius_OrdNet` |
| ordinal Lyapunov (sum of positive exponents) | `net.lyapunov_sum()` (`method="T"` for the alternative) | `LyapunovSum_OrdNet` |
| stationary distribution, entropy rate | `net.stationary_distribution()`, `net.metric_entropy_rate()` | `MyOrdinalMarkov` |
| degree, strength, ANND/ANNS | `in_degree`, `out_degree`, `in_strength`, `out_strength`, `avg_neighbor_degree(weighted=)` | `TopologyOrdinalNetwork` |
| node-link entropy, hierarchical degree | `node_link_entropy()`, `hierarchical_degree(h)` | `nodeLinkEntropy`, `DiscreteMapNetwork_HierarchicalDegree` |
| closed k-walks / UPO detection | `n_cycles(k)`, `cycle_nodes(k)`, `mapping.windows_on_cycles` | `numberOfCycles`, `DiscreteOrdinalNetworkCyclesLocation` |
| clustering, asymmetry, assortativity | `clustering()`, `asymmetry()`, `assortativity()` | `clusteringCoeff`, `asymmetryCoeff`, `Assortativity` |
| centralities | `centrality("betweenness" \| "closeness_in" \| "closeness_out" \| "harmonic" \| "eigenvector" \| "katz" \| "pagerank")` | `CentralityOrdinalNetwork` |
| shortest paths, path length | `shortest_paths()`, `characteristic_path_length()` | `ShortestPaths`, `CharacteristicPathLength` |
| communities / modularity | `communities("leiden")` | `ModularityIndex` |
| random walks | `random_walk(steps, weighted=)` | `SimpleRandomWalk`, `TransitionsRandomWalk` |
| export | `to_igraph()`, `to_networkx()`, `save()` | `ordinal_network_to_list` |

## Ordinal statistics and entropies

```python
stats.permutation_entropy(normalized=True)
stats.conditional_entropy()                    # ConditionalEntropy
stats.forbidden_fraction()                     # CountForbiddenPatterns
stats.topological_permutation_entropy()        # Method II
on.entropy.growth_rate(sweep, range(4, 15))    # Method I
on.entropy.outgrowth_ratio(sweep[m], sweep[m+1]); on.entropy.log_mean_outgrowth_ratio(...)   # Method III
on.entropy.entropy_table(sweep, lyapunov=True) # all estimators for every m
```

Also included:

* `on.metrics`: permutation distances (Hamming, Kendall τ, Spearman footrule and ρ), plus rank numbers
* `on.recurrence`: order recurrence plots and the order structure matrix
* `on.mapping`: maps node properties back to the series or attractor, and gives the 1-D projection of the ordinal partition
* `on.surrogates`: surrogate series from network random walks, in random or constrained variants

## Dynamical systems

`on.systems` ports the MATLAB `dynamical-systems` toolbox, with the same equations,
parameter order and named regimes, running in Rust:

* maps: logistic, cubic, tent, dyadic, bent baker, Gauss, piecewise-linear,
  Hénon, Lozi, Ikeda, Holmes, Tinkerbell, Gingerbread man, area-preserving Hénon,
  Chirikov, Arnold cat, baker's, stiletto, Zaslavsky, solenoid, generalised Hénon,
  folded towel
* flows (fixed-step RK4): Lorenz, Rössler, 4-D Rössler, Chua, Muthuswamy,
  Rabinovich–Fabrikant, Duffing, Van der Pol, Reiterer (9-D), ABC, double gyre
  (three variants), Bickley jet
* delay equation: Mackey–Glass

> **Floating-point orbits for very long series.** In double precision the
> logistic map at r = 4 started from x₀ = 0.4 (after 10⁴ transient steps) lands
> *exactly* on a fixed point at n ≈ 1.55 × 10⁷, and every later value is constant.
> Maps with exact 0/1 arithmetic (tent, dyadic at μ = 2) collapse even sooner, which
> is why μ = 2 − 10⁻¹² is used. `generate`/`generate_to_file` warn when this
> happens, and `stats.n_tied_windows` exposes it in any series. For N ≫ 10⁷, check
> the warning, or use a system whose orbits stay aperiodic for longer (e.g. Hénon).

```python
on.systems.generate("henon", "BenedicksCarleson", n=10**6)
on.systems.poincare("lorenz", n=10**6)          # z = 27 section, dz/dt > 0 (Hermite-interpolated crossings)
on.systems.MAPS["gauss"].regimes                 # named parameter sets
on.systems.lyapunov_spectrum("henon", xy)        # reference values via Jacobians
```

## Differences from the MATLAB code

* Pattern IDs are fixed (the rank among all m! permutations) instead of
  data-dependent `unique` indices.
* `HenonAreaPreservMapTimeSeries` sign error fixed: y' = x sin α **+** (y − x²) cos α
  (Hénon 1969). The MATLAB minus sign gives det J = −cos 2α, a contracting map
  that collapses onto the origin.
* `HammingMetric` typo fixed. `kendall_tau_distance` is now the permutation
  distance its documentation describes. The old behaviour (a lagged Kendall
  correlation of the series) is available as `metrics.kendall_autocorrelation`.
* `lyapunov_sum` excludes nodes outside the support of the leading eigenvector
  rather than dividing by zero.
* When no observed value fits, surrogate reconstruction uses the midpoint of the
  admissible interval, so the requested pattern is preserved.

## Licence

MIT.

# Coming from the MATLAB toolbox

The library ports and extends the MATLAB code behind Sakellariou, Stemler & Small
(2019, 2021).

| Quantity | Python | MATLAB origin |
|---|---|---|
| ln ρ(A), topological entropy (Method IV) | `net.spectral_radius(log=True)` | `SpectralRadius_OrdNet` |
| ordinal Lyapunov (sum of positive exponents) | `net.lyapunov_sum()` (`method="T"` for the alternative) | `LyapunovSum_OrdNet` |
| stationary distribution, entropy rate | `net.stationary_distribution()`, `net.metric_entropy_rate()` | `MyOrdinalMarkov` |
| degree, strength, ANND/ANNS | `in_degree`, `out_degree`, `in_strength`, `out_strength`, `avg_neighbor_degree(weighted=)` | `TopologyOrdinalNetwork` |
| node-link entropy, hierarchical degree | `node_link_entropy()`, `hierarchical_degree(h)` | `nodeLinkEntropy`, `DiscreteMapNetwork_HierarchicalDegree` |
| closed k-walks / UPO detection | `n_cycles(k)`, `cycle_nodes(k)`, `mapping.windows_on_cycles` | `numberOfCycles`, `DiscreteOrdinalNetworkCyclesLocation` |
| clustering, asymmetry, assortativity | `clustering()`, `asymmetry()`, `assortativity()` | `clusteringCoeff`, `asymmetryCoeff`, `Assortativity` |
| centralities | `centrality(...)` | `CentralityOrdinalNetwork` |
| shortest paths, path length | `shortest_paths()`, `characteristic_path_length()` | `ShortestPaths`, `CharacteristicPathLength` |
| communities / modularity | `communities("leiden")` | `ModularityIndex` |
| random walks | `random_walk(steps, weighted=)` | `SimpleRandomWalk`, `TransitionsRandomWalk` |
| conditional entropy, forbidden patterns | `stats.conditional_entropy()`, `stats.forbidden_fraction()` | `ConditionalEntropy`, `CountForbiddenPatterns` |
| outgrowth ratio | `entropy.outgrowth_ratio`, `entropy.log_mean_outgrowth_ratio` | `OutgrowthRatio` |
| surrogates | `surrogates.reconstruct(method="random" | "constrained")` | `GetTimeSeriesReconstruction`, `ConstrainedReconstruction` |
| export | `to_igraph()`, `to_networkx()`, `save()` | `ordinal_network_to_list` |

## Differences

- **Pattern IDs are fixed** (the rank among all m! permutations) instead of
  data-dependent `unique` indices. With the default ranking the ID order equals the
  row order of `unique(s, 'rows')`.
- **`HenonAreaPreservMapTimeSeries` sign error fixed:** y′ = x sin α **+** (y − x²) cos α
  (Hénon 1969). The MATLAB minus sign gives det J = −cos 2α, a contracting map that
  collapses onto the origin.
- **`HammingMetric` typo fixed.** `kendall_tau_distance` is the permutation distance
  its documentation describes; the old behaviour (a lagged Kendall correlation of the
  series) is `metrics.kendall_autocorrelation`.
- **`lyapunov_sum`** excludes nodes outside the support of the leading eigenvector
  rather than dividing by zero.
- **`conditional_entropy`** uses the source marginal of the transition counts, so it
  is a proper conditional entropy also for short series and series with gaps.
- **Surrogate reconstruction:** when no observed value fits, the midpoint of the
  admissible interval is used (or a step proportional to the data range beyond the
  current extreme), so the requested pattern is preserved.
- **Harmonic centrality and characteristic path length** follow the standard
  definitions (incoming paths; unreachable pairs excluded).

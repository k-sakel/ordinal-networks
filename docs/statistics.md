# Ordinal statistics and entropies

`OrdinalSymbolizer.fit` returns an `OrdinalStatistics`: the observed patterns with
their frequencies, and the observed transitions with theirs.

```python
stats = on.OrdinalSymbolizer(m=6).fit(x)
stats.pattern_ids, stats.pattern_counts, stats.pattern_probabilities
stats.permutations()                 # the patterns as permutations
stats.src, stats.dst, stats.weight   # transition edge list (node indices, counts)
stats.summary()
```

Below, $p_i$ is the relative frequency of pattern $i$, $V$ the number of observed
patterns, $w_{ij}$ the number of transitions $i \to j$ and $E$ the number of distinct
transitions.

## From the pattern frequencies

| quantity | call | definition |
|---|---|---|
| permutation entropy | `stats.permutation_entropy(normalized=, base=)` | $-\sum_i p_i \log p_i$; normalised by $\log m!$ |
| Rényi entropy | `stats.renyi_entropy(alpha)` | $\frac{1}{1-\alpha} \log \sum_i p_i^\alpha$ |
| Tsallis entropy | `stats.tsallis_entropy(q)` | $\sum_i p_i \ln_q (1/p_i)$ |
| statistical complexity | `stats.statistical_complexity()` | $C = Q_J[P, U]\, H[P]$ (Jensen–Shannon) |
| complexity–entropy plane | `stats.complexity_entropy()` | the point $(H, C)$ |
| forbidden patterns | `stats.forbidden_fraction()` | $1 - V/m!$ |
| topological permutation entropy | `stats.topological_permutation_entropy()` | $\log_2 V / (m-1)$ |

The statistical complexity treats the $m! - V$ unobserved patterns in closed form, so
it is exact for every m up to 34 without enumerating permutations.

## From the transitions

| quantity | call | definition |
|---|---|---|
| conditional entropy | `stats.conditional_entropy(base=)` | $-\sum_{ij} p_{ij} \log (p_{ij}/p_i)$ with $p_{ij} = w_{ij}/\sum w$ and $p_i = \sum_j p_{ij}$ |
| global node entropy | `stats.global_node_entropy()` | $\sum_i p^{\mathrm{in}}_i H_i$, $H_i$ = entropy of the out-transition probabilities of node $i$ |
| missing transitions | `stats.missing_transition_fraction()` | $1 - E / E_{\max}$ |

`conditional_entropy` and `global_node_entropy` are the same sum, weighted by the
outgoing and the incoming share of each node. They agree up to edge effects of
order 1/N.

$E_{\max}$ is the number of transitions that are combinatorially possible: when
consecutive windows share $m - s$ points ($w = s\tau$, $s < m$) each pattern has
$m!/(m-s)!$ possible successors, so $E_{\max} = m!\, m$ for $\tau = w = 1$; windows
that share no points allow all $(m!)^2$.

## Topological entropy: four estimators

For a sweep over pattern lengths (`sweep = sym.sweep(x, range(2, 21))`):

| method | call | estimator |
|---|---|---|
| I | `on.entropy.growth_rate(sweep, range(4, 15))` | least-squares slope of $\ln V(m)$ against $m$ |
| II | `stats.topological_permutation_entropy()` | $\log_2 V(m) / (m-1)$ |
| III | `on.entropy.log_mean_outgrowth_ratio(sweep[m], sweep[m+1])` | $\ln G(m)$, $G$ the mean number of admissible $(m+1)$-patterns extending an admissible $m$-pattern |
| IV | `stats.network().spectral_radius(log=True)` | $\ln \rho(A)$, $A$ the binary connectivity matrix of the network |

`on.entropy.entropy_table(sweep, lyapunov=True)` returns all of them for every m, in
nats.

Methods I–III converge from above and slowly in m. Method IV is the estimator of
Sakellariou, Stemler & Small (2021); on the logistic map at r = 4 it is within 0.2 %
of ln 2 at m = 12.

## Recurrence, distances and mapping back to the series

- `on.recurrence.order_recurrence_plot(stats.sequence)`, `recurrence_rate`,
  `order_structure_matrix`.
- `on.metrics`: Hamming, Kendall τ, Spearman footrule and ρ distances between
  permutations (all normalised to [0, 1]); rank numbers.
- `on.mapping.property_per_window` / `property_per_point`: any node property as a
  series aligned with the data; `windows_on_cycles`: windows whose pattern lies on a
  closed k-walk (neighbourhoods of unstable periodic orbits);
  `pattern_value_ranges`: the 1-D projection of the ordinal partition.

These need `fit(..., keep_sequence=True)`.

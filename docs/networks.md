# Ordinal partition networks

The ordinal partition network of a series has one node per observed pattern and one
directed edge per observed transition between consecutive windows, weighted by how
often the transition occurs. Self-loops are kept. Read as a Markov chain on the
ordinal partition, it is a finite-state model of the dynamics.

```python
net = stats.network()          # OrdinalNetwork
net.W, net.A, net.T, net.Q     # scipy.sparse matrices
```

| matrix | definition |
|---|---|
| `W` | transition counts, $W_{ij} = \#(i \to j)$ |
| `A` | binary connectivity, $A_{ij} = 1$ if $W_{ij} > 0$ |
| `T` | row-stochastic transition matrix, $T_{ij} = W_{ij} / \sum_k W_{ik}$ |
| `Q` | column-normalised matrix, $Q_{ij} = W_{ij} / \sum_k W_{kj}$ |

## Editing

```python
net = net.prune(min_rel_freq=1e-4)     # drop rare transitions (and nodes left isolated)
net = net.largest_scc()                # restrict to the largest strongly connected component
net.is_irreducible(); net.components("strong")
net.subgraph(nodes); net.without_self_loops()
```

## Spectral and ergodic measures

| quantity | call | notes |
|---|---|---|
| topological entropy | `net.spectral_radius(log=True)` | $\ln \rho(A)$; `-inf` for a network without cycles |
| stationary distribution | `net.stationary_distribution()` | left Perron vector of $T$ |
| entropy rate | `net.metric_entropy_rate()` | $-\sum_i \pi_i \sum_j T_{ij} \ln T_{ij}$ |
| ordinal Lyapunov sum | `net.lyapunov_sum()` | estimate of the sum of positive Lyapunov exponents, from $Q$ (`method="T"` for the row-stochastic variant) |

All in nats. Networks up to 1500 nodes use a dense eigensolver; larger ones use
ARPACK with a fixed start vector, so results are reproducible.

## Local and global topology

| quantity | call |
|---|---|
| degree, strength | `in_degree()`, `out_degree()`, `in_strength()`, `out_strength()`, `degree_distribution()` |
| average neighbour degree / strength | `avg_neighbor_degree(weighted=)` |
| node-link entropy, hierarchical degree | `node_link_entropy()`, `hierarchical_degree(h)` |
| closed k-walks | `n_cycles(k)`, `cycle_diagonal(k)`, `cycle_nodes(k)` |
| clustering, asymmetry, assortativity | `clustering()`, `asymmetry()`, `assortativity()` |
| shortest paths | `shortest_paths(weighted=)`, `characteristic_path_length()` |
| centralities | `centrality("betweenness" | "closeness_in" | "closeness_out" | "harmonic" | "eigenvector" | "katz" | "pagerank")` |
| communities | `communities("leiden" | "leading_eigenvector" | "multilevel")` |

!!! note "Conventions to be aware of"
    - `eigenvector` and `katz` centrality follow **out-links** (a node is central if it
      leads to central nodes). networkx and igraph follow in-links; pass the transposed
      network, `OrdinalNetwork.from_matrix(net.W.T)`, for that convention.
    - `closeness_*` is not rescaled by the number of reachable nodes.
    - `characteristic_path_length` averages over connected ordered pairs only.
    - Path-based measures build dense n × n distance matrices; use them on networks
      of up to a few tens of thousands of nodes.

Betweenness, PageRank and community detection need the optional `[graph]` extra.

## Random walks and surrogates

```python
walk = net.random_walk(steps=10_000, weighted=True, seed=1)        # node indices

stats = on.OrdinalSymbolizer(5).fit(x, keep_sequence=True)
s, n_fallback = on.surrogates.network_surrogate(stats, x, length=10_000, seed=1)
```

A walk on the network is a sequence of overlapping patterns. `surrogates.reconstruct`
turns it into a series: each new pattern fixes the rank of one new point relative to
the previous m − 1, and the value is drawn from the observed values that satisfy
that constraint. Requires `tau = w = 1`.

## Export

```python
net.to_igraph(); net.to_networkx()
net.save("my_network"); on.load_network("my_network")
```

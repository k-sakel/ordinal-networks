# Coming from ordpy

[ordpy](https://github.com/arthurpessa/ordpy) (Pessa & Ribeiro, *Chaos* **31**, 063110,
2021) covers the breadth of Bandt–Pompe methods, including two-dimensional data.
`ordinal-networks` is narrower and built for a different regime: one-dimensional
series that are very long, long patterns, and the ordinal partition network as a
Markov model. Where the two overlap they give the same numbers; the test suite
checks this (`tests/test_crosscheck.py`).

`st` below is `on.OrdinalSymbolizer(m, tau).fit(x)`.

| ordpy | ordinal-networks | notes |
|---|---|---|
| `ordinal_sequence(x, dx, taux)` | `on.OrdinalSymbolizer(m, tau).encode(x)` | IDs instead of permutations; `on.decode(ids, m) - 1` gives ordpy's symbols |
| `ordinal_distribution(x, dx, taux)` | `st.pattern_ids`, `st.pattern_probabilities` | only observed patterns are stored |
| `permutation_entropy(x, dx, taux)` | `st.permutation_entropy(normalized=True)` | ordpy normalises and uses nats by default; here bits, not normalised |
| `complexity_entropy(x, dx, taux)` | `st.complexity_entropy()` | |
| `renyi_entropy(x, alpha, dx)` | `st.renyi_entropy(alpha, normalized=True)` | |
| `tsallis_entropy(x, q, dx)` | `st.tsallis_entropy(q, normalized=True)` | |
| `missing_patterns(x, dx, return_missing=False)` | `st.forbidden_fraction()` | |
| `ordinal_network(x, dx, taux, normalized=False)` | `st.src`, `st.dst`, `st.weight`; `st.network()` | node indices instead of string labels |
| `global_node_entropy(x, dx, taux)` | `st.global_node_entropy()` | |
| `missing_links(x, dx, return_missing=False)` | `st.missing_transition_fraction()` | |
| `overlapping=False` | `w = m * tau` | any slide `w ≥ 1` is allowed |

## Conventions that differ

- **Symbols.** ordpy's permutations are 0-based `argsort` results; the default
  (`"chronological"`) ranking here is the same permutation, 1-based.
- **Ties.** Both give the earlier point the smaller rank. ordpy offers
  `tie_precision` to round the data first; here ties are left as they are and counted
  (`st.n_tied_windows`).
- **Missing values.** Windows containing NaN/Inf are skipped here and break the chain
  of transitions.

## Not in ordinal-networks

Two-dimensional (image) data, weighted permutation entropy and other amplitude-aware
variants, Fisher–Shannon and the Rényi/Tsallis complexity planes, permutation
contrasts, and the smoothness–structure plane. Use ordpy for these.

## Only in ordinal-networks

Pattern lengths up to 34 and series of 10⁹ points; out-of-core results; fixed
decodable pattern IDs; topological-entropy estimators, the ordinal Lyapunov sum and
entropy rate; cycle detection; network measures; surrogates by random walk; Parquet
storage; generators for more than 35 dynamical systems.

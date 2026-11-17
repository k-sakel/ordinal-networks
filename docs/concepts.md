# Concepts and conventions

## Windows

A series $x_0, x_1, \dots, x_{N-1}$ is cut into windows of `m` points:

| parameter | meaning |
|---|---|
| `m` | pattern length, 1 ≤ m ≤ 34 |
| `tau` | embedding delay between the points of a window (default 1) |
| `w` | slide lag between the starts of consecutive windows (default 1 = maximal overlap; `w = m*tau` = no overlap; any w ≥ 1) |
| `ranking` | how a window becomes a permutation: `"chronological"` (default) or `"amplitude"` |

Window $k$ contains $x_{kw + j\tau}$ for $j = 0, \dots, m-1$. A series of $N$ points has
$\lfloor (N - (m-1)\tau - 1)/w \rfloor + 1$ windows. Transitions link consecutive
windows $k \to k+1$.

A series shorter than one window, $(m-1)\tau + 1$ points, raises `ValueError`.

## Rankings

Both rankings describe the same ordering of the window; they write it differently.

- **`"chronological"`** (Bandt & Pompe): $\pi = (p_1, \dots, p_m)$ such that
  $x_{p_1} < x_{p_2} < \dots < x_{p_m}$, i.e. the time indices sorted by value.
  This is MATLAB's `[~, s] = sort(window)` and NumPy's `argsort(window) + 1`.
- **`"amplitude"`** (Small): entry $j$ is the rank of the $j$-th point of the window in
  *descending* order of amplitude (1 = largest).

For the window `(3.0, 2.0, 4.0, 1.0)` the chronological pattern is `(4, 2, 1, 3)` and
the amplitude pattern is `(2, 3, 1, 4)`.

`patterns.convert(ids, m, "chronological", "amplitude")` translates IDs between the two.

## Pattern IDs

Each pattern has a fixed ID in $\{1, \dots, m!\}$: its 1-based lexicographic rank
among all permutations of $(1, \dots, m)$, written in the chosen ranking. ID 0 means
"no pattern" (a window that contained NaN or ±Inf).

IDs are computed directly from the data in $O(m)$ per window and are exact up to
m = 34 (34! < 2¹²⁸). They are comparable across series, chunks and parameter
values, and they decode back to the permutation:

```python
on.decode(stats.pattern_ids[:3], 12)          # 3 x 12 array of permutations (1-based)
on.encode([3, 1, 2])                          # -> 5
on.pattern_of([0.3, 0.9, 0.1])                # ID of one window -> 5
```

For m ≤ 20 IDs are `uint64` arrays. For m > 20 they are arrays of Python ints
(internally two `uint64` arrays, `id_hi` and `id_lo`).

With the default ranking the ID order equals the row order of MATLAB's
`unique(s, 'rows')`.

## Ties

If two points of a window are equal, the earlier one counts as smaller (Bandt &
Pompe). Ties are counted so that you can judge whether they matter:
`stats.n_tied_windows` (windows with at least one tie) and `stats.n_tie_pairs`.

Many ties usually mean quantised data or a floating-point orbit that has collapsed
onto a periodic point (see [Dynamical systems](systems.md)).

## Non-finite values

NaN and ±Inf are counted (`stats.n_nonfinite_points`). Windows containing them are
skipped (`stats.n_nonfinite_windows`), and no transition is recorded into or out of a
skipped window: the chain is never joined across a gap. A series whose windows are
all skipped gives an empty result with a `RuntimeWarning`.

## Units

| quantity | default unit |
|---|---|
| `permutation_entropy`, `conditional_entropy`, `renyi_entropy`, `node_link_entropy`, `topological_permutation_entropy` | **bits** (`base=2`) |
| `spectral_radius(log=True)`, `lyapunov_sum`, `metric_entropy_rate`, `global_node_entropy`, `entropy_table` | **nats** |
| `tsallis_entropy`, `statistical_complexity` | dimensionless |

Multiply bits by ln 2 to compare with nats. Functions that take `base` accept any
logarithm base.

## Results do not depend on performance settings

`threads`, `chunk_size`, `memory_budget`, `spill_dir` and the out-of-core `output=`
mode change speed and memory use only. The pattern and transition counts are
identical in every configuration; the test suite asserts this.

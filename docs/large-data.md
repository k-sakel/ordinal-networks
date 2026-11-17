# Large data

## Input

`fit`, `sweep` and `encode` accept

- any 1-D NumPy array (or a 2-D array / pandas / polars DataFrame with `column=`);
- a path to `.npy` (memory-mapped, recommended for N ~ 10⁹), `.parquet`,
  `.csv` / `.txt` / `.dat`, or `.json`.

```python
on.systems.generate_to_file("henon", "henon_x.npy", n=10**9, column=0)   # chunked, constant memory

sym = on.OrdinalSymbolizer(m=25, threads=0,          # all cores
                           chunk_size=4_000_000,      # windows per chunk
                           memory_budget=3 * 2**30,   # RAM for the counting hash tables
                           spill_dir=None)            # None = system temp dir; False = never spill
stats = sym.fit("henon_x.npy", progress=True)
```

## How it works

1. The series is read in chunks of `chunk_size` windows from the memory map.
2. Each chunk is split across threads. Within a thread the rank vector of the window
   is updated incrementally when consecutive windows overlap (drop the oldest point,
   insert the newest: $O(m)$ per window instead of a sort), and converted to the
   pattern ID through its Lehmer code.
3. Patterns and transitions are counted in per-thread hash tables, which are merged.
4. When the tables outgrow `memory_budget`, they are sorted and spilled to disk as
   runs; at the end all runs are merged with a k-way merge.

## Several pattern lengths in one pass

```python
sweep = sym.sweep("henon_x.npy", m=range(2, 31))      # dict: m -> OrdinalStatistics
on.entropy.entropy_table(sweep)                        # all estimators for every m
```

The memory budget is shared between the pattern lengths of a sweep.

## Results larger than RAM

For large m and N nearly every window is a new pattern, and the result itself (not
the input) is what does not fit. At about 24 bytes per transition and 16 bytes per
pattern, 10⁸ distinct transitions need about 4 GB.

```python
stats = sym.fit("henon_x.npy", output="HENON_m30")    # streamed to memory-mapped .npy files
stats = on.load_statistics("HENON_m30")               # reload later (memory-mapped)
sweep = sym.sweep("henon_x.npy", m=range(2, 31), output_dir="HENON_sweep")
```

## The symbol sequence

By default only frequencies are kept. To map results back onto the series, keep the
node index of every window:

```python
stats = sym.fit(x, keep_sequence=True)                 # stats.sequence, -1 = skipped window
stats = sym.fit(x, sequence_path="seq.npy")            # written to disk instead of RAM
stats.symbol_sequence                                  # pattern ID of every window
```

## Storage

```python
stats.save("LOG_ChaosR4_m25")      # LOG_ChaosR4_m25.nodes.parquet + .edges.parquet
stats = on.load_statistics("LOG_ChaosR4_m25")
```

- `*.nodes.parquet`: `id` (or `id_hi` / `id_lo` for m > 20) and `count`, sorted by ID
- `*.edges.parquet`: `src`, `dst` (node indices) and `weight` (transition count), in
  the smallest sufficient integer types, zstd-compressed
- parameters and counters as JSON in the Parquet metadata
- optional `*.sequence.npy`

Pattern frequencies and the transition edge list together define the Markov chain;
every matrix is rebuilt from them on load.

## Benchmarks

See the README for a table of timings and peak memory up to N = 10⁹, and
`benchmarks/compare.py` for a head-to-head comparison with ordpy and thread scaling.

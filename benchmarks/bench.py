"""Throughput benchmark.

    python benchmarks/bench.py --n 100000000 --m 10 25 --file data.npy

Generates a series of the chosen system on disk (if needed; default Henon x), then times a single
fit per m and a full sweep. Prints windows/second and peak RSS.
"""
import argparse
import os
import time
from pathlib import Path

import numpy as np

import ordinal_networks as on


def rss_gb():
    """Peak resident memory in GB (Linux/macOS via resource, Windows via psutil if installed)."""
    try:
        import resource
        r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return r / 1024 ** 2 if os.uname().sysname != "Darwin" else r / 1024 ** 3
    except ImportError:
        try:
            import psutil
            return psutil.Process().memory_info().peak_wset / 1024 ** 3
        except Exception:
            return float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10 ** 7)
    ap.add_argument("--m", type=int, nargs="+", default=[5, 10, 15, 20, 25, 30])
    ap.add_argument("--file", default="bench_series.npy")
    ap.add_argument("--system", default="henon")
    ap.add_argument("--column", type=int, default=0)
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--output", default=None, help="out-of-core results directory (used for m >= --ooc-from)")
    ap.add_argument("--ooc-from", type=int, default=25)
    a = ap.parse_args()
    p = Path(a.file)
    if not p.exists() or np.load(p, mmap_mode="r").shape[0] != a.n:
        t = time.perf_counter()
        on.systems.generate_to_file(a.system, p, n=a.n, column=a.column)
        print(f"generated {a.n:.0e} points in {time.perf_counter() - t:.1f}s")
    for m in a.m:
        t = time.perf_counter()
        out = None if (a.output is None or m < a.ooc_from) else Path(a.output) / f"m{m}"
        st = on.OrdinalSymbolizer(m).fit(p, output=out)
        dt = time.perf_counter() - t
        print(f"m={m:2d}: {dt:7.1f}s  {st.n_windows / dt / 1e6:6.1f} M windows/s  "
              f"V={st.n_patterns:>11,d}  E={st.n_transitions:>11,d}  tied={st.n_tied_windows}  "
              f"peak RSS {rss_gb():.2f} GB{'  (out-of-core)' if out else ''}")
    if a.sweep:
        t = time.perf_counter()
        sw = on.OrdinalSymbolizer(2).sweep(p, a.m)
        print(f"sweep over m={a.m}: {time.perf_counter() - t:.1f}s")


if __name__ == "__main__":
    main()

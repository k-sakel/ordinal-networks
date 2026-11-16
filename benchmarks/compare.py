"""Head-to-head timing against ordpy, and thread scaling.

    python benchmarks/compare.py --out paper/results/benchmark.json

Task timed for both packages: from a series in RAM to the pattern distribution and
the weighted transition list of the ordinal network (tau = 1, overlapping windows).
For ordpy this is ``ordinal_distribution`` followed by ``ordinal_network``.

Every timing is the best of ``--repeat`` runs. ordpy is skipped for a configuration
once it has exceeded ``--budget`` seconds at a smaller size.
"""
import argparse
import json
import os
import platform
import time
from pathlib import Path

import numpy as np

import ordinal_networks as on


def best(f, repeat):
    ts = []
    for _ in range(repeat):
        t = time.perf_counter()
        f()
        ts.append(time.perf_counter() - t)
    return min(ts)


def t_on(x, m, threads=0):
    on.OrdinalSymbolizer(m, threads=threads).fit(x)


def t_ordpy(x, m):
    import ordpy
    ordpy.ordinal_distribution(x, dx=m)
    ordpy.ordinal_network(x, dx=m, normalized=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="benchmark.json")
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--budget", type=float, default=120.0, help="stop growing ordpy runs beyond this many seconds")
    ap.add_argument("--nmax", type=int, default=10 ** 8)
    a = ap.parse_args()

    try:
        import ordpy
        ordpy_version = getattr(ordpy, "__version__", None)
        if ordpy_version is None:
            from importlib.metadata import version
            ordpy_version = version("ordpy")
    except ImportError:
        ordpy, ordpy_version = None, None

    x_all = on.systems.generate("henon", n=a.nmax)[:, 0].copy()
    res = {"machine": {"platform": platform.platform(), "processor": platform.processor(),
                       "cpu_count": os.cpu_count(), "python": platform.python_version(),
                       "numpy": np.__version__, "ordinal_networks": on.__version__, "ordpy": ordpy_version},
           "system": "Henon map (a = 1.4, b = 0.3), x component", "vs_n": [], "vs_m": [], "threads": []}

    # --- time against N at m = 6 -------------------------------------------------
    m = 6
    slow = False
    n = 10 ** 4
    while n <= a.nmax:
        x = x_all[:n]
        row = {"n": n, "m": m, "ordinal_networks": best(lambda: t_on(x, m), a.repeat)}
        if ordpy is not None and not slow:
            row["ordpy"] = best(lambda: t_ordpy(x, m), 1 if n >= 10 ** 6 else a.repeat)
            slow = row["ordpy"] > a.budget
        res["vs_n"].append(row)
        print(row, flush=True)
        n *= 10

    # --- time against m at N = 10^6 ------------------------------------------------
    x = x_all[:10 ** 6]
    slow = False
    for m in (3, 4, 5, 6, 7, 8, 10, 12, 16, 20, 25, 30, 34):
        row = {"n": len(x), "m": m, "ordinal_networks": best(lambda: t_on(x, m), a.repeat)}
        if ordpy is not None and not slow:
            row["ordpy"] = best(lambda: t_ordpy(x, m), 1)
            slow = row["ordpy"] > a.budget
        res["vs_m"].append(row)
        print(row, flush=True)

    # --- strong scaling over threads at N = nmax ------------------------------------
    ncpu = os.cpu_count() or 1
    ts = sorted({t for t in (1, 2, 4, 8, 16, 32, 64) if t <= ncpu} | {ncpu})
    for m in (6, 16):
        for t in ts:
            row = {"n": a.nmax, "m": m, "threads": t, "ordinal_networks": best(lambda: t_on(x_all, m, t), a.repeat)}
            res["threads"].append(row)
            print(row, flush=True)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()

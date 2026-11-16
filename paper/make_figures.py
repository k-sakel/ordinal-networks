"""Regenerate every figure and number of the manuscript.

    python paper/make_figures.py                 # all figures, reduced sizes (a few minutes)
    python paper/make_figures.py --full          # sizes used in the manuscript
    python paper/make_figures.py --only scale    # one item: quickstart | entropy | agreement | benchmark | scale | collapse

Figures go to ``paper/figures/*.pdf`` (and ``.png``), numbers to ``paper/results/*.json``.
The benchmark figure is drawn from ``paper/results/benchmark.json``, written by
``python benchmarks/compare.py --out paper/results/benchmark.json``.
"""
import argparse
import json
import math
import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import ordinal_networks as on  # noqa: E402

HERE = Path(__file__).parent
FIG = HERE / "figures"
RES = HERE / "results"
LN2 = math.log(2)
H_HENON = 0.4651          # topological entropy of the Henon map (a = 1.4, b = 0.3), Newhouse & Pignataro 1993

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "legend.frameon": False, "figure.dpi": 150, "savefig.bbox": "tight"})


def save(fig, name):
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png", dpi=200)
    plt.close(fig)
    print("saved", name)


def dump(obj, name):
    RES.mkdir(exist_ok=True)
    with open(RES / f"{name}.json", "w") as f:
        json.dump(obj, f, indent=1)


# ------------------------------------------------------------------ Fig. entropy --

def fig_entropy(full):
    """Methods I-IV and the ordinal Lyapunov estimate on the logistic map, r = 4."""
    n = 10 ** 7
    mmax = 20 if full else 16
    t = time.perf_counter()
    x = on.systems.generate("logistic", "ChaosR4", n=n)
    sweep = on.OrdinalSymbolizer(2).sweep(x, range(2, mmax + 2))
    t_sym = time.perf_counter() - t
    tab = on.entropy.entropy_table(sweep, spectral=True, lyapunov=True)
    gamma = on.entropy.growth_rate(sweep, range(4, 15))
    ms = tab["m"][:-1]
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    ax.axhline(LN2, color="0.6", lw=0.8)
    ax.text(ms[-1], LN2 - 0.012, r"$\ln 2$", ha="right", va="top", color="0.4")
    ax.axhline(gamma, color="C3", ls="--", lw=0.8, label="I: growth rate of $V$")
    ax.plot(ms, tab["TPE"][:-1], "d-", ms=3, lw=0.8, label="II: $\\ln V/(m-1)$")
    ax.plot(ms, tab["lnG"][:-1], "s-", ms=3, lw=0.8, label="III: $\\ln G(m)$")
    ax.plot(ms, tab["ln_rho"][:-1], "o-", ms=3, lw=0.8, label="IV: $\\ln\\rho(A)$")
    ax.plot(ms, tab["lyap"][:-1], "^-", ms=3, lw=0.8, color="0.3", label="ordinal Lyapunov")
    ax.set_xlabel("pattern length $m$")
    ax.set_ylabel("estimate (nats)")
    ax.set_ylim(0.45, 1.0)
    ax.legend(fontsize=7)
    save(fig, "entropy_logistic")
    dump({"n": n, "seconds_symbolise_sweep": t_sym, "growth_rate_m4_14": gamma,
          **{k: np.asarray(v).tolist() for k, v in tab.items()}}, "entropy_logistic")


# ---------------------------------------------------------------- Fig. agreement --

def fig_agreement(full):
    """Permutation entropy along the logistic bifurcation diagram and the
    complexity-entropy plane; ordpy values overlaid where it is installed."""
    try:
        import ordpy
    except ImportError:
        ordpy = None
    m, n = 6, 10 ** 5 if full else 2 * 10 ** 4
    rs = np.linspace(3.5, 4.0, 501 if full else 201)
    H = np.empty(len(rs)); C = np.empty(len(rs)); Ho = np.full(len(rs), np.nan); Co = np.full(len(rs), np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for i, r in enumerate(rs):
            x = on.systems.generate("logistic", params=[r], n=n)
            H[i], C[i] = on.OrdinalSymbolizer(m).fit(x).complexity_entropy()
            if ordpy is not None and i % 10 == 0:
                Ho[i], Co[i] = ordpy.complexity_entropy(x, dx=m)
    rng = np.random.default_rng(0)
    extra = {
        "white noise": rng.standard_normal(n),
        "random walk": np.cumsum(rng.standard_normal(n)),
        "Hénon": on.systems.generate("henon", n=n)[:, 0],
        "Lorenz $x$": on.systems.generate("lorenz", n=n, every=10)[:, 0],
        "Mackey–Glass": on.systems.generate("mackey_glass", n=n, every=20),
    }
    pts = {k: on.OrdinalSymbolizer(m).fit(v).complexity_entropy() for k, v in extra.items()}
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.8, 2.6))
    a.plot(rs, H, lw=0.6, color="C0", label="ordinal-networks")
    ok = np.isfinite(Ho)
    if ok.any():
        a.plot(rs[ok], Ho[ok], "o", ms=2.5, mfc="none", mew=0.6, color="C1", label="ordpy")
    a.set_xlabel("$r$"); a.set_ylabel(f"normalised permutation entropy ($m={m}$)"); a.legend(fontsize=7)
    b.scatter(H, C, s=2, c=rs, cmap="viridis", lw=0)
    if ok.any():
        b.plot(Ho[ok], Co[ok], "o", ms=2.5, mfc="none", mew=0.5, color="C1")
    for k, (h, c) in pts.items():
        b.plot(h, c, "k*", ms=5)
        b.annotate(k, (h, c), textcoords="offset points", xytext=(4, 3), fontsize=6.5)
    b.set_xlim(0, 1.02); b.set_ylim(0, 0.55)
    b.set_xlabel("normalised permutation entropy $H$"); b.set_ylabel("statistical complexity $C$")
    fig.tight_layout()
    save(fig, "agreement_ordpy")
    dump({"m": m, "n": n, "max_abs_diff_H": float(np.nanmax(np.abs(H - Ho))) if ok.any() else None,
          "max_abs_diff_C": float(np.nanmax(np.abs(C - Co))) if ok.any() else None,
          "points": {k: list(v) for k, v in pts.items()}}, "agreement_ordpy")


# ---------------------------------------------------------------- Fig. benchmark --

def fig_benchmark(full):
    p = RES / "benchmark.json"
    if not p.exists():
        print("skipping benchmark figure: run benchmarks/compare.py --out paper/results/benchmark.json first")
        return
    with open(p) as f:
        r = json.load(f)
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(6.8, 2.4))
    for key, lab, sty in (("ordpy", "ordpy", "s--"), ("ordinal_networks", "ordinal-networks", "o-")):
        v = [(q["n"], q[key]) for q in r["vs_n"] if key in q]
        a.loglog(*zip(*v), sty, ms=3, lw=0.8, label=lab)
        v = [(q["m"], q[key]) for q in r["vs_m"] if key in q]
        b.semilogy(*zip(*v), sty, ms=3, lw=0.8, label=lab)
    a.set_xlabel("series length $N$"); a.set_ylabel("time (s)"); a.set_title(f"$m = {r['vs_n'][0]['m']}$", fontsize=8)
    a.legend(fontsize=7)
    b.set_xlabel("pattern length $m$"); b.set_title(f"$N = 10^{{{round(math.log10(r['vs_m'][0]['n']))}}}$", fontsize=8)
    for m in sorted({q["m"] for q in r["threads"]}):
        v = [(q["threads"], q["ordinal_networks"]) for q in r["threads"] if q["m"] == m]
        t1 = v[0][1]
        c.plot([q[0] for q in v], [t1 / q[1] for q in v], "o-", ms=3, lw=0.8, label=f"$m = {m}$")
    tmax = max(q["threads"] for q in r["threads"])
    c.plot([1, tmax], [1, tmax], color="0.7", lw=0.6)
    c.set_xlabel("threads"); c.set_ylabel("speed-up")
    c.set_title(f"$N = 10^{{{round(math.log10(r['threads'][0]['n']))}}}$", fontsize=8); c.legend(fontsize=7)
    fig.tight_layout()
    save(fig, "benchmark")


# -------------------------------------------------------------------- Fig. scale --

def fig_scale(full, force=False):
    """What scale buys: convergence of ln rho(A) for the Henon map as N and m grow."""
    nmax = 10 ** 9 if full else 10 ** 8
    mmax = 24 if full else 22
    path = HERE / "data" / f"henon_x_{nmax:.0e}.npy"
    path.parent.mkdir(exist_ok=True)
    if not path.exists():
        on.systems.generate_to_file("henon", path, n=nmax, column=0)
    ns = [10 ** k for k in range(5, round(math.log10(nmax)) + 1)]
    cache = RES / "scale_henon.json"
    out = None
    if cache.exists() and not force:
        with open(cache) as f:
            out = json.load(f)
        if out.get("mmax") != mmax or [r["n"] for r in out["runs"]] != ns:
            out = None
        else:
            print("scale: using cached results (pass --force to recompute; the eigenvalues take about 40 minutes)")
    if out is None:
        out = _scale_compute(path, nmax, mmax, ns)
        dump(out, "scale_henon")
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.8, 2.6))
    for i, run in enumerate(out["runs"]):
        n, ms, h, V = run["n"], run["m"], run["ln_rho"], run["V"]
        col = plt.cm.viridis(i / max(len(ns) - 1, 1))
        lab = f"$N = 10^{{{round(math.log10(n))}}}$"
        a.plot(ms, h, "o-", ms=2.5, lw=0.8, color=col, label=lab)
        b.semilogy(ms, V, "o-", ms=2.5, lw=0.8, color=col, label=lab)
    a.axhline(H_HENON, color="0.6", lw=0.8)
    a.text(2, H_HENON - 0.01, "$h_{top}$ = 0.4651", color="0.4", fontsize=7, va="top")
    a.set_ylim(0.15, 0.9)
    a.set_xlabel("pattern length $m$"); a.set_ylabel(r"$\ln\rho(A)$ (nats)"); a.legend(fontsize=7, ncol=2)
    mm = np.arange(2, mmax + 1)
    b.semilogy(mm, np.exp(H_HENON * mm) * 2.0, color="0.6", lw=0.8)
    b.text(mm[-1], np.exp(H_HENON * mm[-1]) * 1.2, r"$\propto e^{h m}$", color="0.4", fontsize=7, ha="right", va="top")
    b.set_xlabel("pattern length $m$"); b.set_ylabel("observed patterns $V(m)$")
    fig.tight_layout()
    save(fig, "scale_henon")


def _scale_compute(path, nmax, mmax, ns):
    x = np.load(path, mmap_mode="r")
    out = {"mmax": mmax, "runs": []}
    for n in ns:
        t = time.perf_counter()
        sweep = on.OrdinalSymbolizer(2).sweep(x[:n] if n < nmax else path, range(2, mmax + 1))
        t_sym = time.perf_counter() - t
        ms = sorted(sweep)
        V = [sweep[m].n_patterns for m in ms]
        E = [sweep[m].n_transitions for m in ms]
        t = time.perf_counter()
        h = [sweep[m].network().spectral_radius(log=True) for m in ms]
        t_eig = time.perf_counter() - t
        tied = max(sweep[m].n_tied_windows for m in ms)
        out["runs"].append({"n": n, "m": ms, "V": V, "E": E, "ln_rho": h, "seconds_sweep": t_sym,
                            "seconds_eig": t_eig, "max_tied_windows": tied})
        print(f"N={n:.0e}: sweep {t_sym:.1f}s, eigenvalues {t_eig:.1f}s, ln rho(m={ms[-1]}) = {h[-1]:.4f}", flush=True)
    return out


# ----------------------------------------------------------------- Fig. collapse --

def fig_collapse(full):
    """Floating-point collapse of the logistic orbit at r = 4, and its ordinal signature."""
    n = 2 * 10 ** 7
    with warnings.catch_warnings(record=True) as wlist:
        warnings.simplefilter("always")
        x = on.systems.generate("logistic", "ChaosR4", n=n)
    same = np.flatnonzero(x[1:] == x[:-1])
    n_c = int(same[0]) if len(same) else None
    m = 8
    ns = np.unique(np.round(np.logspace(5, math.log10(n), 25)).astype(int))
    tied = []; h = []; pe = []
    for k in ns:
        st = on.OrdinalSymbolizer(m).fit(x[:k])
        tied.append(st.n_tied_windows / st.n_windows)
        h.append(st.network().spectral_radius(log=True))
        pe.append(st.permutation_entropy(base=math.e) / (m - 1))
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.8, 2.4))
    if n_c is not None:
        lo, hi = max(n_c - 40, 0), n_c + 15
        a.plot(np.arange(lo, hi) - n_c, x[lo:hi], ".-", ms=3, lw=0.6)
        a.axvline(0, color="C3", lw=0.6)
    a.set_xlabel(f"$n - n_c$   ($n_c$ = {n_c:,})" if n_c else "$n$"); a.set_ylabel("$x_n$")
    b.semilogx(ns, pe, "o-", ms=2.5, lw=0.8, label=f"$H/(m-1)$, $m={m}$")
    b.set_ylim(0.6, 0.9)
    b.semilogx(ns, h, "s-", ms=2.5, lw=0.8, label=r"$\ln\rho(A)$")
    b.axhline(LN2, color="0.6", lw=0.8)
    if n_c is not None:
        b.axvline(n_c, color="C3", lw=0.6)
    b.set_xlabel("series length $N$"); b.set_ylabel("estimate (nats)"); b.legend(fontsize=7)
    fig.tight_layout()
    save(fig, "collapse_logistic")
    dump({"n_collapse": n_c, "x_collapse": float(x[n_c + 1]) if n_c is not None else None,
          "warning": str(wlist[0].message) if wlist else None, "m": m, "n": ns.tolist(), "tied_fraction": tied,
          "ln_rho": h, "pe_rate": pe}, "collapse_logistic")


def quickstart(full):
    """Output and timing of the quick-start listing (no figure)."""
    x = on.systems.generate("logistic", "ChaosR4", n=10 ** 7)
    ts = []
    for _ in range(3):
        t = time.perf_counter()
        st = on.OrdinalSymbolizer(m=12).fit(x)
        ts.append(time.perf_counter() - t)
    net = st.network()
    dump({"seconds": min(ts), "n_patterns": st.n_patterns, "n_transitions": st.n_transitions,
          "ln_rho": net.spectral_radius(log=True), "lyapunov_sum": net.lyapunov_sum()}, "quickstart")


FIGS = {"quickstart": quickstart, "entropy": fig_entropy, "agreement": fig_agreement, "benchmark": fig_benchmark, "scale": fig_scale,
        "collapse": fig_collapse}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--only", choices=sorted(FIGS), nargs="+")
    ap.add_argument("--force", action="store_true", help="recompute the cached results of the scale figure")
    a = ap.parse_args()
    for name in a.only or FIGS:
        t0 = time.perf_counter()
        if name == "scale":
            fig_scale(a.full, a.force)
        else:
            FIGS[name](a.full)
        print(f"[{name}] {time.perf_counter() - t0:.1f}s", flush=True)

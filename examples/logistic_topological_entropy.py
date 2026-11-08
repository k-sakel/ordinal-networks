"""Reproduce Fig. 3 of Sakellariou, Stemler & Small, "Estimating topological
entropy using ordinal partition networks": Methods I-IV on the logistic map
(r = 4, N = 10^7), plus the ordinal Lyapunov estimate (LyapunovSum_OrdNet).

    python examples/logistic_topological_entropy.py [--n 10000000] [--plot]
"""
import argparse
import math
import time

import ordinal_networks as on

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=10 ** 7)
ap.add_argument("--mmax", type=int, default=20)
ap.add_argument("--plot", action="store_true")
a = ap.parse_args()

t0 = time.perf_counter()
x = on.systems.generate("logistic", "ChaosR4", n=a.n)
sweep = on.OrdinalSymbolizer(2).sweep(x, range(2, a.mmax + 2))
print(f"symbolised {a.n:.0e} points for m = 2..{a.mmax + 1} in {time.perf_counter() - t0:.1f}s")

tab = on.entropy.entropy_table(sweep, spectral=True, lyapunov=True)
gamma = on.entropy.growth_rate(sweep, range(4, 15))
print(f"Method I (growth rate of V, m=4..14): {gamma:.4f}")
print(" m        V        E   TPE*ln2    lnG   ln rho   lyap")
for i, m in enumerate(tab["m"][:-1]):
    print(f"{m:2d} {tab['V'][i]:8d} {tab['E'][i]:8d}  {tab['TPE'][i]:.4f}  {tab['lnG'][i]:.4f}  "
          f"{tab['ln_rho'][i]:.4f}  {tab['lyap'][i]:.4f}")
print(f"exact: h_top = lambda = ln 2 = {math.log(2):.4f}")

if a.plot:
    import matplotlib.pyplot as plt
    ms = tab["m"][:-1]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axhline(math.log(2), color="0.5", lw=1, label="ln 2")
    ax.axhline(gamma, color="C3", ls="--", lw=1, label="I: growth rate")
    ax.plot(ms, tab["TPE"][:-1], "d-", label="II: TPE")
    ax.plot(ms, tab["lnG"][:-1], "s-", label="III: ln G(m)")
    ax.plot(ms, tab["ln_rho"][:-1], "o-", label=r"IV: ln $\rho(A)$")
    ax.plot(ms, tab["lyap"][:-1], "^-", label="ordinal Lyapunov")
    ax.set_xlabel("pattern length m")
    ax.set_ylabel("estimate")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig("logistic_topological_entropy.png", dpi=150)
    print("saved logistic_topological_entropy.png")

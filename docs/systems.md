# Dynamical systems

`on.systems` provides fast generators (Rust) for the test systems used throughout
the ordinal-network literature, with named parameter regimes.

```python
x  = on.systems.generate("logistic", n=10**6)                       # regime "ChaosR4"
xy = on.systems.generate("henon", "BenedicksCarleson", n=10**5)     # (n, 2) array
P  = on.systems.poincare("lorenz", n=10**5)                         # z = 27 section, dz/dt > 0
on.systems.generate_to_file("henon", "henon.npy", n=10**9, column=0)

on.systems.MAPS["gauss"].regimes         # named parameter sets
on.systems.FLOWS["lorenz"].section       # default Poincare section
```

- **Maps:** logistic, cubic, tent, dyadic, bent baker, Gauss, piecewise-linear, Hénon,
  Lozi, Ikeda, Holmes, Tinkerbell, Gingerbread man, area-preserving Hénon, Chirikov,
  Arnold cat, baker's, stiletto, Zaslavsky, solenoid, generalised Hénon, folded towel.
- **Flows** (fixed-step RK4): Lorenz, Rössler, 4-D Rössler, Chua, Muthuswamy,
  Rabinovich–Fabrikant, Duffing, Van der Pol, Reiterer (9-D), ABC, double gyre (three
  variants), Bickley jet.
- **Delay equation:** Mackey–Glass.

`transient` defaults to 10⁴ iterates for maps and 10⁵ steps for flows. `seed`
perturbs the default initial condition; `x0` and `params` override it.

## Poincaré sections

`poincare(name, n, normal=, offset=, direction=)` returns n crossings of the plane
`normal · x = offset`. The crossing inside each RK4 step is located by bisection on
the cubic Hermite interpolant. If fewer than n crossings are found within
`max_steps`, a `ValueError` is raised instead of integrating forever.

## Reference Lyapunov exponents

```python
on.systems.lyapunov_spectrum("henon", xy)                # QR (Benettin) iteration on Jacobians
on.systems.lyapunov_sum_jacobian("logistic", x)          # <ln |det J|>
on.systems.grid_lyapunov_1d("logistic", x, nbins=1000)   # Ulam / grid estimate
```

## Floating-point orbits of very long series

!!! warning
    In double precision the logistic map at r = 4 started from x₀ = 0.4 (after 10⁴
    transient steps) lands *exactly* on the fixed point 0 at n ≈ 1.55 × 10⁷; every
    later value is constant. Maps with exact binary arithmetic (tent, dyadic at μ = 2)
    collapse after about 53 iterations, which is why their default is μ = 2 − 10⁻¹².

`generate` and `generate_to_file` warn when a trajectory collapses, and
`stats.n_tied_windows` exposes the collapse in any series. For N ≫ 10⁷ use a system
whose orbits stay aperiodic for longer, such as the Hénon map.

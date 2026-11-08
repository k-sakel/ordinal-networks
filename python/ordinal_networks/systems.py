"""Test dynamical systems (ported from the MATLAB ``dynamical-systems`` toolbox).

>>> x = generate("logistic", n=10**6)                    # regime "ChaosR4"
>>> xy = generate("henon", "BenedicksCarleson", n=10**5)
>>> P = poincare("lorenz", n=10**5)                      # z = 27 section, dz/dt > 0
>>> generate_to_file("logistic", "log_r4.npy", n=10**9)  # chunked, for huge N

Equations, parameter order and named regimes follow the ``*MapTimeSeries``,
``*_eqns`` and ``*DynamicalRegime`` MATLAB files. Iteration/integration runs
in Rust (fixed-step RK4 for flows).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import _core

PI = math.pi


@dataclass(frozen=True)
class System:
    name: str
    kind: str                     # "map" | "flow" | "dde"
    dim: int
    params: tuple[str, ...]
    regimes: dict = field(default_factory=dict)
    default_regime: str | None = None
    x0: tuple = ()
    dt: float = 0.01              # flows: integration step
    section: tuple | None = None  # flows: (normal, offset, direction) default Poincare section
    note: str = ""


def _r(**kw):
    return {k: tuple(v) if isinstance(v, (list, tuple)) else (v,) for k, v in kw.items()}


MAPS: dict[str, System] = {s.name: s for s in [
    System("logistic", "map", 1, ("r",), _r(
        StableOrigin=0.5, StableNode=1.5, StableNodeOscillations=2.8, StableNodeSlowOscillations=3.0,
        Period2=3.4, Period4=3.5, Period8=3.56, Period16=3.566, Period32=3.5695, Period3=1 + math.sqrt(8),
        PeriodInf=3.57, Chaos1=3.6, Chaos2=3.7, Chaos3=3.8, Chaos4=3.9, ChaosR4=4.0), "ChaosR4", (0.4,)),
    System("cubic", "map", 1, ("a",), _r(
        StableNode=2.5, Period2=3.1, Period4=3.26, Period8=3.295, PreP3Chaos=3.42, PostP3Chaos=3.49,
        PreMergeChaos=3.58, MergeChaos=1 + 0.5 * math.sqrt(27), PostMergeChaos=3.65, PostMergeChaos2=3.78,
        PostMergeChaos3=3.9, ChaosA4=4.0), "ChaosA4", (0.3,), note="x' = (a-1)x - a x^3"),
    System("tent", "map", 1, ("mu",), _r(Chaos=2 - 1e-12), "Chaos", (0.4,),
           note="mu = 2 - 1e-12 avoids the floating-point collapse of the exact tent map"),
    System("dyadic", "map", 1, ("mu",), _r(Chaos=2 - 1e-12), "Chaos", (0.4,), note="x' = mu x mod 1"),
    System("bent_baker", "map", 1, (), _r(Chaos=()), "Chaos", (0.4,)),
    System("gauss", "map", 1, ("alpha", "beta"), _r(
        Period2=(4.7, -0.7), Period4=(4.7, -0.6), Period8=(4.7, -0.525), Period16=(4.7, -0.51),
        Period32s=(4.7, -0.5), Period32=(4.7, -0.495), Mouse=(4.9, -0.48), Chaos=(6.2, -0.486),
        MouseOrbMild=(4.9, -0.54), MouseOrbWild=(4.9, -0.436), RatOrbMild=(6.2, -0.66),
        RatOrbWild=(6.2, -0.38)), "RatOrbWild", (0.1,)),
    System("piecewise_linear", "map", 1, (), _r(Default=()), "Default", (0.3,), note="x' = x log(1/x)"),
    System("henon", "map", 2, ("a", "b"), _r(Chaos=(1.4, 0.3), BenedicksCarleson=(1.96, 0.02)), "Chaos", (0.1, 0.1)),
    System("lozi", "map", 2, ("a", "b"), _r(Chaos=(6 / 5, -2 / 15)), "Chaos", (0.1, 0.1),
           note="MATLAB form (x1, x2) -> (x2, 1 + b x1 - a|x2|)"),
    System("ikeda", "map", 2, ("mu",), _r(Chaos=0.9), "Chaos", (0.1, 0.1)),
    System("holmes", "map", 2, ("a", "b"), _r(Chaos=(2.75, 0.2)), "Chaos", (0.1, 0.1)),
    System("tinkerbell", "map", 2, ("a", "b", "c", "d"), _r(
        Chaos=(0.9, -0.6013, 2.0, 0.5), CircularChaos=(0.3, 0.6, 2.0, 0.27)), "Chaos", (-0.72, -0.64)),
    System("gingerbread_man", "map", 2, (), _r(Chaos=()), "Chaos", (0.5, 3.7)),
    System("henon_area", "map", 2, ("alpha",), _r(
        StarChaos=PI / 2, CircleChaos=0.8, Chaos=math.acos(-0.95), FiveIslands=math.acos(0.22),
        ChaosEmerge=math.acos(0.4), StrongRecurrence=math.acos(0.24)), "CircleChaos", (0.1, 0.1)),
    System("chirikov", "map", 2, ("K",), _r(
        Flat=0.0, PeriodicFlat=0.05, Eye=0.25, WideEye=0.4, ChaosOnset=0.6, QuasiPeriodicChaos=0.971635,
        Islands=1.2, Chaos=2.1), "Chaos", (1.0, 2.0)),
    System("arnold_cat", "map", 2, (), _r(Chaos=()), "Chaos", (0.1234, 0.5678)),
    System("bakers", "map", 2, (), _r(Chaos=()), "Chaos", (0.1234, 0.5678)),
    System("stiletto", "map", 2, (), _r(Chaos=()), "Chaos", (0.1, 0.1)),
    System("zaslavsky", "map", 2, ("Omega", "alpha", "Gamma", "epsilon"), {}, None, (0.1, 0.1),
           note="the MATLAB regime file has no values; pass params explicitly"),
    System("solenoid", "map", 3, (), _r(Chaos=()), "Chaos", (0.1, 0.1, 0.3), note="state (x, y, theta)"),
    System("general_henon", "map", 3, ("a", "b"), _r(HyperChaos=(1.9, 0.03)), "HyperChaos", (0.1, 0.1, 0.1)),
    System("folded_towel", "map", 3, ("a", "b"), _r(HyperChaos=(3.8, 0.2)), "HyperChaos", (0.1, 0.1, 0.1)),
]}

_LORENZ_SECTION = ((0.0, 0.0, 1.0), 27.0, 1.0)

FLOWS: dict[str, System] = {s.name: s for s in [
    System("lorenz", "flow", 3, ("sigma", "rho", "beta"), _r(
        Butterfly=(10, 28, 8 / 3), TorusKnot=(10, 99.96, 8 / 3), Periodic=(100, 200, 8 / 3),
        Chaotic=(25, 200, 8 / 3), R240SPO=(10, 240, 8 / 3)), "Butterfly", (1.0, 1.0, 20.0), 0.005,
        _LORENZ_SECTION),
    System("rossler", "flow", 3, ("a", "b", "c"), _r(
        Original=(0.2, 0.2, 5.7), Alternative=(0.1, 0.1, 14.0), Periodic=(0.3, 2, 4), Period2=(0.35, 2, 4),
        Period3=(0.41, 2, 4), Period4=(0.377, 2, 4), Period5=(0.4, 2, 4), Period6=(0.39067, 2, 4),
        Period8=(0.3848, 2, 4), FourBandChaos=(0.386, 2, 4), BroadBandChaos=(0.398, 2, 4)),
        "BroadBandChaos", (1.0, 1.0, 0.0), 0.01, ((1.0, 0.0, 0.0), 0.0, 1.0)),
    System("rossler4d", "flow", 4, ("a", "b", "c", "d"), _r(HyperChaos=(0.25, 3, 0.5, 0.05)), "HyperChaos",
           (-10.0, -6.0, 0.0, 10.0), 0.01, ((1.0, 0.0, 0.0, 0.0), 0.0, 1.0)),
    System("chua", "flow", 3, ("alpha", "beta", "m0", "m1", "E"), {
        k: (15.6, b, -8 / 7, -5 / 7, 1.0) for k, b in dict(
            Periodx2=50, Period2x2=35, Period4x2=33.8, MildChaosx2=33.6, WildChaosx2=33, WildChaosMerge=25.58).items()
    } | {"DoubleScroll": (9.0, 14 + 2 / 7, -8 / 7, -5 / 7, 1.0), "Xu2008": (9.4, 16.0, -8 / 7, -5 / 7, 1.0)},
        "DoubleScroll", (0.1, 0.0, 0.0), 0.01, ((0.0, 1.0, 0.0), 0.0, 1.0)),
    System("muthuswamy", "flow", 3, ("C", "L", "alpha", "beta"), _r(SingleScroll=(1, 3, 0.6, 1.5)), "SingleScroll",
           (0.1, 0.0, 0.0), 0.01, ((1.0, 0.0, 0.0), 0.0, 1.0)),
    System("rabinovich_fabrikant", "flow", 3, ("alpha", "gamma"), _r(Chaos=(1.1, 0.87)), "Chaos",
           (-1.0, 0.0, 0.5), 0.001, ((1.0, 0.0, 0.0), 0.0, 1.0)),
    System("duffing", "flow", 3, ("alpha", "beta", "gamma", "delta", "omega"), _r(
        PeriodicallyForced=(1, -1, 0.3, 0.2, 1), WeaklyNonlinear=(0.1, 1, 0, 0, 0)), "PeriodicallyForced",
        (1.0, 0.0, 0.0), 0.01, None, note="state (x, dx/dt, phase); use stroboscopic sampling"),
    System("vanderpol", "flow", 3, ("mu", "A", "omega"), _r(Chaos=(8.53, 1.2, 2 * PI / 10)), "Chaos",
           (1.0, 0.0, 0.0), 0.01, None, note="state (x, dx/dt, phase)"),
    System("reiterer", "flow", 9, ("a", "sigma", "R"), {
        k: (0.5, 0.5, R) for k, R in dict(LimitCycle=13.1, Period2=14.1, Period4=14.14, Period8=14.164,
                                          PeriodInf=14.17, SingleScroll=14.22, DoubleLeaf=14.30, PeriodOdd=15.0,
                                          Chaos=24.0, HyperChaos=45.0).items()}, "HyperChaos",
        (0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09), 0.001, None),
    System("abc", "flow", 3, ("A", "B", "C"), _r(Turbulence=(math.sqrt(3), math.sqrt(2), 1.0)), "Turbulence",
           (0.1, 0.2, 0.3), 0.01, None),
    System("double_gyre", "flow", 3, ("A", "epsilon", "omega"), _r(Turbulence=(0.25, 0.25, 2 * PI)), "Turbulence",
           (0.5, 0.5, 0.0), 0.01, None, note="state (x, y, t)"),
    System("double_gyre_auto", "flow", 2, (), _r(Default=()), "Default", (0.3, 0.4), 0.01, None),
    System("double_gyre_roto", "flow", 3, (), _r(Default=()), "Default", (0.3, 0.4, 0.0), 0.01, None,
           note="state (x, y, t)"),
    System("bickley", "flow", 3, ("U0", "L0", "A1", "A2", "A3", "k1", "k2", "k3", "c1", "c2", "c3"), _r(
        Turbulence=(62.66, 1770000.0, 0.0075, 0.15, 0.3, 2 / 6371000, 4 / 6371000, 6 / 6371000,
                    0.1446 * 62.66, 0.205 * 62.66, 0.461 * 62.66)), "Turbulence", (0.0, 1.0e6, 0.0), 1000.0, None,
        note="state (x, y, t), SI units"),
    System("mackey_glass", "dde", 1, ("beta", "gamma", "n", "tau", "alpha"), _r(
        LimitCycle=(0.2, 0.1, 10, 6, 1), Chaos=(0.2, 0.1, 10, 20, 1), Xu2008=(0.2, 0.1, 10, 17, 10)), "Chaos",
        (1.2,), 0.05, None),
]}

SYSTEMS = MAPS | FLOWS


def get_system(name: str) -> System:
    flat = name.lower().replace("-", "").replace(" ", "").replace("_", "")
    lookup = {k.replace("_", ""): k for k in SYSTEMS}
    lookup.update({"roessler": "rossler", "roessler4d": "rossler4d", "bitshift": "dyadic",
                   "henonareapreserv": "henon_area", "standardmap": "chirikov"})
    key = lookup.get(flat, flat)
    if key not in SYSTEMS:
        raise KeyError(f"unknown system {name!r}; available: {sorted(SYSTEMS)}")
    return SYSTEMS[key]


def parameters(name: str, regime: str | None = None) -> tuple:
    s = get_system(name)
    regime = regime or s.default_regime
    if regime is None:
        raise ValueError(f"{s.name} has no default regime; pass params=...")
    if regime not in s.regimes:
        raise KeyError(f"{s.name}: unknown regime {regime!r}; available: {sorted(s.regimes)}")
    return s.regimes[regime]


def _x0(s: System, x0, seed):
    if x0 is not None:
        return [float(v) for v in np.atleast_1d(x0)]
    base = np.array(s.x0, dtype=float)
    if seed is not None:
        base = base + 1e-3 * np.random.default_rng(seed).standard_normal(base.shape)
    return base.tolist()


def _check_collapse(out: np.ndarray, name: str, offset: int = 0):
    """Warn if a map trajectory has collapsed onto an exact fixed point
    (a floating-point artefact, e.g. the logistic map hitting x = 0 or 3/4)."""
    if out.shape[0] < 2:
        return
    same = np.all(out[1:] == out[:-1], axis=1)
    if same.any():
        i = int(np.argmax(same)) + offset
        import warnings
        warnings.warn(f"{name}: trajectory collapsed onto a fixed point at n = {i} (floating-point "
                      f"artefact); later values are constant. Use a shorter series, another x0, or "
                      f"perturbed parameters (cf. tent/dyadic mu = 2 - 1e-12).", RuntimeWarning, stacklevel=3)
        return True
    return False


def generate(name: str, regime: str | None = None, n: int = 100_000, *, params=None, x0=None,
             transient: int | None = None, dt: float | None = None, every: int = 1, seed: int | None = None,
             squeeze: bool = True) -> np.ndarray:
    """Trajectory of a map (n iterates) or flow (n samples every ``every`` RK4 steps of ``dt``).

    ``transient`` defaults to 10^4 iterates (maps) or 10^5 steps (flows).
    1-D systems return a 1-D array when ``squeeze``.
    """
    s = get_system(name)
    p = [float(v) for v in (parameters(s.name, regime) if params is None else params)]
    x = _x0(s, x0, seed)
    if s.kind == "map":
        out, _ = _core.iterate_map(s.name, p, x, int(n), int(10_000 if transient is None else transient))
        _check_collapse(out, s.name)
    elif s.kind == "flow":
        out, _ = _core.integrate_flow(s.name, p, x, float(dt or s.dt), int(n), int(every),
                                      int(100_000 if transient is None else transient))
    else:
        out = _core.mackey_glass(p, float(x[0]), float(dt or s.dt), int(n), int(every),
                                 int(100_000 if transient is None else transient))[:, None]
    return out[:, 0] if (squeeze and out.shape[1] == 1) else out


def poincare(name: str, regime: str | None = None, n: int = 100_000, *, normal=None, offset: float | None = None,
             direction: float | None = None, params=None, x0=None, dt: float | None = None, transient: int = 100_000,
             seed: int | None = None, return_times: bool = False):
    """n crossings of the Poincare section {normal . x = offset} in the given direction.

    Default section: Lorenz z = 27 with dz/dt > 0 (as in Sakellariou et al. 2019);
    see ``FLOWS[name].section`` for the others.
    """
    s = get_system(name)
    if s.kind != "flow":
        raise ValueError(f"{s.name} is not a flow")
    if normal is None:
        if s.section is None:
            raise ValueError(f"{s.name}: no default section; pass normal/offset")
        normal, off0, dir0 = s.section
        offset = off0 if offset is None else offset
        direction = dir0 if direction is None else direction
    p = [float(v) for v in (parameters(s.name, regime) if params is None else params)]
    pts, t, _ = _core.poincare_section(s.name, p, _x0(s, x0, seed), float(dt or s.dt), int(n),
                                       [float(v) for v in normal], float(offset or 0.0),
                                       float(1.0 if direction is None else direction), int(transient))
    return (pts, t) if return_times else pts


def generate_to_file(name: str, path, n: int, regime: str | None = None, *, column: int | None = None,
                     chunk: int = 10_000_000, params=None, x0=None, transient: int | None = None,
                     dt: float | None = None, every: int = 1, seed: int | None = None):
    """Write a long trajectory to ``.npy`` in chunks (constant memory). ``column``
    selects one state variable (1-D output); default keeps all."""
    s = get_system(name)
    p = [float(v) for v in (parameters(s.name, regime) if params is None else params)]
    x = _x0(s, x0, seed)
    if s.kind == "dde":
        raise ValueError("generate_to_file does not support delay equations")
    shape = (n,) if (column is not None or s.dim == 1) else (n, s.dim)
    mm = np.lib.format.open_memmap(str(path), mode="w+", dtype=np.float64, shape=shape)
    tr = transient if transient is not None else (10_000 if s.kind == "map" else 100_000)
    pos = 0
    warned = False
    while pos < n:
        k = min(chunk, n - pos)
        if s.kind == "map":
            out, x = _core.iterate_map(s.name, p, x, k, tr)
            if not warned:
                warned = bool(_check_collapse(out, s.name, pos))
        else:
            out, x = _core.integrate_flow(s.name, p, x, float(dt or s.dt), k, int(every), tr)
            # continue from the state after the last stored sample
        tr = 0
        if len(shape) == 1:
            mm[pos:pos + k] = out[:, 0 if column is None else column]
        else:
            mm[pos:pos + k] = out
        pos += k
    mm.flush()
    return path


# ------------------------------------------------------- Jacobians/Lyapunov --

def jacobian(name: str, X, params=None, regime: str | None = None) -> np.ndarray:
    """Jacobian matrices (n x d x d) of a map at the points X (n x d)."""
    s = get_system(name)
    p = parameters(s.name, regime) if params is None else tuple(params)
    X = np.atleast_2d(np.asarray(X, dtype=float))
    if X.shape[1] != s.dim:
        X = X.reshape(-1, s.dim)
    n = len(X)
    J = np.zeros((n, s.dim, s.dim))
    x = X[:, 0]
    nm = s.name
    if nm == "logistic":
        J[:, 0, 0] = p[0] * (1 - 2 * x)
    elif nm == "cubic":
        J[:, 0, 0] = (p[0] - 1) - 3 * p[0] * x ** 2
    elif nm in ("tent",):
        J[:, 0, 0] = np.where(x < 0.5, p[0], -p[0])
    elif nm == "dyadic":
        J[:, 0, 0] = p[0]
    elif nm == "bent_baker":
        s6 = math.sqrt(6)
        J[:, 0, 0] = 4 * s6 * x ** 2 - 4 * s6 * x + (2 + 2 * s6 / 3)
    elif nm == "gauss":
        J[:, 0, 0] = -2 * p[0] * x * np.exp(-p[0] * x ** 2)
    elif nm == "henon":
        J[:, 0, 0] = -2 * p[0] * x; J[:, 0, 1] = 1; J[:, 1, 0] = p[1]
    elif nm == "lozi":
        y = X[:, 1]
        J[:, 0, 1] = 1; J[:, 1, 0] = p[1]; J[:, 1, 1] = -p[0] * np.sign(y)
    elif nm == "holmes":
        y = X[:, 1]
        J[:, 0, 1] = 1; J[:, 1, 0] = -p[1]; J[:, 1, 1] = p[0] - 3 * y ** 2
    elif nm == "ikeda":
        y = X[:, 1]; u = p[0]
        r2 = 1 + x ** 2 + y ** 2
        t = 0.4 - 6 / r2
        dtdx, dtdy = 12 * x / r2 ** 2, 12 * y / r2 ** 2
        c, sn = np.cos(t), np.sin(t)
        J[:, 0, 0] = u * (c - x * sn * dtdx - y * c * dtdx)
        J[:, 0, 1] = u * (-x * sn * dtdy - sn - y * c * dtdy)
        J[:, 1, 0] = u * (sn + x * c * dtdx - y * sn * dtdx)
        J[:, 1, 1] = u * (x * c * dtdy + c - y * sn * dtdy)
    elif nm == "tinkerbell":
        y = X[:, 1]; a, b, c, d = p
        J[:, 0, 0] = 2 * x + a; J[:, 0, 1] = -2 * y + b; J[:, 1, 0] = 2 * y + c; J[:, 1, 1] = 2 * x + d
    elif nm == "general_henon":
        J[:, 0, 1] = -2 * X[:, 1]; J[:, 0, 2] = -p[1]; J[:, 1, 0] = 1; J[:, 2, 1] = 1
    elif nm == "folded_towel":
        a, b = p
        x1, x2, x3 = X[:, 0], X[:, 1], X[:, 2]
        J[:, 0, 0] = a * (1 - 2 * x1); J[:, 0, 1] = -0.05 * (1 - 2 * x3); J[:, 0, 2] = 0.1 * (x2 + 0.35)
        J[:, 1, 0] = -0.19 * ((x2 + 0.35) * (1 + 2 * x3) - 1)
        J[:, 1, 1] = 0.1 * (1 + 2 * x3) * (1 - 1.9 * x1); J[:, 1, 2] = 0.2 * (x2 + 0.35) * (1 - 1.9 * x1)
        J[:, 2, 1] = b; J[:, 2, 2] = 3.78 * (1 - 2 * x3)
    else:
        raise NotImplementedError(f"no Jacobian for {s.name}")
    return J


def lyapunov_sum_jacobian(name: str, X, params=None, regime: str | None = None) -> float:
    """<ln |det J(x)|> along a trajectory (MATLAB ``DiscreteMapLyapunovSum``):
    the sum of *all* Lyapunov exponents (equals the positive one for 1-D maps)."""
    J = jacobian(name, X, params, regime)
    return float(np.mean(np.log(np.abs(np.linalg.det(J)))))


def lyapunov_spectrum(name: str, X, params=None, regime: str | None = None) -> np.ndarray:
    """Lyapunov spectrum of a map along trajectory X by QR (Benettin) iteration."""
    J = jacobian(name, X, params, regime)
    d = J.shape[1]
    Qm = np.eye(d)
    acc = np.zeros(d)
    for k in range(len(J)):
        Qm, R = np.linalg.qr(J[k] @ Qm)
        acc += np.log(np.abs(np.diag(R)))
    return np.sort(acc / len(J))[::-1]


def grid_lyapunov_1d(name: str, x, nbins: int, params=None, regime: str | None = None, lims=None) -> float:
    """Ulam/grid estimate sum_bins p(bin) ln|f'(centre)| (MATLAB ``GridPartition1D``)."""
    x = np.asarray(x, dtype=float)
    lo, hi = (x.min(), x.max()) if lims is None else lims
    edges = np.linspace(lo, hi, nbins + 1)
    p, _ = np.histogram(x, edges)
    p = p / p.sum()
    c = 0.5 * (edges[1:] + edges[:-1])
    d = np.abs(jacobian(name, c[:, None], params, regime)[:, 0, 0])
    ok = (p > 0) & (d > 0)
    return float(np.sum(p[ok] * np.log(d[ok])))

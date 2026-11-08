"""ordinal-networks: fast ordinal symbolic analysis and ordinal partition networks.

Core workflow::

    import ordinal_networks as on
    x = on.systems.generate("logistic", n=10**7)        # or your own series / .npy file
    stats = on.OrdinalSymbolizer(m=12).fit(x)            # tau=1, w=1 by default
    net = stats.network()
    net.spectral_radius(log=True)                        # topological entropy estimate
    net.lyapunov_sum()                                   # ordinal Lyapunov (LyapunovSum_OrdNet)
"""
from . import entropy, mapping, metrics, patterns, recurrence, surrogates, systems
from .io import load_network, load_statistics
from .network import OrdinalNetwork
from .patterns import MAX_M, decode, encode, factorial, pattern_of
from .symbolize import OrdinalStatistics, OrdinalSymbolizer, load_series, symbolize

__version__ = "0.1.0"

__all__ = [
    "OrdinalSymbolizer", "OrdinalStatistics", "OrdinalNetwork", "symbolize", "load_series",
    "load_statistics", "load_network", "decode", "encode", "factorial", "pattern_of", "MAX_M",
    "entropy", "mapping", "metrics", "patterns", "recurrence", "surrogates", "systems",
]

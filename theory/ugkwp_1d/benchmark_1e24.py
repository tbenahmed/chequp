"""
Initial state of the CHEQUP/WarpX 1D benchmark at n0 = 1e24 m^-3
(Bessel-beam ionization rings), DIGITIZED from the t = 0 comparison plot
(Castro/WarpX n_e and T_e), interpolated linearly. Replace by the actual
profile when available.
"""
import numpy as np

N0 = 1e24
T_HEAVY = 348.0            # K (0.03 eV), as in generate_initial_conditions.py
EV = 11604.518

_r = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
               19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 30, 200.0])
_ne = np.array([0.97, 0.98, 0.98, 0.97, 0.90, 0.75, 0.55, 0.38, 0.22, 0.14, 0.14, 0.18,
                0.22, 0.24, 0.21, 0.16, 0.08, 0.025, 0.003, 0.005, 0.02, 0.035, 0.045,
                0.045, 0.04, 0.028, 0.015, 0.006, 0.002, 0.0, 0.0])
_Te = np.array([2.0, 2.2, 2.3, 2.1, 1.8, 1.4, 1.0, 0.7, 0.45, 0.32, 0.33, 0.45, 0.6,
                0.65, 0.6, 0.45, 0.25, 0.13, 0.08, 0.12, 0.22, 0.30, 0.33, 0.32, 0.28,
                0.20, 0.12, 0.07, 0.05, 0.03, 0.03])
R_NODES_UM = _r
NE_NODES = _ne
TE_NODES_EV = _Te


def profile(r, n0=N0):
    """r [m] -> (n_e [m^-3], T_e [K]); piecewise-linear between the digitized
    nodes (the same interpolation is written into the WarpX input deck)."""
    x = np.clip(np.asarray(r) * 1e6, 0, 200)
    return np.interp(x, _r, _ne) * n0, np.interp(x, _r, _Te) * EV


def parser_expr(values, var='sqrt(x*x+y*y)'):
    """WarpX/AMReX parser expression for the piecewise-linear interpolation of
    `values` on the nodes R_NODES_UM (r in metres)."""
    r = 'rr'
    g = lambda v: f'{float(v):.10g}'
    expr = g(values[-1])
    for k in range(len(_r) - 2, -1, -1):
        r0, r1 = _r[k] * 1e-6, _r[k + 1] * 1e-6
        v0, v1 = values[k], values[k + 1]
        seg = f'({g(v0)}+({g(v1 - v0)})*({r}-{g(r0)})/{g(r1 - r0)})'
        expr = f'if({r}<{g(r1)},{seg},{expr})'
    # AMReX parser: local variable via 'rr = ...; expr'
    return f'rr={var}; {expr}'


def profile_pchip(r, n0=N0):
    """smooth (PCHIP) interpolation of the same nodes, as used in the first
    1e24 m^-3 UGKWP/particle/fluid comparison (bench_1e24.png)."""
    from scipy.interpolate import PchipInterpolator
    x = np.clip(np.asarray(r) * 1e6, 0, 200)
    ne = np.clip(PchipInterpolator(_r, _ne)(x), 0, 1) * n0
    Te = np.exp(PchipInterpolator(_r, np.log(_Te))(x)) * EV
    return ne, Te

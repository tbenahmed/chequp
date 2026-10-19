"""
Initial state of the CHEQUP/WarpX 1D benchmark at n0 = 1e24 m^-3
(Bessel-beam ionization rings), DIGITIZED from the t = 0 comparison plot
(Castro/WarpX n_e and T_e). Replace by the actual profile when available.
"""
import numpy as np
from scipy.interpolate import PchipInterpolator

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
_fne = PchipInterpolator(_r, _ne)
_fTe = PchipInterpolator(_r, np.log(_Te))


def profile(r):
    """r [m] -> (n_e [m^-3], T_e [K])"""
    x = np.clip(np.asarray(r) * 1e6, 0, 200)
    return np.clip(_fne(x), 0, 1) * N0, np.exp(_fTe(x)) * EV

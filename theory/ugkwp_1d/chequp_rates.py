"""
Hydrogen rates ported from CHEQUP (sim_folder/build), so that the kinetic
prototype uses the same electron physics as the Castro model:

  * collisional_ionization.H : direct ionization (binary-encounter cross
    section averaged over a Maxwellian, same 1000-point quadrature),
    excitation-mediated ionization (Gaunt-factor fit), 3-body recombination
    from Saha detailed balance with the Coulomb-regularized temperature;
  * collision_frequencies.H  : electron-ion (Eq. A1) and electron-neutral
    (Eq. A5, LXCat fit) collision frequencies of arXiv:2305.16779;
  * actual_conductivity.H    : electron thermal conductivity (Eq. B1).

All functions take SI inputs (m^-3, K) and return SI outputs.
"""
import numpy as np
from math import exp, log, sqrt, pi

KB = 1.380649e-23
QE = 1.602176634e-19
ME = 9.1093837015e-31
EPS0 = 8.854187817e-12
HBAR = 1.054571817e-34
K_TO_EV = 8.617333262145e-5
CHI_H = 13.598434          # eV
DE_EX_H = 10.6             # eV, effective excitation energy (CHEQUP)
F_OSC_H = 0.416
GAUNT_H = [-7.61094672e-06, 1.42014919e-04, -5.26821513e-04, -2.27233338e-03,
           2.38468040e-03, 5.36211118e-02, 2.56972926e-01]


def _sigma_direct_cm2(ep):
    """CHEQUP sigma_E for H (Z=1, zstar=0), ep in eV, returns cm^2."""
    me_c2 = 510998.0
    norm = 2 * pi * (1 / 137.0) ** 4 * (5.29e-11) ** 2
    ep_mc2 = ep / me_c2
    bp = CHI_H / me_c2
    e = ep_mc2 / bp
    if e <= 1.0:
        return 0.0
    betae2 = 1 - 1 / (1 + ep_mc2) ** 2
    betab2 = 1 - 1 / (1 + bp) ** 2
    s0 = norm / (bp * (betae2 + 2 * betab2))
    ep2 = (1 / (1 + 0.5 * ep_mc2)) ** 2
    a1c = (1 + 2 * ep_mc2) / (1 + e) * ep2
    a2c = (e - 1) * bp * bp * 0.5 * ep2
    a3 = log(betae2 / (1 - betae2)) - betae2 - log(2 * bp)
    sk = s0 * (0.5 * a3 * (1 - 1 / e ** 2) + 1 - 1 / e + a2c - a1c * log(e))
    return sk * 1e4


def _rate_direct_cgs(TeV):
    """CHEQUP compute_rate: same E grid (0..1000 eV, 1000 points)."""
    s = 0.0
    nrm = 0.0
    for i in range(1000):
        E = i * 1.0
        w = sqrt(E) * exp(-E / TeV)
        v = sqrt(2 * E * 1.602176634e-12 / 9.1093837015e-28)
        s += _sigma_direct_cm2(E) * v * w
        nrm += w
    return s / nrm if nrm > 0 else 0.0


_T_MIN, _T_MAX, _NT = 1e-3, 3e2, 100
_LOGT = np.linspace(np.log(_T_MIN), np.log(_T_MAX), _NT)
# CHEQUP's own table (kept for comparison): 1 eV quadrature + linear interpolation
_TABLE = np.array([_rate_direct_cgs(np.exp(lt)) for lt in _LOGT])


def _G_direct_cgs(TeV, nx=4000):
    """G(T) = S_direct(T) * exp(chi/T) [cm^3/s] with a quadrature resolved on
    the scale of T: E = chi + T x, and the exact Maxwellian normalization.
    G is smooth in T, so it can be interpolated safely (S itself varies by
    many orders of magnitude between table nodes at low T)."""
    x = np.linspace(0, 60, nx)
    E = CHI_H + TeV * x
    sig = np.array([_sigma_direct_cm2(e) for e in E])
    v = np.sqrt(2 * E * 1.602176634e-12 / 9.1093837015e-28)
    integrand = sig * v * np.sqrt(E) * np.exp(-x) * TeV
    num = np.sum(0.5 * (integrand[1:] + integrand[:-1]) * np.diff(x))
    norm = 0.5 * np.sqrt(np.pi) * TeV ** 1.5
    return num / norm


_LNG = np.log(np.array([_G_direct_cgs(np.exp(lt)) for lt in _LOGT]))


def _G_direct(TK):
    TeV = np.clip(np.asarray(TK, float) * K_TO_EV, _T_MIN, _T_MAX)
    return np.exp(np.interp(np.log(TeV), _LOGT, _LNG)) * 1e-6


def rate_direct(TK):
    """direct ionization rate coefficient [m^3/s] (accurate at all T)."""
    TeV = np.clip(np.asarray(TK, float) * K_TO_EV, _T_MIN, _T_MAX)
    return _G_direct(TK) * np.exp(-CHI_H / TeV)


def rate_direct_chequp_table(TK):
    """CHEQUP's tabulated direct rate (linear interpolation), for comparison."""
    TeV = np.clip(np.asarray(TK, float) * K_TO_EV, _T_MIN, _T_MAX)
    x = (np.log(TeV) - np.log(_T_MIN)) / (np.log(_T_MAX) - np.log(_T_MIN)) * (_NT - 1)
    i = np.minimum(x.astype(int), _NT - 2)
    w = x - i
    return ((1 - w) * _TABLE[i] + w * _TABLE[i + 1]) * 1e-6


def _G_ex(TK):
    """S_ex * exp(dE/T) [m^3/s] (analytic, CHEQUP Gaunt fit)."""
    TeV = np.clip(np.asarray(TK, float) * K_TO_EV, _T_MIN, _T_MAX)
    x = np.log(TeV)
    poly = sum(GAUNT_H[n] * x ** (6 - n) for n in range(7))
    return 1.57e-7 * F_OSC_H * np.exp(poly) / (DE_EX_H * np.sqrt(TeV)) * 1e-6


def rate_ex(TK):
    """excitation-mediated ionization rate coefficient [m^3/s]."""
    TeV = np.clip(np.asarray(TK, float) * K_TO_EV, _T_MIN, _T_MAX)
    return _G_ex(TK) * np.exp(-DE_EX_H / TeV)


def T_effective(TK, ne):
    """CHEQUP regularization of T_e with the Coulomb temperature."""
    n_cgs = np.maximum(ne * 1e-6, 1e-30)
    a_d = np.cbrt(3 / (4 * np.pi * n_cgs))
    Tc = (4.803204712570263e-10 ** 2 / a_d) / 1.380649e-16
    return np.sqrt(TK ** 2 + Tc ** 2)


def rate_3body(TK, ne):
    """3-body recombination coefficient [m^6/s]: R_rec = K3 n_e^2 n_i, from
    detailed balance as in CHEQUP (both channels, Coulomb-regularized T),
    evaluated without overflow/interpolation artefacts."""
    Teff = T_effective(np.asarray(TK, float), ne)
    lam3 = (2 * np.pi * HBAR ** 2 / (ME * KB * Teff)) ** 1.5     # m^3
    # g(H0) / (2 g(H+)) = 2 / 2 = 1
    return (_G_direct(Teff) + _G_ex(Teff)) * lam3


def sigma_en_H(log_E_eV):
    num = [-4.451560e+01, -4.477831e+01, -6.605789e+02, 9.020842e+01, -1.798055e+02]
    den = [8.890529e-01, 1.461417e+01, -3.442456e+00, 3.765710e+00]
    x = (log_E_eV - 2.606962e+00) / 4.605170e+00
    nv = num[0] + num[1] * x + num[2] * x ** 2 + num[3] * x ** 3 + num[4] * x ** 4
    dv = 1 + den[0] * x + den[1] * x ** 2 + den[2] * x ** 3 + den[3] * x ** 4
    return np.exp(nv / dv)


def collision_frequencies(ne, nn, Te, Th, mH):
    """(nu_ei, nu_en) [1/s] as in CHEQUP get_collision_frequencies (Z=1)."""
    ne_s = np.maximum(ne, 1.0)
    raw = np.log(3 / (2 * np.sqrt(np.pi)) * np.sqrt(
        (4 * np.pi * EPS0 * KB * Te) ** 3 / (QE ** 6 * ne_s * (1 + Te / Th))))
    lam = np.maximum(raw, 0.5 * np.log(2))
    nu_ei = (4 / 3) * np.sqrt(2 * np.pi / ME) * ne * QE ** 4 * lam / \
        ((4 * np.pi * EPS0) ** 2 * (KB * Te) ** 1.5)
    E = 4 / np.pi * KB * Te
    mu = ME * mH / (ME + mH)
    nu_en = nn * 4 / 3 * np.sqrt(2 * E / mu) * sigma_en_H(np.log(E / QE))
    return nu_ei, nu_en


def conductivity(ne, nn, Te, Th, mH):
    """electron thermal conductivity [W/m/K] (CHEQUP Eq. B1, Z=1)."""
    nu_ei, nu_en = collision_frequencies(ne, nn, Te, Th, mH)
    gamma0 = 12.471 * (1 + 0.112) / (1 + 3.386)
    den = (2 * np.pi / 15) * nu_en + nu_ei / gamma0
    with np.errstate(all='ignore'):
        k = np.where(ne > 0, KB ** 2 * ne * Te / (ME * np.maximum(den, 1e-300)), 0.0)
    return k

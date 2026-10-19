"""
HOFI-like 1D cylindrical channel with UGKWP:

  * H+ ions     : kinetic (UGKWP), BGK self-collisions with Coulomb rate
  * H neutrals  : kinetic (UGKWP), BGK self-collisions, VHS H-H
  * electrons   : quasi-neutral fluid (n_e = n_i), internal energy equation
                  (advection with the ions, p dV work), drive the ions via
                  the ambipolar field  E = -grad(p_e) / (e n_e)
  * couplings   : resonant charge exchange H+ + H (constant cross-section),
                  electron-impact ionization of H (Voronov fit),
                  electron-ion energy equilibration (NRL)

All couplings are operator-split from the UGKWP transport/self-collision
step. Charge exchange acts on the distribution: a neutral particle that
undergoes CX is re-drawn from the ion Maxwellian (and vice versa), so fast
CX neutrals appear as a genuinely non-Maxwellian population. Ionization moves
neutral particles (with their velocities) into the ion species.

Not included (yet): electron heat conduction, electron-neutral
collisions, recombination, radiation.

Units: SI.
"""
import time
import numpy as np
from numba import njit
from math import exp, sqrt, log
from ugkwp import Species, prim, valid, match_moments, cell_sort, locate

KB = 1.380649e-23
QE = 1.602176634e-19
ME = 9.1093837015e-31
MH = 1.6735575e-27
EION = 13.598 * QE


# --------------------------------------------------------------------------
# rates
# --------------------------------------------------------------------------
def voronov_H(TeV):
    """electron impact ionization rate coefficient of H [m^3/s] (Voronov 1997)."""
    TeV = np.maximum(TeV, 0.05)
    U = 13.6 / TeV
    return 2.91e-8 * U ** 0.39 * np.exp(-U) / (0.232 + U) * 1e-6


def nu_ei(ne, TeV):
    n_cm3 = np.maximum(ne * 1e-6, 1e-30)
    lnL = np.maximum(24.0 - np.log(np.sqrt(n_cm3) / TeV), 2.0)
    return 2.91e-6 * n_cm3 * lnL * TeV ** (-1.5)


# --------------------------------------------------------------------------
# particle operations
# --------------------------------------------------------------------------
@njit(cache=True)
def _kick(px, pvx, n, xf, a, dt):
    for p in range(n):
        pvx[p] += a[locate(px[p], xf)] * dt


@njit(cache=True)
def _heat(px, pvx, pvy, pvz, n, xf, Up, s):
    for p in range(n):
        i = locate(px[p], xf)
        pvx[p] = Up[i] + s[i] * (pvx[p] - Up[i])
        pvy[p] *= s[i]
        pvz[p] *= s[i]


@njit(cache=True)
def _resample(px, pvx, pvy, pvz, pw, n, xf, ncell, P, Ut, RTt):
    """Each particle is re-drawn with probability P[cell] from the
    Maxwellian (Ut, RTt)[cell]; each re-drawn group is moment-matched to that
    Maxwellian."""
    order, start = cell_sort(px, n, xf, ncell)
    for i in range(ncell):
        m = start[i + 1] - start[i]
        if m == 0 or P[i] <= 0:
            continue
        sel = np.empty(m, dtype=np.int64)
        k = 0
        for j in range(start[i], start[i + 1]):
            if np.random.random() < P[i]:
                sel[k] = order[j]
                k += 1
        if k == 0:
            continue
        s = sqrt(RTt[i])
        mass = 0.0
        for j in range(k):
            p = sel[j]
            pvx[p] = Ut[i] + s * np.random.standard_normal()
            pvy[p] = s * np.random.standard_normal()
            pvz[p] = s * np.random.standard_normal()
            mass += pw[p]
        if k >= 2:
            match_moments(sel[:k], pvx, pvy, pvz, pw, mass, mass * Ut[i],
                          mass * (0.5 * Ut[i] ** 2 + 1.5 * RTt[i]))


@njit(cache=True)
def _select(px, n, xf, P):
    out = np.zeros(n, dtype=np.bool_)
    for p in range(n):
        if np.random.random() < P[locate(px[p], xf)]:
            out[p] = True
    return out


# --------------------------------------------------------------------------
class HOFIChannel:
    def __init__(self, R=100e-6, ncell=200, n0=1e23, r0=12e-6, Te0_eV=5.0,
                 T0=300.0, floor=1e-4, nc=100, full_particle=False, seed=1,
                 sigma_cx=5e-19, d_HH=2.5e-10, omega_HH=0.7, fluid=False):
        self.xf = np.linspace(0, R, ncell + 1)
        xc = 0.5 * (self.xf[1:] + self.xf[:-1])
        f = np.exp(-(xc / r0) ** 4)
        ni = np.maximum(n0 * f, floor * n0)
        nn = np.maximum(n0 * (1 - f), floor * n0)
        RT0 = KB * T0 / MH

        def W_of(n):
            W = np.zeros((ncell, 3))
            W[:, 0] = n * MH
            W[:, 2] = 1.5 * n * MH * RT0
            return W

        mu_ref = 15 * np.sqrt(np.pi * MH * KB * T0) / (2 * np.pi * d_HH ** 2 * (5 - 2 * omega_HH) * (7 - 2 * omega_HH))
        tp_n = [0.0, KB / MH, mu_ref, T0, omega_HH, MH, 0.0]
        tp_i = [1.0, KB / MH, 0.0, 0.0, 0.0, MH, 2.0]
        if fluid:
            # two-fluid Euler limit of the same model (tau -> 0 for self-collisions)
            tp_n = [2.0, KB / MH, 1e-30, 0, 0, MH, 0]
            tp_i = [2.0, KB / MH, 1e-30, 0, 0, MH, 0]
        rwo = 0.0 if full_particle else 2 * floor * n0 * MH
        self.ion = Species(self.xf, 1, W_of(ni), tp_i, nc=nc, seed=seed, full_particle=full_particle,
                           rho_wave_only=rwo, RT_floor=RT0)
        self.neu = Species(self.xf, 1, W_of(nn), tp_n, nc=nc, seed=seed + 1000, full_particle=full_particle,
                           rho_wave_only=rwo, RT_floor=RT0)
        self.Te_arr = T0 + (Te0_eV * 11604.518 - T0) * f   # electron temperature [K]
        self.T0 = T0
        self.sigma_cx = sigma_cx
        self.full = full_particle
        self.t = 0.0
        self.nsteps = 0
        self.V = self.ion.V
        self.A = self.ion.A
        self.xc = xc

    # ------------------------------------------------------------------
    @staticmethod
    def _mom(sp):
        """rho, U, RT; empty cells (possible in the particle reference) give
        rho = 0, U = 0, RT = 0."""
        W = sp.W
        ok = valid(W)
        rho = np.where(ok, W[:, 0], 0.0)
        r = np.where(ok, rho, 1.0)
        U = np.where(ok, W[:, 1] / r, 0.0)
        RT = np.where(ok, (W[:, 2] - 0.5 * r * U * U) / (1.5 * r), 0.0)
        return rho, U, RT

    def ne(self):
        return np.maximum(self.ion.W[:, 0], 0.0) / MH

    def Te(self):
        return self.Te_arr

    @property
    def eps_e(self):
        return 1.5 * self.ne() * KB * self.Te_arr

    def pe(self):
        return self.ne() * KB * self.Te_arr

    def _sync(self, sp):
        """W = W^h + W^p, then repair the wave part if needed."""
        sp.W = sp.Wh + sp.Wp()
        sp.fix_wave()

    # ------------------------------------------------------------------
    def dt_cfl(self, cfl=0.4):
        dr = self.xf[1] - self.xf[0]
        ri = self.ion.W[:, 0]
        ce2 = np.where(ri > 0, GAMMA_E * self.pe() / np.where(ri > 0, ri, 1), 0.0)
        s = max(self.ion.max_speed(ce2), self.neu.max_speed())
        return cfl * dr / s

    def step(self, dt):
        ion, neu = self.ion, self.neu
        # 1) transport + self collisions (UGKWP or particle BGK)
        U_old = self._mom(ion)[1]
        ion.step(dt)
        neu.step(dt)
        U_new = self._mom(ion)[1]
        # 2) electrons: advection with the ions and p dV work
        self._electrons(dt, 0.5 * (U_old + U_new))
        # 3) ambipolar force on the ions
        self._ambipolar(dt)
        # 4) electron-ion energy exchange
        self._ei_exchange(dt)
        # 5) charge exchange
        self._charge_exchange(dt)
        # 6) electron-impact ionization
        self._ionization(dt)
        self.t += dt
        self.nsteps += 1

    # ------------------------------------------------------------------
    def _electrons(self, dt, U):
        """T_e: upwind advection with the ion velocity + adiabatic p dV work,
        dT_e/dt + U dT_e/dr = -(2/3) T_e div(U). Evolving T_e (not the energy
        density) keeps p_e = n_e k T_e consistent with the kinetic ion density."""
        T = self.Te_arr
        uf = np.zeros(len(self.xf))
        uf[1:-1] = 0.5 * (U[1:] + U[:-1])
        dr = self.xf[1] - self.xf[0]
        Tg = np.concatenate([[T[0]], T, [T[-1]]])
        uc = U
        grad_up = np.where(uc > 0, (Tg[1:-1] - Tg[:-2]) / dr, (Tg[2:] - Tg[1:-1]) / dr)
        div = (self.A[1:] * uf[1:] - self.A[:-1] * uf[:-1]) / self.V
        T_new = (T - dt * uc * grad_up) * np.exp(-(2.0 / 3.0) * div * dt)
        self.Te_arr = np.clip(T_new, self.T0, TE_MAX)

    def _ambipolar(self, dt):
        ion = self.ion
        pe = self.pe()
        pg = np.concatenate([[pe[0]], pe, [pe[-1]]])
        dr = self.xf[1] - self.xf[0]
        grad = (pg[2:] - pg[:-2]) / (2 * dr)
        ri = ion.W[:, 0]
        a = np.where(ri > 0, -grad / np.where(ri > 0, ri, 1), 0.0)
        # wave part
        Wh = ion.Wh
        okh = valid(Wh)
        rh = np.where(okh, Wh[:, 0], 0.0)
        dv = a * dt
        Wh[:, 2] += np.where(okh, Wh[:, 1] * dv + 0.5 * rh * dv * dv, 0.0)
        Wh[:, 1] += rh * dv
        # particle part
        _kick(ion.px, ion.pvx, ion.np, ion.xf, a, dt)
        self._sync(ion)
        # work done on the ions is taken from the electrons via p dV (step 2)

    def _add_heat(self, sp, dE):
        """add energy dE [J/m^3] to the thermal energy of species sp."""
        Wp = sp.Wp()
        rho = sp.W[:, 0]
        fh = np.clip(sp.Wh[:, 0] / rho, 0, 1)
        okh = valid(sp.Wh)
        sp.Wh[:, 2] += np.where(okh, dE * fh, 0.0)
        dEp = dE - np.where(okh, dE * fh, 0.0)
        rp = Wp[:, 0]
        Up = np.where(rp > 0, Wp[:, 1] / np.where(rp > 0, rp, 1), 0.0)
        Eth = Wp[:, 2] - 0.5 * rp * Up ** 2
        s = np.sqrt(np.clip(1 + np.where(Eth > 0, dEp / np.where(Eth > 0, Eth, 1), 0.0), 0.25, 4.0))
        _heat(sp.px, sp.pvx, sp.pvy, sp.pvz, sp.np, sp.xf, Up, s)
        self._sync(sp)

    def _ei_exchange(self, dt):
        ne = self.ne()
        Te = self.Te()
        rho, U, RT = self._mom(self.ion)
        Ti = RT * MH / KB
        TeV = Te / 11604.518
        nu = nu_ei(ne, TeV)
        nu_eq = 2 * ME / MH * nu             # dTe/dt = -nu_eq (Te - Ti), same for Ti (n_e = n_i)
        x = (Te - Ti) * (1 - np.exp(-2 * nu_eq * dt)) / 2
        dE = 1.5 * ne * KB * x
        self.Te_arr = np.maximum(self.Te_arr - x, self.T0)
        self._add_heat(self.ion, dE)

    def _charge_exchange(self, dt):
        ion, neu = self.ion, self.neu
        ri, Ui, RTi = self._mom(ion)
        rn, Un, RTn = self._mom(neu)
        ni, nn = ri / MH, rn / MH
        vrel = np.sqrt(8 / np.pi * (RTi + RTn) + (Ui - Un) ** 2)
        k = self.sigma_cx * vrel
        Pn = 1 - np.exp(-ni * k * dt)
        Pi = 1 - np.exp(-nn * k * dt)
        # exact pairwise relaxation of momentum and energy per particle
        D = 1 - np.exp(-(ni + nn) * k * dt)
        both = (ni > 0) & (nn > 0)
        Pn = np.where(both, Pn, 0.0)
        Pi = np.where(both, Pi, 0.0)
        nis = np.where(both, ni, 1.0)
        nns = np.where(both, nn, 1.0)
        pi_, pn_ = ion.W[:, 1] / nis, neu.W[:, 1] / nns
        ei_, en_ = ion.W[:, 2] / nis, neu.W[:, 2] / nns
        dp = np.where(both, (pi_ - pn_) * D, 0.0)
        de = np.where(both, (ei_ - en_) * D, 0.0)
        Wi_t = np.stack([ion.W[:, 0], ion.W[:, 1] - ni * nn / (nis + nns) * dp,
                         ion.W[:, 2] - ni * nn / (nis + nns) * de], 1)
        Wn_t = np.stack([neu.W[:, 0], neu.W[:, 1] + nn * ni / (nis + nns) * dp,
                         neu.W[:, 2] + nn * ni / (nis + nns) * de], 1)
        # distribution-level exchange
        for sp, P, Ut, RTt in ((neu, Pn, Ui, RTi), (ion, Pi, Un, RTn)):
            _resample(sp.px, sp.pvx, sp.pvy, sp.pvz, sp.pw, sp.np, sp.xf, sp.n, P, Ut, RTt)
            okh = valid(sp.Wh)
            rh = np.where(okh, sp.Wh[:, 0], 0.0)
            tgt = np.stack([rh, rh * Ut, rh * (0.5 * Ut ** 2 + 1.5 * RTt)], 1)
            sp.Wh = np.where(okh[:, None], sp.Wh + P[:, None] * (tgt - sp.Wh), 0.0)
        if self.full:
            for sp in (ion, neu):
                sp.W = sp.Wp()
                sp.Wh[:] = 0.0
        else:
            ion.W = Wi_t
            neu.W = Wn_t
            ion.fix_wave()
            neu.fix_wave()

    def _ionization(self, dt):
        ion, neu = self.ion, self.neu
        ne = self.ne()
        S = voronov_H(self.Te() / 11604.518)
        P = 1 - np.exp(-ne * S * dt)
        # particles: move ionized neutrals (with their velocity) to the ions
        sel = _select(neu.px, neu.np, neu.xf, P)
        moved = np.zeros(len(P))
        if sel.any():
            idx = np.nonzero(sel)[0]
            np.add.at(moved, np.minimum(((neu.px[idx] - self.xf[0]) / (self.xf[1] - self.xf[0])).astype(int), neu.n - 1),
                      neu.pw[idx])
            ion._append(neu.px[idx], neu.pvx[idx], neu.pvy[idx], neu.pvz[idx], neu.pw[idx])
            neu._compact(~sel)
        # wave part
        okh = valid(neu.Wh)
        dWh = np.where(okh[:, None], P[:, None] * neu.Wh, 0.0)
        neu.Wh = neu.Wh - dWh
        ion.Wh = np.where(valid(ion.Wh)[:, None], ion.Wh, 0.0) + dWh
        moved = moved / self.V + dWh[:, 0]
        # electrons: new electrons are born cold, each ionization costs 13.6 eV
        dn = moved / MH
        eps = 1.5 * ne * KB * self.Te_arr - dn * EION
        ne_new = ne + dn
        self.Te_arr = np.where(ne_new > 0, eps / (1.5 * np.where(ne_new > 0, ne_new, 1) * KB), self.T0)
        self.Te_arr = np.clip(self.Te_arr, self.T0, TE_MAX)
        if self.full:
            for sp in (ion, neu):
                sp.W = sp.Wp()
                sp.Wh[:] = 0.0
        else:
            self._sync(ion)
            self._sync(neu)

    # ------------------------------------------------------------------
    def diagnostics(self):
        out = dict(r=self.xc, Te=self.Te(), t=self.t)
        for name, sp in (('i', self.ion), ('n', self.neu)):
            d = sp.diagnostics()
            out[f'n_{name}'] = d['rho'] / MH
            out[f'U_{name}'] = d['U']
            out[f'T_{name}'] = d['RT'] * MH / KB
            out[f'fp_{name}'] = d['fp']
            out[f'Pirr_{name}'] = d['Pi_xx']
            out[f'Pitt_{name}'] = d['Pi_yy']
            out[f'q_{name}'] = d['q']
            out[f'np_{name}'] = d['nparticles']
        vth_n = np.sqrt(2 * KB * out['T_n'] / MH)
        out['slip'] = (out['U_i'] - out['U_n']) / vth_n
        return out


GAMMA_E = 5.0 / 3.0
TE_MAX = 100 * 11604.518

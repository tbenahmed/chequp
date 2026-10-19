"""
1D Unified Gas-Kinetic Wave-Particle (UGKWP) method, BGK collision model.

Planar (x) or axisymmetric cylindrical (r) geometry, monatomic gas with
three velocity components (K = 2 internal/transverse degrees of freedom
in the 1D GKS notation, gamma = 5/3).

References
----------
- C. Liu, Y. Zhu, K. Xu, "Unified gas-kinetic wave-particle methods I:
  Continuum and rarefied gas flow", J. Comput. Phys. 401 (2020) 108977.
- C. Liu, K. Xu, "Unified gas-kinetic wave-particle methods IV:
  multi-species gas mixture and plasma transport", Adv. Aerodyn. 3 (2021) 9.
- K. Xu, J.-C. Huang, "A unified gas-kinetic scheme for continuum and
  rarefied flows", J. Comput. Phys. 229 (2010) 7747.

Every cell holds the total macroscopic state W = (rho, rho*U, rho*E).
The distribution is split into
  * a particle part  (non-equilibrium, carried by simulation particles),
  * a wave part      (W^h = W - W^p, an analytic local Maxwellian).
Each time step:
  1. from W^h, the fraction exp(-dt/tau) that would stream freely for the
     whole step is sampled as particles ("hydro particles");
  2. existing particles draw a free-flight time t_c = -tau*ln(eta); those
     with t_c < dt collide during the step and are absorbed into the wave
     part where they stop;
  3. W is updated with the analytic equilibrium flux (GKS integral
     solution), the analytic free-transport flux of the un-sampled wave
     part, and the particle fluxes (counted by deposition);
  4. W^h = W - W^p(survivors).
For tau << dt nothing is sampled and the scheme is a gas-kinetic
(Navier-Stokes/Euler) finite-volume solver; for tau >> dt everything is
carried by particles and the scheme is a particle method for the
collisionless/BGK kinetic equation.
"""
import numpy as np
from math import erfc, exp, sqrt, pi, log
from numba import njit

KINT = 2.0           # transverse degrees of freedom (vy, vz)
GAM = 5.0 / 3.0


# --------------------------------------------------------------------------
# Maxwellian moment helpers
# --------------------------------------------------------------------------
@njit(cache=True)
def prim(W):
    rho = W[0]
    U = W[1] / rho
    eth = W[2] - 0.5 * rho * U * U
    lam = (KINT + 1.0) * rho / (4.0 * eth)      # lam = 1/(2RT)
    return rho, U, lam


@njit(cache=True)
def umoments(U, lam, n, mu, mup, mun):
    mu[0] = 1.0
    mu[1] = U
    s = sqrt(lam) * U
    mup[0] = 0.5 * erfc(-s)
    mun[0] = 0.5 * erfc(s)
    e = exp(-lam * U * U) / (2.0 * sqrt(pi * lam))
    mup[1] = U * mup[0] + e
    mun[1] = U * mun[0] - e
    for k in range(2, n + 1):
        c = (k - 1) / (2.0 * lam)
        mu[k] = U * mu[k - 1] + c * mu[k - 2]
        mup[k] = U * mup[k - 1] + c * mup[k - 2]
        mun[k] = U * mun[k - 1] + c * mun[k - 2]


@njit(cache=True)
def mpsi(k, m, xi2, out):
    """<u^k psi>, psi = (1, u, (u^2+xi^2)/2)."""
    out[0] = m[k]
    out[1] = m[k + 1]
    out[2] = 0.5 * (m[k + 2] + m[k] * xi2)


@njit(cache=True)
def mpsi_a(k, m, xi2, xi4, a, out):
    """<u^k psi (a . psi)>."""
    a0, a1, a2 = a[0], a[1], a[2]
    out[0] = a0 * m[k] + a1 * m[k + 1] + a2 * 0.5 * (m[k + 2] + m[k] * xi2)
    out[1] = a0 * m[k + 1] + a1 * m[k + 2] + a2 * 0.5 * (m[k + 3] + m[k + 1] * xi2)
    out[2] = 0.5 * (a0 * (m[k + 2] + m[k] * xi2)
                    + a1 * (m[k + 3] + m[k + 1] * xi2)
                    + a2 * 0.5 * (m[k + 4] + 2.0 * m[k + 2] * xi2 + m[k] * xi4))


@njit(cache=True)
def solve_slope_linalg(U, lam, b):
    """Reference implementation: solve <psi psi> a = b numerically."""
    mu = np.empty(7); mup = np.empty(7); mun = np.empty(7)
    umoments(U, lam, 6, mu, mup, mun)
    xi2 = KINT / (2.0 * lam)
    xi4 = (KINT * KINT + 2.0 * KINT) / (4.0 * lam * lam)
    M = np.empty((3, 3))
    e = np.zeros(3)
    col = np.empty(3)
    for j in range(3):
        e[:] = 0.0
        e[j] = 1.0
        mpsi_a(0, mu, xi2, xi4, e, col)
        M[:, j] = col
    return np.linalg.solve(M, b)


@njit(cache=True)
def solve_slope(U, lam, b):
    """Closed-form solution of <psi psi> a = b for the Maxwellian (U, lam)."""
    a = np.zeros(3)
    if not (lam > 0.0) or not np.isfinite(lam) or not np.isfinite(U):
        return a
    c = U * U + (KINT + 1.0) / (2.0 * lam)
    R2 = b[1] - U * b[0]
    R3 = 2.0 * b[2] - c * b[0]
    a[2] = 4.0 * lam * lam / (KINT + 1.0) * (R3 - 2.0 * U * R2)
    a[1] = 2.0 * lam * R2 - U * a[2]
    a[0] = b[0] - U * a[1] - 0.5 * a[2] * c
    for q in range(3):
        if not np.isfinite(a[q]):
            a[:] = 0.0
            break
    return a


# --------------------------------------------------------------------------
# Collision time
# --------------------------------------------------------------------------
# tp = [kind, Rgas, mu_ref, T_ref, omega, mass, lnL_min]
#   kind 0: VHS gas,       tau = mu_ref (T/T_ref)^omega / p
#   kind 1: Coulomb ions,  tau = 1/nu_ii  (NRL formulary, Z=1, SI input)
#   kind 2: fluid limit,   tau = tp[2] (tiny): the scheme reduces to its
#           gas-kinetic Euler solver and never creates particles
@njit(cache=True)
def tau_of(rho, lam, tp):
    if tp[0] == 2.0:
        return tp[2]
    RT = 1.0 / (2.0 * lam)
    T = RT / tp[1]
    if tp[0] == 0.0:
        return tp[2] * (T / tp[3]) ** tp[4] / (rho * RT)
    # Coulomb self-collisions of singly charged ions
    n_cm3 = rho / tp[5] * 1e-6
    TeV = max(T / 11604.518, 1e-6)
    lnL = 23.0 - log(sqrt(n_cm3) * TeV ** (-1.5))
    if lnL < tp[6]:
        lnL = tp[6]
    mu_amu = tp[5] / 1.66053907e-27
    nu = 4.8e-8 * n_cm3 * lnL * TeV ** (-1.5) / sqrt(mu_amu)
    if not (nu > 0.0):
        return 1e30
    return 1.0 / nu


# --------------------------------------------------------------------------
# Interface fluxes
@njit(cache=True)
def _ok(W):
    """admissible state: positive density and a thermal energy that is not
    lost in round-off of the total energy."""
    if not (W[0] > 0.0):
        return False
    eth = W[2] - 0.5 * W[1] * W[1] / W[0]
    return eth > 1e-12 * abs(W[2]) and np.isfinite(eth)


# --------------------------------------------------------------------------
@njit(cache=True)
def face_fluxes(WL, WR, dWdx, WhL, WhR, shL, shR, tauL, tauR,
                flagL, flagR, dt, tp, Feq, Ffr):
    """
    Feq: equilibrium (collision) part of the UGKS integral-solution flux.
    Ffr: free transport of the un-sampled wave part.
    Both are time-integrated over dt, per unit area.
    """
    mu = np.empty(8); mup = np.empty(8); mun = np.empty(8)
    tmp = np.empty(3); tmp2 = np.empty(3)

    # ---- equilibrium state at the interface
    for q in range(3):
        Feq[q] = 0.0
    W0 = np.zeros(3)
    okL = _ok(WL)
    okR = _ok(WR)
    pL = 0.0
    pR = 0.0
    if okL:
        rL, UL, lL = prim(WL)
        pL = rL / (2 * lL)
        umoments(UL, lL, 7, mu, mup, mun)
        mpsi(0, mup, KINT / (2 * lL), tmp)
        W0 += rL * tmp
    if okR:
        rR, UR, lR = prim(WR)
        pR = rR / (2 * lR)
        umoments(UR, lR, 7, mu, mup, mun)
        mpsi(0, mun, KINT / (2 * lR), tmp)
        W0 += rR * tmp
    ok0 = _ok(W0)
    if ok0:
        r0, U0, l0 = prim(W0)
        xi2 = KINT / (2 * l0)
        xi4 = (KINT * KINT + 2 * KINT) / (4 * l0 * l0)
        umoments(U0, l0, 7, mu, mup, mun)
        a = solve_slope(U0, l0, dWdx / r0)
        mpsi_a(1, mu, xi2, xi4, a, tmp)
        A = solve_slope(U0, l0, -tmp)
        tau = tau_of(r0, l0, tp) + dt * abs(pL - pR) / (pL + pR)
        ex = exp(-dt / tau)
        C1 = dt - tau * (1 - ex)
        C2 = -tau * dt + 2 * tau * tau * (1 - ex) - tau * dt * ex
        C3 = 0.5 * dt * dt - tau * dt + tau * tau * (1 - ex)
        mpsi(1, mu, xi2, tmp)
        for q in range(3):
            Feq[q] = r0 * C1 * tmp[q]
        mpsi_a(2, mu, xi2, xi4, a, tmp)
        for q in range(3):
            Feq[q] += r0 * C2 * tmp[q]
        mpsi_a(1, mu, xi2, xi4, A, tmp)
        for q in range(3):
            Feq[q] += r0 * C3 * tmp[q]

    # ---- free transport of the wave part (upwind)
    for q in range(3):
        Ffr[q] = 0.0
    for side in range(2):
        if side == 0:
            Wh = WhL; sh = shL; tauc = tauL; flag = flagL
        else:
            Wh = WhR; sh = shR; tauc = tauR; flag = flagR
        if not _ok(Wh):
            continue
        rh, Uh, lh = prim(Wh)
        if not (lh > 0.0) or not np.isfinite(lh):
            continue
        ex = exp(-dt / tauc)
        C4 = tauc * (1 - ex)
        C5 = tauc * dt * ex - tauc * tauc * (1 - ex)
        if flag:
            C4 -= dt * ex
            C5 += 0.5 * dt * dt * ex
        umoments(Uh, lh, 7, mu, mup, mun)
        m = mup if side == 0 else mun
        xi2h = KINT / (2 * lh)
        xi4h = (KINT * KINT + 2 * KINT) / (4 * lh * lh)
        ah = solve_slope(Uh, lh, sh / rh)
        mpsi(1, m, xi2h, tmp)
        mpsi_a(2, m, xi2h, xi4h, ah, tmp2)
        for q in range(3):
            Ffr[q] += rh * (C4 * tmp[q] + C5 * tmp2[q])


# --------------------------------------------------------------------------
# Particles
# --------------------------------------------------------------------------
@njit(cache=True)
def locate(x, xf):
    """cell index of position x on the (uniform) face grid xf."""
    n = xf.shape[0] - 1
    i = int((x - xf[0]) / (xf[1] - xf[0]))
    if i < 0:
        i = 0
    if i > n - 1:
        i = n - 1
    return i


@njit(cache=True)
def stream(x, vx, vy, vz, t, xf, cyl):
    """
    Free flight for time t with specular walls.
    planar: walls at xf[0], xf[-1].
    cyl   : x = r, (vx, vy) = (v_r, v_theta), outer specular wall at xf[-1].
    Returns new (x, vx, vy).
    """
    x0 = xf[0]
    x1 = xf[xf.shape[0] - 1]
    if cyl == 0:
        xn = x + vx * t
        for _ in range(100):
            if xn < x0:
                xn = 2 * x0 - xn
                vx = -vx
            elif xn > x1:
                xn = 2 * x1 - xn
                vx = -vx
            else:
                break
        return xn, vx, vy
    # cylindrical: Cartesian plane (X, Y), start at (x, 0)
    X = x
    Y = 0.0
    VX = vx
    VY = vy
    rem = t
    R = x1
    for _ in range(100):
        # solve |P + V s|^2 = R^2 for s > 0
        aa = VX * VX + VY * VY
        if aa == 0.0:
            break
        bb = X * VX + Y * VY
        cc = X * X + Y * Y - R * R
        disc = bb * bb - aa * cc
        if disc < 0:
            disc = 0.0
        s = (-bb + sqrt(disc)) / aa
        if s >= rem:
            X += VX * rem
            Y += VY * rem
            break
        X += VX * s
        Y += VY * s
        rem -= s
        rr = sqrt(X * X + Y * Y)
        nx = X / rr
        ny = Y / rr
        vn = VX * nx + VY * ny
        VX -= 2 * vn * nx
        VY -= 2 * vn * ny
        # nudge inside
        X *= (1 - 1e-12)
        Y *= (1 - 1e-12)
    r = sqrt(X * X + Y * Y)
    if r > 0:
        vr = (X * VX + Y * VY) / r
        vt = (X * VY - Y * VX) / r
    else:
        vr = sqrt(VX * VX + VY * VY)
        vt = 0.0
    return r, vr, vt


@njit(cache=True)
def particle_step(px, pvx, pvy, pvz, pw, ptf, n, xf, cyl, V, dt,
                  dW, Wp_new, keep):
    """
    Stream all particles for their free time ptf. Deposit -start/+end
    moments into dW (per unit volume). Survivors (ptf >= dt) are flagged in
    keep and their moments accumulated into Wp_new.
    """
    for p in range(n):
        i0 = locate(px[p], xf)
        w = pw[p]
        e0 = 0.5 * (pvx[p] ** 2 + pvy[p] ** 2 + pvz[p] ** 2)
        dW[i0, 0] -= w / V[i0]
        dW[i0, 1] -= w * pvx[p] / V[i0]
        dW[i0, 2] -= w * e0 / V[i0]
        xn, vxn, vyn = stream(px[p], pvx[p], pvy[p], pvz[p], ptf[p], xf, cyl)
        px[p] = xn
        pvx[p] = vxn
        pvy[p] = vyn
        i1 = locate(xn, xf)
        e1 = 0.5 * (vxn ** 2 + vyn ** 2 + pvz[p] ** 2)
        dW[i1, 0] += w / V[i1]
        dW[i1, 1] += w * vxn / V[i1]
        dW[i1, 2] += w * e1 / V[i1]
        if ptf[p] >= dt:
            keep[p] = True
            Wp_new[i1, 0] += w / V[i1]
            Wp_new[i1, 1] += w * vxn / V[i1]
            Wp_new[i1, 2] += w * e1 / V[i1]
        else:
            keep[p] = False


@njit(cache=True)
def particle_moments(px, pvx, pvy, pvz, pw, n, xf, V, ncell):
    Wp = np.zeros((ncell, 3))
    for p in range(n):
        i = locate(px[p], xf)
        Wp[i, 0] += pw[p] / V[i]
        Wp[i, 1] += pw[p] * pvx[p] / V[i]
        Wp[i, 2] += pw[p] * 0.5 * (pvx[p] ** 2 + pvy[p] ** 2 + pvz[p] ** 2) / V[i]
    return Wp


@njit(cache=True)
def cell_sort(px, n, xf, ncell):
    """Return (order, start) so particles of cell i are order[start[i]:start[i+1]]."""
    cnt = np.zeros(ncell + 1, dtype=np.int64)
    idx = np.empty(n, dtype=np.int64)
    for p in range(n):
        i = locate(px[p], xf)
        idx[p] = i
        cnt[i + 1] += 1
    for i in range(ncell):
        cnt[i + 1] += cnt[i]
    start = cnt.copy()
    pos = cnt[:-1].copy()
    order = np.empty(n, dtype=np.int64)
    for p in range(n):
        i = idx[p]
        order[pos[i]] = p
        pos[i] += 1
    return order, start


@njit(cache=True)
def match_moments(sel, pvx, pvy, pvz, pw, mass, mom, energy):
    """
    Shift/scale the velocities (and uniformly rescale weights) of the
    particles listed in sel so that their total mass, x-momentum and energy
    equal (mass, mom, energy). Mean transverse velocities are set to zero.
    Returns False if impossible.
    """
    m = sel.shape[0]
    if m < 2 or mass <= 0:
        return False
    wsum = 0.0
    for j in range(m):
        wsum += pw[sel[j]]
    f = mass / wsum
    ux = 0.0; uy = 0.0; uz = 0.0
    for j in range(m):
        p = sel[j]
        pw[p] *= f
        ux += pw[p] * pvx[p]; uy += pw[p] * pvy[p]; uz += pw[p] * pvz[p]
    ux /= mass; uy /= mass; uz /= mass
    th = 0.0
    for j in range(m):
        p = sel[j]
        th += 0.5 * pw[p] * ((pvx[p] - ux) ** 2 + (pvy[p] - uy) ** 2 + (pvz[p] - uz) ** 2)
    U = mom / mass
    th_t = energy - 0.5 * mass * U * U
    if th_t <= 0 or th <= 0:
        return False
    s = sqrt(th_t / th)
    for j in range(m):
        p = sel[j]
        pvx[p] = U + s * (pvx[p] - ux)
        pvy[p] = s * (pvy[p] - uy)
        pvz[p] = s * (pvz[p] - uz)
    return True


@njit(cache=True)
def sample_cells(Wh, frac, nsamp, xf, cyl, V, out_x, out_vx, out_vy, out_vz, out_w):
    """Sample nsamp[i] Maxwellian particles carrying frac[i]*Wh[i] in cell i."""
    k = 0
    for i in range(Wh.shape[0]):
        m = nsamp[i]
        if m == 0:
            continue
        rho, U, lam = prim(Wh[i])
        s = sqrt(1.0 / (2.0 * lam))
        mass = frac[i] * Wh[i, 0] * V[i]
        w = mass / m
        for j in range(m):
            if cyl == 0:
                out_x[k + j] = xf[i] + (xf[i + 1] - xf[i]) * np.random.random()
            else:
                out_x[k + j] = sqrt(xf[i] ** 2 + (xf[i + 1] ** 2 - xf[i] ** 2) * np.random.random())
            out_vx[k + j] = U + s * np.random.standard_normal()
            out_vy[k + j] = s * np.random.standard_normal()
            out_vz[k + j] = s * np.random.standard_normal()
            out_w[k + j] = w
        sel = np.arange(k, k + m)
        match_moments(sel, out_vx, out_vy, out_vz, out_w,
                      mass, frac[i] * Wh[i, 1] * V[i], frac[i] * Wh[i, 2] * V[i])
        k += m
    return k


# --------------------------------------------------------------------------
# Reconstruction
# --------------------------------------------------------------------------
def vanleer(a, b):
    s = a * b
    with np.errstate(all='ignore'):
        out = np.where(s > 0, 2 * s / (a + b), 0.0)
    return out


def valid(W):
    rho = W[..., 0]
    eth = W[..., 2] - 0.5 * W[..., 1] ** 2 / np.where(rho > 0, rho, 1)
    return (rho > 0) & (eth > 1e-12 * np.abs(W[..., 2])) & np.isfinite(eth)


def mirror(W):
    g = W.copy()
    g[..., 1] *= -1
    return g


def reconstruct(W, xc, xf):
    """van Leer slopes on conservative variables with reflective ghosts.
    Returns slopes (n,3), face-left states (n+1,3), face-right states (n+1,3)."""
    Wg = np.vstack([mirror(W[:1]), W, mirror(W[-1:])])
    xg = np.concatenate([[2 * xf[0] - xc[0]], xc, [2 * xf[-1] - xc[-1]]])
    dl = (Wg[1:-1] - Wg[:-2]) / (xg[1:-1] - xg[:-2])[:, None]
    dr = (Wg[2:] - Wg[1:-1]) / (xg[2:] - xg[1:-1])[:, None]
    s = vanleer(dl, dr)
    Wm = W - s * (xc - xf[:-1])[:, None]      # value at left face of cell
    Wp = W + s * (xf[1:] - xc)[:, None]       # value at right face of cell
    bad = ~(valid(Wm) & valid(Wp))
    s[bad] = 0.0
    Wm[bad] = W[bad]
    Wp[bad] = W[bad]
    # face states
    WL = np.vstack([mirror(Wm[:1]), Wp])      # left state at face f
    WR = np.vstack([Wm, mirror(Wp[-1:])])     # right state at face f
    sL = np.vstack([s[:1] * np.array([-1, 1, -1]), s])
    sR = np.vstack([s, s[-1:] * np.array([-1, 1, -1])])
    return s, WL, WR, sL, sR


# --------------------------------------------------------------------------
# Species container and time step
# --------------------------------------------------------------------------
class Species:
    def __init__(self, xf, cyl, W0, tp, nc=200, nmin=8, cap=4_000_000, seed=0,
                 full_particle=False, rho_wave_only=0.0, RT_floor=0.0):
        self.xf = np.asarray(xf, float)
        self.cyl = int(cyl)
        self.n = len(xf) - 1
        self.xc = 0.5 * (self.xf[1:] + self.xf[:-1])
        if cyl:
            self.V = 0.5 * (self.xf[1:] ** 2 - self.xf[:-1] ** 2)
            self.A = self.xf.copy()
        else:
            self.V = np.diff(self.xf)
            self.A = np.ones_like(self.xf)
        self.W = np.array(W0, float)
        self.tp = np.array(tp, float)
        self.nc = nc
        self.nmin = nmin
        self.cap = cap
        self.full_particle = full_particle
        # cells below this density (numerical floors) are kept wave-only:
        # no hydro particles are sampled there and particles entering them
        # are absorbed into the wave part
        self.rho_wave_only = rho_wave_only
        # positivity guard: temperature floor applied where the update yields
        # a non-positive internal energy (energy added is accumulated)
        self.RT_floor = RT_floor
        self.floor_energy_added = 0.0
        self.floor_mass_removed = 0.0
        self.px = np.empty(cap); self.pvx = np.empty(cap)
        self.pvy = np.empty(cap); self.pvz = np.empty(cap)
        self.pw = np.empty(cap)
        self.np = 0
        np.random.seed(seed)
        _seed(seed)
        self.Wh = self.W.copy()
        if full_particle:
            self._convert_all_to_particles()

    # ---------------------------------------------------------------
    def tau_cells(self, W=None):
        W = self.W if W is None else W
        out = np.full(self.n, 1e30)
        ok = valid(W)
        for i in np.nonzero(ok)[0]:
            rho, U, lam = prim(W[i])
            out[i] = tau_of(rho, lam, self.tp)
        return out

    def Wp(self):
        return particle_moments(self.px, self.pvx, self.pvy, self.pvz, self.pw,
                                self.np, self.xf, self.V, self.n)

    def _append(self, x, vx, vy, vz, w):
        m = len(x)
        if self.np + m > self.cap:
            raise RuntimeError("particle capacity exceeded")
        s = slice(self.np, self.np + m)
        self.px[s] = x; self.pvx[s] = vx; self.pvy[s] = vy; self.pvz[s] = vz; self.pw[s] = w
        self.np += m

    def _convert_all_to_particles(self):
        frac = np.ones(self.n)
        nsamp = np.full(self.n, self.nc, dtype=np.int64)
        tot = int(nsamp.sum())
        bx = np.empty(tot); bvx = np.empty(tot); bvy = np.empty(tot); bvz = np.empty(tot); bw = np.empty(tot)
        k = sample_cells(self.Wh, frac, nsamp, self.xf, self.cyl, self.V, bx, bvx, bvy, bvz, bw)
        self._append(bx[:k], bvx[:k], bvy[:k], bvz[:k], bw[:k])
        self.Wh[:] = 0.0

    # ---------------------------------------------------------------
    def fix_wave(self):
        """Set W^h = W - W^p and repair cells where it is not admissible
        (or negligible) by matching the cell's particles to W."""
        Wp = self.Wp()
        Wh = self.W - Wp
        rho = self.W[:, 0]
        bad = (~valid(Wh)) | (Wh[:, 0] < 1e-4 * rho)
        # a wave residue whose temperature or velocity is far from the cell's
        # is round-off/statistical debris of W - W^p: hand it to the particles
        okW = valid(self.W)
        okh = valid(Wh) & okW
        with np.errstate(all='ignore'):
            r = np.where(okW, rho, 1.0)
            U = self.W[:, 1] / r
            RT = (self.W[:, 2] - 0.5 * r * U * U) / (1.5 * r)
            rh = np.where(okh, Wh[:, 0], 1.0)
            Uh = Wh[:, 1] / rh
            RTh = (Wh[:, 2] - 0.5 * rh * Uh * Uh) / (1.5 * rh)
            odd = okh & ((RTh > 10 * RT) | (RTh < 0.1 * RT) | (np.abs(Uh - U) > 3 * np.sqrt(RT)))
        bad |= odd
        bad &= Wp[:, 0] > 0
        if np.any(bad):
            order, start = cell_sort(self.px, self.np, self.xf, self.n)
            drop = np.zeros(self.np, dtype=np.bool_)
            for i in np.nonzero(bad)[0]:
                sel = order[start[i]:start[i + 1]]
                ok = match_moments(sel, self.pvx, self.pvy, self.pvz, self.pw,
                                   self.W[i, 0] * self.V[i], self.W[i, 1] * self.V[i],
                                   self.W[i, 2] * self.V[i])
                if not ok:
                    drop[sel] = True            # give everything back to the wave
            if np.any(drop):
                self._compact(~drop)
            Wh = self.W - self.Wp()
        Wh[~valid(Wh)] = 0.0                    # round-off residue of repaired cells
        self.Wh = Wh

    def _compact(self, keep):
        k = int(keep.sum())
        for arr in (self.px, self.pvx, self.pvy, self.pvz, self.pw):
            arr[:k] = arr[:self.np][keep]
        self.np = k

    # ---------------------------------------------------------------
    def sound_speed2(self):
        ok = valid(self.W)
        rho = np.where(ok, self.W[:, 0], 1.0)
        U = np.where(ok, self.W[:, 1] / rho, 0.0)
        RT = np.where(ok, (self.W[:, 2] - 0.5 * rho * U * U) / (1.5 * rho), 0.0)
        return U, GAM * RT

    def max_speed(self, extra_c2=0.0):
        U, c2 = self.sound_speed2()
        return np.max(np.abs(U) + 3.0 * np.sqrt(c2 + extra_c2))

    def step(self, dt):
        if self.full_particle:
            return self._step_particle_bgk(dt)
        n = self.n
        tau = self.tau_cells()
        tau = np.where(self.W[:, 0] < self.rho_wave_only, 1e-30, tau)
        ex = np.exp(-dt / tau)
        Wh = self.Wh
        hvalid = valid(Wh)
        rho = self.W[:, 0]

        # --- how many hydro particles to sample (fraction exp(-dt/tau) of W^h)
        want = self.nc * ex * np.where(hvalid, Wh[:, 0] / np.where(rho > 0, rho, 1.0), 0.0)
        nsamp = np.round(want).astype(np.int64)
        flag = (nsamp >= self.nmin) & hvalid
        nsamp[~flag] = 0

        # --- wave fluxes (equilibrium part from total W, free transport from W^h)
        s, WL, WR, sL, sR = reconstruct(self.W, self.xc, self.xf)
        Whr = np.where(hvalid[:, None], Wh, 0.0)
        hs, hWL, hWR, hsL, hsR = reconstruct(Whr, self.xc, self.xf)
        Wg = np.vstack([mirror(self.W[:1]), self.W, mirror(self.W[-1:])])
        xg = np.concatenate([[2 * self.xf[0] - self.xc[0]], self.xc, [2 * self.xf[-1] - self.xc[-1]]])
        dWdx = (Wg[1:] - Wg[:-1]) / (xg[1:] - xg[:-1])[:, None]
        # across strong density jumps the second-order (slope/time-derivative)
        # terms of the equilibrium flux can break positivity: use first order there
        with np.errstate(all='ignore'):
            rmin = np.minimum(Wg[1:, 0], Wg[:-1, 0])
            rmax = np.maximum(Wg[1:, 0], Wg[:-1, 0])
            steep = ~(rmax <= 2.0 * rmin)
        dWdx[steep] = 0.0
        tauL = np.concatenate([[tau[0]], tau])
        tauR = np.concatenate([tau, [tau[-1]]])
        flL = np.concatenate([[flag[0]], flag])
        flR = np.concatenate([flag, [flag[-1]]])
        hvL = np.concatenate([[hvalid[0]], hvalid])
        hvR = np.concatenate([hvalid, [hvalid[-1]]])
        F = _all_faces(WL, WR, dWdx, hWL, hWR, hsL, hsR, hvL, hvR, tauL, tauR,
                       flL, flR, dt, self.tp, self.A)
        dW = -(F[1:] - F[:-1]) / self.V[:, None]
        if self.cyl:
            # geometric (hoop) source of the wave part: + p_wave / r
            okW = valid(self.W)
            rs = np.where(okW, rho, 1.0)
            p_tot = np.where(okW, (self.W[:, 2] - 0.5 * self.W[:, 1] ** 2 / rs) / 1.5, 0.0)
            rh = np.where(hvalid, Wh[:, 0], 1.0)
            ph = np.where(hvalid, (Wh[:, 2] - 0.5 * Wh[:, 1] ** 2 / rh) / 1.5, 0.0)
            C1 = dt - tau * (1 - ex)
            C4 = tau * (1 - ex) - np.where(flag, dt * ex, 0.0)
            dW[:, 1] += (C1 * p_tot + C4 * ph) * np.diff(self.xf) / self.V

        # --- particles: free-flight times of existing particles, new hydro particles
        nold = self.np
        tf = np.empty(self.np + int(nsamp.sum()))
        if nold:
            ci = _cells_of(self.px, nold, self.xf)
            tc = -tau[ci] * np.log(np.random.random(nold))
            tf[:nold] = np.minimum(tc, dt)
        tot = int(nsamp.sum())
        if tot:
            bx = np.empty(tot); bvx = np.empty(tot); bvy = np.empty(tot); bvz = np.empty(tot); bw = np.empty(tot)
            k = sample_cells(Wh, ex, nsamp, self.xf, self.cyl, self.V, bx, bvx, bvy, bvz, bw)
            self._append(bx[:k], bvx[:k], bvy[:k], bvz[:k], bw[:k])
            tf[nold:self.np] = dt
        keep = np.zeros(self.np, dtype=np.bool_)
        Wp_new = np.zeros((n, 3))
        particle_step(self.px, self.pvx, self.pvy, self.pvz, self.pw, tf, self.np,
                      self.xf, self.cyl, self.V, dt, dW, Wp_new, keep)
        self.W = self.W + dW
        self._positivity()
        self._compact(keep)
        self.fix_wave()

    def _positivity(self):
        W = self.W
        neg = W[:, 0] <= 0
        if np.any(neg & (W[:, 0] < 0)):
            self.floor_mass_removed += float(-(W[neg, 0] * self.V[neg]).sum())
        W[neg] = 0.0
        pos = ~neg
        r = np.where(pos, W[:, 0], 1.0)
        Emin = 0.5 * W[:, 1] ** 2 / r + 1.5 * r * self.RT_floor
        low = pos & ((W[:, 2] - 0.5 * W[:, 1] ** 2 / r) <= 1e-12 * np.abs(W[:, 2]))
        if np.any(low):
            self.floor_energy_added += float(((Emin - W[:, 2])[low] * self.V[low]).sum())
            W[low, 2] = Emin[low]

    def _step_particle_bgk(self, dt):
        """Reference: pure stochastic particle BGK (all mass in particles).
        Free flight dt, then each particle collides with probability
        1 - exp(-dt/tau) and is re-drawn from the local Maxwellian;
        collided particles of a cell are moment-matched so that collisions
        conserve mass, momentum and energy exactly."""
        n = self.n
        tf = np.full(self.np, dt)
        keep = np.zeros(self.np, dtype=np.bool_)
        dW = np.zeros((n, 3)); Wp_new = np.zeros((n, 3))
        particle_step(self.px, self.pvx, self.pvy, self.pvz, self.pw, tf, self.np,
                      self.xf, self.cyl, self.V, dt, dW, Wp_new, keep)
        self.W = self.W + dW
        tau = self.tau_cells()
        _bgk_collide(self.px, self.pvx, self.pvy, self.pvz, self.pw, self.np,
                     self.xf, self.n, self.W, self.V, tau, dt)
        self.W = self.Wp()
        self.Wh = np.zeros_like(self.W)

    # ---------------------------------------------------------------
    def diagnostics(self):
        """rho, U, T(RT), particle mass fraction, Pi_xx/p, q_x/(p*sqrt(2RT))."""
        ok = valid(self.W)
        rho = np.where(ok, self.W[:, 0], np.nan)
        U = self.W[:, 1] / rho
        RT = (self.W[:, 2] - 0.5 * rho * U * U) / (1.5 * rho)
        p = rho * RT
        Pxx = np.zeros(self.n); Pyy = np.zeros(self.n); qx = np.zeros(self.n)
        _stress(self.px, self.pvx, self.pvy, self.pvz, self.pw, self.np, self.xf, self.V, U, Pxx, Pyy, qx)
        Wp = self.Wp()
        fp = Wp[:, 0] / rho
        # wave part is a Maxwellian (isotropic, no heat flux) in its own frame;
        # in the cell frame add rho_h (U_h - U)^2 to Pxx and its energy flux
        Wh = self.Wh
        okh = valid(Wh)
        rh = np.where(okh, Wh[:, 0], 0.0)
        Uh = np.where(okh, Wh[:, 1] / np.where(rh > 0, rh, 1), np.nan_to_num(U))
        RTh = np.where(okh, (Wh[:, 2] - 0.5 * rh * Uh ** 2) / (1.5 * np.where(rh > 0, rh, 1)), 0.0)
        c = Uh - U
        Pxx += rh * (RTh + c * c)
        Pyy += rh * RTh
        qx += rh * c * (0.5 * c * c + 2.5 * RTh)
        Pi_xx = (Pxx - p) / p
        Pi_yy = (Pyy - p) / p
        q = qx / (p * np.sqrt(2 * RT))
        return dict(x=self.xc, rho=rho, U=U, RT=RT, fp=fp, Pi_xx=Pi_xx, Pi_yy=Pi_yy, q=q,
                    nparticles=self.np)


@njit(cache=True)
def _bgk_collide(px, pvx, pvy, pvz, pw, n, xf, ncell, W, V, tau, dt):
    order, start = cell_sort(px, n, xf, ncell)
    for i in range(ncell):
        m = start[i + 1] - start[i]
        if m < 3:
            continue
        P = 1.0 - exp(-dt / tau[i])
        rho, U, lam = prim(W[i])
        if not (lam > 0):
            continue
        s = sqrt(1.0 / (2.0 * lam))
        col = np.empty(m, dtype=np.int64)
        k = 0
        for j in range(start[i], start[i + 1]):
            if np.random.random() < P:
                col[k] = order[j]
                k += 1
        if k < 3:
            continue
        sel = col[:k]
        mass = 0.0; mom = 0.0; en = 0.0
        for j in range(k):
            p = sel[j]
            mass += pw[p]; mom += pw[p] * pvx[p]
            en += 0.5 * pw[p] * (pvx[p] ** 2 + pvy[p] ** 2 + pvz[p] ** 2)
        for j in range(k):
            p = sel[j]
            pvx[p] = U + s * np.random.standard_normal()
            pvy[p] = s * np.random.standard_normal()
            pvz[p] = s * np.random.standard_normal()
        match_moments(sel, pvx, pvy, pvz, pw, mass, mom, en)


@njit(cache=True)
def _seed(s):
    np.random.seed(s)


@njit(cache=True)
def _cells_of(px, n, xf):
    out = np.empty(n, dtype=np.int64)
    for p in range(n):
        out[p] = locate(px[p], xf)
    return out


@njit(cache=True)
def _stress(px, pvx, pvy, pvz, pw, n, xf, V, U, Pxx, Pyy, qx):
    for p in range(n):
        i = locate(px[p], xf)
        cx = pvx[p] - U[i]
        w = pw[p] / V[i]
        c2 = cx * cx + pvy[p] ** 2 + pvz[p] ** 2
        Pxx[i] += w * cx * cx
        Pyy[i] += w * pvy[p] ** 2
        qx[i] += w * cx * 0.5 * c2


@njit(cache=True)
def _all_faces(WL, WR, dWdx, hWL, hWR, hsL, hsR, hvL, hvR, tauL, tauR, flL, flR, dt, tp, A):
    nf = WL.shape[0]
    F = np.zeros((nf, 3))
    Feq = np.empty(3); Ffr = np.empty(3)
    z = np.zeros(3)
    for f in range(nf):
        if A[f] == 0.0:
            continue
        hl = hWL[f] if hvL[f] else z
        hr = hWR[f] if hvR[f] else z
        face_fluxes(WL[f], WR[f], dWdx[f], hl, hr, hsL[f], hsR[f], tauL[f], tauR[f],
                    flL[f], flR[f], dt, tp, Feq, Ffr)
        for q in range(3):
            F[f, q] = A[f] * (Feq[q] + Ffr[q])
    return F

"""
Reference solvers for the planar validation:
  * dvm_bgk : deterministic discrete-velocity BGK solver (Chu-reduced, 3V
              monatomic), i.e. the full kinetic solution of the same model;
  * euler_exact : exact Riemann solution of the Euler equations (gamma=5/3).
"""
import numpy as np
from numba import njit

K = 2.0


@njit(cache=True)
def _moments(g, h, u, wu):
    n = g.shape[0]
    W = np.zeros((n, 3))
    for i in range(n):
        r = 0.0; m = 0.0; e = 0.0
        for k in range(u.shape[0]):
            r += wu[k] * g[i, k]
            m += wu[k] * u[k] * g[i, k]
            e += wu[k] * 0.5 * (u[k] * u[k] * g[i, k] + h[i, k])
        W[i, 0] = r; W[i, 1] = m; W[i, 2] = e
    return W


@njit(cache=True)
def _maxwell(W, u, G, H):
    n = W.shape[0]
    for i in range(n):
        rho = W[i, 0]; U = W[i, 1] / rho
        eth = W[i, 2] - 0.5 * rho * U * U
        lam = (K + 1) * rho / (4 * eth)
        for k in range(u.shape[0]):
            G[i, k] = rho * np.sqrt(lam / np.pi) * np.exp(-lam * (u[k] - U) ** 2)
            H[i, k] = K / (2 * lam) * G[i, k]


@njit(cache=True)
def _minmod(a, b):
    if a * b <= 0:
        return 0.0
    return a if abs(a) < abs(b) else b


@njit(cache=True)
def _transport(f, u, dx, dt):
    n, nv = f.shape
    out = f.copy()
    # ghost cells: specular walls f(-u) mirror
    fg = np.empty((n + 4, nv))
    fg[2:n + 2] = f
    for k in range(nv):
        kk = nv - 1 - k
        fg[1, k] = f[0, kk]; fg[0, k] = f[1, kk]
        fg[n + 2, k] = f[n - 1, kk]; fg[n + 3, k] = f[n - 2, kk]
    flux = np.zeros((n + 1, nv))
    for j in range(n + 1):
        jj = j + 1  # left cell index in fg
        for k in range(nv):
            if u[k] > 0:
                c = jj
                s = _minmod(fg[c, k] - fg[c - 1, k], fg[c + 1, k] - fg[c, k])
                fv = fg[c, k] + 0.5 * (1 - u[k] * dt / dx) * s
            else:
                c = jj + 1
                s = _minmod(fg[c, k] - fg[c - 1, k], fg[c + 1, k] - fg[c, k])
                fv = fg[c, k] - 0.5 * (1 + u[k] * dt / dx) * s
            flux[j, k] = u[k] * fv
    for i in range(n):
        for k in range(nv):
            out[i, k] = f[i, k] - dt / dx * (flux[i + 1, k] - flux[i, k])
    return out


def dvm_bgk(W0, x_faces, t_end, tp_mu, nv=161, umax=None, cfl=0.8):
    """tp_mu = (mu_ref, T_ref, omega, Rgas)."""
    from ugkwp import tau_of, prim
    n = len(x_faces) - 1
    dx = x_faces[1] - x_faces[0]
    RTmax = np.max((W0[:, 2] - 0.5 * W0[:, 1] ** 2 / W0[:, 0]) / (1.5 * W0[:, 0]))
    if umax is None:
        umax = 7.0 * np.sqrt(RTmax) + np.max(np.abs(W0[:, 1] / W0[:, 0]))
    u = np.linspace(-umax, umax, nv)
    wu = np.full(nv, u[1] - u[0])
    G = np.empty((n, nv)); H = np.empty((n, nv))
    _maxwell(W0, u, G, H)
    g, h = G.copy(), H.copy()
    tp = np.array([0.0, tp_mu[3], tp_mu[0], tp_mu[1], tp_mu[2], 1.0, 0.0])
    t = 0.0
    dt0 = cfl * dx / umax
    while t < t_end - 1e-14:
        dt = min(dt0, t_end - t)
        g = _transport(g, u, dx, dt)
        h = _transport(h, u, dx, dt)
        W = _moments(g, h, u, wu)
        _maxwell(W, u, G, H)
        tau = np.array([tau_of(prim(W[i])[0], prim(W[i])[2], tp) for i in range(n)])
        a = (dt / tau)[:, None]
        g = (g + a * G) / (1 + a)
        h = (h + a * H) / (1 + a)
        t += dt
    W = _moments(g, h, u, wu)
    # heat flux and stress for diagnostics
    rho = W[:, 0]; U = W[:, 1] / rho
    RT = (W[:, 2] - 0.5 * rho * U * U) / (1.5 * rho)
    c = u[None, :] - U[:, None]
    Pxx = np.sum(wu * c * c * g, axis=1)
    qx = np.sum(wu * c * 0.5 * (c * c * g + h), axis=1)
    p = rho * RT
    return dict(x=0.5 * (x_faces[1:] + x_faces[:-1]), rho=rho, U=U, RT=RT,
                Pi_xx=(Pxx - p) / p, q=qx / (p * np.sqrt(2 * RT)))


def euler_exact(x, x0, t, L, R, gam=5.0 / 3.0):
    """Exact Riemann solution; L, R = (rho, u, p)."""
    rl, ul, pl = L
    rr, ur, pr = R
    cl = np.sqrt(gam * pl / rl); cr = np.sqrt(gam * pr / rr)

    def fK(p, rk, pk, ck):
        if p > pk:
            A = 2 / ((gam + 1) * rk); B = (gam - 1) / (gam + 1) * pk
            return (p - pk) * np.sqrt(A / (p + B))
        return 2 * ck / (gam - 1) * ((p / pk) ** ((gam - 1) / (2 * gam)) - 1)

    lo, hi = 1e-10, 10 * max(pl, pr)
    for _ in range(200):
        pm = 0.5 * (lo + hi)
        if fK(pm, rl, pl, cl) + fK(pm, rr, pr, cr) + ur - ul > 0:
            hi = pm
        else:
            lo = pm
    um = 0.5 * (ul + ur) + 0.5 * (fK(pm, rr, pr, cr) - fK(pm, rl, pl, cl))
    out = np.zeros((len(x), 3))
    g1 = (gam - 1) / (gam + 1)
    for j, xx in enumerate(x):
        s = (xx - x0) / t
        if s < um:
            if pm > pl:
                rs = rl * ((pm / pl + g1) / (g1 * pm / pl + 1))
                S = ul - cl * np.sqrt((gam + 1) / (2 * gam) * pm / pl + (gam - 1) / (2 * gam))
                out[j] = (rl, ul, pl) if s < S else (rs, um, pm)
            else:
                rs = rl * (pm / pl) ** (1 / gam)
                cs = cl * (pm / pl) ** ((gam - 1) / (2 * gam))
                if s < ul - cl:
                    out[j] = (rl, ul, pl)
                elif s > um - cs:
                    out[j] = (rs, um, pm)
                else:
                    uu = 2 / (gam + 1) * (cl + (gam - 1) / 2 * ul + s)
                    c = 2 / (gam + 1) * (cl + (gam - 1) / 2 * (ul - s))
                    rho = rl * (c / cl) ** (2 / (gam - 1))
                    out[j] = (rho, uu, pl * (c / cl) ** (2 * gam / (gam - 1)))
        else:
            if pm > pr:
                rs = rr * ((pm / pr + g1) / (g1 * pm / pr + 1))
                S = ur + cr * np.sqrt((gam + 1) / (2 * gam) * pm / pr + (gam - 1) / (2 * gam))
                out[j] = (rs, um, pm) if s < S else (rr, ur, pr)
            else:
                rs = rr * (pm / pr) ** (1 / gam)
                cs = cr * (pm / pr) ** ((gam - 1) / (2 * gam))
                if s > ur + cr:
                    out[j] = (rr, ur, pr)
                elif s < um + cs:
                    out[j] = (rs, um, pm)
                else:
                    uu = 2 / (gam + 1) * (-cr + (gam - 1) / 2 * ur + s)
                    c = 2 / (gam + 1) * (cr - (gam - 1) / 2 * (ur - s))
                    rho = rr * (c / cr) ** (2 / (gam - 1))
                    out[j] = (rho, uu, pr * (c / cr) ** (2 * gam / (gam - 1)))
    return dict(rho=out[:, 0], U=out[:, 1], RT=out[:, 2] / out[:, 0])

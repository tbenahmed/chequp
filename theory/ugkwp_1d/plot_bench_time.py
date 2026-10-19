"""
1e24 m^-3 benchmark over the full run (0-2 ns, every 0.1 ns), zoomed to
r <= 50 um:
  bench_1e24_profiles.png : radial profiles at six times
  bench_1e24_rt.png       : r-t maps (kinetic reference, UGKWP, fluid) and
                            the fluid and UGKWP deviations from the reference
"""
import glob
import warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, TwoSlopeNorm

warnings.filterwarnings('ignore')
EV = 11604.518
RMAX = 50.0


def load(kind):
    return [np.load(f) for f in sorted(glob.glob(f'bench_{kind}_s*.npz'))]


S = {k: load(k) for k in ('ref', 'ugkwp', 'fluid')}
times = sorted({float(k.split('/')[0]) for k in S['ref'][0].files if '/' in k})
key = lambda t: f'{t:.3e}'
r = S['ref'][0][f'{key(0.0)}/r'] * 1e6
m = r <= RMAX


def ens(kind, t, q):
    return np.nanmean([d[f'{key(t)}/{q}'] for d in S[kind]], axis=0)


def field(kind, q):
    return np.array([ens(kind, t, q) for t in times])     # (nt, nr)


# --------------------------------------------------------------------------
# 1) radial profiles at selected times
TSEL = [0.0, 0.5e-9, 1.0e-9, 2.0e-9, 3.0e-9, 4.0e-9]
ST = {'ref': dict(lw=4, alpha=0.3, color=None), 'ugkwp': dict(lw=1.4), 'fluid': dict(lw=1.4, ls='--')}
LAB = {'ref': 'full-particle BGK, kinetic reference (4 seeds)', 'ugkwp': 'UGKWP (4 seeds)',
       'fluid': 'two-fluid Euler'}
fig, ax = plt.subplots(6, len(TSEL), figsize=(4.2 * len(TSEL), 19), sharex=True, sharey='row')
for j, t in enumerate(TSEL):
    for k in S:
        st = {kk: v for kk, v in ST[k].items() if v is not None}
        g = lambda q: ens(k, t, q)[m]
        rr = r[m]
        ax[0, j].plot(rr, g('n_i'), color='tab:blue', **st)
        ax[1, j].plot(rr, g('n_n'), color='tab:blue', **st)
        ax[2, j].semilogy(rr, g('Te') / EV, color='tab:blue', **st)
        ax[2, j].semilogy(rr, g('T_i') / EV, color='tab:red', **st)
        ax[2, j].semilogy(rr, g('T_n') / EV, color='tab:green', **st)
        ax[3, j].plot(rr, g('U_i') * 1e-3, color='tab:red', **st)
        ax[3, j].plot(rr, g('U_n') * 1e-3, color='tab:green', **st)
        ax[4, j].plot(rr, g('Pirr_n'), color='tab:green', **st)
        ax[4, j].plot(rr, g('Pitt_n'), color='tab:purple', **st)
        if k == 'ugkwp':
            ax[5, j].plot(rr, g('fp_i'), color='tab:red', **st)
            ax[5, j].plot(rr, g('fp_n'), color='tab:green', **st)
    ax[0, j].set_title(f't = {t * 1e9:.1f} ns', fontsize=14, fontweight='bold')
ax[0, 0].set_ylabel('$n_e = n_{H^+}$ [m$^{-3}$]')
ax[1, 0].set_ylabel('$n_{H0}$ [m$^{-3}$]')
ax[2, 0].set_ylabel('T [eV]\nblue $T_e$, red $T_i$, green $T_n$')
ax[3, 0].set_ylabel('radial velocity [km/s]\nred H$^+$, green H$^0$')
ax[4, 0].set_ylabel('H$^0$ stress anisotropy\n' + r'green $\Pi_{rr}/p$, purple $\Pi_{\theta\theta}/p$')
ax[5, 0].set_ylabel('UGKWP mass fraction\nin particles\nred H$^+$, green H$^0$')
ax[0, 0].set_ylim(0, 1.05e24)
ax[1, 0].set_ylim(0, 2.3e24)
ax[2, 0].set_ylim(1.5e-2, 4)
ax[4, 0].set_ylim(-1, 1.5)
ax[5, 0].set_ylim(-0.05, 1.05)
for a in ax.flat:
    a.grid(alpha=0.3)
    a.set_xlim(0, RMAX)
for a in ax[4]:
    a.axhline(0, color='0.6', lw=0.6)
for a in ax[-1]:
    a.set_xlabel(r'$r$ [$\mu$m]')
handles = [plt.Line2D([], [], color='0.3', **{kk: v for kk, v in ST[k].items() if v is not None}) for k in S]
fig.legend(handles, [LAB[k] for k in S], loc='upper center', ncol=3, fontsize=13, bbox_to_anchor=(0.5, 1.0))
fig.tight_layout(rect=(0, 0, 1, 0.98))
fig.savefig('bench_1e24_profiles.png', dpi=85)

# --------------------------------------------------------------------------
# 2) r-t maps
T = np.array(times) * 1e9
qty = [('n_i', '$n_e$ [m$^{-3}$]', 1.0, 'Blues', dict(vmin=0, vmax=1e24)),
       ('n_n', '$n_{H0}$ [m$^{-3}$]', 1.0, 'Greens', dict(vmin=0, vmax=2e24)),
       ('Te', '$T_e$ [eV]', 1 / EV, 'Oranges', dict(norm=LogNorm(0.03, 2.5))),
       ('T_i', '$T_i$ [eV]', 1 / EV, 'Reds', dict(norm=LogNorm(0.03, 2.5))),
       ('T_n', '$T_n$ [eV]', 1 / EV, 'Purples', dict(norm=LogNorm(0.03, 1.0)))]
cols = ['ref', 'ugkwp', 'fluid']
titles = ['kinetic reference', 'UGKWP', 'two-fluid Euler', 'UGKWP vs reference', 'fluid vs reference']
fig, ax = plt.subplots(len(qty), 5, figsize=(23, 4.0 * len(qty)), sharex=True, sharey=True)
for i, (q, lab, sc, cmap, kw) in enumerate(qty):
    F = {k: field(k, q)[:, m] * sc for k in cols}
    for j, k in enumerate(cols):
        im = ax[i, j].pcolormesh(r[m], T, F[k], cmap=cmap, shading='nearest', **kw)
        if j == 2:
            fig.colorbar(im, ax=ax[i, :3].tolist(), pad=0.01, fraction=0.02).set_label(lab)
    floor = 1e-2 * 1e24 if q.startswith('n') else 0
    for j, k in ((3, 'ugkwp'), (4, 'fluid')):
        with np.errstate(all='ignore'):
            rel = np.log(F[k] / F['ref'])
        if q == 'n_i':
            rel = np.where(F['ref'] > 1e22, rel, np.nan)
        if q == 'T_i':
            rel = np.where(field('ref', 'n_i')[:, m] > 1e22, rel, np.nan)
        im2 = ax[i, j].pcolormesh(r[m], T, rel, cmap='RdBu_r', shading='nearest',
                                  norm=TwoSlopeNorm(0, -1, 1))
    fig.colorbar(im2, ax=ax[i, 3:].tolist(), pad=0.01, fraction=0.03).set_label(r'$\ln(y/y_{ref})$')
for j, tl in enumerate(titles):
    ax[0, j].set_title(tl, fontsize=14, fontweight='bold')
for a in ax[:, 0]:
    a.set_ylabel('t [ns]')
for a in ax[-1]:
    a.set_xlabel(r'$r$ [$\mu$m]')
fig.suptitle(r'H, $n_0=10^{24}$ m$^{-3}$, Bessel-ring initial state: full run, $r\leq50\,\mu$m '
             '(ion quantities masked where $n_e<10^{22}$ m$^{-3}$ in the difference maps)', y=1.0, fontsize=14)
fig.savefig('bench_1e24_rt.png', dpi=80, bbox_inches='tight')
print('ok', len(times), 'snapshots')

"""
Plot the 1e24 m^-3 benchmark: UGKWP vs full-particle reference vs two-fluid
Euler, in the layout of the Castro/WarpX comparison (n_e, n_H0, temperatures)
plus ion-neutral slip, neutral non-equilibrium and UGKWP particle fraction.
"""
import glob
import warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
EV = 11604.518
SNAP = ['0.000e+00', '5.000e-10', '1.000e-09', '2.000e-09']


def load(kind):
    return [np.load(f) for f in sorted(glob.glob(f'bench_{kind}_s*.npz'))]


def ens(runs, key):
    return np.nanmean([r[key] for r in runs], axis=0)


S = {k: load(k) for k in ('ref', 'ugkwp', 'fluid')}
S = {k: v for k, v in S.items() if v}
ST = {'ref': dict(lw=4, alpha=0.3), 'ugkwp': dict(lw=1.3), 'fluid': dict(lw=1.3, ls='--')}
LAB = {'ref': 'full-particle BGK (kinetic reference)', 'ugkwp': 'UGKWP', 'fluid': 'two-fluid Euler'}

fig, ax = plt.subplots(6, len(SNAP), figsize=(5 * len(SNAP), 21), sharex=True)
for j, ts in enumerate(SNAP):
    for k, runs in S.items():
        g = lambda q: ens(runs, f'{ts}/{q}')
        r = runs[0][f'{ts}/r'] * 1e6
        st = ST[k]
        ax[0, j].plot(r, g('n_i'), color='tab:blue', **st)
        ax[1, j].plot(r, g('n_n'), color='tab:blue', **st)
        ax[2, j].semilogy(r, g('Te') / EV, color='tab:blue', **st)
        ax[2, j].semilogy(r, g('T_i') / EV, color='tab:red', **st)
        ax[2, j].semilogy(r, g('T_n') / EV, color='tab:green', **st)
        ax[3, j].plot(r, g('slip'), color='tab:green', **st)
        ax[4, j].plot(r, g('Pirr_n'), color='tab:blue', **st)
        ax[4, j].plot(r, g('Pitt_n'), color='tab:purple', **st)
        ax[4, j].plot(r, g('q_n'), color='tab:orange', **st)
        if k == 'ugkwp':
            ax[5, j].plot(r, g('fp_i'), color='tab:red', **st)
            ax[5, j].plot(r, g('fp_n'), color='tab:green', **st)
    ax[0, j].set_title(f't = {float(ts) * 1e9:.1f} ns', fontsize=14, fontweight='bold')
ax[0, 0].set_ylabel('$n_e = n_{H^+}$ [m$^{-3}$]')
ax[1, 0].set_ylabel('$n_{H0}$ [m$^{-3}$]')
ax[2, 0].set_ylabel('T [eV]\nblue $T_e$, red $T_i$, green $T_n$')
ax[3, 0].set_ylabel(r'slip $(U_i-U_n)/v_{th,n}$')
ax[4, 0].set_ylabel('H$^0$ non-equilibrium\n' + r'blue $\Pi_{rr}/p$, purple $\Pi_{\theta\theta}/p$,'
                    '\n' + r'orange $q_r/(p v_{th})$')
ax[5, 0].set_ylabel('UGKWP particle mass fraction\nred H$^+$, green H$^0$')
for a in ax[0]:
    a.set_ylim(0, 1.15e24)
for a in ax[1]:
    a.set_ylim(0, 3.05e24)
for a in ax[2]:
    a.set_ylim(1e-2, 4)
for a in ax[3]:
    a.set_ylim(-1, 4)
for a in ax[4]:
    a.set_ylim(-1, 2)
    a.axhline(0, color='0.6', lw=0.6)
for a in ax[5]:
    a.set_ylim(-0.05, 1.05)
for a in ax.flat:
    a.grid(alpha=0.3)
    a.set_xlim(0, 83)
for a in ax[-1]:
    a.set_xlabel(r'$r$ [$\mu$m]')
handles = [plt.Line2D([], [], color='k', **ST[k]) for k in S]
fig.legend(handles, [LAB[k] + (f' ({len(S[k])} seeds)' if k != 'fluid' else '') for k in S],
           loc='upper center', ncol=3, fontsize=13, bbox_to_anchor=(0.5, 1.0))
fig.tight_layout(rect=(0, 0, 1, 0.985))
fig.savefig('bench_1e24.png', dpi=90)
for k, runs in S.items():
    a = np.array([r['info'] for r in runs])
    print(f'{k:6s} cpu {a[:, 0].mean():8.1f} s  steps {a[:, 1].mean():8.0f}  mean particles {a[:, 2].mean():8.0f}  '
          f'mass {a[:, 3].mean():.6e}')

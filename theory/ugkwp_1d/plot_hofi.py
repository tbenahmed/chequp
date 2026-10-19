"""
Plots for the HOFI-like channel:
  hofi_fine.png   : fine grid (0.5 um): full-particle reference, UGKWP, two-fluid Euler
  hofi_coarse.png : coarse grid (2 um) UGKWP / particle / fluid vs. fine reference
and an L1 error table vs. the fine-grid reference.
"""
import glob
import warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
SNAP = ['8.600e-10', '1.730e-09']
EV = 11604.518


def load(kind, n):
    return [np.load(f) for f in sorted(glob.glob(f'hofi_{kind}_n{n}_s*.npz'))]


def ens(runs, key):
    return np.nanmean([r[key] for r in runs], axis=0)


def info(runs):
    a = np.array([r['info'] for r in runs])
    return a[:, 0].mean(), a[:, 1].mean(), a[:, 2].mean()


SETS = {
    'ref200': load('ref', 200), 'ugkwp200': load('ugkwp', 200), 'fluid200': load('fluid', 200),
    'ref50': load('ref', 50), 'ugkwp50': load('ugkwp', 50), 'fluid50': load('fluid', 50),
}
STYLE = {
    'ref200': dict(lw=3.5, alpha=0.35, ls='-'),
    'ugkwp200': dict(lw=1.3, ls='-'),
    'fluid200': dict(lw=1.3, ls='--'),
    'ugkwp50': dict(lw=1.6, ls='-', marker='o', ms=3),
    'ref50': dict(lw=1.0, ls=':'),
    'fluid50': dict(lw=1.3, ls='--'),
}
LABEL = {
    'ref200': 'full-particle BGK, 0.5 µm (reference)',
    'ugkwp200': 'UGKWP, 0.5 µm',
    'fluid200': 'two-fluid Euler, 0.5 µm',
    'ugkwp50': 'UGKWP, 2 µm',
    'ref50': 'full-particle BGK, 2 µm',
    'fluid50': 'two-fluid Euler, 2 µm',
}


def figure(keys, fname, title):
    fig, ax = plt.subplots(6, 2, figsize=(14, 21), sharex=True)
    for j, ts in enumerate(SNAP):
        for key in keys:
            runs = SETS[key]
            r = runs[0][f'{ts}/r'] * 1e6
            st = STYLE[key]
            g = lambda q: ens(runs, f'{ts}/{q}')
            ax[0, j].semilogy(r, g('n_n'), color='tab:blue', **st)
            ax[0, j].semilogy(r, g('n_i'), color='tab:red', **st)
            ax[1, j].semilogy(r, g('T_n') / EV, color='tab:blue', **st)
            ax[1, j].semilogy(r, g('T_i') / EV, color='tab:red', **st)
            ax[1, j].semilogy(r, g('Te') / EV, color='tab:orange', **st)
            ax[2, j].plot(r, g('U_n') * 1e-3, color='tab:blue', **st)
            ax[2, j].plot(r, g('U_i') * 1e-3, color='tab:red', **st)
            ax[3, j].plot(r, g('slip'), color='tab:green', **st)
            ax[4, j].plot(r, g('Pirr_n'), color='tab:blue', **st)
            ax[4, j].plot(r, g('Pitt_n'), color='tab:purple', **st)
            ax[4, j].plot(r, g('q_n'), color='tab:orange', **st)
            if key.startswith('ugkwp'):
                ax[5, j].plot(r, g('fp_n'), color='tab:blue', **st)
                ax[5, j].plot(r, g('fp_i'), color='tab:red', **st)
        ax[0, j].set_title(f't = {float(ts) * 1e9:.2f} ns')
    ax[0, 0].set_ylabel('density [m$^{-3}$]\n(blue H$^0$, red H$^+$)')
    ax[1, 0].set_ylabel('temperature [eV]\n(blue $T_n$, red $T_i$, orange $T_e$)')
    ax[2, 0].set_ylabel('radial velocity [km/s]\n(blue H$^0$, red H$^+$)')
    ax[3, 0].set_ylabel(r'ion-neutral slip $(U_i-U_n)/v_{th,n}$')
    ax[4, 0].set_ylabel('neutral non-equilibrium\n' + r'blue $\Pi_{rr}/p$, purple $\Pi_{\theta\theta}/p$,'
                        '\n' + r'orange $q_r/(p\,v_{th})$')
    ax[5, 0].set_ylabel('UGKWP mass fraction\ncarried by particles\n(blue H$^0$, red H$^+$)')
    for a in ax[0]:
        a.set_ylim(1e18, 3e23)
    for a in ax[1]:
        a.set_ylim(0.015, 10)
    for a in ax[3]:
        a.set_ylim(-1, 6)
    for a in ax[4]:
        a.set_ylim(-1.5, 3)
        a.axhline(0, color='0.6', lw=0.6)
    for a in ax[5]:
        a.set_ylim(-0.05, 1.05)
    for a in ax.flat:
        a.grid(alpha=0.3)
        a.set_xlim(0, 100)
    for a in ax[-1]:
        a.set_xlabel(r'r [$\mu$m]')
    handles = [plt.Line2D([], [], color='k', **STYLE[k]) for k in keys]
    fig.legend(handles, [LABEL[k] for k in keys], loc='upper center', ncol=len(keys), fontsize=11,
               bbox_to_anchor=(0.5, 0.995))
    fig.suptitle(title, y=1.012, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(fname, dpi=100, bbox_inches='tight')


figure(['ref200', 'ugkwp200', 'fluid200'], 'hofi_fine.png',
       'HOFI-like H channel, $n_0=10^{23}$ m$^{-3}$, $T_{e0}$=5 eV: kinetic reference vs UGKWP vs fluid '
       '(mean of 4 seeds)')
figure(['ref200', 'ugkwp50', 'ref50', 'fluid50'], 'hofi_coarse.png',
       'Coarse grid (2 µm) vs fine-grid kinetic reference: does UGKWP keep the kinetic answer?')

# ---- error table vs fine reference (r < 60 um, where the plasma is)
ref = SETS['ref200']
rows = []
for key in ['ugkwp200', 'fluid200', 'ugkwp50', 'ref50', 'fluid50']:
    errs = []
    for ts in SNAP:
        rr = ref[0][f'{ts}/r']
        # compare only where the species is physically present (> 1e-3 n0 in
        # the reference), i.e. not in the artificial 1e-4 n0 density floors
        pres = {sp: ens(ref, f'{ts}/n_{sp}') > 1e-3 * 1e23 for sp in 'in'}
        for q, log in (('n_i', 1), ('n_n', 1), ('T_i', 1), ('T_n', 1), ('U_i', 0), ('Pirr_n', 0)):
            m = pres[q.split('_')[1][0]]
            yref = ens(ref, f'{ts}/{q}')
            y = np.interp(rr, SETS[key][0][f'{ts}/r'], ens(SETS[key], f'{ts}/{q}'))
            if log:
                e = np.nanmean(np.abs(np.log(y[m] / yref[m])))
            elif q == 'U_i':
                e = np.nanmean(np.abs(y[m] - yref[m])) / np.nanmax(np.abs(yref[m]))
            else:
                e = np.nanmean(np.abs(y[m] - yref[m]))
            errs.append(e)
    cpu, steps, npm = info(SETS[key])
    rows.append((key, errs, cpu, steps, npm))
cpu, steps, npm = info(ref)
hdr = ['n_i', 'n_n', 'T_i', 'T_n', 'U_i', 'Pi_rr,n']
with open('hofi_errors.txt', 'w') as f:
    line = f"{'case':<10}" + ''.join(f'{h + "@" + t:>14}' for t in ('0.86', '1.73') for h in hdr) + \
           f"{'cpu[s]':>9}{'steps':>7}{'particles':>11}"
    print(line); f.write(line + '\n')
    line = f"{'ref200':<10}" + ''.join(f'{"-":>14}' for _ in range(12)) + f'{cpu:9.1f}{steps:7.0f}{npm:11.0f}'
    print(line); f.write(line + '\n')
    for key, errs, cpu, steps, npm in rows:
        line = f'{key:<10}' + ''.join(f'{e:14.3f}' for e in errs) + f'{cpu:9.1f}{steps:7.0f}{npm:11.0f}'
        print(line); f.write(line + '\n')
    f.write('\nerrors vs ref200 (mean of 4 seeds each): n, T -> mean |ln(y/y_ref)|; '
            'U_i -> mean |dU| / max|U_ref|; Pi_rr -> mean |dPi|;\n'
            'evaluated where the species density exceeds 1e-3 n0 in the reference\n')

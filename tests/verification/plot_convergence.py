"""
Plot time-convergence of one field (rel. L2 error vs finest dt) from the JSON files
written by analyze_time_convergence.py.

usage: python plot_convergence.py [--qty=Te] out.png label1=res1.json[@time] label2=res2.json[@time] ...
       (qty: Te, rho, f, ion, vr; time in seconds; default: last common output time)
"""
import sys, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

args = sys.argv[1:]
qty = 'Te'
if args[0].startswith('--qty='):
    qty = args.pop(0).split('=', 1)[1]
NAMES = {'Te': 'Te', 'rho': 'density', 'f': 'f = E_h/E', 'ion': 'ionization fraction', 'vr': 'v_r'}

SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']   # categorical slots 1-4
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'

fig, ax = plt.subplots(figsize=(7, 4.6), dpi=150)
fig.patch.set_facecolor('#fcfcfb'); ax.set_facecolor('#fcfcfb')
for i, spec in enumerate(args[1:]):
    label, _, rest = spec.rpartition('=')
    path, _, t = rest.partition('@')
    res = json.load(open(path))
    t = t or f"{res['times'][-1]:g}"
    dt = np.array(res['dt'][:-1]); e = np.array(res['err_ref'][qty][t])
    c = SERIES[i % len(SERIES)]
    ax.loglog(dt, e, '-o', color=c, lw=2, ms=6, mec='#fcfcfb', mew=2, label=label)
# first-order reference slope
x = np.array([1.5e-12, 1.5e-10]); ax.loglog(x, 3e-3 * x / x[0], '--', color=INK2, lw=1)
ax.text(3e-12, 3e-3 * 3e-12 / 1.5e-12 * 0.62, 'slope 1', color=INK2, fontsize=9, rotation=27)
ax.set_xlabel('time step dt [s]', color=INK2); ax.set_ylabel(f'{NAMES[qty]} rel. L2 error vs finest dt', color=INK2)
ax.grid(True, which='major', color=GRID, lw=0.8); ax.tick_params(colors=INK2)
for s in ax.spines.values(): s.set_color(GRID)
ax.legend(frameon=False, fontsize=9, labelcolor=INK)
ax.set_title(f'Implicit (JFNK) conduction: time convergence of {NAMES[qty]}', color=INK, fontsize=11, loc='left')
fig.tight_layout(); fig.savefig(args[0], facecolor=fig.get_facecolor())

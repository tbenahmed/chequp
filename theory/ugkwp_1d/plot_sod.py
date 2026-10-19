import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

d = json.load(open('results_sod.json'))
cases = ['0.0001', '0.01', '1.0']
qty = [('rho', r'$\rho$'), ('U', r'$U$'), ('RT', r'$RT$'), ('q', r'$q_x/(p\sqrt{2RT})$'), ('fp', 'particle mass fraction')]
fig, ax = plt.subplots(len(qty), 3, figsize=(13, 13), sharex=True)
for j, c in enumerate(cases):
    r = d[c]['ref']; u = d[c]['ugkwp']; e = d['euler']
    for i, (k, lab) in enumerate(qty):
        a = ax[i, j]
        if k in e and k != 'q':
            a.plot(e['x'], e[k], color='0.6', lw=1, ls='--', label='Euler (exact)')
        if k in r:
            a.plot(r['x'], r[k], 'k', lw=1.5, label='DVM-BGK (full kinetic)')
        a.plot(u['x'], u[k], 'o', ms=2.0, color='tab:red', alpha=0.45, label='UGKWP (one run)')
        a.plot(u['x'], d[c]['ens'][k], '-', lw=1.2, color='tab:red', label='UGKWP (mean of 8 seeds)')
        if i == 0:
            a.set_title(f"Kn = {float(c):g}   ({u['nstep']} steps, {u['cpu']:.1f} s)")
        if j == 0:
            a.set_ylabel(lab)
        a.grid(alpha=0.3)
    ax[-1, j].set_xlabel('x')
    ax[-1, j].set_ylim(-0.05, 1.05)
ax[0, 0].legend(fontsize=8)
fig.suptitle('Sod shock tube, BGK (VHS $\\omega$=0.81), t = 0.15: UGKWP vs full kinetic', y=0.995)
fig.tight_layout()
fig.savefig('sod_validation.png', dpi=130)

# L1 errors vs DVM (interpolated on UGKWP grid)
for c in cases:
    r = d[c]['ref']; u = d[c]['ugkwp']
    msg = []
    for k in ['rho', 'U', 'RT']:
        ri = np.interp(u['x'], r['x'], r[k])
        e1 = np.mean(np.abs(np.array(u[k]) - ri)) / np.mean(np.abs(ri))
        e8 = np.mean(np.abs(np.array(d[c]['ens'][k]) - ri)) / np.mean(np.abs(ri))
        msg.append(f"{k}: 1 run {e1:.2%} / 8 seeds {e8:.2%}")
    print(c, ' '.join(msg))

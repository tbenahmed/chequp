"""
Density / shock-front diagnostics vs dt for a dt-refinement series.

At each common output time prints, for every dt:
  R_peak  : radius of the density maximum (cell resolution)
  R_front : sub-cell shock-front radius, where rho crosses (rho_max + rho_0)/2
            on the outer side of the peak (linear interpolation)
  rho_max / rho_0
plus the relative L2 density error vs the finest dt.
In 2D the radial profile is taken at the axial mid-plane (z = 300 um).

usage: python density_front.py <run_prefix> [rho_0 = 1.67e-6]
"""
import os, sys, glob
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plotfile_reader import read_plotfile, fields, by_time

def radial_profile(pf):
    rho = fields(pf)['rho']
    dx = (pf['hi'][0] - pf['lo'][0]) / pf['shape'][0]
    r = pf['lo'][0] + (np.arange(pf['shape'][0]) + 0.5) * dx
    return r, (rho[:, rho.shape[1] // 2] if rho.ndim == 2 else rho), rho

def front(r, prof, rho0):
    i = int(np.argmax(prof)); half = 0.5 * (prof[i] + rho0)
    for j in range(i, len(prof) - 1):
        if prof[j] >= half > prof[j + 1]:
            return r[j] + (prof[j] - half) / (prof[j] - prof[j + 1]) * (r[j + 1] - r[j]), r[i], prof[i]
    return np.nan, r[i], prof[i]

prefix = sys.argv[1]; rho0 = float(sys.argv[2]) if len(sys.argv) > 2 else 1.67e-6
runs = sorted(glob.glob(prefix + '*'), key=lambda p: -float(p[len(prefix):]))
tb = [by_time(r) for r in runs]
for t in [t for t in sorted(set.intersection(*[set(b) for b in tb])) if t > 0]:
    data = [radial_profile(read_plotfile(b[t])) for b in tb]
    ref = data[-1][2]; w = (np.arange(ref.shape[0]) + 0.5)
    w = w[:, None] * np.ones(ref.shape[1:])[None, :] if ref.ndim == 2 else w
    print(f"t = {t*1e9:g} ns")
    for run, (r, prof, full) in zip(runs, data):
        Rf, Rp, pk = front(r, prof, rho0)
        err = np.sqrt(np.sum(w * (full - ref)**2) / np.sum(w * ref**2))
        print(f"  dt={float(run[len(prefix):]):9.3g}  R_peak={Rp*1e4:6.1f} um  R_front={Rf*1e4:7.2f} um"
              f"  rho_max/rho_0={pk/rho0:6.3f}  rho L2 err={err:.2e}")

"""
Check that a z-uniform 2D rz run reproduces the 1D cylindrical run
(same radial grid and dt): prints max |1D - 2D| / max|1D| and the z
non-uniformity of the 2D solution at each common output time.

usage: python compare_1d_2d.py <run_1d> <run_2d>
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plotfile_reader import read_plotfile, fields, by_time

a, b = by_time(sys.argv[1]), by_time(sys.argv[2])
for t in sorted(set(a) & set(b)):
    if t == 0:
        continue
    f1, f2 = fields(read_plotfile(a[t])), fields(read_plotfile(b[t]))
    out = []
    for q in ['Te', 'rho', 'f', 'ion']:
        q2 = f2[q]
        dz = np.abs(q2 - q2[:, :1]).max() / np.abs(q2).max()
        d12 = np.abs(f1[q][:, None] - q2).max() / np.abs(f1[q]).max()
        out.append(f"{q}: 1D-vs-2D {d12:.1e}, z-nonuniformity {dz:.1e}")
    print(f"t = {t*1e9:g} ns  " + "; ".join(out))

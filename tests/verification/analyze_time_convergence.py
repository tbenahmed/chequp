"""
Time-convergence analysis of a dt-refinement series.

For a series of runs <prefix><dt> (same grid, fixed dt), prints at every
common output time:
  - the relative L2 error of each field against the finest-dt run,
  - the observed order p = log2(d_k / d_{k+1}) from successive differences
    d_k = |u(dt_k) - u(dt_{k+1})|, which does not rely on the reference.
L2 norms are weighted by the cylindrical cell volume (proportional to r).
v_r is normalised by max|v_r| of the reference (it changes sign).

usage: python analyze_time_convergence.py <run_prefix> [out.json]
   e.g. python analyze_time_convergence.py work/runs/h1d_dt res_h1d.json
"""
import os, sys, glob, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plotfile_reader import read_plotfile, fields, by_time

QTY = ['Te', 'rho', 'f', 'ion', 'vr']

def r_weight(a):
    r = np.arange(a.shape[0]) + 0.5
    return r[:, None] * np.ones(a.shape[1:])[None, :] if a.ndim == 2 else r

def error(a, b, q, vmax):
    w = r_weight(b)
    if q == 'vr':
        return np.sqrt(np.sum(w * (a - b)**2) / np.sum(w)) / vmax if vmax > 0 else np.nan
    return np.sqrt(np.sum(w * (a - b)**2) / np.sum(w * b**2))

def main(prefix, out=None):
    runs = sorted(glob.glob(prefix + '*'), key=lambda p: -float(p[len(prefix):]))
    dts = [float(r[len(prefix):]) for r in runs]
    tb = [by_time(r) for r in runs]
    times = [t for t in sorted(set.intersection(*[set(t) for t in tb])) if t > 0]
    res = {'dt': dts, 'times': times, 'err_ref': {}, 'diff_succ': {}, 'order': {}}
    print(f"{prefix}: dt = {dts}")
    for t in times:
        F = [fields(read_plotfile(b[t])) for b in tb]
        ref = F[-1]
        print(f"\n t = {t*1e9:g} ns  rel. L2 error vs finest dt ({dts[-1]:.3g} s)"
              f" | observed order from successive differences")
        for q in QTY:
            vmax = np.abs(ref['vr']).max()
            e = [error(F[i][q], ref[q], q, vmax) for i in range(len(dts) - 1)]
            d = [error(F[i][q], F[i+1][q], q, vmax) for i in range(len(dts) - 1)]
            p = [np.log2(d[i] / d[i+1]) if d[i+1] > 0 else np.nan for i in range(len(d) - 1)]
            for k, v in (('err_ref', e), ('diff_succ', d), ('order', p)):
                res[k].setdefault(q, {})[f"{t:g}"] = [float(x) for x in v]
            if np.all(np.array(e) == 0) or np.any(np.isnan(e)):
                continue  # field not evolved in this configuration
            print(f"  {q:4s} " + " ".join(f"{x:9.2e}" for x in e)
                  + "  | p = " + " ".join(f"{x:5.2f}" for x in p))
    if out:
        with open(out, 'w') as f:
            json.dump(res, f, indent=1)

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)

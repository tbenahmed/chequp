"""
Validation: Sod shock tube (planar, BGK, VHS omega=0.81) at three Knudsen
numbers. UGKWP vs. deterministic DVM-BGK (full kinetic) and exact Euler.
"""
import sys
import time
import json
import numpy as np
from ugkwp import Species
from reference import dvm_bgk, euler_exact

OMEGA = 0.81
T_END = 0.15
NSEED = 8


def mu_ref_for(Kn):
    # VHS reference viscosity giving mean free path Kn (rho=1, RT=1, L=1)
    # (Bird: lambda = 2(5-2w)(7-2w)/15 * mu / (rho sqrt(2 pi RT)))
    return Kn * 15 * np.sqrt(2 * np.pi) / (2 * (5 - 2 * OMEGA) * (7 - 2 * OMEGA))


def initial(xf):
    xc = 0.5 * (xf[1:] + xf[:-1])
    rho = np.where(xc < 0.5, 1.0, 0.125)
    p = np.where(xc < 0.5, 1.0, 0.1)
    W = np.zeros((len(xc), 3))
    W[:, 0] = rho
    W[:, 2] = 1.5 * p
    return W


def run_ugkwp(Kn, ncell=200, nc=400, seed=1, cfl=0.5, full=False):
    xf = np.linspace(0, 1, ncell + 1)
    tp = [0.0, 1.0, mu_ref_for(Kn), 1.0, OMEGA, 1.0, 0.0]
    sp = Species(xf, 0, initial(xf), tp, nc=nc, seed=seed, full_particle=full)
    t = 0.0
    nstep = 0
    t0 = time.time()
    npart = []
    while t < T_END - 1e-12:
        dt = cfl * (xf[1] - xf[0]) / sp.max_speed()
        if t + dt > T_END - 1e-12:
            dt = T_END - t
        elif t + 2 * dt > T_END:
            dt = 0.5 * (T_END - t)      # avoid a tiny last step
        sp.step(dt)
        t += dt
        nstep += 1
        npart.append(sp.np)
    d = sp.diagnostics()
    d['cpu'] = time.time() - t0
    d['nstep'] = nstep
    d['np_mean'] = float(np.mean(npart))
    return d


def main():
    out = {}
    for Kn in [1e-4, 1e-2, 1.0]:
        print(f"Kn={Kn}", flush=True)
        ref = dvm_bgk(initial(np.linspace(0, 1, 1001)), np.linspace(0, 1, 1001), T_END,
                      (mu_ref_for(Kn), 1.0, OMEGA, 1.0), nv=201)
        runs = [run_ugkwp(Kn, seed=s) for s in range(1, NSEED + 1)]
        u = runs[0]
        print(f"  UGKWP: {u['nstep']} steps, {u['cpu']:.1f}s, mean particles {u['np_mean']:.0f}", flush=True)
        ens = {k: np.mean([r[k] for r in runs], axis=0).tolist()
               for k in ['rho', 'U', 'RT', 'q', 'Pi_xx', 'fp']}
        out[str(Kn)] = dict(ref={k: np.asarray(v).tolist() for k, v in ref.items()},
                            ugkwp={k: (np.asarray(v).tolist() if isinstance(v, np.ndarray) else v)
                                   for k, v in u.items()},
                            ens=ens)
    xe = np.linspace(0, 1, 1000)
    ex = euler_exact(xe, 0.5, T_END, (1, 0, 1), (0.125, 0, 0.1))
    out['euler'] = dict(x=xe.tolist(), **{k: v.tolist() for k, v in ex.items()})
    json.dump(out, open('results_sod.json', 'w'))


if __name__ == '__main__':
    main()

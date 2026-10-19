"""
CHEQUP/WarpX 1D benchmark case (n0 = 1e24 m^-3, Bessel-ring ionization,
digitized initial state) with CHEQUP electron physics:
UGKWP, full-particle BGK reference, two-fluid Euler.

usage: python3 run_bench.py {ugkwp|ref|fluid} [seed]
"""
import sys
import time
import numpy as np
from hofi import HOFIChannel, MH
from benchmark_1e24 import profile_pchip as profile, N0, T_HEAVY

SNAP = list(np.round(np.arange(0, 4.0001e-9, 0.1e-9), 13))


def tau_min_ions(ch):
    tau = ch.ion.tau_cells()
    mask = ch.ion.W[:, 0] / MH > 1e-2 * N0
    return tau[mask].min() if mask.any() else np.inf


def run(kind, seed):
    full = kind == 'ref'
    ch = HOFIChannel(R=120e-6, ncell=240, n0=N0, T0=T_HEAVY, profile=profile, electrons='chequp',
                     seed=seed, nc=100, full_particle=full, fluid=(kind == 'fluid'))
    out = {}
    t0 = time.time()
    npart = []
    for ts in SNAP:
        while ch.t < ts - 1e-18:
            dt = ch.dt_cfl()
            if full:
                dt = min(dt, 0.5 * tau_min_ions(ch))
            rem = ts - ch.t
            dt = rem if dt >= rem else (0.5 * rem if rem < 1.3 * dt else dt)
            ch.step(dt)
            npart.append(ch.ion.np + ch.neu.np)
            if ch.nsteps % 2000 == 0:
                print(f"[{kind} s{seed}] t={ch.t * 1e9:.3f} ns steps={ch.nsteps} particles={npart[-1]} "
                      f"cpu={time.time() - t0:.0f}s", flush=True)
        for k, v in ch.diagnostics().items():
            out[f'{ts:.3e}/{k}'] = np.asarray(v)
    info = np.array([time.time() - t0, ch.nsteps, np.mean(npart) if npart else 0,
                     ch.ion.W[:, 0] @ ch.V + ch.neu.W[:, 0] @ ch.V])
    print(kind, seed, dict(cpu=info[0], steps=info[1], np_mean=info[2], mass=info[3]), flush=True)
    np.savez(f'bench_{kind}_s{seed}.npz', info=info, **out)


if __name__ == '__main__':
    run(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 1)

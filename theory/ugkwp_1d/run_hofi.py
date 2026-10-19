"""
HOFI-like channel (1D cylindrical, H, n0 = 1e23 m^-3): UGKWP vs. a
full-particle BGK reference using the identical physical model.
"""
import sys
import time
import numpy as np
from hofi import HOFIChannel, MH

SNAP = [0.86e-9, 1.73e-9]


def tau_min_ions(ch):
    tau = ch.ion.tau_cells()
    mask = ch.ion.W[:, 0] / MH > 1e-2 * 1e23
    return tau[mask].min() if mask.any() else np.inf


def run(full, seed=1, nc=100, ncell=200, label='', fluid=False):
    ch = HOFIChannel(full_particle=full, seed=seed, nc=nc, ncell=ncell, fluid=fluid)
    snaps = {}
    t0 = time.time()
    npart = []
    k = 0
    for ts in SNAP:
        while ch.t < ts - 1e-18:
            dt = ch.dt_cfl()
            if full:
                dt = min(dt, 0.5 * tau_min_ions(ch))
            dt = min(dt, ts - ch.t)
            if ts - ch.t - dt < 0.3 * dt and ts - ch.t > dt:
                dt = 0.5 * (ts - ch.t)
            ch.step(dt)
            npart.append(ch.ion.np + ch.neu.np)
            k += 1
            if k % 200 == 0:
                print(f"  [{label}] t={ch.t*1e9:.3f} ns  steps={ch.nsteps}  particles={npart[-1]}  "
                      f"cpu={time.time()-t0:.0f}s", flush=True)
        snaps[ts] = ch.diagnostics()
    return snaps, dict(cpu=time.time() - t0, steps=ch.nsteps, np_mean=float(np.mean(npart)),
                       np_max=int(np.max(npart)),
                       floor_energy_J_per_rad=ch.ion.floor_energy_added + ch.neu.floor_energy_added,
                       floor_mass_kg_per_rad=ch.ion.floor_mass_removed + ch.neu.floor_mass_removed,
                       total_mass_kg_per_rad=float(ch.ion.W[:, 0] @ ch.V + ch.neu.W[:, 0] @ ch.V))


if __name__ == '__main__':
    which = sys.argv[1]
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    ncell = int(sys.argv[3]) if len(sys.argv) > 3 else 200
    snaps, info = run(which == 'ref', seed=seed, ncell=ncell, label=which, fluid=(which == 'fluid'))
    print(which, info)
    out = {'info': info}
    for ts, d in snaps.items():
        for k, v in d.items():
            out[f'{ts:.3e}/{k}'] = v
    np.savez(f'hofi_{which}_n{ncell}_s{seed}.npz', **{k: np.asarray(v) for k, v in out.items() if k != 'info'},
             info=np.array([info['cpu'], info['steps'], info['np_mean'], info['np_max']]))

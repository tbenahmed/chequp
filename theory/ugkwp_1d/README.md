# 1D UGKWP prototype for HOFI channels

Prototype of the **Unified Gas-Kinetic Wave-Particle (UGKWP)** method
(Liu, Zhu & Xu, JCP 2020; Liu & Xu, Adv. Aerodyn. 2021) in 1D planar and
cylindrical geometry, written to test whether a multiscale kinetic method
could replace or complement the fluid description in CHEQUP.
Python + numba, standalone (not coupled to Castro).

| file | content |
|---|---|
| `ugkwp.py` | UGKWP core: GKS equilibrium flux, analytic free transport of the wave part, particle tracking (planar / exact cylindrical), BGK collisions; also a pure particle-BGK mode used as the kinetic reference and a fluid mode (`tau -> 0`, the gas-kinetic Euler limit) |
| `reference.py` | deterministic discrete-velocity BGK solver (full kinetic) and exact Euler Riemann solver |
| `run_sod.py`, `plot_sod.py` | validation on the Sod shock tube at Kn = 1e-4, 1e-2, 1 |
| `hofi.py` | two kinetic species (H+, H) + electron fluid, charge exchange, ionization, e-i equilibration, ambipolar field |
| `run_hofi.py`, `run_all_hofi.sh`, `plot_hofi.py` | HOFI-like channel: UGKWP vs full-particle reference vs two-fluid Euler |

```
python3 run_sod.py && python3 plot_sod.py               # ~40 s
./run_all_hofi.sh 200 && ./run_all_hofi.sh 50 && python3 plot_hofi.py   # ~3 min
```

## 1. Validation: Sod shock tube (`sod_validation.png`)

Same BGK model (VHS, omega = 0.81) solved by UGKWP (200 cells, 400 particles
per cell) and by a deterministic DVM (1000 cells x 201 velocities).
UGKWP matches the full-kinetic solution from the near-continuum regime
(Kn = 1e-4, where it approaches the exact Euler solution) to the
free-molecular regime (Kn = 1), including the heat flux.
Averaging 8 seeds reduces the L1 error by ~sqrt(8)
(e.g. density 4.1% -> 1.4%), i.e. the remaining difference is statistical
noise, not bias. Mass is conserved to machine precision.

## 2. HOFI-like channel (`hofi_fine.png`, `hofi_coarse.png`, `hofi_errors.txt`)

Hydrogen, n0 = 1e23 m^-3, ionized core exp(-(r/12 um)^4), Te0 = 5 eV,
T_i = T_n = 300 K, outputs at 0.86 and 1.73 ns.

Model (identical in the three solvers):
* H+ and H are kinetic species (BGK self-collisions: Coulomb for ions,
  VHS H-H for neutrals);
* resonant charge exchange (sigma = 5e-19 m^2) acting on the distribution:
  a neutral that charge-exchanges is re-drawn from the ion Maxwellian, which
  produces the fast, non-Maxwellian CX neutrals;
* electron-impact ionization (Voronov) moves neutral particles, with their
  velocity, into the ion species, and costs 13.6 eV of electron energy;
* electrons: quasi-neutral fluid, adiabatic advection with the ions,
  e-i equilibration, ambipolar force -grad(p_e) on the ions.

Solvers: **full-particle BGK** reference (dt <= tau_ii/2), **UGKWP**, and
**two-fluid Euler** (the same code with tau -> 0, i.e. separate ion and neutral
fluids coupled by charge exchange: already more general than CHEQUP's
single fluid).

### Results

* **UGKWP agrees with the kinetic reference within statistical noise** on
  densities, velocities, temperatures, ion-neutral slip and the neutral
  non-equilibrium (Pi_rr/p ~ 1.3, Pi_tt/p ~ -0.6, q/(p v_th) > 2 from CX
  neutrals). Errors (0.07-0.27 in |ln| units) are at the level of the
  reference's own seed-to-seed scatter (0.13-0.21).
* **The fluid model departs systematically**, beyond noise:
  - neutral temperature in the core ~2-5x too high (in the kinetic solution
    the core neutrals are cold atoms streaming in from the wall, which a
    fluid cannot represent);
  - neutral density near the axis ~5-10x too high at 1.73 ns;
  - ion-neutral slip underestimated by ~1/3 at 5-20 um;
  - sharp ion front instead of the extended fast-ion tail;
  - zero neutral stress/heat flux by construction.
* **In this regime UGKWP is essentially a particle method**: more than 90%
  of both species' mass is carried by particles wherever they are physically
  present. tau_ii (tens of ps once the ions are heated) and the CX and H-H
  times (~ns) all exceed dt, and the cell Knudsen number is O(1).
  Accordingly, the full-particle BGK solver is as accurate and as cheap
  here, also on the 4x coarser grid. UGKWP's advantage, the automatic,
  conservative switch to the fluid limit without a Kn threshold or
  interface, will only pay off where part of the domain is collisional
  (higher density, colder/denser plasma, longer times, larger cells).

### Limitations of this prototype

BGK collision operators (Pr = 1, no Fokker-Planck for ions); constant CX
cross-section; no electron heat conduction, electron-neutral collisions,
recombination or radiation; ion/neutral floors (1e-4 n0) are kept wave-only;
electron energy equation is non-conservative (T_e advection); not coupled
to Castro/AMReX. The particle reference loses ~3e-5 of the total mass over
the run (UGKWP and fluid conserve to round-off).

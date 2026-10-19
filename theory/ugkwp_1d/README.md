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
| `chequp_rates.py` | CHEQUP hydrogen rates (ionization, 3-body recombination, nu_ei, nu_en, conductivity), with an accurate low-T_e evaluation |
| `benchmark_1e24.py`, `run_bench.py`, `plot_bench.py` | CHEQUP/WarpX 1e24 m^-3 benchmark case (Bessel-ring initial state) |

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

## 3. CHEQUP/WarpX benchmark case, n0 = 1e24 m^-3 (`bench_1e24.png`)

Initial state **digitized** from the t = 0 Castro/WarpX comparison plot
(Bessel-ring ionization: n_e = 1e24 m^-3 on axis, rings at 13 and 23 um;
T_e = 2.3 / 0.65 / 0.33 eV; T_heavy = 0.03 eV) — replace
`benchmark_1e24.profile` by the real profile when available.
Electron physics switched to CHEQUP's (`electrons='chequp'`): implicit
electron conduction with the CHEQUP conductivity, e-i and e-n energy
exchange (nu_ei, nu_en of arXiv:2305.16779), CHEQUP ionization
(direct + excitation channel) and Saha 3-body recombination; recombining ions
become neutral particles with their velocity. Outputs at 0, 0.5, 1, 2 ns,
r in [0, 120] um, 0.5 um cells.

| | cpu / run | steps | particles |
|---|---|---|---|
| full-particle BGK reference | 36 s | 1588 | 48 000 |
| UGKWP | 3.9 s | 362 | 26 000 |
| two-fluid Euler | 1.5 s | 350 | 0 |

* At this density the ions are collisional in the core: **UGKWP carries only
  20-60% of the ion mass as particles there and 100% at the expanding ion
  front**, so here it is genuinely multiscale and ~9x cheaper than the
  particle reference (it does not need to resolve tau_ii ~ 0.05 ps).
* **UGKWP agrees with the kinetic reference within statistical noise**
  (errors 0.01-0.08 in |ln| units vs. reference seed-to-seed scatter
  0.01-0.10).
* **Two-fluid Euler is fine until ~1 ns but departs at 2 ns**: the neutral
  shock piles up to n_H0 = 2.0e24 m^-3 (kinetic: 1.2e24, a broad
  front), core T_n = 0.31 eV (kinetic 0.13-0.15 eV), T_e on axis 15-20%
  high; errors on T_i, T_n, T_e and n_i are 2-8x the noise level.
* The fall of n_e on axis (1e24 -> 2e23 m^-3 by 1 ns) is expansion, not
  recombination: the number of electrons per unit length changes by <0.3%.

### Issue found in CHEQUP's rates

`collisional_ionization.H` interpolates the direct ionization rate
linearly on a log-T table and integrates the Maxwellian with a 1 eV step.
At low T_e the rate varies by ~e^56 between nodes, so it is overestimated by
1.7x (1 eV), 14x (0.3 eV), 4e7x (0.1 eV); `get_rate_3body` multiplies it by
exp(chi/T_e), so 3-body recombination is overestimated by the same factors
(and is non-monotonic in T_e). `chequp_rates.py` tabulates the smooth
G = S exp(chi/T) with a T-resolved quadrature instead (identical to the
CHEQUP table above ~3 eV); this prototype uses that version.

### Limitations of this prototype

BGK collision operators (Pr = 1, no Fokker-Planck for ions); constant CX
cross-section (5e-19 m^2); no radiation; electron conduction,
e-n collisions and recombination only in the `electrons='chequp'` case; ion/neutral floors (1e-4 n0) are kept wave-only;
electron energy equation is non-conservative (T_e advection); not coupled
to Castro/AMReX. The particle reference loses ~3e-5 of the total mass over
the run (UGKWP and fluid conserve to round-off).

"""
Initial conditions for the time-convergence / reproducibility study of the
implicit (JFNK) electron thermal diffusion scheme.

Simple H plasma expansion, DESY/COMSOL-like profile (27 eV super-Gaussian Te,
ionized core, rho = 1.67e-6 g/cm^3), written in openPMD format:
  ic_1d.h5           : 1D cylindrical (r)
  ic_2d_uniformz.h5  : 2D rz, same radial profile, uniform along z
                       (must reproduce the 1D solution)
  ic_2d_column.h5    : 2D rz, finite-length plasma column (super-Gaussian
                       envelope in z, centred at z = 300 um)

Usage: python make_ic.py [output_dir]
"""
import os, sys, re
import numpy as np
from scipy.constants import atomic_mass

here = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(here, "../../initial_condition"))
from ionization_routines import save_to_openpmd

out = sys.argv[1] if len(sys.argv) > 1 else "."
m_u_cgs = atomic_mass * 1e3
with open(os.path.join(here, "../../sim_folder/build/species.net")) as f:
    species = re.findall(r'\n\s.*\s([A-Z][a-z]*\d)', f.read())

sigma1, sigma2, Te_max = 38e-6, 35e-6, 27.0   # m, m, eV
Ta = 2000 * 8.617333262145e-5                 # background temperature in eV
n_total = (1.67e-6 / m_u_cgs) * 1e6           # m^-3

def profiles(r):
    ion = (1 - 1e-3) * np.exp(-np.power(r * r / (2 * sigma1**2), 12)) + 1e-3
    Te = Te_max * np.exp(-np.power(r * r / (2 * sigma2**2), 3)) + Ta
    return ion, Te

def densities(ion):
    d = np.zeros(ion.shape + (len(species),))
    d[..., species.index('H0')] = (1 - ion) * n_total
    d[..., species.index('H1')] = ion * n_total
    return d

# 1D
r = np.arange(0, 6e-4 + 1e-6, 1e-6)
ion, Te = profiles(r)
save_to_openpmd({'r': [r.min(), r.max()]}, densities(ion), Te,
                os.path.join(out, 'ic_1d.h5'), species)

# 2D rz, uniform in z
r = np.linspace(0, 6e-4, 601); z = np.linspace(0, 6e-4, 7)
R, Z = np.meshgrid(r, z, indexing='ij')
ion, Te = profiles(R)
save_to_openpmd({'r': [r.min(), r.max()], 'z': [z.min(), z.max()]}, densities(ion), Te,
                os.path.join(out, 'ic_2d_uniformz.h5'), species)

# 2D rz, finite plasma column
r = np.linspace(0, 6e-4, 301); z = np.linspace(0, 6e-4, 301)
R, Z = np.meshgrid(r, z, indexing='ij')
envz = np.exp(-np.power((Z - 3e-4)**2 / (2 * (100e-6)**2), 3))
ion_r, Te_r = profiles(R)
ion = (ion_r - 1e-3) * envz + 1e-3
Te = (Te_r - Ta) * envz + Ta
save_to_openpmd({'r': [r.min(), r.max()], 'z': [z.min(), z.max()]}, densities(ion), Te,
                os.path.join(out, 'ic_2d_column.h5'), species)

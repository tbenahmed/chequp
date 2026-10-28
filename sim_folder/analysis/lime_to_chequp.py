"""
Convert LIME openPMD diagnostics (openpmd_*.h5) into a 2D CHEQUP input file.

Everything is read from the LIME output files themselves:
  - r axis : from the openPMD grid attributes (gridSpacing, gridGlobalOffset, gridUnitSI)
  - z axis : one point per output file, from its "time" attribute
             (z = c * t, or z = t if LIME already stores the distance there)

For every output file, the first row of the stored mode-0 array is read
(identical to np.array(ds[0]).T[:, 0]) and stacked along z,
so each field has shape (Nz, Nr). Only that row is read from disk.
The densities are then mapped onto the CHEQUP species list (read from
species.net) and written with save_to_openpmd.

Species: "H" (H0, H1), "N" (N0..N5), "Ar" (Ar0..Ar8), "He" (He0..He2),
or a list of them. Te_eV is always included.

Typical usage
-------------
    from lime_to_chequp import lime_to_chequp

    data = lime_to_chequp("path/to/Test_LIME/diags/hdf5", "CHEQUP_sim/2d_input.h5", species=["H", "N"], plot=True)
"""
import glob
import os
import re

import h5py
import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as spc

try:
    from tqdm.auto import tqdm
except ImportError:                                  # no tqdm: plain iteration
    def tqdm(it, **kwargs):
        return it
sys.path.append("/data/dust/user/benahmed/src/Codes/chequp/initial_condition")
from ionization_routines import save_to_openpmd 

SPECIES_NET = "/data/dust/user/benahmed/src/Codes/chequp/sim_folder/build/species.net"

# ---------------------------------------------------------------------------
# Species definitions:
#   dataset : LIME dataset name pattern
#   levels  : output name -> ion level
#   factor  : multiplier applied to the density (H comes from H2 -> 2 atoms)
# Adjust "dataset" if your LIME species are named differently.
# ---------------------------------------------------------------------------
SPECIES = {
    "H":  {"dataset": "n_H2_ionlevel_{}", "factor": 2, "levels": {"n_H0": 0, "n_H1": 2}},
    "N":  {"dataset": "n_N_ionlevel_{}",  "factor": 1, "levels": {f"n_N{k}":  k for k in range(6)}},  # N0..N5
    "Ar": {"dataset": "n_Ar_ionlevel_{}", "factor": 1, "levels": {f"n_Ar{k}": k for k in range(9)}},  # Ar0..Ar8
    "He": {"dataset": "n_He_ionlevel_{}", "factor": 1, "levels": {f"n_He{k}": k for k in range(3)}},  # He0..He2
}
TE_NAME, TE_KEY = "Te_eV", "temperature_eV_electron"


def _species_list(species):
    return [species] if isinstance(species, str) else list(species)


def _field_map(species):
    """{output_name: dataset_key} for the requested species, plus Te."""
    fields = {}
    for sp in _species_list(species):
        if sp not in SPECIES:
            raise ValueError(f"Unknown species '{sp}'. Choose from {list(SPECIES)}")
        pattern = SPECIES[sp]["dataset"]
        for name, lvl in SPECIES[sp]["levels"].items():
            fields[name] = pattern.format(lvl)
    fields[TE_NAME] = TE_KEY
    return fields


def _sorted_files(input_dir):
    """openpmd_<digits>.h5 files sorted by step (other diagnostics are ignored)."""
    pat = re.compile(r"openpmd_(\d+)\.h5$")
    files = [p for p in glob.glob(os.path.join(input_dir, "openpmd_*.h5"))
             if pat.search(os.path.basename(p))]
    if not files:
        raise FileNotFoundError(f"No openpmd_<step>.h5 files in {input_dir}")
    return sorted(files, key=lambda p: int(pat.search(os.path.basename(p)).group(1)))


def _str(x):
    return x.decode() if isinstance(x, bytes) else str(x)


def _r_axis(ds):
    """r coordinates (m) of the kept axis (last storage axis) from openPMD attributes."""
    n = ds.shape[-1]
    labels = [_str(l) for l in ds.attrs.get("axisLabels", [])]
    if _str(ds.attrs.get("dataOrder", "C")) == "F":
        labels = labels[::-1]
    i = len(labels) - 1                              # attrs are per axis, last = kept axis
    if labels and labels[i] != "r":
        print(f"Warning: kept axis is labelled '{labels[i]}', not 'r' (labels {labels})")
    spacing = np.atleast_1d(ds.attrs["gridSpacing"])
    offset = np.atleast_1d(ds.attrs["gridGlobalOffset"])
    pos = np.atleast_1d(ds.attrs.get("position", np.zeros(len(spacing))))
    if _str(ds.attrs.get("dataOrder", "C")) == "F":
        spacing, offset, pos = spacing[::-1], offset[::-1], pos[::-1]
    unit = ds.attrs.get("gridUnitSI", 1.0)
    r = (offset[i] + spacing[i] * (np.arange(n) + pos[i])) * unit
    r_edges = (offset[i] * unit, (offset[i] + spacing[i] * n) * unit)
    return r, r_edges


def read_species_net(path=SPECIES_NET):
    """CHEQUP species keys (e.g. ['H0', 'H1', 'N0', ...]) from species.net."""
    with open(path, "r") as f:
        return re.findall(r"\n\s.*\s([A-Z][a-z]*\d)", f.read())


def load_lime(input_dir, species="H"):
    """
    Stack the kept profile of every openpmd_*.h5 along z.

    Returns dict with "z" (Nz,), "r" (Nr,), "r_edges", "time", "iteration"
    and one (Nz, Nr) array per field.
    """
    fields = _field_map(species)
    files = _sorted_files(input_dir)
    Nz = len(files)

    # Grid and size from the first file
    with h5py.File(files[0], "r") as f:
        meshes = f["data"][next(iter(f["data"]))]["meshes"]
        missing = [k for k in fields.values() if k not in meshes]
        if missing:
            raise KeyError(f"Missing in {files[0]}: {missing}\n"
                           f"Available: {sorted(meshes.keys())}")
        r, r_edges = _r_axis(meshes[TE_KEY])
    Nr = len(r)

    stack = {name: np.empty((Nz, Nr)) for name in fields}
    times = np.empty(Nz)
    iters = np.empty(Nz, dtype=int)
    for i, fp in enumerate(tqdm(files, desc="Reading LIME outputs", unit="file")):
        with h5py.File(fp, "r") as f:
            it = next(iter(f["data"]))               # iteration number, e.g. "500"
            grp = f["data"][it]
            meshes = grp["meshes"]
            iters[i] = int(it)
            times[i] = grp.attrs["time"] * grp.attrs.get("timeUnitSI", 1.0)
            for name, key in fields.items():
                stack[name][i] = meshes[key][0, 0, :]    # == np.array(ds[0]).T[:, 0]

    # z of each output: LIME may store z/c (seconds) or z (metres) in "time"
    z = times * spc.c
    if abs(z[-1]) > 10:                              # > 10 m: "time" was already a distance
        z = times.copy()
        print("z axis: 'time' attribute used directly as distance")
    else:
        print("z axis: z = c * time")

    data = {"z": z, "r": r, "r_edges": r_edges, "time": times, "iteration": iters}
    data.update(stack)                               # (Nz, Nr)
    return data


def lime_to_chequp(input_dir="path_to_lime_output/diags/hdf5",
                   output="CHEQUP_sim/2d_input.h5",
                   species="H",
                   species_net=SPECIES_NET,
                   density_floor=1e18,
                   te_offset=0.06,
                   plot=False):
    """
    Read LIME openPMD outputs and write the 2D CHEQUP input file.

    Parameters
    ----------
    input_dir     : folder containing openpmd_*.h5
    output        : CHEQUP input .h5 path (None to skip writing)
    species       : "H", "N", "Ar", "He" or a list, e.g. ["H", "N"]
    species_net   : path to CHEQUP species.net
    density_floor : added to every requested density (avoids zeros)
    te_offset     : added to Te_eV (eV)
    plot          : plot each LIME field (x axis = z, y axis = r)

    Returns
    -------
    dict with "z", "r", "time", "iteration", one (Nz, Nr) array per LIME field,
    "densities" (Nz, Nr, Nspecies), "species_keys" and "grid".
    """
    data = load_lime(input_dir, species)
    grid = {"r": list(data["r_edges"]), "z": [data["z"][0], data["z"][-1]]}
    print(f"Grid from h5: z = [{grid['z'][0]:.3e}, {grid['z'][1]:.3e}] m ({len(data['z'])} outputs), "
          f"r = [{grid['r'][0]:.3e}, {grid['r'][1]:.3e}] m ({len(data['r'])} cells)")

    species_keys = read_species_net(species_net)

    # Map LIME densities onto the CHEQUP species axis in one step
    names = [n for sp in _species_list(species) for n in SPECIES[sp]["levels"]]
    skipped = [n[2:] for n in names if n[2:] not in species_keys]
    if skipped:
        print(f"Warning: {skipped} not in species.net, skipped")
    names = [n for n in names if n[2:] in species_keys]
    idx = [species_keys.index(n[2:]) for n in names]
    factor = np.array([SPECIES[sp]["factor"] for sp in _species_list(species)
                       for n in SPECIES[sp]["levels"] if n in names])

    densities = np.zeros(data[TE_NAME].shape + (len(species_keys),))
    densities[..., idx] = np.stack([data[n] for n in names], axis=-1) * factor + density_floor

    data.update({"densities": densities, "species_keys": species_keys, "grid": grid})

    if output:
        # save_to_openpmd reverses the axes, so pass (Nr, Nz) to get (z, r) in the file
        save_to_openpmd(grid,
                        np.ascontiguousarray(densities.transpose(1, 0, 2)),   # (Nr, Nz, Nspecies)
                        np.ascontiguousarray((data[TE_NAME] + te_offset).T),  # (Nr, Nz)
                        output, species_keys)
        print(f"Wrote CHEQUP input ({len(species_keys)} species, {len(data['z'])} z steps) -> {output}")

    if plot:
        plot_fields(data, list(_field_map(species)))
    return data


def plot_fields(data, names):
    """r-z map of each field: x axis = z [mm], y axis = r [µm]."""
    z, r = data["z"], data["r"]
    for name in names:
        plt.figure()
        plt.pcolormesh(z * 1e3, r * 1e6, data[name].T, shading="auto")
        plt.xlabel("z [mm]")
        plt.ylabel("r [µm]")
        plt.colorbar(label=name)
        plt.title(name)
    plt.show()
"""
CHEQUP to FBPIC Profile Converter Module.

This module provides the :class:`CHEQUPToFBPIC` class to process multi-species
hydrodynamics outputs from CHEQUP (Castro framework) and convert them into openPMD-compliant
HDF5 density files for initialization in FBPIC simulations.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import openpmd_api as io

# Ensure access to the analysis tool path
from analysis_tool import CastroSimulation


class CHEQUPToFBPIC:
    """Converts CHEQUP hydrodynamics output files to openPMD profiles for FBPIC.

    Handles extraction of species densities across various charge states, calculates
    the resulting electron profile to satisfy quasineutrality, and writes openPMD
    HDF5 files configured for r-z (thetaMode) geometry.

    Attributes:
        ATOMIC_MASS (float): Mass unit conversion factor in kg (1.66e-30 kg).
        SPECIES_CONFIG (dict): Dictionary mapping species symbols to their atomic
            number Z and maximum tracked charge state.
        sim_dir (str): Directory path containing CHEQUP simulation plotfiles.
        plt_pattern (str): Pattern string matching CHEQUP plotfile folders.
        cs (CastroSimulation): Instance of the CHEQUP simulation reader interface.
    """

    ATOMIC_MASS = 1.66e-30  # kg

    SPECIES_CONFIG = {
        'H':  {'z_num': 1,  'max_Z': 1},
        'He': {'z_num': 2,  'max_Z': 2},
        'N':  {'z_num': 7,  'max_Z': 7},
        'Ar': {'z_num': 18, 'max_Z': 8}  # Max ionization state tracked
    }

    def __init__(self, sim_dir='.', plt_pattern='plt_2d_*'):
        """Initializes the CHEQUPToFBPIC converter interface.

        Args:
            sim_dir (str, optional): Root directory path containing CHEQUP plotfiles.
                Defaults to '.'.
            plt_pattern (str, optional): Folder pattern for simulation output directories.
                Defaults to 'plt_2d_*'.
        """
        self.sim_dir = sim_dir
        self.plt_pattern = plt_pattern
        self.cs = CastroSimulation(self.sim_dir, self.plt_pattern)

    def force_quasineutrality(self, ion_charge_densities):
        """Calculates electron density required to enforce local quasineutrality.

        Evaluates the sum: :math:`n_e = \\sum_i Z_i n_i` for all active ion
        species components.

        Args:
            ion_charge_densities (dict): Dictionary mapping species names to 2D
                numpy arrays of free electron contributions (:math:`Z_i n_i`).

        Returns:
            numpy.ndarray: 2D array representing the total quasineutral electron density profile.
        """
        first_ion = next(iter(ion_charge_densities.values()))
        n_e = np.zeros_like(first_ion)
        
        for sp_name, n_sp_e in ion_charge_densities.items():
            n_e += n_sp_e

        return n_e

    def get_species_density(self, t_hydro, species_name):
        """Extracts and sums all ionization states for a specified element.

        Iterates through tracked ionization states (e.g., `rho_Ar0` to `rho_Ar8`),
        converts mass density fields into number density arrays (in :math:`\\text{m}^{-3}`),
        and accumulates total atomic and electron contributions.

        Args:
            t_hydro (float): Hydrodynamics simulation timestamp (in seconds).
            species_name (str): Element symbol (e.g., 'H', 'He', 'N', 'Ar').

        Returns:
            tuple[numpy.ndarray, numpy.ndarray]:
                - **total_density** (*numpy.ndarray*): Total atomic number density.
                - **total_electrons** (*numpy.ndarray*): Total electron contribution from this species.

        Raises:
            ValueError: If `species_name` is not present in :attr:`SPECIES_CONFIG`.
        """
        if species_name not in self.SPECIES_CONFIG:
            raise ValueError(f"Species '{species_name}' is not configured in SPECIES_CONFIG.")

        cfg = self.SPECIES_CONFIG[species_name]
        max_Z = cfg['max_Z']

        total_density = None
        total_electrons = None

        for Z in range(max_Z + 1):
            field_name = f'rho_{species_name}{Z}'
            try:
                field_data = self.cs.get_field(t_hydro, quantity=field_name, level=0)['q'] / self.ATOMIC_MASS
            except KeyError:
                # Skip if a specific charge state field is missing from simulation output
                continue

            if total_density is None:
                total_density = np.zeros_like(field_data)
                total_electrons = np.zeros_like(field_data)

            total_density += field_data
            total_electrons += field_data * Z

        return total_density, total_electrons

    def process_all_species(self, t_hydro, target_species=('H', 'He', 'N', 'Ar'), quasineutral=True):
        """Extracts species profiles and calculates final grid coordinates and electron profile.

        Args:
            t_hydro (float): Hydrodynamics simulation timestamp (in seconds).
            target_species (tuple[str, ...], optional): Sequence of target species symbols.
                Defaults to ('H', 'He', 'N', 'Ar').
            quasineutral (bool, optional): Whether to enforce quasineutrality for $n_e$.
                Defaults to True.

        Returns:
            tuple[numpy.ndarray, numpy.ndarray, dict, numpy.ndarray]:
                - **r** (*numpy.ndarray*): 1D array of radial spatial coordinates.
                - **z** (*numpy.ndarray*): 1D array of longitudinal spatial coordinates.
                - **species_densities** (*dict*): Dictionary mapping species name to 2D density array.
                - **density_e** (*numpy.ndarray*): 2D array of computed electron density.
        """
        species_densities = {}
        ion_electron_contributions = {}

        for sp in target_species:
            n_sp, n_e_sp = self.get_species_density(t_hydro, species_name=sp)
            
            if n_sp is not None:
                species_densities[sp] = n_sp
                ion_electron_contributions[sp] = n_e_sp

        if quasineutral:
            density_e = self.force_quasineutrality(ion_electron_contributions)
        else:
            density_e = sum(ion_electron_contributions.values())

        # Extract grid vectors directly from simulation object
        ref_field = self.cs.get_field(t_hydro, quantity='rho_H0', level=0)
        r = ref_field['r'] if 'r' in ref_field else np.linspace(0, 150e-6, density_e.shape[0])
        z = ref_field['z'] if 'z' in ref_field else np.linspace(0, 1e-2, density_e.shape[1])

        return r, z, species_densities, density_e

    def plot_profiles(self, r, z, species_densities, density_e):
        """Generates and displays 2D spatial density maps for each species and electrons.

        Args:
            r (numpy.ndarray): 1D array of radial coordinates (in meters).
            z (numpy.ndarray): 1D array of longitudinal coordinates (in meters).
            species_densities (dict): Dictionary mapping species name to 2D density array.
            density_e (numpy.ndarray): 2D array of computed electron density.
        """
        num_plots = len(species_densities) + 1
        fig, axes = plt.subplots(num_plots, 1, figsize=(10, 3 * num_plots), sharex=True, sharey=True)

        if num_plots == 1:
            axes = [axes]

        # Plot scale conversions (m -> mm for z, m -> um for r)
        z_mm = z * 1e3
        r_um = r * 1e6
        extent = [z_mm[0], z_mm[-1], r_um[0], r_um[-1]]

        # Plot individual species profiles
        for i, (sp_name, density_matrix) in enumerate(species_densities.items()):
            ax = axes[i]
            im = ax.imshow(
                density_matrix, 
                aspect='auto', 
                origin='lower', 
                extent=extent, 
                cmap='viridis'
            )
            ax.set_ylabel(r'$r$ [$\mu$m]')
            ax.set_title(f'Species Density: {sp_name}')
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label(r'$n$ [m$^{-3}$]')

        # Plot quasineutral electron profile
        ax_e = axes[-1]
        im_e = ax_e.imshow(
            density_e, 
            aspect='auto', 
            origin='lower', 
            extent=extent, 
            cmap='inferno'
        )
        ax_e.set_xlabel('$z$ [mm]')
        ax_e.set_ylabel(r'$r$ [$\mu$m]')
        ax_e.set_title('Electron Density (Quasineutral $n_e$)')
        cbar_e = fig.colorbar(im_e, ax=ax_e)
        cbar_e.set_label(r'$n_e$ [m$^{-3}$]')

        plt.tight_layout()
        plt.show()

    def _store_species_in_file_rz(self, z_1d, r_1d, series, density_data, species_name):
        """Writes a single species mesh component into the openPMD series.

        Configures openPMD metadata appropriate for cylindrical `thetaMode` geometry
        and writes scalar density records.

        Args:
            z_1d (numpy.ndarray): 1D array of longitudinal coordinates (z-axis).
            r_1d (numpy.ndarray): 1D array of radial coordinates (r-axis).
            series (openpmd_api.Series): Active openPMD file series instance.
            density_data (numpy.ndarray): 2D array shaped (Nz, Nr) representing species density.
            species_name (str): OpenPMD record name assigned to the scalar field.

        Raises:
            AssertionError: If `density_data.shape` does not match `(len(z_1d), len(r_1d))`.
        """
        assert density_data.shape == (len(z_1d), len(r_1d)), \
            f"Data shape {density_data.shape} does not match grids ({len(z_1d)}, {len(r_1d)})"

        it = series.iterations[0]
        density = it.meshes[species_name]

        # Set mesh grid properties and geometry metadata
        density.grid_spacing = np.array([
            (z_1d[-1] - z_1d[0]) / (len(z_1d) - 1),
            (r_1d[-1] - r_1d[0]) / (len(r_1d) - 1)
        ])
        density.grid_global_offset = [z_1d[0], r_1d[0]]
        density.axis_labels = ["z", "r"]
        density.geometry = io.Geometry.thetaMode
        density.unit_dimension = {
            io.Unit_Dimension.L: -3,
        }
        
        density_d = density[io.Mesh_Record_Component.SCALAR]
        density_d.position = [0, 0, 0]

        dataset = io.Dataset(density_data.dtype, density_data.shape)
        density_d.reset_dataset(dataset)
        density_d.store_chunk(density_data)

    def export_fbpic_profile(self, t_hydro, target_species=('H', 'He', 'N', 'Ar'), 
                            quasineutral=True, output_dir='.', filename="plasma_density.h5",
                            plot=False):
        """Exports the processed plasma density profiles to an openPMD HDF5 file.

        Args:
            t_hydro (float): Hydrodynamics simulation timestamp (in seconds).
            target_species (tuple[str, ...], optional): List of target species symbols.
                Defaults to ('H', 'He', 'N', 'Ar').
            quasineutral (bool, optional): Whether to compute electron profile via
                quasineutrality condition. Defaults to True.
            output_dir (str, optional): Target output folder directory. Defaults to '.'.
            filename (str, optional): Output HDF5 file name. Defaults to "plasma_density.h5".
            plot (bool, optional): If True, renders the interactive matplotlib figure window.
                Defaults to False.
        """
        r, z, species_densities, density_e = self.process_all_species(
            t_hydro, target_species=target_species, quasineutral=quasineutral
        )
        
        file_path = os.path.join(output_dir, filename)
        if os.path.exists(file_path):
            os.remove(file_path)
        
        series = io.Series(file_path, io.Access.create)
        series.set_software("PIP4_Profile_Generator", "1.0") 

        # Write atomic species profiles (Transposed to (Nz, Nr))
        for sp_name, density_matrix in species_densities.items():
            self._store_species_in_file_rz(z, r, series, density_matrix.T, f"density_{sp_name}")

        # Write quasineutral electron profile
        self._store_species_in_file_rz(z, r, series, density_e.T, "density_e")

        series.flush()
        del series
        print(f"Successfully exported multi-species profile to {file_path}")

        # Display plot window on screen without saving to disk
        if plot:
            self.plot_profiles(r, z, species_densities, density_e)
import os
import sys
import numpy as np
from scipy.ndimage import gaussian_filter
import openpmd_api as io

code_path = '/data/dust/user/benahmed/src/Codes/chequp'
sys.path.append(f"{code_path}/sim_folder/analysis")
from analysis_tool import CastroSimulation


def write_FBPIC_profile(input_path='.', output_path='.', t_hydro=0.0e-9):
    """
    Extract plasma species density profiles from a Castro hydrodynamics simulation
    and export them to an openPMD-compliant HDF5 file formatted for FBPIC in 2D
    cylindrical (:math:`r-z`) geometry.

    :param input_path: Path to the directory containing Castro simulation plotfiles 
                       (e.g., ``plt_2d_*``). Defaults to current working directory ('.').
    :type input_path: str, optional
    :param output_path: Directory where the output HDF5 file (``plasma_density.h5``) 
                        will be created. Defaults to current working directory ('.').
    :type output_path: str, optional
    :param t_hydro: Hydrodynamic simulation time (in seconds) at which to extract 
                    the density profiles. Defaults to 0.0e-9.
    :type t_hydro: float, optional

    :returns: None
    :rtype: None

    .. note::
       The generated file follows openPMD standards using the ``thetaMode`` geometry 
       (cylindrical :math:`r-z`) and records scalar density datasets for Hydrogen, 
       Helium, Argon, Nitrogen, and calculated free electrons.
    """

    def store_species_in_file_rz(z_1d, r_1d, series, density_data, species_name):
        """
        Store a single 2D species density grid into an openPMD series using 
        cylindrical :math:`r-z` coordinates.

        :param z_1d: 1D array representing the spatial grid points along the longitudinal Z axis (meters).
        :type z_1d: numpy.ndarray
        :param r_1d: 1D array representing the spatial grid points along the radial R axis (meters).
        :type r_1d: numpy.ndarray
        :param series: Active openPMD file series write instance.
        :type series: openpmd_api.Series
        :param density_data: 2D array of species density values matching shape ``(Nz, Nr)``.
        :type density_data: numpy.ndarray
        :param species_name: Name of the mesh record to write in the openPMD file 
                             (e.g., 'density_e', 'density_H').
        :type species_name: str

        :raises AssertionError: If ``density_data.shape`` does not match ``(len(z_1d), len(r_1d))``.
        """
        # Optional but highly recommended safety check:
        # Ensures the data array shape matches (Nz, Nr) exactly before saving.
        assert density_data.shape == (len(z_1d), len(r_1d)), \
            f"Data shape {density_data.shape} does not match grids ({len(z_1d)}, {len(r_1d)})"

        it = series.iterations[0]
        density = it.meshes[species_name]

        # Z is index 0, R is index 1
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

    def write_plasma_profile(output_dir, z, r, density_H, density_He, density_Ar, density_N, density_e):
        """
        Create the target HDF5 file and write all calculated species density profiles to it.

        :param output_dir: Directory path where ``plasma_density.h5`` will be saved.
        :type output_dir: str
        :param z: 1D spatial coordinate array for the Z-axis in meters.
        :type z: numpy.ndarray
        :param r: 1D spatial coordinate array for the R-axis in meters.
        :type r: numpy.ndarray
        :param density_H: Transposed 2D array of Hydrogen density with shape ``(Nz, Nr)``.
        :type density_H: numpy.ndarray
        :param density_He: Transposed 2D array of Helium density with shape ``(Nz, Nr)``.
        :type density_He: numpy.ndarray
        :param density_Ar: Transposed 2D array of Argon density with shape ``(Nz, Nr)``.
        :type density_Ar: numpy.ndarray
        :param density_N: Transposed 2D array of Nitrogen density with shape ``(Nz, Nr)``.
        :type density_N: numpy.ndarray
        :param density_e: Transposed 2D array of calculated Electron density with shape ``(Nz, Nr)``.
        :type density_e: numpy.ndarray
        """
        file_path = os.path.join(output_dir, "plasma_density.h5")
        if os.path.exists(file_path):
            os.remove(file_path)
        
        # Create the openPMD series
        series = io.Series(file_path, io.Access.create)
        series.set_software("PIP4_Profile_Generator", "1.0") 

        # Assuming density_H was originally created as (Nr, Nz). 
        # Transposing it with .T makes it (Nz, Nr), which perfectly aligns with our (z, r) inputs.
        store_species_in_file_rz(z, r, series, density_H, "density_H")
        store_species_in_file_rz(z, r, series, density_He, "density_He")
        store_species_in_file_rz(z, r, series, density_Ar, "density_Ar")
        store_species_in_file_rz(z, r, series, density_N, "density_N")
        store_species_in_file_rz(z, r, series, density_e, "density_e")

        # Flush and safely close
        series.flush()
        del series
        
    def get_domain_extents(header_path):
        """
        Parse the Castro plotfile Header file to extract spatial domain maximum limits.

        :param header_path: Full file path to the plotfile ``Header`` file.
        :type header_path: str
        :returns: Upper physical spatial extent in radial (R) and longitudinal (Z) dimensions.
        :rtype: tuple(float, float)
        """
        with open(header_path, "r") as f:
            lines = [line.strip() for line in f if line.strip()]
        num_vars = int(lines[1])
        prob_hi_idx = 2 + num_vars + 4
        r_max, z_max = map(float, lines[prob_hi_idx].split())
        return r_max, z_max

    print('Loading CHEQUP...')
    cs = CastroSimulation(input_path, 'plt_2d_*')
    atomic_mass = 1.66e-30

    # Hydrogen
    density_H = cs.get_field(t_hydro, quantity='rho_H1', level=0)['q'] / atomic_mass
    # Helium
    density_He = cs.get_field(t_hydro, quantity='rho_He1', level=0)['q'] / atomic_mass
    density_He += cs.get_field(t_hydro, quantity='rho_He2', level=0)['q'] / atomic_mass
    # Argon
    density_Ar = cs.get_field(t_hydro, quantity='rho_Ar1', level=0)['q'] / (39.9 * atomic_mass)
    for Z in range(2, 9):
        density_Ar += cs.get_field(t_hydro, quantity=f'rho_Ar{Z}', level=0)['q'] / (39.9 * atomic_mass)
    # Nitrogen
    density_N = cs.get_field(t_hydro, quantity='rho_N1', level=0)['q'] / (14.0 * atomic_mass)
    for Z in range(2, 6):
        density_N += cs.get_field(t_hydro, quantity=f'rho_N{Z}', level=0)['q'] / (14.0 * atomic_mass)

    density_e = density_H + density_Ar * 8.0 + 5.0 * density_N + 2 * density_He
    # Geometry
    r_max, z_max = get_domain_extents(f"{input_path}/plt_2d_00000/Header")
    r = np.linspace(0, r_max*1e-2, density_e.shape[0])
    z = np.linspace(0, z_max*1e-2, density_e.shape[0])
    
    write_plasma_profile(
            output_dir=output_path,
            z=z,
            r=r,
            density_H=density_H.T,
            density_He=density_He.T,
            density_Ar=density_Ar.T,
            density_N=density_N.T,
            density_e=density_e.T
        )
    print(f'Wrote {output_path}/plasma_density.h5')
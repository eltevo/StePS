#!/usr/bin/env python3


import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..','StePS_IC','src'))
from os.path import exists
import argparse
import numpy as np
import time
from scipy.special import erf
from nbodykit.lab import HDFCatalog, ConvolvedFFTPower, ArrayCatalog, FKPCatalog
from inputoutput import Load_params_from_HDF5_snap


_VERSION="v0.0.1.0"
_YEAR="2025-2026"
_AUTHOR="Gabor Racz"
_DESCRIPTION="StePS Power Spectrum Estimator based on the FKP method for spherical and cylindrical zoom-in simulations."

#Setting up the units of distance and time
UNIT_T=47.14829951063323 #Unit time in Gy
UNIT_V=20.738652969925447 #Unit velocity in km/s
UNIT_D=3.0856775814671917e24#=1Mpc Unit distance in cm

def get_mean_displacement(data_coords, glass_coords, mean_particle_separation):
    """
    Calculate the mean displacement between data and glass coordinates.
    Parameters:
     - data_coords : np.ndarray
        The 3D coordinates of the data particles.
     - glass_coords : np.ndarray
        The 3D coordinates of the glass particles.
     - mean_particle_separation : float
        The mean particle separation in the simulation, used for normalization.
    Returns:
     - mean_disp : float
        The mean displacement between data and glass coordinates, in the units of mean particle separation.
    """
    if len(data_coords) != len(glass_coords):
        raise ValueError("Data and glass coordinates must have the same length.")
    
    # Calculate the displacement vector lengths
    displacement = np.sqrt(np.sum((data_coords - glass_coords)**2, axis=1))  # Euclidean distance
    
    # Calculate the mean displacement
    mean_disp = np.double(np.mean(displacement)) / mean_particle_separation
    return mean_disp

def estimate_power_spectrum_fkp(data, randoms, Lbox, P0=1e4, n_radial_bins=32, Nmesh=64, Geometry='spherical', Verbose=False, Lzsim=0.0):
    """
    Estimates the power spectrum of a non-periodic zoom-in simulation
    using the FKP estimator with position-dependent weights.

    Parameters
    ----------
    data : ArrayCatalog
        The data catalog containing the simulation snapshot.
    randoms : ArrayCatalog
        The random catalog containing random points for estimating the power spectrum.
    Lbox : float
        The linear size of the periodic box of the 3D Fourier transformation.
    P0 : float, optional
        The characteristic power spectrum amplitude for FKP weights.
    n_radial_bins : int, optional
        The number of radial bins to use for estimating n(r).
    Nmesh : int, optional
        The number of mesh points for the FFT.
    Geometry : str, optional
        The geometry of the simulation, either 'spherical' or 'cylindrical'.
    Verbose : bool, optional
        If True, print detailed information about the estimation process.
    Lzsim : float, optional
        The length of the simulation box in "z" direction for cylindrical geometry. If 0.0, it defaults to Lbox.
    Returns
    -------
    pk : nbodykit.binned_statistic.BinnedStatistic
        The estimated power spectrum.
    """
    # --- Checking the input parameters ---
    if Geometry not in ['spherical', 'cylindrical']:
        raise ValueError("Geometry must be either 'spherical' or 'cylindrical'.")

    # Ensure that the 'Radius' and 'Position' column is present in both data and randoms
    if 'Radius' not in data.columns:
        raise ValueError("Data catalog must contain a 'Radius' column.")
    if 'Radius' not in randoms.columns:
        raise ValueError("Randoms catalog must contain a 'Radius' column.")
    if 'Position' not in data.columns:
        raise ValueError("Data catalog must contain a 'Position' column.")
    if 'Position' not in randoms.columns:
        raise ValueError("Randoms catalog must contain a 'Position' column.")

    # --- Cutting the data and randoms to the specified radius ---

    # Determine the radial bins
    rmin = 0.0
    rmax = np.max([data['Radius'].max(), randoms['Radius'].max()]) 
    radial_bins = np.linspace(rmin, rmax, n_radial_bins + 1)
    bin_centers = 0.5 * (radial_bins[:-1] + radial_bins[1:])

    if Verbose:
        start_time = time.time()
        print(f"\tEstimating power spectrum using FKP estimator...")
        print(f"\tGeometry: {Geometry}")
        print(f"\tBox size: {Lbox:.2f} Mpc")
        print(f"\tData catalog contains {len(data)} particles.")
        print(f"\tRandoms catalog contains {len(randoms)} particles.")
        print(f"\tUsing {n_radial_bins} radial bins from {rmin:.2f} to {rmax:.2f} Mpc.")

    # Count randoms in each radial shell
    counts, _ = np.histogram(randoms['Radius'], bins=radial_bins)

    # Calculate the volume of each spherical shell
    if Geometry == 'spherical':
        shell_volumes = 4./3. * np.pi * (radial_bins[1:]**3 - radial_bins[:-1]**3)
    elif Geometry == 'cylindrical':
        if Lzsim <= 0.0:
            Lzsim = Lbox
        # For cylindrical geometry, we assume a height equal to the simulation cylinder length
        shell_volumes = np.pi * (radial_bins[1:]**2 - radial_bins[:-1]**2) * Lzsim

    # Estimate the number density in each shell
    n_r = counts / shell_volumes

    # --- Assign n(R) and FKP weights to both data and randoms ---

    # Interpolate n(r) to the position of each data and random particle
    data['NZ'] = np.interp(np.array(data['Radius']), bin_centers, np.array(n_r), left=n_r[0], right=n_r[-1])
    randoms['NZ'] = np.interp(np.array(randoms['Radius']), bin_centers, np.array(n_r), left=n_r[0], right=n_r[-1])

    # Calculate the FKP weights
    data['FKPWeight'] = 1.0 / (1.0 + data['NZ'] * P0)
    randoms['FKPWeight'] = 1.0 / (1.0 + randoms['NZ'] * P0)
    
    # The 'Weight' column is used for completeness, which is 1.0 here
    # as the randoms already encode the selection.
    data['Weight'] = 1.0
    randoms['Weight'] = 1.0

    # --- Compute the power spectrum using ConvolvedFFTPower ---

    # Combine data and randoms into an FKPCatalog (https://nbodykit.readthedocs.io/en/binder/api/_autosummary/nbodykit.source.catalog.fkp.html)
    fkp = FKPCatalog(data, randoms,BoxSize=Lbox)

    # Convert to a mesh, specifying the weight columns
    # The box size is automatically determined from the randoms
    mesh = fkp.to_mesh(Nmesh=Nmesh, nbar='NZ', fkp_weight='FKPWeight', comp_weight='Weight')

    # Compute the power spectrum multipoles (here, just the monopole l=0)
    r = ConvolvedFFTPower(mesh, poles=[0], dk=0.01, kmin=0.01)

    if Verbose:
        elapsed_time = time.time() - start_time
        print(f"\tPower spectrum estimation completed in {elapsed_time:.2f} seconds.")

    return r

if __name__ == '__main__':
    start = time.time()
    # --- Welcome message and version information ---
    print(f"StePS Power Spectrum Estimator {_VERSION} by {_AUTHOR}, {_YEAR}")
    print(f"\n\tThis program is free software; you can redistribute it and/or modify\n\tit under the terms of the GNU General Public License as published by\n\tthe Free Software Foundation; either version 2 of the License,\n\tor (at your option) any later version.\n\n\tThis program is distributed in the hope that it will be useful,\n\tbut WITHOUT ANY WARRANTY; without even the implied warranty of\n\tMERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the\n\tGNU General Public License for more details.\n\n")

    # --- Parse command line arguments ---
    parser = argparse.ArgumentParser(description=_DESCRIPTION)
    parser.add_argument('data_path', type=str, help='Path to the HDF5 file containing the simulation snapshot.')
    parser.add_argument('random_path', type=str, help='Path to the HDF5 file containing the random catalog.')
    parser.add_argument('glass_path', type=str, help='Path to the HDF5 file containing the initial glass of the simulation.')
    parser.add_argument('output_path', type=str, help='Path to save the output power spectrum results in ASCII format.')
    parser.add_argument('--P0', type=float, default=1e4, help='Characteristic power spectrum amplitude for FKP weights (default: 1e4).')
    parser.add_argument('--n_radial_bins', type=int, default=2, help='Number of concentric spheres used for estimating the power spectrum (default: 2).')
    parser.add_argument('--n_FKP_radial_bins', type=int, default=32, help='Number of radial bins for estimating n(r) within a single FKP estimator (default: 32).')
    parser.add_argument('--Nmesh', type=int, default=256, help='Number of mesh points for the FFT (default: 256).')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output.')
    parser.add_argument('--ShotNoise', action='store_true', help='Calculate and subtract shot noise from the power spectrum.')
    parser.add_argument('--Geometry', type=str, default='spherical', choices=['spherical', 'cylindrical'], help='Geometry of the simulation (default: spherical).')
    args = parser.parse_args()

    # --- Check if the input files exist ---
    if not exists(args.data_path):
        print(f"Error: Data file '{args.data_path}' does not exist.")
        sys.exit(1)
    if not exists(args.random_path):
        print(f"Error: Randoms file '{args.random_path}' does not exist.")
        sys.exit(1)
    if not exists(args.glass_path):
        print(f"Error: Glass file '{args.glass_path}' does not exist.")
        sys.exit(1)
    
    # --- Load the data and randoms from HDF5 files ---
    data_hdf5 = HDFCatalog(args.data_path, dataset='PartType1')
    randoms_hdf5 = HDFCatalog(args.random_path, dataset='PartType1')
    glass_hdf5 = HDFCatalog(args.glass_path, dataset='PartType1')

    # --- Load the cosmology parameters from the data catalog ---
    Parameters = Load_params_from_HDF5_snap(args.data_path, LEGACY_MODE=False)
    # Calculate the (z=0) critical and matter density
    rho_crit = 3*(Parameters['h']*100)**2/(8*np.pi)/UNIT_V/UNIT_V #in internal units [10^11 M_sun/Mpc^3]
    rho_mean = Parameters['Om']*rho_crit

    # --- Preparing the data and randoms ---
    if args.verbose:
        print(f"-------------------------------------\nLoaded data from {args.data_path} with {len(data_hdf5)} particles.")
        print(f"Loaded randoms from {args.random_path} with {len(randoms_hdf5)} particles.")
        print(f"Loaded glass from {args.glass_path} with {len(glass_hdf5)} particles.\n--------------------------------------\n")
    # Calculate the radial distance for each data and random particle
    if args.Geometry == 'spherical':
        randoms_hdf5['Radius'] = np.sqrt(randoms_hdf5['Coordinates'][:, 0]**2 + randoms_hdf5['Coordinates'][:, 1]**2 + randoms_hdf5['Coordinates'][:, 2]**2)
        data_hdf5['Radius'] = np.sqrt(data_hdf5['Coordinates'][:, 0]**2 + data_hdf5['Coordinates'][:, 1]**2 + data_hdf5['Coordinates'][:, 2]**2)
        glass_hdf5['Radius'] = np.sqrt(glass_hdf5['Coordinates'][:, 0]**2 + glass_hdf5['Coordinates'][:, 1]**2 + glass_hdf5['Coordinates'][:, 2]**2)
    elif args.Geometry == 'cylindrical':
        randoms_hdf5['Radius'] = np.sqrt(randoms_hdf5['Coordinates'][:, 0]**2 + randoms_hdf5['Coordinates'][:, 1]**2)
        data_hdf5['Radius'] = np.sqrt(data_hdf5['Coordinates'][:, 0]**2 + data_hdf5['Coordinates'][:, 1]**2)
        glass_hdf5['Radius'] = np.sqrt(glass_hdf5['Coordinates'][:, 0]**2 + glass_hdf5['Coordinates'][:, 1]**2)
    Rmax = np.double(data_hdf5['Radius'].max())  # Maximum radius in the data
    # getting the minimal and maximal particle masses, and the spatial resolution
    min_mass = np.double(data_hdf5['Masses'].min())
    Npart_min = data_hdf5['Masses'][data_hdf5['Masses'] <= 1.1*min_mass].size  # Number of particles with close to minimal mass
    max_mass = np.double(data_hdf5['Masses'].max())
    tot_mass = np.double(data_hdf5['Masses'].sum())  # Total mass of the particles in the data catalog
    if args.Geometry == 'spherical':
        Rsim = np.cbrt(3 * tot_mass / (4 * np.pi * rho_mean)) # The radius of the simulation sphere in Mpc, calculated from the total mass and the mean density
        k_min = np.pi/Rsim # The minimal resolved wavenumber in the simulation volume, in 1/Mpc
    elif args.Geometry == 'cylindrical':
        Lsim = np.max((np.max(np.array(data_hdf5['Coordinates'][:, 2])), np.max(np.array(randoms_hdf5['Coordinates'][:, 2])))) - np.min((np.min(np.array(data_hdf5['Coordinates'][:, 2])), np.min(np.array(randoms_hdf5['Coordinates'][:, 2]))))  # The length of the simulation cylinder in Mpc
        Rsim = np.sqrt(tot_mass / (np.pi * rho_mean * Lsim)) # The radius of the simulation cylinder in Mpc, calculated from the total mass and the mean density)
        k_min = 2*np.pi/np.max((2*Rsim,Lsim)) # The minimal resolved wavenumber in the simulation volume, in 1/Mpc
    dmean_Rsim = np.cbrt(max_mass/rho_mean) # The spatial resolution at Rsim, in Mpc
    dmean_Rcentral = np.cbrt(min_mass/rho_mean) # The spatial resolution at Rcentral, in Mpc
    k_max_Rsim = 2*np.pi/dmean_Rsim # The maximal resolved wavenumber my the maximal mass particles in 1/Mpc
    
    # the maximum radius of the minimal mass particle defines the radius of the central, constant resolution region
    Rcentral = np.double(data_hdf5['Radius'][data_hdf5['Masses'] <= 1.1*min_mass].max())  # Maximum radius of the central region


    # --- Print some information about the data ---
    print("\nCosmological Parameters:\n------------------------")
    print(f"Redshift = \t\t{Parameters['Redshift']}\nOmega_m =\t\t{Parameters['Om']:.4f}\nOmega_lambda =\t\t{Parameters['Ol']:.4f}\nOmega_radiation =\t{Parameters['Or']:.4f}\nH0 =\t\t\t{100.0*Parameters['h']:.2f} km/s/Mpc")
    if Parameters["Model"] == "wCDM" or Parameters["Model"] == "w0waCDM":
        print(f"w0 = {Parameters['w0']:.2f}")
    if Parameters["Model"] == "w0waCDM":
        print(f"wa = {Parameters['wa']:.2f}")
    if args.verbose:
        print(f"Critical density =\t{1e11*rho_crit:.4e} M_sun/Mpc^3\nMean density =\t\t{1e11*rho_mean:.4e} M_sun/Mpc^3\n")
        
    print(f"\nSimulation geometry:\n--------------------\nRsim =\t\t\t{Rsim:.2f} Mpc\nRmax =\t\t\t{Rmax:.2f} Mpc\nRcentral =\t\t{Rcentral:.2f} Mpc")
    if args.Geometry == 'cylindrical':
        print(f"Lsim =\t\t\t{Lsim:.2f} Mpc")
    print(f"Min particle mass =\t{1e11*min_mass:.4e} M_sun\nMax particle mass =\t{1e11*max_mass:.4e} M_sun\n")
    print(f"Mean particle separation at Rsim = \t{dmean_Rsim:.2f} Mpc \t(k_min={k_min:.5f} Mpc^-1; k_max = {k_max_Rsim:.4f} Mpc^-1; k_max/k_min={k_max_Rsim/k_min:.1f})\nMean particle separation at Rcentral = \t{dmean_Rcentral:.2f} Mpc \t(k_min = {2*np.pi/Rcentral:.4f} Mpc^-1; k_max = {2*np.pi/dmean_Rcentral:.4f} Mpc^-1)\n")
    print(f"Number of particles in the data catalog: {len(data_hdf5)}")
    print(f"Number of minimal mass particles in the data catalog: {Npart_min}")

    # --- Estimate the power spectrum ---
    # we use different "rmax" radius for the FKP estimator to estimate the power spectrum at different scales
    rmax_array = np.linspace(Rsim, Rcentral, args.n_radial_bins)
    print("Using the following radial bins to calculate the Power-spectrum: ",rmax_array, "Mpc")
    k_final = np.array([],dtype=np.double)  # Initialize an empty array for k values
    P_final = np.array([],dtype=np.double) # Initialize an empty array for P(k) values
    for i in range(len(rmax_array)):
        # Set the maximum radius for the analysis
        rmax = rmax_array[i]
        # Filter the data and randoms to only include particles within the specified radius
        data_mask = data_hdf5['Radius'] <= rmax
        randoms_mask = randoms_hdf5['Radius'] <= rmax
        glass_mask = glass_hdf5['Radius'] <= rmax
        data = ArrayCatalog({'Radius': data_hdf5['Radius'][data_mask],
                            'Position': data_hdf5['Coordinates'][data_mask],
                            'Masses': data_hdf5['Masses'][data_mask]})
        randoms = ArrayCatalog({'Radius': randoms_hdf5['Radius'][randoms_mask],
                                'Position': randoms_hdf5['Coordinates'][randoms_mask]})
        glass = ArrayCatalog({'Radius': glass_hdf5['Radius'][glass_mask],
                              'Position': glass_hdf5['Coordinates'][glass_mask],
                              'Masses': glass_hdf5['Masses'][glass_mask]})
        mass_res_bin = np.double(data['Masses'].max())
        d_mean_bin = np.cbrt(3 * mass_res_bin / (4 * np.pi * rho_mean))  # The spatial resolution at the current rmax, in Mpc
        # Calculate the mean displacement between data and glass coordinates, but only for the particles with maximal mass
        displacement_mask = np.logical_and(data_hdf5['Masses'] <= mass_res_bin, data_hdf5['Masses'] > 0.95*mass_res_bin)  # Use a mask to only consider particles with mass smaller than the current bin's mass resolution
        mean_displacement = get_mean_displacement(data_hdf5['Coordinates'][displacement_mask], glass_hdf5['Coordinates'][displacement_mask], d_mean_bin)
        if i == 0:
            k_min_bin = np.pi/rmax   # The minimum wavenumber for the current bin, in 1/Mpc
            k_max_bin = 0.75*np.pi/d_mean_bin  # The maximum wavenumber for the current bin, in 1/Mpc, limited to 75% of the Nyquist frequency to be conservative
        else:
            k_min_bin = k_max_bin_prev # The minimum wavenumber for the current bin is the maximum wavenumber of the previous bin, in 1/Mpc
            k_max_bin = 2.0*np.pi/d_mean_bin  # The maximum wavenumber for the current bin, in 1/Mpc
        if args.verbose:
            print(f"\nBin {i}: Estimating power spectrum with FKP estimator using rmax = {rmax:.2f} Mpc. Mass resolution = {1e11*mass_res_bin:.4e} Msol, k_min = {k_min_bin:.4f} Mpc^-1, k_max = {k_max_bin:.4f} Mpc^-1, dmean = {d_mean_bin:.2f} Mpc")
            print(f"Mean displacement for rmax = {rmax:.2f} Mpc: {mean_displacement:.4f} times the mean particle separation ({d_mean_bin:.2f} Mpc)")
        # The number of FKP radial bins is defined for the smallest rmax, so we need to scale it for larger rmax values
        FKP_radial_bins = int(args.n_FKP_radial_bins * (rmax/Rcentral))
        if args.Geometry == 'spherical':
            pk_result_rand = estimate_power_spectrum_fkp(data, randoms, 4.0*rmax, P0=args.P0, n_radial_bins=FKP_radial_bins, Nmesh=args.Nmesh, Verbose=args.verbose)
            pk_result_glass = estimate_power_spectrum_fkp(data, glass, 4.0*rmax, P0=args.P0, n_radial_bins=FKP_radial_bins, Nmesh=args.Nmesh, Verbose=args.verbose)
        elif args.Geometry == 'cylindrical':
            Lestimatorbox = np.max((2*rmax, Lsim))  # The box size for the cylindrical estimator is the maximum of 2*Rsim and Lsim
            pk_result_rand = estimate_power_spectrum_fkp(data, randoms, Lestimatorbox, P0=args.P0, n_radial_bins=FKP_radial_bins, Nmesh=args.Nmesh, Verbose=args.verbose, Geometry=args.Geometry, Lzsim=Lsim)
            pk_result_glass = estimate_power_spectrum_fkp(data, glass, Lestimatorbox, P0=args.P0, n_radial_bins=FKP_radial_bins, Nmesh=args.Nmesh, Verbose=args.verbose, Geometry=args.Geometry, Lzsim=Lsim)
        Poisson_ratio = erf(0.25*mean_displacement) # if this is close to 1, the particles can be considered as Poisson distributed, if close to 0, they are close to glass
        #Poisson_ratio = 1.0
        if args.verbose:
                print(f"The Poisson ratio in this bin: {Poisson_ratio:.4f}")
        if args.ShotNoise:
            # The power spectrum monopole is in the 'power' column, after subtracting shot noise
            Pk = Poisson_ratio*(pk_result_rand.poles['power_0'].real - pk_result_rand.attrs['shotnoise']/2) + (1-Poisson_ratio)*pk_result_glass.poles['power_0'].real
        else:
            # The power spectrum monopole is in the 'power' column, without subtracting shot noise
            Pk = Poisson_ratio*pk_result_rand.poles['power_0'].real + (1-Poisson_ratio)*pk_result_glass.poles['power_0'].real
        k_max_bin = np.min([pk_result_rand.poles['k'][~np.isnan(pk_result_rand.poles['k'])].max(),k_max_bin])  # The maximum wavenumber for the current bin, in 1/Mpc
        k_max_bin_prev = k_max_bin  # Update the previous kmax for the next iteration
        # Store the k values and P(k) for this bin to the final lists within k_min_bin and k_max_bin
        k_mask = np.logical_and(pk_result_rand.poles['k']>=k_min_bin, pk_result_rand.poles['k']<k_max_bin)
        k_final = np.append(k_final,np.array(pk_result_rand.poles['k'][k_mask]))
        P_final = np.append(P_final,np.array(Pk[k_mask]))
        if args.verbose:
            print(pk_result_rand.poles)
    # --- Save the results ---
    
    # --- Save the results to an ASCII file ---
    with open(args.output_path, 'w') as f:
        f.write("# Real-space DM power spectrum, calculated with StePS_Pk.py " + _VERSION + "\n")
        f.write(f"# Cosmological parameters: z={Parameters['Redshift']}, Omega_m={Parameters['Om']:.4f}, Omega_lambda={Parameters['Ol']:.4f}, h={Parameters['h']:.4f}")
        if Parameters["Model"] == "wCDM" or Parameters["Model"] == "w0waCDM":
            f.write(f"# w0={Parameters['w0']:.2f}")
        if Parameters["Model"] == "w0waCDM":
            f.write(f"# wa={Parameters['wa']:.2f}")
        f.write("\n")
        f.write(f"# Simulation parameters: ")
        if args.Geometry == 'cylindrical':
            f.write(f"Geometry: Cylindrical, Rsim={Rsim:.2f} Mpc, Rcentral={Rcentral:.2f} Mpc,  Lsim={Lsim:.2f} Mpc, ")
        else:
            f.write(f"Geometry: Spherical, Rsim={Rsim:.2f} Mpc, Rcentral={Rcentral:.2f} Mpc, ")
        f.write(f"Min particle mass={1e11*min_mass:.4e} M_sun, Max particle mass={1e11*max_mass:.4e} M_sun\n")
        f.write(f"# Power spectrum estimator parameters: P0={args.P0}, Number of radial bins={args.n_radial_bins}, Number of n(r) bins={args.n_FKP_radial_bins}, Nmesh={args.Nmesh}, ShotNoise subtracted={args.ShotNoise}\n")
        f.write("# k [1/Mpc]  P(k) [(1/Mpc)^3]\n")
        for k, p in zip(k_final, P_final):
            f.write(f"{k:.7f} {p:.7e}\n")
    end = time.time()
    print(f"\nPower spectrum results saved to '{args.output_path}'.")
    print(f"Total execution time: {end - start:.2f} seconds.\n")
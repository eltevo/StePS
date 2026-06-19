#!/usr/bin/env python3


import argparse
import h5py
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.stats import binned_statistic_dd
import sys
import os
# adding ../Utils/ to the system path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Utils'))
from inputoutput import save_cosmic_web_hdf5


_VERSION="v0.0.1.0"
_YEAR="2026"
_AUTHOR="Gabor Racz"
_DESCRIPTION="StePS Cosmic Web Finder using the tidal tensor method based on particle accelerations."

'''
This script classifies the cosmic web structure in StePS simulations using the tidal tensor method, leveraging particle accelerations directly from the snapshot data.
Based on Hahn et al. (2007) [https://arxiv.org/abs/astro-ph/0610280]
The tidal tensor Tij is defined as the Hessian of the gravitational potential \phi:
Tij=d2\phi/dxidxj

Since the acceleration vector is a=-\nabla \phi, we can rewrite the components of the tidal tensor directly as the spatial derivatives of the acceleration field:
Tij=-dai/dxj
By binning the particle accelerations onto a grid, applying a Gaussian smoothing filter to suppress shot noise (essential for defining cosmic web scales as outlined in Hahn et al. 2007), and using finite differences, one can calculate Tij effortlessly without any FFT (which requires T^3 topology).
'''

def classify_cosmic_web(snapshot_path, geometry, grid_res, L=None, Lz=None, r_smooth=2.0, output_path="cosmic_web.h5"):
    print(f"--- Loading snapshot data from {snapshot_path} ---")
    with h5py.File(snapshot_path, 'r') as f:
        header = f['Header'].attrs
        
        if L is None:
            if 'BoxSize' in header:
                val = header['BoxSize']
                L = float(val[0]) if isinstance(val, np.ndarray) else float(val)
                print(f"Read L = {L:.4f} from HDF5 Header")
            else:
                raise ValueError("L was not provided as an argument and 'BoxSize' was not found in the header.")

        if geometry == 'S1R2' and Lz is None:
            if 'Lz' in header:
                val = header['Lz']
                Lz = float(val[0]) if isinstance(val, np.ndarray) else float(val)
                print(f"Read Lz = {Lz:.4f} from HDF5 Header")
            elif 'BoxSize' in header and isinstance(header['BoxSize'], np.ndarray) and len(header['BoxSize']) >= 3:
                Lz = float(header['BoxSize'][2])
                print(f"Read Lz = {Lz:.4f} from Header['BoxSize'][2]")
            else:
                raise ValueError("Lz was not provided as an argument and not found in the header for S1xR2 geometry.")

        pos = f['PartType1/Coordinates'][:]  # Shape (N, 3) [Mpc]
        acc = f['PartType1/Accelerations'][:]   # Shape (N, 3) [Internal Units]

    # 1. Define grid bounds and padding rules based on geometry
    if geometry == 'R3':
        Nx = Ny = Nz = grid_res
        x_min, x_max = -L / 2.0, L / 2.0
        y_min, y_max = -L / 2.0, L / 2.0
        z_min, z_max = -L / 2.0, L / 2.0
        smoothing_modes = 'nearest'
        pad_modes = ['edge', 'edge', 'edge']
        
    elif geometry == 'S1R2':
        Nx = Ny = grid_res
        Nz = int(grid_res * Lz / L) 
        x_min, x_max = -L / 2.0, L / 2.0
        y_min, y_max = -L / 2.0, L / 2.0
        z_min, z_max = 0.0, Lz 
        smoothing_modes = ['nearest', 'nearest', 'wrap']
        pad_modes = ['edge', 'edge', 'wrap']
        
    elif geometry == 'T3':
        Nx = Ny = Nz = grid_res
        x_min, x_max = 0.0, L
        y_min, y_max = 0.0, L
        z_min, z_max = 0.0, L
        smoothing_modes = 'wrap'
        pad_modes = ['wrap', 'wrap', 'wrap']
        
    else:
        raise ValueError("Geometry must be 'R3', 'S1R2', or 'T3'")

    edges = [
        np.linspace(x_min, x_max, Nx + 1),
        np.linspace(y_min, y_max, Ny + 1),
        np.linspace(z_min, z_max, Nz + 1)
    ]
    dx = (x_max - x_min) / Nx
    dy = (y_max - y_min) / Ny
    dz = (z_max - z_min) / Nz

    print(f"--- Binning acceleration to a {Nx}x{Ny}x{Nz} grid ---")
    grid_ax, _, _ = binned_statistic_dd(pos, acc[:, 0], statistic='mean', bins=edges)
    grid_ay, _, _ = binned_statistic_dd(pos, acc[:, 1], statistic='mean', bins=edges)
    grid_az, _, _ = binned_statistic_dd(pos, acc[:, 2], statistic='mean', bins=edges)

    grid_ax = np.nan_to_num(grid_ax, nan=0.0)
    grid_ay = np.nan_to_num(grid_ay, nan=0.0)
    grid_az = np.nan_to_num(grid_az, nan=0.0)

    # 2. Smooth the acceleration fields
    sigma_grid = r_smooth / dx
    print(f"--- Smoothing acceleration field (Sigma = {r_smooth} Mpc / {sigma_grid:.2f} cells) ---")
    
    grid_ax = gaussian_filter(grid_ax, sigma=sigma_grid, mode=smoothing_modes)
    grid_ay = gaussian_filter(grid_ay, sigma=sigma_grid, mode=smoothing_modes)
    grid_az = gaussian_filter(grid_az, sigma=sigma_grid, mode=smoothing_modes)

    # 3. Compute gradients with custom boundary padding
    print("--- Computing Tidal Tensor (T_ij) via finite differences ---")
    
    def compute_gradient(field, h_spacing):
        padded = np.pad(field, ((1,1), (0,0), (0,0)), mode=pad_modes[0])
        padded = np.pad(padded, ((0,0), (1,1), (0,0)), mode=pad_modes[1])
        padded = np.pad(padded, ((0,0), (0,0), (1,1)), mode=pad_modes[2])
        
        grad_x = np.gradient(padded, h_spacing[0], axis=0)[1:-1, 1:-1, 1:-1]
        grad_y = np.gradient(padded, h_spacing[1], axis=1)[1:-1, 1:-1, 1:-1]
        grad_z = np.gradient(padded, h_spacing[2], axis=2)[1:-1, 1:-1, 1:-1]
        return grad_x, grad_y, grad_z

    h = [dx, dy, dz]
    dT_x = compute_gradient(grid_ax, h)
    dT_y = compute_gradient(grid_ay, h)
    dT_z = compute_gradient(grid_az, h)

    T = np.zeros((Nx, Ny, Nz, 3, 3))
    T[..., 0, 0], T[..., 0, 1], T[..., 0, 2] = -dT_x[0], -dT_x[1], -dT_x[2]
    T[..., 1, 0], T[..., 1, 1], T[..., 1, 2] = -dT_y[0], -dT_y[1], -dT_y[2]
    T[..., 2, 0], T[..., 2, 1], T[..., 2, 2] = -dT_z[0], -dT_z[1], -dT_z[2]

    # Enforce strict symmetry T_ij = 0.5 * (T_ij + T_ji)
    T = 0.5 * (T + np.swapaxes(T, -1, -2))

    print("--- Performing Eigendecomposition ---")
    # w: eigenvalues, v: eigenvectors
    w, v = np.linalg.eigh(T)

    num_positive = np.sum(w > 0, axis=-1)
    structure_type = num_positive + 1  # 1=void, 2=sheet, 3=filament, 4=cluster

    # Prepare metadata for the HDF5 file
    metadata = {
        'geometry': geometry,
        'grid_res_Nx': Nx,
        'grid_res_Ny': Ny,
        'grid_res_Nz': Nz,
        'L': L,
        'Lz': Lz if Lz else L,
        'r_smooth': r_smooth,
        'sigma_grid': sigma_grid
    }

    # Save output data directly to HDF5
    save_cosmic_web_hdf5(
        output_path=output_path,
        structure_type=structure_type,
        eigenvalues=w,
        eigenvectors=v,
        metadata=metadata
    )
    print("Done!")

if __name__ == "__main__":
    #--- Welcome message and version information ---
    print(f"StePS Cosmic Web Finder {_VERSION} by {_AUTHOR}, {_YEAR}")
    print(f"\n\tThis program is free software; you can redistribute it and/or modify\n\tit under the terms of the GNU General Public License as published by\n\tthe Free Software Foundation; either version 2 of the License,\n\tor (at your option) any later version.\n\n\tThis program is distributed in the hope that it will be useful,\n\tbut WITHOUT ANY WARRANTY; without even the implied warranty of\n\tMERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the\n\tGNU General Public License for more details.\n\n")
    print(f"\nNote: The input snapshot must contain 'PartType1/Coordinates' and 'PartType1/Accelerations' datasets.\nThe latter can be turned on in StePS with the \"-DSAVE_ACCELERATIONS\" compile-time flag.\n\n")
    parser = argparse.ArgumentParser(description=_DESCRIPTION)
    parser.add_argument("-i", "--input", type=str, required=True, help="Path to HDF5 snapshot file")
    parser.add_argument("--geom", type=str, choices=['R3', 'S1R2', 'T3'], required=True, help="Geometry type")
    parser.add_argument("--res", type=int, default=128, help="Grid resolution per axis")
    # Note: L and Lz are now optional since they can be read from the header
    parser.add_argument("--L", type=float, default=None, help="Box length L (Mpc). If omitted, reads 'BoxSize' from Header.")
    parser.add_argument("--Lz", type=float, default=None, help="Periodic length Lz (Mpc) for S1R2. If omitted, reads 'Lz' from Header.")
    parser.add_argument("--rsmooth", type=float, default=2.0, help="Gaussian smoothing scale [Mpc]")
    parser.add_argument("-o", "--out", type=str, default="cosmic_web.h5", help="Output h5 filename (.h5)")

    args = parser.parse_args()
    classify_cosmic_web(
        snapshot_path=args.input, 
        geometry=args.geom, 
        grid_res=args.res, 
        L=args.L, 
        Lz=args.Lz, 
        r_smooth=args.rsmooth, 
        output_path=args.out
    )
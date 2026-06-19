#!/usr/bin/env python3

import numpy as np
import pyvista as pv
import argparse
import sys
import os

# adding ../Utils/ to the system path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Utils'))
from inputoutput import load_density_hdf5, load_cosmic_web_hdf5

_VERSION="v0.0.1.0"
_YEAR="2026"
_AUTHOR="Gabor Racz"
_DESCRIPTION="StePS 3D Volumetric Density Viewer"

def visualize_density_3d(density_path, cw_path=None, mode='all', vmin=None, vmax=None):
    print(f"--- Loading Density Data from {density_path} ---")
    try:
        density_data, density_attrs = load_density_hdf5(density_path, verbose=True)
        delta = density_data['overdensity']
    except Exception as e:
        print(f"Error loading density file: {e}")
        return

    Nx_d, Ny_d, Nz_d = delta.shape
    print(f"Density field resolution: {Nx_d}x{Ny_d}x{Nz_d}")

    # Compute log10(delta + 1) with a safe lower bound for absolute cosmic voids
    log_delta = np.log10(np.clip(delta + 1.0, 1e-5, None)).astype(np.float32)

    # Extract configuration constraints from the primary density file
    L_d = density_attrs.get('L', 1.0)
    Lz_d = density_attrs.get('Lz', L_d)
    geometry = density_attrs.get('geometry', 'R3')

    # Handling environmental masking if a Cosmic Web file is provided
    if cw_path is not None and mode != 'all':
        print(f"--- Loading Cosmic Web Classification from {cw_path} ---")
        try:
            cw_data, cw_attrs = load_cosmic_web_hdf5(cw_path)
            structure_type = cw_data['structure_type']
        except Exception as e:
            print(f"Error loading cosmic web file: {e}")
            return

        # Validate physical dimension compatibility across files
        L_cw = cw_attrs.get('L', 1.0)
        Lz_cw = cw_attrs.get('Lz', L_cw)
        if not (np.isclose(L_d, L_cw, rtol=1e-3) and np.isclose(Lz_d, Lz_cw, rtol=1e-3)):
            print(f"WARNING: Physical scale mismatch! Density: L={L_d}, Lz={Lz_d} | Cosmic Web: L={L_cw}, Lz={Lz_cw}")
            print("Proceeding anyway assuming identical spatial bounds bounding boxes...")

        Nx_c, Ny_c, Nz_c = structure_type.shape
        print(f"Cosmic Web array resolution: {Nx_c}x{Ny_c}x{Nz_c}")

        # Vectorized Nearest-Neighbor index resampling to bridge different resolutions
        if (Nx_d, Ny_d, Nz_d) != (Nx_c, Ny_c, Nz_c):
            print("\tResolutions differ. Resampling Cosmic Web grid using nearest-neighbor mapping...")
            x_idx = np.clip((np.arange(Nx_d) * (Nx_c / Nx_d)).astype(int), 0, Nx_c - 1)
            y_idx = np.clip((np.arange(Ny_d) * (Ny_c / Ny_d)).astype(int), 0, Ny_c - 1)
            z_idx = np.clip((np.arange(Nz_d) * (Nz_c / Nz_d)).astype(int), 0, Nz_c - 1)
            # Advanced indexing via index-crossproduct mesh matching
            cw_resampled = structure_type[np.ix_(x_idx, y_idx, z_idx)]
        else:
            cw_resampled = structure_type

        # Map environment modes to explicit structure type IDs
        mode_map = {'voids': 1, 'sheets': 2, 'filaments': 3, 'clusters': 4}
        target_id = mode_map[mode]

        print(f"--- Masking Density Field: Isolating {mode.upper()} (ID: {target_id}) ---")
        # Turn off voxels that do not match the environment mask by assigning NaN
        mask = (cw_resampled != target_id)
        log_delta[mask] = np.nan
        
        valid_count = np.sum(~mask)
        print(f"Kept {valid_count} voxels out of {log_delta.size} belonging to {mode.upper()}.")
        if valid_count == 0:
            print("ERROR: No matching voxels found for this structural filter mode. Verify your classification bounds.")
            return
            
    elif mode != 'all' and cw_path is None:
        raise ValueError(f"Mode '{mode}' requires an environmental classification file passed via --cosmic_web.")

    # Color range normalization adjustments
    if vmin is None:
        vmin = float(np.nanmin(log_delta)) if not np.all(np.isnan(log_delta)) else 0.0
    if vmax is None:
        vmax = float(np.nanmax(log_delta)) if not np.all(np.isnan(log_delta)) else 1.0
    print(f"Color bounds set to: vmin = {vmin:.2f}, vmax = {vmax:.2f}")

    # Build the spatial grid representation
    dx, dy = L_d / Nx_d, L_d / Ny_d
    dz = Lz_d / Nz_d if geometry == 'S1R2' else L_d / Nz_d
    
    if geometry in ['R3', 'S1R2']:
        origin = (-L_d / 2.0, -L_d / 2.0, 0.0 if geometry == 'S1R2' else -L_d / 2.0)
    else: # T3
        origin = (0.0, 0.0, 0.0)

    grid = pv.ImageData()
    grid.dimensions = (Nx_d, Ny_d, Nz_d)
    grid.spacing = (dx, dy, dz)
    grid.origin = origin

    # Flatten coordinates utilizing Fortran ordering mechanics
    grid.point_data['LogDensity'] = log_delta.flatten(order='F')

    # Assemble the Volumetric Render Window
    print("--- Generating Volumetric Render ---")
    pl = pv.Plotter()

    # Pass the grid configuration directly; NaNs will dynamically appear invisible
    vol = pl.add_volume(
        grid,
        scalars='LogDensity',
        cmap='magma',
        opacity='linear', # The environment mask handles transparency now!
        clim=[vmin, vmax],
        shade=True,
        mapper='smart'
    )

    # Frame styles layout details
    pl.add_bounding_box(color='white', line_width=1.0)
    pl.show_grid(color='gray', xtitle='X [Mpc]', ytitle='Y [Mpc]', ztitle='Z [Mpc]', font_size=10)
    pl.add_axes()
    pl.set_background('black')
    
    title_text = f"StePS Density Field: {mode.upper()} Mask Profile" if cw_path else "StePS Full Density Field"
    pl.add_text(f"{title_text}\nScalar: log10(delta + 1)", font_size=11, color='white')

    print("--- Opening Interactive Window ---")
    pl.show()

if __name__ == "__main__":
    print(f"StePS Density 3V Environment Viewer {_VERSION} by {_AUTHOR}, {_YEAR}\n")

    parser = argparse.ArgumentParser(description=_DESCRIPTION)
    parser.add_argument("-i", "--input", type=str, required=True, help="Path to the overdensity field .h5 file")
    parser.add_argument("-c", "--cosmic_web", type=str, default=None, help="Optional cosmic web classification file for structural masking")
    parser.add_argument("--mode", type=str, choices=['all', 'clusters', 'filaments', 'sheets', 'voids'], default='all', help="Target structural group to isolate")
    parser.add_argument("--vmin", type=float, default=None, help="Manually override min log scalar colorbar boundary")
    parser.add_argument("--vmax", type=float, default=None, help="Manually override max log scalar colorbar boundary")

    args = parser.parse_args()
    visualize_density_3d(args.input, cw_path=args.cosmic_web, mode=args.mode, vmin=args.vmin, vmax=args.vmax)
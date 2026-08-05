#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys
import os

# adding ../Utils/ to the system path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Utils'))
from inputoutput import load_density_hdf5

_VERSION="v0.0.2.0"
_YEAR="2026"
_AUTHOR="Gabor Racz"
_DESCRIPTION="StePS 2D visualization tool for overdensity fields."

def visualize_density_slice(h5_path, slice_axis='z', slice_idx=None, slice_thickness=1, vmin=None, vmax=None, title=None, cmap='magma'):
    print(f"Loading data from {h5_path}...")
    try:
        data, attrs = load_density_hdf5(h5_path)
        delta = data['overdensity']
    except Exception as e:
        print(f"Error loading {h5_path}: {e}")
        return

    Nx, Ny, Nz = delta.shape
    print(f"Loaded density shape: {delta.shape} (Nx={Nx}, Ny={Ny}, Nz={Nz})")
    
    if slice_idx is None:
        if slice_axis == 'x': slice_idx = Nx // 2
        elif slice_axis == 'y': slice_idx = Ny // 2
        else: slice_idx = Nz // 2

    slice_idx = int(slice_idx)
    slice_thickness = int(slice_thickness)
    if slice_thickness < 1:
        raise ValueError("slice_thickness must be a positive integer")

    observer_axis = {'x': 0, 'y': 1, 'z': 2}[slice_axis]
    axis_size = delta.shape[observer_axis]
    if slice_idx < 0 or slice_idx >= axis_size:
        raise ValueError(f"slice_idx must be between 0 and {axis_size - 1}")

    slice_thickness = min(slice_thickness, axis_size)
    slice_start = max(0, min(slice_idx - slice_thickness // 2, axis_size - slice_thickness))
    slice_stop = slice_start + slice_thickness
    print(f"Extracting {slice_thickness} pixel slice along {slice_axis.upper()}-axis from index {slice_start} to {slice_stop - 1}...")
    
    if slice_axis == 'x':
        slice_2d = np.mean(delta[slice_start:slice_stop, :, :], axis=0)
        xlabel, ylabel = r'$y [$Mpc$]$', r'$z [$Mpc$]$'
    elif slice_axis == 'y':
        slice_2d = np.mean(delta[:, slice_start:slice_stop, :], axis=1)
        xlabel, ylabel = r'$x [$Mpc$]$', r'$z [$Mpc$]$'
    elif slice_axis == 'z':
        slice_2d = np.mean(delta[:, :, slice_start:slice_stop], axis=2)
        xlabel, ylabel = r'$x [$Mpc$]$', r'$y [$Mpc$]$'
    else:
        raise ValueError("slice_axis must be 'x', 'y', or 'z'")

    # Calculate log10(delta + 1). 
    # Add a small epsilon to delta + 1 to prevent log10(0) inside empty voids.
    log_delta = np.log10(np.clip(slice_2d + 1.0, 1e-6, None))

    # Calculate physical extent
    L = attrs['L']
    Lz = attrs.get('Lz', L)
    geometry = attrs.get('geometry', 'R3')
    observer_length = Lz if geometry == 'S1R2' else L
    slice_coord_idx = slice_start + (slice_thickness - 1) / 2
    slice_coord = (slice_coord_idx / axis_size - 0.5) * observer_length
    thickness_mpc = slice_thickness * observer_length / axis_size
    
    if geometry == 'R3':
        extent = [-L/2, L/2, -L/2, L/2]
    elif geometry == 'S1R2':
        if slice_axis == 'z':
            extent = [-L/2, L/2, -L/2, L/2]
            slice_coord = (slice_coord_idx / axis_size) * Lz
        else:
            extent = [-L/2, L/2, 0, Lz]
            slice_coord = (slice_coord_idx / axis_size - 0.5) * Lz
    elif geometry == 'T3':
        extent = [0, L, 0, L]
        slice_coord = (slice_coord_idx / axis_size) * L

    # Plotting
    fig, ax = plt.subplots(figsize=(9, 8), dpi=120)
    
    # 'magma' is an excellent perceptually uniform colormap for density maps
    cax = ax.imshow(log_delta.T, cmap=cmap, origin='lower', interpolation='nearest', 
                    extent=extent, vmin=vmin, vmax=vmax)

    if title==None:
        ax.set_title(f"Dark Matter Overdensity Field\n(Slice {slice_axis.upper()} = {slice_coord:.2f} Mpc, thickness = {thickness_mpc:.2f} Mpc)", fontsize=13, pad=15)
    else:
        ax.set_title(title, fontsize=13, pad=15)

    cbar = fig.colorbar(cax, shrink=0.82, pad=0.04)
    cbar.set_label(r'$\log_{10}(\delta + 1)$', fontsize=13, rotation=270, labelpad=20)
    cbar.ax.tick_params(labelsize=11)
    
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print(f"StePS Density visualization {_VERSION} by {_AUTHOR}, {_YEAR}")

    parser = argparse.ArgumentParser(description=_DESCRIPTION)
    parser.add_argument("-i", "--input", required=True, type=str, default="density.h5", help="Path to the generated .h5 density file")
    parser.add_argument("--axis", type=str, choices=['x', 'y', 'z'], default='z', help="Axis to slice along")
    parser.add_argument("--index", type=int, default=None, help="Grid index of the slice center (defaults to the middle)")
    parser.add_argument("--thickness", type=int, default=1, help="Slice thickness in pixels (defaults to 1)")
    parser.add_argument("--vmin", type=float, default=None, help="Minimum log10(delta+1) for colorbar")
    parser.add_argument("--vmax", type=float, default=None, help="Maximum log10(delta+1) for colorbar")
    parser.add_argument("--cmap", type=str, default='magma', help="Colormap for the density field")
    args = parser.parse_args()
    visualize_density_slice(args.input, slice_axis=args.axis, slice_idx=args.index, slice_thickness=args.thickness, vmin=args.vmin, vmax=args.vmax, cmap=args.cmap)
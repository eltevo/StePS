#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys
import os

# adding ../Utils/ to the system path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Utils'))
from inputoutput import load_density_hdf5

_VERSION="v0.0.1.1"
_YEAR="2026"
_AUTHOR="Gabor Racz"
_DESCRIPTION="StePS 2D visualization tool for log(overdensity) fields."

def visualize_density_slice(h5_path, slice_axis='z', slice_idx=None, vmin=None, vmax=None, title=None):
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
    print(f"Extracting 2D slice along {slice_axis.upper()}-axis at index {slice_idx}...")
    
    if slice_axis == 'x':
        slice_2d = delta[slice_idx, :, :]
        xlabel, ylabel = r'$y [$Mpc$]$', r'$z [$Mpc$]$'
    elif slice_axis == 'y':
        slice_2d = delta[:, slice_idx, :]
        xlabel, ylabel = r'$x [$Mpc$]$', r'$z [$Mpc$]$'
    elif slice_axis == 'z':
        slice_2d = delta[:, :, slice_idx]
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
    
    if geometry == 'R3':
        extent = [-L/2, L/2, -L/2, L/2]
        slice_coord = (slice_idx / delta.shape[0] - 0.5) * L
    elif geometry == 'S1R2':
        if slice_axis == 'z':
            extent = [-L/2, L/2, -L/2, L/2]
            slice_coord = (slice_idx / delta.shape[2]) * Lz
        else:
            extent = [-L/2, L/2, 0, Lz]
            slice_coord = (slice_idx / delta.shape[0] - 0.5) * Lz
    elif geometry == 'T3':
        extent = [0, L, 0, L]
        slice_coord = (slice_idx / delta.shape[0]) * L

    # Plotting
    fig, ax = plt.subplots(figsize=(9, 8), dpi=120)
    
    # 'magma' is an excellent perceptually uniform colormap for density maps
    cax = ax.imshow(log_delta.T, cmap='magma', origin='lower', interpolation='nearest', 
                    extent=extent, vmin=vmin, vmax=vmax)

    if title==None:
        ax.set_title(f"Dark Matter Overdensity Field\n(Slice {slice_axis.upper()} = {slice_coord:.2f} Mpc)", fontsize=13, pad=15)
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
    parser.add_argument("--index", type=int, default=None, help="Grid index of the slice (defaults to the middle)")
    parser.add_argument("--vmin", type=float, default=None, help="Minimum log10(delta+1) for colorbar")
    parser.add_argument("--vmax", type=float, default=None, help="Maximum log10(delta+1) for colorbar")
    
    args = parser.parse_args()
    visualize_density_slice(args.input, slice_axis=args.axis, slice_idx=args.index, vmin=args.vmin, vmax=args.vmax)
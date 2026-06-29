#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import argparse
import sys
import os
# adding ../Utils/ to the system path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Utils'))
from inputoutput import load_cosmic_web_hdf5

_VERSION="v0.0.1.2"
_YEAR="2026"
_AUTHOR="Gabor Racz"
_DESCRIPTION="StePS Cosmic Web slice visualization tool for 2D cross-sections of the cosmic web classification."

def visualize_cosmic_web_slice(h5_path, slice_axis='z', slice_idx=None):
    print(f"Loading data from {h5_path}...")
    try:
        data, attrs = load_cosmic_web_hdf5(h5_path)
        structure_type = data['structure_type']
    except Exception as e:
        print(f"Error loading {h5_path}: {e}")
        return

    Nx, Ny, Nz = structure_type.shape
    print(f"Loaded data shape: {structure_type.shape} (Nx={Nx}, Ny={Ny}, Nz={Nz})")
    
    # Default to the middle slice if no index is provided
    if slice_idx is None:
        if slice_axis == 'x': slice_idx = Nx // 2
        elif slice_axis == 'y': slice_idx = Ny // 2
        else: slice_idx = Nz // 2

    # Ensure the slice index is within bounds
    slice_idx = int(slice_idx)
    
    print(f"Extracting 2D slice along {slice_axis.upper()}-axis at index {slice_idx}...")
    
    if slice_axis == 'x':
        slice_2d = structure_type[slice_idx, :, :]
        xlabel, ylabel = r'$y [$Mpc$]$', r'$z [$Mpc$]$'
    elif slice_axis == 'y':
        slice_2d = structure_type[:, slice_idx, :]
        xlabel, ylabel = r'$x [$Mpc$]$', r'$z [$Mpc$]$'
    elif slice_axis == 'z':
        slice_2d = structure_type[:, :, slice_idx]
        xlabel, ylabel = r'$x [$Mpc$]$', r'$y [$Mpc$]$'
    else:
        raise ValueError("slice_axis must be 'x', 'y', or 'z'")

    # Define a discrete color palette
    # 1: Void (Black)
    # 2: Sheet/Wall (Dark Slate Blue)
    # 3: Filament (Orange)
    # 4: Cluster/Knot (Bright Yellow)
    colors = ['#000000', '#2E4053', '#E67E22', '#F1C40F']
    cmap = ListedColormap(colors)
    
    # BoundaryNorm ensures the integers map strictly to the distinct colors
    bounds = [0.5, 1.5, 2.5, 3.5, 4.5]
    norm = BoundaryNorm(bounds, cmap.N)

    # calculating the physical extent of the slice for proper axis labeling
    L = attrs['L']
    Lz = attrs.get('Lz', L)  # Use Lz if available
    if attrs['geometry'] == 'R3':
        extent = [-L/2, L/2, -L/2, L/2]
        slice_coordinate = (slice_idx / structure_type.shape[0] - 0.5) * L
    elif attrs['geometry'] == 'S1R2':
        if slice_axis == 'z':
            extent = [-L/2, L/2, -L/2, L/2]
            slice_coordinate = (slice_idx / structure_type.shape[2]) * Lz
        else:
            extent = [-L/2, L/2, 0, Lz]
            slice_coordinate = (slice_idx / structure_type.shape[0] - 0.5) * Lz
    elif attrs['geometry'] == 'T3':
        extent = [0, L, 0, L]
        slice_coordinate = (slice_idx / structure_type.shape[0]) * L
    else:
        raise ValueError("Unknown geometry type in attributes")

    # Plotting
    fig, ax = plt.subplots(figsize=(9, 8), dpi=120)
    
    # origin='lower' ensures (0,0) is at the bottom left
    cax = ax.imshow(slice_2d.T, cmap=cmap, norm=norm, origin='lower', interpolation='nearest', extent=extent)
    
    ax.set_title(f"Cosmic Web Classification\n(Slice {slice_axis.upper()} = {slice_coordinate} Mpc)", fontsize=12, pad=15)

    # Configure Colorbar
    cbar = fig.colorbar(cax, ticks=[1, 2, 3, 4], shrink=0.82, pad=0.04)
    cbar.ax.set_yticklabels(['Void', 'Sheet', 'Filament', 'Cluster'], fontsize=11)
    
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    #--- Welcome message and version information ---
    print(f"StePS Cosmic Web visualization {_VERSION} by {_AUTHOR}, {_YEAR}")
    print(f"\n\tThis program is free software; you can redistribute it and/or modify\n\tit under the terms of the GNU General Public License as published by\n\tthe Free Software Foundation; either version 2 of the License,\n\tor (at your option) any later version.\n\n\tThis program is distributed in the hope that it will be useful,\n\tbut WITHOUT ANY WARRANTY; without even the implied warranty of\n\tMERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the\n\tGNU General Public License for more details.\n\n")

    parser = argparse.ArgumentParser(description=_DESCRIPTION)
    parser.add_argument("-i", "--input", required=True, type=str, default="cosmic_web.npz", help="Path to the generated .npz file")
    parser.add_argument("--axis", type=str, choices=['x', 'y', 'z'], default='z', help="Axis to slice along")
    parser.add_argument("--index", type=int, default=None, help="Grid index of the slice (defaults to the middle)")
    
    args = parser.parse_args()
    
    visualize_cosmic_web_slice(args.input, slice_axis=args.axis, slice_idx=args.index)
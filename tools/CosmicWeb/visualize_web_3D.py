#!/usr/bin/env python3

import numpy as np
import pyvista as pv
import argparse
from matplotlib.colors import ListedColormap
import sys
import os
# adding ../Utils/ to the system path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Utils'))
from inputoutput import load_cosmic_web_hdf5

_VERSION="v0.0.2.0"
_YEAR="2026"
_AUTHOR="Gabor Racz"
_DESCRIPTION="StePS 3D Volumetric Visualization of the Cosmic Web."

def visualize_cosmic_web_3d(h5_path, mode='clusters'):
    print(f"--- Loading data from {h5_path} ---")
    try:
        data, attrs = load_cosmic_web_hdf5(h5_path)
        # Extract the 3D structure array (Shape: Nx, Ny, Nz)
        structure_type = data['structure_type']
    except Exception as e:
        print(f"Error loading {h5_path}: {e}")
        return
    
    # Extract configuration constraints from the primary density file
    L = attrs.get('L', 1.0)
    Lz = attrs.get('Lz', L)
    geometry = attrs.get('geometry', 'R3')

    # Preparing the Grid
    print("--- Preparing 3D Grid ---")
    # Cast back to float32 or int32 to prevent PyVista from assuming a 0-255 image texture
    structure_float = structure_type.astype(np.float32)
    
    # PyVista requires the 3D array to be flattened in Fortran order ('F') 
    # to correctly map the data coordinates to the spatial grid.
    flat_data = structure_float.flatten(order='F')

    # Create a spatial container (ImageData) for the volume
    Nx, Ny, Nz =structure_float.shape
    dx, dy = L / Nx, L / Ny
    dz = Lz / Nz if geometry == 'S1R2' else L / Nz

    if geometry in ['R3', 'S1R2']:
        origin = (-L / 2.0, -L / 2.0, 0.0 if geometry == 'S1R2' else -L / 2.0)
    else: # T3
        origin = (0.0, 0.0, 0.0)

    grid = pv.ImageData()
    grid.dimensions = (Nx, Ny, Nz)
    grid.spacing = (dx, dy, dz)
    grid.origin = origin

    # Assign the classification data to the grid's points
    grid.point_data['Structure'] = flat_data

    # 2. Define Colors and Opacity
    # 1: Void, 2: Sheet, 3: Filament, 4: Cluster
    colors = ["#FFFFFF", "#3395FD", "#FF8F2D", '#F1C40F']
    cmap = ListedColormap(colors)

    # Define our 4-point Opacity Transfer Functions
    if mode == 'clusters':
        opacity = [0.0, 0.0, 0.0, 1.0]
    elif mode == 'filaments':
        opacity = [0.0, 0.0, 1.0, 0.1]
    elif mode == 'sheets':
        opacity = [0.0, 1.0, 0.0, 0.0]
    elif mode == 'voids':
        opacity = [1.0, 0.0, 0.0, 0.0]
    else: # 'all'
        opacity = [0.0, 0.05, 0.4, 1.0]

    # Rendering the Scene
    print("--- Generating Volumetric Render ---")
    pl = pv.Plotter()

    structure_annotations = {
        1.0: "Void",
        2.0: "Sheet",
        3.0: "Filament",
        4.0: "Cluster"
    }

    # Defining the colorbar
    colorbar_features = {
        "title": "Structure Type",
        "color": "white",     
        "label_font_size": 12,
        "title_font_size": 14,
        "shadow": True,
        "vertical": True,
        "n_labels": 0
    }

    # Add the volume rendering
    vol = pl.add_volume(
        grid,
        scalars='Structure',
        cmap=cmap,
        opacity=opacity,
        clim=[1, 4],     # FORCE the color/opacity limits to explicitly map onto our 1-4 categories
        shade=True,      # Enables 3D lighting/shadows
        mapper='smart',
        annotations=structure_annotations,
        scalar_bar_args=colorbar_features,
        show_scalar_bar=True
    )

    # Scene aesthetics
    pl.add_bounding_box(color='white', line_width=1.0)
    pl.show_grid(color='gray', xtitle='X [Mpc]', ytitle='Y [Mpc]', ztitle='Z [Mpc]', font_size=10)
    pl.add_axes()
    pl.set_background('black')
    pl.add_text(f'Cosmic Web: {mode.upper()} Mode\nRotate with Mouse', font_size=12, color='white')

    print("--- Opening Interactive Window ---")
    pl.show()

if __name__ == "__main__":
    #--- Welcome message and version information ---
    print(f"StePS Cosmic Web 3D visualization {_VERSION} by {_AUTHOR}, {_YEAR}")
    print(f"\n\tThis program is free software; you can redistribute it and/or modify\n\tit under the terms of the GNU General Public License as published by\n\tthe Free Software Foundation; either version 2 of the License,\n\tor (at your option) any later version.\n\n\tThis program is distributed in the hope that it will be useful,\n\tbut WITHOUT ANY WARRANTY; without even the implied warranty of\n\tMERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the\n\tGNU General Public License for more details.\n\n")

    parser = argparse.ArgumentParser(description=_DESCRIPTION)
    parser.add_argument("-i", "--input", type=str, required=True, help="Path to the generated .h5 file")
    parser.add_argument("--mode", type=str, choices=['clusters', 'filaments', 'sheets', 'voids', 'all'], default='all', help="Focus mode for visualization")
    
    args = parser.parse_args()
    visualize_cosmic_web_3d(args.input, mode=args.mode)
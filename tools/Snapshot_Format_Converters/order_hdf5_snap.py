#!/usr/bin/env python3

#********************************************************************************#
#  order_hdf5_snap.py - Script for ordering particle data in HDF5 snapshots      #
#                                                                                #
#    Copyright (C) 2026 Gabor Racz                                               #
#                                                                                #
#    This program is free software; you can redistribute it and/or modify        #
#    it under the terms of the GNU General Public License as published by        #
#    the Free Software Foundation; either version 2 of the License, or           #
#    (at your option) any later version.                                         #
#                                                                                #
#    This program is distributed in the hope that it will be useful,             #
#    but WITHOUT ANY WARRANTY; without even the implied warranty of              #
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the               #
#    GNU General Public License for more details.                                #
#********************************************************************************#

import h5py
import numpy as np
import argparse
import sys

_VERSION = "1.0.0"
_AUTHOR = "Gabor Racz"
_DATE = "2026"
_DESCRIPTION = "A script for ordering particle data in HDF5 snapshots."
_DESCRIPTION_LONG = "A script for ordering particle data in HDF5 snapshots.\n\tThis can be useful to order particles based on specific geometric criteria.\n\tProper ordering can speed up parallel Octree/Direct simulations by evening out the computational load through the particle order."

def order_snapshot(input_path, output_path, order_by='z'):
    """
    Reads an HDF5 snapshot, orders particles based on a specific coordinate criterion,
    and saves to a new file.
    
    Parameters:
    - input_path: str, path to the input HDF5 snapshot
    - output_path: str, path to save the ordered HDF5 snapshot
    - order_by: str, 'z', 'r', 'rho', or 'phi'
    """
    valid_orders = ['z', 'r', 'rho', 'phi']
    if order_by not in valid_orders:
        raise ValueError(f"Invalid order_by. Choose from {valid_orders}")

    try:
        with h5py.File(input_path, 'r') as f_in, h5py.File(output_path, 'w') as f_out:
            print(f"Processing: {input_path} (Ordering by: {order_by})...")

            # Copy Header
            if "Header" in f_in:
                f_in.copy("Header", f_out)
                print("\tSuccessfully copied Header group.")

            for group_name in f_in.keys():
                if not group_name.startswith("PartType"):
                    continue
                
                print(f"\tOrdering group: {group_name}...")
                group_in = f_in[group_name]
                group_out = f_out.create_group(group_name)

                # Get Coordinates to calculate sorting indices
                if "Coordinates" not in group_in:
                    print(f"\tWarning: No Coordinates in {group_name}. Skipping...")
                    continue

                coords = group_in["Coordinates"][:]
                x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]

                # Define the sorting key based on user selection
                if order_by == 'z':
                    key = z
                elif order_by == 'r':
                    # Spherical radius: sqrt(x^2 + y^2 + z^2)
                    key = np.sqrt(x**2 + y**2 + z**2)
                elif order_by == 'rho':
                    # Cylindrical radius: sqrt(x^2 + y^2)
                    key = np.sqrt(x**2 + y**2)
                elif order_by == 'phi':
                    # Cylindrical angle: atan2(y, x) returns values in [-pi, pi]
                    key = np.arctan2(y, x)

                # Generate sorting indices (ascending)
                indices = np.argsort(key)

                # Apply indices to all datasets in the group
                for ds_name in group_in.keys():
                    data = group_in[ds_name][:]
                    ordered_data = data[indices]
                    
                    # Write dataset and preserve attributes
                    ds_out = group_out.create_dataset(ds_name, data=ordered_data)
                    for attr_name, attr_val in group_in[ds_name].attrs.items():
                        ds_out.attrs[attr_name] = attr_val

            print(f"...Ordering finished. Saved to: {output_path}.\n")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Beginning of the script
    # Welcome message
    print("\norder_hdf5_snap.py v%s\n\t%s\n\tCopyright (C) %s %s\n" % (_VERSION,_DESCRIPTION,_DATE,_AUTHOR))
    print("\nThis program comes with ABSOLUTELY NO WARRANTY.\nThis is free software, and you are welcome to redistribute it under certain conditions.\nSee the file LICENSE for details.\n\n")
    parser = argparse.ArgumentParser(description=_DESCRIPTION_LONG)
    parser.add_argument("-i", "--input", required=True, help="Path to input HDF5 snapshot")
    parser.add_argument("-o", "--output", required=True, help="Path to output ordered HDF5 snapshot")
    parser.add_argument("-r", "--order", choices=['z', 'r', 'rho', 'phi'], default='phi', help="Order by coordinate: 'z', 'r', 'rho', or 'phi' (default: 'phi')")

    args = parser.parse_args()
    order_snapshot(args.input, args.output, args.order)
#!/usr/bin/env python3

#********************************************************************************#
#  scale_hdf5_glass.py - Script for rescaling particle data in HDF5 snapshots    #
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
_DESCRIPTION = "A script for rescaling particle data in HDF5 snapshots."
_DESCRIPTION_LONG = "A script for rescaling particle data in HDF5 snapshots. This can be useful for rescaling glasses to match the desired resolution of a StePS simulation."

def scale_snapshot(input_path, output_path, scale_factor):
    """
    Reads an HDF5 snapshot, rescales particle coordinates, and saves to a new file.
    
    Parameters:
    - input_path: str, path to the input HDF5 snapshot
    - output_path: str, path to save the rescaled HDF5 snapshot
    - scale_factor: float, factor applied to coordinates and length-like header attributes
    """
    scale_factor = float(scale_factor)

    try:
        with h5py.File(input_path, 'r') as f_in, h5py.File(output_path, 'w') as f_out:
            print(f"Processing: {input_path} (Scale factor: {scale_factor})...")

            # Copy Header
            if "Header" in f_in:
                f_in.copy("Header", f_out)
                header = f_out["Header"]
                for attr_name in ("BoxSize", "SimulationRadius"):
                    if attr_name in header.attrs:
                        header.attrs[attr_name] = header.attrs[attr_name] * scale_factor
                print("\tSuccessfully copied Header group.")

            for group_name in f_in.keys():
                if not group_name.startswith("PartType"):
                    continue
                
                print(f"\tScaling group: {group_name}...")
                group_in = f_in[group_name]
                group_out = f_out.create_group(group_name)

                # Get Coordinates to rescale positions
                if "Coordinates" not in group_in:
                    print(f"\tWarning: No Coordinates in {group_name}. Skipping...")
                    continue
                # Get Masses to rescale individual particle masses if needed (optional)
                if "Masses" not in group_in:
                    print(f"\tNote: No Masses in {group_name}. Only coordinates will be scaled.")

                for ds_name in group_in.keys():
                    data = group_in[ds_name][:]
                    if ds_name == "Coordinates":
                        data = data * scale_factor
                    if ds_name == "Masses":
                        data = data * (scale_factor ** 3)  # Assuming mass scales with volume
                    # Write dataset and preserve attributes
                    ds_out = group_out.create_dataset(ds_name, data=data)
                    for attr_name, attr_val in group_in[ds_name].attrs.items():
                        ds_out.attrs[attr_name] = attr_val

            print(f"...Scaling finished. Saved to: {output_path}.\n")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Beginning of the script
    # Welcome message
    print("\nscale_hdf5_glass.py v%s\n\t%s\n\tCopyright (C) %s %s\n" % (_VERSION,_DESCRIPTION,_DATE,_AUTHOR))
    print("\nThis program comes with ABSOLUTELY NO WARRANTY.\nThis is free software, and you are welcome to redistribute it under certain conditions.\nSee the file LICENSE for details.\n\n")
    parser = argparse.ArgumentParser(description=_DESCRIPTION_LONG)
    parser.add_argument("-i", "--input", required=True, help="Path to input HDF5 snapshot")
    parser.add_argument("-o", "--output", required=True, help="Path to output rescaled HDF5 snapshot")
    parser.add_argument("-s", "--scale-factor", required=True, type=float, help="Scale factor applied to coordinates and length-like header attributes")

    args = parser.parse_args()
    scale_snapshot(args.input, args.output, args.scale_factor)
#!/usr/bin/env python3

#********************************************************************************#
#  shuffle_hdf5_snap.py - Script for shuffling particle data in HDF5 snapshots   #
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
_DESCRIPTION = "A script for shuffling particle data in HDF5 snapshots."
_DESCRIPTION_LONG = "A script for shuffling particle data in HDF5 snapshots.\n\tThis can be useful to break correlations in the data while preserving the overall structure and header information of the snapshot.\n\tSince the ICs are ordered in a specific geometric way in StePS, shuffling in certain cases can speed up parallel Octree simulations by randomizing particle order."

def shuffle_snapshot(input_path, output_path, seed=137):
    """
    Reads an HDF5 snapshot, shuffles particle data consistently across 
    datasets, and saves to a new file while preserving the header.
    Parameters:
    - input_path: str, path to the input HDF5 snapshot
    - output_path: str, path to save the shuffled HDF5 snapshot
    - seed: int, random seed for reproducibility (default: 137)
    """
    # Setting the random seed
    np.random.seed(seed)
    
    try:
        with h5py.File(input_path, 'r') as f_in, h5py.File(output_path, 'w') as f_out:
            print(f"Processing: {input_path}...")

            # Copy Header and its attributes exactly as they are
            if "Header" in f_in:
                f_in.copy("Header", f_out)
                print("\tSuccessfully copied Header group and attributes.")
            else:
                print("\tWarning: 'Header' group not found in input file.")

            # Iterate through all groups to find particle data (e.g., PartType0, PartType1...)
            for group_name in f_in.keys():
                if not group_name.startswith("PartType"):
                    continue
                
                print(f"\tShuffling group: {group_name}...")
                group_in = f_in[group_name]
                group_out = f_out.create_group(group_name)

                # Find a reference dataset to determine particle count (assuming all datasets in the group have the same number of particles)
                ref_ds_name = list(group_in.keys())[0]
                num_particles = group_in[ref_ds_name].shape[0]

                # Generate shuffled indices, and shuffle and write each dataset within the PartType group
                indices = np.arange(num_particles)
                np.random.shuffle(indices)
                for ds_name in group_in.keys():
                    data = group_in[ds_name][:]
                    
                    # Apply the same shuffled indices to all datasets in this group
                    shuffled_data = data[indices]
                    
                    # Create dataset in output with same dtype and shape
                    group_out.create_dataset(ds_name, data=shuffled_data)
                    
                    # Copy attributes of the dataset if they exist (e.g., units)
                    for attr_name, attr_val in group_in[ds_name].attrs.items():
                        group_out[ds_name].attrs[attr_name] = attr_val

            print(f"...shuffle finished. Saved to: {output_path}.\n")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Beginning of the script
    # Welcome message
    print("\nshuffle_hdf5_snap.py v%s\n\t%s\n\tCopyright (C) %s %s\n" % (_VERSION,_DESCRIPTION,_DATE,_AUTHOR))
    print("\nThis program comes with ABSOLUTELY NO WARRANTY.\nThis is free software, and you are welcome to redistribute it under certain conditions.\nSee the file LICENSE for details.\n\n")
    parser = argparse.ArgumentParser(description=_DESCRIPTION_LONG)
    parser.add_argument("-i", "--input", required=True, help="Path to input HDF5 snapshot")
    parser.add_argument("-o", "--output", required=True, help="Path to output shuffled HDF5 snapshot")
    parser.add_argument("-s", "--seed", type=int, default=137, help="Optional random seed (default: 137)")

    args = parser.parse_args()
    shuffle_snapshot(args.input, args.output, args.seed)
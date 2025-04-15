#!/usr/bin/env python3

#*******************************************************************************#
#  hdf52ascii_snapshot.py - A hdf5 to ASCII file converter for StePS            #
#     (STEreographically Projected cosmological Simulations) snapshots.         #
#    Copyright (C) 2017-2025 Gabor Racz                                         #
#                                                                               #
#    This program is free software; you can redistribute it and/or modify       #
#    it under the terms of the GNU General Public License as published by       #
#    the Free Software Foundation; either version 2 of the License, or          #
#    (at your option) any later version.                                        #
#                                                                               #
#    This program is distributed in the hope that it will be useful,            #
#    but WITHOUT ANY WARRANTY; without even the implied warranty of             #
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the              #
#    GNU General Public License for more details.                               #
#*******************************************************************************#

import numpy as np
import h5py
import sys
import time
from os.path import exists
# %matplotlib inline

_VERSION = "1.0.0"
_AUTHOR = "Gabor Racz"
_DATE = "2017-2025"
_DESCRIPTION = "HDF5 to ASCII file converter for StePS (STEreographically Projected cosmological Simulations) snapshots."

opt_list=['-steps', '-rockstar']
opt_desc=['-steps: (Default) StePS compatible output format.\n\tThe stored columns are X Y X VX VY VZ M\n', '-rockstar: Rockstar halo finder compatible output format.\n\tThe stored columns are X Y Z VX VY VZ ID\n']

# Beginning of the script

# Welcome message
print("\nhdf52ascii_snapshot.py v%s\n\t%s\n\tCopyright (C) %s %s\n" % (_VERSION,_DESCRIPTION,_DATE,_AUTHOR))
print("\nThis program comes with ABSOLUTELY NO WARRANTY.\nThis is free software, and you are welcome to redistribute it under certain conditions.\nSee the file LICENSE for details.\n\n")

if len(sys.argv) != 3 and len(sys.argv) != 4:
    print("Error:")
    print("usage: ./hdf52ascii_snapshot.py <input HDF5 snapshot> <output ASCII snapshot>\nExiting.")
    sys.exit(2)
if len(sys.argv) == 4 and str(sys.argv[3]) not in opt_list:
    print("Error: unkonwn option %s" % sys.argv[3])
    print("Supported options:")
    for i in range(len(opt_desc)):
        print("\t%s" % opt_desc[i])
    print("usage: ./hdf52ascii_snapshot.py <input HDF5 snapshot> <output ASCII snapshot> <output format (optional)>\nExiting.")
    sys.exit(2)

fmt="STEPS" # default format
if len(sys.argv) == 4:
    if sys.argv[3] == '-steps':
        fmt="STEPS"
    elif sys.argv[3] == '-rockstar':
        fmt="ROCKSTAR"
        print("\nWarning: in ROCKSTAR compatible output format, the mass of the particles is not saved in the output file. This is because the ROCKSTAR format does not support it.")
    else:
        print("Error: unkown option %s" % sys.argv[3])
        print("Supported options:")
        for i in range(len(opt_desc)):
            print("\t%s" % opt_desc[i])
        print("usage: ./hdf52ascii_snapshot.py <input HDF5 snapshot> <output ASCII snapshot> <option>\nExiting.")
        sys.exit(2)

#reading and converting the HDF5 file
if not exists(str(sys.argv[1])):
    print("Error: input file %s does not exist.\nExiting." % str(sys.argv[1]))
    sys.exit(2)
print("Reading the input HDF5 file...")
start = time.time()
HDF5_snapshot = h5py.File(str(sys.argv[1]), "r")
N = int(HDF5_snapshot['/Header'].attrs['NumPart_ThisFile'][1])
SCALE_FACTOR = HDF5_snapshot['/Header'].attrs['Time']
if SCALE_FACTOR == 0:
    print("\n\t!!! Warning: the scale factor is zero. Is this a valid snapshot?")
    print("\t    Note: In ROCKSTAR format, this will result zero velocities in the output.\n")
M_min = HDF5_snapshot['/Header'].attrs['MassTable'][1]
if M_min == 0:
    M_min = np.min(HDF5_snapshot['/PartType1/Masses'])
R_max = HDF5_snapshot['/Header'].attrs['BoxSize']
print("Number of particles:\t%i\nMinimal mass:\t%f*10e11 M_sol(/h)\nLinear size:\t%f Mpc(/h)\nScale factor:\t%f" % (N,M_min,R_max,SCALE_FACTOR))
ASCII_snapshot = np.zeros((N,7), dtype=np.double)
ASCII_snapshot[:,0:3] = HDF5_snapshot['/PartType1/Coordinates']
ASCII_snapshot[:,3:6] = HDF5_snapshot['/PartType1/Velocities']
if fmt == "STEPS":
   ASCII_snapshot[:,6] = HDF5_snapshot['/PartType1/Masses']
elif fmt == "ROCKSTAR":
    ASCII_snapshot[:,6] = np.double(HDF5_snapshot['/PartType1/ParticleIDs'])
    ASCII_snapshot[:,3:6] *= np.sqrt(SCALE_FACTOR) # StePS uses the same convention as GADGET, and the output velocities are divided by the scale factor. Rockstar ASCII does not do this.
HDF5_snapshot.close()
end = time.time()
print("..done in %fs. \n\n" % (end-start))
#Saving the ascii file
outputfilename = str(sys.argv[2])
print("Saving the %s ASCII file..." % outputfilename)
start = time.time()
np.savetxt(outputfilename, ASCII_snapshot, delimiter='\t', fmt='%.9f %.9f %.9f %.9f %.9f %.9f %i')
end = time.time()
print("..done in %fs. \n\n" % (end-start))

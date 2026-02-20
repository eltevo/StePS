#!/usr/bin/env python3

#********************************************************************************#
#  hdf52gadget.py - Script for converting a hdf5 snapshot to gadget binary       #
#                   format.                                                      #
#    Copyright (C) 2025 Gabor Racz                                               #
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

from os.path import exists
import numpy as np
import h5py
import sys
import time
from subprocess import call
# adding ../../StePS_IC/src/ to the system path
sys.path.insert(0, '../../StePS_IC/src/')
from inputoutput import *

_VERSION = "1.0.0"
_AUTHOR = "Gabor Racz"
_DATE = "2025"
_DESCRIPTION = "HDF5 to GADGET file converter for StePS (STEreographically Projected cosmological Simulations) snapshots."

UNIT_D=3.0856775814671917e24#=1Mpc Unit distance in cm (in the StePS code)

# Beginning of the script

# Welcome message
print("\nhdf52gadget.py v%s\n\t%s\n\tCopyright (C) %s %s\n" % (_VERSION,_DESCRIPTION,_DATE,_AUTHOR))
print("\nThis program comes with ABSOLUTELY NO WARRANTY.\nThis is free software, and you are welcome to redistribute it under certain conditions.\nSee the file LICENSE for details.\n\n")
print("Warning: only the particle coordinates and velocities are saved in the output file. Masses will be ignored.")

if len(sys.argv) != 3 and len(sys.argv) != 4:
    print("Error:")
    print("usage: ./hdf52gadget.py <input HDF5 snapshot> <output GADGET snapshot>\nExiting.")
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
Lbox = HDF5_snapshot['/Header'].attrs['BoxSize']
print("Number of particles:\t%i\nMinimal mass:\t%f*10e11 M_sol(/h)\nLinear size:\t%f Mpc(/h)\nScale factor:\t%f" % (N,M_min,Lbox,SCALE_FACTOR))
ASCII_snapshot = np.zeros((N,7), dtype=np.double)
ASCII_snapshot[:,0:3] = HDF5_snapshot['/PartType1/Coordinates']
ASCII_snapshot[:,3:6] = HDF5_snapshot['/PartType1/Velocities']
#ASCII_snapshot[:,6] = HDF5_snapshot['/PartType1/Masses']
HDF5_snapshot.close()
end = time.time()
print("..done in %fs. \n\n" % (end-start))
#Saving the ascii file
outputfilename = str(sys.argv[2])+".dat"
print("Saving the temporary %s ASCII file..." % outputfilename)
start = time.time()
np.savetxt(outputfilename, ASCII_snapshot, delimiter='\t', fmt='%.9f %.9f %.9f %.9f %.9f %.9f %i')
end = time.time()
print("..done in %fs. \n\n" % (end-start))
print("Converting the temporary ASCII file to GADGET format...")
start = time.time()
ascii2gadget(outputfilename, str(sys.argv[2]), Lbox, 1.0, UNIT_D)
end = time.time()
print("..done in %fs. \n\n" % (end-start))
#removing the temporary ASCII file
call(["rm", outputfilename])
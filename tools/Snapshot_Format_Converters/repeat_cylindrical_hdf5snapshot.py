#!/usr/bin/env python3

#********************************************************************************#
#  repeat_cylindrical_hdf5snapshot.py - Script for repeating a hdf5 snapshot     #
#                   periodically only in z direction                             #
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
_DESCRIPTION = "A script for repeating a cylindrical hdf5 StePS (STEreographically Projected cosmological Simulations) snapshot in the z direction."

precision = 0 #default 32bit floating point precision in the output

# Beginning of the script

# Welcome message
print("\nrepeat_cylindrical_hdf5snapshot.py v%s\n\t%s\n\tCopyright (C) %s %s\n" % (_VERSION,_DESCRIPTION,_DATE,_AUTHOR))
print("\nThis program comes with ABSOLUTELY NO WARRANTY.\nThis is free software, and you are welcome to redistribute it under certain conditions.\nSee the file LICENSE for details.\n\n")
print("Warning: only the particle coordinates and velocities are saved in the output file. Masses will be ignored.")

if len(sys.argv) != 4 and len(sys.argv) != 5:
    print("Error:")
    print("usage: ./repeat_cylindrical_hdf5snapshot.py <input HDF5 snapshot> <output HDF5 snapshot> <z repetition factor> (<periodicity in z direction (in Mpc)>)\nExiting.")
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
M_min = HDF5_snapshot['/Header'].attrs['MassTable'][1]
if M_min == 0:
    M_min = np.min(HDF5_snapshot['/PartType1/Masses'])
if len(sys.argv) == 5:
    Lbox = float(sys.argv[4])  #periodicity in z direction in Mpc
else:
    Lbox = HDF5_snapshot['/Header'].attrs['BoxSize']
Redshift = 1.0 / SCALE_FACTOR - 1.0
OmegaM = HDF5_snapshot['/Header'].attrs['Omega0']
OmegaL = HDF5_snapshot['/Header'].attrs['OmegaLambda']
HubbleParam = HDF5_snapshot['/Header'].attrs['HubbleParam']
print("Parameters of the loaded snapshot:")
print("\tNumber of particles: %i" % N)
print("\tScale factor: %f" % SCALE_FACTOR)
print("\tMinimum mass: %f" % M_min)
print("\tBox size: %f" % Lbox)
print("\tRedshift: %f" % Redshift)
print("\tOmegaM: %f" % OmegaM)
print("\tOmegaL: %f" % OmegaL)
print("\tHubble parameter: %f" % HubbleParam)
#repeating the snapshot in the z direction
z_repetition_factor = int(sys.argv[3])
if z_repetition_factor <= 0:
    raise ValueError("The z repetition factor must be a positive integer.")
print("Repeating the snapshot in the z direction by a factor of %i..." % z_repetition_factor)
if precision == 0:
    dataarray = np.zeros((N * z_repetition_factor, 7), dtype=np.float32) # x y z vx vy vz M
else:
    dataarray = np.zeros((N * z_repetition_factor, 7), dtype=np.float64) # x y z vx vy vz M
dataarray[:, 0:3] = np.repeat(HDF5_snapshot['/PartType1/Coordinates'][:], z_repetition_factor, axis=0)
for i in range(0,z_repetition_factor):
    dataarray[N*(i):N*(i+1), 0:3] = HDF5_snapshot['/PartType1/Coordinates'][:]
    dataarray[N*(i):N*(i+1), 2] += Lbox * i  #shifting the repeated images in the z direction
    dataarray[N*(i):N*(i+1), 3:6] = HDF5_snapshot['/PartType1/Velocities'][:]
    dataarray[N*(i):N*(i+1), 6] = HDF5_snapshot['/PartType1/Masses'][:]
writeHDF5snapshot(dataarray, str(sys.argv[2]), z_repetition_factor*Lbox, Redshift, OmegaM, OmegaL, HubbleParam, precision)
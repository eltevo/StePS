#!/usr/bin/env python3

#*******************************************************************************#
#  hdependent2hindependent_hdf5.py - Script for converting h dependent snapshot #
#                                    to h independent units
#     STEreographically Projected cosmological Simulations                      #
#    Copyright (C) 2022 Gabor Racz                                              #
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

from os.path import exists
import numpy as np
import h5py
import sys

# adding ../../StePS_IC/src/ to the system path
sys.path.insert(0, '../../StePS_IC/src/')
from inputoutput import *

_VERSION="v0.0.0.1dev"
precision = 0 #32bit

#Beginning of the script
print("\nThis is hdependent2hindependent_hdf5.py version %s.\nCopyright (C) 2022 Gabor Racz\n\nThis script is used to convert h dependent snapshot to h independent units.\n" % _VERSION)
if len(sys.argv) != 3:
    print("Error: wrong number of arguments!")
    print("usage: ./hdependent2hindependent_hdf5.py <input hdf5 file> <output hdf5 file>\nExiting.")
    sys.exit(2)
start = time.time()

#loading the input file:
print("Loading cosmological parameters...")
HDF5_snapshot = h5py.File(str(sys.argv[1]), "r")
Linearsize  = np.double(HDF5_snapshot['/Header'].attrs['BoxSize'])
Redshift    = np.double(HDF5_snapshot['/Header'].attrs['Redshift'])
OmegaM      = np.double(HDF5_snapshot['/Header'].attrs['Omega0'])
OmegaL      = np.double(HDF5_snapshot['/Header'].attrs['OmegaLambda'])
HubbleParam = np.double(HDF5_snapshot['/Header'].attrs['HubbleParam'])
print("\tLinear size\t%fMpc\n\tRedshift\t%f\n\tOmegaM\t\t%f\n\tOmegaL\t\t%f\n\tHubbleParam\t%f\n"%(Linearsize, Redshift, OmegaM, OmegaL, HubbleParam))
print("Loading particle data...")
Coordinates, Velocities, Masses = Load_snapshot(str(sys.argv[1]), RETURN_VELOCITIES=True)

#Saving the h independent particle data
dataarray = np.vstack((np.hstack((Coordinates,Velocities)).T,Masses)).T
print("\tConverting the snapshot to H0 independent units (h=%f)..."%HubbleParam)
#coordinates
dataarray[:,0:3] *= HubbleParam
#masses
dataarray[:,6] *= HubbleParam
print("\t...done\n")

writeHDF5snapshot(dataarray, str(sys.argv[2]), Linearsize*HubbleParam, Redshift, OmegaM, OmegaL, HubbleParam, precision)


end = time.time()
print("The snapshot conversion took %fs.\n" % (end-start))

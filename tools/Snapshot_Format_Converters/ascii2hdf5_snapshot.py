#!/usr/bin/env python3

#*******************************************************************************#
#  ascii2hdf5_snapshot.py - An ASCII to hdf5 file converter for StePS           #
#     (STEreographically Projected cosmological Simulations) snapshots.         #
#    Copyright (C) 2017-2022 Gabor Racz                                         #
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
# %matplotlib inline

#Beginning of the script
if len(sys.argv) != 4:
    print("Error:")
    print("usage: ./ascii2hdf5_snapshot.py <input ASCII snapshot> <output HDF5 snapshot> <precision 0: 32bit 1: 64bit>\nExiting.")
    sys.exit(2)

if int(sys.argv[3]) != 0 and int(sys.argv[3]) != 1:
    print("Error:")
    print("Unkown output precision.\nExiting.")
    sys.exit(2)

print("Reading the input ASCII file...")
start = time.time()
ASCII_snapshot=np.fromfile(str(sys.argv[1]), count=-1, sep='\t', dtype=np.float64)
ASCII_snapshot = ASCII_snapshot.reshape(int(len(ASCII_snapshot)/7),7)
N=len(ASCII_snapshot)
M_min = np.min(ASCII_snapshot[:,6])
R_max = np.max(np.abs(ASCII_snapshot[:,0:3]))
print("Number of particles:\t%i\nMinimal mass:\t%f*10e11M_sol\nMaximal radius:\t%fMpc" % (N,M_min,R_max))
end = time.time()
print("..done in %fs. \n\n" % (end-start))

print("Saving the snapshot in HDF5 format...")
start = time.time()
HDF5_snapshot = h5py.File(str(sys.argv[2]), "w")
#Creating the header
header_group = HDF5_snapshot.create_group("/Header")
#Writing the header attributes
header_group.attrs['NumPart_ThisFile'] = np.array([0,N,0,0,0,0],dtype=np.uint32)
header_group.attrs['NumPart_Total'] = np.array([0,N,0,0,0,0],dtype=np.uint32)
header_group.attrs['NumPart_Total_HighWord'] = np.array([0,0,0,0,0,0],dtype=np.uint32)
header_group.attrs['MassTable'] = np.array([0,0,0,0,0,0],dtype=np.float64)
header_group.attrs['Time'] = np.double(1.0)
header_group.attrs['Redshift'] = np.double(0.0)
header_group.attrs['Lbox'] = np.double(R_max*2.01)
header_group.attrs['NumFilesPerSnapshot'] = int(1)
header_group.attrs['Omega0'] = np.double(1.0)
header_group.attrs['OmegaLambda'] = np.double(0.0)
header_group.attrs['HubbleParam'] = np.double(1.0)
header_group.attrs['Flag_Sfr'] = int(0)
header_group.attrs['Flag_Cooling'] = int(0)
header_group.attrs['Flag_StellarAge'] = int(0)
header_group.attrs['Flag_Metals'] = int(0)
header_group.attrs['Flag_Feedback'] = int(0)
header_group.attrs['Flag_Entropy_ICs'] = int(0)
#Header created.
#Creating datasets for the particle data
particle_group = HDF5_snapshot.create_group("/PartType1")
if int(sys.argv[3]) == 0:
    HDF5datatype = 'float32'
    npdatatype = np.float32
if int(sys.argv[3]) == 1:
    HDF5datatype = 'double'
    npdatatype = np.float64
X = particle_group.create_dataset("Coordinates", (N,3),dtype=HDF5datatype)
V = particle_group.create_dataset("Velocities", (N,3),dtype=HDF5datatype)
IDs = particle_group.create_dataset("ParticleIDs", (N,),dtype='uint64')
M = particle_group.create_dataset("Masses", (N,),dtype=HDF5datatype)
#Saving the particle data
X[:,:] = ASCII_snapshot[:,0:3]
V[:,:] = ASCII_snapshot[:,3:6]
M[:] = ASCII_snapshot[:,6]
IDs[:] = np.arange(N, dtype=np.uint64)
HDF5_snapshot.close()
end = time.time()
print("..done in %fs. \n\n" % (end-start))

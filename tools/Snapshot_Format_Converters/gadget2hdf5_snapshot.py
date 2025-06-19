#!/usr/bin/env python3

#*******************************************************************************#
#  gadget2hdf5_snapshot.py - A gadget binary to hdf5 file converter for StePS   #
#     (STEreographically Projected cosmological Simulations) snapshots.         #
#    Copyright (C) 2025 Gabor Racz                                              #
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
sys.path.insert(0, '../../StePS_IC/src/')
from inputoutput import *

#Setting up the units of distance and time
UNIT_T=47.14829951063323 #Unit time in Gy
UNIT_V=20.738652969925447 #Unit velocity in km/s
UNIT_D=3.0856775814671917e24#=1Mpc Unit distance in cm

#Beginning of the script
if len(sys.argv) != 4:
    print("Error:")
    print("usage: ./gadget2hdf5_snapshot.py <input gadget snapshot> <output HDF5 snapshot> <output precision>\nExiting.")
    sys.exit(2)

if int(sys.argv[3]) != 0 and int(sys.argv[3]) != 1:
    print("Error:")
    print("Unkown output precision.\nExiting.")
    sys.exit(2)

print("Reading the input gadget file...")
start = time.time()
SnapParams = Load_params_from_gadget_snap(sys.argv[1])
N = SnapParams['Ntot']
Lbox = SnapParams['Lbox']
Om = SnapParams['OmegaM']
Ol = SnapParams['OmegaL']
H0 = SnapParams['HubbleParam']
z = SnapParams['Redshift']
partmass_table = SnapParams['PartMassTable']
print("\tN = %d" %N)
print("\tLbox = %f" %Lbox)
print("\tOmegaM = %f" %Om)
print("\tOmegaL = %f" %Ol)
print("\tH0 = %f" %H0)
print("\tz = %f" %z)
if partmass_table[1] == 0.0:
    #in gadget format, we assume constant mass for particle type 1 (we do not use this format for zoom-in simulations)
    #if this is zero, we calculate the mass from the cosmological parameters
    #Calculating the density from the cosmological Parameters
    rho_crit = H0**2/(8*np.pi)/UNIT_V/UNIT_V #in internal units
    rho_mean = Om*rho_crit
    partmass_table[1] = rho_mean*Lbox*Lbox*Lbox/N * 10.0 * (H0/100.0) # gadget uses 1e10 Msun/h units
print("\tPartMassTable = ", partmass_table)
Coordinates, Velocities, Masses, ParticleIDs = Load_snapshot(sys.argv[1],CONSTANT_RES=True,RETURN_VELOCITIES=True,RETURN_IDs=True,SILENT=False,DOUBLEPRECISION=False)
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
header_group.attrs['MassTable'] = np.array(partmass_table,dtype=np.float64)
header_group.attrs['Time'] = np.double(1.0/(z+1.0))
header_group.attrs['Redshift'] = np.double(z)
header_group.attrs['Lbox'] = np.double(Lbox)
header_group.attrs['BoxSize'] = np.double(Lbox)
header_group.attrs['NumFilesPerSnapshot'] = int(1)
header_group.attrs['Omega0'] = np.double(Om)
header_group.attrs['OmegaLambda'] = np.double(Ol)
header_group.attrs['HubbleParam'] = np.double(H0/100.0)
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
#Saving the particle data
X[:,:] = Coordinates
V[:,:] = Velocities
IDs[:] = ParticleIDs
HDF5_snapshot.close()
end = time.time()
print("..done in %fs. \n\n" % (end-start))

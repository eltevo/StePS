#!/usr/bin/env python3

#*******************************************************************************#
#  RotatingIC.py - Adding initial rotation to                                   #
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
from inputoutput import *

_VERSION="v0.0.0.1dev"
precision = 0 #32bit

#Beginning of the script
print("\nThis is RotatingIC.py version %s.\nCopyright (C) 2022 Gabor Racz\n\nThis script is used to add initial angular velocity to a StePS Initial Condition.\n" % _VERSION)
if len(sys.argv) != 7:
    print("Error: wrong number of arguments!")
    print("usage: ./RotateIC.py <input file> <output hdf5 file> <x component of the angular velocity> <y component of the angular velocity> <z component of the angular velocity> <Angular velocity in 1/Gy>\nExiting.")
    sys.exit(2)
start = time.time()
#Setting up the units of distance and time
UNIT_T=31556952.0*1e9 #=1Gy Unit time in s
UNIT_D=3.0856775814671917e19#=1Mpc Unit distance in km
lightspeed = 299792.458 #km/s

#normalizing the components of the angular velocity
Rx = np.double(sys.argv[3]) #unnormalized x component of the rotation vector
Ry = np.double(sys.argv[4]) #unnormalized y component of the rotation vector
Rz = np.double(sys.argv[5]) #unnormalized z component of the rotation vector
Mag = np.double(sys.argv[6]) #Length of the rotation vector
Norm = np.sqrt(Rx**2 + Ry**2 + Rz**2)
if precision == 0:
    R = Mag*np.array([Rx,Ry,Rz],dtype=np.float32)/Norm #Rotation vector as np array
elif precision == 1:
    R = Mag*np.array([Rx,Ry,Rz],dtype=np.double)/Norm #Rotation vector as np array

print("The angular velocity vector: [%.7f/Gy, %.7f/Gy, %.7f/Gy] = [%.7fkm/s/Mpc, %.7fkm/s/Mpc, %.7fkm/s/Mpc]\n" % (R[0],R[1],R[2],R[0]*UNIT_D/UNIT_T,R[1]*UNIT_D/UNIT_T, R[2]*UNIT_D/UNIT_T))

#setting the rotation vector to internal (StePS) units
R *= UNIT_D/UNIT_T #km/s/Mpc

#loading the input file:
print("Loading cosmological parameters...")
HDF5_snapshot = h5py.File(str(sys.argv[1]), "r")
Linearsize  = np.double(HDF5_snapshot['/Header'].attrs['BoxSize'])
Redshift    = np.double(HDF5_snapshot['/Header'].attrs['Redshift'])
OmegaM      = np.double(HDF5_snapshot['/Header'].attrs['Omega0'])
OmegaL      = np.double(HDF5_snapshot['/Header'].attrs['OmegaLambda'])
HubbleParam = np.double(HDF5_snapshot['/Header'].attrs['HubbleParam'])
print("\tLinear size\t%fMpc\n\tRedshift\t%f\n\tOmegaM\t%f\n\tOmegaL\t%f\n\tHubbleParam\t%f\n"%(Linearsize, Redshift, OmegaM, OmegaL, HubbleParam))
print("Loading particle data...")
Coordinates, Velocities, Masses = Load_snapshot(str(sys.argv[1]), RETURN_VELOCITIES=True)

#Calculating the velocity field of the rotation:
Rot_field = np.cross(R,Coordinates)
MaxVel = np.max(Rot_field)
print("Maximal rotational velocity: v_max = %e km/s (= %e c).\n" % (MaxVel, MaxVel/lightspeed))

#Adding the rotation field to the initial velocity field:
Velocities += Rot_field

#Saving the particle data
dataarray = np.vstack((np.hstack((Coordinates,Velocities)).T,Masses)).T
writeHDF5snapshot(dataarray, str(sys.argv[2]), Linearsize, Redshift, OmegaM, OmegaL, HubbleParam, precision)

end = time.time()
print("The IC making took %fs.\n" % (end-start))

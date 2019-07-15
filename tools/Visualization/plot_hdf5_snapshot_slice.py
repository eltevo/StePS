#!/usr/bin/env python3

#*******************************************************************************#
#  plot_hdf5_snapshot_slice.py - A snapshot plotting script for                 #
#     STEreographically Projected cosmological Simulations                      #
#    Copyright (C) 2019 Gabor Racz                                              #
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


import matplotlib
matplotlib.use('Qt5Agg')
import numpy as np
import h5py
import sys
import matplotlib.pyplot as plt
import time
# %matplotlib inline

#Beginning of the script
if len(sys.argv) != 4:
    print("Error:")
    print("usage: ./plot_hdf5_snapshot.py <input hdf5 snapshot file> <maximal plotted radius in Mpc> <0:4degree slice  1: 20Mpc thick slice>\nExiting.")
    sys.exit(2)

#Parameters of the plot
R_plot = np.float64(sys.argv[2]); #Mpc
plot_mode = np.int(sys.argv[3])

print("Reading the input HDF5 file...")
start = time.time()
HDF5_snapshot = h5py.File(str(sys.argv[1]), "r")
N_StePS = np.int(HDF5_snapshot['/Header'].attrs['NumPart_ThisFile'][1])
StePS_coordinates=np.zeros((N_StePS,4), dtype=np.float64)
redshift = np.double(HDF5_snapshot['/Header'].attrs['Redshift'])
StePS_coordinates[:,0:3] = HDF5_snapshot['/PartType1/Coordinates'] #reading the coordinates
StePS_coordinates[:,3] = HDF5_snapshot['/PartType1/Masses'] #reading the masses
end = time.time()
print("..done in %fs. \n\n" % (end-start))

print("Calculating map...")
start = time.time()
if plot_mode == 0:
    alpha = 4.0/180.0*np.pi #4 degree viewing angle in RAD
    N_StePS_slice=0
    for i in range(0,N_StePS):
        if StePS_coordinates[i,2]>-alpha*np.sqrt(StePS_coordinates[i,0]**2 + StePS_coordinates[i,1]**2) and StePS_coordinates[i,2]<alpha*np.sqrt(StePS_coordinates[i,0]**2 + StePS_coordinates[i,1]**2) and R_plot>np.sqrt(StePS_coordinates[i,0]**2 + StePS_coordinates[i,1]**2 +  StePS_coordinates[i,2]**2):
            N_StePS_slice+=1

    StePS_slice = np.zeros( (N_StePS_slice,3), dtype=np.float64)
    j=0
    for i in range(0,N_StePS):
        if StePS_coordinates[i,2]>-alpha*np.sqrt(StePS_coordinates[i,0]**2 + StePS_coordinates[i,1]**2) and StePS_coordinates[i,2]<alpha*np.sqrt(StePS_coordinates[i,0]**2 + StePS_coordinates[i,1]**2) and R_plot>np.sqrt(StePS_coordinates[i,0]**2 + StePS_coordinates[i,1]**2 + StePS_coordinates[i,2]**2):
            StePS_slice[j,0] = StePS_coordinates[i,0]
            StePS_slice[j,1] = StePS_coordinates[i,1]
            StePS_slice[j,2] = StePS_coordinates[i,3]
            j+=1

    end = time.time()
    print("..done in %fs. \n\n" % (end-start))

    plt.figure(figsize=(6,6))
    axes = plt.gca()
    axes.set_xlim([-R_plot,R_plot])
    axes.set_ylim([-R_plot,R_plot])
    plt.title("StePS Simulation, z=%.2f" % redshift)
    plt.scatter(StePS_slice[:,0], StePS_slice[:,1], marker='o', c='b', s=StePS_slice[:,2]/5000)
    plt.xlabel('x[Mpc]'); plt.ylabel('y[Mpc]'); plt.grid()
    plt.show()
else:
    alpha = 10 #20Mpc thick slice
    N_StePS_slice=0
    for i in range(0,N_StePS):
        if StePS_coordinates[i,2]>-alpha and StePS_coordinates[i,2]<alpha and R_plot>np.sqrt(StePS_coordinates[i,0]**2 + StePS_coordinates[i,1]**2 +  StePS_coordinates[i,2]**2):
            N_StePS_slice+=1

    StePS_slice = np.zeros( (N_StePS_slice,3), dtype=np.float64)
    j=0
    for i in range(0,N_StePS):
        if StePS_coordinates[i,2]>-alpha and StePS_coordinates[i,2]<alpha and R_plot>np.sqrt(StePS_coordinates[i,0]**2 + StePS_coordinates[i,1]**2 + StePS_coordinates[i,2]**2):
            StePS_slice[j,0] = StePS_coordinates[i,0]
            StePS_slice[j,1] = StePS_coordinates[i,1]
            StePS_slice[j,2] = StePS_coordinates[i,3]
            j+=1

    end = time.time()
    print("..done in %fs. \n\n" % (end-start))

    plt.figure(figsize=(6,6))
    axes = plt.gca()
    axes.set_xlim([-R_plot,R_plot])
    axes.set_ylim([-R_plot,R_plot])
    plt.title("StePS Simulation, z=%.2f" % redshift)
    plt.scatter(StePS_slice[:,0], StePS_slice[:,1], marker='o', c='b', s=StePS_slice[:,2]/5000)
    plt.xlabel('x[Mpc]'); plt.ylabel('y[Mpc]'); plt.grid()
    plt.show()

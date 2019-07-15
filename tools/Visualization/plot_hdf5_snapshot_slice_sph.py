#!/usr/bin/env python3

#*******************************************************************************#
#  plot_hdf5_snapshot_slice_sph.py - A snapshot plotting script for             #
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
import matplotlib.pyplot as plt
import numpy as np
import h5py
import sphviewer as sph
import sys
import time

#Beginning of the script
if len(sys.argv) != 3:
    print("Error:")
    print("usage: ./plot_hdf5_snapshot_slice_sph.py <input hdf5 snapshot file> <maximal plotted radius in Mpc>\nExiting.")
    sys.exit(2)

#Parameters of the plot
R_plot = np.float64(sys.argv[2]); #Mpc

print("Reading the input HDF5 file...")
start = time.time()
HDF5_snapshot = h5py.File(str(sys.argv[1]), "r")
N_StePS = np.int(HDF5_snapshot['/Header'].attrs['NumPart_ThisFile'][1])
redshift = np.double(HDF5_snapshot['/Header'].attrs['Redshift'])
print("z=%.4f" % redshift)
title = "z = %.2f" % redshift
StePS_coordinates=np.zeros((N_StePS,4), dtype=np.float64)
StePS_coordinates[:,0:3] = HDF5_snapshot['/PartType1/Coordinates'] #reading the coordinates
StePS_coordinates[:,3] = HDF5_snapshot['/PartType1/Masses'] #reading the masses
end = time.time()
print("..done in %fs. \n\n" % (end-start))

print("Calculating map...")
start = time.time()
alpha = 10 #20Mpc thick slice
N_StePS_slice=0
plot_R_limit = np.sqrt(2)*R_plot
for i in range(0,N_StePS):
    if StePS_coordinates[i,2]>-alpha and StePS_coordinates[i,2]<alpha and plot_R_limit>np.sqrt(StePS_coordinates[i,0]**2 + StePS_coordinates[i,1]**2 +  StePS_coordinates[i,2]**2):
        N_StePS_slice+=1

StePS_slice = np.zeros( (N_StePS_slice,4), dtype=np.float64)
j=0
for i in range(0,N_StePS):
    if StePS_coordinates[i,2]>-alpha and StePS_coordinates[i,2]<alpha and plot_R_limit>np.sqrt(StePS_coordinates[i,0]**2 + StePS_coordinates[i,1]**2 + StePS_coordinates[i,2]**2):
        StePS_slice[j,0] = StePS_coordinates[i,0]
        StePS_slice[j,1] = StePS_coordinates[i,1]
        StePS_slice[j,2] = StePS_coordinates[i,2]
        StePS_slice[j,3] = StePS_coordinates[i,3]
        j+=1

end = time.time()
print("..done in %fs. \n\n" % (end-start))
cmap='inferno'

extent=[-R_plot,R_plot,-R_plot,R_plot]
Particles = sph.Particles(StePS_slice[:,0:3].T,StePS_slice[:,3].T)
Scene = sph.Scene(Particles)
Scene.update_camera(r='infinity', extent=extent)
Render = sph.Render(Scene)
Render.set_logscale()
img = Render.get_image()
fig = plt.figure(1,figsize=(6,6))
ax1 = fig.add_subplot(111)
ax1.imshow(img, extent=extent, origin='lower', cmap=cmap)
ax1.set_xlabel('x[Mpc]', size=10)
ax1.set_ylabel('y[Mpc]', size=10)
plt.title(title)
plt.show()

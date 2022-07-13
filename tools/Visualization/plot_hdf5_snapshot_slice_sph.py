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
import yaml

#Beginning of the script
if len(sys.argv) != 3 and len(sys.argv) != 2:
    print("Error:")
    print("usage:\n\t./plot_hdf5_snapshot_slice_sph.py <input hdf5 snapshot file> <maximal plotted radius in Mpc>\n\tor:\n\t./plot_hdf5_snapshot_slice_sph.py <input yaml file>\nExiting.")
    sys.exit(2)

if len(sys.argv) == 3:
    #Parameters of the plot
    R_plot = np.float64(sys.argv[2]); #Mpc
    infilename      = str(sys.argv[1])
    outfilename     = "None"
    resolution      = "Auto"
    cmap            = 'inferno'
    figsize         = "Auto"
    title           = "Auto"
    slice_axis      = "Z"
else:
    #loading the parameters of the plot from yaml file
    document        = open(str(sys.argv[1]))
    Params          = yaml.safe_load(document)
    infilename      = str(Params["INFILENAME"])
    outfilename     = str(Params["OUTFILENAME"])
    resolution      = Params["RESOLUTION"]
    R_plot          = Params["R_PLOT"]
    alpha           = np.double(Params["SLICE_THICKNESS"])*0.5
    cmap            = Params["CMAP"]
    figsize         = Params["FIGSIZE"]
    title           = Params["TITLE"]
    slice_axis      = Params["SLICE_AXIS"]
    if slice_axis != "X" and slice_axis != "Y" and slice_axis != "Z":
        print("WARNING: Unknown Axis. Setting \"Z\".")
        slice_axis = "Z"
print("Reading the input HDF5 file...")
start = time.time()
HDF5_snapshot = h5py.File(infilename, "r")
N_StePS = int(HDF5_snapshot['/Header'].attrs['NumPart_ThisFile'][1])
redshift = np.double(HDF5_snapshot['/Header'].attrs['Redshift'])
print("z=%.4f" % redshift)
if title == "Auto":
    title = "z = %.2f" % redshift
StePS_coordinates=np.zeros((N_StePS,4), dtype=np.float64)
StePS_coordinates[:,0:3] = HDF5_snapshot['/PartType1/Coordinates'] #reading the coordinates
if slice_axis  == "Y":
    plot_indexes = [0,2,1] #x-z-Y
elif slice_axis  == "X":
    plot_indexes = [2,1,0] #z-y-X
else:
    plot_indexes = [0,1,2] #x-y-Z
StePS_coordinates[:,3] = HDF5_snapshot['/PartType1/Masses'] #reading the masses
end = time.time()
print("..done in %fs. \n\n" % (end-start))

print("Calculating map...")
start = time.time()
alpha = 10 #20Mpc thick slice
N_StePS_slice=0
plot_R_limit = np.sqrt(2)*R_plot
for i in range(0,N_StePS):
    if StePS_coordinates[i,plot_indexes[2]]>-alpha and StePS_coordinates[i,plot_indexes[2]]<alpha and plot_R_limit>np.sqrt(StePS_coordinates[i,0]**2 + StePS_coordinates[i,1]**2 +  StePS_coordinates[i,2]**2):
        N_StePS_slice+=1

StePS_slice = np.zeros( (N_StePS_slice,4), dtype=np.float64)
j=0
for i in range(0,N_StePS):
    if StePS_coordinates[i,plot_indexes[2]]>-alpha and StePS_coordinates[i,plot_indexes[2]]<alpha and plot_R_limit>np.sqrt(StePS_coordinates[i,0]**2 + StePS_coordinates[i,1]**2 + StePS_coordinates[i,2]**2):
        StePS_slice[j,plot_indexes[0]] = StePS_coordinates[i,0]
        StePS_slice[j,plot_indexes[1]] = StePS_coordinates[i,1]
        StePS_slice[j,plot_indexes[2]] = StePS_coordinates[i,2]
        StePS_slice[j,3] = StePS_coordinates[i,3]
        j+=1

end = time.time()
print("..done in %fs. \n\n" % (end-start))

extent=[-R_plot,R_plot,-R_plot,R_plot]
Particles = sph.Particles(StePS_slice[:,0:3],StePS_slice[:,3])
Scene = sph.Scene(Particles)
if resolution =='Auto':
    Scene.update_camera(r='infinity', extent=extent)
else:
    Scene.update_camera(r='infinity', extent=extent, xsize=resolution, ysize=resolution)
Render = sph.Render(Scene)
Render.set_logscale()
img = Render.get_image()
if figsize=="Auto":
    fig = plt.figure(1,figsize=(6,7))
else:
    fig = plt.figure(1,figsize=(figsize,7/6*figsize))
ax1 = fig.add_subplot(111)
ax1.imshow(img, extent=extent, origin='lower', cmap=cmap)
if slice_axis == "Z":
    ax1.set_xlabel('x[Mpc]', size=10)
    ax1.set_ylabel('y[Mpc]', size=10)
elif slice_axis == "Y":
    ax1.set_xlabel('x[Mpc]', size=10)
    ax1.set_ylabel('z[Mpc]', size=10)
else:
    ax1.set_xlabel('z[Mpc]', size=10)
    ax1.set_ylabel('y[Mpc]', size=10)
plt.title(title)
if outfilename!= "None":
    plt.savefig(outfilename)
plt.show()

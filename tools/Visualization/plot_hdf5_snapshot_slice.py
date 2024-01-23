#!/usr/bin/env python3

#*******************************************************************************#
#  plot_hdf5_snapshot_slice.py - A snapshot plotting script for                 #
#     STEreographically Projected cosmological Simulations                      #
#    Copyright (C) 2019-2022 Gabor Racz                                         #
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
import yaml
# %matplotlib inline

#Beginning of the script
if len(sys.argv)!=2 and len(sys.argv)!=4 and len(sys.argv)!=5 :
    print("Error:")
    print("usage:\n\t./plot_hdf5_snapshot_slice.py <input hdf5 snapshot file> <maximal plotted radius in Mpc> <0:4degree slice  1: 20Mpc thick slice> <(optional)quantity for plotting 0: particle coordinates; 1: velocity field>\nor\n\t./plot_hdf5_snapshot_slice.py <input yaml file>\nExiting.")
    sys.exit(2)

if len(sys.argv)!=2:
    #Parameters of the plot
    R_plot = np.float64(sys.argv[2]); #Mpc
    R_cutoff = "Auto"
    plot_mode = int(sys.argv[3])
    if plot_mode == 0:
        alpha = 4.0/180.0*np.pi #4 degree viewing angle in RAD
    elif plot_mode == 1:
        alpha = 10 #20Mpc thick slice
    else:
        print("Error: Unknown plot mode. Setting it to \"Slice\"\n")
        plot_mode = 1
        alpha = 10 #20Mpc thick slice
    if len(sys.argv) == 5:
        quantity = int(sys.argv[4])
    else:
        quantity = 0 #only plotting particle coordinates
    outfilename = "None"
    slice_axis = "Z"
    infilename = str(sys.argv[1])
    figsize=6
    title="Auto"
    color = 'blue'
    cmap='inferno'
    marker = 'o'
    slice_axis  == "Z"
    arrow_scale = 5e4 #ideal value for the example z=127 IC visualization
    logvelcols = True
else:
    #loading the parameters of the plot from yaml file
    document        = open(str(sys.argv[1]))
    Params          = yaml.safe_load(document)
    infilename      = str(Params["INFILENAME"])
    outfilename     = str(Params["OUTFILENAME"])
    if Params["QUANTITY"] == "Coordinates":
        quantity = 0
    elif Params["QUANTITY"] == "Velocities":
        quantity = 1
    else:
        print("Warning: Unkown quantity. Setting quantity variable to \"Coordinates\".\n")
        quantity = 0
    R_plot          = Params["R_PLOT"]
    R_cutoff        = Params["R_CUTOFF"]
    if Params["PLOTMODE"] == "Slice":
        plot_mode = 1
        alpha = np.double(Params["SLICE_THICKNESS"])*0.5
    elif Params["PLOTMODE"] == "Wedge":
        plot_mode = 0
        alpha = Params["WEDGE_ANGLE"]/180.0*np.pi #4 degree viewing angle in RAD
    else:
        print("Warning: Unkown plot mode. Setting it to \"Slice\".\n ")
        plot_mode = 1
        alpha = np.double(Params["SLICE_THICKNESS"])*0.5
    cmap            = Params["CMAP"]
    color           = Params["COLOR"]
    marker          = Params["MARKER"]
    figsize         = Params["FIGSIZE"]
    title           = Params["TITLE"]
    slice_axis      = Params["SLICE_AXIS"]
    arrow_scale     = Params["ARROW_SCALE"]
    cmap            = Params["CMAP"]
    logvelcols      = Params["LOG_VELOCITY_COLORS"]
    if slice_axis != "X" and slice_axis != "Y" and slice_axis != "Z":
        print("WARNING: Unknown Axis. Setting \"Z\".")
        slice_axis = "Z"

print("Reading the input HDF5 file...")
start = time.time()
HDF5_snapshot = h5py.File(infilename, "r")
N_StePS = int(HDF5_snapshot['/Header'].attrs['NumPart_ThisFile'][1])
StePS_coordinates=np.zeros((N_StePS,4), dtype=np.float64)
redshift = np.double(HDF5_snapshot['/Header'].attrs['Redshift'])
if title=="Auto":
    title = "StePS Simulation, z=%.2f" % redshift
StePS_coordinates[:,0:3] = HDF5_snapshot['/PartType1/Coordinates'] #reading the coordinates
if slice_axis  == "Y":
    plot_indexes = [0,2,1] #x-z-Y
elif slice_axis  == "X":
    plot_indexes = [2,1,0] #z-y-X
else:
    plot_indexes = [0,1,2] #x-y-Z
if R_cutoff == "Auto":
    R_cutoff = R_plot
StePS_coordinates[:,3] = HDF5_snapshot['/PartType1/Masses'] #reading the masses
if quantity == 1:
    StePS_velocities=np.zeros((N_StePS,3), dtype=np.float64)
    StePS_velocities[:,0:3] = HDF5_snapshot['/PartType1/Velocities'] #reading the coordinates

end = time.time()
print("..done in %fs. \n\n" % (end-start))

print("Calculating map...")
start = time.time()
if plot_mode == 0:
    N_StePS_slice=0
    for i in range(0,N_StePS):
        if StePS_coordinates[i,plot_indexes[2]]>-alpha*np.sqrt(StePS_coordinates[i,plot_indexes[0]]**2 + StePS_coordinates[i,plot_indexes[1]]**2) and StePS_coordinates[i,plot_indexes[2]]<alpha*np.sqrt(StePS_coordinates[i,plot_indexes[0]]**2 + StePS_coordinates[i,plot_indexes[1]]**2) and R_cutoff>np.sqrt(StePS_coordinates[i,0]**2 + StePS_coordinates[i,1]**2 +  StePS_coordinates[i,2]**2):
            N_StePS_slice+=1

    StePS_slice = np.zeros( (N_StePS_slice,3), dtype=np.float64)
    if quantity == 1:
        StePS_vel_slice = np.zeros( (N_StePS_slice,3), dtype=np.float64)
    j=0
    for i in range(0,N_StePS):
        if StePS_coordinates[i,plot_indexes[2]]>-alpha*np.sqrt(StePS_coordinates[i,plot_indexes[0]]**2 + StePS_coordinates[i,plot_indexes[1]]**2) and StePS_coordinates[i,plot_indexes[2]]<alpha*np.sqrt(StePS_coordinates[i,plot_indexes[0]]**2 + StePS_coordinates[i,plot_indexes[1]]**2) and R_cutoff>np.sqrt(StePS_coordinates[i,0]**2 + StePS_coordinates[i,1]**2 +  StePS_coordinates[i,2]**2):
            StePS_slice[j,0] = StePS_coordinates[i,plot_indexes[0]]
            StePS_slice[j,1] = StePS_coordinates[i,plot_indexes[1]]
            StePS_slice[j,2] = StePS_coordinates[i,3]
            if quantity == 1:
                StePS_vel_slice[j,0] = StePS_velocities[i,plot_indexes[0]]
                StePS_vel_slice[j,1] = StePS_velocities[i,plot_indexes[1]]
                StePS_vel_slice[j,2] = StePS_velocities[i,plot_indexes[2]]
            j+=1

    end = time.time()
    print("..done in %fs. \n\n" % (end-start))

    plt.figure(figsize=(figsize,figsize))
    axes = plt.gca()
    axes.set_xlim([-R_plot,R_plot])
    axes.set_ylim([-R_plot,R_plot])
    plt.title(title)
    if quantity == 0:
        plt.scatter(StePS_slice[:,0], StePS_slice[:,1], marker=marker, c=color, s=StePS_slice[:,2]/5000)
    elif quantity == 1:
        if arrow_scale>0:
            plt.quiver(StePS_slice[:,0], StePS_slice[:,1], StePS_vel_slice[:,0], StePS_vel_slice[:,1], np.sqrt(StePS_vel_slice[:,0]**2+StePS_vel_slice[:,1]**2 + StePS_vel_slice[:,2]**2), cmap=cmap, units='height', scale=arrow_scale, headwidth=0.5, headaxislength=1, headlength=1)
        else:
            #arrows are unit length and only show direction. Colors proportional to the magnitude
            length = np.sqrt(StePS_vel_slice[:,0]**2 + StePS_vel_slice[:,1]**2)
            if logvelcols:
                plt.quiver(StePS_slice[:,0], StePS_slice[:,1], StePS_vel_slice[:,0]/length, StePS_vel_slice[:,1]/length, np.log10(np.sqrt(StePS_vel_slice[:,0]**2+StePS_vel_slice[:,1]**2 + StePS_vel_slice[:,2]**2)), cmap=cmap, scale=-1.0*arrow_scale, units='height')
            else:
                plt.quiver(StePS_slice[:,0], StePS_slice[:,1], StePS_vel_slice[:,0]/length, StePS_vel_slice[:,1]/length, np.sqrt(StePS_vel_slice[:,0]**2+StePS_vel_slice[:,1]**2 + StePS_vel_slice[:,2]**2), cmap=cmap, scale=-1.0*arrow_scale, units='height')

    else:
        print("Error: Unkown quantity to plot. Exiting.\n")
        exit()
    if slice_axis == "Z":
        plt.xlabel('x[Mpc]', size=10)
        plt.ylabel('y[Mpc]', size=10)
    elif slice_axis == "Y":
        plt.xlabel('x[Mpc]', size=10)
        plt.ylabel('z[Mpc]', size=10)
    else:
        plt.xlabel('z[Mpc]', size=10)
        plt.ylabel('y[Mpc]', size=10)
    plt.grid()
    if outfilename!= "None":
        plt.savefig(outfilename)
    plt.show()
else:
    N_StePS_slice=0
    for i in range(0,N_StePS):
        if StePS_coordinates[i,plot_indexes[2]]>-alpha and StePS_coordinates[i,plot_indexes[2]]<alpha and R_cutoff>np.sqrt(StePS_coordinates[i,0]**2 + StePS_coordinates[i,1]**2 +  StePS_coordinates[i,2]**2):
            N_StePS_slice+=1

    StePS_slice = np.zeros( (N_StePS_slice,3), dtype=np.float64)
    if quantity == 1:
        StePS_vel_slice = np.zeros( (N_StePS_slice,3), dtype=np.float64)
    j=0
    for i in range(0,N_StePS):
        if StePS_coordinates[i,plot_indexes[2]]>-alpha and StePS_coordinates[i,plot_indexes[2]]<alpha and R_cutoff>np.sqrt(StePS_coordinates[i,0]**2 + StePS_coordinates[i,1]**2 + StePS_coordinates[i,2]**2):
            StePS_slice[j,0] = StePS_coordinates[i,plot_indexes[0]]
            StePS_slice[j,1] = StePS_coordinates[i,plot_indexes[1]]
            StePS_slice[j,2] = StePS_coordinates[i,3]
            if quantity == 1:
                StePS_vel_slice[j,0] = StePS_velocities[i,plot_indexes[0]]
                StePS_vel_slice[j,1] = StePS_velocities[i,plot_indexes[1]]
                StePS_vel_slice[j,2] = StePS_velocities[i,plot_indexes[2]]
            j+=1

    end = time.time()
    print("..done in %fs. \n\n" % (end-start))

    plt.figure(figsize=(figsize,figsize))
    axes = plt.gca()
    axes.set_xlim([-R_plot,R_plot])
    axes.set_ylim([-R_plot,R_plot])
    plt.title(title)
    if quantity == 0:
        plt.scatter(StePS_slice[:,0], StePS_slice[:,1], marker=marker, c=color, s=StePS_slice[:,2]/5000)
    elif quantity == 1:
        if arrow_scale>0:
            plt.quiver(StePS_slice[:,0], StePS_slice[:,1], StePS_vel_slice[:,0], StePS_vel_slice[:,1], np.sqrt(StePS_vel_slice[:,0]**2+StePS_vel_slice[:,1]**2 + StePS_vel_slice[:,2]**2), cmap=cmap, units='height', scale=arrow_scale, headwidth=0.5, headaxislength=1, headlength=1)
        else:
            #arrows are unit length and only show direction. Colors proportional to the magnitude
            length = np.sqrt(StePS_vel_slice[:,0]**2 + StePS_vel_slice[:,1]**2)
            if logvelcols:
                plt.quiver(StePS_slice[:,0], StePS_slice[:,1], StePS_vel_slice[:,0]/length, StePS_vel_slice[:,1]/length, np.log10(np.sqrt(StePS_vel_slice[:,0]**2+StePS_vel_slice[:,1]**2 + StePS_vel_slice[:,2]**2)), cmap=cmap, scale=-1.0*arrow_scale, units='height')
            else:
                plt.quiver(StePS_slice[:,0], StePS_slice[:,1], StePS_vel_slice[:,0]/length, StePS_vel_slice[:,1]/length, np.sqrt(StePS_vel_slice[:,0]**2+StePS_vel_slice[:,1]**2 + StePS_vel_slice[:,2]**2), cmap=cmap, scale=-1.0*arrow_scale, units='height')

    else:
        print("Error: Unkown quantity to plot. Exiting.\n")
        exit()
    if slice_axis == "Z":
        plt.xlabel('x[Mpc]', size=10)
        plt.ylabel('y[Mpc]', size=10)
    elif slice_axis == "Y":
        plt.xlabel('x[Mpc]', size=10)
        plt.ylabel('z[Mpc]', size=10)
    else:
        plt.xlabel('z[Mpc]', size=10)
        plt.ylabel('y[Mpc]', size=10)
    plt.grid()
    if outfilename!= "None":
        plt.savefig(outfilename)
    plt.show()

#!/usr/bin/env python3

#*******************************************************************************#
#  CreateAnimationFromSimulation.py - a script for animating                    #
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


import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
import h5py
import sphviewer as sph
import time
import sys
import yaml

_VERSION = "v0.0.0.2dev"
_YEAR    = "2022"

def full_frame(width=None, height=None):
	import matplotlib as mpl
	mpl.rcParams['savefig.pad_inches'] = 0
	figsize = None if width is None else (width, height)
	fig = plt.figure(figsize=figsize)
	ax = plt.axes([0,0,1,1], frameon=False)
	ax.get_xaxis().set_visible(False)
	ax.get_yaxis().set_visible(False)
	plt.autoscale(tight=True)


#Begininng of the script
print("----------------------------------------------------------------------------------------------\nCreateAnimationFromSimulation.py %s\n (A script for animating StePS simulations.)\n\n Gabor Racz, %s\n\tJet Propulsion Laboratory, California Institute of Technology | Pasadena, CA, USA\n----------------------------------------------------------------------------------------------\n\n" % (_VERSION,_YEAR))


if len(sys.argv) != 2:
        print("usage: ./CreateAnimationFromSimulation.py <input yaml file>")
        sys.exit(2)
Plot_start = time.time()
props = dict(boxstyle='round', facecolor='white', alpha=0.65)

#loading the parameters of the plot
document = open(str(sys.argv[1]))
Params   = yaml.safe_load(document)
infilebasename  = Params["INFILE_BASENAME"]
ICfile          = Params["ICFILE"]
outbasefilename = Params["OUTFILE_BASENAME"]
center          = Params["CENTER"]
R_plot          = Params["R_PLOT"]
R_camera        = Params["R_CAMERA"]
sizex           = Params["RESOLUTION"][0]
sizey           = Params["RESOLUTION"][1]
N_snapshots     = Params["NSNAPSHOTS"]
N_interp        = Params["NINTERPOLATION"]
Frames          = N_interp*(N_snapshots)+1
Rotation        = Params["ROT_IN_DEGREES"]
delta_phi       = -1.0*Rotation/np.float64(Frames)
theta           = Params["INIT_THETA_IN_DEGREES"]
phi             = 0.0
cmap            = Params["CMAP"]
comoving        = Params["COMOVING"]

#Reading the first (last) snapshot
k=N_snapshots-1
infile=infilebasename + '_%04d' %k + '.hdf5'
print("Reading the (last) %s input HDF5 file..." % infile)
start = time.time()
HDF5_snapshot = h5py.File(infile, "r")
Npart = int(HDF5_snapshot['/Header'].attrs['NumPart_ThisFile'][1]) #total number of particles
redshift_1 = np.double(HDF5_snapshot['/Header'].attrs['Redshift'])
time_1 = np.double(HDF5_snapshot['/Header'].attrs['Time']) #in Gy
Coordinates_0=np.zeros((Npart,4), dtype=np.float64)
Coordinates_1=np.zeros((Npart,4), dtype=np.float64)
Plotted_coordinates=np.zeros((Npart,4), dtype=np.float64)
Coordinates_1[:,0:3] = HDF5_snapshot['/PartType1/Coordinates'] #reading the coordinates
Coordinates_1[:,3] = HDF5_snapshot['/PartType1/Masses'] #reading the masses
Plotted_coordinates[:,3] = Coordinates_1[:,3]
end = time.time()
print("..done in %fs. \n" % (end-start))

l=Frames-1 #the actual number of the frame
#making the video
for k in reversed(range(0,N_snapshots-1)):
	Coordinates_0 = Coordinates_1
	time_0        = time_1
	redshift_0    = redshift_1
	#reading the input snapshot
	infile=infilebasename + '_%04d' %k + '.hdf5'
	print("Reading the %s input HDF5 file..." % infile)
	start = time.time()
	Plot_start= time.time()
	HDF5_snapshot = h5py.File(infile, "r")
	Npart = int(HDF5_snapshot['/Header'].attrs['NumPart_ThisFile'][1])
	Coordinates_1=np.zeros((Npart,4), dtype=np.float64)
	Coordinates_1[:,0:3] = HDF5_snapshot['/PartType1/Coordinates'] #reading the coordinates
	Coordinates_1[:,3] = HDF5_snapshot['/PartType1/Masses']
	redshift_1 = np.double(HDF5_snapshot['/Header'].attrs['Redshift'])
	time_1 = np.double(HDF5_snapshot['/Header'].attrs['Time']) #in Gy
	end = time.time()
	print("..done in %fs. \n" % (end-start))
	#doing the interpolation...
	for h in range(0,N_interp):
		beta = np.double(h)/np.double(N_interp)
		Plotted_coordinates[:,0:3] = ((1.0-beta)*Coordinates_0[:,0:3]+beta*Coordinates_1[:,0:3]) #Coordinates
		Plotted_time = ((1-beta)*time_0 + beta*time_1) #time
		Plotted_redshift = ((1-beta)*redshift_0 + beta*redshift_1) #redshift
		if comoving:
			textstr = "Redshift = %.4f" % (Plotted_redshift)
		else:
			textstr = "Age = %.3f Gy" % (Plotted_time)
		#calculating the map
		print("Calculating the interploated %i. map..." % h)
		start = time.time()
		indexes = np.sqrt(Plotted_coordinates[:,0]**2 + Plotted_coordinates[:,1]**2 + Plotted_coordinates[:,2]**2) <= R_plot
		Nplot = len(Plotted_coordinates[indexes])
		print("Number of plotted particles: %i" % Nplot)
		end = time.time()
		print("..done in %fs. \n" % (end-start))
		Particles = sph.Particles(Plotted_coordinates[indexes,0:3],Plotted_coordinates[indexes,3].T)
		Scene = sph.Scene(Particles)
		#Rendering
		print("Rendering Frame %i..." % l)
		outfilename = outbasefilename + '%06g' %l + '.png'
		start = time.time()
		Scene.update_camera(x=center[0],y=center[1],z=center[2],r=R_camera, xsize=sizex, ysize=sizey, t=theta, p=phi)
		Render = sph.Render(Scene)
		Render.set_logscale()
		img = Render.get_image()
		if (l==Frames-1):
			vmax = np.max(img)
			vmin = np.min(img)
		end = time.time()
		print("..done in %fs. \n" % (end-start))
		full_frame(width=sizex/100, height=sizey/100)
		plt.text(25, 0.95*sizey, textstr, fontsize=16, verticalalignment='top', bbox=props)
		plt.imshow(img, origin='lower', cmap=cmap, vmax=vmax, vmin=vmin)
		plt.savefig(outfilename,format='png',bbox_inches = 'tight', facecolor='black')
		phi += delta_phi
		l -= 1
#interpolating between the first snapshot and IC
Coordinates_0 = Coordinates_1
time_0 = time_1
redshift_0 = redshift_1
#reading the IC
print("Reading the %s input HDF5 IC file..." % ICfile)
start = time.time()
Plot_start= time.time()
HDF5_snapshot = h5py.File(ICfile, "r")
Npart = int(HDF5_snapshot['/Header'].attrs['NumPart_ThisFile'][1])
Coordinates_1=np.zeros((Npart,4), dtype=np.float64)
Coordinates_1[:,0:3] = HDF5_snapshot['/PartType1/Coordinates'] #reading the coordinates
Coordinates_1[:,3] = HDF5_snapshot['/PartType1/Masses']
redshift_1 = np.double(HDF5_snapshot['/Header'].attrs['Redshift'])
time_1 = np.double(HDF5_snapshot['/Header'].attrs['Time']) #in Gy #this is not correct!
end = time.time()
print("..done in %fs. \n" % (end-start))
#doing the interpolation...
for h in range(0,N_interp+1):
	beta = np.double(h)/np.double(N_interp)
	Plotted_coordinates[:,0:3] = ((1.0-beta)*Coordinates_0[:,0:3]+beta*Coordinates_1[:,0:3]) #Coordinates
	Plotted_time = ((1-beta)*time_0 + beta*time_1) #time
	Plotted_redshift = ((1-beta)*redshift_0 + beta*redshift_1) #redshift
	if comoving:
		textstr = "Redshift = %.4f" % (Plotted_redshift)
	else:
		textstr = "Age = %.3f Gy" % (Plotted_time)
	#calculating the map
	print("Calculating the interploated %i. map..." % h)
	start = time.time()
	indexes = np.sqrt(Plotted_coordinates[:,0]**2 + Plotted_coordinates[:,1]**2 + Plotted_coordinates[:,2]**2) <= R_plot
	Nplot = len(Plotted_coordinates[indexes])
	print("Number of plotted particles: %i" % Nplot)
	end = time.time()
	print("..done in %fs. \n" % (end-start))
	Particles = sph.Particles(Plotted_coordinates[indexes,0:3],Plotted_coordinates[indexes,3].T)
	Scene = sph.Scene(Particles)
	#Rendering
	print("Rendering Frame %i..." % l)
	outfilename = outbasefilename + '%06g' %l + '.png'
	start = time.time()
	Scene.update_camera(x=center[0],y=center[1],z=center[2],r=R_camera, xsize=sizex, ysize=sizey, t=theta, p=phi)
	Render = sph.Render(Scene)
	Render.set_logscale()
	img = Render.get_image()
	end = time.time()
	print("..done in %fs. \n" % (end-start))
	full_frame(width=sizex/100, height=sizey/100)
	plt.text(25, 0.95*sizey, textstr, fontsize=16, verticalalignment='top', bbox=props)
	plt.imshow(img, origin='lower', cmap=cmap, vmax=vmax, vmin=vmin)
	plt.savefig(outfilename,format='png',bbox_inches = 'tight', facecolor='black')
	phi += delta_phi
	l -= 1
Plot_end=time.time()
print("Total run time: %f s (=%f h)" % (Plot_end-Plot_start, (Plot_end-Plot_start)/3600))

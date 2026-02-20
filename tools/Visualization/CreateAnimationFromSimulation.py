#!/usr/bin/env python3

#*******************************************************************************#
#  CreateAnimationFromSimulation.py - a script for animating                    #
#     STEreographically Projected cosmological Simulations                      #
#    Copyright (C) 2022-2026 Gabor Racz                                         #
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


from statistics import mode
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

_VERSION = "v0.0.2.0"
_YEAR    = "2022-2026"

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
print("----------------------------------------------------------------------------------------------\nCreateAnimationFromSimulation.py %s\n (A script for animating StePS simulations.)\n\n Gabor Racz, %s\n\tUniversity of Helsinki | Helsinki, Finland\n\tJet Propulsion Laboratory, California Institute of Technology | Pasadena, CA, USA\n----------------------------------------------------------------------------------------------\n\n" % (_VERSION,_YEAR))


if len(sys.argv) != 2:
        print("usage: ./CreateAnimationFromSimulation.py <input yaml file>")
        sys.exit(2)
Plot_start = time.time()
props = dict(boxstyle='round', facecolor='white', alpha=0.65)

#loading the parameters of the plot
document = open(str(sys.argv[1]))
Params   = yaml.safe_load(document)
infilebasename  = Params["INFILE_BASENAME"]
topology        = Params["TOPOLOGY"]
if (topology != "SPHERICAL" and topology != "CYLINDRICAL"):
	print("Error: unknown topology: %s. Only 'SPHERICAL' and 'CYLINDRICAL' are supported." % topology)
	sys.exit(2)
visualisation_mode = Params["MODE"]
if (visualisation_mode != "2D" and visualisation_mode != "3D"):
	print("Error: unknown visualisation mode: %s. Only '2D' and '3D' are supported." % visualisation_mode)
	sys.exit(2)
ICfile          = Params["ICFILE"]
outbasefilename = Params["OUTFILE_BASENAME"]
center          = Params["CENTER"]
R_plot          = Params["R_PLOT"]
slice_thickness = Params["SLICE_THICKNESS"]
R_camera        = Params["R_CAMERA"]
sizex           = Params["RESOLUTION"][0]
sizey           = Params["RESOLUTION"][1]
if visualisation_mode == "2D":
	extent=[-R_camera,R_camera,-R_camera*sizey/sizex,R_camera*sizey/sizex]
N_snapshots     = Params["NSNAPSHOTS"]
N_interp        = Params["NINTERPOLATION"]
Frames          = N_interp*(N_snapshots)+1
Rotation        = Params["ROT_IN_DEGREES"]
delta_phi       = -1.0*Rotation/np.float64(Frames)
theta           = Params["INIT_THETA_IN_DEGREES"]
phi             = 0.0
cmap            = Params["CMAP"]
if "VMAX" in Params:
	vmax        = Params["VMAX"]
else:
	vmax        = np.nan
if "VMIN" in Params:
	vmin        = Params["VMIN"]
else:
	vmin        = np.nan
comoving        = Params["COMOVING"]
if "STATIC" in Params:
	static      = Params["STATIC"]
else:
	static      = False

#Printing the parameters
print("Input parameters:")
print("Input file basename: %s" % infilebasename)
print("Initial conditions file: %s" % ICfile)
print("Output file basename: %s" % outbasefilename)
print("Topology: %s" % topology)
print("Visualisation mode: %s" % visualisation_mode)
print("Center: %s" % str(center))
print("Plotting radius: %f Mpc" % R_plot)
if topology == "CYLINDRICAL":
	print("Slice thickness: %f Mpc" % slice_thickness)
print("Camera distance: %f Mpc" % R_camera)
print("Resolution: %ix%i" % (sizex,sizey))
print("Number of snapshots: %i" % N_snapshots)
print("Number of interpolated frames between snapshots: %i" % N_interp)
print("Total number of frames: %i" % Frames)
print("Rotation: %f degrees" % Rotation)
print("Colormap: %s" % cmap)
if not np.isnan(vmax):
	print("Color scale vmax: %e" % vmax)
else:
	print("Color scale vmax: auto")
if not np.isnan(vmin):
	print("Color scale vmin: %e" % vmin)
print("Comoving coordinates: %s" % str(comoving))
print("Static image: %s" % str(static))
print("\n")

if static:
	#Reading the first (last) snapshot
	k=N_snapshots-1
	infile=infilebasename + '_%04d' %k + '.hdf5'
	print("Reading the (last) %s input HDF5 file..." % infile)
	start = time.time()
	HDF5_snapshot = h5py.File(infile, "r")
	Npart = int(HDF5_snapshot['/Header'].attrs['NumPart_ThisFile'][1]) #total number of particles
	redshift = np.double(HDF5_snapshot['/Header'].attrs['Redshift'])
	time_1 = np.double(HDF5_snapshot['/Header'].attrs['Time']) #in Gy
	Plotted_coordinates=np.zeros((Npart,4), dtype=np.float64)
	Plotted_coordinates[:,0:3] = HDF5_snapshot['/PartType1/Coordinates'] #reading the coordinates
	Plotted_coordinates[:,3] = HDF5_snapshot['/PartType1/Masses'] #reading the masses
	end = time.time()
	print("..done in %fs. \n" % (end-start))
	if comoving:
		if redshift < 0.0:
			redshift = 0.0
			print("Warning: negative redshift value read from the snapshot. Setting it to zero.")
		textstr = "Redshift = %.4f" % (redshift)
	else:
		textstr = "Age = %.3f Gy" % (time_1)
	#calculating the map
	print("Calculating the map...")
	if topology == "SPHERICAL":
		indexes = np.sqrt(Plotted_coordinates[:,0]**2 + Plotted_coordinates[:,1]**2 + Plotted_coordinates[:,2]**2) <= R_plot
	elif topology == "CYLINDRICAL":
		indexes = np.logical_and(np.sqrt(Plotted_coordinates[:,0]**2 + Plotted_coordinates[:,1]**2) <= R_plot, np.abs(Plotted_coordinates[:,2]-center[2]) <= slice_thickness/2.0)
	Nplot = len(Plotted_coordinates[indexes])
	print("Considered particles: %i" % Plotted_coordinates.shape[0])
	print("Number of plotted particles: %i" % Nplot)
	end = time.time()
	print("..done in %fs. \n" % (end-start))
	#Rendering
	for k in range(0,Frames):
		print("Rendering Frame %i..." % k)
		try:
			outfilename = outbasefilename + '%06g' %k + '.png'
			start = time.time()
			Particles = sph.Particles(Plotted_coordinates[indexes,0:3],Plotted_coordinates[indexes,3].T)
			Scene = sph.Scene(Particles)
			if visualisation_mode == "2D":
				Scene.update_camera(r='infinity', extent=extent, xsize=sizex, ysize=sizey)
			elif visualisation_mode == "3D":
				Scene.update_camera(x=center[0],y=center[1],z=center[2],r=R_camera, xsize=sizex, ysize=sizey, t=theta, p=phi)
			#the pysphviewer rendering can fail for some frames, especially if the number of particles is large. In this case, we skip the frame and continue with the next one.
			Render = sph.Render(Scene)
			Render.set_logscale()
			img = Render.get_image()
			if (k==0):
				if np.isnan(vmax):
					vmax = np.max(img)
				if np.isnan(vmin):
					vmin = np.min(img)
			print("Color scale: vmin=%e, vmax=%e" % (vmin,vmax))
			end = time.time()
			print("..done in %fs. \n" % (end-start))
			full_frame(width=sizex/100, height=sizey/100)
			plt.text(25, 0.95*sizey, textstr, fontsize=16, verticalalignment='top', bbox=props)
			plt.imshow(img, origin='lower', cmap=cmap, vmax=vmax, vmin=vmin)
			plt.savefig(outfilename,format='png',bbox_inches = 'tight', facecolor='black')
			plt.close()
		except:
			print("Error: rendering frame %i failed. Skipping this frame." % k)
		phi += delta_phi
else:
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
	if N_interp<=1:
		#no interpolation, just plotting the first snapshot
		Plotted_coordinates[:,0:3] = Coordinates_1[:,0:3] #Coordinates
		Plotted_time = time_1 #time
		Plotted_redshift = redshift_1 #redshift
		if comoving:
			if redshift_1 < 0.0:
				redshift_1 = 0.0
				print("Warning: negative redshift value read from the snapshot. Setting it to zero.")
			textstr = "Redshift = %.4f" % (redshift_1)
		else:
			textstr = "Age = %.3f Gy" % (Plotted_time)
		start = time.time()
		print("Calculating the first map...")
		if topology == "SPHERICAL":
			indexes = np.sqrt(Plotted_coordinates[:,0]**2 + Plotted_coordinates[:,1]**2 + Plotted_coordinates[:,2]**2) <= R_plot
		elif topology == "CYLINDRICAL":
			indexes = np.logical_and(np.sqrt(Plotted_coordinates[:,0]**2 + Plotted_coordinates[:,1]**2) <= R_plot, np.abs(Plotted_coordinates[:,2]-center[2]) <= slice_thickness/2.0)
		Nplot = len(Plotted_coordinates[indexes])
		print("Number of plotted particles: %i" % Nplot)
		end = time.time()
		print("..done in %fs. \n" % (end-start))
		#Rendering
		print("Rendering Frame %i..." % (Frames-1))
		try:
			outfilename = outbasefilename + '%06g' %(Frames) + '.png'
			start = time.time()
			Particles = sph.Particles(Plotted_coordinates[indexes,0:3],Plotted_coordinates[indexes,3].T)
			Scene = sph.Scene(Particles)
			if visualisation_mode == "2D":
				Scene.update_camera(r='infinity', extent=extent, xsize=sizex, ysize=sizey)
			elif visualisation_mode == "3D":
				Scene.update_camera(x=center[0],y=center[1],z=center[2],r=R_camera, xsize=sizex, ysize=sizey, t=theta, p=phi)
			Render = sph.Render(Scene)
			Render.set_logscale()
			img = Render.get_image()
			if np.isnan(vmax):
				vmax = np.max(img)
			if np.isnan(vmin):
				vmin = np.min(img)
			print("Color scale: vmin=%e, vmax=%e" % (vmin,vmax))
			end = time.time()
			print("..done in %fs. \n" % (end-start))
			full_frame(width=sizex/100, height=sizey/100)
			plt.text(25, 0.95*sizey, textstr, fontsize=16, verticalalignment='top', bbox=props)
			plt.imshow(img, origin='lower', cmap=cmap, vmax=vmax, vmin=vmin)
			plt.savefig(outfilename,format='png',bbox_inches = 'tight', facecolor='black')
			plt.close()
		except:
			print("Error: rendering frame %i failed. Skipping this frame." % (Frames-1))
		phi += delta_phi

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
		if N_interp<=1:
			#no interpolation, just plotting the snapshots
			Plotted_coordinates[:,0:3] = Coordinates_1[:,0:3] #Coordinates
			Plotted_time = time_1 #time
			Plotted_redshift = redshift_1 #redshift
			if comoving:
				textstr = "Redshift = %.4f" % (Plotted_redshift)
			else:
				textstr = "Age = %.3f Gy" % (Plotted_time)
			#calculating the map
			print("Calculating the %i. map..." % l)
			start = time.time()
			if topology == "SPHERICAL":
				indexes = np.sqrt(Plotted_coordinates[:,0]**2 + Plotted_coordinates[:,1]**2 + Plotted_coordinates[:,2]**2) <= R_plot
			elif topology == "CYLINDRICAL":
				indexes = np.logical_and(np.sqrt(Plotted_coordinates[:,0]**2 + Plotted_coordinates[:,1]**2) <= R_plot, np.abs(Plotted_coordinates[:,2]-center[2]) <= slice_thickness/2.0)
			Nplot = len(Plotted_coordinates[indexes])
			print("Number of plotted particles: %i" % Nplot)
			end = time.time()
			print("..done in %fs. \n" % (end-start))
			#Rendering
			print("Rendering Frame %i..." % l)
			try:
				outfilename = outbasefilename + '%06g' %l + '.png'
				start = time.time()
				Particles = sph.Particles(Plotted_coordinates[indexes,0:3],Plotted_coordinates[indexes,3].T)
				Scene = sph.Scene(Particles)
				if visualisation_mode == "2D":
					Scene.update_camera(r='infinity', extent=extent, xsize=sizex, ysize=sizey)
				elif visualisation_mode == "3D":
					Scene.update_camera(x=center[0],y=center[1],z=center[2],r=R_camera, xsize=sizex, ysize=sizey, t=theta, p=phi)
				Render = sph.Render(Scene)
				Render.set_logscale()
				img = Render.get_image()
				if (l==Frames-1):
					if np.isnan(vmax):
						vmax = np.max(img)
					if np.isnan(vmin):
						vmin = np.min(img)
					print("Color scale: vmin=%e, vmax=%e" % (vmin,vmax))
				end = time.time()
				print("..done in %fs. \n" % (end-start))
				full_frame(width=sizex/100, height=sizey/100)
				plt.text(25, 0.95*sizey, textstr, fontsize=16, verticalalignment='top', bbox=props)
				plt.imshow(img, origin='lower', cmap=cmap, vmax=vmax, vmin=vmin)
				plt.savefig(outfilename,format='png',bbox_inches = 'tight', facecolor='black')
				plt.close()
			except:
				print("Error: rendering frame %i failed. Skipping this frame." % l)
			phi += delta_phi
			l -= 1
		else:
			#doing the interpolation...
			for h in range(0,N_interp):
				beta = np.double(h)/np.double(N_interp)
				Plotted_coordinates[:,0:3] = ((1.0-beta)*Coordinates_0[:,0:3]+beta*Coordinates_1[:,0:3]) #Coordinates
				Plotted_time = ((1-beta)*time_0 + beta*time_1) #time
				Plotted_redshift = ((1-beta)*redshift_0 + beta*redshift_1) #redshift
				if comoving:
					if Plotted_redshift < 0.0:
						Plotted_redshift = 0.0
						print("Warning: negative redshift value read from the snapshot. Setting it to zero.")
					textstr = "Redshift = %.4f" % (Plotted_redshift)
				else:
					textstr = "Age = %.3f Gy" % (Plotted_time)
				#calculating the map
				print("Calculating the interploated %i. map..." % h)
				start = time.time()
				if topology == "SPHERICAL":
					indexes = np.sqrt(Plotted_coordinates[:,0]**2 + Plotted_coordinates[:,1]**2 + Plotted_coordinates[:,2]**2) <= R_plot
				elif topology == "CYLINDRICAL":
					indexes = np.logical_and(np.sqrt(Plotted_coordinates[:,0]**2 + Plotted_coordinates[:,1]**2) <= R_plot, np.abs(Plotted_coordinates[:,2]-center[2]) <= slice_thickness/2.0)
				Nplot = len(Plotted_coordinates[indexes])
				print("Number of plotted particles: %i" % Nplot)
				end = time.time()
				print("..done in %fs. \n" % (end-start))
				#Rendering
				print("Rendering Frame %i..." % l)
				try:
					outfilename = outbasefilename + '%06g' %l + '.png'
					start = time.time()
					Particles = sph.Particles(Plotted_coordinates[indexes,0:3],Plotted_coordinates[indexes,3].T)
					Scene = sph.Scene(Particles)
					if visualisation_mode == "2D":
						Scene.update_camera(r='infinity', extent=extent, xsize=sizex, ysize=sizey)
					elif visualisation_mode == "3D":
						Scene.update_camera(x=center[0],y=center[1],z=center[2],r=R_camera, xsize=sizex, ysize=sizey, t=theta, p=phi)
					Render = sph.Render(Scene)
					Render.set_logscale()
					img = Render.get_image()
					if (l==Frames-1):
						if np.isnan(vmax):
							vmax = np.max(img)
						if np.isnan(vmin):
							vmin = np.min(img)
						print("Color scale: vmin=%e, vmax=%e" % (vmin,vmax))
					end = time.time()
					print("..done in %fs. \n" % (end-start))
					full_frame(width=sizex/100, height=sizey/100)
					plt.text(25, 0.95*sizey, textstr, fontsize=16, verticalalignment='top', bbox=props)
					plt.imshow(img, origin='lower', cmap=cmap, vmax=vmax, vmin=vmin)
					plt.savefig(outfilename,format='png',bbox_inches = 'tight', facecolor='black')
					plt.close()
				except:
					print("Error: rendering frame %i failed. Skipping this frame." % l)
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
	if N_interp<=1:
		#no interpolation, just plotting the snapshots
		Plotted_coordinates[:,0:3] = Coordinates_1[:,0:3] #Coordinates
		Plotted_time = time_1 #time
		Plotted_redshift = redshift_1 #redshift
		if comoving:
			textstr = "Redshift = %.4f" % (Plotted_redshift)
		else:
			textstr = "Age = %.3f Gy" % (Plotted_time)
		#calculating the map
		print("Calculating the %i. map..." % l)
		start = time.time()
		if topology == "SPHERICAL":
			indexes = np.sqrt(Plotted_coordinates[:,0]**2 + Plotted_coordinates[:,1]**2 + Plotted_coordinates[:,2]**2) <= R_plot
		elif topology == "CYLINDRICAL":
			indexes = np.logical_and(np.sqrt(Plotted_coordinates[:,0]**2 + Plotted_coordinates[:,1]**2) <= R_plot, np.abs(Plotted_coordinates[:,2]-center[2]) <= slice_thickness/2.0)
		Nplot = len(Plotted_coordinates[indexes])
		print("Number of plotted particles: %i" % Nplot)
		end = time.time()
		print("..done in %fs. \n" % (end-start))
		#Rendering
		print("Rendering Frame %i..." % l)
		try:
			outfilename = outbasefilename + '%06g' %l + '.png'
			start = time.time()
			Particles = sph.Particles(Plotted_coordinates[indexes,0:3],Plotted_coordinates[indexes,3].T)
			Scene = sph.Scene(Particles)
			if visualisation_mode == "2D":
				Scene.update_camera(r='infinity', extent=extent, xsize=sizex, ysize=sizey)
			elif visualisation_mode == "3D":
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
			plt.close()
		except:
			print("Error: rendering frame %i failed. Skipping this frame." % l)
		phi += delta_phi
		l -= 1
	else:
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
			if topology == "SPHERICAL":
				indexes = np.sqrt(Plotted_coordinates[:,0]**2 + Plotted_coordinates[:,1]**2 + Plotted_coordinates[:,2]**2) <= R_plot
			elif topology == "CYLINDRICAL":
				indexes = np.logical_and(np.sqrt(Plotted_coordinates[:,0]**2 + Plotted_coordinates[:,1]**2) <= R_plot, np.abs(Plotted_coordinates[:,2]-center[2]) <= slice_thickness/2.0)
			Nplot = len(Plotted_coordinates[indexes])
			print("Number of plotted particles: %i" % Nplot)
			end = time.time()
			print("..done in %fs. \n" % (end-start))
			#Rendering
			print("Rendering Frame %i..." % l)
			try:
				start = time.time()
				outfilename = outbasefilename + '%06g' %l + '.png'
				Particles = sph.Particles(Plotted_coordinates[indexes,0:3],Plotted_coordinates[indexes,3].T)
				Scene = sph.Scene(Particles)
				if visualisation_mode == "2D":
					Scene.update_camera(r='infinity', extent=extent, xsize=sizex, ysize=sizey)
				elif visualisation_mode == "3D":
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
				plt.close()
			except:
				print("Error: rendering frame %i failed. Skipping this frame." % l)
			phi += delta_phi
			l -= 1
	Plot_end=time.time()
	print("Total run time: %f s (=%f h)" % (Plot_end-Plot_start, (Plot_end-Plot_start)/3600))

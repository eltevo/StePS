#!/usr/bin/env python3

#*******************************************************************************#
#  plot_halo_from_catalog.py - a halo visualization script for                  #
#      STEreographically Projected cosmological Simulations                     #
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

__VERSION__ = "0.0.2"
__AUTHOR__ = "Gabor Racz"
__YEAR__ = "2025"

# Importing libraries
import matplotlib
matplotlib.use('Qt5Agg')
import numpy as np
import h5py
import sys
import matplotlib.pyplot as plt
import StePS_HF
from StePS_HF.halo_params import calculate_halo_shape
#from StePS_HF.halo_params import get_total_energy
import time

# Welcome message
print("This is plot_halo_from_catalog.py version %sr" % (__VERSION__))
print("    A halo visualization script for STEreographically Projected cosmological Simulations")
print("    Copyright (C) %s %s" % (__YEAR__, __AUTHOR__))
if len(sys.argv) != 3:
        print("Error: wrong number of arguments.")
        print("usage: ./plot_halo_from_catalog.py <hdf5 halo catalog> <halo ID>\nExiting.")
        sys.exit(1)
halocatalog_filename = sys.argv[1]
haloID = int(sys.argv[2])

# Loading the halo catalog
halocatalog = h5py.File(halocatalog_filename, 'r')
if not "/Particles" in halocatalog:
    print("Error: no particle data group is saved in the halo catalog file %s\nExiting." % halocatalog_filename)
    sys.exit(2)
Nhalos = halocatalog["/Header"].attrs["Nhalos"]
print("Number of halos in the catalog: %i" % Nhalos)
if haloID < 0 or haloID >= Nhalos:
    print("Error: invalid halo ID %i, valid range is [0,%i]\nExiting." % (haloID, Nhalos-1))
    sys.exit(3)
if not "/Particles/Halo_%i" % haloID in halocatalog:
    print("Error: no particle data is saved for halo ID %i in the halo catalog file %s\nExiting." % (haloID, halocatalog_filename))
    sys.exit(4)

# Default values
SaveRadius = 1.0 # in Rvir units
Mpcunits = "Mpc" # Mpc or h^-1 Mpc
kpcunits = "kpc" # kpc or h^-1 kpc
Msolunits = "Msol" # Msol or Msol/h

# Printing the header information
print("\nHalo catalog file: %s\nHeader information:\n-------------------" % halocatalog_filename)
for key in halocatalog["/Header"].attrs.keys():
    print("%s: %s" % (key, str(halocatalog["/Header"].attrs[key])))
    if key=="ParticleSaveRadius":
        SaveRadius = halocatalog["/Header"].attrs[key]
print("\nHalo information for halo #%i:\n---------------------------------" % haloID)
# Loading the halo particle data
hindep_units = halocatalog["/Header"].attrs["HindependentUnits"] # true if the coordinates are in h^-1 Mpc, false if in Mpc
halo_part_group = halocatalog["/Particles/Halo_%i" % haloID]
part_coords = halo_part_group["Coordinates"][:]
part_masses = halo_part_group["Masses"][:]
#part_vels = halo_part_group["Velocities"][:]
# Loading the halo properties
Npart = int(halocatalog["/Halos/Npart"][haloID])
Mvir = halocatalog["/Halos/Mvir"][haloID]
print("Number of particles in halo # %i: %i" % (haloID, Npart))
print("Number of particles loaded: %i" % part_coords.shape[0])
halo_coords = halocatalog["/Halos/Coordinates"][haloID]
if hindep_units:
    Mpcunits = "Mpc / h"
    kpcunits = "kpc / h"
    Msolunits = "Msol / h"
print("Halo center coordinates: ", halo_coords, Mpcunits)
print("Halo virial mass: %.4e %s" % (halocatalog["/Halos/Mvir"][haloID], Msolunits))
print("Available halo properties:", end="")
Radius = {}
for key in halocatalog["/Halos"].keys():
    if key != "Coordinates" and key != "Npart":
        print(" %s" % (key), end="")
    # saving the radius variables for later use
    if key.startswith("R"):
        Radius[key] = halocatalog["/Halos/%s" % key][haloID] / 1000.0 # converting from kpc to Mpc
print("\n")
#printing all halo properties
#for key in halocatalog["/Halos"].keys():
#    print("%s: %s" % (key, str(halocatalog["/Halos/%s" % key][haloID])))
#print("\n")
halocatalog.close()

#calculating the halo shape parameters
print("Calculating the halo shape parameters...")
start_time = time.time()
#selecting particles within Rvir
mask = np.linalg.norm(part_coords, axis=1) <= Radius["Rvir"]
shape_params = calculate_halo_shape(part_coords[mask], part_masses[mask], np.array([0.0,0.0,0.0])) # saved particle coordinates are relative to the halo center
end_time = time.time()
print("...done. Time elapsed: %.2e seconds." % (end_time - start_time))
print("\nHalo shape parameters:")
for key in shape_params.keys():
    print("%s: %s" % (key, str(shape_params[key])))
eigen_vector_a = shape_params["eigenvectors"][:,0]*Radius["Rvir"]
eigen_vector_b = shape_params["eigenvectors"][:,1]*shape_params["axis_lengths"]["b"]/shape_params["axis_lengths"]["a"] * Radius["Rvir"]
eigen_vector_c = shape_params["eigenvectors"][:,2]*shape_params["axis_lengths"]["c"]/shape_params["axis_lengths"]["a"] * Radius["Rvir"]
print("Eigenvector a length: ", np.linalg.norm(eigen_vector_a))
print("Eigenvector b length: ", np.linalg.norm(eigen_vector_b))
print("Eigenvector c length: ", np.linalg.norm(eigen_vector_c))
print("")

##calculating halo energy, if wanted
#print("Calculating the halo energy with octree method...")
#start_time = time.time()
#Etot, Ekin, Epot_octree = get_total_energy(part_coords[mask],part_vels[mask], part_masses[mask]*1e11, 1.0, 0.0, np.full(part_masses[mask].shape, 0.01), boundary="STEPS", boxsize=0.0, method="octree")
#end_time = time.time()
#print("...done. Time elapsed: %.2e seconds." % (end_time - start_time))
#print("Calculating the halo energy with direct method...")
#start_time = time.time()
#Etot, Ekin, Epot_direct = get_total_energy(part_coords[mask], part_vels[mask], part_masses[mask]*1e11, 1.0, 0.0, np.full(part_masses[mask].shape, 0.01), boundary="STEPS", boxsize=0.0, method="direct")
#end_time = time.time()
#print("...done. Time elapsed: %.2e seconds." % (end_time - start_time))
#print("\nHalo energy values:")
#print("Halo total kinetic energy: %.4e (Msol *  (km/s)^2 )" % Ekin)
#print("Halo total potential energy (octree): %.4e (Msol *  (km/s)^2 )" % Epot_octree)
#print("Halo total potential energy (direct): %.4e (Msol *  (km/s)^2 )" % Epot_direct)
#print("Difference between octree and direct potential energy: %.4f percent" % (100.0 * np.abs(Epot_octree - Epot_direct) / np.abs(Epot_direct)))
#print("Energy ratio (2*Ekin/Epot): %.4f" % (2.0*Ekin/Epot_direct))
#print("")

# plotting the 3D distribution of the halo particles
print("Plotting the 3D distribution of the halo particles...")
plt.figure(figsize=(8, 8))
ax = plt.axes(projection='3d')
ax.scatter3D(part_coords[:,0], part_coords[:,1], part_coords[:,2], s=1, c='black', alpha=0.25, marker='o', label='Halo particles')
if hindep_units:
    ax.set_xlabel(r'$x [h^{-1}$ Mpc$]$')
    ax.set_ylabel(r'$y [h^{-1}$ Mpc$]$')
    ax.set_zlabel(r'$z [h^{-1}$ Mpc$]$')
else:
    ax.set_xlabel(r'$x [$Mpc$]$')
    ax.set_ylabel(r'$y [$Mpc$]$')
    ax.set_zlabel(r'$z [$Mpc$]$')
# plotting the halo radius if available
Rvir = Radius["Rvir"]
# plotting a sphere with radius Rvir around the halo center
u = np.linspace(0, 2 * np.pi, 30)
v = np.linspace(0, np.pi, 30)
x = Rvir * np.outer(np.cos(u), np.sin(v))
y = Rvir * np.outer(np.sin(u), np.sin(v))
z = Rvir * np.outer(np.ones(np.size(u)), np.cos(v)) 
ax.plot_wireframe(x, y, z, color='blue', alpha=0.05, label=r'$R_{vir}=%0.2f %s $' % (Rvir*1000.0, kpcunits))
if "R200c" in Radius and Radius["R200c"] > 0.0:
    R200 = Radius["R200c"]
    x = R200 * np.outer(np.cos(u), np.sin(v))
    y = R200 * np.outer(np.sin(u), np.sin(v))
    z = R200 * np.outer(np.ones(np.size(u)), np.cos(v)) 
    ax.plot_wireframe(x, y, z, color='red', alpha=0.05, label=r'$R_{200c}=%0.2f %s $' % (R200*1000.0, kpcunits))
if "R500c" in Radius and Radius["R500c"] > 0.0:
    R500 = Radius["R500c"]
    x = R500 * np.outer(np.cos(u), np.sin(v))
    y = R500 * np.outer(np.sin(u), np.sin(v))
    z = R500 * np.outer(np.ones(np.size(u)), np.cos(v)) 
    ax.plot_wireframe(x, y, z, color='green', alpha=0.05, label=r'$R_{500c}=%0.2f %s $' % (R500*1000.0, kpcunits))
# plotting the eigenvectors
ax.quiver(0, 0, 0, eigen_vector_a[0], eigen_vector_a[1], eigen_vector_a[2], color='red', length=np.linalg.norm(eigen_vector_a), normalize=True, label='Eigenvector a (scaled to Rvir)')
ax.quiver(0, 0, 0, eigen_vector_b[0], eigen_vector_b[1], eigen_vector_b[2], color='green', length=np.linalg.norm(eigen_vector_b), normalize=True, label='Eigenvector b (scaled to Rvir)')
ax.quiver(0, 0, 0, eigen_vector_c[0], eigen_vector_c[1], eigen_vector_c[2], color='blue', length=np.linalg.norm(eigen_vector_c), normalize=True, label='Eigenvector c (scaled to Rvir)')
# setting the axes limits
ax.set_xlim3d(-Rvir*SaveRadius,+Rvir*SaveRadius)
ax.set_ylim3d(-Rvir*SaveRadius,+Rvir*SaveRadius)
ax.set_zlim3d(-Rvir*SaveRadius,+Rvir*SaveRadius)
ax.set_box_aspect([1,1,1])  # equal aspect ratio
ax.legend()
ax.set_title("Halo #%i, Mvir=%.4e %s" % (haloID, Mvir, Msolunits))
plt.show()

print("...done.")
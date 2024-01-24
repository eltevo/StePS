#!/usr/bin/env python3

#*******************************************************************************#
#  StePS_HF.py - a Halo Finder script for                                       #
#      STEreographically Projected cosmological Simulations                     #
#    Copyright (C) 2024 Gabor Racz                                              #
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

#*******************************************************************#
# Base (planned) algorithm:                                         #
#   1. Load the snapshot                                            #
#   2. Reconstruct the density field with Voronoi tessellation      #
#       -> every particle will have a local estimated density       #
#          (rho_i = m_i/V_{i, voronoi})                             #
#   3. Select the largest (unflagged) density particle              #
#      (this will be the center of the halo)                        #
#   4. We grow a sphere around this centre, and stop when the       #
#      mean density within the sphere falls below a desired         #
#      critical value.                                              #
#       -> summing particle masses                                  #
#       -> once the limit reached, flag particles in the halo       #
#   5. GOTO 3. until we run out of halos                            #
#   6. Save the catalog                                             #
#*******************************************************************#

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..','StePS_IC','src'))
from os.path import exists
import time
import yaml
import numpy as np
from scipy.spatial import Voronoi, ConvexHull
import astropy.units as u
from astropy.cosmology import LambdaCDM, wCDM, w0waCDM, z_at_value
from inputoutput import *

_VERSION="v0.0.0.1"
_YEAR="2024"

#defining functions
def voronoi_volumes(points, SILENT=False):
    if SILENT==False:
        v_start = time.time()
        print("Calculating voronoi tessellation...")
    v = Voronoi(points)
    if SILENT==False:
        v_end = time.time()
        print("...done in %.2f s." % (v_end-v_start))
    if SILENT==False:
        v_start = time.time()
        print("Calculating voronoi volumes...")
    vol = np.zeros(v.npoints)
    for i, reg_num in enumerate(v.point_region):
        indices = v.regions[reg_num]
        if -1 in indices: # some regions can be opened
            vol[i] = np.inf
        else:
            vol[i] = ConvexHull(v.vertices[indices]).volume #NameError: name 'ConvexHull' is not defined
    if SILENT==False:
        v_end = time.time()
        print("...done in %.2f s." % (v_end-v_start))
    return vol

#defining classes
class StePS_Particle_Catalog:
    '''
    A class for storing particle information.
    Stored information: ID, coordinate components, velocity components, mass, Parent halo ID, density
    '''
    def __init__(self, FILENAME):
        print("Creating a new particle catalog by loading %s\n" % FILENAME)
        self.sourcefile = FILENAME
        self.Coordinates, self.Velocities, self.Masses, self.IDs = Load_snapshot(FILENAME,CONSTANT_RES=False,RETURN_VELOCITIES=True,RETURN_IDs=True,SILENT=True)
        self.HaloParentIDs = -1*np.ones(len(self.Masses),dtype=np.int64)
        self.Density= np.zeros(len(self.Masses),dtype=np.double)
        self.Npart = len(self.Masses)
    def printlines(self,lines):
        print("ID\t(X      Y      Z) [Mpc/h]\t\t(Vx      Vy      Vz) [km/s]\t\tM[1e11Msol/h]\tDensity[rho/rho_crit]\tParentID\n----------------------------------------------------------------------------------------------------------------------------------------")
        for line in lines:
            print("%-1i\t(%+-10.2f %+-10.2f %+-7.2f)\t\t(%+-10.2f %+-10.2f %+-7.2f)\t\t%-8.3g\t%-8.3g\t\t%i" % (self.IDs[line], self.Coordinates[line,0], self.Coordinates[line,1], self.Coordinates[line,2], self.Velocities[line,0], self.Velocities[line,1], self.Velocities[line,2], self.Masses[line], self.Density[line], self.HaloParentIDs[line]))

#class StePS_Particle_Catalog:
#    '''
#    A class for storing halo catalogs
#    '''

#Beginning of the script
print("\n+-----------------------------------------------------------------------------------------------+\n|StePS_HF.py %s\t\t\t\t\t\t\t\t\t\t|\n| (STEreographically Projected cosmological Simulations Halo Finder)\t\t\t\t|\n+-----------------------------------------------------------------------------------------------+\n| Copyright (C) %s Gabor Racz\t\t\t\t\t\t\t\t\t|\n|\tJet Propulsion Laboratory, California Institute of Technology | Pasadena, CA, USA\t|\n|\tDepartment of Physics of Complex Systems, Eotvos Lorand University | Budapest, Hungary  |\n|\tDepartment of Physics & Astronomy, Johns Hopkins University | Baltimore, MD, USA\t|\n+-----------------------------------------------------------------------------------------------+\n"%(_VERSION, _YEAR))
if len(sys.argv) != 2:
    print("Error: missing yaml file!")
    print("usage: ./StePS_HF.py <input yaml file>\nExiting.")
    sys.exit(2)
start = time.time()
print("Reading the %s paramfile...\n" % str(sys.argv[1]))
document = open(str(sys.argv[1]))
Params = yaml.safe_load(document)
print("Cosmological Parameters:\n------------------------\nOmega_m:\t%f\t(Ommh2=%f; Omch2=%f)\nOmega_lambda:\t%f\nOmega_k:\t%f\nOmega_b:\t%f\t(Ombh2=%f)\nH0:\t\t%f km/s/Mpc\nDark energy model:\t%s" % (Params['OMEGAM'], Params['OMEGAM'] * (Params['H0']/100.0)**2, (Params['OMEGAM'] - Params['OMEGAB']) * (Params['H0']/100.0)**2, Params['OMEGAL'], 1.0-Params['OMEGAM']-Params['OMEGAL'], Params['OMEGAB'], (Params['OMEGAB']) * (Params['H0']/100.0)**2, Params['H0'], Params['DARKENERGYMODEL']))
if Params['DARKENERGYMODEL'] == 'Lambda':
    print("\n")
elif Params['DARKENERGYMODEL'] == 'w0':
    print("w = %f\n" % Params['DARKENERGYPARAMS'][0])
elif Params['DARKENERGYMODEL'] == 'CPL':
    print("w0 = %f\nwa = %f\n" % (Params['DARKENERGYPARAMS'][0], Params['DARKENERGYPARAMS'][1]))
else:
    print("Error: unkown dark energy parametrization!\nExiting.\n")
    sys.exit(2)

#loading the input particle snapshot
p = StePS_Particle_Catalog(Params['INFILE'])
#calculating voronoi voronoi volumes:
p.Density = voronoi_volumes(p.Coordinates)
p.printlines([1,100,1000,10000,100000,1000000,1700000])

end = time.time()
print("SO halo finding finished under %fs.\n" % (end-start))

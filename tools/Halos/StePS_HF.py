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
#   2. Reconstruct the density field with                           #
#      Voronoi tessellation / 10th nearest neighbor method          #
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
from scipy.spatial import Voronoi, ConvexHull, KDTree
import astropy.units as u
from astropy.cosmology import LambdaCDM, wCDM, w0waCDM, z_at_value
from inputoutput import *

_VERSION="v0.0.0.3"
_YEAR="2024"

#defining functions
def voronoi_volumes(points, SILENT=False):
    if SILENT==False:
        v_start = time.time()
        print("Calculating Voronoi tessellation...")
    v = Voronoi(points)
    if SILENT==False:
        v_end = time.time()
        print("...done in %.2f s." % (v_end-v_start))
    if SILENT==False:
        v_start = time.time()
        print("Calculating Voronoi volumes...")
    vol = np.zeros(v.npoints)
    for i, reg_num in enumerate(v.point_region):
        indices = v.regions[reg_num]
        if -1 in indices: # some regions can be opened
            vol[i] = np.inf
        else:
            vol[i] = ConvexHull(v.vertices[indices]).volume #NameError: name 'ConvexHull' is not defined
    if SILENT==False:
        v_end = time.time()
        print("...done in %.2f s.\n" % (v_end-v_start))
    return vol

def get_center_of_mass(r, m):
    return np.sum((r.T*m).T,axis=0)/np.sum(m)


def calculate_halo_params(p, idx, halo_particleindexes, HaloID, massdefnames, massdefdenstable, npartmin, centermode):
    Masslist = np.zeros(len(massdefdenstable), dtype=np.float32)
    Rlist = np.zeros(len(massdefdenstable), dtype=np.float32)
    #sorting particles by distance from the central particle
    distances = np.sqrt(np.sum(np.power((p.Coordinates[halo_particleindexes]-p.Coordinates[idx]),2),axis=1)) # calculating Euclidian particle distances from the center
    sorted_idx = distances.argsort() # sorting
    if centermode == "CENTRALPARTICLE":
        # using the particle with the highest estimated density as center
        Center = p.Coordinates[idx]
    elif centermode == "CENTEROFMASSNPARTMIN":
        # using the innermost npartmin particles to calculate the center of mass
        Center = get_center_of_mass(p.Coordinates[halo_particleindexes][sorted_idx][:npartmin], p.Masses[halo_particleindexes][sorted_idx][:npartmin])
        distances = np.sqrt(np.sum(np.power((p.Coordinates[halo_particleindexes]-Center),2),axis=1)) #recalculating distances due to the new center
        sorted_idx = distances.argsort()
    else:
        raise Exception("Error: unkown CENTERMODE parameter %s." % (centermode))
    #calculating enclosed density
    M_enc = np.cumsum(p.Masses[halo_particleindexes][sorted_idx])
    rho_enc= M_enc / (4.0*np.pi/3.0*np.power(distances[sorted_idx],3)) # enclosed mass / enclosed volume
    print(rho_enc)
    print(massdefdenstable)
    #calculating the parameters for each mass definitions
    R = np.zeros(len(massdefdenstable))
    M = np.zeros(len(massdefdenstable))
    V = np.zeros((3,len(massdefdenstable)))
    Vrms = np.zeros(len(massdefdenstable))
    Vmax = np.zeros(len(massdefdenstable))
    J = np.zeros((3,len(massdefdenstable)))
    for i in range(0,len(massdefdenstable)):
        radius_idx = massdefdenstable[i] <= rho_enc
        max_radi_idx = np.where(radius_idx == False)[0][0] # apply cut when the density first fall below the limit
        if i==0:
            p.HaloParentIDs[halo_particleindexes][sorted_idx][:max_radi_idx] = HaloID #Setting the HaloParentIDs for the primary mass definition
        R[i] = distances[sorted_idx][:max_radi_idx][-1] #Radii
        M[i] = M_enc[sorted_idx][:max_radi_idx][-1] #Mass
        V[i] = get_center_of_mass(p.Velocities[halo_particleindexes][sorted_idx][:max_radi_idx], p.Masses[halo_particleindexes][sorted_idx][:max_radi_idx]) #Velocity; the formula for calculating the mean velocity is the same as for the COM
        Vrms[i] = np.sqrt(np.sum(np.power(p.Velocities[halo_particleindexes][sorted_idx][:max_radi_idx] - V[i],2))/len(p.Velocities[halo_particleindexes][sorted_idx][:max_radi_idx])) # root mean square velocity
        Vmax[i] = np.max(np.sqrt(np.sum(np.power(p.Velocities[halo_particleindexes][sorted_idx][:max_radi_idx] - V[i],2), axis=1))) # maximal velocity
        #J[i] = #angular momentum
    print("Radii:\t",R)
    print("Mass:\t",M)
    print("Mean Velocity:\t",V)
    print("Vrms:\t",Vrms)
    print("Vmax:\t",Vmax)

    return returndict #[Center, velocitymassdef1, , massdef1, massdef2, ..., rdef1, rdef2, ...]

#defining classes
class StePS_Particle_Catalog:
    '''
    A class for storing particle information.
    Stored information: ID, coordinate components, velocity components, mass, Parent halo ID, density
    '''
    def __init__(self, FILENAME, D_UNITS, V_UNITS, M_UNITS):
        print("Creating a new particle catalog by loading %s\n" % FILENAME)
        self.sourcefile = FILENAME
        self.Coordinates, self.Velocities, self.Masses, self.IDs = Load_snapshot(FILENAME,CONSTANT_RES=False,RETURN_VELOCITIES=True,RETURN_IDs=True,SILENT=True)
        self.HaloParentIDs = -1*np.ones(len(self.Masses),dtype=np.int64)
        self.Density= np.zeros(len(self.Masses),dtype=np.double)
        self.Npart = len(self.Masses)
        #converting the input data to StePS units (Mpc, km/s, 1e11Msol)
        self.Coordinates *= D_UNITS
        self.Velocities *= V_UNITS
        self.Masses *= (M_UNITS / 1e11)
    def printlines(self,lines):
        print("ID\t(X      Y      Z) [Mpc/h]\t\t(Vx      Vy      Vz) [km/s]\t\tM[1e11Msol/h]\tDensity[rho/rho_crit]\tParentID\n----------------------------------------------------------------------------------------------------------------------------------------")
        for line in lines:
            print("%-1i\t(%+-10.2f %+-10.2f %+-7.2f)\t\t(%+-10.2f %+-10.2f %+-7.2f)\t\t%-8.3g\t%-8.3g\t\t%i" % (self.IDs[line], self.Coordinates[line,0], self.Coordinates[line,1], self.Coordinates[line,2], self.Velocities[line,0], self.Velocities[line,1], self.Velocities[line,2], self.Masses[line], self.Density[line], self.HaloParentIDs[line]))

class StePS_Halo_Catalog:
    '''
    A class for storing halo catalogs
    '''
    def __init__(self, H0, Om, Ol, DE_Model, Mdef):
        self.Header = {
            "H0": H0,
            "OmegaM": Om,
            "OmegaL": Ol,
            "DE_Model": DE_Model,
            "Nhalos": 0,
            "MassDefinitions": Mdef,
        }
        self.IDs = np.array([],dtype=np.uint32)
        self.Npart = np.array([],dtype=np.uint32)
        self.Coordinates = np.array([None,3],dtype=np.float32)
        self.Velocities = np.array([None,3],dtype=np.float32)
        self.Masses = np.array([None,len(Mdef)],dtype=np.float32)
    def add_halo(id,npart,coordinates,velocities,masses):
        return



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
print("Cosmological Parameters:\n------------------------\nOmega_m:\t\t%f\t(Ommh2=%f; Omch2=%f)\nOmega_lambda:\t\t%f\nOmega_k:\t\t%f\nOmega_b:\t\t%f\t(Ombh2=%f)\nH0:\t\t\t%f km/s/Mpc\nDark energy model:\t%s" % (Params['OMEGAM'], Params['OMEGAM'] * (Params['H0']/100.0)**2, (Params['OMEGAM'] - Params['OMEGAB']) * (Params['H0']/100.0)**2, Params['OMEGAL'], 1.0-Params['OMEGAM']-Params['OMEGAL'], Params['OMEGAB'], (Params['OMEGAB']) * (Params['H0']/100.0)**2, Params['H0'], Params['DARKENERGYMODEL']))
if Params['DARKENERGYMODEL'] == 'Lambda':
    print("\n")
elif Params['DARKENERGYMODEL'] == 'w0':
    print("w = %f\n" % Params['DARKENERGYPARAMS'][0])
elif Params['DARKENERGYMODEL'] == 'CPL':
    print("w0 = %f\nwa = %f\n" % (Params['DARKENERGYPARAMS'][0], Params['DARKENERGYPARAMS'][1]))
else:
    print("Error: unkown dark energy parametrization!\nExiting.\n")
    sys.exit(2)
#Setting up the units of distance and time (the usual StePS internal units)
UNIT_T=47.14829951063323 #Unit time in Gy
UNIT_V=20.738652969925447 #Unit velocity in km/s
UNIT_D=3.0856775814671917e24 #=1Mpc Unit distance in cm

#calculating relevant cosmological quantities
rho_c = 3*Params['H0']**2/(8*np.pi)/UNIT_V/UNIT_V #critical density in internal units
rho_b = rho_c * Params['OMEGAM'] #background density in internal units

print("Snapshot Parameters:\n------------------------\nRadius:\t\t\t%.6gMpc/h\nDistance units:\t\t%.2gMpc/h\nVelocity units:\t\t%.2gkm/s\nMass units:\t\t%.2gMsol/h\n" % (np.double(Params['RSIM']),np.double(Params['UNIT_D_IN_MPC']), np.double(Params['UNIT_V_IN_KMPS']), np.double(Params['UNIT_M_IN_MSOL'])))

print("Halo Finder Parameters:\n---------------------------\nInitial Density Estimation:\t\t%s\nSearch radius alpha parameter:\t\t%.2f\nNumber of KDTree worker threads:\t%i\nMinimal particle number:\t\t%i\nHalo center mode:\t\t\t%s\nMass definitions:" %(Params["INITIAL_DENSITY_MODE"],np.double(Params["SEARCH_RADIUS_ALPHA"]),int(Params["KDWORKERS"]), int(Params["NPARTMIN"]), Params["CENTERMODE"] ))
Nmassdef = len(Params["MASSDEF"]) #total number of mass definitions.
for i in range(0,Nmassdef):
    print("\t\t- %s" % Params["MASSDEF"][i])
# generating density levels for the different mass definitions:
massdefdenstable = np.zeros(Nmassdef,dtype=np.float32) # array containing the mass definition density levels
for i in range(0,Nmassdef):
    massdefdenstable[i] = np.float32(Params["MASSDEF"][i][:-1])
    if Params["MASSDEF"][i][-1] == "c" or Params["MASSDEF"][i][-1] == "C":
        massdefdenstable[i] *= rho_c
    elif Params["MASSDEF"][i][-1] == "b" or Params["MASSDEF"][i][-1] == "B":
        massdefdenstable[i] *= rho_b
    else:
        raise Exception(f"Error: Unrecognized mass definition %s." % Params["MASSDEF"][i])
npartmin = int(Params["NPARTMIN"])
if Params["INITIAL_DENSITY_MODE"] != "Voronoi" and Params["INITIAL_DENSITY_MODE"] != "10th neighbor":
    raise Exception(f"Error: Unknown initial density estimation method %s." % Params["INITIAL_DENSITY_MODE"])
else:
    DensMode = Params["INITIAL_DENSITY_MODE"]
if Params["CENTERMODE"] != "CENTRALPARTICLE" and Params["CENTERMODE"] != "CENTEROFMASSNPARTMIN":
    raise Exception("Error: unkown CENTERMODE parameter %s." % Params["CENTERMODE"])
alpha = np.double(Params["SEARCH_RADIUS_ALPHA"])
kdworkers = int(Params["KDWORKERS"])


# Loading the input particle snapshot
p = StePS_Particle_Catalog(Params['INFILE'], np.double(Params['UNIT_D_IN_MPC']), np.double(Params['UNIT_V_IN_KMPS']), np.double(Params['UNIT_M_IN_MSOL']))

# Building KDTree for a quick nearest-neighbor lookup
tree = KDTree(p.Coordinates,leafsize=10, compact_nodes=True, balanced_tree=True, boxsize=None)

# Density reconstruction
if DensMode == "Voronoi":
    # Calculating voronoi volumes:
    p.Density = p.Masses/voronoi_volumes(p.Coordinates)/rho_c
elif DensMode == "10th neighbor":
    # searching for the distance of the 10th nearest neighbor for all particle
    print("Density reconstruction with 10th nearest neighbor method...")
    kd_start = time.time()
    for i in range(0,p.Npart):
        d,idx = tree.query(p.Coordinates[i,:], k=10, workers=kdworkers)
        p.Density[i] = (p.Masses[i]/(d[9]**3))
    p.Density *= 10/(4.0*np.pi/3.0)/rho_c #assuming 10 particles with p.Masses[i] with in r<d spherical volume.
    kd_end = time.time()
    print("...done in %.2f s.\n" % (kd_end-kd_start))

#p.printlines([0,1000,1000000,1700000])
p.printlines(np.arange(0,p.Npart))

# Identifying halos using Spherical Overdensity (SO) method
halos = StePS_Halo_Catalog(np.double(Params["H0"]), np.double(Params["OMEGAM"]), np.double(Params["OMEGAL"]), Params["DARKENERGYMODEL"], Params["MASSDEF"])
halo_ID = 0
while True:
    #selecting the largest density particle with parentID=-1
    idx = np.argmax(p.Density[p.HaloParentIDs == -1])
    #Query the kd-tree for nearest neighbors.
    search_radius = alpha*np.cbrt(p.Masses[idx]/rho_b) #In StePS simulations, the particles are more density packed at the center. The typical particle separation is proportional to the cubic root of the particle mass.
    halo_particleindexes = tree.query_ball_point(p.Coordinates[idx], search_radius, p=2.0, eps=0, workers=kdworkers, return_sorted=False)
    if len(halo_particleindexes) >= npartmin:
        print("Search radius for halo #%i: %.2fMpc/h" % (halo_ID, search_radius))
        print("Number of particles in the search radius of halo #%i:" % (halo_ID),len(halo_particleindexes))
        calculate_halo_params(p, idx, halo_particleindexes, halo_ID, Params["MASSDEF"], massdefdenstable, npartmin, Params["CENTERMODE"])

end = time.time()
print("SO halo finding finished under %fs.\n" % (end-start))

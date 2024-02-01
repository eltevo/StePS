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
from scipy.optimize import curve_fit
import astropy.units as u
from astropy.cosmology import LambdaCDM, wCDM, w0waCDM, z_at_value
from inputoutput import *

_VERSION="v0.0.1.0"
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

def NFW_profile(r,rho0,Rs):
    rpRs = r/Rs
    return rho0/(rpRs*np.power((1.0+rpRs),2))

def Hz(z, H0, Om, Ol, DE_model, DE_params):
    if DE_model == "Lambda":
        cosmo = LambdaCDM(H0=H0, Om0=Om, Ode0=Ol)
    elif DE_model == 'w0':
        cosmo = wCDM(H0=H0, Om0=Om, Ode0=Ol,w0=DE_params[0])
    elif DE_model == 'CPL':
        cosmo = w0waCDM(H0=H0, Om0=Om, Ode0=Ol,w0=DE_params[0],wa=DE_params[1])
    else:
        raise Exception("Error: unkown dark energy parametrization!\nExiting.\n")
    return cosmo.H(z).value

def get_Delta_c(z, H0, Om, Ol, DE_model, DE_params):
    """
    Virial overdensity constant calculation
    """
    if DE_model == "Lambda":
        cosmo = LambdaCDM(H0=H0, Om0=Om, Ode0=Ol)
    elif DE_model == 'w0':
        cosmo = wCDM(H0=H0, Om0=Om, Ode0=Ol,w0=DE_params[0])
    elif DE_model == 'CPL':
        cosmo = w0waCDM(H0=H0, Om0=Om, Ode0=Ol,w0=DE_params[0],wa=DE_params[1])
    else:
        raise Exception("Error: unkown dark energy parametrization!\nExiting.\n")
    x = cosmo.Om(z)-1.0
    return 18.0*np.pi**2 + 82.0*x + 39.0*x**2

def get_1D_radial_profile(r,M,Nbins,background_density=0.0):

    """
    This function reconstructs the 1D density profile of a halo
    by using equal "Npart" radial binning
    ---------------------------
    input:
            -r: distances from center
            -M: particle masses
            -Nbins: Number or radial bins
            -method: binning method. Must be "NGP" or "CIC"
    """
    rmax = r[-1]
    NpartTot = len(r)
    NpartPerBin = int(np.floor(NpartTot/Nbins))
    #Allocating memory for the profile
    r_bin_limits = np.zeros(Nbins,dtype=np.double)
    r_bin_centers = np.zeros(Nbins,dtype=np.double)
    rho_bins = np.zeros(Nbins,dtype=np.double)
    # i=0 bin:
    r_bin_limits[0] = r[NpartPerBin-1] #bin upper limit
    r_bin_centers[0] = 0.5*(r_bin_limits[0]) #bin center
    rho_bins[0] = np.sum(M[:NpartPerBin])/(4.0*np.pi/3.0*r_bin_limits[0]**3)
    for i in range(1,Nbins):
        r_bin_limits[i] = r[NpartPerBin*(i+1)-1] #bin upper limit
        r_bin_centers[i] = (0.5*(r_bin_limits[i] + r_bin_limits[i-1])) #bin center
        rho_bins[i] = np.sum(M[NpartPerBin*i:NpartPerBin*(i+1)])/(4.0*np.pi/3.0*(r_bin_limits[i]**3 - r_bin_limits[i-1]**3))
    out_idx = rho_bins > 0.0 # selecting non-empty bins
    rho_bins[out_idx] -= background_density #removing background density
    return r_bin_centers[out_idx], rho_bins[out_idx]





def calculate_halo_params(p, idx, halo_particleindexes, HaloID, massdefnames, massdefdenstable, npartmin, centermode,rho_b=0.0):
    Masslist = np.zeros(len(massdefdenstable), dtype=np.float32)
    Rlist = np.zeros(len(massdefdenstable), dtype=np.float32)
    # sorting particles by distance from the central particle
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
    # calculating enclosed density
    M_enc = np.cumsum(p.Masses[halo_particleindexes][sorted_idx]) # enclosed mass
    V_enc = 4.0*np.pi/3.0*np.power(distances[sorted_idx],3) # enclosed volume
    if centermode == "CENTRALPARTICLE":
        # In this mode, the first bin have zero volume, so we have to do this
        rho_enc = np.zeros(len(V_enc))
        rho_enc[1:]= M_enc[1:] / V_enc[1:] # enclosed mass / enclosed volume
        rho_enc[0] = rho_enc[1] # the first bin can
    else:
        rho_enc = M_enc / V_enc
    # calculating the parameters for each mass definitions (+ virial mass)
    R = np.zeros(len(massdefdenstable))
    M = np.zeros(len(massdefdenstable))
    V = np.zeros((len(massdefdenstable),3))
    Vrms = np.zeros(len(massdefdenstable))
    Vmax = np.zeros(len(massdefdenstable))
    J = np.zeros((len(massdefdenstable),3))
    Spin_Peebles = np.double(0.0)
    Spin_Bullock = np.double(0.0)
    for i in range(0,len(massdefdenstable)):
        radius_idx = massdefdenstable[i] <= rho_enc
        max_radi_idx = np.where(radius_idx == False)[0][0] # apply cut when the density first fall below the limit
        if i == 0:
            Npart = max_radi_idx+1
            if Npart < npartmin:
                if max_radi_idx == 0:
                    p.set_HaloParentIDs(p.IDs[halo_particleindexes][sorted_idx][0],-2)
                else:
                    p.set_HaloParentIDs(p.IDs[halo_particleindexes][sorted_idx][:max_radi_idx],-2) # setting the HaloParentIDs to -2, since this group has too less patricles
                return None
        if max_radi_idx > 0:
            # if max_radi_idx==0, then this mass definition is not applicable,
            #  because the halo doesn't have high enough density even at the center.
            R[i] = distances[sorted_idx][:max_radi_idx][-1] #Radii
            M[i] = M_enc[sorted_idx][:max_radi_idx][-1] #Mass
            V[i] = get_center_of_mass(p.Velocities[halo_particleindexes][sorted_idx][:max_radi_idx], p.Masses[halo_particleindexes][sorted_idx][:max_radi_idx]) #Velocity; the formula for calculating the mean velocity is the same as for the COM
            Vrms[i] = np.sqrt(np.sum(np.power(p.Velocities[halo_particleindexes][sorted_idx][:max_radi_idx] - V[i],2))/len(p.Velocities[halo_particleindexes][sorted_idx][:max_radi_idx])) # root mean square velocity
            Vmax[i] = np.max(np.sqrt(np.sum(np.power(p.Velocities[halo_particleindexes][sorted_idx][:max_radi_idx] - V[i],2), axis=1))) # maximal velocity
            #J[i] = #angular momentum in (Msun/h) * (Mpc/h) * km/s physical (non-comoving) units
            if i == 0:
                p.set_HaloParentIDs(p.IDs[halo_particleindexes][sorted_idx][:max_radi_idx],HaloID) # setting the HaloParentIDs of the particles that are in this halo (within Rvir)
                # print("Mvir=%f Rvir=%f Npart=%i" % (M[0], R[0], Npart))
                # calculating 1D profile and NFW parameters within r<Rvir
                if Npart < 2*npartmin+1:
                    NprofileBins = int(np.floor(npartmin/2))-1
                elif Npart < 4*npartmin+1:
                    NprofileBins = npartmin
                else:
                    NprofileBins = 2*npartmin
                rbin_centers, rho_r = get_1D_radial_profile(distances[sorted_idx][:max_radi_idx],p.Masses[halo_particleindexes][sorted_idx][:max_radi_idx],NprofileBins,background_density=rho_b) # equal volume radial bins
                # fitting NFW profile
                par,cov = curve_fit(NFW_profile, rbin_centers, rho_r, p0=[rho_r[0]/2,R[0]],maxfev = 38400)
                rho0_NFW = par[0]
                Rs_NFW = par[1] #NFW scale radius
    #Spin parameters. Spins are dimensionless. Overview: https://arxiv.org/abs/1501.03280 (definitions: eq.1 and eq.4)
    #Spin_Peebles[i] =  Peebles Spin Parameter (1969) https://ui.adsabs.harvard.edu/abs/1969ApJ...155..393P/abstract
    #Spin_Bullock[i] =  BullockSpinParameter (2001) https://ui.adsabs.harvard.edu/abs/2001ApJ...555..240B/abstract
    #generating output dictionary containing all calculated quantities
    returndict = {
    "ID": HaloID,
    "Coordinates": Center,
    "Npart": Npart,
    "Rs": Rs_NFW * 1.0e3,
    "Mvir": M[0] * 1.0e11,
    "Rvir": R[0] * 1.0e3,
    "Vvir": V[0],
    "VRMSvir": Vrms[0],
    "VMAXvir": Vmax[0]
    }
    #saving all other quantities
    for i in range(0,len(massdefnames)):
        returndict["M"+massdefnames[i]] = M[i+1] * 1.0e11 # the output is in Msol/h
        returndict["R"+massdefnames[i]] = R[i+1] * 1.0e3 # the output in in kpc/h
        returndict["V"+massdefnames[i]] = V[i+1]
        returndict["VRMS"+massdefnames[i]] = Vrms[i+1]
        returndict["VMAX"+massdefnames[i]] = Vmax[i+1]
    return returndict

#defining classes
class StePS_Particle_Catalog:
    '''
    A class for storing particle information.
    Stored information: ID, coordinate components, velocity components, mass, Parent halo ID, density
    '''

    def __init__(self, FILENAME, D_UNITS, V_UNITS, M_UNITS, REDSHIFT=0.0, FORCE_RES=0.0):
        print("Creating a new particle catalog by loading %s\n" % FILENAME)
        if FILENAME[-4:] == 'hdf5':
            self.Redshift, self.Om, self.Ol, self.H0, self.Npart = Load_params_from_HDF5_snap(FILENAME)
        else:
            self.Redshift = REDSHIFT
        self.a = 1.0/(self.Redshift+1.0) # scale factor
        self.sourcefile = FILENAME
        self.Coordinates, self.Velocities, self.Masses, self.IDs = Load_snapshot(FILENAME,CONSTANT_RES=False,RETURN_VELOCITIES=True,RETURN_IDs=True,SILENT=True)
        self.HaloParentIDs = -1*np.ones(len(self.Masses),dtype=np.int64)
        self.Density= np.zeros(len(self.Masses),dtype=np.double)
        # converting the input data to StePS units (Mpc, km/s, 1e11Msol)
        self.Coordinates *= D_UNITS
        self.Velocities *= V_UNITS
        self.Velocities *= np.sqrt(self.a) #physical velocities, assuming Gadget convention
        self.Masses *= (M_UNITS / 1e11)
        if FILENAME[-4:] != 'hdf5':
            self.Npart = len(self.Masses)
        # setting softening lengths
        Minmass = np.min(self.Masses)
        self.SoftLength = np.cbrt(self.Masses/Minmass)*FORCE_RES
        return
    def set_HaloParentIDs(self, PartIDs, ParentID,SILENT=True):
        idx = np.in1d(p.IDs,PartIDs)
        self.HaloParentIDs[idx] = ParentID
        if SILENT==False:
            print("Halo parent IDs got updated for ", PartIDs, " to ", ParentID)

    def printlines(self,lines):
        print("ID\t(X      Y      Z) [Mpc/h]\t\t(Vx      Vy      Vz) [km/s]\t\tM[1e11Msol/h]\tDensity[rho/rho_crit]\tParentID\n----------------------------------------------------------------------------------------------------------------------------------------")
        for line in lines:
            print("%-1i\t(%+-10.2f %+-10.2f %+-7.2f)\t\t(%+-10.2f %+-10.2f %+-7.2f)\t\t%-8.3g\t%-8.3g\t\t%i" % (self.IDs[line], self.Coordinates[line,0], self.Coordinates[line,1], self.Coordinates[line,2], self.Velocities[line,0], self.Velocities[line,1], self.Velocities[line,2], self.Masses[line], self.Density[line], self.HaloParentIDs[line]))
        print("\n")
        return

class StePS_Halo_Catalog:
    '''
    A class for storing halo catalogs
    '''
    def __init__(self, H0, Om, Ol, DE_Model, DE_Params, z, rho_c, rho_b, PrimaryMDef, SecondaryMdefList):
        #During initialization we fill the header
        Mdef = [PrimaryMDef]
        Mdef.append(SecondaryMdefList)
        self.Header = {
            "Redshift": z,
            "H0": H0,
            "OmegaM": Om,
            "OmegaL": Ol,
            "DE_Model": DE_Model,
            "DE_Params": DE_Params,
            "Nhalos": 0,
            "MassDefinitions": Mdef,
            "rho_c": rho_c,
            "rho_b": rho_b
        }
        self.Nhalos = 0
        self.DataTable = [] # empty list
    def add_halo(self,haloparamdict):
        self.DataTable.append(haloparamdict)
        self.Nhalos += 1
        self.Header["Nhalos"] = self.Nhalos
        #print(haloparamdict)
        return
    def print_halos(self,haloIDlist,Mdef="vir"):
        print("\nID\tNpart\t(X      Y      Z) [Mpc/h]\t\t(Vx      Vy      Vz) [km/s]\t\tM"+Mdef+"[Msol/h]\t\tR"+Mdef+"[kpc/h]")
        print("------------------------------------------------------------------------------------------------------------------------------------")
        for line in haloIDlist:
            print("%-1i\t%-1i\t(%+-10.2f %+-10.2f %+-7.2f)\t\t(%+-10.2f %+-10.2f %+-7.2f)\t\t%-8.4e\t\t%-8.2f" % (self.DataTable[line]["ID"],self.DataTable[line]["Npart"], self.DataTable[line]["Coordinates"][0], self.DataTable[line]["Coordinates"][1], self.DataTable[line]["Coordinates"][2], self.DataTable[line]["V"+Mdef][0], self.DataTable[line]["V"+Mdef][1], self.DataTable[line]["V"+Mdef][2],self.DataTable[line]["M"+Mdef],self.DataTable[line]["R"+Mdef]))
        print("\n")
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
#setting the redshift (and cosmological parameters, if the input in HDF5 format)
if Params['INFILE'][-4:] == 'hdf5':
    redshift, Omega_m, Omega_l, H0, Npart = Load_params_from_HDF5_snap(Params['INFILE'])
else:
    redshift = np.double(Params['REDSHIFT'])
    Omega_m = np.double(Params['OMEGAM'])
    Omega_l = np.double(Params['OMEGAL'])
    H0 = np.double(Params['H0'])
print("Cosmological Parameters:\n------------------------\n\u03A9_m:\t\t\t%f\t(Ommh2=%f; Omch2=%f)\n\u03A9_lambda:\t\t%f\n\u03A9_k:\t\t\t%f\n\u03A9_b:\t\t\t%f\t(Ombh2=%f)\nH0:\t\t\t%f km/s/Mpc\nDark energy model:\t%s" % (Omega_m, Omega_m * (Params['H0']/100.0)**2, (Omega_m - Params['OMEGAB']) * (Params['H0']/100.0)**2, Omega_l, 1.0-Omega_m-Omega_l, Params['OMEGAB'], (Params['OMEGAB']) * (Params['H0']/100.0)**2, Params['H0'], Params['DARKENERGYMODEL']))
if Params['DARKENERGYMODEL'] == 'Lambda':
    print("\t\t\t(w = -1)")
elif Params['DARKENERGYMODEL'] == 'w0':
    print("\t\t\tw = %f" % Params['DARKENERGYPARAMS'][0])
elif Params['DARKENERGYMODEL'] == 'CPL':
    print("\t\t\tw0 = %f\n\t\t\twa = %f" % (Params['DARKENERGYPARAMS'][0], Params['DARKENERGYPARAMS'][1]))
else:
    print("Error: unkown dark energy parametrization!\nExiting.\n")
    sys.exit(2)
# Setting up the units of distance and time (the usual StePS internal units)
UNIT_T=47.14829951063323 #Unit time in Gy
UNIT_V=20.738652969925447 #Unit velocity in km/s
UNIT_D=3.0856775814671917e24 #=1Mpc Unit distance in cm

# Calculating relevant cosmological quantities
rho_c = 3.0*Hz(redshift, H0, Omega_m, Omega_l, Params["DARKENERGYMODEL"], Params["DARKENERGYPARAMS"])**2/(8*np.pi)/UNIT_V/UNIT_V/(redshift+1)**3 #comoving critical density in internal units (G=1)
rho_b = 3.0*Params['H0']**2/(8*np.pi)/UNIT_V/UNIT_V * Omega_m #background density in internal units (G=1) [the comoving background density is redshift independent]
print("\u03C1_c (comoving):\t\t%.4e h^2Msol/Mpc^3\n\u03C1_b (comoving):\t\t%.4e h^2Msol/Mpc^3\n" % (rho_c*1e11, rho_b*1e11))
Delta_c = get_Delta_c(redshift, H0, Omega_m, Omega_l, Params["DARKENERGYMODEL"], Params["DARKENERGYPARAMS"]) #Virial overdensity constant

print("Snapshot Parameters:\n------------------------\nRedshift:\t\t%.4f\nRadius:\t\t\t%.6gMpc/h\nSoftening Length:\t%.4gMpc/h\nDistance units:\t\t%.2gMpc/h\nVelocity units:\t\t%.2gkm/s\nMass units:\t\t%.2gMsol/h\n" % (redshift,np.double(Params['RSIM']),np.double(Params['PARTICLE_RADII']),np.double(Params['UNIT_D_IN_MPC']), np.double(Params['UNIT_V_IN_KMPS']), np.double(Params['UNIT_M_IN_MSOL'])))

print("Halo Finder Parameters:\n---------------------------\nInitial Density Estimation:\t\t%s\nSearch radius alpha parameter:\t\t%.2f\nNumber of KDTree worker threads:\t%i\nMinimal particle number:\t\t%i\nHalo center mode:\t\t\t%s\nMass definitions:" %(Params["INITIAL_DENSITY_MODE"],np.double(Params["SEARCH_RADIUS_ALPHA"]),int(Params["KDWORKERS"]), int(Params["NPARTMIN"]), Params["CENTERMODE"] ))
Nmassdef = len(Params["MASSDEF"])+1 #total number of mass definitions.
print("\t\t- %s" % "Vir [\u0394c = %.2f] (Default primary definition, cannot be changed)" % Delta_c)
for i in range(0,Nmassdef-1):
    print("\t\t- %s" % Params["MASSDEF"][i])
# generating density levels for the different mass definitions:
massdefdenstable = np.zeros(Nmassdef,dtype=np.float32) # array containing the mass definition density levels.
massdefdenstable[0] = Delta_c * rho_c # virial density parameter (Delta_c*rho_crit)
for i in range(1,Nmassdef):
    massdefdenstable[i] = np.double(Params["MASSDEF"][i-1][:-1])
    if Params["MASSDEF"][i-1][-1] == "c" or Params["MASSDEF"][i-1][-1] == "C":
        massdefdenstable[i] *= rho_c
    elif Params["MASSDEF"][i-1][-1] == "b" or Params["MASSDEF"][i-1][-1] == "B":
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
p = StePS_Particle_Catalog(Params['INFILE'], np.double(Params['UNIT_D_IN_MPC']), np.double(Params['UNIT_V_IN_KMPS']), np.double(Params['UNIT_M_IN_MSOL']),REDSHIFT=np.double(Params['REDSHIFT']),FORCE_RES=np.double(Params['PARTICLE_RADII']))

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

p.printlines([0,1000,119784,458323,473220,1000000,1700000])
#p.printlines(np.arange(0,p.Npart))

# Identifying halos using Spherical Overdensity (SO) method
halos = StePS_Halo_Catalog(np.double(Params["H0"]), np.double(Params["OMEGAM"]), np.double(Params["OMEGAL"]), Params["DARKENERGYMODEL"], Params["DARKENERGYPARAMS"], redshift, rho_c, rho_b, "Vir", Params["MASSDEF"])
halo_ID = 0
while True:
    #selecting the largest density particle with parentID=-1
    idx = p.IDs[p.HaloParentIDs == -1][np.argmax(p.Density[p.HaloParentIDs == -1])]
    maxdens = p.Density[idx]
    #Query the kd-tree for nearest neighbors.
    search_radius = alpha*np.cbrt(p.Masses[idx]/rho_b) #In StePS simulations, the particles are more density packed at the center. The typical particle separation is proportional to the cubic root of the particle mass.
    halo_particleindexes = tree.query_ball_point(p.Coordinates[idx], search_radius, p=2.0, eps=0, workers=kdworkers, return_sorted=False)
    if len(halo_particleindexes) >= npartmin:
        #print("\nCentral estimated density for halo #%i: %.2f \u03C1_c" % (halo_ID, maxdens))
        #print("\tCentral coordinate of halo #%i: " % (halo_ID), p.Coordinates[idx])
        #print("\tID of the central particle of halo #%i: %i" % (halo_ID,idx))
        #print("\tSearch radius for halo #%i: %.2fMpc/h" % (halo_ID, search_radius))
        #print("\tNumber of particles in the search radius of halo #%i:" % (halo_ID),len(halo_particleindexes))
        halo_params = calculate_halo_params(p, idx, halo_particleindexes, halo_ID, Params["MASSDEF"], massdefdenstable, npartmin, Params["CENTERMODE"],rho_b=rho_b)
        if halo_params != None:
            halos.add_halo(halo_params) #adding the identified halo to the catalog
            halo_ID +=1
        #else:
        #    print("This candidate didn't had enough partilces.")
    if maxdens <= Delta_c:
        print("Central estimated density for the last halo candidate #%i: %.2f \u03C1_c" % (halo_ID, maxdens))
        #This means that in the center, we did not reach Delta_c*rho_c.
        #After this, we will not find new halos.
        break;
end = time.time()
halos.print_halos(np.arange(0,halos.Nhalos),Mdef="vir")
p.printlines([0,1000,119784,458323,473220,1000000,1700000])
print("SO halo finding finished under %fs.\n" % (end-start))

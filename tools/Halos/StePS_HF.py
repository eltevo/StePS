#!/usr/bin/env python3

#*******************************************************************************#
#  StePS_HF.py - a Halo Finder script for                                       #
#      STEreographically Projected cosmological Simulations                     #
#    Copyright (C) 2024-2025 Gabor Racz                                         #
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
# Base algorithm:                                                   #
#   1. Load the snapshot                                            #
#   2. Reconstruct the density field with                           #
#      Voronoi tessellation / Nth nearest neighbor method          #
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
import copy
from mpi4py import MPI
import time
from datetime import datetime
import yaml
import numpy as np
from scipy.spatial import KDTree
from inputoutput import *
import StePS_HF
from StePS_HF import _VERSION, _AUTHOR, _YEAR
from StePS_HF import G, H2RHO_C
from StePS_HF.geometry import *
from StePS_HF.halo_params import *

# Defining functions
def calculate_halo_params(p, idx, halo_particleindexes, HaloID, massdefnames, massdefdenstable, npartmin, centermode, hnow, boundonly=False, unbound_threshold=0.0, boundaries="STEPS", Lbox=0, rho_b=0.0):
    """
    Function for calculating various halo parameters
    Input:
        - p: a StePS_Particle_Catalog containing the particles of the simulation
        - idx: ID of the central particle candidate of the halo
        - halo_particleindexes: list of indexes of the particles that can be potentially part of the halo
        - HaloID: ID of the new halo
        - massdefnames: a list containing the mass definition names
        - massdefdenstable: an array containing the density limits of the mass definitions
        - npartmin: minimal particle number of a halo
        - centermode: method for identifying the center of the halo. Implemented methods:
            -> "CENTRALPARTICLE": the coordinates of the central particle is the halo center (fast)
            -> "CENTEROFMASSNPARTMIN": using center of mass of the most inner npartmin particles as center (more physical)
        - hnow: Hubble parameter at the current scale factor in km/s/Mpc units
        - boundonly: If True, only bound particles will be considered during the parameter estimation.
        - unbound_threshold: threshold for the unbound particles.
        - boundaries: boundary condition. Must be "STEPS", "PERIODIC", or "CYLINDRICAL"
        - rho_b: background density
    Returns:
        - returndict: A dictionary containing all calculated halo parameters such us position, velocity, mass, radius, angular momentum, spin parameters, scale radius, energy, velocity dispersion, etc.

    """
    # Sorting particles by distance from the central particle
    if boundaries=="STEPS" or boundaries=="CYLINDRICAL":
        distances = np.sqrt(np.sum((p.Coordinates[halo_particleindexes]-p.Coordinates[idx])**2,axis=1)) # calculating Euclidian particle distances from the center
    elif boundaries=="PERIODIC":
        # using only the nearest periodic images in the distance calculation
        distances = get_periodic_distances(p.Coordinates[halo_particleindexes],p.Coordinates[idx],Lbox) # calculating Euclidian particle distances from the center in toroidal space
    else:
        raise Exception("Error: unkonwn boundary condition %s." % (boundaries))
    sorted_idx = distances.argsort() # sorting
    # Finding the center of the halo
    if centermode == "CENTRALPARTICLE":
        # using the particle with the highest estimated density as center
        Center = p.Coordinates[idx]
    elif centermode == "CENTEROFMASSNPARTMIN":
        # using the innermost npartmin particles to calculate the center of mass
        Center = get_center_of_mass(p.Coordinates[halo_particleindexes][sorted_idx][:npartmin], p.Masses[halo_particleindexes][sorted_idx][:npartmin], boundaries=boundaries, boxsize=Lbox)
        if boundaries=="STEPS" or boundaries=="CYLINDRICAL":
            distances = np.sqrt(np.sum(np.power((p.Coordinates[halo_particleindexes]-Center),2),axis=1)) #recalculating distances due to the new center
        elif boundaries=="PERIODIC":
            # forcing the center to be within the box
            Center = np.mod(Center, Lbox)
            distances = get_periodic_distances(p.Coordinates[halo_particleindexes], Center, Lbox)
        else:
            raise Exception("Error: unkonwn boundary condition %s." % (boundaries))
        sorted_idx = distances.argsort()
    elif centermode == "CENTEROFMASS":
        # using the center of mass of all particles within the estimated quarter-mass radius as center
        M_enc = np.cumsum(p.Masses[halo_particleindexes][sorted_idx]) # enclosed mass
        V_enc = 4.0*np.pi/3.0*np.power(distances[sorted_idx],3) # enclosed volume
        rho_enc = M_enc[1:] / V_enc[1:] # enclosed mass / enclosed volume
        com_radi_idx = int( np.where( (massdefdenstable[0] <= rho_enc) == False)[0][0] * 0.25 )+1
        Center = get_center_of_mass(p.Coordinates[halo_particleindexes][sorted_idx][:com_radi_idx], p.Masses[halo_particleindexes][sorted_idx][:com_radi_idx], boundaries=boundaries, boxsize=Lbox)
        if boundaries=="STEPS" or boundaries=="CYLINDRICAL":
            distances = np.sqrt(np.sum(np.power((p.Coordinates[halo_particleindexes]-Center),2),axis=1)) #recalculating distances due to the new center
        elif boundaries=="PERIODIC":
            # forcing the center to be within the box
            Center = np.mod(Center, Lbox)
            distances = get_periodic_distances(p.Coordinates[halo_particleindexes], Center, Lbox)
        else:
            raise Exception("Error: unkonwn boundary condition %s." % (boundaries))
        sorted_idx = distances.argsort()
    else:
        raise Exception("Error: unkown CENTERMODE parameter %s." % (centermode))
    Ntota = len(distances)# total number of particles in the analysis (the numbers of the particle within the halo will be smaller)
    # calculating enclosed density
    M_enc = np.cumsum(p.Masses[halo_particleindexes][sorted_idx]) # enclosed mass
    V_enc = 4.0*np.pi/3.0*np.power(distances[sorted_idx],3) # enclosed volume
    if centermode == "CENTRALPARTICLE":
        # In this mode, the first bin have zero volume, so we have to do this
        rho_enc = np.zeros(len(V_enc))
        rho_enc[1:]= M_enc[1:] / V_enc[1:] # enclosed mass / enclosed volume
        rho_enc[0] = rho_enc[1]
    else:
        rho_enc = M_enc / V_enc
    # Flagging unbound particles, if needed (this can have significant effect in the runtime, if the seach radius is too large)
    if boundonly:
        # For this, we have a first estimate on the bulk velocity of the halo. Using SO Mvir definition for this:
        radius_idx = massdefdenstable[0] <= rho_enc
        max_radi_idx_so = np.where(radius_idx == False)[0][0] # apply cut when the density first fall below the limit
        if (max_radi_idx_so+1)<npartmin:
            # In this case, the SO definition has smaller number of particles then npartmin. This also means that the BO halo definition will not have enough particles.
            p.set_HaloParentIDs(p.IDs[idx],-2) # since this group doesn't have enough patricles, the "central" particle IDs had to be set to -2
            if max_radi_idx_so == 0:
                p.set_HaloParentIDs(p.IDs[halo_particleindexes][sorted_idx][0],-2)
            else:
                p.set_HaloParentIDs(p.IDs[halo_particleindexes][sorted_idx][:max_radi_idx_so],-2) # setting the HaloParentIDs to -2, since this group has too less patricles
            return None
        M_SO = M_enc[:max_radi_idx_so][-1] #Mass
        R_SO = distances[sorted_idx][:max_radi_idx_so][-1] #Radii
        N_SO = len(distances[sorted_idx][:max_radi_idx_so]) #Number of particles
        V_SO = get_center_of_mass(p.Velocities[halo_particleindexes][sorted_idx][:max_radi_idx_so], p.Masses[halo_particleindexes][sorted_idx][:max_radi_idx_so]) #Velocity; the formula for calculating the mean velocity is the same as for the CoM
        # Calculating individual energies
        if np.min(massdefdenstable) == massdefdenstable[0]:
            T,U = get_individual_energy(p.Coordinates[halo_particleindexes][sorted_idx][:max_radi_idx_so]-Center,p.Velocities[halo_particleindexes][sorted_idx][:max_radi_idx_so]-V_SO,p.Masses[halo_particleindexes][sorted_idx][:max_radi_idx_so]*1e11,p.a,hnow,p.SoftLength[halo_particleindexes][sorted_idx][:max_radi_idx_so], boundary=boundaries, boxsize=Lbox)
        else:
            # if the default mass definition has not have the minimal density threshold, then we have to use the minimal one to calculate the total energy
            # this is not the best solution, but it is fast and works
            radius_minrho_idx = np.min(massdefdenstable) <= rho_enc
            max_radi_minrho_idx_so = np.where(radius_minrho_idx == False)[0][0] # apply cut when the density first fall below the limit
            T,U = get_individual_energy(p.Coordinates[halo_particleindexes][sorted_idx][:max_radi_minrho_idx_so]-Center,p.Velocities[halo_particleindexes][sorted_idx][:max_radi_minrho_idx_so]-V_SO,p.Masses[halo_particleindexes][sorted_idx][:max_radi_minrho_idx_so]*1e11,p.a,hnow,p.SoftLength[halo_particleindexes][sorted_idx][:max_radi_minrho_idx_so], boundary=boundaries, boxsize=Lbox)
        bound = (T+U)<0.0 # a bool array. If the particle is bound True; False if not
        bound_ratio = np.sum(bound[:max_radi_idx_so])/len(bound[:max_radi_idx_so]) # bound ratio
        if (bound_ratio < unbound_threshold) or (N_SO < npartmin):
            # this means that the halo is not bound enough, or does not have enough particles, so we have to set the HaloParentIDs to -2
            p.set_HaloParentIDs(p.IDs[idx],-2) # since this group doesn't have enough patricles, the "central" particle IDs had to be set to -2
            if max_radi_idx_so == 0:
                p.set_HaloParentIDs(p.IDs[halo_particleindexes][sorted_idx][0],-2)
            else:
                p.set_HaloParentIDs(p.IDs[halo_particleindexes][sorted_idx][:max_radi_idx_so],-2) # setting the HaloParentIDs to -2, since this group has too less patricles
            return None
        # After this, the enclosed density and mass has to be re-calculated
        bound = np.pad(bound, (0, len(distances)-len(bound)), 'constant', constant_values=(False, False)) # padding the bound array to the same length as the distances
        M_enc = np.cumsum(p.Masses[halo_particleindexes][sorted_idx]*np.double(bound)) # enclosed (bound) mass
        V_enc = 4.0*np.pi/3.0*np.power(distances[sorted_idx],3) # enclosed (bound) volume
        if centermode == "CENTRALPARTICLE":
            # In this mode, the first bin have zero volume, so we have to do this
            rho_enc = np.zeros(len(V_enc))
            rho_enc[1:]= M_enc[1:] / V_enc[1:] # enclosed mass / enclosed volume
            rho_enc[0] = rho_enc[1]
        else:
            rho_enc = M_enc / V_enc
    else:
        bound = np.ones(len(distances),dtype='bool') # using all particles in SO mode.
    # calculating the parameters for each mass definitions (+ virial mass)
    R = np.zeros(len(massdefdenstable)) # Radii
    M = np.zeros(len(massdefdenstable)) # Masses
    V = np.zeros((len(massdefdenstable),3)) # Bulk velocity vector
    Vrms = np.zeros(len(massdefdenstable)) # velocity dispersion within the halo
    Vcirc = np.zeros(len(massdefdenstable)) # circular orbital velocity at Radius
    J = np.zeros((len(massdefdenstable),3))
    for i in range(0,len(massdefdenstable)):
        radius_idx = massdefdenstable[i] <= rho_enc
        max_radi_idx = np.where(radius_idx == False)[0][0] # apply cut when the density first fall below the limit
        if i == 0:
            Npart = max_radi_idx
            #print("Menc=",M_enc)
            #print("R=",distances[sorted_idx][bound])
            #print("Venc=",V_enc)
            #print(rho_enc/massdefdenstable[i])
            if Npart < npartmin:
                if boundonly:
                    p.set_HaloParentIDs(p.IDs[idx],-2) # since this group doesn't have enough patricles, the "central" particle IDs had to be set to -2
                    if max_radi_idx == 0:
                        p.set_HaloParentIDs(p.IDs[halo_particleindexes][sorted_idx][0],-2)
                    else:
                        p.set_HaloParentIDs(p.IDs[halo_particleindexes][sorted_idx][:max_radi_idx_so],-2) # setting the HaloParentIDs to -2, since this group has too less patricles
                    return None
                else:
                    p.set_HaloParentIDs(p.IDs[idx],-2) # since this group doesn't have enough patricles, the "central" particle IDs had to be set to -2
                    if max_radi_idx == 0:
                        p.set_HaloParentIDs(p.IDs[halo_particleindexes][sorted_idx][0],-2)
                    else:
                        p.set_HaloParentIDs(p.IDs[halo_particleindexes][sorted_idx][bound][:max_radi_idx],-2) # setting the HaloParentIDs to -2, since this group has too less patricles
                    return None
            if centermode == "CENTRALPARTICLE":
                Vmax = np.max(np.sqrt(G*1e11*M_enc[1:]/(distances[sorted_idx][1:]*p.a))) # Maximal circular velocity of the halo
            else:
                Vmax = np.max(np.sqrt(G*1e11*M_enc[:]/(distances[sorted_idx][:]*p.a))) # Maximal circular velocity of the halo
        if max_radi_idx > 0:
            # if max_radi_idx==0, then this mass definition is not applicable,
            #  because the halo doesn't have high enough density even at the center.
            M[i] = M_enc[max_radi_idx-1] #Mass
            if i == 0:
                MassMmaxPart = M[i]/npartmin #Mass of particles, if the halo is built from npartmin particles
                #getting the resolved volume from the p.UniqueMasses and p.UniqueMassVolumes
                massresID = find_nearest_id(p.UniqueMasses,MassMmaxPart)
                if p.UniqueMasses[massresID]>MassMmaxPart: # if the find mass bin is larger than the mass, then we have to use the next one (assuming increasing order of mass bins)
                    massresID -= 1 # if the find mass bin is larger than the mass, then we have to use the next one (assuming increasing order of mass bins)
                vresolved = p.UniqueMassVolumes[massresID]
            # The halo radius can be directly calculated from the mass and density treshold
            R[i] = np.cbrt((3.0/(np.pi*4))*M[i]/massdefdenstable[i])
            # Or alternatively, we can use the coordinate of the last particle:
            #R[i] = distances[sorted_idx][bound][max_radi_idx-1] #Radii
            V[i] = get_center_of_mass(p.Velocities[halo_particleindexes][sorted_idx][bound][:max_radi_idx], p.Masses[halo_particleindexes][sorted_idx][bound][:max_radi_idx]) #Velocity; the formula for calculating the mean velocity is the same as for the CoM
            Vrms[i] = np.sqrt(np.sum(np.power(p.Velocities[halo_particleindexes][sorted_idx][bound][:max_radi_idx] - V[i],2))/len(p.Velocities[halo_particleindexes][sorted_idx][bound][:max_radi_idx])) # root mean square velocity
            if boundaries=="STEPS" or boundaries=="CYLINDRICAL":
                J[i] = get_angular_momentum((p.Coordinates[halo_particleindexes][sorted_idx][bound][:max_radi_idx]-Center),p.Velocities[halo_particleindexes][sorted_idx][bound][:max_radi_idx] - V[i],p.Masses[halo_particleindexes][sorted_idx][bound][:max_radi_idx],p.a,hnow) #angular momentum in (Msun/h) * (Mpc/h) * km/s physical (non-comoving) units
            elif boundaries=="PERIODIC":
                J[i] = get_angular_momentum(get_periodic_distance_vec(p.Coordinates[halo_particleindexes][sorted_idx][bound][:max_radi_idx],Center,Lbox),p.Velocities[halo_particleindexes][sorted_idx][bound][:max_radi_idx] - V[i],p.Masses[halo_particleindexes][sorted_idx][bound][:max_radi_idx],p.a,hnow) #angular momentum in (Msun/h) * (Mpc/h) * km/s physical (non-comoving) units
            else:
                raise Exception("Error: unkonwn boundary condition %s." % (boundaries))
            Vcirc[i] = np.sqrt(G*1e11*M[i]/(R[i]*p.a))
            if i == 0:
                p.set_HaloParentIDs(p.IDs[idx],-2) # even if the "central" particle is not bound, this will ensure to not to check this group again.
                if boundonly:
                    p.set_HaloParentIDs(p.IDs[halo_particleindexes][sorted_idx][:max_radi_idx_so],HaloID) # setting the HaloParentIDs of the particles that are in this halo (within RvirSO)
                else:
                    p.set_HaloParentIDs(p.IDs[halo_particleindexes][sorted_idx][:max_radi_idx],HaloID) # setting the HaloParentIDs of the particles that are in this halo (within Rvir)
                c_klypin, Rs_klypin = get_Rs_Klypin(Vmax,Vcirc[0],R[0])
                if boundonly:
                    Ekin = np.sum(T)
                    Epot = np.sum(U)
                    MSSO = np.sum(p.Masses[halo_particleindexes][distances<=R[i]]) # total mass within Rvir (Strict Spherical Overdensity)
                    MboundPerMtot = M[i] / MSSO #Total bounded mass ratio within Rvir
                    Energy = Ekin+Epot
                #else:
                #    Energy, Ekin, Epot = get_total_energy((p.Coordinates[halo_particleindexes][sorted_idx][:max_radi_idx]-Center),(p.Velocities[halo_particleindexes][sorted_idx][:max_radi_idx] - V[i]),p.Masses[halo_particleindexes][sorted_idx][:max_radi_idx]*1e11,p.a,hnow,p.SoftLength[halo_particleindexes][sorted_idx][:max_radi_idx],boundary=boundaries,boxsize=Lbox) #Total energy of the halo. Needed for Peebles spin parameter
    #Spin parameters. Spins are dimensionless. Overview: https://arxiv.org/abs/1501.03280 (definitions: eq.1 and eq.4)
    absJvir = np.sqrt(np.sum(np.power(J[0],2)))
    if boundonly:
        Spin_Peebles = absJvir*1e11*np.sqrt(np.abs(Energy))/(G*np.power(M[0]*1e11,2.5)) # Peebles Spin Parameter (1969) https://ui.adsabs.harvard.edu/abs/1969ApJ...155..393P/abstract
    else:
        Spin_Peebles = 0.0
        Energy = 0.0
        Ekin = 0.0
        Epot = 1.0
    Spin_Bullock = get_bullock_spin(absJvir*1e11,M[0]*1e11,R[0]*p.a) # Bullock Spin Parameter (2001) https://ui.adsabs.harvard.edu/abs/2001ApJ...555..240B/abstract
    #generating output dictionary containing all calculated quantities
    returndict = {
    "ID": HaloID,
    "Npart": Npart,
    "VolResolved": vresolved,
    "MvirSO": 0.0,
    "RvirSO": 0.0,
    "Mvir": M[0] * 1.0e11,
    "Coordinates": Center,
    "Rvir": R[0] * 1.0e3,
    "Vvir": V[0],
    "VRMSvir": Vrms[0],
    "Vcircvir": Vcirc[0],
    "VMax": Vmax,
    "Rs_klypin":Rs_klypin * 1.0e3,
    "Jvir": J[0] * 1e11,
    "Spin_Peebles": Spin_Peebles,
    "Spin_Bullock": Spin_Bullock,
    "Energy": Energy,
    "T/|U|": Ekin/np.abs(Epot)
    }
    if boundonly:
        returndict["MvirSO"] = M_SO * 1.0e11 # the output masses are in Msol
        returndict["RvirSO"] = R_SO * 1.0e3 # the output radii are in kpc
        returndict["MboundRatio"] = MboundPerMtot
    else:
        del returndict["MvirSO"]
        del returndict["RvirSO"]
        del returndict["Energy"]
        del returndict["T/|U|"]
        del returndict["Spin_Peebles"]
    #saving all other quantities
    for i in range(0,len(massdefnames)):
        returndict["M"+massdefnames[i]] = M[i+1] * 1.0e11 # the output masses are in Msol
        returndict["R"+massdefnames[i]] = R[i+1] * 1.0e3 # the output radii are in kpc
        returndict["V"+massdefnames[i]] = V[i+1]
        returndict["VRMS"+massdefnames[i]] = Vrms[i+1]
        returndict["Vcirc"+massdefnames[i]] = Vcirc[i+1]
        returndict["J"+massdefnames[i]] = J[i+1] * 1e11 # the output angular momenta in Msun * Mpc * km/s (physical)
    return returndict

def SetParentThreadIDs(p, R_center, d_r, delta_theta, N_MPI_threads, MPI_rank, BOUNDARIES="STEPS"):
    '''
    Function for setting the ParentThreadIDs of the particles in a halo catalog.
    This function is used to assign particles to MPI threads based on their position in the simulation volume.

    Input:
        - p: a StePS_Particle_Catalog containing the particles of the simulation
        - R_center: the central radius of high resolution region in Mpc units
        - d_r: the width of the overlap region in Mpc units
        - delta_theta: the width of the overlap region in radians
        - N_MPI_threads: the number of MPI threads
        - MPI_rank: the rank of the current MPI thread
        - BOUNDARIES: the boundary condition used in the simulation. Can be "STEPS" or "CYLINDRICAL".
    Returns:
        - ParentThreadID: an array containing the ParentThreadIDs of the particles.
    '''
    ParentThreadID= -1*np.ones(len(p.Masses),dtype=int)
    if N_MPI_threads <= 1:
        return ParentThreadID
    Nslices = N_MPI_threads - 1
    d_theta = 2.0*np.pi/Nslices
    if BOUNDARIES == "STEPS":
        r = np.sqrt(np.sum(np.power(p.Coordinates,2.0),axis=1))
    elif BOUNDARIES == "CYLINDRICAL":
        r = np.sqrt(np.sum(np.power(p.Coordinates[:,0:2],2.0),axis=1))
    if MPI_rank == 0:
        # the first MPI thread always gets the particles in the center
        ParentThreadID[r<R_center+0.5*d_r] = MPI_rank
        return ParentThreadID
    elif MPI_rank == 1 and N_MPI_threads == 2:
        # the second MPI thread gets the particles in the outer region, if there are only two MPI threads
        ParentThreadID[r>R_center-0.5*d_r] = MPI_rank
        return ParentThreadID
    else:
        # the rest of the MPI threads get the particles in tangential slices
        # using equal-angle cuts in the X-Y plane
        # for this, we use polar coordinates
        theta = np.arctan2(p.Coordinates[:,0], p.Coordinates[:,1])+np.pi
        # Note: the overlap region has a fixed width of delta_theta
        if MPI_rank == 1:
            # the second thread needs special threatment due to the overlap region is around 0 degrees
            ParentThreadID[np.logical_and(theta<=d_theta+0.5*delta_theta,r>R_center-0.5*d_r)] = MPI_rank
            ParentThreadID[np.logical_and(theta>=2*np.pi-0.5*delta_theta,r>R_center-0.5*d_r)] = MPI_rank
        elif MPI_rank == N_MPI_threads-1:
            # the last thread needs special threatment due to the overlap region is around 2*pi degrees
            ParentThreadID[np.logical_and(theta>=(N_MPI_threads-2)*d_theta-0.5*delta_theta,r>R_center-0.5*d_r)] = MPI_rank
            ParentThreadID[np.logical_and(theta<=0.5*delta_theta,r>R_center-0.5*d_r)] = MPI_rank
        else:
            ParentThreadID[np.logical_and(np.logical_and(theta>=(MPI_rank-1)*d_theta-0.5*delta_theta,theta<=(MPI_rank)*d_theta+0.5*delta_theta),r>R_center-0.5*d_r)] = MPI_rank
        return ParentThreadID

def IsHaloInThreadSubvolume(r,theta, z, R_center, N_MPI_threads, MPI_rank):
    if N_MPI_threads <= 1:
        return True
    elif N_MPI_threads == 2:
        if MPI_rank == 0:
            if (r<=R_center):
                return True
            else:
                return False
        else:
            if (r>R_center):
                return True
            else:
                return False
    else:
        # for three and more MPI threads, we use equal-angle cuts in the X-Y plane
        N_slices = N_MPI_threads - 1 # number of tangential slices
        if MPI_rank == 0:
            if (r<=R_center):
                return True
            else:
                return False
        else:
            if r<=R_center:
                return False 
            elif int(np.floor((theta)/(2.0*np.pi)*N_slices)) == MPI_rank-1:
                return True
            else:
                return False

# Defining classes
class StePS_Particle_Catalog:
    '''
    A class for storing particle information.
    Stored information: ID, coordinate components, velocity components, mass, Parent halo ID, density
    '''

    def __init__(self, FILENAME, D_UNITS, V_UNITS, M_UNITS, H_INDEPENDENT_UNITS, HUBBLE, REDSHIFT=0.0, FORCE_RES=0.0, BOUNDARIES="STEPS", LBOX=0.0, D_OVERLAP=0.0, VERBOSE=False):
        print("\nLoading particle data from %s" % FILENAME)
        if FILENAME[-4:] == 'hdf5':
            self.Redshift, self.Om, self.Ol, self.H0, self.Npart = Load_params_from_HDF5_snap(FILENAME)
        else:
            self.H0 = HUBBLE
            self.Redshift = REDSHIFT
        self.h = self.H0 / 100.0
        self.a = 1.0/(self.Redshift+1.0) # scale factor
        self.sourcefile = FILENAME
        self.Lbox = LBOX # box size in Mpc
        self.Coordinates, self.Velocities, self.Masses, self.IDs = Load_snapshot(FILENAME,CONSTANT_RES=False,RETURN_VELOCITIES=True,RETURN_IDs=True,SILENT=True)
        # converting the input data to StePS units (Mpc, km/s, 1e11Msol)
        if H_INDEPENDENT_UNITS:
            self.Coordinates *= D_UNITS/self.h
            self.Masses *= (M_UNITS / 1e11)/self.h
            self.Lbox *= D_UNITS/self.h
        else:
            self.Coordinates *= D_UNITS
            self.Masses *= (M_UNITS / 1e11) # 1e11msol(/h) units
            self.Lbox *= D_UNITS
        self.Velocities *= V_UNITS
        self.Velocities *= np.sqrt(self.a) # km/s physical velocities, assuming StePS (and Gadget) convention
        if FILENAME[-4:] != 'hdf5':
            self.Npart = len(self.Masses)
        # setting softening lengths
        Minmass = np.min(self.Masses)
        self.SoftLength = np.cbrt(self.Masses/Minmass)*FORCE_RES
        if BOUNDARIES == "CYLINDRICAL":
            print("Number of loaded particles in the cylindrical topology: %i" % (self.Npart))
            if VERBOSE:
                print("Applying periodic boundary conditions in the z direction with D_OVERLAP=%f Mpc and Lz=%f Mpc." % (D_OVERLAP, LBOX))
            # in cylindrical topology, the periodic boundary conditions are appied only in the z direction.
            self.Coordinates[:,2] = np.mod(self.Coordinates[:,2], self.Lbox) # forcing the z coordinate to be within the box
            # after this, we extend the volume in the z direction by D_OVERLAP in both directions by using periodic copies
            # copying the bottom of the cylinder to the top
            mask = np.logical_and(self.Coordinates[:,2] >= 0.0, self.Coordinates[:,2] <= D_OVERLAP)
            if VERBOSE:
                print("Top: Adding %i periodic copies in the z direction with D_OVERLAP=%f Mpc." % (len(self.Coordinates[mask,:]),D_OVERLAP))
            self.Coordinates = np.concatenate((self.Coordinates, self.Coordinates[mask,:]+np.array([0,0,self.Lbox,])), axis=0)
            self.Velocities = np.concatenate((self.Velocities, self.Velocities[mask,:]), axis=0)
            self.Masses = np.concatenate((self.Masses, self.Masses[mask]), axis=0)
            self.IDs = np.concatenate((self.IDs, self.IDs[mask]), axis=0)
            # copying the top of the cylinder to the bottom
            mask = np.logical_and(self.Coordinates[:,2] >= self.Lbox-D_OVERLAP, self.Coordinates[:,2] <= self.Lbox)
            if VERBOSE:
                print("Bottom: Adding %i periodic copies in the z direction with D_OVERLAP=%f Mpc." % (len(self.Coordinates[mask,:]),D_OVERLAP))
            self.Coordinates = np.concatenate((self.Coordinates, self.Coordinates[mask,:]-np.array([0,0,self.Lbox,])), axis=0)
            self.Velocities = np.concatenate((self.Velocities, self.Velocities[mask,:]), axis=0)
            self.Masses = np.concatenate((self.Masses, self.Masses[mask]), axis=0)
            self.IDs = np.concatenate((self.IDs, self.IDs[mask]), axis=0)
            # after this, we have to recalculate the total number of particles
            self.Npart = len(self.Masses)
            if VERBOSE:
                print("After periodic copies, the number of particles in the particle catalog is: %i\n" % (self.Npart))
        else:
            print("The number of loaded particles in the particle catalog is: %i\n" % (self.Npart))
        self.HaloParentIDs = -1*np.ones(len(self.Masses),dtype=np.int64)
        self.Density= np.zeros(len(self.Masses),dtype=np.double)
        # reconstructing the radius-mass resolution function
        self.UniqueMasses = np.unique(self.Masses)
        self.UniqueMassRadii = np.zeros_like(self.UniqueMasses)
        if BOUNDARIES == "PERIODIC":
            # Assuming that the Volume of Interest (VOI) is at the center of the box
            for i in range(0,len(self.UniqueMasses)-1):
                self.UniqueMassRadii[i] = np.mean(np.sqrt(np.sum(np.power(self.Coordinates[self.Masses == self.UniqueMasses[i],:]-self.Lbox*0.5,2),axis=1)))
        elif BOUNDARIES == "STEPS":
            for i in range(2,len(self.UniqueMasses)):
                self.UniqueMassRadii[i] = np.mean(np.sqrt(np.sum(np.power(self.Coordinates[self.Masses == self.UniqueMasses[i],:],2),axis=1)))
            # the first two mass bin is threated differently. (Usually we are in or close to the constant mass resolution limit)
        elif BOUNDARIES == "CYLINDRICAL":
            for i in range(2,len(self.UniqueMasses)):
                self.UniqueMassRadii[i] = np.mean(np.sqrt(np.sum(np.power(self.Coordinates[self.Masses == self.UniqueMasses[i],0:2],2),axis=1)))
            # the first two mass bin is threated differently. (Usually we are in or close to the constant mass resolution limit)
        else:
            raise Exception("Error: unkonwn boundary condition %s." % (BOUNDARIES))
        if BOUNDARIES == "STEPS" or BOUNDARIES == "CYLINDRICAL":
            # linearly extrapolating from i=3 and i=4 to i=0 and i=1
            fit = np.polyfit(self.UniqueMasses[2:6], self.UniqueMassRadii[2:6], 1) # linear fit
            self.UniqueMassRadii[1] = fit[0] * self.UniqueMasses[1] + fit[1] # extrapolating to i=1 
            self.UniqueMassRadii[0] = fit[0] * self.UniqueMasses[0] + fit[1] # extrapolating to i=0
        elif BOUNDARIES == "PERIODIC":
            # the last bin is needed a special treatment, since this volume is cubic, and not spherical
            self.UniqueMassRadii[-1] = self.Lbox * 0.5 # the radius of the last bin is half of the box size
            self.UniqueMassVolumes = 4/3 * np.pi * self.UniqueMassRadii**3 # Assuming spherical zoom-in regions
        # Calculating resolved volumes
        if BOUNDARIES == "STEPS":
            self.UniqueMassVolumes = 4/3 * np.pi * self.UniqueMassRadii**3 # Assuming spherical zoom-in regions
        elif BOUNDARIES == "CYLINDRICAL":
            self.UniqueMassVolumes = np.pi * self.UniqueMassRadii**2 * self.Lbox # Assuming cylindrical zoom-in regions
        elif BOUNDARIES == "PERIODIC":
            self.UniqueMassVolumes[-1] = self.Lbox**3 # the volume of the last bin is the box volume
        if VERBOSE:
            print("Unique masses and radii have been calculated.")
            print("R[Mpc]\tV[Mpc^3]\tM[Msol]\n--------------------------")
            for i in range(0,len(self.UniqueMasses)):
                print("%.3f\t%e\t%e" % (self.UniqueMassRadii[i], self.UniqueMassVolumes[i], self.UniqueMasses[i]*1e11)) # the output masses are in Msol
        return

    def set_HaloParentIDs(self, PartIDs, ParentID,SILENT=True):
        idx = np.in1d(p.IDs,PartIDs)
        self.HaloParentIDs[idx] = ParentID
        if SILENT==False:
            print("Halo parent IDs got updated for ", PartIDs, " to ", ParentID)
        return

    def printlines(self,lines):
        print("ID\t(X      Y      Z) [Mpc]\t\t(Vx      Vy      Vz) [km/s]\t\tM[1e11Msol]\tDensity[rho/rho_crit]\tParentID\n----------------------------------------------------------------------------------------------------------------------------------------")
        for line in lines:
            print("%-1i\t(%+-10.2f %+-10.2f %+-7.2f)\t\t(%+-10.2f %+-10.2f %+-7.2f)\t\t%-8.3g\t%-8.3g\t\t%i" % (self.IDs[line], self.Coordinates[line,0], self.Coordinates[line,1], self.Coordinates[line,2], self.Velocities[line,0], self.Velocities[line,1], self.Velocities[line,2], self.Masses[line], self.Density[line], self.HaloParentIDs[line]))
        print("\n")
        return

class StePS_Halo_Catalog:
    '''
    A class for storing halo catalogs
    '''
    def __init__(self, H0, Hnow, Om, Ol, DE_Model, DE_Params, z, rho_c, rho_b, PrimaryMDef, SecondaryMdefList, Centermode, DensityMode, Npartmin, RemoveUnBoundParts, HindepUnis=False, Boundaries="STEPS", Rsim=0.0, Lbox=0.0):
        #During initialization we fill the header
        Mdef = [PrimaryMDef]
        Mdef.append(SecondaryMdefList)
        self.Header = {
            "Redshift": z,
            "H0": H0,
            "H(z)": Hnow,
            "OmegaM": Om,
            "OmegaL": Ol,
            "DE_Model": DE_Model,
            "DE_Params": DE_Params,
            "Nhalos": 0,
            "MassDefinitions": Mdef,
            "rho_c": rho_c,
            "rho_b": rho_b,
            "CenterMode": Centermode,
            "DensityMode": DensityMode,
            "Npart_min": Npartmin,
            "RemoveUnBoundParts": RemoveUnBoundParts,
            "HindependentUnits": HindepUnis,
            "Boundaries": Boundaries
        }
        if Boundaries == "PERIODIC":
            self.Header["Lbox"] = Lbox
        elif Boundaries == "STEPS":
            self.Header["Rsim"] = Rsim
        elif Boundaries == "CYLINDRICAL":
            self.Header["Rsim"] = Rsim
            self.Header["Lbox"] = Lbox
        self.h = H0 / 100.0
        self.Nhalos = 0 # total number of halos in the catalog
        self.DataTable = [] # table containing all calculated halo parameters
        self.DenstyEstimation = [] # table containing all initial halo central density estimation

    def add_halo(self,haloparamdict,centraldensity):
        self.DataTable.append(haloparamdict)
        self.DenstyEstimation.append(centraldensity)
        self.Nhalos += 1
        self.Header["Nhalos"] = self.Nhalos
        return

    def add_catalog(self,halocatalog,overlaps=True,MPI_parent_thread=0, boundaries="STEPS", Lbox=0.0):
        Nhalos_stored = copy.deepcopy(self.Nhalos)
        outID=0
        if overlaps:
            for j in range(0,halocatalog.Nhalos):
                if boundaries == "STEPS":
                    r_halo = np.sqrt(np.sum(halocatalog.DataTable[j]["Coordinates"]**2))
                elif boundaries == "CYLINDRICAL":
                    r_halo = np.sqrt(np.sum(halocatalog.DataTable[j]["Coordinates"][:2]**2))
                theta_halo = np.arctan2(halocatalog.DataTable[j]["Coordinates"][0], halocatalog.DataTable[j]["Coordinates"][1])+np.pi
                # if halo center is within the subvolume of the parent thread, we add it to the catalog
                if boundaries == "STEPS" or boundaries == "PERIODIC":
                    if IsHaloInThreadSubvolume(r_halo,theta_halo, halocatalog.DataTable[j]["Coordinates"][2], r_master, size, MPI_parent_thread):
                        halocatalog.DataTable[j]["ID"] = outID + Nhalos_stored
                        self.DataTable.append(halocatalog.DataTable[j])
                        self.DenstyEstimation.append(halocatalog.DenstyEstimation[j])
                        outID += 1
                        self.Nhalos += 1
                elif boundaries == "CYLINDRICAL" and halocatalog.DataTable[j]["Coordinates"][2] >= 0.0 and halocatalog.DataTable[j]["Coordinates"][2] <= Lbox:
                    if IsHaloInThreadSubvolume(r_halo,theta_halo, halocatalog.DataTable[j]["Coordinates"][2], r_master, size, MPI_parent_thread):
                        halocatalog.DataTable[j]["ID"] = outID + Nhalos_stored
                        self.DataTable.append(halocatalog.DataTable[j])
                        self.DenstyEstimation.append(halocatalog.DenstyEstimation[j])
                        outID += 1
                        self.Nhalos += 1

        else:
            for j in range(0,halocatalog.Nhalos):
                if boundaries == "STEPS" or boundaries == "PERIODIC":
                    halocatalog.DataTable[j]["ID"] = outID + Nhalos_stored
                    self.DataTable.append(halocatalog.DataTable[j])
                    self.DenstyEstimation.append(halocatalog.DenstyEstimation[j])
                    outID += 1
                elif boundaries == "CYLINDRICAL" and halocatalog.DataTable[j]["Coordinates"][2] >= 0.0 and halocatalog.DataTable[j]["Coordinates"][2] <= Lbox:
                    halocatalog.DataTable[j]["ID"] = outID + Nhalos_stored
                    self.DataTable.append(halocatalog.DataTable[j])
                    self.DenstyEstimation.append(halocatalog.DenstyEstimation[j])
                    outID += 1
        self.Nhalos = len(self.DataTable)
        self.Header["Nhalos"] = self.Nhalos
        print("Nhalos after merge: ", self.Nhalos)
        return

    def print_halos(self,haloIDlist,Mdef="vir"):
        print("\nID\tNpart\t(X      Y      Z) [Mpc]\t\t\t(Vx      Vy      Vz) [km/s]\t\tM"+Mdef+"[Msol]\t\tR"+Mdef+"[kpc]\t\t(Jx      Jy      Jz)[Msol * Mpc * km/s]")
        print("---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
        for line in haloIDlist:
            print("%-1i\t%-1i\t(%+-10.2f %+-10.2f %+-7.2f)\t\t(%+-10.2f %+-10.2f %+-7.2f)\t\t%-8.4e\t\t%-8.2f\t\t(%+-10.2f %+-10.2f %+-7.2f)" % (self.DataTable[line]["ID"],self.DataTable[line]["Npart"], self.DataTable[line]["Coordinates"][0], self.DataTable[line]["Coordinates"][1], self.DataTable[line]["Coordinates"][2], self.DataTable[line]["V"+Mdef][0], self.DataTable[line]["V"+Mdef][1], self.DataTable[line]["V"+Mdef][2],self.DataTable[line]["M"+Mdef],self.DataTable[line]["R"+Mdef],self.DataTable[line]["J"+Mdef][0],self.DataTable[line]["J"+Mdef][1],self.DataTable[line]["J"+Mdef][2]))
        print("\n")
        return

    def save_ascii_catalog(self, filename):
        if self.Nhalos >= 1:
            #generating the header
            columnlist = ""
            columnlist_numbered = ""
            fields = 0 #number of columns in the file
            for key in self.DataTable[0].keys():
                datatype = type(self.DataTable[0][key])
                if datatype == np.ndarray:
                    #this is a vector
                    if key == "Coordinates":
                        column = "X Y Z "
                        columnlist_numbered += "%i-X %i-Y %i-Z " % (fields+1,fields+2,fields+3)
                    else:
                        column = key + "_X "+ key + "_Y " + key + "_Z "
                        columnlist_numbered += "%i-"%(fields+1) + key + "_X " + "%i-"%(fields+2) + key + "_Y " + "%i-"%(fields+3) + key + "_Z "
                    fields += 3
                else:
                    column = key + " "
                    fields += 1
                    columnlist_numbered += "%i-"%fields + column
                columnlist += column
            now = datetime.now()
            dt_string = now.strftime("%d/%m/%Y %H:%M")
            header = columnlist + "\n  /--------------------------------------------------------------------------\\\n | Halo catalog generated by StePS_HF.py version %s at %s |\n  \\--------------------------------------------------------------------------/\n  +---------------------------\n  | Parameters of the catalog:\n  +---------------------------\n" % (_VERSION, dt_string)
            for key in self.Header.keys():
                if type(self.Header[key]) == list:
                    header += "  | " + key +": \n"
                    for i in range(0,len(self.Header[key])):
                        if type(self.Header[key][i]) == list:
                            for j in range(0,len(self.Header[key][i])):
                                header += "  |\t\t* %s" % self.Header[key][i][j] + "\n"
                        else:
                            header += "  |\t\t* %s" % self.Header[key][i] + "\n"
                else:
                    header += "  | " + key +": %s\n" % self.Header[key]
            header += "  | Ncolumns: %i\n" % fields
            header += "  +---------------------------\n"
            if self.Header["HindependentUnits"]:
                header += "  | Units:\n  |\t Masses are in Msol/h \n  |\t Positions in Mpc/h (comoving)\n  |\t Velocities in km / s (physical)\n  |\t Halo Radii in kpc/h (comoving)\n  |\t Volumes in Mpc^3/h (comoving)\n  |\t Halo energies in (Msol/h) * (km/s)^2 (physical) \n  |\t Angular momenta in (Msun/h) * (Mpc/h) * km/s (physical)\n  |\t Spins are dimensionless\n  +---------------------------"
            else:
                header += "  | Units:\n  |\t Masses are in Msol \n  |\t Positions in Mpc (comoving)\n  |\t Velocities in km / s (physical)\n  |\t Halo Radii in kpc (comoving)\n  |\t Volumes in Mpc^3 (comoving)\n  |\t Halo energies in Msol * (km/s)^2 (physical) \n  |\t Angular momenta in Msun * Mpc * km/s (physical)\n  |\t Spins are dimensionless\n  +---------------------------"
            header += "\n" + columnlist_numbered
            #generating the output array
            outarray = np.zeros((self.Header["Nhalos"],fields), dtype=np.double)
            #figuring out the DataTable -> outarray mapping by using the first row:
            mapdict = { }
            i=0
            fmtstring = ""
            for key in self.DataTable[0].keys():
                if type(self.DataTable[0][key]) == np.ndarray:
                    # this is a 3D vector
                    mapdict[key] = [i,i+1,i+2]
                    i += 3
                    fmtstring += "%.7g %.7g %.7g "
                else:
                    # this is a scalar
                    mapdict[key] = i
                    i += 1
                    if type(self.DataTable[0][key]) == int or type(self.DataTable[0][key]) == np.int64 or type(self.DataTable[0][key]) == np.int32 or type(self.DataTable[0][key]) == np.uint64 or type(self.DataTable[0][key]) == np.uint32:
                        fmtstring += "%i "
                    else:
                        fmtstring += "%.7g "
            # storing the data into a numpy array
            # in an decreasing central density order
            sorted_idx = np.array(self.DenstyEstimation).argsort()[::-1]
            #print("Sorted_idx array: ", sorted_idx)
            for i in range(0,self.Nhalos):
                for key in self.DataTable[0].keys():
                    j = mapdict[key]
                    if type(mapdict[key]) == np.array:
                        outarray[i][j[0]:j[2]] = self.DataTable[sorted_idx[i]][key]
                    else:
                        if self.Header["HindependentUnits"]:
                            if key == "Coordinates":
                                outarray[i][j] = self.DataTable[sorted_idx[i]][key] * self.h
                            elif key[0]=="M" or key[0]=="R" or key=="Energy":
                                #print("Saving parameter in /h independent units: ", key)
                                outarray[i][j] = self.DataTable[sorted_idx[i]][key] * self.h
                            elif key[0]=="J":
                                #print("Saving angular momentum in /h physical units: ", key)
                                outarray[i][j] = self.DataTable[sorted_idx[i]][key] * (self.h**2)
                            elif key == "VolResolved":
                                #print("Saving volume in /h independent units: ", key)
                                outarray[i][j] = self.DataTable[sorted_idx[i]][key] * (self.h**3)
                            else:
                                outarray[i][j] = self.DataTable[sorted_idx[i]][key]
                        else:
                            outarray[i][j] = self.DataTable[sorted_idx[i]][key]
            np.savetxt(filename+".dat",outarray,fmt=fmtstring[:-1],header=header,comments='#')
        else:
            print("The halo catalog is empty. No file is saved.");
        return

    def save_hdf5_catalog(self, filename, save_particles=False, part_save_radius=1.0, particle_catalog=None, kdtree=None, precision=0):
        '''
        Function for saving the halo catalog in HDF5 format.
        If save_particles is True, then the particles belonging to each halo are also saved in the file:
            * The particles are saved in groups named "Halo_<haloID>" under the "Particles" group.
            * The particles are saved in datasets named "Coordinates", "Velocities", "Masses", "IDs".
            * The particles are saved within a radius of part_save_radius * Rvir, and the coordinates are given in comoving units.
            * The particle data is taken from the provided particle_catalog (StePS_Particle_Catalog) and the kdtree (scipy.spatial.cKDTree) is used for finding the particles within the given radius.
        Input:
            - filename: the name of the output file (without .hdf5 extension)
            - save_particles: boolean, if True, the particles belonging to each halo are also saved
            - part_save_radius: the radius (in units of Rvir) within which the particles are saved
            - particle_catalog: the StePS_Particle_Catalog containing the particles of the simulation
            - kdtree: the scipy.spatial.cKDTree built from the particle_catalog
            - precision: 0 for 32bit float, 1 for 64bit float
        '''
        if self.Nhalos >= 1:
            if int(precision) == 0:
                HDF5datatype = 'float32'
                npdatatype = np.float32
                print(" using 32bit precision.")
            if int(precision) == 1:
                HDF5datatype = 'double'
                npdatatype = np.float64
                print(" using 64bit precision.")
            HDF5_snapshot = h5py.File(filename+".hdf5", "w")
            #generating the header
            header_group = HDF5_snapshot.create_group("/Header")
            header_group.attrs['HaloFinder'] = "StePS_HF.py"
            header_group.attrs['HaloFinderVersion'] = _VERSION
            now = datetime.now()
            header_group.attrs['Date'] = now.strftime("%d/%m/%Y %H:%M")
            header_group.attrs['NumHalos'] = np.uint32(self.Nhalos)
            header_group.attrs['SaveParticles'] = bool(save_particles)
            if save_particles:
                header_group.attrs['ParticleSaveRadius'] = np.float32(part_save_radius)
            for key in self.Header.keys():
                if key != "MassDefinitions":
                    header_group.attrs[key] = self.Header[key]
                else:
                    mdeflist = []
                    mdeflist.append(self.Header[key][0])
                    for i in range(0,len(self.Header[key][1])):
                        mdeflist.append(self.Header[key][1][i])
                    header_group.attrs[key] = mdeflist
            #Header created.
            #Creating datasets for the halo data
            halo_group = HDF5_snapshot.create_group("/Halos")
            for key in self.DataTable[0].keys():
                if type(self.DataTable[0][key]) == np.ndarray:
                    #Vector quantity
                    dataset = halo_group.create_dataset(key, (self.Nhalos,3),dtype=HDF5datatype)
                    outarray = np.zeros((self.Nhalos,3),dtype=npdatatype)
                    for i in range(0,self.Nhalos):
                        if self.Header["HindependentUnits"]:
                            if key == "Coordinates":
                                outarray[i] = npdatatype(self.DataTable[i][key]* self.h)
                            elif key[0]=="J":
                                outarray[i] = npdatatype(self.DataTable[i][key]* self.h**2)
                            else:
                                outarray[i] = npdatatype(self.DataTable[i][key])
                        else:
                            outarray[i] = npdatatype(self.DataTable[i][key])
                    dataset[:,:] = outarray
                elif type(self.DataTable[0][key]) == int or type(self.DataTable[0][key]) == np.uint32 or type(self.DataTable[0][key]) == np.uint64:
                    #ID or particle number
                    dataset = halo_group.create_dataset(key, (self.Nhalos,),dtype='uint32')
                    outarray = np.zeros((self.Nhalos),dtype=np.uint32)
                    for i in range(0,self.Nhalos):
                        outarray[i] = npdatatype(self.DataTable[i][key])
                    dataset[:] = outarray
                else:
                    #everything else. the data type should be scalar float in this case.
                    if key == "T/|U|":
                        dataset = halo_group.create_dataset("TperAbsU", (self.Nhalos,),dtype=HDF5datatype)
                    else:
                        dataset = halo_group.create_dataset(key, (self.Nhalos,),dtype=HDF5datatype)
                    outarray = np.zeros(self.Nhalos,dtype=npdatatype)
                    for i in range(0,self.Nhalos):
                        outarray[i] = npdatatype(self.DataTable[i][key])
                    if self.Header["HindependentUnits"]:
                        if key[0]=="M" or key[0]=="R" or key=="Energy":
                            dataset[:] = outarray * self.h
                        elif key == "VolResolved":
                            dataset[:] = outarray * (self.h**3)
                        else:
                            dataset[:] = outarray
                    else:
                        dataset[:] = outarray
            if save_particles:
                if particle_catalog is None:
                    raise Exception("Error: particle catalog is not provided, cannot save particle data.")
                if kdtree is None:
                    raise Exception("Error: kd-tree is not provided, cannot save particle data.")
                print("Saving particle data within %.2f x Rvir..." % (part_save_radius))
                save_start = time.time()
                #Creating datasets for the particle data
                particle_root_group = HDF5_snapshot.create_group("/Particles")
                for i in range(0,self.Nhalos):
                    #creating a group for storing the halo particles
                    haloid = self.DataTable[i]["ID"]
                    #print("\tSaving particles for halo ID %i (%i / %i)" % (haloid, i+1, self.Nhalos), end="\r")
                    halo_part_group = particle_root_group.create_group("/Particles/Halo_%i" % haloid)
                    #selecting the particles belonging to this halo.
                    Rvir = self.DataTable[i]["Rvir"]/1000.0 # converting Rvir from kpc to Mpc
                    halo_center = self.DataTable[i]["Coordinates"]
                    # searching the particles within the given radius
                    part_indices = kdtree.query_ball_point(halo_center, part_save_radius * Rvir)
                    Npart_halo = len(part_indices)
                    if Npart_halo < 1:
                        # this should not happen, but just in case
                        raise Exception("Error: no particles found for halo ID %i within %f x Rvir = %f Mpc." % (haloid, part_save_radius, part_save_radius * Rvir))
                    # creating datasets for the particle properties
                    coord_dataset = halo_part_group.create_dataset("Coordinates", (Npart_halo,3),dtype=HDF5datatype)
                    vel_dataset = halo_part_group.create_dataset("Velocities", (Npart_halo,3),dtype=HDF5datatype)
                    mass_dataset = halo_part_group.create_dataset("Masses", (Npart_halo,),dtype=HDF5datatype)
                    id_dataset = halo_part_group.create_dataset("IDs", (Npart_halo,),dtype='uint64')
                    # filling the datasets with the correctly sized selection
                    coord_dataset[:] = np.array((particle_catalog.Coordinates[part_indices]-halo_center),dtype=npdatatype) # centering the coordinates on the halo center
                    if self.Header["Boundaries"] == "PERIODIC":
                        coord_dataset[:] = np.mod(coord_dataset[:]+0.5*self.Header["Lbox"], self.Header["Lbox"])-0.5*self.Header["Lbox"] # applying periodic boundary conditions
                    elif self.Header["Boundaries"] == "CYLINDRICAL":
                        coord_dataset[:,2] = np.mod(coord_dataset[:,2]+0.5*self.Header["Lbox"], self.Header["Lbox"])-0.5*self.Header["Lbox"] # applying periodic boundary conditions in the z direction
                    if self.Header["HindependentUnits"]:
                        coord_dataset[:] = coord_dataset[:] * self.h # converting to comoving Mpc/h
                    vel_dataset[:] = np.array(particle_catalog.Velocities[part_indices],dtype=npdatatype)
                    mass_dataset[:] = np.array(particle_catalog.Masses[part_indices],dtype=npdatatype)
                    if self.Header["HindependentUnits"]:
                        mass_dataset[:] = mass_dataset[:] * self.h # converting to Msol/h
                    id_dataset[:] = np.array(particle_catalog.IDs[part_indices],dtype=np.uint64)

                print("Particle data saved in %f seconds." % (time.time()-save_start))
            HDF5_snapshot.close()
        else:
            print("The halo catalog is empty. No file is saved.");
        return



#Beginning of the script

#initializing MPI
comm = MPI.COMM_WORLD
size = comm.Get_size() #total number of MPI threads
rank = comm.Get_rank() #rank of this MPI thread

#parameters that are used in all threads
Params = None
ERROR = 0
VERBOSE = False
rho_c = None
rho_b = None
Delta_c = None
Nmassdef = None
massdefdenstable = None
p = None
redshift = None
r_central = None
r_master = None
Hubble_now = None
dist_unitMpc = None
dist_unitkpc = None
mass_unit = None
L_box = None
save_particles = False
part_save_radius = 1.0

#Welcome message
if rank == 0:
    print("\n+-----------------------------------------------------------------------------------------------+\n|StePS_HF.py %s\t\t\t\t\t\t\t\t\t\t|\n| (STEreographically Projected cosmological Simulations Halo Finder)\t\t\t\t|\n+-----------------------------------------------------------------------------------------------+\n| Copyright (C) %s %s\t\t\t\t\t\t\t\t|\n|\tDepartment of Physics, University of Helsinki | Helsinki, Finland\t\t\t|\n|\tJet Propulsion Laboratory, California Institute of Technology | Pasadena, CA, USA\t|\n+-----------------------------------------------------------------------------------------------+\n"%(_VERSION, _YEAR, _AUTHOR))
    if len(sys.argv) != 2 and len(sys.argv) != 3:
        print("Error: missing yaml file!")
        print("usage: ./StePS_HF.py <input yaml file>\nExiting.")
        ERROR = 1
    if len(sys.argv) == 3:
        if sys.argv[2] == "--verbose":
            VERBOSE = True
        else:
            print("Error: unknown command line argument %s\nExiting." % str(sys.argv[2]))
            ERROR = 1
    start = time.time()
ERROR = comm.bcast(ERROR, root=0)
if ERROR>0:
    sys.exit(ERROR)
if rank == 0:
    print("Reading the %s paramfile...\n" % str(sys.argv[1]))
    try:
        document = open(str(sys.argv[1]))
        Params = yaml.safe_load(document)
    except:
        print("Error: invalid input; cannot read yaml file\nExiting.")
        ERROR = 2
ERROR = comm.bcast(ERROR, root=0)
if ERROR>0:
    sys.exit(ERROR)

#Bcasting the loaded parameters
Params = comm.bcast(Params, root=0)
min_mass_force_res = np.double(Params['PARTICLE_RADII'])
npartmin = int(Params["NPARTMIN"])
search_radius = np.double(Params["SEARCH_RADIUS"])
if Params["H_INDEPENDENT_UNITS"]:
    search_radius /= (Params["H0"]/100.0) #converting the input comoving search radius from Mpc/h to Mpc
kdworkers = int(Params["KDWORKERS"])
# parameters of the parallelisation
delta_theta = np.double(Params["DELTA_THETA"])/180.0*np.pi #RAD Thickness of rangential coordinate in which duplicates are searched
d_r = np.double(Params["DELTA_R"]) #thicknes of the shell in which duplicates are searched in Mpc units
if "SAVEPARTICLES" in Params:
    save_particles = bool(Params["SAVEPARTICLES"])
    if Params["OUTFORMAT"] == "ASCII" and save_particles:
        if rank == 0:
            print("Warning: particle saving is only supported for HDF5 output format. Setting SAVEPARTICLES to False for ASCII output.\n")
        save_particles = False
if "PARTICLE_SAVE_RADIUS" in Params:
    part_save_radius = np.double(Params["PARTICLE_SAVE_RADIUS"])


if rank == 0:
    # checking the input parameters
    if Params['BOUNDARIES'] not in ["STEPS", "PERIODIC", "CYLINDRICAL"]:
        print("Error: unknown boundary condition %s\nExiting." % Params['BOUNDARIES'])
        ERROR = 3
    if Params['OUTFORMAT'] not in ["ASCII", "HDF5", "BOTH"]:
        print("Error: unknown output format %s\nExiting." % Params['OUTFORMAT'])
        ERROR = 3
    if Params['INITIAL_DENSITY_MODE'] not in ["Voronoi", "Nth neighbor"]:
        print("Error: unknown initial density mode %s\nExiting." % Params['INITIAL_DENSITY_MODE'])
        ERROR = 3
    if Params['CENTERMODE'] not in ["CENTRALPARTICLE", "CENTEROFMASSNPARTMIN"]:
        print("Error: unknown center mode %s\nExiting." % Params['CENTERMODE'])
        ERROR = 3
    # checking if the input file exists
    if not exists(Params['INFILE']):
        print("Error: input file %s does not exist!\nExiting." % Params['INFILE'])
        ERROR = 2
    # setting the redshift (and cosmological parameters, if the input in HDF5 format)
    if Params['H_INDEPENDENT_UNITS']:
        dist_unitMpc = "Mpc/h"
        dist_unitkpc = "kpc/h"
        mass_unit = "Msol/h"
    else:
        dist_unitMpc = "Mpc"
        dist_unitkpc = "kpc"
        mass_unit = "Msol"
    if Params['INFILE'][-4:] == 'hdf5':
        redshift, Omega_m, Omega_l, H0, Npart = Load_params_from_HDF5_snap(Params['INFILE'])
    else:
        redshift = np.double(Params['REDSHIFT'])
        Omega_m = np.double(Params['OMEGAM'])
        Omega_l = np.double(Params['OMEGAL'])
        H0 = np.double(Params['H0'])
    print("Cosmological Parameters:\n------------------------\n\u03A9_m:\t\t\t%f\t(Ommh2=%f; Omch2=%f)\n\u03A9_lambda:\t\t%f\n\u03A9_k:\t\t\t%f\n\u03A9_b:\t\t\t%f\t(Ombh2=%f)\nH0:\t\t\t%f km/s/Mpc\nDark energy model:\t%s" % (Omega_m, Omega_m * (Params['H0']/100.0)**2, (Omega_m - Params['OMEGAB']) * (Params['H0']/100.0)**2, Omega_l, 1.0-Omega_m-Omega_l, Params['OMEGAB'], (Params['OMEGAB']) * (Params['H0']/100.0)**2, Params['H0'], Params['DARKENERGYMODEL']))
    # checking the dark energy model
    if Params['DARKENERGYMODEL'] == 'Lambda':
        print("\t\t\t(w = -1)")
    elif Params['DARKENERGYMODEL'] == 'w0':
        print("\t\t\tw = %f" % Params['DARKENERGYPARAMS'][0])
    elif Params['DARKENERGYMODEL'] == 'CPL':
        print("\t\t\tw0 = %f\n\t\t\twa = %f" % (Params['DARKENERGYPARAMS'][0], Params['DARKENERGYPARAMS'][1]))
    else:
        print("Error: unkown dark energy parametrization!\nExiting.\n")
        ERROR=3
ERROR = comm.bcast(ERROR, root=0)
if ERROR>0:
    sys.exit(ERROR)
dist_unitMpc = comm.bcast(dist_unitMpc, root=0)
mass_unit = comm.bcast(mass_unit, root=0)
dist_unitkpc = comm.bcast(dist_unitkpc, root=0)

if rank == 0:
    # Calculating relevant cosmological quantities
    Hubble_now = Hz(redshift, H0, Omega_m, Omega_l, Params["DARKENERGYMODEL"], Params["DARKENERGYPARAMS"])
    rho_c = H2RHO_C*(Hubble_now/100.0)**2/(redshift+1)**3 #comoving critical density in internal units (G=1)
    rho_b = (Params['H0']/100.0)**2 * H2RHO_C * Omega_m #background density in internal units (G=1) [the comoving background density is redshift independent]
    print("\u03C1_c (comoving):\t\t%.6e Msol/Mpc^3 (=%.6e Msol*h^2/Mpc^3)\n\u03C1_b (comoving):\t\t%.6e Msol/Mpc^3 (=%.6e Msol*h^2/Mpc^3)\n" % (rho_c*1e11, rho_c*1e11/(H0/100.0)**2, rho_b*1e11, rho_b*1e11/(H0/100.0)**2))
    Delta_c = get_Delta_c(redshift, H0, Omega_m, Omega_l, Params["DARKENERGYMODEL"], Params["DARKENERGYPARAMS"]) #Virial overdensity constant
    print("Snapshot Parameters:\n"
            "--------------------")
    if Params["BOUNDARIES"] == "STEPS":
        print("Topology:\t\tSpherical\n"
              "Radius:\t\t\t%.6g %s"
              % (
                    np.double(Params['RSIM']), dist_unitMpc
                )
              )
    elif Params["BOUNDARIES"] == "PERIODIC":
        L_box = np.double(Params['LBOX'])
        print("Topology:\t\tPeriodic [3-torus]\n"
              "Boxsize:\t\t%.6g %s"
              % (
                    np.double(Params['LBOX']), dist_unitMpc
                )
              )
        if Params["H_INDEPENDENT_UNITS"]:
            L_box /= np.double(Params['H0'])/100.0
        if Params["INITIAL_DENSITY_MODE"] == "Voronoi":
            print("Error: Voronoi density estimation is not supported with periodic boundaries.\nExiting.")
            ERROR = 5
    elif Params["BOUNDARIES"] == "CYLINDRICAL":
        L_box = np.double(Params['LBOX'])
        print("Topology:\t\tCylindrical\n"
            "Boxsize:\t\t%.6g %s\n"
            "Radius:\t\t\t%.6g %s"
            % (
                np.double(Params['LBOX']), dist_unitMpc,
                np.double(Params['RSIM']), dist_unitMpc,
            )
        )
    else:
        print("Error: Unrecognized boundary conditions %s.\nExiting." % Params["BOUNDARIES"])
        ERROR = 3
    print("Redshift:\t\t%.4f\n"
          "H(z):\t\t\t%.4f km/s/Mpc\n"
          "Softening length:\t%.4g %s\n"
          "Distance units:\t\t%.2g %s\n"
          "Velocity units:\t\t%.2g km/s\n"
          "Mass units:\t\t%.2g %s\n"
          % (
              redshift,
              Hubble_now,
              min_mass_force_res*1000.0,dist_unitkpc,
              np.double(Params['UNIT_D_IN_MPC']), dist_unitMpc,
              np.double(Params['UNIT_V_IN_KMPS']),
              np.double(Params['UNIT_M_IN_MSOL']), mass_unit
              )
    )
    print("Halo Finder Parameters:\n-----------------------\nHalo catalog file:\t\t\t%s\nInitial Density Estimation:\t\t%s\nSearch radius:\t\t\t\t%.2f %s\nNumber of KDTree worker threads:\t%i\nMinimal particle number:\t\t%i\nHalo center mode:\t\t\t%s" %(Params["OUTFILE"],Params["INITIAL_DENSITY_MODE"],np.double(Params["SEARCH_RADIUS"]), dist_unitMpc, int(Params["KDWORKERS"]), int(Params["NPARTMIN"]), Params["CENTERMODE"] ))
    if Params["BOUNDONLYMODE"]:
        print("Spherical Overdensity Mode:\t\tBound Only (BO)")
        print("Unbound threshold:\t\t\t%.2f" % Params["UNBOUND_THRESHOLD"])
    else:
        print("Spherical Overdensity Mode:\t\tStrict Spherical Overdensity (SO)")
    print("Save halo particles:\t\t\t%r" % save_particles)
    if save_particles:
        print("Particle save radius:\t\t\t(%.2f x Rvir)" % (part_save_radius))
    print("Mass definitions:")
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
            print("Error: Unrecognized mass definition %s.\nExiting." % Params["MASSDEF"][i])
            ERROR=4
    if size>1 and Params["BOUNDARIES"] == "PERIODIC":
        print("Error: periodic boundary conditions are not supported with MPI parallelization yet.\nExiting.")
        ERROR=5
    if VERBOSE:
        print("\nMass definition density levels (comoving):\n------------------------------------------\nVirial:\t\t%.6e Msol/Mpc^3 (=%.6e Msol/h^2/Mpc^3)" % (massdefdenstable[0]*1e11, massdefdenstable[0]*1e11/(H0/100.0)**2))
        for i in range(1,Nmassdef):
            print("%s:\t\t%.6e Msol/Mpc^3 (=%.6e Msol/h^2/Mpc^3)" % (Params["MASSDEF"][i-1],massdefdenstable[i]*1e11, massdefdenstable[i]*1e11/(H0/100.0)**2))
        print("------------------------------------------\n")
ERROR = comm.bcast(ERROR, root=0)
if ERROR>0:
    sys.exit(ERROR)
else:
    #Bcasting calculated quantities
    rho_c = comm.bcast(rho_c, root=0)
    rho_b = comm.bcast(rho_b, root=0)
    Delta_c = comm.bcast(Delta_c, root=0)
    Nmassdef = comm.bcast(Nmassdef, root=0)
    massdefdenstable = comm.bcast(massdefdenstable, root=0)
    redshift = comm.bcast(redshift, root=0)
    Hubble_now = comm.bcast(Hubble_now, root=0)
    if Params["BOUNDARIES"] == "PERIODIC" or Params["BOUNDARIES"] == "CYLINDRICAL":
        L_box = comm.bcast(L_box, root=0)

# Setting the parameters for the halo finder
DensMode = Params["INITIAL_DENSITY_MODE"]
if Params["BOUNDONLYMODE"]:
    UnboudThreshold = np.double(Params["UNBOUND_THRESHOLD"])
else:
    UnboudThreshold = 0.0
sys.stdout.flush()
if rank == 0:
    # Loading the input particle snapshot
    p = StePS_Particle_Catalog(Params['INFILE'], np.double(Params['UNIT_D_IN_MPC']), np.double(Params['UNIT_V_IN_KMPS']), np.double(Params['UNIT_M_IN_MSOL']), Params["H_INDEPENDENT_UNITS"],H0,REDSHIFT=np.double(Params['REDSHIFT']),FORCE_RES=min_mass_force_res, BOUNDARIES=Params["BOUNDARIES"], LBOX=np.double(Params['LBOX']), D_OVERLAP=np.double(Params['DELTA_R']))
    # Plotting some basic information about the snapshot
    if VERBOSE:
        print("Particle data:\n--------------\nNpart:\t\t\t%i\nMasses:\t\t\tmin: %.7f * 10^11 Msol\tmax: %.7f * 10^11 Msol\nCoordinates:\t\tmin: (%.4f,%.4f,%.4f) Mpc\tmax: (%.4f,%.4f,%.4f) Mpc\nVelocities:\t\tmin: (%.2f,%.2f,%.2f) km/s\tmax: (%.2f,%.2f,%.2f) km/s\n--------------\n" % (p.Npart,np.min(p.Masses), np.max(p.Masses), np.min(p.Coordinates[:,0]), np.min(p.Coordinates[:,1]), np.min(p.Coordinates[:,2]), np.max(p.Coordinates[:,0]), np.max(p.Coordinates[:,1]), np.max(p.Coordinates[:,2]), np.min(p.Velocities[:,0]), np.min(p.Velocities[:,1]), np.min(p.Velocities[:,2]), np.max(p.Velocities[:,0]), np.max(p.Velocities[:,1]), np.max(p.Velocities[:,2])))
    if Params["BOUNDARIES"] == "STEPS":
        # Estimating the central region of the simulation. This is the maximal distance of the 2xminimum-mass particles from the origin.
        r_central = np.max(np.sqrt(np.sum(np.power(p.Coordinates[p.Masses <= 2.0*np.min(p.Masses)],2.0),axis=1)))
        # Calculating the radius of the master thread's subvolume
        r_part = np.sqrt(np.sum(np.power(p.Coordinates[:,:],2.0),axis=1))
        if size==1:
            r_master = np.max(r_part)
        else:
            #sorting all particles in increasing order
            sorted_idx = r_part.argsort()
            Ncentral = np.uint64(0.3*(p.Npart / (size-1)))
            r_master = r_part[sorted_idx][Ncentral]
        print("Estimated radius of the high-res central region of the simulation: %.2f %s" % (r_central,dist_unitMpc))
        print("Radius of the subvolume of the master thread: %.2f %s" % (r_master,dist_unitMpc))
        if size>1:
            print("Thickness of the overlap region: %.3f %s" % (d_r,dist_unitMpc))
        if size>2:
            print("Angle of the tangential slices of the slave threads: %.2f degrees" % (360.0/(size-1)))
            print("Thickness of the tangential overlap region: %.2f degrees" % (180.0*delta_theta/np.pi))
        print("\n")
        # Building KDTree for a quick nearest-neighbor lookup
        tree = KDTree(p.Coordinates,leafsize=10, compact_nodes=True, balanced_tree=True, boxsize=None)
    elif Params["BOUNDARIES"] == "CYLINDRICAL":
        # Estimating the central region of the simulation. This is the maximal distance of the 2xminimum-mass particles from the origin.
        r_central = np.max(np.sqrt((np.power(p.Coordinates[p.Masses <= 2.0*np.min(p.Masses),0],2.0) + np.power(p.Coordinates[p.Masses <= 2.0*np.min(p.Masses),1],2.0))))
        # Calculating the radius of the master thread's subvolume
        r_part = np.sqrt(np.power(p.Coordinates[:,:2],2.0).sum(axis=1))
        if size==1:
            r_master = np.max(r_part)
        else:
            #sorting all particles in increasing order
            sorted_idx = r_part.argsort()
            Ncentral = np.uint64(0.5*(p.Npart / (size-1)))
            r_master = r_part[sorted_idx][Ncentral]
        print("Estimated radius of the high-res central region of the simulation: %.2f Mpc" % (r_central))
        print("Radius of the subvolume of the master thread: %.2f Mpc" % (r_master))
        if size>1:
            print("Thickness of the overlap region: %.3f Mpc" % (d_r))
        if size>2:
            print("Angle of the tangential slices of the slave threads: %.2f degrees" % (360.0/(size-1)))
            print("Thickness of the tangential overlap region: %.2f degrees" % (180.0*delta_theta/np.pi))
        print("\n")
        # Building KDTree for a quick nearest-neighbor lookup
        tree = KDTree(p.Coordinates,leafsize=10, compact_nodes=True, balanced_tree=True, boxsize=None)
    elif Params["BOUNDARIES"] == "PERIODIC":
        # Building KDTree for a quick nearest-neighbor lookup
        tree = KDTree(p.Coordinates,leafsize=10, compact_nodes=True, balanced_tree=True, boxsize=L_box)
    # Density reconstruction on the master thread
    if DensMode == "Voronoi":
        if Params["BOUNDARIES"] == "STEPS" or Params["BOUNDARIES"] == "CYLINDRICAL":
            # Calculating voronoi volumes:
            p.Density = p.Masses/voronoi_volumes(p.Coordinates)/rho_c
        else:
            print("Error: Voronoi density estimation is not implemented for fully periodic boundary conditions.\nExiting.")
            ERROR = 7
            sys.exit(ERROR)
    elif DensMode == "Nth neighbor":
        # searching for the distance of the Nth nearest neighbor for all particle
        print("Density reconstruction with %ith nearest neighbor method..."%npartmin)
        sys.stdout.flush()
        kd_start = time.time()
        for i in range(0,p.Npart):
            d,idx = tree.query(p.Coordinates[i,:], k=npartmin+1, workers=kdworkers)
            Mtot = np.sum(p.Masses[idx])
            p.Density[i] = ((Mtot)/(d[npartmin]**3))
        p.Density *= 1.0/(4.0*np.pi/3.0)/rho_c #the estimated density in units of rho_c
        kd_end = time.time()
        print("...done in %.2f s.\n" % (kd_end-kd_start))
    print("min(rho) = %g rho_c\nmax(rho) = %g rho_c\n" % (np.min(p.Density), np.max(p.Density)))

# Bcasting the particle data
# All MPI threads will have a complete copy of the particles. Since the typical spherical StePS snapshot is a few 100MB, this will not cause a large memory requirement
# On the other hand, this can cause large memory footprint for larger simulations (e.g.: cylindrical StePS simulations)
p = comm.bcast(p, root=0)
r_central = comm.bcast(r_central, root=0) # radius of the central region that will be analysed in the master thread
r_master = comm.bcast(r_master, root=0) # radius of the subvolume of the master thread
if rank > 0:
    if Params["BOUNDARIES"] == "STEPS" or Params["BOUNDARIES"] == "CYLINDRICAL":
        # Building KDTree for a quick nearest-neighbor lookup for every other thread.
        tree = KDTree(p.Coordinates,leafsize=10, compact_nodes=True, balanced_tree=True, boxsize=None)
    elif Params["BOUNDARIES"] == "PERIODIC":
        # Building KDTree for a quick nearest-neighbor lookup for every other thread.
        tree = KDTree(p.Coordinates,leafsize=10, compact_nodes=True, balanced_tree=True, boxsize=L_box)

if size>1:
    #Distributing the halo candidates evenly between the threads
    ParentThreadID = SetParentThreadIDs(p, r_master, d_r, delta_theta, size, rank, Params["BOUNDARIES"]) # this is a local variable for every thread
    #print("ParentThreadID: ", ParentThreadID)
else:
    ParentThreadID = np.zeros(p.Npart, dtype=np.int32)

# Identifying halos using Spherical Overdensity (SO) method
print("MPI Rank %i: Identifying halos and calculating halo parameters..." % rank)
sys.stdout.flush()
id_start = time.time()
if Params["BOUNDARIES"] == "STEPS":
    halos = StePS_Halo_Catalog(np.double(Params["H0"]),np.double(Hubble_now), np.double(Params["OMEGAM"]), np.double(Params["OMEGAL"]), Params["DARKENERGYMODEL"], Params["DARKENERGYPARAMS"], redshift, rho_c, rho_b, "Vir", Params["MASSDEF"], Params["CENTERMODE"], Params["INITIAL_DENSITY_MODE"], Params["NPARTMIN"], Params["BOUNDONLYMODE"], HindepUnis=Params["H_INDEPENDENT_UNITS"], Boundaries=Params["BOUNDARIES"], Rsim=Params["RSIM"])
elif Params["BOUNDARIES"] == "PERIODIC" or Params["BOUNDARIES"] == "CYLINDRICAL":
    halos = StePS_Halo_Catalog(np.double(Params["H0"]),np.double(Hubble_now), np.double(Params["OMEGAM"]), np.double(Params["OMEGAL"]), Params["DARKENERGYMODEL"], Params["DARKENERGYPARAMS"], redshift, rho_c, rho_b, "Vir", Params["MASSDEF"], Params["CENTERMODE"], Params["INITIAL_DENSITY_MODE"], Params["NPARTMIN"], Params["BOUNDONLYMODE"], HindepUnis=Params["H_INDEPENDENT_UNITS"], Boundaries=Params["BOUNDARIES"], Rsim=Params["RSIM"], Lbox=L_box)
else:
    print("Error: Unrecognized boundary conditions %s.\nExiting." % Params["BOUNDARIES"])
    ERROR = 3
    sys.exit(ERROR)
halo_ID = 0
while True:
    #selecting the largest density particle with parentID=-1
    idx = p.IDs[np.logical_and(p.HaloParentIDs == -1, ParentThreadID == rank)][np.argmax(p.Density[np.logical_and(p.HaloParentIDs == -1, ParentThreadID == rank)])]
    maxdens = p.Density[idx]
    if maxdens>0.5*Delta_c:
        #Query the kd-tree for nearest neighbors.
        if search_radius <= np.cbrt(p.Masses[idx]/rho_b):
            #if the search radius is smaller than the typical particle separation, increase it
            this_search_radius = np.cbrt(p.Masses[idx]/rho_b)
        else:
            this_search_radius = search_radius
        halo_particleindexes = tree.query_ball_point(p.Coordinates[idx], this_search_radius, p=2.0, eps=0, workers=kdworkers, return_sorted=False)
        if len(halo_particleindexes) >= npartmin:
            #print("\nCentral estimated density for halo #%i: %.2f \u03C1_c" % (halo_ID, maxdens))
            #print("\tCentral coordinate of halo #%i: " % (halo_ID), p.Coordinates[idx])
            #print("\tID of the central particle of halo #%i: %i" % (halo_ID,idx))
            #print("\tSearch radius for halo #%i: %.2fMpc/h" % (halo_ID, search_radius))
            #print("\tNumber of particles in the search radius of halo #%i:" % (halo_ID),len(halo_particleindexes))
            if Params["BOUNDARIES"] == "STEPS":
                halo_params = calculate_halo_params(p, idx, halo_particleindexes, halo_ID, Params["MASSDEF"], massdefdenstable, npartmin, Params["CENTERMODE"],np.double(Hubble_now),boundonly=Params["BOUNDONLYMODE"], unbound_threshold=UnboudThreshold, rho_b=rho_b,boundaries=Params["BOUNDARIES"])
            elif Params["BOUNDARIES"] == "PERIODIC" or Params["BOUNDARIES"] == "CYLINDRICAL":
                halo_params = calculate_halo_params(p, idx, halo_particleindexes, halo_ID, Params["MASSDEF"], massdefdenstable, npartmin, Params["CENTERMODE"],np.double(Hubble_now),boundonly=Params["BOUNDONLYMODE"], unbound_threshold=UnboudThreshold, rho_b=rho_b,boundaries=Params["BOUNDARIES"],Lbox=L_box)
            if halo_params != None:
                halos.add_halo(halo_params, maxdens) #adding the identified halo to the catalog
                halo_ID +=1
                if VERBOSE:
                    print("MPI Rank %i: Halo #%i added to the (local) catalog. (Npart = %i, Mvir=%e Msol, Rvir=%.2f kpc)" % (rank, halo_ID-1, halos.DataTable[halo_ID-1]["Npart"], halos.DataTable[halo_ID-1]["Mvir"], halos.DataTable[halo_ID-1]["Rvir"]))
            #else:
            #    print("This candidate didn't had enough partilces.")
    else:
        #print("Central estimated density for the last halo candidate #%i: %.2f \u03C1_c" % (halo_ID, maxdens))
        #This means that in the center, we did not reach Delta_c*rho_c.
        #After this, we will not find new halos.
        print("MPI Rank %i: Total number of identified halos: "%rank, halos.Nhalos)
        break;
id_end = time.time()
print("MPI Rank %i: ...done in %.2f s.\n" % (rank, id_end-id_start))
sys.stdout.flush()
#collecting all catalog into the main thread
if rank == 0:
    print("MPI Rank %i: Collecting and merging all calculated catalogs to the master thread."%rank)
    if Params["BOUNDARIES"] == "STEPS":
        halos_final = StePS_Halo_Catalog(np.double(Params["H0"]), np.double(Hubble_now), np.double(Params["OMEGAM"]), np.double(Params["OMEGAL"]), Params["DARKENERGYMODEL"], Params["DARKENERGYPARAMS"], redshift, rho_c, rho_b, "Vir", Params["MASSDEF"], Params["CENTERMODE"], Params["INITIAL_DENSITY_MODE"], Params["NPARTMIN"], Params["BOUNDONLYMODE"], HindepUnis=Params["H_INDEPENDENT_UNITS"], Boundaries=Params["BOUNDARIES"], Rsim=Params["RSIM"])
    elif Params["BOUNDARIES"] == "PERIODIC" or Params["BOUNDARIES"] == "CYLINDRICAL":
        halos_final = StePS_Halo_Catalog(np.double(Params["H0"]), np.double(Hubble_now), np.double(Params["OMEGAM"]), np.double(Params["OMEGAL"]), Params["DARKENERGYMODEL"], Params["DARKENERGYPARAMS"], redshift, rho_c, rho_b, "Vir", Params["MASSDEF"], Params["CENTERMODE"], Params["INITIAL_DENSITY_MODE"], Params["NPARTMIN"], Params["BOUNDONLYMODE"], HindepUnis=Params["H_INDEPENDENT_UNITS"], Boundaries=Params["BOUNDARIES"], Rsim=Params["RSIM"], Lbox=L_box)
    # adding the catalog of the master thread to the final catalog
    if size > 1:
        halos_final.add_catalog(halos,overlaps=True, MPI_parent_thread=0, boundaries=Params["BOUNDARIES"], Lbox=L_box)
    else:
        halos_final.add_catalog(halos,overlaps=False, MPI_parent_thread=0, boundaries=Params["BOUNDARIES"], Lbox=L_box)
#collecting all remaining halos from the slave threads
if size > 1:
    if rank != 0:
        comm.send(halos, 0, tag=rank)
    else:
        for i in range(1,size):
            print("MPI Rank %i: Receiving halos from Rank %i."%(rank,i))
            halos = comm.recv(buf=None, source=i, tag=i)
            halos_final.add_catalog(halos,overlaps=True,MPI_parent_thread=i, boundaries=Params["BOUNDARIES"], Lbox=L_box)
        print("MPI Rank %i: Halo catalogs merged."%(rank))
        print("MPI Rank %i: The total size of the generated halo catalog is: %i\n"%(rank,halos_final.Nhalos))

if rank == 0:
    if halos.Nhalos > 0:
        if Params["OUTFORMAT"] == "ASCII" or Params["OUTFORMAT"] == "BOTH":
            print("Saving the generated catalog to %s.dat" % Params["OUTFILE"])
            halos_final.save_ascii_catalog(Params["OUTFILE"])
        if Params["OUTFORMAT"] == "HDF5" or Params["OUTFORMAT"] == "BOTH":
            print("Saving the generated catalog to %s.hdf5" % Params["OUTFILE"], end='')
            halos_final.save_hdf5_catalog(Params["OUTFILE"],save_particles=save_particles, part_save_radius=part_save_radius, particle_catalog=p, kdtree=tree)
    end = time.time()
    if Params["BOUNDONLYMODE"]:
        print("\nBO halo finding finished under %fs.\n" % (end-start))
    else:
        print("\nSO halo finding finished under %fs.\n" % (end-start))

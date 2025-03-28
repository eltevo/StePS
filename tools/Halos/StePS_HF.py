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
from scipy.spatial import Voronoi, ConvexHull, KDTree
from scipy.optimize import fsolve, curve_fit
import astropy.units as u
from astropy.cosmology import LambdaCDM, wCDM, w0waCDM, z_at_value
from inputoutput import *

_VERSION="v0.2.2.0"
_YEAR="2024-2025"

# Global variables (constants)
G  = 4.3009172706e-9 # gravitational constant G in Mpc/Msol*(km/s)^2 units
# usual StePS internal units
UNIT_T=47.14829951063323 #Unit time in Gy
UNIT_V=20.738652969925447 #Unit velocity in km/s
UNIT_D=3.0856775814671917e24 #=1Mpc Unit distance in cm

# Defining functions
def get_periodic_distance_vec(Coords1,Coords2,Lbox):
    return np.mod(Coords1 - Coords2 + Lbox / 2, Lbox) - Lbox / 2

def get_periodic_distances(Coords1,Coords2,Lbox):
    return np.sqrt(np.sum(np.power(np.mod(Coords1 - Coords2 + Lbox / 2, Lbox) - Lbox / 2, 2), axis=1))

def voronoi_volumes(points, SILENT=False):
    """
    Function for calculating voronoi volumes
    Input:
        - points: array containing the coordinates of the input particles
    Returns:
        - vol: array containing the volumes of all cells
    """
    if SILENT==False:
        v_start = time.time()
        print("Calculating Voronoi tessellation...")
        sys.stdout.flush()
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

def get_center_of_mass(r, m, boundaries="StePS", boxsize=0):
    """
    Function for calculating the center of mass of a particle system
    Input:
        - r: particle coordinates
        - m: particle masses
        - boundaries: boundary condition. Must be "StePS" or "Periodic"
        - boxsize: linear box size in the same units as the coorinates
    Returns:
        - Center of mass coordinates
    """
    if boundaries=="StePS":
        return np.sum((r.T*m).T,axis=0)/np.sum(m)
    elif boundaries=="Periodic":
        #assuming that the box size is significantly larger than the halo size (moving the first particle to the center)
        dr = boxsize/2 - r[0,:]
        r =  np.mod(r+dr,boxsize)
        com = np.sum((r.T*m).T,axis=0)/np.sum(m)
        return com-dr# shifting the center back to the original position and returning the center of mass

    else:
        raise Exception("Error: unknown boundary condition %s." % (boundaries))

def get_angular_momentum(r,v,m):
    """
    Function for calculating the angular momentum of a particle system.
    Assumed input:
        - r: CoM (physical) coordinates.
        - v: (physical) velocities.
        - m: particle masses.
    Returns:
        - J: angular momentum vector
    """
    p = m.reshape((len(m),1))*v #Individual linear momenta: mass x position
    J = np.cross(r,p)# Individual orbital angular momenta" (position vector) x (linear momentum)
    return np.sum(J,axis=0) #returning the total angular momentum vector

def cubic_spline_potential(r, h):
    """
    Function for calculating the potential of Cubic spline kernel (Monaghan & Lattanzio, 1985)
    Input:
        - r: distance from the particle
        - h: softening length
    Returns:
        - cubic spline potential value (assuming unit masses and G=1 units)
    """
    q = r / h
    kernel_value = np.zeros_like(q)
    # Define the kernel function for the cubic spline softening
    mask = np.logical_and(q >= 0, q < 0.5)
    kernel_value[mask] = 32.0*r[mask]**5/(15.0*h[mask]**6) - 9.6*r[mask]**4/h[mask]**5 + 16.0*r[mask]**2/(3.0*h[mask]**3) - 8.0/(3.0*h[mask])
    mask = np.logical_and(q >= 0.5, q < 1.0)
    kernel_value[mask] = -32.0*r[mask]**5/(15.0*h[mask]**6) + 9.6*r[mask]**4/h[mask]**5 - 16.0*r[mask]**3/h[mask]**4+32.0*r[mask]**2/(3.0*h[mask]**3) + 1/(15.0*r[mask]) - 16.0/(5.0*h[mask])
    mask = q >= 1.0
    kernel_value[mask] = -1/r[mask]
    return kernel_value

def get_individual_energy(r,v,m,force_res,boundary="STEPS",boxsize=0.0):
    """
    Function for calculating the individual energy of a particle system by using direct summation of the potential.
    Expected input:
        - r: CoM (physical) coordinates in Mpc.
        - v: (physical) velocities in km/s.
        - m: particle masses in Msol.
        - force_res: (physical) softening length of each particle
    Returns:
        - Ekin: Kinetic energy in (Msol * (km/s)^2 ) units
        - Epot: Potential energy in (Msol * (km/s)^2 ) units
    """
    Nparticle = len(m) #number of input particles
    Ekin = 0.5*m*np.sum(v**2,axis=1) # kinetic energy of the individual particles (Ekin = 0.5*m*v^2)
    #calculating the potential energy
    Epot = np.zeros(Nparticle,dtype=np.double)
    for i in range(0,Nparticle):
        idx = np.where(np.arange(0,Nparticle)!=i)
        if boundary == "STEPS":
            dist = np.sqrt(np.sum(( r[idx] - r[i])**2, axis=1))
        elif boundary == "PERIODIC":
            dist = get_periodic_distances(r[idx], r[i], boxsize)
        else:
            raise Exception("Error:")
        Epot[i] += m[i]*np.sum(m[idx]*cubic_spline_potential(dist,force_res[idx]+force_res[i]))
    Epot *= G
    return Ekin, Epot

def get_total_energy(r,v,m,force_res,boundary="STEPS",boxsize=0.0):
        """
        Function for calculating the total energy of a particle system.
        Expected input:
            - r: CoM (physical) coordinates in Mpc.
            - v: (physical) velocities in km/s.
            - m: particle masses in Msol.
            - force_res: (physical) softening length of each particle
        Returns:
            - TotE: Total energy in of the system in (Msol * (km/s)^2 ) units
            - TotEkin: Total kinetic energy of the system in (Msol * (km/s)^2 ) units
            - TotEpot: Total potential energy of the system in (Msol * (km/s)^2 ) units
        """
        Ekin,Epot = get_individual_energy(r,v,m,force_res,boundary=boundary,boxsize=boxsize)
        TotEkin = np.sum(Ekin) #total kinetic energy of the halo
        TotEpot = np.sum(Epot) #total potential energy of the halo
        return TotEkin+TotEpot, TotEkin, TotEpot

def get_Rs_Klypin(vmax,v200,R200):
    """
    Function for calculatin the c concentration and Rs scale length based on Klypin Vmax method.
    Input:
        - vmax: maximal circular velocity of the halo
        - v200: circular velocity at the virial radius
        - R200: virial radius
    Returns:
        - c: Rvir/Rs concentration of the halo
        - Rs: Scale radius of the halo
    Details:
        -> Francisco Prada, Anatoly A. Klypin, Antonio J. Cuesta, Juan E. Betancort-Rijo, Joel Primack (2012) https://academic.oup.com/mnras/article/423/4/3018/987360
        -> Klypin, Anatoly A. ; Trujillo-Gomez, Sebastian ; Primack, Joel (2011) https://ui.adsabs.harvard.edu/abs/2011ApJ...740..102K/abstract
        -> Klypin, Anatoly ; Kravtsov, Andrey V. ; Bullock, James S. ; Primack, Joel R. (2001) https://ui.adsabs.harvard.edu/abs/2001ApJ...554..903K/abstract
    """
    vmaxperv200sqr = (vmax/v200)**2
    #using a polynomial fit of the solution to quickly estimate the initial guess
    p=np.poly1d([ 3.02509213e-03, -1.20296527e-01, 2.34770468e+00, 2.07672808e+01, -2.43183200e+01, 5.44330487e+00])
    init_guess = p(vmax/v200)
    def f_klypin(x):
        return 0.216*x/(np.log(1+x)-x/(1+x))-vmaxperv200sqr
    c = fsolve(f_klypin,init_guess)[0] # numerically solving the transcendental equation above
    Rs = R200/c
    return c, Rs

def get_bullock_spin(jvir,mvir,rvir):
    """
    Function for calculating Bullock spin parameter
    Expected input:
        - jvir: length of the total angular momentum vector within a virilized sphere in  Msun * Mpc * km/s
        - mvir: virial mass in Msol
        - rvir: virial radius in physical Mpc
    """
    vvir = np.sqrt(G*mvir/rvir) #circular velocity at the virial radius [Vvir^2 = G*Mvir/Rvir] (physical km/s)
    S_Bullock = jvir/ (np.sqrt(2) * mvir * rvir * vvir)
    return S_Bullock

def NFW_profile(r,rho0,Rs):
    """
    A function for calculating the Navarro-Frenk-White profile
    Input:
        - r: distance from the center
        - rho0: density parameter of the NFW profile
        - Rs: scale lenght of the halo
    Returns:
        - NFW profile values at "r" input distances
    """
    rpRs = r/Rs
    return rho0/(rpRs*np.power((1.0+rpRs),2))

def Hz(z, H0, Om, Ol, DE_model, DE_params):
    """
    Hubble parameter at given redshift using astropy.
    Input:
        - z: redshift
        - H0: Hubble constant in km/s/Mpc units
        - Om: non-relativistic matter density parameter
        - Ol: dark energy density parameter
        - DE_model: Dark Energy model name. must be "Lambda", "w0", or "CPL"
        - DE_params: list of the parameters of the DE model.
            -> "Lambda": not used
            -> "w0": [w0]
            -> "CPL": [w0,wa]
    Returns:
        - Hz: Hubble parameter at z redshift in km/s/Mpc units
    """
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
    Virial overdensity constant calculation, see eq 6. of https://ui.adsabs.harvard.edu/abs/1998ApJ...495...80B/abstract
    (Assuming Omega_r=0)
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


def calculate_halo_params(p, idx, halo_particleindexes, HaloID, massdefnames, massdefdenstable, npartmin, centermode, boundonly=False, boundaries="StePS", Lbox=0, rho_b=0.0):
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
        - boundonly: If True, only bound particles will be considered during the parameter estimation.
        - rho_b: background densiy
    Returns:
        - returndict: A dictionary containing all calculated halo parameters such us position, velocity, mass, radius, angular momentum, spin parameters, scale radius, energy, velocity dispersion, etc.

    """
    # Sorting particles by distance from the central particle
    if boundaries=="STEPS":
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
        if boundaries=="STEPS":
            distances = np.sqrt(np.sum(np.power((p.Coordinates[halo_particleindexes]-Center),2),axis=1)) #recalculating distances due to the new center
        elif boundaries=="PERIODIC":
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
        rho_enc[0] = rho_enc[1] # the first bin can
    else:
        rho_enc = M_enc / V_enc
    # Flagging unbound particles, if needed (this can have significant effect in the runtime, if the seach radius is too large)
    if boundonly:
        # For this, we have a first estimate on the bulk velocity of the halo. Using SO Mvir definition for this:
        radius_idx = massdefdenstable[0] <= rho_enc
        max_radi_idx = np.where(radius_idx == False)[0][0] # apply cut when the density first fall below the limit
        if max_radi_idx<npartmin:
            # In this case, the SO definition has smaller number of particles then npartmin. This also means that the BO halo definition will not have enough particles.
            p.set_HaloParentIDs(p.IDs[idx],-2) # since this group doesn't have enough patricles, the "central" particle IDs had to be set to -2
            if max_radi_idx == 0:
                p.set_HaloParentIDs(p.IDs[halo_particleindexes][sorted_idx][0],-2)
            else:
                p.set_HaloParentIDs(p.IDs[halo_particleindexes][sorted_idx][:max_radi_idx],-2) # setting the HaloParentIDs to -2, since this group has too less patricles
            return None
        R_SO = distances[sorted_idx][:max_radi_idx][-1] #Radii
        M_SO = M_enc[:max_radi_idx][-1] #Mass
        V_SO = get_center_of_mass(p.Velocities[halo_particleindexes][sorted_idx][:max_radi_idx], p.Masses[halo_particleindexes][sorted_idx][:max_radi_idx]) #Velocity; the formula for calculating the mean velocity is the same as for the CoM
        # Calculating individual energies
        T,U = get_individual_energy(p.Coordinates[halo_particleindexes][sorted_idx],p.Velocities[halo_particleindexes][sorted_idx]-V_SO,p.Masses[halo_particleindexes][sorted_idx]*1e11,p.SoftLength[halo_particleindexes][sorted_idx], boundary=boundaries, boxsize=Lbox)
        bound = (T+U)<0.0 # a bool array. If the particle is bound True; False if not
        # After this, the enclosed density and mass has to be re-calculated
        M_enc = np.cumsum(p.Masses[halo_particleindexes][sorted_idx][bound]) # enclosed (bound) mass
        V_enc = 4.0*np.pi/3.0*np.power(distances[sorted_idx][bound],3) # enclosed (bound) volume
        if centermode == "CENTRALPARTICLE":
            # In this mode, the first bin have zero volume, so we have to do this
            rho_enc = np.zeros(len(V_enc))
            rho_enc[1:]= M_enc[1:] / V_enc[1:] # enclosed mass / enclosed volume
            rho_enc[0] = rho_enc[1] # the first bin can
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
            Npart = max_radi_idx+1
            if Npart < npartmin:
                p.set_HaloParentIDs(p.IDs[idx],-2) # since this group doesn't have enough patricles, the "central" particle IDs had to be set to -2
                if max_radi_idx == 0:
                    p.set_HaloParentIDs(p.IDs[halo_particleindexes][sorted_idx][0],-2)
                else:
                    p.set_HaloParentIDs(p.IDs[halo_particleindexes][sorted_idx][bound][:max_radi_idx],-2) # setting the HaloParentIDs to -2, since this group has too less patricles
                return None
            Vmax = np.max(np.sqrt(G*1e11*M_enc[:]/(distances[sorted_idx][bound][:]*p.a))) # Maximal circular velocity of the halo
        if max_radi_idx > 0:
            # if max_radi_idx==0, then this mass definition is not applicable,
            #  because the halo doesn't have high enough density even at the center.
            R[i] = distances[sorted_idx][bound][:max_radi_idx][-1] #Radii
            M[i] = M_enc[:max_radi_idx][-1] #Mass
            V[i] = get_center_of_mass(p.Velocities[halo_particleindexes][sorted_idx][bound][:max_radi_idx], p.Masses[halo_particleindexes][sorted_idx][bound][:max_radi_idx]) #Velocity; the formula for calculating the mean velocity is the same as for the CoM
            Vrms[i] = np.sqrt(np.sum(np.power(p.Velocities[halo_particleindexes][sorted_idx][bound][:max_radi_idx] - V[i],2))/len(p.Velocities[halo_particleindexes][sorted_idx][bound][:max_radi_idx])) # root mean square velocity
            if boundaries=="STEPS":
                J[i] = get_angular_momentum((p.Coordinates[halo_particleindexes][sorted_idx][bound][:max_radi_idx]-Center)*p.a,p.Velocities[halo_particleindexes][sorted_idx][bound][:max_radi_idx] - V[i],p.Masses[halo_particleindexes][sorted_idx][bound][:max_radi_idx]) #angular momentum in (Msun/h) * (Mpc/h) * km/s physical (non-comoving) units
            elif boundaries=="PERIODIC":
                J[i] = get_angular_momentum(get_periodic_distance_vec(p.Coordinates[halo_particleindexes][sorted_idx][bound][:max_radi_idx],Center,Lbox)*p.a,p.Velocities[halo_particleindexes][sorted_idx][bound][:max_radi_idx] - V[i],p.Masses[halo_particleindexes][sorted_idx][bound][:max_radi_idx]) #angular momentum in (Msun/h) * (Mpc/h) * km/s physical (non-comoving) units
            else:
                raise Exception("Error: unkonwn boundary condition %s." % (boundaries))
            Vcirc[i] = np.sqrt(G*1e11*M[i]/(R[i]*p.a))
            if i == 0:
                p.set_HaloParentIDs(p.IDs[idx],-2) # even if the "central" particle is not bound, this will ensure to not to check this group again.
                p.set_HaloParentIDs(p.IDs[halo_particleindexes][sorted_idx][bound][:max_radi_idx],HaloID) # setting the HaloParentIDs of the particles that are in this halo (within Rvir)
                c_klypin, Rs_klypin = get_Rs_Klypin(Vmax,Vcirc[0],R[0])
                if boundonly:
                    Ekin = np.sum(T[bound][:max_radi_idx])
                    Epot = np.sum(U[bound][:max_radi_idx])
                    MSSO = np.sum(p.Masses[halo_particleindexes][distances<=R[i]]) # total mass within Rvir (Strict Spherical Overdensity)
                    MboundPerMtot = M[i] / MSSO #Total bounded mass ratio within Rvir
                    Energy = np.sum(Ekin+Epot)
                else:
                    Energy, Ekin, Epot = get_total_energy((p.Coordinates[halo_particleindexes][sorted_idx][:max_radi_idx]-Center)*p.a,p.Velocities[halo_particleindexes][sorted_idx][:max_radi_idx] - V[i],p.Masses[halo_particleindexes][sorted_idx][:max_radi_idx]*1e11,p.SoftLength[halo_particleindexes][sorted_idx][:max_radi_idx],boundary=boundaries,boxsize=Lbox) #Total energy of the halo. Needed for Peebles spin parameter
    #Spin parameters. Spins are dimensionless. Overview: https://arxiv.org/abs/1501.03280 (definitions: eq.1 and eq.4)
    absJvir = np.sqrt(np.sum(np.power(J[0],2)))
    Spin_Peebles = absJvir*1e11*np.sqrt(np.abs(Energy))/(G*np.power(M[0]*1e11,2.5)) # Peebles Spin Parameter (1969) https://ui.adsabs.harvard.edu/abs/1969ApJ...155..393P/abstract
    Spin_Bullock = get_bullock_spin(absJvir*1e11,M[0]*1e11,R[0]*p.a) # Bullock Spin Parameter (2001) https://ui.adsabs.harvard.edu/abs/2001ApJ...555..240B/abstract
    #generating output dictionary containing all calculated quantities
    returndict = {
    "ID": HaloID,
    "Npart": Npart,
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
    #saving all other quantities
    for i in range(0,len(massdefnames)):
        returndict["M"+massdefnames[i]] = M[i+1] * 1.0e11 # the output masses are in Msol
        returndict["R"+massdefnames[i]] = R[i+1] * 1.0e3 # the output radii are in kpc
        returndict["V"+massdefnames[i]] = V[i+1]
        returndict["VRMS"+massdefnames[i]] = Vrms[i+1]
        returndict["Vcirc"+massdefnames[i]] = Vcirc[i+1]
        returndict["J"+massdefnames[i]] = J[i+1] * 1e11 # the output angular momenta in Msun * Mpc * km/s (physical)
    return returndict

def SetParentThreadIDs(p, R_center, d_r, delta_theta, N_MPI_threads, MPI_rank):
    ParentThreadID= -1*np.ones(len(p.Masses),dtype=int)
    if N_MPI_threads <= 1:
        return ParentThreadID
    Nslices = N_MPI_threads - 1
    d_theta = 2.0*np.pi/Nslices
    r = np.sqrt(np.sum(np.power(p.Coordinates,2.0),axis=1))
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

    def __init__(self, FILENAME, D_UNITS, V_UNITS, M_UNITS, H_INDEPENDENT_UNITS, HUBBLE, REDSHIFT=0.0, FORCE_RES=0.0):
        print("Loading particle data from %s\n" % FILENAME)
        if FILENAME[-4:] == 'hdf5':
            self.Redshift, self.Om, self.Ol, self.H0, self.Npart = Load_params_from_HDF5_snap(FILENAME)
        else:
            self.H0 = HUBBLE
            self.Redshift = REDSHIFT
        self.h = self.H0 / 100.0
        self.a = 1.0/(self.Redshift+1.0) # scale factor
        self.sourcefile = FILENAME
        self.Coordinates, self.Velocities, self.Masses, self.IDs = Load_snapshot(FILENAME,CONSTANT_RES=False,RETURN_VELOCITIES=True,RETURN_IDs=True,SILENT=True)
        self.HaloParentIDs = -1*np.ones(len(self.Masses),dtype=np.int64)
        self.Density= np.zeros(len(self.Masses),dtype=np.double)
        # converting the input data to StePS units (Mpc, km/s, 1e11Msol)
        if H_INDEPENDENT_UNITS:
            self.Coordinates *= D_UNITS/self.h
            self.Masses *= (M_UNITS / 1e11)/self.h
        else:
            self.Coordinates *= D_UNITS
            self.Masses *= (M_UNITS / 1e11) # 1e11msol(/h) units
        self.Velocities *= V_UNITS
        self.Velocities *= np.sqrt(self.a) # km/s physical velocities, assuming Gadget convention
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
    def __init__(self, H0, Om, Ol, DE_Model, DE_Params, z, rho_c, rho_b, PrimaryMDef, SecondaryMdefList, Centermode, DensityMode, Npartmin, RemoveUnBoundParts):
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
            "rho_b": rho_b,
            "CenterMode": Centermode,
            "DensityMode": DensityMode,
            "Npart_min": Npartmin,
            "RemoveUnBoundParts": RemoveUnBoundParts
        }
        self.Nhalos = 0 # total number of halos in the catalog
        self.DataTable = [] # table containing all calculated halo parameters
        self.DenstyEstimation = [] # table containing all initial halo central density estimation

    def add_halo(self,haloparamdict,centraldensity):
        self.DataTable.append(haloparamdict)
        self.DenstyEstimation.append(centraldensity)
        self.Nhalos += 1
        self.Header["Nhalos"] = self.Nhalos
        return

    def add_catalog(self,halocatalog,overlaps=True,MPI_parent_thread=0):
        Nhalos_stored = copy.deepcopy(self.Nhalos)
        if overlaps:
            for j in range(0,halocatalog.Nhalos):
                r_halo = np.sqrt(np.sum(halocatalog.DataTable[j]["Coordinates"]**2))
                theta_halo = np.arctan2(halocatalog.DataTable[j]["Coordinates"][0], halocatalog.DataTable[j]["Coordinates"][1])+np.pi
                # if halo center is within the subvolume of the parent thread, we add it to the catalog
                if IsHaloInThreadSubvolume(r_halo,theta_halo, halocatalog.DataTable[j]["Coordinates"][2], r_master, size, MPI_parent_thread):
                    halocatalog.DataTable[j]["ID"] += Nhalos_stored
                    self.DataTable.append(halocatalog.DataTable[j])
                    self.DenstyEstimation.append(halocatalog.DenstyEstimation[j])
                    self.Nhalos += 1
        else:
            for j in range(0,halocatalog.Nhalos):
                halocatalog.DataTable[j]["ID"] += self.Nhalos
                self.DataTable.append(halocatalog.DataTable[j])
                self.DenstyEstimation.append(halocatalog.DenstyEstimation[j])
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
            header += "  | Units:\n  |\t Masses are in Msol \n  |\t Positions in Mpc (comoving)\n  |\t Velocities in km / s (physical)\n  |\t Halo Radii in kpc (comoving)\n  |\t Halo energies in Msol * (km/s)^2 (physical) \n  |\t Angular momenta in Msun * Mpc * km/s (physical)\n  |\t Spins are dimensionless\n  +---------------------------"
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
                        outarray[i][j] = self.DataTable[sorted_idx[i]][key]
            np.savetxt(filename+".dat",outarray,fmt=fmtstring,header=header)
        else:
            print("The halo catalog is empty. No file is saved.");
        return

    def save_hdf5_catalog(self, filename, save_particles=False, precision=0):
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
                        dataset = halo_group.create_dataset("Tper|U|", (self.Nhalos,),dtype=HDF5datatype)
                    else:
                        dataset = halo_group.create_dataset(key, (self.Nhalos,),dtype=HDF5datatype)
                    outarray = np.zeros(self.Nhalos,dtype=npdatatype)
                    for i in range(0,self.Nhalos):
                        outarray[i] = npdatatype(self.DataTable[i][key])
                    dataset[:] = outarray
            if save_particles:
                #Creating datasets for the particle data
                particle_group = HDF5_snapshot.create_group("/PartType1")
                print("Warning: Saving particle data is not implemented yet.")
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
rho_c = None
rho_b = None
Delta_c = None
Nmassdef = None
massdefdenstable = None
p = None
redshift = None
r_central = None
r_master = None

#Welcome message
if rank == 0:
    print("\n+-----------------------------------------------------------------------------------------------+\n|StePS_HF.py %s\t\t\t\t\t\t\t\t\t\t|\n| (STEreographically Projected cosmological Simulations Halo Finder)\t\t\t\t|\n+-----------------------------------------------------------------------------------------------+\n| Copyright (C) %s Gabor Racz\t\t\t\t\t\t\t\t|\n|\tDepartment of Physics, University of Helsinki | Helsinki, Finland\t\t\t|\n|\tJet Propulsion Laboratory, California Institute of Technology | Pasadena, CA, USA\t|\n|\tDepartment of Physics of Complex Systems, Eotvos Lorand University | Budapest, Hungary  |\n|\tDepartment of Physics & Astronomy, Johns Hopkins University | Baltimore, MD, USA\t|\n+-----------------------------------------------------------------------------------------------+\n"%(_VERSION, _YEAR))
    if len(sys.argv) != 2:
        print("Error: missing yaml file!")
        print("usage: ./StePS_HF.py <input yaml file>\nExiting.")
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
alpha = np.double(Params["SEARCH_RADIUS_ALPHA"])
kdworkers = int(Params["KDWORKERS"])
# parameters of the parallelisation
delta_theta = np.double(Params["DELTA_THETA"])/180.0*np.pi #RAD Thickness of rangential coordinate in which duplicates are searched
d_r = np.double(Params["DELTA_R"]) #thicknes of the shell in which duplicates are searched in Mpc units


if rank == 0:
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
        ERROR=3
ERROR = comm.bcast(ERROR, root=0)
if ERROR>0:
    sys.exit(ERROR)

if rank == 0:
    # Calculating relevant cosmological quantities
    rho_c = 3.0*Hz(redshift, H0, Omega_m, Omega_l, Params["DARKENERGYMODEL"], Params["DARKENERGYPARAMS"])**2/(8*np.pi)/UNIT_V/UNIT_V/(redshift+1)**3 #comoving critical density in internal units (G=1)
    rho_b = 3.0*Params['H0']**2/(8*np.pi)/UNIT_V/UNIT_V * Omega_m #background density in internal units (G=1) [the comoving background density is redshift independent]
    print("\u03C1_c (comoving):\t\t%.4e Msol/Mpc^3\n\u03C1_b (comoving):\t\t%.4e Msol/Mpc^3\n" % (rho_c*1e11, rho_b*1e11))
    Delta_c = get_Delta_c(redshift, H0, Omega_m, Omega_l, Params["DARKENERGYMODEL"], Params["DARKENERGYPARAMS"]) #Virial overdensity constant
    if Params["BOUNDARIES"] == "STEPS":
        if Params["H_INDEPENDENT_UNITS"]:
            print("Snapshot Parameters:\n--------------------\nRedshift:\t\t%.4f\nRadius:\t\t\t%.6g Mpc/h\nSoftening length:\t%.4g Mpc/h\nDistance units:\t\t%.2g Mpc/h\nVelocity units:\t\t%.2g km/s\nMass units:\t\t%.2g Msol/h\n" % (redshift,np.double(Params['RSIM']),min_mass_force_res,np.double(Params['UNIT_D_IN_MPC']), np.double(Params['UNIT_V_IN_KMPS']), np.double(Params['UNIT_M_IN_MSOL'])))
        else:
            print("Snapshot Parameters:\n--------------------\nRedshift:\t\t%.4f\nRadius:\t\t\t%.6g Mpc\nSoftening length:\t%.4g Mpc\nDistance units:\t\t%.2g Mpc\nVelocity units:\t\t%.2g km/s\nMass units:\t\t%.2g Msol\n" % (redshift,np.double(Params['RSIM']),min_mass_force_res,np.double(Params['UNIT_D_IN_MPC']), np.double(Params['UNIT_V_IN_KMPS']), np.double(Params['UNIT_M_IN_MSOL'])))
    elif Params["BOUNDARIES"] == "PERIODIC":
        if Params["H_INDEPENDENT_UNITS"]:
            print("Snapshot Parameters:\n--------------------\nRedshift:\t\t%.4f\nBoxsize:\t\t%.6g Mpc/h\nSoftening length:\t%.4g Mpc/h\nDistance units:\t\t%.2g Mpc/h\nVelocity units:\t\t%.2g km/s\nMass units:\t\t%.2g Msol/h\n" % (redshift,np.double(Params['LBOX']),min_mass_force_res,np.double(Params['UNIT_D_IN_MPC']), np.double(Params['UNIT_V_IN_KMPS']), np.double(Params['UNIT_M_IN_MSOL'])))
        else:
            print("Snapshot Parameters:\n--------------------\nRedshift:\t\t%.4f\nBoxsize:\t\t%.6g Mpc\nSoftening length:\t%.4g Mpc\nDistance units:\t\t%.2g Mpc\nVelocity units:\t\t%.2g km/s\nMass units:\t\t%.2g Msol\n" % (redshift,np.double(Params['LBOX']),min_mass_force_res,np.double(Params['UNIT_D_IN_MPC']), np.double(Params['UNIT_V_IN_KMPS']), np.double(Params['UNIT_M_IN_MSOL'])))
        if Params["INITIAL_DENSITY_MODE"] == "Voronoi":
            print("Error: Voronoi density estimation is not supported with periodic boundaries.\nExiting.")
            ERROR = 5
    else:
        print("Error: Unrecognized boundary conditions %s.\nExiting." % Params["BOUNDARIES"])
        ERROR = 4
    print("Halo Finder Parameters:\n-----------------------\nHalo catalog file:\t\t\t%s\nInitial Density Estimation:\t\t%s\nSearch radius alpha parameter:\t\t%.2f\nNumber of KDTree worker threads:\t%i\nMinimal particle number:\t\t%i\nHalo center mode:\t\t\t%s" %(Params["OUTFILE"],Params["INITIAL_DENSITY_MODE"],np.double(Params["SEARCH_RADIUS_ALPHA"]),int(Params["KDWORKERS"]), int(Params["NPARTMIN"]), Params["CENTERMODE"] ))
    if Params["BOUNDONLYMODE"]:
        print("Spherical Overdensity Mode:\t\tBound Only (BO)")
    else:
        print("Spherical Overdensity Mode:\t\tStrict Spherical Overdensity (SO)")
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

if rank == 0:
    print("\n")
    if Params["INITIAL_DENSITY_MODE"] != "Voronoi" and Params["INITIAL_DENSITY_MODE"] != "Nth neighbor":
        print("Error: Unknown initial density estimation method %s.\nExiting." % Params["INITIAL_DENSITY_MODE"])
        ERROR = 5
    if Params["CENTERMODE"] != "CENTRALPARTICLE" and Params["CENTERMODE"] != "CENTEROFMASSNPARTMIN":
        print("Error: unkown CENTERMODE parameter %s.\nExiting." % Params["CENTERMODE"])
        ERROR = 6
ERROR = comm.bcast(ERROR, root=0)
if ERROR>0:
    sys.exit(ERROR)
DensMode = Params["INITIAL_DENSITY_MODE"]
sys.stdout.flush()
if rank == 0:
    # Loading the input particle snapshot
    p = StePS_Particle_Catalog(Params['INFILE'], np.double(Params['UNIT_D_IN_MPC']), np.double(Params['UNIT_V_IN_KMPS']), np.double(Params['UNIT_M_IN_MSOL']), Params["H_INDEPENDENT_UNITS"],H0,REDSHIFT=np.double(Params['REDSHIFT']),FORCE_RES=min_mass_force_res)
    if Params["BOUNDARIES"] == "StePS":
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
        print("Estimated radius of the high-res central region of the simulation: %.2f Mpc/h" % r_central)
        print("Radius of the subvolume of the master thread: %.2f Mpc/h" % r_master)
        if size>1:
            print("Thickness of the overlap region: %.3f Mpc/h" % (d_r))
        if size>2:
            print("Angle of the tangential slices of the slave threads: %.2f degrees" % (360.0/(size-1)))
            print("Thickness of the tangential overlap region: %.2f degrees" % (180.0*delta_theta/np.pi))
        print("\n")
        # Building KDTree for a quick nearest-neighbor lookup
        tree = KDTree(p.Coordinates,leafsize=10, compact_nodes=True, balanced_tree=True, boxsize=None)
    elif Params["BOUNDARIES"] == "PERIODIC":
        # Building KDTree for a quick nearest-neighbor lookup
        tree = KDTree(p.Coordinates,leafsize=10, compact_nodes=True, balanced_tree=True, boxsize=np.double(Params['LBOX']))
    # Density reconstruction on the master thread
    if DensMode == "Voronoi":
        if Params["BOUNDARIES"] == "StePS":
            # Calculating voronoi volumes:
            p.Density = p.Masses/voronoi_volumes(p.Coordinates)/rho_c
        elif Params["BOUNDARIES"] == "PERIODIC":
            print("Error: Voronoi density estimation is not implemented for periodic boundary conditions.\nExiting.")
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
    print("min(rho) = %g/rho_c\nmax(rho) = %g/rho_c\n" % (np.min(p.Density), np.max(p.Density)))


# Bcasting the particle data
# All MPI threads will have a complete copy of the particles. Since the typical StePS snapshotis a few 100MB, this will not cause a large memory requirement
# On the other hand, for larger simulations, this can cause large memory foot print.
p = comm.bcast(p, root=0)
r_central = comm.bcast(r_central, root=0) # radius of the central region that will be analysed in the master thread
r_master = comm.bcast(r_master, root=0) # radius of the subvolume of the master thread
if rank > 0:
    if Params["BOUNDARIES"] == "StePS":
        # Building KDTree for a quick nearest-neighbor lookup for every other thread.
        tree = KDTree(p.Coordinates,leafsize=10, compact_nodes=True, balanced_tree=True, boxsize=None)
    elif Params["BOUNDARIES"] == "PERIODIC":
        # Building KDTree for a quick nearest-neighbor lookup for every other thread.
        tree = KDTree(p.Coordinates,leafsize=10, compact_nodes=True, balanced_tree=True, boxsize=np.double(Params['LBOX']))

if size>1:
    #Distributing the halo candidates evenly between the threads
    ParentThreadID = SetParentThreadIDs(p, r_master, d_r, delta_theta,size, rank) # this is a local variable for every thread
    #print("ParentThreadID: ", ParentThreadID)
else:
    ParentThreadID = np.zeros(p.Npart, dtype=np.int32)

# Identifying halos using Spherical Overdensity (SO) method
print("MPI Rank %i: Identifying halos and calculating halo parameters..." % rank)
sys.stdout.flush()
id_start = time.time()
halos = StePS_Halo_Catalog(np.double(Params["H0"]), np.double(Params["OMEGAM"]), np.double(Params["OMEGAL"]), Params["DARKENERGYMODEL"], Params["DARKENERGYPARAMS"], redshift, rho_c, rho_b, "Vir", Params["MASSDEF"], Params["CENTERMODE"], Params["INITIAL_DENSITY_MODE"], Params["NPARTMIN"], Params["BOUNDONLYMODE"])
halo_ID = 0
while True:
    #selecting the largest density particle with parentID=-1
    idx = p.IDs[np.logical_and(p.HaloParentIDs == -1, ParentThreadID == rank)][np.argmax(p.Density[np.logical_and(p.HaloParentIDs == -1, ParentThreadID == rank)])]
    maxdens = p.Density[idx]
    if maxdens>0.5*Delta_c:
        #Query the kd-tree for nearest neighbors.
        search_radius = alpha*np.cbrt(p.Masses[idx]/rho_b) #In StePS simulations, the particles are more density packed at the center. The typical particle separation is proportional to the cubic root of the particle mass.
        halo_particleindexes = tree.query_ball_point(p.Coordinates[idx], search_radius, p=2.0, eps=0, workers=kdworkers, return_sorted=False)
        if len(halo_particleindexes) >= npartmin:
            #print("\nCentral estimated density for halo #%i: %.2f \u03C1_c" % (halo_ID, maxdens))
            #print("\tCentral coordinate of halo #%i: " % (halo_ID), p.Coordinates[idx])
            #print("\tID of the central particle of halo #%i: %i" % (halo_ID,idx))
            #print("\tSearch radius for halo #%i: %.2fMpc/h" % (halo_ID, search_radius))
            #print("\tNumber of particles in the search radius of halo #%i:" % (halo_ID),len(halo_particleindexes))
            halo_params = calculate_halo_params(p, idx, halo_particleindexes, halo_ID, Params["MASSDEF"], massdefdenstable, npartmin, Params["CENTERMODE"],boundonly=Params["BOUNDONLYMODE"], rho_b=rho_b,boundaies=Params["BOUNDARIES"],Lbox=np.double(Params["LBOX"]))
            if halo_params != None:
                halos.add_halo(halo_params, maxdens) #adding the identified halo to the catalog
                halo_ID +=1
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
    halos_final = StePS_Halo_Catalog(np.double(Params["H0"]), np.double(Params["OMEGAM"]), np.double(Params["OMEGAL"]), Params["DARKENERGYMODEL"], Params["DARKENERGYPARAMS"], redshift, rho_c, rho_b, "Vir", Params["MASSDEF"], Params["CENTERMODE"], Params["INITIAL_DENSITY_MODE"], Params["NPARTMIN"], Params["BOUNDONLYMODE"])
    # adding the catalog of the master thread to the final catalog
    if size > 1:
        halos_final.add_catalog(halos,overlaps=True, MPI_parent_thread=0)
    else:
        halos_final.add_catalog(halos,overlaps=False, MPI_parent_thread=0)
#collecting all remaining halos from the slave threads
if size > 1:
    if rank != 0:
        comm.send(halos, 0, tag=rank)
    else:
        for i in range(1,size):
            print("MPI Rank %i: Receiving halos from Rank %i."%(rank,i))
            halos = comm.recv(buf=None, source=i, tag=i)
            halos_final.add_catalog(halos,overlaps=True,MPI_parent_thread=i)
        print("MPI Rank %i: Halo catalogs merged."%(rank))
        print("MPI Rank %i: The total size of the generated halo catalog is: %i\n"%(rank,halos_final.Nhalos))

if rank == 0:
    if halos.Nhalos > 0:
        if Params["OUTFORMAT"] == "ASCII" or Params["OUTFORMAT"] == "BOTH":
            print("Saving the generated catalog to %s.dat" % Params["OUTFILE"])
            halos_final.save_ascii_catalog(Params["OUTFILE"])
        if Params["OUTFORMAT"] == "HDF5" or Params["OUTFORMAT"] == "BOTH":
            print("Saving the generated catalog to %s.hdf5" % Params["OUTFILE"], end='')
            halos_final.save_hdf5_catalog(Params["OUTFILE"],save_particles=Params["SAVEPARTICLES"])
    end = time.time()
    print("\nSO halo finding finished under %fs.\n" % (end-start))

#!/usr/bin/env python3

#*******************************************************************************#
#  StePS_IC.py - An initial condition generator for                             #
#     STEreographically Projected cosmological Simulations                      #
#    Copyright (C) 2017-2024 Gabor Racz                                         #
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

import numpy as np
import camb
from camb import model, initialpower
from camb.dark_energy import DarkEnergyFluid

from colossus.cosmology import cosmology #Colossus is used for linear growth calculation

def get_Linear_GrowthFunction(z, H0, OmegaM, OmegaB, OmegaL, s8, ns, DE, DE_params, SILENT=False):
    '''
    For w0 and CPL dark energy, we use the Colossus implementation of Linder & Jenkins 2003 Eq. 11.
    https://arxiv.org/pdf/astro-ph/0305286.pdf
    '''
    #Calculating the curvature
    OmegaK = 1.0-OmegaM-OmegaL
    if np.abs(OmegaK)<=1e-5:
        flat=True
        print("Flat cosmology.")
    else:
        flat=False
        print("Non-flat cosmology, OmegaK=%e, OmegaM=%.4f, OmegaL=%.4f" %(OmegaK, OmegaM, OmegaL))
    Tcmb = 0.001 # Zero CMB temperature can cause issues in colossus. This small value shouldn't cause any significant errors at late times in relevant cosmologies. (non-zero OmegaR will be implemented in the future.)
    if DE == 'Lambda':
        if flat==True:
            params = {'flat': flat, 'H0': H0, 'Om0': OmegaM, 'Ob0': OmegaB, 'sigma8': s8, 'ns': ns, 'Tcmb0': Tcmb}
        else:
            params = {'flat': flat, 'H0': H0, 'Om0': OmegaM, 'Ob0': OmegaB, 'Ode0': OmegaL, 'sigma8': s8, 'ns': ns, 'Tcmb0': Tcmb}
        cosmo = cosmology.setCosmology('LCDM', **params)
    elif DE == 'w0':
        if flat==True:
            params = {'flat': flat, 'H0': H0, 'Om0': OmegaM, 'Ob0': OmegaB, 'sigma8': s8, 'ns': ns, 'de_model': 'w0', 'w0': DE_params[0], 'Tcmb0': Tcmb}
        else:
            params = {'flat': flat, 'H0': H0, 'Om0': OmegaM, 'Ob0': OmegaB, 'Ode0': OmegaL, 'sigma8': s8, 'ns': ns, 'de_model': 'w0', 'w0': DE_params[0], 'Tcmb0': Tcmb}
        cosmo = cosmology.setCosmology('wCDM', **params)
    elif DE == 'CPL':
        if flat==True:
            params = {'flat': flat, 'H0': H0, 'Om0': OmegaM, 'Ob0': OmegaB, 'sigma8': s8, 'ns': ns, 'de_model': 'w0wa', 'w0': DE_params[0], 'wa': DE_params[1], 'Tcmb0': Tcmb}
        else:
            params = {'flat': flat, 'H0': H0, 'Om0': OmegaM, 'Ob0': OmegaB, 'Ode0': OmegaL, 'sigma8': s8, 'ns': ns, 'de_model': 'w0wa', 'w0': DE_params[0], 'wa': DE_params[1], 'Tcmb0': Tcmb}
        cosmo = cosmology.setCosmology('w0waCDM', **params)
    Dlin = cosmo.growthFactorUnnormalized(z)/cosmo.growthFactorUnnormalized(0.0)
    if not SILENT:
        print("Initial normalized linear growth: D(z=%.2f)/D(z=0) = %e" % (z, Dlin))
    return Dlin

def get_CAMB_Linear_SPECTRUM(H0=73.0, ombh2=0.024, omch2=0.1092445, omk=0.0,ns=1.0,redshift=127,kmin=0.01,kmax=1.0,npoints=1024,sigma8='Auto',DE='Lambda',DE_params=None):
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, omk=omk)
    if DE != 'Lambda':
        if DE == 'w0':
            pars.DarkEnergy = DarkEnergyFluid()
            pars.DarkEnergy.set_params(w=DE_params[0])
        elif DE == 'CPL':
            pars.DarkEnergy = DarkEnergyFluid()
            pars.DarkEnergy.set_params(w=DE_params[0],wa=DE_params[1])
    pars.InitPower.set_params(ns=ns)
    pars.set_matter_power(redshifts=[0.0], kmax=kmax)
    #Linear spectra
    pars.NonLinear = model.NonLinear_none
    results = camb.get_results(pars)
    s8 = results.get_sigma8()
    print("original s8: %f"%s8)
    pars.set_matter_power(redshifts=[0.0], kmax=kmax)
    results = camb.get_results(pars)
    s8 = results.get_sigma8()
    #calculating normalized linear growth function
    omm=(ombh2+omch2)/((H0/100.0)**2) #non-relativistic matter density
    omb=ombh2/((H0/100.0)**2) #baryonic matter density
    oml=1.0-omk-omm # dark energy density
    DzD0 = get_Linear_GrowthFunction(redshift, H0, omm, omb, oml, s8, ns, DE, DE_params )
    #calculating the z=0 linear P(k)
    kh, z, pk = results.get_matter_power_spectrum(minkh=kmin, maxkh=kmax, npoints = npoints)
    if sigma8 != 'Auto':
        pars.InitPower.set_params(As=2e-09*(sigma8/s8)**2, ns=ns, nrun=0, nrunrun=0.0, r=0.0, nt=None, ntrun=0.0, pivot_scalar=0.05, pivot_tensor=0.05, parameterization='tensor_param_rpivot')
        pars.set_matter_power(redshifts=[0.0], kmax=kmax)
        results = camb.get_results(pars)
        s8 = results.get_sigma8()
        print("rescaled s8: %f"%s8)
        kh, z, pk = results.get_matter_power_spectrum(minkh=kmin, maxkh=kmax, npoints = npoints)
    return kh, pk[0]*DzD0**2

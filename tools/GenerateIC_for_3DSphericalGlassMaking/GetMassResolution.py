#!/usr/bin/env python3

#*******************************************************************************#
#  GetMassResolution.py - Mass resolution calculator for StePS simulations      #
#     STEreographically Projected cosmological Simulations                      #
#    Copyright (C) 2018-2025 Gabor Racz                                         #
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

from os.path import exists
import numpy as np
import yaml
import sys
from pathlib import Path
# adding ../Utils/ to the system path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "Utils"))
import matplotlib
import matplotlib.pyplot as plt

_VERSION = "v0.0.0.1"
_YEAR    = "2026"
_NAME    = "GetMassResolution"
_AUTHORS = "Gabor Racz"


#Begininng of the script
# Welcome message
print("+-------------------------------------------------------------------------------------------------------+\n|%s %s\t\t\t\t\t\t\t\t\t\t|\n| (A mass resolution calculator for StePS simulations.)\t\t\t\t\t\t\t|\n+-------------------------------------------------------------------------------------------------------+\n|\t\t\t\t\t\t\t\t\t\t\t\t\t|\n| %s, %s\t\t\t\t\t\t\t\t\t\t\t|\n|\tDepartment of Physics, University of Helsinki | Helsinki, Finland\t\t\t\t|\n|\tJet Propulsion Laboratory, California Institute of Technology | Pasadena, CA, USA\t\t|\n|\tDepartment of Physics of Complex Systems, Eotvos Lorand University | Budapest, Hungary\t\t|\n|\tDepartment of Physics & Astronomy, Johns Hopkins University | Baltimore, MD, USA\t\t|\n+-------------------------------------------------------------------------------------------------------+\n\n" % (_NAME,_VERSION,_AUTHORS,_YEAR))

#read the input parameters (from the first argument)
if len(sys.argv) != 2:
        print("usage: ./%s.py <input yaml file>" % _NAME)
        sys.exit(2)
print("Reading the %s paramfile...\n" % str(sys.argv[1]))
document = open(str(sys.argv[1]))
Params = yaml.safe_load(document)
document.close()
#Checking the input parameters
if 'BOUNDARY' not in Params:
    print("Error: BOUNDARY not defined in the input file!\nExiting...\n")
    sys.exit(2)
if Params['BOUNDARY'] == "PDS":
    # Use the same shell and exterior mass plan as the PDS IC generator.
    import GenerateIC_for_PDSGlassMaking as pdsgen
    (R_curv, hubble, omega_m, omega_l, omega_k, n_part, seed, baseout,
     _ignored, zoom_settings) = pdsgen.read_parameters(str(sys.argv[1]))
    rho_mean = 3.0*(hubble/pdsgen.UNIT_V)**2/(8.0*np.pi)*omega_m
    V_pds = np.pi**2*R_curv**3/60.0
    r_max = pdsgen.dodecahedron_lengths(R_curv)[1]
    def report(label, value):
        print(f"{label + ':':<44} {value}")

    print("Glass parameters:\n-----------------")
    report("Boundary condition", "Poincare Dodecahedral Space (S^3/I*)")
    report("Output file", baseout)
    report("S^3 curvature radius", f"{R_curv:.12g} Mpc")
    report("Random seed", seed)
    print("\nCosmological parameters:\n------------------------")
    report("Omega_lambda", f"{omega_l:.12g}")
    report("Omega_m", f"{omega_m:.12g}")
    report("Omega_k", f"{omega_k:.12g}")
    report("H0", f"{hubble:.12g} km/s/Mpc")
    from pds_zoom import zoom_config, plan_zoom
    config = zoom_config(zoom_settings)
    if config is None:
        Mass_res = rho_mean*V_pds/n_part
        print("\nResolution:")
        report("Homogeneous particle mass", pdsgen.format_mass(Mass_res))
        r = np.array([0.0, r_max])
        Mass = np.array([Mass_res, Mass_res])
        m_min = Mass_res
        n_ref = n_part
    else:
        meta, rows, rmap = plan_zoom(config, R_curv, seed, rho_mean, pdsgen._VERSION)
        r_zoom = meta['zoom_radius']
        r = np.array([row['chi_lo'] * R_curv for row in rows] + [r_zoom, r_max])
        Mass = np.array([row['particle_mass'] for row in rows] + [meta['exterior_particle_mass']]*2)
        if meta['central_particle_count']:
            r = np.r_[0.0, r]
            Mass = np.r_[meta['central_particle_mass'], Mass]
        m_min = meta['minimum_particle_mass']
        n_ref = meta['total_particle_count']
        print("\nZoom load:")
        report("Zoom-in radius", f"{r_zoom:.12g} Mpc")
        report("Injectivity radius", f"{meta['injectivity_radius']:.12g} Mpc")
        report("Particles inside zoom region", meta['zoom_particle_count'])
        report("Particles outside zoom region", meta['exterior_particle_count'])
        report("Nominal / retained radial bins", f"{config['NRBINS']} / {meta['radial_shell_count']}")
        report("Particles per shell", config['NSHELL'])
        report("High-resolution core radius", f"{meta['constant_resolution_radius']:.12g} Mpc")
        report("High-resolution core particles", meta['central_particle_count'])
        if meta['central_particle_count']:
            report("High-resolution core particle mass", pdsgen.format_mass(meta['central_particle_mass']))
        report("Edge / exterior particle mass", pdsgen.format_mass(meta['exterior_particle_mass']))
        report("Edge mass rounding correction", f"{meta['edge_mass_relative_correction']:.6g}")
        report("Minimum / maximum particle mass",
               f"{pdsgen.format_mass(m_min)} / {pdsgen.format_mass(meta['maximum_particle_mass'])}")
    report("Total number of particles", n_ref)
    report("Position and mass storage", f"{n_ref * 32 / 2**30:.3f} GiB (32 bytes/particle)")
    if config is not None:
        report("Additional azimuth sort arrays", f"{n_ref * 16 / 2**30:.3f} GiB, plus working memory")
    min_sep = np.cbrt(m_min/rho_mean)
    report("Minimum-mass particle separation", f"{min_sep:.12g} Mpc")
    report("Recommended softening length", f"{min_sep/40.0:.12g} Mpc")
    print()
    figscale = 0.75
    plt.figure(figsize=(7*figscale,4.25*figscale))
    plt.xlabel(r'Geodesic distance from zoom centre $[\mathrm{Mpc}]$')
    plt.ylabel(r'$M[\mathrm{M}_{\odot}]$')
    plt.gca().set_xlim([0.0, r_max])
    plt.step(r, Mass*1e11, where='post', c='teal', label="StePS Resolution")
    plt.yscale('log')
    plt.title(r"Initial glass mass resolution (distance from the zoom-in centre)")
    plt.tight_layout()
    if Params.get('SAVEPLOTS', False) == True:
        plt.savefig(str(Path(baseout).with_suffix(''))+"_MassResolution_vs_Radius.pdf", format='pdf')
    plt.show()
    sys.exit(0)
from compactification import *  # imported late: not needed for PDS
print("Glass parameters:\n----------------------")
if Params['BOUNDARY'] == "SPHERICAL":
    print("Boundary condition:\t\t\t\tSpherical (3D StePS)")
    print("Output file:\t\t\t\t\t%s\nDiameter of the 4D hypersphere:\t\t\t%f Mpc\nRadius of the simulation volume:\t\t%f Mpc\nRandom seed:\t\t\t\t\t%i" % (Params['BASEOUT'], Params['D_S'], Params['RSIM'], Params['RANDSEED'] ))
elif Params['BOUNDARY'] == "CYLINDRICAL":
    print("Boundary conditions:\t\t\t\tCylindrical (2D StePS)")
    print("Output file:\t\t\t\t\t%s\nDiameter of the 3D sphere:\t\t\t%f Mpc\nRadius of the simulation volume:\t\t%f Mpc\nLinear size in the Z direction:\t\t\t%f Mpc\nRandom seed:\t\t\t\t\t%i" % (Params['BASEOUT'], Params['D_S'], Params['RSIM'], Params['LZSIM'], Params['RANDSEED'] ))
else:
    print("Error: Unknown boundary condition %s!\nExiting...\n" % Params['BOUNDARY'])
    sys.exit(2)
np.random.seed(Params['RANDSEED'])
if Params['BIN_MODE'] == 0:
    print("Binning mode:\t\t\t\t\tConstant size binning in the \"omega\" compact coordinate.")
    print("Input Periodic Glass:\t\t\t\t%s" % Params['GLASSFILE'])
    print("Radius of constant resolution:\t\t\t%fMpc" % Params['RCRIT'] )
    last_cell_size = Params['NRBINS']*np.pi/(2*np.arctan(Params['RSIM']/Params['D_S']))-Params['NRBINS']
if Params['BIN_MODE'] == 1:
    print("Binning mode:\t\t\t\t\tConstant shell volumes in the compact space.")
if (Params['BIN_MODE'] > 1) or (Params['BIN_MODE'] < 0):
    print("Error: Binning mode = %i\n Unknown binning mode!\n Exiting." % Params['BIN_MODE'])
    sys.exit(2)

    
if Params['BOUNDARY'] == "SPHERICAL":
    print("Number of particles per spherical shell:\t%i\nNumber of radial bins:\t\t\t\t%i\n" % (Params['NSHELL'], Params['NRBINS']))
if Params['BOUNDARY'] == "CYLINDRICAL":
    print("Number of particles per cylindrical shell:\t%i\nNumber of radial bins:\t\t\t\t%i\n" % (Params['NSHELL'], Params['NRBINS']))
print("Cosmological parameters:\n------------------------\nOmega_lambda\t%f\nOmega_m\t\t%f\nOmega_k\t\t%f\nH0\t\t%f(km/s)/Mpc\n" % (Params['OMEGA_L'], Params['OMEGA_M'], 1.0-Params['OMEGA_M']-Params['OMEGA_L'], Params['HUBBLE_CONSTANT']))
#Calculating the mean density:
rho_crit = 3*Params['HUBBLE_CONSTANT']**2/(8*np.pi)*0.0482191394711204*0.0482191394711204 #G=1
rho_mean = rho_crit*Params['OMEGA_M']
#Calculating the resolutions:
if Params['BOUNDARY'] == "SPHERICAL":
    if Params['BIN_MODE'] == 0:
        i_crit = Calculate_i_r(Params['RCRIT'], Params['D_S'], Params['NRBINS'], last_cell_size)
        i = np.arange(Params['NRBINS'])
        r = Calculate_r_i(i, Params['D_S'], Params['NRBINS'], last_cell_size)
        Mass = np.zeros(Params['NRBINS'])
        Mass_res_inside = Mass_resolution_i_3D(i_crit, Params['D_S'], Params['NRBINS'], Params['NSHELL'], rho_mean, last_cell_size)
        #Calculating the total number of particles:
        N_part_inside = int((4.0*np.pi/3.0*Calculate_rlimits_i(i_crit, Params['D_S'], Params['NRBINS'], last_cell_size)**3)*rho_mean/Mass_res_inside)
        #recalculating the mass resolution inside RCRIT
        Mass_res_inside = ((4.0*np.pi/3.0*Calculate_rlimits_i(i_crit, Params['D_S'], Params['NRBINS'], last_cell_size)**3)*rho_mean/N_part_inside)
        N_part_outside = (Params['NRBINS']-i_crit)*Params['NSHELL']
        N_part = N_part_inside + N_part_outside
        #Allocating memory for the particles
        print("Total number of particles =\t\t%i\n" % N_part)
        print("Number of particles inside the constant resolution region =\t%i\n" % N_part_inside)
        print("Number of particles outside the constant resolution region =\t%i\n" % N_part_outside)
        print("The Mass resolution inside the constant resolution region = %lf 10e11Msol\n" % Mass_res_inside)
        for j in i:
            if j>=i_crit:
                Mass[j] = Mass_resolution_i_3D(j, Params['D_S'], Params['NRBINS'], Params['NSHELL'], rho_mean, last_cell_size)
            else:
                Mass[j] = Mass_res_inside
    elif Params['BIN_MODE'] == 1:
        i = np.arange(Params['NRBINS'])
        N_part = np.int64(Params['NRBINS'])*np.int64(Params['NSHELL'])
        r = Calculate_r_i_cvol(i, Params['D_S'], Params['NRBINS'], Params['RSIM'])
        Mass = Mass_resolution_i_3D_cvol(i, Params['D_S'], Params['NRBINS'], Params['NSHELL'], rho_mean, Params['RSIM'])
        print("Total number of particles =\t\t%i\n" % N_part)
        print("The Mass resolution at the tangent point = %lf 10e11Msol\n" % Mass[0])
        print("The calculated real-space bins:\nr_min\t\tr_i\t\tr_max\t\tMass_part")
        print(np.array(( Calculate_rlimits_i_3D_cvol(i, Params['D_S'], Params['NRBINS'], Params['RSIM']),r, Calculate_rlimits_i_3D_cvol(i+1, Params['D_S'], Params['NRBINS'], Params['RSIM']),Mass)).T)
elif Params['BOUNDARY'] == "CYLINDRICAL":
    if Params['BIN_MODE'] == 0:
        i_crit = Calculate_i_r(Params['RCRIT'], Params['D_S'], Params['NRBINS'], last_cell_size)
        i = np.arange(Params['NRBINS'])
        r = Calculate_r_i(i, Params['D_S'], Params['NRBINS'], last_cell_size)
        Mass = np.zeros(Params['NRBINS'])
        Mass_res_inside = Mass_resolution_i_2D(i_crit, Params['LZSIM'], Params['D_S'], Params['NRBINS'], Params['NSHELL'], rho_mean, last_cell_size)
        #Calculating the total number of particles:
        N_part_inside = int((Params['LZSIM']*np.pi*Calculate_rlimits_i(i_crit, Params['D_S'], Params['NRBINS'], last_cell_size)**2)*rho_mean/Mass_res_inside)
        #recalculating the mass resolution inside RCRIT
        Mass_res_inside = ((Params['LZSIM']*np.pi*Calculate_rlimits_i(i_crit, Params['D_S'], Params['NRBINS'], last_cell_size)**2)*rho_mean/N_part_inside)
        N_part_outside = (Params['NRBINS']-i_crit)*Params['NSHELL']
        N_part = N_part_inside + N_part_outside
        #Allocating memory for the particles
        print("Total number of particles =\t\t%i\n" % N_part)
        print("Number of particles inside the constant resolution region =\t%i\n" % N_part_inside)
        print("Number of particles outside the constant resolution region =\t%i\n" % N_part_outside)
        print("The Mass resolution inside the constant resolution region = %lf 10e11Msol\n" % Mass_res_inside)
        for j in i:
            if j>=i_crit:
                Mass[j] = Mass_resolution_i_2D(j, Params['LZSIM'],Params['D_S'], Params['NRBINS'], Params['NSHELL'], rho_mean, last_cell_size)
            else:
                Mass[j] = Mass_res_inside
    elif Params['BIN_MODE'] == 1:
        #i = np.arange(Params['NRBINS'])
        #N_part = np.int64(Params['NRBINS'])*np.int64(Params['NSHELL'])
        #r = Calculate_r_i_cvol(i, Params['D_S'], Params['NRBINS'], Params['RSIM'])
        #Mass = Mass_resolution_i_2D_cvol(i, Params['LZSIM'], Params['D_S'], Params['NRBINS'], Params['NSHELL'], rho_mean, Params['RSIM'])
        #print("Total number of particles =\t\t%i\n" % N_part)
        #print("The Mass resolution at the tangent point = %lf 10e11Msol\n" % Mass[0])
        #print("The calculated real-space bins:\nr_min\t\tr_i\t\tr_max\t\tMass_part")
        #print(np.array(( Calculate_rlimits_i_2D_cvol(i, Params['D_S'], Params['NRBINS'], Params['RSIM']),r, Calculate_rlimits_i_2D_cvol(i+1, Params['D_S'], Params['NRBINS'], Params['RSIM']),Mass)).T)
        raise NotImplementedError("Cylindrical binning mode 1 is not implemented yet.")


#Mill_res = 0.86/(73.0/100.0)/100 + 0*i
figscale = 0.75
plt.figure(figsize=(7*figscale,4.25*figscale))
if Params['BOUNDARY'] == "SPHERICAL":
    plt.xlabel(r'$r[\mathrm{Mpc}]$')
elif Params['BOUNDARY'] == "CYLINDRICAL":
    plt.xlabel(r'$\varrho[\mathrm{Mpc}]$')
plt.ylabel(r'$M[\mathrm{M}_{\odot}]$')
axes = plt.gca()
axes.set_xlim([0.0,Params['RSIM']])
#plt.grid()
color = 'teal'
plt.semilogy(r,Mass*1e11, c=color, label="StePS Resolution")
#if Params['BIN_MODE'] == 0:
#    Mass_R5 = Mass_res_inside*(r/Params['RCRIT'])**5
#    plt.semilogy(r[r>Params['RCRIT']],Mass_R5[r>Params['RCRIT']], '--', c='b', label=r'$M(R)=M_p(R_c)\cdot\left(\frac{R}{R_c}\right)^5$')
#plt.semilogy(r,Mill_res, c='r', label="Millennium Resolution")
#plt.legend()
plt.title(r"Initial glass mass resolution")
plt.tight_layout()
if Params['SAVEPLOTS'] == True:
    plt.savefig(Params['BASEOUT'][:-5]+"_MassResolution_vs_Radius.pdf", format='pdf')
plt.show()

#the simulation volume is:
if Params['BOUNDARY'] == "SPHERICAL":
    Tot_V = 4*np.pi/3*Params['RSIM']**3
    if Params['BIN_MODE'] == 0:
        V_central = 4*np.pi/3*Params['RCRIT']**3
elif Params['BOUNDARY'] == "CYLINDRICAL":
    Tot_V = np.pi*Params['RSIM']**2*Params['LZSIM']
    if Params['BIN_MODE'] == 0:
        V_central = np.pi*Params['RCRIT']**2*Params['LZSIM']


if Params['BIN_MODE'] == 0:
    print("The mean particle separation in the constant resolution region = %f Mpc\nThe recommended softening length is %f Mpc\n" % (np.cbrt(V_central/N_part_inside), np.cbrt(V_central/N_part_inside)/40.0))

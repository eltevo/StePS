#!/usr/bin/env python3

#*******************************************************************************#
#  StePS_IC.py - An initial condition generator for                             #
#     STEreographically Projected cosmological Simulations                      #
#    Copyright (C) 2017-2018 Gabor Racz                                         #
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

import sys
import time
import yaml
import numpy as np
import astropy.units as u
from astropy.cosmology import LambdaCDM, z_at_value
from ascii2gadget import *
from write_ICparamfile import *
from write_HDF5_snapshot import *
from subprocess import call
from pynverse import inversefunc

#some basic function for the stereographic projection
#Functions for the constant omega binning method
def Calculate_rlimits_i(i, d_s, N_r_bin, last_cell_size):
    r_i = d_s*np.tan((i)*np.pi/(2.0*(N_r_bin+last_cell_size)))
    return r_i;
def Calculate_r_i(i, d_s, N_r_bin, last_cell_size):
    lower_limit = Calculate_rlimits_i(i, d_s, N_r_bin, last_cell_size)
    upper_limit = Calculate_rlimits_i(i+1, d_s, N_r_bin, last_cell_size)
    #simple assumption with "conical frustum"
    r_i = 0.25*(upper_limit-lower_limit)*(lower_limit*lower_limit+2*lower_limit*upper_limit+3*upper_limit*upper_limit)/(lower_limit*lower_limit+lower_limit*upper_limit+upper_limit*upper_limit)+lower_limit
    return r_i;

#Functions for the constant volume binning method (constant volume in the compact space)
def Calculate_rlimits_i_cvol(i, d_s, N_r_bin, R_sim):
    '''
    Calculates the lower limit of the i-th bin for the constant volume binning in the
    non-compact space (constant volume in the compact space)

    i = the ID of the boundary
    d_s = the diameter of the 4D sphere
    N_r_bin = Number of the radial bins
    R_sim = the radius of the simulation volume in real space
    '''
    omega_max = 2.0*np.arctan(R_sim/d_s)
    V_unit_bin = (2.0*omega_max-np.sin(2.0*omega_max))/N_r_bin
    V_unit_to_i = i*V_unit_bin
    #inverting numerically the x-sin(x) function
    func = (lambda x: x-np.sin(x))
    omega_i = inversefunc(func, y_values=V_unit_to_i)/2.0
    r_i = d_s*np.tan(omega_i/2)
    return r_i;
def Calculate_r_i_cvol(i, d_s, N_r_bin, R_sim):
    lower_limit = Calculate_rlimits_i_cvol(i, d_s, N_r_bin, R_sim)
    upper_limit = Calculate_rlimits_i_cvol(i+1, d_s, N_r_bin, R_sim)
    #Calculating the center of the bin with "conical frustum"
    r_i = 0.25*(upper_limit-lower_limit)*(lower_limit*lower_limit+2*lower_limit*upper_limit+3*upper_limit*upper_limit)/(lower_limit*lower_limit+lower_limit*upper_limit+upper_limit*upper_limit)+lower_limit
    return r_i;

#Beginning of the script

print("+-----------------------------------------------------------------------------------------------+\n" \
"|   _____ _       _____   _____    _____ _____               \t\t\t\t\t|\n" \
"|  / ____| |     |  __ \ / ____|  |_   _/ ____|              \t\t\t\t\t|\n" \
"| | (___ | |_ ___| |__) | (___      | || |       _ __  _   _ \t\t\t\t\t|\n" \
"|  \___ \| __/ _ \  ___/ \___ \     | || |      | '_ \| | | |\t\t\t\t\t|\n" \
"|  ____) | ||  __/ |     ____) |____| || |____ _| |_) | |_| |\t\t\t\t\t|\n" \
"| |_____/ \__\___|_|    |_______________\_____(_) .__/ \__, |\t\t\t\t\t|\n" \
"|                                               | |     __/ |\t\t\t\t\t|\n" \
"|                                               |_|    |___/ \t\t\t\t\t|\n" \
"|StePS_IC.py v0.2.0.2\t\t\t\t\t\t\t\t\t\t|\n| (an IC generator python script for STEreographically Projected cosmological Simulations)\t|\n+-----------------------------------------------------------------------------------------------+\n| Copyright (C) 2018 Gabor Racz\t\t\t\t\t\t\t\t\t|\n|\tDepartment of Physics of Complex Systems, Eotvos Lorand University | Budapest, Hungary  |\n|\tDepartment of Physics & Astronomy, Johns Hopkins University | Baltimore, MD, USA\t|\n+-----------------------------------------------------------------------------------------------+\n")
print("+---------------------------------------------------------------+\n" \
"| StePS_IC.py comes with ABSOLUTELY NO WARRANTY.                |\n" \
"| This is free software, and you are welcome to redistribute it |\n" \
"| under certain conditions. See the LICENSE file for details.   |\n" \
"+---------------------------------------------------------------+\n\n")
if len(sys.argv) != 2:
    print("Error: missing yaml file!")
    print("usage: ./StePS_IC.py <input yaml file>\nExiting.")
    sys.exit(2)
start = time.time()
#Setting up the units of distance and time
UNIT_T=47.14829951063323 #Unit time in Gy
UNIT_V=20.738652969925447 #Unit velocity in km/s
UNIT_D=3.0856775814671917e24#=1Mpc Unit distance in cm
print("Reading the %s paramfile...\n" % str(sys.argv[1]))
document = open(str(sys.argv[1]))
Params = yaml.load(document)
print("Cosmological Parameters:\n------------------------\nOmega_m:\t%f\nOmega_lambda:\t%f\nOmega_k:\t%f\nOmega_b:\t%f\nH0:\t\t%f km/s/Mpc\nRedshift:\t%f\nSigma8:\t\t%f\n" % (Params['OMEGAM'], Params['OMEGAL'], 1.0-Params['OMEGAM']-Params['OMEGAL'], Params['OMEGAB'], Params['H0'], Params['REDSHIFT'], Params['SIGMA8']))
Params['INPUTSPECTRUM_UNITLENGTH_IN_CM'] = np.float64(Params['INPUTSPECTRUM_UNITLENGTH_IN_CM'])
print("IC parameters:\n--------------\nLbox:\t\t\t%f Mpc\nRsim:\t\t\t%f Mpc\nVOI_x:\t\t\t%f Mpc\nVOI_y:\t\t\t%f Mpc\nVOI_z:\t\t\t%f Mpc\nSeed:\t\t\t%i\nSpheremode:\t\t%i\nWhichSpectrum:\t\t%i\nFileWithInputSpectrum:\t%s\nInputSpectrum_UnitLength_in_cm\t%e\nReNormalizeInputSpectrum:\t%i\nShapeGamma:\t\t%f\nPrimordialIndex:\t%f\nNgrid samples:\t\t%i\nGlassFile:\t\t%s\nOutDir:\t\t\t%s\nFileBase:\t\t%s\nComoving IC:\t\t%s\nNumber of MPI tasks:\t%i" % (Params['LBOX'], Params['RSIM'], Params['VOIX'], Params['VOIY'], Params['VOIZ'], Params['SEED'], Params['SPHEREMODE'], Params['WHICHSPECTRUM'], Params['FILEWITHINPUTSPECTRUM'], Params['INPUTSPECTRUM_UNITLENGTH_IN_CM'], Params['RENORMALIZEINPUTSPECTRUM'], Params['SHAPEGAMMA'], Params['PRIMORDIALINDEX'], Params['NGRIDSAMPLES'], Params['GLASSFILE'], Params['OUTDIR'], Params['FILEBASE'], Params['COMOVINGIC'], Params['MPITASKS']))
Params['UNITLENGTH_IN_CM'] = np.float64(Params['UNITLENGTH_IN_CM'])
Params['UNITMASS_IN_G'] = np.float64(Params['UNITMASS_IN_G'])
Params['UNITVELOCITY_IN_CM_PER_S'] = np.float64(Params['UNITVELOCITY_IN_CM_PER_S'])
if Params['COMOVINGIC'] != 0 and Params['COMOVINGIC'] != 1:
    print("Error: the COMOVINGIC parameter should be 1 or 0!\nExiting.\n")
    sys.exit(2)
print("IC generator parameters:\n------------------------")
if Params['ICGENERATORTYPE'] == 0:
    print("IC generator:\t2LPTic")
elif Params['ICGENERATORTYPE'] == 1:
    print("IC generator:\tNgenIC")
elif Params['ICGENERATORTYPE'] == 2:
    print("IC generator:\tL-genIC")
else:
    print("Error: unkown IC generator!\nExiting.\n")
    sys.exit(2)
print("Executable:\t%s\n" % Params['EXECUTABLE'])
if (Params['BIN_MODE'] > 1) or (Params['BIN_MODE'] < 0):
    print("Error: unkown binning mode %i!\nExiting.\n" % Params['BIN_MODE'])
    sys.exit(2)
if Params['BIN_MODE'] == 0:
    print("Binning mode:\tConstant size binning in the \"omega\" compact coordinate.")
if Params['BIN_MODE'] == 1:
    print("Binning mode:\tConstant shell volumes in the compact space.")
if Params['OUTPUTFORMAT'] == 0:
    print("Output format:\t ASCII")
if Params['OUTPUTFORMAT'] == 2:
    print("Output format:\t HDF5")
    if Params['OUTPUTPRECISION'] == 0:
        print("Output precision:\t32bit")
    if Params['OUTPUTPRECISION'] == 1:
        print("Output precision:\t64bit")
    if (Params['OUTPUTPRECISION'] != 1) and (Params['OUTPUTPRECISION'] != 0):
        printf("Error: unkown output percision were set.\nExiting.")
        sys.exit()
if (Params['OUTPUTFORMAT'] != 0) and (Params['OUTPUTFORMAT'] != 2):
    print("Error: unkown output format\nExiting.")
    sys.exit()
#Calculating the density from the cosmological Parameters
rho_crit = 3*Params['H0']**2/(8*np.pi)/UNIT_V/UNIT_V #in internal units
rho_mean = Params['OMEGAM']*rho_crit
#Loading the input glass:
print("Loading the %s input glass file..." % Params['GLASSFILE'])
input_glass = np.fromfile(Params['GLASSFILE'], count=-1, sep='\t', dtype=np.float32)
input_glass = input_glass.reshape(int(len(input_glass)/7),7)
Npart = len(input_glass)
print("...done.")
#Calculating the total mass, and the average density
M_tot = np.sum(input_glass[:,6])
V_sim = 4.0*np.pi/3.0*Params['RSIM']**3
OmegaM_mean_input = (M_tot/V_sim)/rho_crit
if np.absolute(OmegaM_mean_input/Params['OMEGAM']-1.0) < 1e-9:
    print("\nThe cosmological Omega_m parameter, calculated from the particle masses:\tOmega_m=%f\n" % (OmegaM_mean_input))
else:
    input_glass[:,6] = input_glass[:,6] * Params['OMEGAM'] / OmegaM_mean_input
    print("\nThe particle masses were rescaled to fit with the cosmological parameter Omega_m=%f\n" % (Params['OMEGAM']))
original_glass = np.copy(input_glass)
#Calculating the Mass list:
Mass_list = np.unique(input_glass[:,6])
print("Number of different masses:\t%i\n" % len(Mass_list))
#Periodically shifting the input glass:
print("Periodically shifting the input glass...")
input_glass[:,0] = input_glass[:,0]+Params['VOIX']
input_glass[:,1] = input_glass[:,1]+Params['VOIY']
input_glass[:,2] = input_glass[:,2]+Params['VOIZ']
for i in range(0, Npart):
    for k in range(0,3):
        if input_glass[i,k]<0:
            input_glass[i,k] += Params['LBOX']
        if input_glass[i,k]>Params['LBOX']:
            input_glass[i,k] -= Params['LBOX']
print("...done.\n")
#Converting the input glass to gadget format:
print("Converting the input glass to Gadget format...")
output_glassfile = Params['OUTDIR'] + Params['FILEBASE'] + "_Glass_tmp.dat"
np.savetxt(output_glassfile, input_glass, delimiter='\t')
gadget_glassfile = Params['OUTDIR'] + Params['FILEBASE'] + "_GLASS"
ascii2gadget(output_glassfile, gadget_glassfile, Params['LBOX'], Params['H0'], Params['UNITLENGTH_IN_CM'])
M_tot_box = rho_mean*Params['LBOX']**3
print("...done.\n")
if Params['NMESH'] == 0:
    #Calculating the Nsample-Mass function:
    Nsample_func = np.zeros((len(Mass_list),2))
    Nsample_func[:,0] = Mass_list[:]
    Nsample_func[:,1] = np.uint32(np.cbrt(M_tot_box/Mass_list[:]))
    Nsample_tab = np.zeros(Params['NGRIDSAMPLES'],dtype=np.uint32)
    Mass_tab = np.zeros(Params['NGRIDSAMPLES'],dtype=np.uint32)
    delta_Nsample = np.uint32(np.ceil(len(Mass_list)/Params['NGRIDSAMPLES']))
    print("The generated Nsample list:")
    print("ID\tNsample\tMass(in 10e11Msol)")
    Nsample_tab[len(Nsample_tab)-1] = Nsample_func[0,1]
    print("%i\t%i\t%e" % (len(Nsample_tab)-1, Nsample_tab[len(Nsample_tab)-1], Nsample_func[0,0]))
    for i in range(len(Nsample_tab)-2, -1, -1):
        Nsample_tab[i] = Nsample_func[len(Nsample_func)-1-i*delta_Nsample,1]
        Mass_tab[i] = Nsample_func[len(Nsample_func)-1-i*delta_Nsample,0]
        print("%i\t%i\t%e" % (i, Nsample_tab[i], Mass_tab[i]))
    #generating paramfiles
    paramfile_name=len(Nsample_tab)*[None]
    for i in range(0,len(Nsample_tab)):
        paramfile_name[i] = Params['OUTDIR'] + Params['FILEBASE'] + "_%i" % i + ".param"
        if Params['ICGENERATORTYPE'] == 0:
            Write_2LPTic_paramfile(paramfile_name[i], Nsample_tab[i], Nsample_tab[i], Params['LBOX']*UNIT_D/Params['UNITLENGTH_IN_CM']*(Params['H0']/100.0), Params['FILEBASE'] + "_%i" % i, Params['OUTDIR'], gadget_glassfile, Params['OMEGAM'], Params['OMEGAL'], Params['OMEGAB'], Params['H0']/100.0, Params['REDSHIFT'], Params['SIGMA8'], Params['SPHEREMODE'], Params['WHICHSPECTRUM'], Params['FILEWITHINPUTSPECTRUM'], Params['SHAPEGAMMA'], Params['PRIMORDIALINDEX'], Params['SEED'],Params['UNITLENGTH_IN_CM'],Params['UNITMASS_IN_G'],Params['UNITVELOCITY_IN_CM_PER_S'],Params['INPUTSPECTRUM_UNITLENGTH_IN_CM'])
        elif Params['ICGENERATORTYPE'] == 1:
            Write_NgenIC_paramfile(paramfile_name[i], Nsample_tab[i], Nsample_tab[i], Params['LBOX']*UNIT_D/Params['UNITLENGTH_IN_CM']*(Params['H0']/100.0), Params['FILEBASE'] + "_%i" % i, Params['OUTDIR'], gadget_glassfile, Params['OMEGAM'], Params['OMEGAL'], Params['OMEGAB'], Params['H0']/100.0, Params['REDSHIFT'], Params['SIGMA8'], Params['SPHEREMODE'], Params['WHICHSPECTRUM'], Params['FILEWITHINPUTSPECTRUM'], Params['RENORMALIZEINPUTSPECTRUM'], Params['SHAPEGAMMA'], Params['PRIMORDIALINDEX'], Params['SEED'],Params['UNITLENGTH_IN_CM'],Params['UNITMASS_IN_G'],Params['UNITVELOCITY_IN_CM_PER_S'],Params['INPUTSPECTRUM_UNITLENGTH_IN_CM'])
        elif Params['ICGENERATORTYPE'] == 2:
            Write_LgenIC_paramfile(paramfile_name[i], Nsample_tab[i], Nsample_tab[i], Params['LBOX']*UNIT_D/Params['UNITLENGTH_IN_CM']*(Params['H0']/100.0), Params['FILEBASE'] + "_%i" % i, Params['OUTDIR'], gadget_glassfile, Params['OMEGAM'], Params['OMEGAL'], Params['OMEGAB'], Params['H0']/100.0, Params['REDSHIFT'], Params['SIGMA8'], Params['SPHEREMODE'], Params['WHICHSPECTRUM'], Params['FILEWITHINPUTSPECTRUM'], Params['SHAPEGAMMA'], Params['PRIMORDIALINDEX'], Params['SEED'],Params['UNITLENGTH_IN_CM'],Params['UNITMASS_IN_G'],Params['UNITVELOCITY_IN_CM_PER_S'],Params['INPUTSPECTRUM_UNITLENGTH_IN_CM'])
        else:
            print("Error: unkown IC generator!\nExiting.\n")
            sys.exit(2)
    #generating the ICs
    for i in range(0,len(Nsample_tab)):
        print("\n--------------------------------\nExecuting:\n " + "mpirun " + Params['EXECUTABLE'] + " " + paramfile_name[i] + "\nNsample=%i\n\n" % Nsample_tab[i])
        call(["mpirun", "-n", str(Params['MPITASKS']), Params['EXECUTABLE'], paramfile_name[i]])
    #Calculating the displacement field for every NSAMPLE:
    print("Calculating the displacement and velocity field for every Nsample...")
    Disp_field = np.zeros( ( Params['NGRIDSAMPLES'], Npart, 3), dtype=np.float32)
    Vel_field = np.zeros( ( Params['NGRIDSAMPLES'], Npart, 3), dtype=np.float32)
    for i in range(0,len(Nsample_tab)):
        print("    i=%i\tNsample=%i" % (i,Nsample_tab[i]))
        X_tmp = np.zeros( (Npart,3), dtype=np.float32)
        V_tmp = np.zeros( (Npart,3), dtype=np.float32)
        if Params['MPITASKS'] == 1:
            #reading only 1 gadget file
            if Params['ICGENERATORTYPE'] == 2:
                print("    Loading the " + Params['OUTDIR'] + Params['FILEBASE'] + "_%i" % i + ".0" + " file...")
                snapshot = glio.GadgetSnapshot(Params['OUTDIR'] + Params['FILEBASE'] + "_%i" % i + ".0")
            else:
                print("    Loading the " + Params['OUTDIR'] + Params['FILEBASE'] + "_%i" % i + " file...")
                snapshot = glio.GadgetSnapshot(Params['OUTDIR'] + Params['FILEBASE'] + "_%i" % i)
                snapshot.load()
                X_tmp = snapshot.pos[1] / (Params['H0'] / 100.0) * Params['UNITLENGTH_IN_CM']/UNIT_D
                V_tmp = snapshot.vel[1]
        else:
            #reading multiple gadget file
            for j in range(0,Params['MPITASKS']):
                print("    Loading the " + Params['OUTDIR'] + Params['FILEBASE'] + "_%i" % i + ".%i" % j + " file...")
                snapshot = glio.GadgetSnapshot(Params['OUTDIR'] + Params['FILEBASE'] + "_%i" % i + ".%i" % j)
                snapshot.load()
                N_in_this_file=snapshot.header.npart[1]
                for k in range(0,N_in_this_file):
                    #The IDs are shifted with 1
                    index_of_this_particle=snapshot.ID[1][k]-1
                    X_tmp[index_of_this_particle] = snapshot.pos[1][k] / (Params['H0'] / 100.0) * Params['UNITLENGTH_IN_CM']/UNIT_D
                    V_tmp[index_of_this_particle] = snapshot.vel[1][k]

        print("    ...done.\n    Calculating the displacement field...")
        Disp_field[i,:,:] = X_tmp-input_glass[:,0:3]
        for j in range(0,Npart):
            for k in range(0,3):
                if np.absolute(Disp_field[i,j,k]) >= Params['LBOX']/2.0:
                    if Disp_field[i,j,k]>0:
                        Disp_field[i,j,k] -= Params['LBOX']
                    else:
                        Disp_field[i,j,k] += Params['LBOX']
        print("    Average displacement: %f Mpc" %  np.mean(np.sqrt(Disp_field[i,:,0]**2 + Disp_field[i,:,1]**2 + Disp_field[i,:,2]**2)))
        print("    Maximal displacement: %f Mpc" % np.max(np.sqrt(Disp_field[i,:,0]**2 + Disp_field[i,:,1]**2 + Disp_field[i,:,2]**2)))
        print("    ...done.\n    Calculating the velocity field...")
        Vel_field[i,:,:] = V_tmp #saving in km/s
        print("    Average velocity: %f km/s" % (np.mean(np.sqrt(Vel_field[i,:,0]**2 + Vel_field[i,:,1]**2 + Vel_field[i,:,2]**2))))
        print("    ...done.\n")
    print("...done\n")
    del(V_tmp)
    del(X_tmp)
    print("Interpolating between the different Nsamples and generating the final IC...")
    IC = np.zeros((Npart,7))
    IC[:,6] = original_glass[:,6] #masses
    for i in range(0,Npart):
        for k in range(0,3):
            #interpolation in the coordinate-space
            IC[i,k] = original_glass[i,k] + np.interp(IC[i,6],Mass_tab,Disp_field[:,i,k])
            #interpolation in the velocity-space
            IC[i,k+3] = original_glass[i,k+3] + np.interp(IC[i,6],Mass_tab,Vel_field[:,i,k])
    print("...done\n")
else:
    #In this case, the script only generates one displacement field
    paramfile_name = Params['OUTDIR'] + Params['FILEBASE'] + ".param"
    Nsample = np.uint32(np.ceil(np.cbrt(M_tot_box/np.min(Mass_list))))
    if Params['ICGENERATORTYPE'] == 0:
        Write_2LPTic_paramfile(paramfile_name, Params['NMESH'], Nsample, Params['LBOX']*UNIT_D/Params['UNITLENGTH_IN_CM']*(Params['H0']/100.0), Params['FILEBASE'], Params['OUTDIR'], gadget_glassfile, Params['OMEGAM'], Params['OMEGAL'], Params['OMEGAB'], Params['H0']/100.0, Params['REDSHIFT'], Params['SIGMA8'], Params['SPHEREMODE'], Params['WHICHSPECTRUM'], Params['FILEWITHINPUTSPECTRUM'], Params['SHAPEGAMMA'], Params['PRIMORDIALINDEX'], Params['SEED'],Params['UNITLENGTH_IN_CM'],Params['UNITMASS_IN_G'],Params['UNITVELOCITY_IN_CM_PER_S'],Params['INPUTSPECTRUM_UNITLENGTH_IN_CM'])
    elif Params['ICGENERATORTYPE'] == 1:
        Write_NgenIC_paramfile(paramfile_name, Params['NMESH'], Nsample, Params['LBOX']*UNIT_D/Params['UNITLENGTH_IN_CM']*(Params['H0']/100.0), Params['FILEBASE'], Params['OUTDIR'], gadget_glassfile, Params['OMEGAM'], Params['OMEGAL'], Params['OMEGAB'], Params['H0']/100.0, Params['REDSHIFT'], Params['SIGMA8'], Params['SPHEREMODE'], Params['WHICHSPECTRUM'], Params['FILEWITHINPUTSPECTRUM'], Params['RENORMALIZEINPUTSPECTRUM'], Params['SHAPEGAMMA'], Params['PRIMORDIALINDEX'], Params['SEED'],Params['UNITLENGTH_IN_CM'],Params['UNITMASS_IN_G'],Params['UNITVELOCITY_IN_CM_PER_S'],Params['INPUTSPECTRUM_UNITLENGTH_IN_CM'])
    elif Params['ICGENERATORTYPE'] == 2:
        Write_LgenIC_paramfile(paramfile_name, Params['NMESH'], Nsample, Params['LBOX']*UNIT_D/Params['UNITLENGTH_IN_CM']*(Params['H0']/100.0), Params['FILEBASE'], Params['OUTDIR'], gadget_glassfile, Params['OMEGAM'], Params['OMEGAL'], Params['OMEGAB'], Params['H0']/100.0, Params['REDSHIFT'], Params['SIGMA8'], Params['SPHEREMODE'], Params['WHICHSPECTRUM'], Params['FILEWITHINPUTSPECTRUM'], Params['SHAPEGAMMA'], Params['PRIMORDIALINDEX'], Params['SEED'],Params['UNITLENGTH_IN_CM'],Params['UNITMASS_IN_G'],Params['UNITVELOCITY_IN_CM_PER_S'],Params['INPUTSPECTRUM_UNITLENGTH_IN_CM'])
    else:
        print("Error: unkown IC generator!\nExiting.\n")
        sys.exit(2)
    print("\n--------------------------------\nExecuting:\n " + "mpirun " + Params['EXECUTABLE'] + " " + paramfile_name + "\nNsample=%i\n\n" % Nsample)
    call(["mpirun", "-np", str(Params['MPITASKS']), Params['EXECUTABLE'], paramfile_name])
    #Calculating the displacement field:
    print("Calculating the displacement and velocity field...")
    Disp_field = np.zeros( (Npart, 3), dtype=np.float32)
    Vel_field = np.zeros( (Npart, 3), dtype=np.float32)
    if Params['MPITASKS'] == 1:
        #reading only 1 gadget file
        if Params['ICGENERATORTYPE'] == 2:
            print("    Loading the " + Params['OUTDIR'] + Params['FILEBASE'] + ".0" + " file...")
            snapshot = glio.GadgetSnapshot(Params['OUTDIR'] + Params['FILEBASE'] + ".0")
        else:
            print("    Loading the " + Params['OUTDIR'] + Params['FILEBASE'] + " file...")
            snapshot = glio.GadgetSnapshot(Params['OUTDIR'] + Params['FILEBASE'])
            snapshot.load()
            Disp_field = snapshot.pos[1] / (Params['H0'] / 100.0) * Params['UNITLENGTH_IN_CM']/UNIT_D
            Vel_field = snapshot.vel[1]
    else:
        #reading multiple gadget file
        for j in range(0,Params['MPITASKS']):
            print("    Loading the " + Params['OUTDIR'] + Params['FILEBASE'] + ".%i" % j + " file...")
            snapshot = glio.GadgetSnapshot(Params['OUTDIR'] + Params['FILEBASE'] + ".%i" % j)
            snapshot.load()
            N_in_this_file=snapshot.header.npart[1]
            for k in range(0,N_in_this_file):
                #The IDs are shifted with 1
                index_of_this_particle=snapshot.ID[1][k]-1
                Disp_field[index_of_this_particle] = snapshot.pos[1][k] / (Params['H0'] / 100.0) * Params['UNITLENGTH_IN_CM']/UNIT_D
                Vel_field[index_of_this_particle] = snapshot.vel[1][k]
    print("    ...done.\n    Calculating the displacement field...")
    Disp_field[:,0:3] = Disp_field[:,0:3]-input_glass[:,0:3]
    for j in range(0,Npart):
        for k in range(0,3):
            if np.absolute(Disp_field[j,k]) >= Params['LBOX']/2.0:
                if Disp_field[j,k]>0:
                    Disp_field[j,k] -= Params['LBOX']
                else:
                    Disp_field[j,k] += Params['LBOX']
    print("    Average displacement: %f Mpc" %  np.mean(np.sqrt(Disp_field[:,0]**2 + Disp_field[:,1]**2 + Disp_field[:,2]**2)))
    print("    Maximal displacement: %f Mpc" % np.max(np.sqrt(Disp_field[:,0]**2 + Disp_field[:,1]**2 + Disp_field[:,2]**2)))
    print("    Average velocity: %f km/s" % (np.mean(np.sqrt(Vel_field[:,0]**2 + Vel_field[:,1]**2 + Vel_field[:,2]**2))))
    print("...done.\n")
    IC = np.zeros((Npart,7))
    IC[:,6] = original_glass[:,6] #masses
    IC[:,0:3] = original_glass[:,0:3] + Disp_field[:,:] #coordinates
    IC[:,3:6] = Vel_field[:,:] #velocities
    print("...done\n")

if Params['COMOVINGIC'] == 0:
    print("Rescaling the IC and adding the Hubble flow for non-comoving simulation...")
    a_start = 1.0/(Params['REDSHIFT']+1.0)
    Hubble_start = Params['H0']*np.sqrt(np.power(a_start, -3.0)*Params['OMEGAM'] + Params['OMEGAL'] + np.power(a_start, -2.0)*(1-Params['OMEGAM']-Params['OMEGAL']))
    for i in range(0,Npart):
        for k in range(0,3):
            IC[i,k] = IC[i,k] * a_start
            IC[i,k+3] = IC[i,k+3] * np.sqrt(a_start)
            IC[i,k+3] = IC[i,k+3] + IC[i,k]*Hubble_start
    print("...done\n")
if Params['OUTPUTFORMAT'] == 0:
    outputfilename = Params['OUTDIR'] + Params['FILEBASE'] + ".dat"
if Params['OUTPUTFORMAT'] == 2:
    outputfilename = Params['OUTDIR'] + Params['FILEBASE'] + ".hdf5"
print("Saving %s IC file..." % outputfilename)
if Params['OUTPUTFORMAT'] == 0:
    np.savetxt(outputfilename, IC, delimiter='\t')
if Params['OUTPUTFORMAT'] == 2:
    writeHDF5snapshot(IC, outputfilename, np.double(2.0*Params['RSIM']), Params['REDSHIFT'], Params['OUTPUTPRECISION'])
print("...done\n")
print("Calculating redshifts for the spherical shells...")
#calculating the comoving distances of the particles
shell_limits = np.zeros(np.uint64(Params['NRBINS'])+np.uint64(1), dtype=np.float64)
z_list = np.zeros(np.uint64(Params['NRBINS']), dtype=np.float64)
if Params['BIN_MODE'] == 0:
    last_cell_size = Params['NRBINS']*np.pi/(2*np.arctan(Params['RSIM']/Params['D_S']))-Params['NRBINS']
i = np.arange(Params['NRBINS'])
if Params['BIN_MODE'] == 0:
    r_list = Calculate_r_i(i, Params['D_S'], Params['NRBINS'], last_cell_size)
if Params['BIN_MODE'] == 1:
    r_list = Calculate_r_i_cvol(i, Params['D_S'], Params['NRBINS'], Params['RSIM'])
del(i)
i = np.arange(Params['NRBINS']+1)
if Params['BIN_MODE'] == 0:
    shell_limits = Calculate_rlimits_i(i, Params['D_S'], Params['NRBINS'], last_cell_size)
if Params['BIN_MODE'] == 1:
    shell_limits = Calculate_rlimits_i_cvol(i, Params['D_S'], Params['NRBINS'], Params['RSIM'])
#calculating redshift-comoving distance function for the redshift cone
cosmo = LambdaCDM(H0=Params['H0'], Om0=Params['OMEGAM'], Ode0=Params['OMEGAL'])
for i in range(0,len(z_list)):
    z_list[i] = z_at_value(cosmo.comoving_distance, r_list[i]*u.Mpc)
if Params['OUTPUTFORMAT'] == 0:
    outputfilename = Params['OUTDIR'] + Params['FILEBASE'] + ".dat_zbins"
if Params['OUTPUTFORMAT'] == 2:
    outputfilename = Params['OUTDIR'] + Params['FILEBASE'] + ".hdf5_zbins"
np.savetxt(outputfilename, z_list)
if Params['OUTPUTFORMAT'] == 0:
    outputfilename = Params['OUTDIR'] + Params['FILEBASE'] + ".dat_zbins_rlimits"
if Params['OUTPUTFORMAT'] == 2:
    outputfilename = Params['OUTDIR'] + Params['FILEBASE'] + ".hdf5_zbins_rlimits"
np.savetxt(outputfilename, shell_limits)
print("...done\n")
print("Deleting the temporary files...")
if Params['NMESH'] == 0:
    for i in range(0,len(Nsample_tab)):
        call(["rm", "-f", paramfile_name[i]])
        if Params['MPITASKS'] == 1:
            if Params['ICGENERATORTYPE'] == 2:
                call(["rm", "-f", (Params['OUTDIR'] + Params['FILEBASE'] + "_%i" % i + ".0")])
                call(["rm", "-f", (Params['OUTDIR'] + "inputspec_" + Params['FILEBASE'] + "_%i" % i + ".txt")])
            else:
                call(["rm", "-f", (Params['OUTDIR'] + Params['FILEBASE'] + "_%i" % i)])
                call(["rm", "-f", (Params['OUTDIR'] + "inputspec_" + Params['FILEBASE'] + "_%i" % i + ".txt")])
        else:
            for j in range(0,Params['MPITASKS']):
                call(["rm", "-f", (Params['OUTDIR'] + Params['FILEBASE'] + "_%i" % i + ".%i" % j)])
                call(["rm", "-f", (Params['OUTDIR'] + "inputspec_" + Params['FILEBASE'] + "_%i" % i + ".txt")])
else:
    call(["rm", "-f", paramfile_name])
    if Params['MPITASKS'] == 1:
        if Params['ICGENERATORTYPE'] == 2:
            call(["rm", "-f", (Params['OUTDIR'] + Params['FILEBASE'] + ".0")])
            call(["rm", "-f", (Params['OUTDIR'] + "inputspec_" + Params['FILEBASE'] + ".txt")])
        else:
            call(["rm", "-f", (Params['OUTDIR'] + Params['FILEBASE'])])
            call(["rm", "-f", (Params['OUTDIR'] + "inputspec_" + Params['FILEBASE'] + ".txt")])
    else:
        for j in range(0,Params['MPITASKS']):
            call(["rm", "-f", (Params['OUTDIR'] + Params['FILEBASE'] + ".%i" % j)])
            call(["rm", "-f", (Params['OUTDIR'] + "inputspec_" + Params['FILEBASE'] + ".txt")])
call(["rm", "-f", (Params['OUTDIR'] + Params['FILEBASE'] + "_GLASS")])
call(["rm", "-f", (Params['OUTDIR'] + Params['FILEBASE'] + "_GLASS_Masses")])
call(["rm", "-f", (Params['OUTDIR'] + Params['FILEBASE'] + "_Glass_tmp.dat")])
print("...done.\n")
end = time.time()
print("The IC making took %fs.\n" % (end-start))

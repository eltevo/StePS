#!/usr/bin/env python3

#*******************************************************************************#
#  StePS_IC.py - An initial condition generator for                             #
#     STEreographically Projected cosmological Simulations                      #
#    Copyright (C) 2017-2022 Gabor Racz                                         #
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

def Write_2LPTic_paramfile(PARAMFILENAME, NMESH, NSAMPLE, LBOX, OUTFILENAME, OUTFILEDIR, INPUTFILE, Om, Ol, Ob, h, Z, SIGMA8,SPHEREMODE,WHICHSPECTRUM,FILEWITHINPUTSPECTRUM,SHAPEGAMMA,PRIMORDIALINEX,SEED,UNITLENGTH_IN_CM,UNITMASS_IN_G,UNITVELOCITY_IN_CM_PER_S,INPUTSPECTRUM_UNITLENGTH_IN_CM, PHASE_SHIFT_ENABLED, PHASESHIFT, FIXED_AMPLITUDES_ENABLED, FIXEDAMPLITUDES, RENORMALIZEINPUTSPECTRUM):
    outstring = "Nmesh            %i       %% This is the size of the FFT grid used to\n" \
    "                           %% compute the displacement field. One\n" \
    "                           %% should have Nmesh >= Nsample.\n" \
    "\n" \
    "Nsample          %i       %% sets the maximum k that the code uses,\n" \
    "                           %% i.e. this effectively determines the\n" \
    "                           %% Nyquist frequency that the code assumes,\n" \
    "                           %% k_Nyquist = 2*PI/Box * Nsample/2\n" \
    "                           %% Normally, one chooses Nsample such that\n" \
    "                           %% Ntot =  Nsample^3, where Ntot is the\n" \
    "                           %% total number of particles\n" \
    "\n" \
    "\n" \
    "Box              %.15f %% Periodic box size of simulation\n" \
    "\n"  \
    "FileBase         %s                 %% Base-filename of output files\n" \
    "OutputDir        %s               %% Directory for output\n" \
    "\n" \
    "GlassFile        %s  %% File with unperturbed glass or\n" \
    "                                  %% Cartesian grid\n" \
    "\n" \
    "OmegaDM_2ndSpecies      0\n" \
    "GlassTileFac            1\n" \
    "WDM_On                  0\n" \
    "WDM_Vtherm_On           0\n" \
    "WDM_PartMass_in_kev     1\n" \
    "\n" \
    "Omega            %.8f       %% Total matter density  (at z=0)\n" \
    "OmegaLambda      %.8f       %% Cosmological constant (at z=0)\n" \
    "OmegaBaryon      %.8f       %% Baryon density        (at z=0)\n" \
    "HubbleParam      %.8f       %% Hubble paramater (may be used for power spec parameterization)\n" \
    "\n" \
    "Redshift         %.15f        %% Starting redshift\n" \
    "\n" \
    "Sigma8           %.10f       %% power spectrum normalization\n" \
    "\n" \
    "\n" \
    "\n" \
    "SphereMode       %i         %% if \"1\" only modes with |k| < k_Nyquist are\n" \
    "                           %% used (i.e. a sphere in k-space), otherwise modes with\n" \
    "                           %% |k_x|,|k_y|,|k_z| < k_Nyquist are used\n" \
    "                           %% (i.e. a cube in k-space)\n" \
    "\n" \
    "\n" \
    "WhichSpectrum    %i         %% \"1\" selects Eisenstein & Hu spectrum,\n" \
    "                           %% \"2\" selects a tabulated power spectrum in\n" \
    "                           %% the file 'FileWithInputSpectrum'\n" \
    "                           %% otherwise, Efstathiou parametrization is used\n" \
    "\n" \
    "\n" \
    "FileWithInputSpectrum   %s  %% filename of tabulated input\n" \
    "                                            %% spectrum (if used)\n" \
    "InputSpectrum_UnitLength_in_cm  %s %% defines length unit of tabulated\n" \
    "                                            %% input spectrum in cm/h.\n" \
    "                                            %% Note: This can be chosen different from UnitLength_in_cm\n" \
    "\n" \
    "\n" \
    "\n" \
    "ShapeGamma       %f      %% only needed for Efstathiou power spectrum\n" \
    "PrimordialIndex  %f       %% may be used to tilt the primordial index,\n" \
    "                           %% primordial spectrum is k^PrimordialIndex\n" \
    "\n" \
    "\n" \
    "Seed             %i    %%  seed for IC-generator\n" \
    "\n" \
    "\n" \
    "NumFilesWrittenInParallel 1  %% limits the number of files that are\n" \
    "                             %% written in parallel when outputting\n" \
    "\n" \
    "\n" \
    "UnitLength_in_cm          %e   %% defines length unit of output (in cm/h)\n" \
    "UnitMass_in_g             %e      %% defines mass unit of output (in g/h)\n" \
    "UnitVelocity_in_cm_per_s  %e           %% defines velocity unit of output (in cm/sec)\n\n\n" \
    % (NMESH, NSAMPLE, LBOX, OUTFILENAME, OUTFILEDIR, INPUTFILE, Om, Ol, Ob, h, Z, SIGMA8, SPHEREMODE,WHICHSPECTRUM, FILEWITHINPUTSPECTRUM,INPUTSPECTRUM_UNITLENGTH_IN_CM,SHAPEGAMMA,PRIMORDIALINEX,SEED,UNITLENGTH_IN_CM,UNITMASS_IN_G,UNITVELOCITY_IN_CM_PER_S)
    if bool(PHASE_SHIFT_ENABLED)==True:
        outstring = outstring + "PhaseShift             %.14f        %% Phase shift in degrees\n\n\n" % PHASESHIFT
    if bool(FIXED_AMPLITUDES_ENABLED)==True:
        outstring = outstring + "FixedAmplitudes             %i        %% If set zero, the delta amplitudes will be generated from Raleigh distribution. If set to one, all generated amplitude will have fixed (average) value\n\n\n" % FIXEDAMPLITUDES
        outstring = outstring + "ReNormalizeInputSpectrum    %i        %% Renormalization of the input tabulated spectrum\n\n\n" % RENORMALIZEINPUTSPECTRUM
    paramfile = open(PARAMFILENAME, "w")
    paramfile.write(outstring)
    paramfile.close()
    return;

def Write_NgenIC_paramfile(PARAMFILENAME, NMESH, NSAMPLE, LBOX, OUTFILENAME, OUTFILEDIR, INPUTFILE, Om, Ol, Ob, h, Z, SIGMA8,SPHEREMODE,WHICHSPECTRUM,FILEWITHINPUTSPECTRUM,RENORMALIZEINPUTSPECTRUM,SHAPEGAMMA,PRIMORDIALINEX,SEED,UNITLENGTH_IN_CM,UNITMASS_IN_G,UNITVELOCITY_IN_CM_PER_S,INPUTSPECTRUM_UNITLENGTH_IN_CM, PHASE_SHIFT_ENABLED, PHASESHIFT, FIXED_AMPLITUDES_ENABLED, FIXEDAMPLITUDES):
    outstring = "Nmesh            %i       %% This is the size of the FFT grid used to\n" \
    "                           %% compute the displacement field. One\n" \
    "                           %% should have Nmesh >= Nsample.\n" \
    "\n" \
    "Nsample          %i       %% sets the maximum k that the code uses,\n" \
    "                           %% i.e. this effectively determines the\n" \
    "                           %% Nyquist frequency that the code assumes,\n" \
    "                           %% k_Nyquist = 2*PI/Box * Nsample/2\n" \
    "                           %% Normally, one chooses Nsample such that\n" \
    "                           %% Ntot =  Nsample^3, where Ntot is the\n" \
    "                           %% total number of particles\n" \
    "\n" \
    "\n" \
    "Box              %.15f %% Periodic box size of simulation\n" \
    "\n"  \
    "FileBase         %s                 %% Base-filename of output files\n" \
    "OutputDir        %s               %% Directory for output\n" \
    "\n" \
    "GlassFile        %s  %% File with unperturbed glass or\n" \
    "                                  %% Cartesian grid\n" \
    "\n" \
    "TileFac         1                %% Number of times the glass file is\n" \
    "                                 %% tiled in each dimension (must be\n" \
    "                                 %% an integer)\n" \
    "\n" \
    "Omega            %.8f       %% Total matter density  (at z=0)\n" \
    "OmegaLambda      %.8f       %% Cosmological constant (at z=0)\n" \
    "OmegaBaryon      %.8f       %% Baryon density        (at z=0)\n" \
    "HubbleParam      %.8f       %% Hubble paramater (may be used for power spec parameterization)\n" \
    "\n" \
    "Redshift         %.15f        %% Starting redshift\n" \
    "\n" \
    "Sigma8           %.10f       %% power spectrum normalization\n" \
    "\n" \
    "\n" \
    "\n" \
    "SphereMode       %i         %% if \"1\" only modes with |k| < k_Nyquist are\n" \
    "                           %% used (i.e. a sphere in k-space), otherwise modes with\n" \
    "                           %% |k_x|,|k_y|,|k_z| < k_Nyquist are used\n" \
    "                           %% (i.e. a cube in k-space)\n" \
    "\n" \
    "\n" \
    "WhichSpectrum    %i         %% \"1\" selects Eisenstein & Hu spectrum,\n" \
    "                           %% \"2\" selects a tabulated power spectrum in\n" \
    "                           %% the file 'FileWithInputSpectrum'\n" \
    "                           %% otherwise, Efstathiou parametrization is used\n" \
    "\n" \
    "\n" \
    "FileWithInputSpectrum   %s  %% filename of tabulated input\n" \
    "                                            %% spectrum (if used)\n" \
    "InputSpectrum_UnitLength_in_cm  %e %% defines length unit of tabulated\n" \
    "                                            %% input spectrum in cm/h.\n" \
    "                                            %% Note: This can be chosen different from UnitLength_in_cm\n" \
    "\n" \
    "ReNormalizeInputSpectrum   %i                %% if set to zero, the\n" \
    "                                        %% tabulated spectrum is\n" \
    "                                        %% assumed to be normalized\n" \
    "                                        %% already in its amplitude to\n" \
    "                                        %% the starting redshift,\n" \
    "                                        %% otherwise this is recomputed\n" \
    "                                        %% based on the specified sigma8\n" \
    "\n" \
    "\n" \
    "ShapeGamma       %f      %% only needed for Efstathiou power spectrum\n" \
    "PrimordialIndex  %f       %% may be used to tilt the primordial index,\n" \
    "                           %% primordial spectrum is k^PrimordialIndex\n" \
    "\n" \
    "\n" \
    "Seed             %i    %%  seed for IC-generator\n" \
    "\n" \
    "\n" \
    "NumFilesWrittenInParallel 1  %% limits the number of files that are\n" \
    "                             %% written in parallel when outputting\n" \
    "\n" \
    "\n" \
    "UnitLength_in_cm          %e   %% defines length unit of output (in cm/h)\n" \
    "UnitMass_in_g             %e      %% defines mass unit of output (in g/h)\n" \
    "UnitVelocity_in_cm_per_s  %e           %% defines velocity unit of output (in cm/sec)\n\n\n" \
    % (NMESH, NSAMPLE, LBOX, OUTFILENAME, OUTFILEDIR, INPUTFILE, Om, Ol, Ob, h, Z, SIGMA8 ,SPHEREMODE,WHICHSPECTRUM,FILEWITHINPUTSPECTRUM,INPUTSPECTRUM_UNITLENGTH_IN_CM,RENORMALIZEINPUTSPECTRUM,SHAPEGAMMA,PRIMORDIALINEX,SEED,UNITLENGTH_IN_CM,UNITMASS_IN_G,UNITVELOCITY_IN_CM_PER_S)
    if bool(PHASE_SHIFT_ENABLED)==True:
        outstring = outstring + "PhaseShift             %.14f        %% Phase shift in degrees\n\n\n" % PHASESHIFT
    if bool(FIXED_AMPLITUDES_ENABLED)==True:
        outstring = outstring + "FixedAmplitudes             %i        %% If set zero, the delta amplitudes will be generated from Raleigh distribution. If set to one, all generated amplitude will have fixed (average) value\n\n\n" % FIXEDAMPLITUDES
    paramfile = open(PARAMFILENAME, "w")
    paramfile.write(outstring)
    paramfile.close()
    return;

def Write_LgenIC_paramfile(PARAMFILENAME, NMESH, NSAMPLE, LBOX, OUTFILENAME, OUTFILEDIR, INPUTFILE, Om, Ol, Ob, h, Z, SIGMA8,SPHEREMODE,WHICHSPECTRUM,FILEWITHINPUTSPECTRUM,SHAPEGAMMA,PRIMORDIALINEX,SEED,UNITLENGTH_IN_CM,UNITMASS_IN_G,UNITVELOCITY_IN_CM_PER_S,INPUTSPECTRUM_UNITLENGTH_IN_CM, PHASE_SHIFT_ENABLED, PHASESHIFT, FIXED_AMPLITUDES_ENABLED, FIXEDAMPLITUDES):
    outstring = "Nmesh            %i       %% This is the size of the FFT grid used to\n" \
    "                           %% compute the displacement field. One\n" \
    "                           %% should have Nmesh >= Nsample.\n" \
    "\n" \
    "Nsample          %i       %% sets the maximum k that the code uses,\n" \
    "                           %% i.e. this effectively determines the\n" \
    "                           %% Nyquist frequency that the code assumes,\n" \
    "                           %% k_Nyquist = 2*PI/Box * Nsample/2\n" \
    "                           %% Normally, one chooses Nsample such that\n" \
    "                           %% Ntot =  Nsample^3, where Ntot is the\n" \
    "                           %% total number of particles\n" \
    "\n" \
    "\n" \
    "Box              %.15f %% Periodic box size of simulation\n" \
    "\n"  \
    "FileBase         %s                 %% Base-filename of output files\n" \
    "OutputDir        %s               %% Directory for output\n" \
    "\n" \
    "GlassFile        %s  %% File with unperturbed glass or\n" \
    "                                  %% Cartesian grid\n" \
    "\n" \
    "GlassTileFac         1                %% Number of times the glass file is\n" \
    "                                 %% tiled in each dimension (must be\n" \
    "                                 %% an integer)\n" \
    "\n" \
    "Omega            %.8f       %% Total matter density  (at z=0)\n" \
    "OmegaLambda      %.8f       %% Cosmological constant (at z=0)\n" \
    "OmegaBaryon      %.8f       %% Baryon density        (at z=0)\n" \
    "HubbleParam      %.8f       %% Hubble paramater (may be used for power spec parameterization)\n" \
    "\n" \
    "Redshift         %.15f        %% Starting redshift\n" \
    "\n" \
    "Sigma8           %.10f       %% power spectrum normalization\n" \
    "\n" \
    "\n" \
    "\n" \
    "SphereMode       %i         %% if \"1\" only modes with |k| < k_Nyquist are\n" \
    "                           %% used (i.e. a sphere in k-space), otherwise modes with\n" \
    "                           %% |k_x|,|k_y|,|k_z| < k_Nyquist are used\n" \
    "                           %% (i.e. a cube in k-space)\n" \
    "\n" \
    "\n" \
    "WhichSpectrum    %i         %% \"1\" selects Eisenstein & Hu spectrum,\n" \
    "                           %% \"2\" selects a tabulated power spectrum in\n" \
    "                           %% the file 'FileWithInputSpectrum'\n" \
    "                           %% otherwise, Efstathiou parametrization is used\n" \
    "\n" \
    "\n" \
    "FileWithInputSpectrum   %s  %% filename of tabulated input\n" \
    "                                            %% spectrum (if used)\n" \
    "InputSpectrum_UnitLength_in_cm  %e %% defines length unit of tabulated\n" \
    "                                            %% input spectrum in cm/h.\n" \
    "                                            %% Note: This can be chosen different from UnitLength_in_cm\n" \
    "\n" \
    "                                        %% tabulated spectrum is\n" \
    "                                        %% assumed to be normalized\n" \
    "                                        %% already in its amplitude to\n" \
    "                                        %% the starting redshift,\n" \
    "                                        %% otherwise this is recomputed\n" \
    "                                        %% based on the specified sigma8\n" \
    "\n" \
    "\n" \
    "ShapeGamma       %f      %% only needed for Efstathiou power spectrum\n" \
    "PrimordialIndex  %f       %% may be used to tilt the primordial index,\n" \
    "                           %% primordial spectrum is k^PrimordialIndex\n" \
    "\n" \
    "\n" \
    "Seed             %i    %%  seed for IC-generator\n" \
    "\n" \
    "\n" \
    "NumFilesWrittenInParallel 1  %% limits the number of files that are\n" \
    "                             %% written in parallel when outputting\n" \
    "\n" \
    "\n" \
    "UnitLength_in_cm          %e   %% defines length unit of output (in cm/h)\n" \
    "UnitMass_in_g             %e      %% defines mass unit of output (in g/h)\n" \
    "UnitVelocity_in_cm_per_s  %e           %% defines velocity unit of output (in cm/sec)\n\n\n" \
    % (NMESH, NSAMPLE, LBOX, OUTFILENAME, OUTFILEDIR, INPUTFILE, Om, Ol, Ob, h, Z, SIGMA8 ,SPHEREMODE,WHICHSPECTRUM,FILEWITHINPUTSPECTRUM,INPUTSPECTRUM_UNITLENGTH_IN_CM,SHAPEGAMMA,PRIMORDIALINEX,SEED,UNITLENGTH_IN_CM,UNITMASS_IN_G,UNITVELOCITY_IN_CM_PER_S)
    if bool(PHASE_SHIFT_ENABLED)==True:
        outstring = outstring + "PhaseShift             %.14f        %% Phase shift in degrees\n\n\n" % PHASESHIFT
    if bool(FIXED_AMPLITUDES_ENABLED)==True:
        outstring = outstring + "FixedAmplitudes             %i        %% If set zero, the delta amplitudes will be generated from Raleigh distribution. If set to one, all generated amplitude will have fixed (average) value\n\n\n" % FIXEDAMPLITUDES
    paramfile = open(PARAMFILENAME, "w")
    paramfile.write(outstring)
    paramfile.close()
    return;

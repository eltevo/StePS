   _____ _       _____   _____    _____ _____               				
  / ____| |     |  __ \ / ____|  |_   _/ ____|              
 | (___ | |_ ___| |__) | (___      | || |       _ __  _   _ 
  \___ \| __/ _ \  ___/ \___ \     | || |      | '_ \| | | |
  ____) | ||  __/ |     ____) |____| || |____ _| |_) | |_| |
 |_____/ \__\___|_|    |_______________\_____(_) .__/ \__, |
                                               | |     __/ |
                                               |_|    |___/ 
StePS_IC.py - an IC generator python script for STEreographically Projected cosmological Simulations

v0.2.0.2
Copyright (C) 2018 Gábor Rácz
	Department of Physics of Complex Systems, Eotvos Lorand University | Budapest, Hungary
	Department of Physics & Astronomy, Johns Hopkins University | Baltimore, MD, USA
ragraat@caesar.elte.hu

+----------------------------------------------------------------------+
| This program is free software; you can redistribute it and/or modify |
| it under the terms of the GNU General Public License as published by |
| the Free Software Foundation; either version 2 of the License, or    |
| (at your option) any later version.                                  |
|                                                                      |
| This program is distributed in the hope that it will be useful,      |
| but WITHOUT ANY WARRANTY; without even the implied warranty of       |
| MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the        |
| GNU General Public License for more details.                         |
+----------------------------------------------------------------------+

This code is under development!

This is an IC generator script for StePS simulations.
-written in python3
-uses an external code such as N-GenIC(http://ascl.net/1502.003) or 2LPTic(https://ascl.net/1201.005) to calculate the displacement field
-reads an input glass, and perturbates its particles
-the output can be in ASCII or in HDF5 format.

*********************************************************************************************

Depencencies:
	Python:
	-glio
	-h5py
	-yaml
	-astropy
	External:
	-NgenIC or 2LPTic

Running the script:
	type:
	./StePS_IC.py <input yaml file>

The example yaml file in the ./examples library:

RSIM: 1860.05314437555		%Linear size of the simulation

LBOX: 3720.2			%Linear size of the cube where the displacement field will be calculated

OMEGAM: 0.3089			%\ 
OMEGAL: 0.6911			%-Cosmological omega parameters
OMEGAB: 0.0000			%/

H0: 67.74			%Hubble constant

REDSHIFT: 31.0			%initial redshift

SIGMA8: 0.8159			%P(k) normalization

SPHEREMODE: 0			%Sphere mode

WHICHSPECTRUM: 0		% "0" = Efstathiou spectrum, 
				% "1" = Eisenstein & Hu spectrum,
				% "2" = a tabulated power spectrum

INPUTSPECTRUM_UNITLENGTH_IN_CM: 3.085678e24	%defines length unit of
					 	%tabulated input spectrum in cm/h

RENORMALIZEINPUTSPECTRUM: 0	

SHAPEGAMMA: 0.21		% needed for Efstathiou power spectrum

PRIMORDIALINDEX: 1.0		% needed for Efstathiou power spectrum

SEED: 123456			% random seed for the displacement field calculation

VOIX: 500.0			%\
VOIY: 500.0			%-x,y,z coordinates of the Volume Of Interest
VOIZ: 500.0			%/

NGRIDSAMPLES: 7			% Number of the calculated displacement fields

NMESH: 0			% Size of the grid used to compute the
				% displacement field. if set zero, 
				% automatically generated values will be used

COMOVINGIC: 1			% "0": output will be in non-comoving coordinates
				% "1": output will be in comoving coordinates

NRBINS: 224			% Number of radial bins

D_S: 105.0			% diameter of the four dimensional hypersphere

BIN_MODE: 0			% This parameter tells what radial binning
				% method was used in the pre-initial
				% condition making.
				% "0": Constant binning in the "omega"
				%      coordinate
				% "1": Constant volume binning in the
				%      compact space

GLASSFILE: ./3DGlasses/Glass_Nr224_Nhp32.dat	%Pre-initial condition

OUTDIR: ./ICs/			%Output directory

FILEBASE: IC_LCDM_SP_1860Mpc_Nr224_Nhp32_ds105_z31	%base filename for the output files

ICGENERATORTYPE: 0		% "0": 2LPTic
				% "1": NgenIC

EXECUTABLE: ../2LPTic/2LPTic	% Location of the external executable

FILEWITHINPUTSPECTRUM: ./input_spectrum.txt	% Input spectrum file

UNITLENGTH_IN_CM: 3.085678e24	
UNITMASS_IN_G: 1.989e43
UNITVELOCITY_IN_CM_PER_S: 1e5
MPITASKS: 2			% Number of MPI task that will be used in the IC making
OUTPUTFORMAT: 2			% Format of the output IC:
				%  "0": ASCII
				%  "2": HDF5

OUTPUTPRECISION: 0		% "0": 32bit IC
				% "1": 64bit IC

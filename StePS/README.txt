   _____ _       _____   _____
  / ____| |     |  __ \ / ____|
 | (___ | |_ ___| |__) | (___
  \___ \| __/ _ \  ___/ \___ \
  ____) | ||  __/ |     ____) |
 |_____/ \__\___|_|    |_____/

StePS - STEreographically Projected cosmological Simulations

v1.4.0.0
Copyright (C) 2017-2025 Gábor Rácz - gabor.racz@helsinki.fi
  Department of Physics, University of Helsinki | Gustaf Hällströmin katu 2, Helsinki, Finland
  Jet Propulsion Laboratory, California Institute of Technology | 4800 Oak Grove Drive, Pasadena, CA, 91109, USA
  Department of Physics of Complex Systems, Eotvos Lorand University | Pf. 32, H-1518 Budapest, Hungary
  Department of Physics & Astronomy, Johns Hopkins University | 3400 N. Charles Street, Baltimore, MD 21218

Contributors:
  2025 Viola Varga - viola.varga@helsinki.fi
    Department of Physics, University of Helsinki | Gustaf Hällströmin katu 2, Helsinki, Finland 
  2025 Balázs Pál - pal.balazs@ttk.elte.hu
    Department of Physics of Complex Systems, Eotvos Lorand University | Pf. 32, H-1518 Budapest, Hungary
    Institute for Particle and Nuclear Physics, HUN-REN Wigner Research Centre for Physics | Pf. 49, H-1525 Budapest, Hungary.



    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

An N-body code for compactified cosmological simulations.

Main features:
- Optimized to run dark matter only N-body simulations in LambdaCDM, wCDM or w0waCDM cosmology.
- Running simulations with other models are possible by using external tabulated expansion histories.
- Able to run standard periodic, cylindrical, and non-periodic spherical cosmological simulations.
- Direct [CPU & GPU], Octree (a.k.a. Barnes-Hut)[CPU only], and randomized Octree [CPU only] force calculation.
- Can be used to make periodic, quasi-periodic, cylindrical or spherical glass.
- Available for GNU/Linux and Darwin (macOS).
- Written in C++ with MPI, OpenMP and CUDA parallelization.
- Able to use multiple GPUs simultaneously in a large computing cluster.
- Supported Initial Condition formats are HDF5, Gadget2, and ASCII.
- Supported output formats are ASCII and HDF5.


*********************************************************************************************

Downloading the code:
	From command line, the source files of the StePS code can be downloaded with the

	$ git clone https://github.com/eltevo/StePS

	command.

Required libraries:
	For the successful compilation, the code needs the OpenMPI (https://www.open-mpi.org/) library. Other MPI implementations should work too.
	Optional libraries:
		-CUDA (https://developer.nvidia.com/cuda-downloads) Use only if you want to accelerate the simulations with Nvidia GPUs. Note that only GNU/Linux is supported for CUDA.
		-HDF5 (https://support.hdfgroup.org/HDF5/) This is used for reading and writing HDF5 files

Compiling:
	1, Navigate to the StePS source directory by typing
		$ cd StePS/StePS
	   to the command line.
	2, Copy the Template-LinuxGCC-Makefile, Template-LinuxICC-Makefile, or Template-Darwin-Makefile (depending on your OS and compiler) to ./Makefile by
		$ cp Template-LinuxGCC-Makefile Makefile
	   or by
		$ cp Template-LinuxICC-Makefile Makefile
	   or by
		$ cp Template-Darwin-Makefile Makefile
	3, You should specify the library directories in the Makefile. For editing the Makefile, you should type:
		$ gedit Makefile
	    for GNU/Linux. For macOS type:
		$ open -a TextEdit Makefile

	4, Define compile time options: Some features of the StePS code are controlled with compile time options in the Makefile. With this technique a more optimized executable can be generated. The following options can be found in the Makefile:
		-USE SINGLE PRECISION
			If this is set, the code will use 32bit precision in the force calculation, otherwise 64bit calculation will be used. The 32bit force calculation is ∼ 32 times faster on Nvidia GTX GPUs compared to the 64bit force calculation, and it uses half as much memory. The speedup with Nvidia Tesla cards by using single precision is ~2.
		-GLASSMAKING
			This option should be set for glass making. In this case the code will use reversed gravity. For periodic glasses, using high precision Ewald forces is highly advised. (High precision Ewald forces can be set in the parameter file)
		-HAVE_HDF5
			If this option is set, then the generated executable will be able to write the output files in HDF5 format.
		-PERIODIC
			Set this if the simulations will use periodic boundary condition. Note that there is a similar option in the parameter file. If the two options are contradicting each other, then the program will exit with an error message.
		-COSMOPARAM
			Parametrization of the background cosmology. Possible values:
				0: standard Lambda-CDM parametrization (default)
				1: wCDM dark energy parametrization
				2: w0waCDM (a.k.a. CPL) dark energy parametrization
				-1: the expansion history will be read from an external ASCII file
								 (columns: t[Gy] a(t) H(t)[km/s/Mpc])
	5, After you saved the Makefile, compile the code with the

		$ make

	command. After the successful compilation, you can find the compiled binary in the "build/" directory

*********************************************************************************************

Once you compiled the code, you can simply run it by typing:
	$ export OMP_NUM_THREADS=<Number of shared memory OMP threads per MPI tasks>
	$ mpirun -np <number of MPI tasks> ./build/StePS <parameterfile>
  or
  $ mpirun -np <number of MPI tasks> ./buildStePS <parameterfile> <Number of shared memory OMP threads per MPI tasks>
where the parameterfile specifies the parameters of the simulation.

If you compiled the code with CUDA, you can simply run it by typing:
	$ export OMP_NUM_THREADS=<Number of GPUs per tasks>
  $ mpirun -np <number of MPI tasks> ./build/StePS_CUDA <parameterfile> <Number of GPUs per tasks>

*********************************************************************************************

Example simulations:

The StePS code git repository contains runnable example simulations in the
  StePS/examples
directory.

Standard comoving Lambda-CDM simulation:
  The first example simulation simulates the evolution of the dark matter structures in comoving coordinates in standard LCDM cosmology with Planck 2018 parameters. The initial conditions for this simulation can be found in the
    /Users/gaborr/Work/StePS/StePS/examples/ic/IC_LCDM_SP_1860Mpc_Nr224_Nhp32_ds105_z63_VOI100.hdf5
  file. This file was generated with the StePS_IC.py script.
  This example simulation can be started with the
    $ ./build/StePS_CUDA ./examples/LCDM_SP_1860_com_VOI100.param 1
    command for GPU accelerated simulation (on one node with one GPU), or with
    $ ./build/StePS ./examples/LCDM_SP_1860_com_VOI100.param
    command for CPU only simulation.
  Since this simulation contains ~1.8million particles, using GPU acceleration is highly advised.
  After a successful run, the results of this simulation can be found in the
    StePS/examples/LCDM_SP_1860_com_VOI100
  directory in hdf5 format.

Standard non-comoving Lambda-CDM simulation:
  The second example simulation is almost the same as the first (standard LCDM cosmology with Planck 2018 parameters), but it uses non-comoving physical coordinates.
  The initial condition (IC) for this example is not in the StePS git repository, but it can be generated by the StePS_IC.py script by navigating to its directory with the
    $ cd ../StePS_IC/src
  command, and start the IC generation with
    $ ./StePS_IC.py ../examples/LCDM_SP_1860Mpc_Nr224_Nhp32_ds105_noncomoving.yaml
  This is a highly memory intensive process: depending on your system, you need to have at least 32GB of RAM to be able to generate the IC. Note that you have to have a working NgenIC of 2LPTic binary to generate IC with StePS_IC.py. For more information about the IC generation see the README.txt at the StePS_IC directory.
  Once the IC is available, and you navigated back to the StePS directory with the
    $ cd ../../StePS/
    command, you can start the simulation with
    $ ./build/StePS_CUDA ./examples/LCDM_SP_1860_noncom_VOI100.param 1
    for GPU accelerated simulation (on one node with one GPU), or with
    $ ./build/StePS ./examples/LCDM_SP_1860_noncom_VOI100.param
  for CPU only simulation.
  After a successful run, the results of this simulation can be found in the
    StePS/examples/LCDM_SP_1860_noncom_VOI100
  directory in hdf5 format.



*********************************************************************************************

Output format for the particle data files if the output format is set to ASCII:

z*.dat, t*.dat:
        x[Mpc]  y[Mpc]  z[Mpc]  v_x[km/s] v_y[km/s] v_z[km/s] M[1e11M_sol]

redshift_cone.dat:
	x[Mpc]  y[Mpc]  z[Mpc]  v_x[km/s] v_y[km/s] v_z[km/s] M[1e11M_sol]	R[Mpc]	z(=Redshift)


Output format for the logfile:

Logfile.dat:
	t[Gy]	error	h[Gy](=length of timestep)	a(=scalefactor)	z(=Redshift)	H[km/s/Mpc](=Hubble parameter)	q(=deceleration parameter)	Omega_m
*********************************************************************************************

The example parameterfile (comoving LCDM):
Cosmological parameters:
------------------------
Omega_b         0.0			%Barionic matter density parameter
Omega_lambda    0.6889			%Cosmological constant density parameters
Omega_dm        0.3111			%Dark matter density parameter
Omega_r         0.0			%Radiation density parameter
HubbleConstant  67.66			%Hubble-constant
a_start         0.015625			%Initial scalefactor ( in both COMOVING_INTEGRATION=1 and COMOVING_INTEGRATION=0 cases)
a_max           1.0				%The final scalefactor (if COMOVING_INTEGRATION=1) or the final physical time (if COMOVING_INTEGRATION=0)


Simulation parameters:
-----------------------
COSMOLOGY       1			%1=cosmological simulation 0=traditional n-body sim.
IS_PERIODIC     0						%Boundary condition 0=vacuum boundaries, 1=nearest images (a.k.a. quasi-periodic), 2=Ewald forces, 3>=high precision Ewald forces
COMOVING_INTEGRATION    1					%Comoving integration 0=no, 1=yes, used only when  COSMOLOGY=1
L_BOX           1860.0531					%Linear size of the simulation volume
IC_FILE 	./examples/ic/IC_LCDM_SP_1860Mpc_Nr224_Nhp32_ds105_z63_VOI100_notcomoving.hdf5	%ic file
IC_FORMAT       2						%Ic file format 0: ascii, 1:GADGET, 2:(Gadget-)HDF5
OUT_DIR         ./examples/LCDM_SP_1860_noncom_VOI100/		%output directory
OUT_LST         ./examples/outtimes2.txt	%output list file (if it is not availabe, then FIRST_T_OUT and H_OUT parameters will be used)
OUTPUT_TIME_VARIABLE	1					%Output time variable used in OUT_LST or in FIRST_T_OUT and H_OUT. 0: physical time in Gy, 1: redshift
OUTPUT_FORMAT   2						%Output format 0: ASCII 2: (Gadget-)HDF5
REDSHIFT_CONE   0						%0: standard output files 1: one output redshift cone file
MIN_REDSHIFT    0.02477117					%The minimal output redshift. Lower redshifts considered 0. Only used in redshift cone simulations.
ACC_PARAM	0.005						%Accuracy parameter (using 0.012 results ~1% accuracy in the power spectrum)
RADIAL_FORCE_ACCURACY   1000    %Sets the number of integration steps when calculating the force lookup table, only used in cylindrical simulations
RADIAL_FORCE_TABLE_SIZE 1000    %Sets the size of the force lookup table, only used in cylindrical simulations
STEP_MIN           0.00025						%Minimal timestep length (in Gy)
STEP_MAX           0.03125						%Maximal timestep length (in Gy)
PARTICLE_RADII   0.134226516867827				%Softening length of particle with minimal mass (in comoving units, if COMOVING_INTEGRATION=1, otherwise in physical units)
FIRST_T_OUT     0.50						%First output time in Gy, if OUTPUT_TIME_VARIABLE=0; First output redshift, if OUTPUT_TIME_VARIABLE=1;
H_OUT           0.50						%Output frequency in Gy, if OUTPUT_TIME_VARIABLE=0; Output frequency in redshift, if OUTPUT_TIME_VARIABLE=1;
SNAPSHOT_START_NUMBER	0					%Initial snapshot number. Useful for restarting simulations.
H_INDEPENDENT_UNITS   0         %Units of the I/O files. 0: i/o in Mpc, Msol, etc. (default); 1: i/o in Mpc/h, Msol/h, etc.
TIME_LIMIT_IN_MIN     3600      %Simulation wall-clock time limit in minutes. If 0, or not defined, then no time limit will be considered.

Optional BH parameters: % These parameters only used in octree force calculation mode in spherical and cylindrical topology
-----------------------
RADIAL_BH_FORCE_CORRECTION          1     % 0: No radial octree force correction. 1: Radial correction of the Harnes-Hut force calculation is on.
GLASS_FILE_FOR_BH_FORCE_CORRECTION  None  % Location of the initial glass file for estimating the correction. If None, the IC will be used to calculate the correction table.
RADIAL_BH_FORCE_TABLE_SIZE          128   % Size of the radial correction table.
RADIAL_BH_FORCE_TABLE_ITERATION     16    % Number of iterations used in the radial correction table calculation, if randomized BH force calculation is turned on in the makefile.

Optional cosmological parameters:    %These parameters are only needed when alternative cosmology parametrizations are turned on in the makefile.
---------------------------------
w0    -0.9                        %Dark energy equation of state at z=0 in wCDM and w0waCDM parametrization. (LCDM: w0=-1.0)
wa    0.1                         %Negative derivative of the dark energy equation of state in w0waCDM parametrization. (LCDM: wa=0.0)
EXPANSION_FILE      ./wpwaCDM.dat %input file with tabulated expansion history. Columns in the file: age [Gy], scale factor [dimensionless], Hubble parameter [km/s/Mpc]
INTERPOLATION_ORDER 3             %order of the interpolation while using tabulated expansion history (recommended value: 3) (possible values: 1,2,or 3)

*********************************************************************************************

Acknowledgement
  The development of this code has been supported by Department of Physics of Complex Systems, ELTE.
  GR would like to thank the Department of Physics & Astronomy, JHU for supporting this work.
  GR acknowledges sponsorship of a NASA Postdoctoral Program Fellowship. GR was supported by JPL, which is run under contract by California Institute of Technology for NASA.
  The developer acknowledges support from the National Science Foundation (NSF) award 1616974.
  GR acknowledges the support of the Research Council of Finland grant 354905 and the support by the European Research Council via ERC Consolidator grant KETJU (no. 818930).

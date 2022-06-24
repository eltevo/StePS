   _____ _       _____   _____
  / ____| |     |  __ \ / ____|
 | (___ | |_ ___| |__) | (___
  \___ \| __/ _ \  ___/ \___ \
  ____) | ||  __/ |     ____) |
 |_____/ \__\___|_|    |_____/

StePS - STEreographically Projected cosmological Simulations

v1.0.1.0
Copyright (C) 2017-2022 Gábor Rácz
  Jet Propulsion Laboratory, California Institute of Technology | 4800 Oak Grove Drive, Pasadena, CA, 91109, USA
  Department of Physics of Complex Systems, Eotvos Lorand University | Pf. 32, H-1518 Budapest, Hungary
  Department of Physics & Astronomy, Johns Hopkins University | 3400 N. Charles Street, Baltimore, MD 21218
gabor.racz@jpl.nasa.gov

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

A direct N-body code for compactified cosmological simulations.

Main features:
- Optimized to run dark matter only N-body simulations in LambdaCDM, wCDM or w0waCDM cosmology.
- Running simulations with different models are possible by using external tabulated expansion histories.
- Able to run standard periodic and non-periodic spherical cosmological simulations.
- Can be used to make periodic, quasi-periodic or spherical glass.
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
	$ mpirun -n <number of MPI tasks> ./StePS <parameterfile>
  or
  $ mpirun -n <number of MPI tasks> ./StePS <parameterfile> <Number of shared memory OMP threads per MPI tasks>
where the parameterfile specifies the parameters of the simulation.

If you compiled the code with CUDA, you can simply run it by typing:
	$ export OMP_NUM_THREADS=<Number of GPUs per tasks>
  $ mpirun -n <number of MPI tasks> ./StePS_CUDA <parameterfile> <Number of GPUs per tasks>


*********************************************************************************************

Output format for the particle data files in ASCII format:

z*.dat, t*.dat:
        x[Mpc]  y[Mpc]  z[Mpc]  v_x[km/s] v_y[km/s] v_z[km/s] M[1e11M_sol]

redshift_cone.dat:
	x[Mpc]  y[Mpc]  z[Mpc]  v_x[km/s] v_y[km/s] v_z[km/s] M[1e11M_sol]	R[Mpc]	z(=Redshift)


Output format for the logfile:

Logfile.dat:
	t[Gy]	error	h[Gy](=length of timestep)	a(=scalefactor)	z(=Redshift)	H[km/s/Mpc](=Hubble parameter)	q(=deceleration parameter)	Omega_m
*********************************************************************************************

The example parameterfile:
Cosmological parameters:
------------------------
Omega_b         0.0			%Barionic matter density parameter
Omega_lambda    0.6911			%Cosmological constant density parameters
Omega_dm        0.3089			%Dark matter density parameter
Omega_r         0.0			%Radiation density parameter
HubbleConstant  67.74			%Hubble-constant
a_start         0.05			%Initial scalefactor
a_max           1.0				%The final scalefactor


Simulation parameters:
-----------------------
COSMOLOGY       1			%1=cosmological simulation 0=traditional n-body sim.
IS_PERIODIC     0						%Boundary condition 0=none, 1=nearest images, 2=Ewald forces, 3=high precision Ewald forces
COMOVING_INTEGRATION    1					%Comoving integration 0=no, 1=yes, used only when  COSMOLOGY=1
L_BOX           1860.0531					%Linear size of the simulation volume
IC_FILE         ../examples/ic/IC_SP_LCDM_1260_343M_com_VOI_1000.dat	%ic file
IC_FORMAT       0						%Ic file format 0: ascii, 1:GADGET, 2:(Gadget-)HDF5
OUT_DIR         ../examples/LCDM_SP_1260_343M_com_VOI_1000/		%output directory
OUT_LST         ../examples/ic/IC_SP_LCDM_1260_343M_com_VOI_1000.dat_zbins	%output list file
OUTPUT_TIME_VARIABLE	1					%Output time variable 0: physical time, 1: redshift
OUTPUT_FORMAT   2						%Output format 0: ASCII 2: (Gadget-)HDF5
REDSHIFT_CONE   1						%0: standard output files 1: one output redshift cone file
MIN_REDSHIFT    0.02477117					%The minimal output redshift. Lower redshifts considered 0. Only used in redshift cone simulations.
ACC_PARAM	0.030						%Accuracy parameter
STEP_MIN           0.00025						%Minimal timestep length (in Gy)
STEP_MAX           0.03125						%Maximal timestep length (in Gy)
PARTICLE_RADII   0.134226516867827				%Softening length of particle with minimal mass
FIRST_T_OUT     0.50						%First output time
H_OUT           0.50						%Output frequency
SNAPSHOT_START_NUMBER	0					%Initial snapshot number. Useful for restarting simulations.
H_INDEPENDENT_UNITS  0         %Units of the I/O files. 0: i/o in Mpc, Msol, etc. (default); 1: i/o in Mpc/h, Msol/h, etc.

Optional parameters:    %These parameters are only needed when alternative cosmology parametrizations are turned on in the makefile.
--------------------
w0    -0.9                        %Dark energy equation of state at z=0 in wCDM and w0waCDM parametrization. (LCDM: w0=-1.0)
wa    0.1                         %Negative derivative of the dark energy equation of state in w0waCDM parametrization. (LCDM: wa=0.0)
EXPANSION_FILE      ./wpwaCDM.dat %input file with tabulated expansion history. Columns in the file: age [Gy], scale factor [dimensionless], Hubble parameter [km/s/Mpc]
INTERPOLATION_ORDER 3             %order of the interpolation while using tabulated expansion history (recommended value: 3) (possible values: 1,2,or 3)

*********************************************************************************************

Acknowledgement
  The development of this code has been supported by Department of Physics of Complex Systems, ELTE.
  GR would like to thank the Department of Physics & Astronomy, JHU for supporting this work.
  GR acknowledges sponsorship of a NASA Postdoctoral Program Fellowship. GR was supported by JPL, which is run under contract by California Institute of Technology for NASA.

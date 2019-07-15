   _____ _       _____   _____ 
  / ____| |     |  __ \ / ____|
 | (___ | |_ ___| |__) | (___ 
  \___ \| __/ _ \  ___/ \___ \
  ____) | ||  __/ |     ____) |
 |_____/ \__\___|_|    |_____/

StePS - STEreographically Projected cosmological Simulations

v0.3.7.2
Copyright (C) 2017-2019 Gábor Rácz
	Department of Physics of Complex Systems, Eotvos Lorand University | Budapest, Hungary
	Department of Physics & Astronomy, Johns Hopkins University | Baltimore, MD, USA
ragraat@caesar.elte.hu

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

This code is under development!

Cosmological simulation code for compactified cosmological simulations.
- written in C++
- parallelized with MPI, OpenMP and CUDA
- able to use multiple GPUs in a large computing cluster
- direct force calculation
- can read HDF5, Gadget2 and ASCII IC formats
- the output is in ASCII or HDF5 format
- able to run standard periodic and spherical cosmological simulations
- able to make periodic, quasiperiodic or spherical glass
- in this early version the code does not make a difference between baryonic and dark matter (dark matter only simulations)

*********************************************************************************************

Downloading the code:
	Under Linux, the source files of the StePS code can be downloaded with the 

	$ git clone https://github.com/eltevo/StePS

	command.

Installation:
	For the successful compilation, the code needs the OpenMPI (https://www.open-mpi.org/) library. Other MPI implementations should work too.
	Optional libraries:
		-CUDA (https://developer.nvidia.com/cuda-downloads) Use only if you want to accelerate the simulations with Nvidia GPUs
		-HDF5 (https://support.hdfgroup.org/HDF5/) This is used for reading and writing HDF5 files 
	You should specify the library directories in the Makefile. For editing the Makefile, you should type:

	$ cd StePS/StePS/src
	$ gedit Makefile

	Some features of the StePS code are controlled with compile time options in the Makefile. With this technique a more optimized executable can be generated. The following options can be found in the Makefile:
		-USE SINGLE PRECISION 
			If this is set, the code will use 32bit precision in the force calculation, otherwise 64bit calculation will be used. The 32bit force calculation is ∼ 32 times faster on Nvidia GTX GPUs compared to the 64bit force calculation, and it uses half as much memory. The speedup with Nvidia Tesla cards by using single pecision is ~2.
		-GLASSMAKING
			This option should be set for glass making. In this case the code will use reversed gravity.
		-HAVE_HDF5
			If this option is set, then the generated executable will be able to write the output files in HDF5 format.
		-PERIODIC
			Set this if the simulations will use periodic boundary condition. Note that there is a similar option in the parameter file. If the two options are contradicting each other, then the program will exit with an error message.

	After you saved the Makefile, the code can be compiled with the

	$ make

	command.

*********************************************************************************************

Once you compiled the code, you can simply run it by typing:
	export OMP_NUM_THREADS=<Number of shared memory OMP threads per MPI tasks>
	mpirun -n <number of MPI tasks> ./StePS <parameterfile>
where the parameterfile specifies the parameters of the simulation.

If you comiled the code for CUDA simulation,  you can simply run it by typing:
	export OMP_NUM_THREADS=<Number of GPUs per tasks>
        mpirun -n <number of MPI tasks> ./StePS_CUDA <parameterfile> <Number of GPUs per tasks>


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
COSMOLOGY       1			%1=cosmological simulation 0=traditional n-body sim.
Omega_b         0.0			%\
Omega_lambda    0.6911			%-Cosmological Omega parameters
Omega_dm        0.3089			%/
Omega_r         0.0			%|
H0              67.74			%Hubble-constant
a_start         0.05			%Initial scalefactor


Simulation parameters:
-----------------------
IS_PERIODIC     0						%Boundary condition 0=none, 1=nearest images, 2=ewald forces
COMOVING_INTEGRATION    1					%Comoving integration 0=no, 1=yes, used only when  COSMOLOGY=1
L_box           1860.0531					%Linear size of the simulation volume
IC_FILE         ../examples/ic/IC_SP_LCDM_1260_343M_com_VOI_1000.dat	%ic file
IC_FORMAT       0						%Ic file format 0: ascii, 1:GADGET, 2:(Gadget-)HDF5
OUT_DIR         ../examples/LCDM_SP_1260_343M_com_VOI_1000/		%output directory
OUT_LST         ../examples/ic/IC_SP_LCDM_1260_343M_com_VOI_1000.dat_zbins	%output list file
OUTPUT_TIME_VARIABLE	1					%Output time variable 0: physical time, 1: redshift
OUTPUT_FORMAT   1						%Output format 0: ASCII 2: (Gadget-)HDF5
REDSHIFT_CONE   1						%0: standard output files 1: one output redshift cone file
MIN_REDSHIFT    0.02477117					%The minimal output redshift. Lower redshifts considered 0. Only used in redshift cone simulations.
a_max           1.0						%The final scalefactor
ACC_PARAM	0.030						%Accuracy parameter
h_min           0.00025						%Minimal timestep length (in Gy)
h_max           0.03125						%Maximal timestep length (in Gy)
ParticleRadi    0.134226516867827				%Softening length of particle with minimal mass
FIRST_T_OUT     0.50						%First output time
H_OUT           0.50						%Output frequency
SNAPSHOT_START_NUMBER	0					%Initial snapshot number. Useful for restarting simulations.

*********************************************************************************************


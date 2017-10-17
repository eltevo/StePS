StePS - STEreographically Projected cosmological Simulations

v0.1.2.1

Gábor Rácz, 2017
Department of Physics of Complex Systems, Eötvös Loránd University
ragraat@caesar.elte.hu

Cosmological simulation code for compactified cosmological simulations.
- written in C++
- parallelized with openmp and CUDA
- read Gadget2 and ascii IC formats
- output in ascii format
- Able to run standard periodic and spherical cosmological simulations
- in this early version the code does not make difference between baryonic and dark matter (dark matter only simulations)

*********************************************************************************************

Installation:
	You should modify the Makefile, to tell the compiler where can it find the necessary libraries. After the modification, simply type:

        make

*********************************************************************************************

Once you compiled the code, you can simply run it by typing:
	./StePS <parameterfile>
where the parameterfile specifies the parameters of the simulation.

*********************************************************************************************

Output format for the particle data files:

z*.dat, t*.dat:
        x[Mpc]  y[Mpc]  z[Mpc]  v_x[20.7386814448645km/s] v_y[20.7386814448645km/s] v_z[20.7386814448645km/s] M[1e11M_sol]

redshift_cone.dat:
	x[Mpc]  y[Mpc]  z[Mpc]  v_x[20.7386814448645km/s] v_y[20.7386814448645km/s] v_z[20.7386814448645km/s] M[1e11M_sol]	R[Mpc]	z(=Redshift)


Output format for the logfile:

Logfile.dat:
	t[Gy]	error	h[Gy](=length of timestep)	a(=scalefactor)	z(=Redshift)	H[km/s/Mpc](=Hubble parameter)	q(=deceleration parameter)	Omega_m
*********************************************************************************************

The example parameterfile:
Cosmological parameters:
------------------------
COSMOLOGY       1			%1=cosmological simulation 0=traditional n-body sim.
Particle_mass   1.0			%mass of the particles. used only in non-cosmological simulations.
Omega_b         0.0			%\
Omega_lambda    0.6911			%-cosmological Omega parameters
Omega_dm        0.3089			%/
Omega_r         0.0			%|
H0              67.74			%Hubble-constant
a_start         0.05			%Initial scalefactor


Simulation parameters:
-----------------------
IS_PERIODIC     0						%Boundary condition 0=none, 1=nearest images, 2=ewald forces
COMOVING_INTEGRATION    1					%Comoving integration 0=no, 1=yes, used only when  COSMOLOGY=1
L_box           1860.0531					%linear size of the simulation volume
IC_FILE         ../examples/ic/IC_SP_LCDM_1260_343M_com_VOI_1000.dat	%ic file
IC_FORMAT       0						%ic file format 0: ascii, 1:GADGET
N_particle      1470017						%Number of particles in the IC
OUT_DIR         ../examples/LCDM_SP_1260_343M_com_VOI_1000/		%output directory
OUT_LST         ../examples/ic/IC_SP_LCDM_1260_343M_com_VOI_1000.dat_zbins	%output redshift list
OUTPUT_FORMAT   1						%output format 0: time, 1: redshift
REDSHIFT_CONE   1						%0: standard output files 1: one output redshift cone file
MIN_REDSHIFT    0.02477117					%The minimal output redshift. Lower redshifts considered 0.
a_max           1.0						%The final scalefactor
h_0             0.0000265121272579267				%initial timestep length
mean_err        0.030						%specified error
h_min           0.00000530242545158534				%minimal timestep length
h_max           0.000662803181448164				%maximal timestep length
ParticleRadi    0.134226516867827				%smoothing length of particle with minimal mass
FIRST_T_OUT     0.50						%first output time
H_OUT           0.50						%output frequency


*********************************************************************************************

The source code provided here is mainly for reference purposes.

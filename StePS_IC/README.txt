StePS_IC - Initial Conditions for STEreographically Projected cosmological Simulations

v0.3.0.0

Gábor Rácz, 2017-2018
	Department of Physics of Complex Systems, Eotvos Lorand University | Budapest, Hungary
	Department of Physics & Astronomy, Johns Hopkins University | Baltimore, MD, USA
ragraat@caesar.elte.hu

IC generator for compactified cosmological simulations.
- written in C++
- read Gadget2 and ascii IC formats
- output in ascii format
- Able to calculate spherical cosmological initial conditions using stereographic projection
- in this early version the code does not make difference between baryonic and dark matter (dark matter only simulations)

Note: you should use periodic initial conditions generated from "glass" as an input. ICs generated from "grid" can cause artificial distortion in the output particle distribution.

*********************************************************************************************

Installation:

	To compile the code, first install the following libraries:
	-Healpix 3.31 c++ library (http://healpix.sourceforge.net/downloads.php)
	-kdtree (https://github.com/jtsiomb/kdtree)

	You should modify the Makefile, to tell the compiler where can it find the necessary libraries. After the modification, simply type:

        make

*********************************************************************************************

Once you compiled the code, you can simply run it by typing:
	./StePS_IC <parameterfile>
where the parameterfile specifies the parameters of the IC generation.

*********************************************************************************************

Output format for the IC file:
        x[Mpc]  y[Mpc]  z[Mpc]  v_x[20.7386814448645km/s] v_y[20.7386814448645km/s] v_z[20.7386814448645km/s] M[1e11M_sol]


*********************************************************************************************

The example parameterfile:


Cosmological Parameters:
------------------------
Particle_mass   1.0			%mass of the particles. used only in ascii input ICs
Omega_b         0.0			%\
Omega_lambda    0.6911			%-cosmological Omega parameters
Omega_dm        0.3089			%/
Omega_r         0.0			%|
H0              67.74			%Hubble-constant
startH_0        3367.90520605054	%Initial Hubble parameter. used only in non-comoving case
a_start         0.05			%Initial scalefactor


Parameters of the IC:
-----------------------
L_box           1860.0531				%Linear size of the input periodic box
IC_FILE         ../../SPCS/ICs/IC_LCDM_1260_343M	%Input file base name
IC_FORMAT       1					%Input file format 0: ascii, 1:GADGET
N_particle      64310000				%Max number of particles in one input file
OUT_FILE        ./ic/IC_SP_LCDM_1260_343M_com_VOI_1000.dat	%Output filename
a_max           1					%maximal scalefactor
SPHERE_DIAMETER 105.0					%Diameter of the four dimensional hypersphere
R_CUT           930.026572187777			%Radius of the output IC. R_CUT should be <= L_box/2
N_SIDE          32					%Healpix binning parameter
R_GRID          224					%number of radial bins
FOR_COMOVING    1					%using comoving(1) or noncomoving(0) coordinates
RANDOM_ROTATION 0					%Rotations of the shells are enabled or disabled 
RANDOM_SEED     1234					%Random seed for the shell rotation
NUMBER_OF_INPUT_FILES   8				%number of input files
N_IC_tot        343000000				%total number of the input particles
VOI_X           1000.0					%x coordinate of the touching point
VOI_Y           1000.0					%y coordinate of the touching point
VOI_Z           930.026572187777			%z coordinate of the touching point
TileFac		3					%number of copies of the original periodic box per axe
SphericalGlassFILE	glass.dat			%Input spherical glass file with theta and phi coordinates

*********************************************************************************************

The source code provided here is mainly for reference purposes.

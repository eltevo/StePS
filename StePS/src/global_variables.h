/********************************************************************************/
/*  StePS - STEreographically Projected cosmological Simulations                */
/*    Copyright (C) 2017-2025 Gabor Racz                                        */
/*                                                                              */
/*    This program is free software; you can redistribute it and/or modify      */
/*    it under the terms of the GNU General Public License as published by      */
/*    the Free Software Foundation; either version 2 of the License, or         */
/*    (at your option) any later version.                                       */
/*                                                                              */
/*    This program is distributed in the hope that it will be useful,           */
/*    but WITHOUT ANY WARRANTY; without even the implied warranty of            */
/*    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the             */
/*    GNU General Public License for more details.                              */
/********************************************************************************/

#define pi 3.14159265358979323846264338327950288419716939937510
#define UNIT_T 47.14829951063323 //Unit time in Gy
#define UNIT_V 20.738652969925447 //Unit velocity in km/s

#ifdef GLASS_MAKING //Newtonian gravitational constant (in internal units)
#define G -1.0 //Gravity is repulsive, if Glassmaking is on.
#else
#define G 1.0 //Normal gravity
#endif

#ifdef USE_SINGLE_PRECISION
typedef float REAL;
#else
typedef double REAL;
#endif

#ifdef USE_BH
extern REAL THETA;//default value for the opening angle (used in BH forces)
#endif

extern int IS_PERIODIC; //periodic boundary conditions, 0=none, 1=nearest images, 2=ewald forces, 3>=ewald forces with increased cut-off radius
extern int COSMOLOGY; //Cosmological Simulation, 0=no, 1=yes
extern int COMOVING_INTEGRATION; //Comoving integration 0=no, 1=yes, used only when  COSMOLOGY=1
extern REAL L; //Size of the simulation box
extern REAL Rsim; //Radius of the simulation volume
extern char IC_FILE[1024]; //input file
extern char OUT_DIR[1024]; //output directory
extern char OUT_LST[1024]; //output redshift list file. only used when OUTPUT_TIME_VARIABLE=1
extern int IC_FORMAT; // 0: ascii, 1:GADGET
extern int OUTPUT_FORMAT; //Output format 0: ASCII 2:HDF5
extern int OUTPUT_TIME_VARIABLE; // 0: time, 1: redshift
extern double MIN_REDSHIFT; //The minimal output redshift. Lower redshifts considered 0.
extern int REDSHIFT_CONE; // 0: standard output files 1: one output redshift cone file
extern int HAVE_OUT_LIST;
extern double TIME_LIMIT_IN_MINS; //Simulation wall-clock time limit in minutes.
extern int H0_INDEPENDENT_UNITS; //0: i/o in Mpc, Msol, etc. 1: i/o in Mpc/h, Msol/h, etc.
extern double *out_list; //Output redshits
extern double *r_bin_limits; //bin limints in Dc for redshift cone simulations
extern int out_list_size; //Number of output redshits
extern unsigned int N_snapshot; //number of written out snapshots
extern bool ForceError; //true, if any errors encountered over the force calculation
extern bool* IN_CONE;

extern int n_GPU; //number of cuda capable GPUs
//variables for MPI
extern int numtasks, rank;
extern int N_mpi_thread; //Number of calculated forces in one MPI thread
extern int ID_MPI_min, ID_MPI_max; //max and min ID of of calculated forces in one MPI thread
extern MPI_Status Stat;
extern REAL* F_buffer; //buffer for force copy
extern int BUFFER_start_ID;

extern int e[2202][4]; //ewald space

extern REAL x4, err, errmax, ACC_PARAM; //variables used for error calculations
extern double h, h_min, h_max,  t_next; //actual stepsize, minimal and maximal stepsize, next time for output
extern double a_max,t_bigbang; //maximal scalefactor; Age of Big Bang

extern double FIRST_T_OUT, H_OUT; //First output time, output frequency in Gy

extern bool Allocate_memory; //if true, memory will be allocated for the next loaded snapshot. if false, the memory is already allocated.

extern REAL* M; //Particle masses
extern REAL *SOFT_LENGTH; //particle softening lengths
extern REAL M_tmp;
extern int N; //Number of particles
extern int t; //Number of the actual timestep
extern REAL* x; //particle coordinates
extern REAL* v; //and velocities
extern REAL* F; //Forces
extern REAL w[3]; //Parameters for smoothing in force calculation
extern REAL beta; //Particle radii
extern REAL ParticleRadi; //Particle radii; readed from parameter file
extern REAL rho_part; //One particle density
//extern REAL SOFT_CONST[8]; //Parameters for smoothing in force calculation
extern REAL M_min; //minimal particle mass
extern REAL mass_in_unit_sphere; //Mass in unit sphere
#ifdef HAVE_HDF5
extern int HDF5_redshiftcone_firstshell;
extern int N_redshiftcone; //number of particles written out to the redshiftcone file
#endif
//Cosmological parameters
extern double Omega_b,Omega_lambda,Omega_dm,Omega_r,Omega_k,Omega_m,H0,Hubble_param, Decel_param, delta_Hubble_param; //needed for all cosmological models
#if COSMOPARAM==1
extern double w0; //Dark energy equation of state at all redshifts. (LCDM: w0=-1.0)
#elif COSMOPARAM==2
extern double w0; //Dark energy equation of state at z=0. (LCDM: w0=-1.0)
extern double wa; //Negative derivative of the dark energy equation of state. (LCDM: wa=0.0)
#elif COSMOPARAM==-1
extern char EXPANSION_FILE[1024]; //input file with expansion history
extern int N_expansion_tab; //number of rows in the expansion history tab
extern int expansion_index; //index of the current value in the expansion history
extern double** expansion_tab; //expansion history tab (columns: t, a, H)
extern int INTERPOLATION_ORDER; //order of the interpolation (1,2,or 3)
#endif
extern double rho_crit; //Critical density
extern double a, a_start, a_prev, a_tmp; //Scalefactor, scalefactor at the starting time, previous scalefactor
extern double T, delta_a, Omega_m_eff; //Physical time, change of scalefactor, effectve Omega_m
#if defined(PERIODIC_Z)
//Variables only used in cylindrical simmetrical simulations
extern int ewald_max; //number of images in the z direction
extern REAL ewald_cut; //cutoff radius for the ewald summation
extern int RADIAL_FORCE_TABLE_SIZE; //size of the lookup table for the radial force calculation
extern int RADIAL_FORCE_ACCURACY; //number of points used in the integration for the lookup table
extern REAL *RADIAL_FORCE_TABLE; //lookup table for the radial force calculation
#endif
#ifdef USE_BH
extern int RADIAL_BH_FORCE_CORRECTION; //0: no correction, 1: correction for the radial force calculation based on the glass or initial radial BH forces (In the case of cylindrical and spherical simulations)
extern char GLASS_FILE_FOR_BH_FORCE_CORRECTION[1024]; //glass file used for the radial BH force correction. if "None", the IC file will be used to calculate the radial BH force correction.
extern int RADIAL_BH_FORCE_TABLE_SIZE; //size of the lookup table for the radial BH force correction calculation
extern REAL* RADIAL_BH_FORCE_TABLE; //lookup table for the radial BH force correction calculation
extern int* RADIAL_BH_N_TABLE; //table for the number of particles in a shell in the radial BH force correction calculation
extern int N_radial_bh_force_correction; //number of particles used in the radial BH force correction
extern int RADIAL_BH_FORCE_TABLE_ITERATION; //number of iterations for the radial BH force correction table calculation (only used in randomised BH force calculation)
extern bool USE_RADIAL_BH_CORRECTION; //true, if the radial BH force correction table is ready to use
#endif
//Functions
//Initial timestep length calculation
double calculate_init_h();
//Functions used for the Friedmann-equation
double friedmann_solver_step(double a0, double h);
double CALCULATE_Hubble_param(double a);
void recalculate_softening();
//This function calculates the deceleration parameter
double CALCULATE_decel_param(double a);

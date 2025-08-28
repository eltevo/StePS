/********************************************************************************/
/*  StePS - STEreographically Projected cosmological Simulations                */
/*    Copyright (C) 2017-2025 Gabor Racz, Balazs Pal                            */
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

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <math.h>
#include <omp.h>
#include <time.h>
#include <algorithm>
#include <unistd.h>
#include <cstring>
#include "mpi.h"
#include "global_variables.h"
#ifdef HAVE_HDF5
#include <hdf5.h>
#endif

#ifdef USE_SINGLE_PRECISION
typedef float REAL;
#else
typedef double REAL;
#endif

int t,N,el;
int e[2202][4];
REAL SOFT_CONST[8];
REAL w[3];
double a_max;
REAL* x;
REAL* v;
REAL* F;
bool* IN_CONE;
double h, h_min, h_max, T, t_next, t_bigbang;
REAL ACC_PARAM;
double FIRST_T_OUT, H_OUT; //First output time, output frequency in Gy
double rho_crit; //Critical density
REAL mass_in_unit_sphere; //Mass in unit sphere
bool ForceError = false;

int n_GPU; //number of cuda capable GPUs
int numtasks, rank; //Variables for MPI
int N_mpi_thread; //Number of calculated forces in one MPI thread
int ID_MPI_min, ID_MPI_max; //max and min ID of of calculated forces in one MPI thread
MPI_Status Stat;
int BUFFER_start_ID;
REAL* F_buffer;

REAL x4, err, errmax;
REAL beta, ParticleRadi, rho_part, M_min;

#ifdef USE_BH
REAL THETA;//default value for the opening angle (used in BH forces)
#endif

int IS_PERIODIC, COSMOLOGY;
int COMOVING_INTEGRATION; //Comoving integration 0=no, 1=yes, used only when  COSMOLOGY=1
REAL L, Rsim; //linear size of the simulation volume
char IC_FILE[1024];
char OUT_DIR[1024];
char OUT_LST[1024]; //output redshift list file. only used when OUTPUT_TIME_VARIABLE=1
extern char __BUILD_DATE;
int IC_FORMAT; // 0: ASCII, 1:GADGET
int OUTPUT_FORMAT; // 0:ASCII, 2:HDF5
int OUTPUT_TIME_VARIABLE; // 0: time, 1: redshift
double MIN_REDSHIFT; //The minimal output redshift. Lower redshifts considered 0.
int REDSHIFT_CONE; // 0: standard output files 1: one output redshift cone file
int HAVE_OUT_LIST; // 0: output list not found. 1: output list found
double TIME_LIMIT_IN_MINS; //Simulation wall-clock time limit in minutes.
int H0_INDEPENDENT_UNITS; //0: i/o in Mpc, Msol, etc. 1: i/o in Mpc/h, Msol/h, etc.
double *out_list; //Output redshits
double *r_bin_limits; //bin limints in Dc for redshift cone simulations
int out_list_size; //Number of output redshits
unsigned int N_snapshot; //number of written out snapshots

double Omega_b,Omega_lambda,Omega_dm,Omega_r,Omega_k,Omega_m,H0,Hubble_param, Decel_param, delta_Hubble_param; //Cosmologycal parameters
#if COSMOPARAM==1
double w0; //Dark energy equation of state at all redshifts. (LCDM: w0=-1.0)
#elif COSMOPARAM==2
double w0; //Dark energy equation of state at z=0. (LCDM: w0=-1.0)
double wa; //Negative derivative of the dark energy equation of state. (LCDM: wa=0.0)
#elif COSMOPARAM==-1
char EXPANSION_FILE[1024]; //input file with expansion history
int N_expansion_tab; //number of rows in the expansion history tab
int expansion_index; //index of the current value in the expansion history
double** expansion_tab; //expansion history tab (columns: t, a, H)
int INTERPOLATION_ORDER; //order of the interpolation (1,2,or 3)
#endif
#if defined(PERIODIC_Z)
//Variables only used in cylindrical simmetrical simulations
//Variables only used in cylindrical simmetrical simulations
int ewald_max; //number of images in the z direction
REAL ewald_cut; //cutoff radius for the ewald summation
int RADIAL_FORCE_TABLE_SIZE; //size of the lookup table for the radial force calculation
int RADIAL_FORCE_ACCURACY; //number of points used in the integration for the lookup table
REAL *RADIAL_FORCE_TABLE; //lookup table for the radial force calculation
bool CYLINDRICAL_LOOKUP_TABLE_CALCULATED=false; //true, if the cylindrical lookup table is calculated
#endif
#ifdef USE_BH
int RADIAL_BH_FORCE_CORRECTION; //0: no correction, 1: correction for the radial force calculation based on the glass or initial radial BH forces (In the case of cylindrical and spherical simulations)
char GLASS_FILE_FOR_BH_FORCE_CORRECTION[1024]; //glass file used for the radial BH force correction. if "None", the IC file will be used to calculate the radial BH force correction.
int RADIAL_BH_FORCE_TABLE_SIZE; //size of the lookup table for the radial BH force correction calculation
REAL* RADIAL_BH_FORCE_TABLE; //lookup table for the radial BH force correction calculation
int* RADIAL_BH_N_TABLE; //table for the number of particles in a shell in the radial BH force correction calculation
int N_radial_bh_force_correction; //number of particles used in the radial BH force correction
int RADIAL_BH_FORCE_TABLE_ITERATION; //number of iterations for the radial BH force correction table calculation (only used in randomised BH force calculation)
bool USE_RADIAL_BH_CORRECTION; //true, if the radial BH force correction table is ready to use
#endif
double epsilon=1;
double sigma=1;
bool Allocate_memory; //if true, memory will be allocated for the next loaded snapshot. if false, the memory is already allocated.
REAL* M;//Particle mass
REAL* SOFT_LENGTH; //particle softening lengths
REAL M_tmp;
double a, a_start,a_prev,a_tmp;//Scalefactor, scalefactor at the starting time, previous scalefactor
double Omega_m_eff; //Effective Omega_m
double delta_a;


// function declarations
void read_param(FILE *param_file);
void step(REAL* x, REAL* v, REAL* F);
void calculate_softening_length(REAL *SOFT_LENGTH, REAL *M, int N);
void forces(REAL* x, REAL* F, int ID_min, int ID_max);
void forces_periodic(REAL*x, REAL*F, int ID_min, int ID_max);
void forces_periodic_z(REAL*x, REAL*F, int ID_min, int ID_max);
double friedmann_solver_start(double a0, double t0, double h, double a_start);
double friedmann_solver_step(double a0, double h);
int ewald_space(REAL R, int ewald_index[2102][4]);
double CALCULATE_Hubble_param(double a);
double CALCULATE_decel_param(double a);
//Functions used in MPI parallelisation
void BCAST_global_parameters();
#ifdef USE_BH
void get_radial_bh_force_correction_table(REAL *RADIAL_BH_FORCE_TABLE, int *RADIAL_BH_N_TABLE, int TABLE_SIZE, REAL *F, REAL *x, int N);
#endif

//Input/Output functions
int file_exist(char *file_name);
int dir_exist(char *dir_name);
int load_IC(char *IC_FILE, int IC_FORMAT);
int read_OUT_LST();
void write_redshift_cone(REAL *x, REAL *v, double *limits, int z_index, int delta_z_index, int ALL);
void write_ascii_snapshot(REAL* x, REAL *v);
void Log_write();
#ifdef HAVE_HDF5
int N_redshiftcone, HDF5_redshiftcone_firstshell;
//Functions for HDF5 I/O
void write_hdf5_snapshot(REAL *x, REAL *v, REAL *M);
void write_header_attributes_in_hdf5(hid_t handle);
#endif
#if COSMOPARAM==-1
void read_expansion_history(char* filename);
#endif
#if defined(PERIODIC_Z)
void get_cylindrical_force_table(REAL* FORCE_TABLE, REAL R, REAL Lz, int TABLE_SIZE, int RADIAL_FORCE_ACCURACY);
#endif

int main(int argc, char *argv[])
{
	//initialize MPI
	MPI_Init(&argc,&argv);
	// get number of tasks
	MPI_Comm_size(MPI_COMM_WORLD,&numtasks);
	// get my rank
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);
	#ifndef USE_CUDA
	// get number of OMP threads
	int omp_threads;
	#pragma omp parallel
	{
			omp_threads = omp_get_num_threads();
	}
	#endif
	if(rank == 0)
	{
		printf("+-----------------------------------------------------------------------------------------------+\n|   _____ _       _____   _____ \t\t\t\t\t\t\t\t|\n|  / ____| |     |  __ \\ / ____|\t\t\t\t\t\t\t\t|\n| | (___ | |_ ___| |__) | (___  \t\t\t\t\t\t\t\t|\n|  \\___ \\| __/ _ \\  ___/ \\___ \\ \t\t\t\t\t\t\t\t|\n|  ____) | ||  __/ |     ____) |\t\t\t\t\t\t\t\t|\n| |_____/ \\__\\___|_|    |_____/ \t\t\t\t\t\t\t\t|\n|StePS %s\t\t\t\t\t\t\t\t\t\t\t|\n| (STEreographically Projected cosmological Simulations)\t\t\t\t\t|\n+-----------------------------------------------------------------------------------------------+\n| Copyright (C) 2017-2025 Gabor Racz et al.\t\t\t\t\t\t\t|\n|\tDepartment of Physics, University of Helsinki | Helsinki, Finland\t\t\t|\n|\tJet Propulsion Laboratory, California Institute of Technology | Pasadena, CA, USA\t|\n|\tDepartment of Physics of Complex Systems, Eotvos Lorand University | Budapest, Hungary\t|\n|\tDepartment of Physics & Astronomy, Johns Hopkins University | Baltimore, MD, USA\t|\n|\t\t\t\t\t\t\t\t\t\t\t\t|\n|", PROGRAM_VERSION);
		printf("Build date: %s\t\t\t\t\t\t\t|\n|",  BUILD_DATE);
		printf("Compiled with: %s", COMPILER_VERSION);
		unsigned long int I;
		for(I = 0; I<10-((sizeof(COMPILER_VERSION)-1)/8); I++)
			printf("\t");
		printf("|\n+-----------------------------------------------------------------------------------------------+\n\n");
		printf("+---------------------------------------------------------------+\n| StePS comes with ABSOLUTELY NO WARRANTY.\t\t\t|\n| This is free software, and you are welcome to redistribute it\t|\n| under certain conditions. See the LICENSE file for details.\t|\n+---------------------------------------------------------------+\n\n");
	}
	char HOSTNAME_BUF[1024];
	if(rank == 0)
	{
		gethostname(HOSTNAME_BUF, sizeof(HOSTNAME_BUF));
		printf("\tRunning on %s.\n", HOSTNAME_BUF);
	}
	#ifdef USE_CUDA
	if(rank == 0)
		printf("\tUsing CUDA capable GPUs for force calculation.\n");
	#endif
	#ifdef GLASS_MAKING
	if(rank == 0)
		printf("\tGlass making.\n");
	#endif
	#ifdef USE_SINGLE_PRECISION
	if(rank == 0)
		printf("\tSingle precision (32bit) force calculation.\n");
	#else
	if(rank == 0)
		printf("\tDouble precision (64bit) force calculation.\n");
	#endif
	#if defined(PERIODIC)
	if(rank == 0)
		printf("\tPeriodic boundary conditions. (Periodic 3-torus topology)\n");
	#elif defined(PERIODIC_Z)
	if(rank == 0)
		printf("\tPeriodic boundary conditions in the z direction. (Cylindrically symmetric topology)\n");
	#else
	if(rank == 0)
		printf("\tNon-periodic boundary conditions. (Spherically symmetric topology)\n");
	#endif
	#if defined(USE_BH)
	THETA = (REAL) USE_BH; //Opening angle for the octree
	#if !defined(RANDOMIZE_BH)
	if(rank == 0)
		printf("\tForce calculation method: Barnes-Hut tree (Octree) algorithm with opening angle (theta) %.2f.\n", THETA);
	#else 
	if(rank == 0)
		printf("\tForce calculation method: Randomized Barnes-Hut tree (Octree) algorithm.\n\t\tOctree opening angle (theta): %.2f\n\t\tRandom seed: %i\n", THETA, RANDOMIZE_BH);
	srand(RANDOMIZE_BH);
	#endif
	#else
	if(rank == 0)
		printf("\tForce calculation method: Direct summation.\n");
	#endif
	#if COSMOPARAM==0 || !defined(COSMOPARAM)
	if(rank == 0)
		printf("\tBackground cosmology: FLRW cosmology with Standard Lambda-Cold Dark Matter parametrization. (LCDM)\n\n");
	#elif COSMOPARAM==1
	if(rank == 0)
		printf("\tBackground cosmology: FLRW cosmology with a constant dark energy equation of state. (wCDM)\n\n");
	#elif COSMOPARAM==2
	if(rank == 0)
		printf("\tBackground cosmology: FLRW cosmology with a CPL dark energy equation of state (w0waCDM)\n\n");
	#elif COSMOPARAM==-1
	if(rank == 0)
		printf("\tBackground cosmology: FLRW cosmology with a tabulated expansion history. \n\n");
	#endif
	if(numtasks != 1 && rank == 0)
	{
		printf("Number of MPI tasks: %i\n", numtasks);
		#if !defined(USE_CUDA)
			printf("Number of OpenMP threads per MPI tasks: %i\n", omp_threads);
			printf("Total number of OpenMP threads: %i\n\n", numtasks*omp_threads);
		#else
			printf("\n");
		#endif
	}
	if(numtasks == 1 && rank == 0)
	{
		printf("Running in OpenMP mode (Number of MPI tasks: %i).\n", numtasks);
		#if !defined(USE_CUDA)
			printf("Total number of OpenMP threads: %i\n\n", omp_threads);
		#else
			printf("\n");
		#endif
	}
	#ifndef USE_CUDA
	if(rank == 0 && argc == 3)
	{
		omp_threads = atoi( argv[2] );
		omp_set_num_threads( atoi( argv[2] ) );
		printf("Numer of OpenMP threads per MPI tasks set to %i.\n", atoi( argv[2] ));
	}
	#endif
	int i,j;
	bool F_buffer_allocated = false; //F_buffer is not allocated yet
	int CONE_ALL=0;
	Allocate_memory = true; //Before loading the first snapshot, memory will be allocated for the particle data arrays.
	N_snapshot = 0; //The snapshot start number is 0 by default
	TIME_LIMIT_IN_MINS = 0; //There is no wall-clock time limit by default
	H0_INDEPENDENT_UNITS = 0; //StePS uses H0 dependent units by default
	OUTPUT_TIME_VARIABLE = -1;
	if( argc < 2 )
	{
		if(rank == 0)
		{
			fprintf(stderr, "Missing parameter file!\n");
			fprintf(stderr, "Call with: ./%s  <parameter file>\n", PROGRAMNAME);
		}
		return (-1);
	}
	else if(argc > 3)
	{
		if(rank == 0)
		{
			fprintf(stderr, "Too many arguments!\n");
			#ifndef USE_CUDA
				fprintf(stderr, "Call with: ./%s  <parameter file>\n", PROGRAMNAME);
			#else
				fprintf(stderr, "Call with: ./%s  <parameter file> \'i\', where \'i\' is the number of the CUDA capable GPUs per node.\nif \'i\' is not set, than one GPU per MPI task will be used.\n", PROGRAMNAME);
			#endif
		}
		return (-1);
	}
	//the rank=0 thread reads the paramfile, and bcast the variables to the other threads
	if(rank == 0)
	{
		if(file_exist(argv[1]) == 0)
		{
			fprintf(stderr, "Error: The %s parameter file does not exist!\nExiting.\n", argv[1]);
			return (-1);
		}
		FILE *param_file;
		param_file = fopen(argv[1], "r");
		read_param(param_file);
		if(dir_exist(OUT_DIR) == 0)
		{
			fprintf(stderr, "Error: The %s output directory does not exist!\nExiting.\n",OUT_DIR);
			return (-1);
		}
	}
	BCAST_global_parameters();
	#ifdef PERIODIC
		if(IS_PERIODIC < 1 || IS_PERIODIC > 4)
		{
			if(rank == 0)
				fprintf(stderr, "Error: Bad boundary condition were set in the paramfile!\nThis executable are able to deal with periodic simulation only.\nExiting.\n");
			return (-2);
		}

		if(IS_PERIODIC>1)
		{
			if(IS_PERIODIC==2)
			{
				if(rank == 0)
					printf("Ewald force calculation is on. (Ewald cut is 2.6*L)\n");
				el = ewald_space(3.6,e);
			}
 			else if(IS_PERIODIC==3)
			{
				if(rank == 0)
					printf("High precision Ewald force calculation is on. (Ewald cut is 4.6*L)\n");
				el = ewald_space(5.6,e);
			}
			else
			{
				if(rank == 0)
					printf("High precision Ewald force calculation is on. (Ewald cut is 5.6*L)\n");
				el = ewald_space(6.6,e);
			}
		}
		else
		{
			printf("Quasi-periodic boundary conditions.\n");
			el = 2202;
			for(i=0;i<el;i++)
			{
				for(j=0;j<3;j++)
				{
					e[i][j]=0.0;
				}
			}
			i=0;
			j=0;
		}
	#elif defined(PERIODIC_Z)
		if(IS_PERIODIC < 1)
		{
			if(rank == 0)
				fprintf(stderr, "Error: Bad boundary conditions were set in the paramfile!\nThis executable is able to run semi-periodic and periodic simulations in z direction only.\nExiting.\n");
			return (-2);
		}
		#ifdef USE_BH
			REAL MaxNodeSize;
			if(2*Rsim < L)
			{
				//if the radius is smaller than half of the box size, we use the periodicity length as the root node size
				MaxNodeSize = L;
			}
			else
			{
				//if the radius is larger than half of the box size, we use the double of diameter as the root node size
				MaxNodeSize = 2.0*Rsim;
			}
			if (MaxNodeSize/THETA < (((REAL) (IS_PERIODIC+1)) - 0.4)*L)
			{
				if(rank == 0)
				{
					printf("Warning: Using too many periodic images in the \"z\" direction:\n\tNodeSize/THETA = %.2f Mpc < Ewald_cut = %.2f Mpc.\nPlease consider decreasing the Theta opening angle, or decreasing the number of periodic images.\n", MaxNodeSize/THETA, L*(((REAL) (IS_PERIODIC+1)) - 0.4));
					printf("You can set the repeated periodic images in the z direction to %i (Boundary condition %i) to avoid not-resolved periodic images.\n", 2*((int) floor(MaxNodeSize/THETA/L + 0.4) - 1)+1, (int) floor(MaxNodeSize/THETA/L + 0.4) - 1);
				}
			}
		#endif
		if(IS_PERIODIC>= 2)
		{
			printf("Ewald summation is on in z direction. Using %i images. (Boundary condition %i)\n\n", 2*IS_PERIODIC+1, IS_PERIODIC);
		}
		else
		{
			printf("Warning: Quasi-periodic boundary conditions only in the z direction.\n         Using only one periodic image in this geometry can easily cause inaccurate forces.\n\n");
		}
	#else
		if(IS_PERIODIC  != 0)
		{
			if(rank == 0)
				fprintf(stderr, "Error: Bad boundary conditions were set in the paramfile!\nThis executable is able to run non-periodic (spherical) simulations only.\nExiting.\n");
			return (-2);
		}
	#endif
	if(OUTPUT_TIME_VARIABLE != 0 && OUTPUT_TIME_VARIABLE !=1)
	{
		if(rank == 0)
			fprintf(stderr, "Error: bad OUTPUT time variable %i!\nExiting.\n", OUTPUT_TIME_VARIABLE);
		return (-2);
	}
	if(OUTPUT_TIME_VARIABLE == 1 && COSMOLOGY != 1)
	{
		if(rank == 0)
			fprintf(stderr, "Error: you can not use redshift output format in non-cosmological simulations. \nExiting.\n");
		return (-2);
	}
	if(H0 == 0.0 && COSMOLOGY == 1)
	{
    #if !defined(COSMOPARAM) || COSMOPARAM>=0
		if(rank == 0)
			fprintf(stderr, "Error: Hubble constant is set to zero in a cosmological simulation. This must be a mistake.\nExiting.\n");
		return (-2);
		#else
		if(rank == 0)
			printf("Warning: Hubble constant is set to zero in a cosmological simulation. \nSince the expansion history read from an external file, this is not necessarily an error.\nPlease make sure that the Hubble constant was set correctly during the initial condition generation.\n\n");
		#endif
	}
	if(rank == 0)
	{
		if(file_exist(OUT_LST) == 0)
		{
			HAVE_OUT_LIST = 0;
			printf("Output list not found. Using the FIRST_T_OUT and H_OUT variables for calculating the output");
		}
		else
		{
			HAVE_OUT_LIST = 1;
			printf("Output list found. Using the contents of this file for the output");
		}
		if(OUTPUT_TIME_VARIABLE == 1)
		{
			printf(" redshifts.\n");
			if(COSMOLOGY == 1 && COMOVING_INTEGRATION == 0)
			{
					fprintf(stderr, "Error: only output physical times can be used in non-comoving cosmological simulations.\nExiting.\n");
					return (-2);
			}
		}
		else
			printf(" times.\n");
	}
	if(HAVE_OUT_LIST==1 && rank==0)
	{
		if(0 != read_OUT_LST())
		{
			fprintf(stderr, "Exiting.\n");
			return (-2);
		}
	}
	#if defined(USE_BH) && !defined(PERIODIC)
	USE_RADIAL_BH_CORRECTION = false; //The radial BH force table iteration is not done yet
	if(RADIAL_BH_FORCE_CORRECTION == 1)
	{
		if(rank == 0)
		{
			if(file_exist(GLASS_FILE_FOR_BH_FORCE_CORRECTION) == 0 && strcmp(GLASS_FILE_FOR_BH_FORCE_CORRECTION, "None") != 0)
			{
				fprintf(stderr, "Error: The %s glass file does not exist!\n Using the initial condition file for radial BH force correction.\n", GLASS_FILE_FOR_BH_FORCE_CORRECTION);
				strcpy(GLASS_FILE_FOR_BH_FORCE_CORRECTION, "None");
			}
			if(strcmp(GLASS_FILE_FOR_BH_FORCE_CORRECTION, "None") != 0)
			{
				printf("\nRadial BH force correction is on. Using the %s glass file for the radial BH force correction.\n", GLASS_FILE_FOR_BH_FORCE_CORRECTION);
			}
			else
			{
				printf("\nRadial BH force correction is on. Using the initial condition file for the radial BH force correction.\n");
				strcpy(GLASS_FILE_FOR_BH_FORCE_CORRECTION, IC_FILE);
			}
			//Loading the initial glass or IC file for the radial BH force correction (master thread only)
			if(load_IC(GLASS_FILE_FOR_BH_FORCE_CORRECTION, IC_FORMAT) != 0)
			{
				fprintf(stderr, "Error: failed to load the %s file for the radial BH force correction.\nExiting.\n", GLASS_FILE_FOR_BH_FORCE_CORRECTION);
				return (-1);
			}
			N_mpi_thread = (N/numtasks) + (N%numtasks);
			//Calculating softening lengths for the radial BH force correction
			//Converting units, if needed
			if(H0_INDEPENDENT_UNITS != 0 && COSMOLOGY == 1)
			{
				if(H0==0.0)
				{
					fprintf(stderr, "Error: Hubble constant is zero while using H0 independent units. This must be a mistake.\nExiting.\n");
					return (-2);
				}
				for(i=0;i<N;i++)
				{
					for(j=0;j<3;j++)
					{
						x[3*i + j] /= (H0*UNIT_V/100.0); //converting coordinates
					}
					M[i] /= (H0*UNIT_V/100.0); //converting masses
				}
			}
			calculate_softening_length(SOFT_LENGTH, M, N); //Calculating the softening lengths for the particles
			if(numtasks > 1 && F_buffer_allocated == false)
			{
				//Allocating memory for the F_buffer
				if(!(F_buffer = (REAL*)malloc(3*(N/numtasks)*sizeof(REAL))))
				{
					fprintf(stderr, "MPI task %i: failed to allocate memory for F_buffer.\n", rank);
					exit(-2);
				}
				F_buffer_allocated = true; //F_buffer is allocated
			}
		}
		//Bcasting the number of particles
		MPI_Bcast(&N,1,MPI_INT,0,MPI_COMM_WORLD);
		if(rank != 0)
		{
			//Allocating memory for the slave processes
			N_mpi_thread = N/numtasks;

			//Allocating memory for the coordinates
			if(!(x = (REAL*)malloc(3*N*sizeof(REAL))))
			{
				fprintf(stderr, "MPI task %i: failed to allocate memory for x.\n", rank);
				exit(-2);
			}
			//Allocating memory for the forces. There is no need to allocate for N forces. N/numtasks should be enough
			if(!(F = (REAL*)malloc((3*N_mpi_thread)*sizeof(REAL))))
			{
				fprintf(stderr, "MPI task %i: failed to allocate memory for F.\n", rank);
				exit(-2);
			}
			//Allocating memory for the masses
				if(!(M = (REAL*)malloc(N*sizeof(REAL))))
			{
				fprintf(stderr, "MPI task %i: failed to allocate memory for M.\n", rank);
				exit(-2);
			}
			//Allocating memory for the softening lengths
				if(!(SOFT_LENGTH = (REAL*)malloc(N*sizeof(REAL))))
			{
				fprintf(stderr, "MPI task %i: failed to allocate memory for SOFT_LENGTH.\n", rank);
				exit(-2);
			}
			Allocate_memory = false; //Now the memory is already allocated for the particle data arrays.
		}
		//Bcasting the particle data
		#ifdef SINGLE_PRECISION
			MPI_Bcast(&M_min,1,MPI_FLOAT,0,MPI_COMM_WORLD);
			MPI_Bcast(&rho_part,1,MPI_FLOAT,0,MPI_COMM_WORLD);
			MPI_Bcast(x,3*N,MPI_FLOAT,0,MPI_COMM_WORLD);
			MPI_Bcast(M,N,MPI_FLOAT,0,MPI_COMM_WORLD);
			MPI_Bcast(SOFT_LENGTH,N,MPI_FLOAT,0,MPI_COMM_WORLD);
		#else
			MPI_Bcast(&M_min,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
			MPI_Bcast(&rho_part,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
			MPI_Bcast(x,3*N,MPI_DOUBLE,0,MPI_COMM_WORLD);
			MPI_Bcast(M,N,MPI_DOUBLE,0,MPI_COMM_WORLD);
			MPI_Bcast(SOFT_LENGTH,N,MPI_DOUBLE,0,MPI_COMM_WORLD);
		#endif
		// Calculating cylindrical or spherical radial forces of the last pixel (mass_in_unit_sphere)
		a = 1.0; //Setting the scale factor to 1.
		rho_crit = 3.0*H0*H0/(8.0*pi); //Calculating the critical density
		#if defined(PERIODIC_Z)
			if(CYLINDRICAL_LOOKUP_TABLE_CALCULATED == false)
			{
				// in cylindrically symmetric simulations, the radial force is proportional to the mass in a unit cylinder
				mass_in_unit_sphere = (REAL) (2.0*pi*rho_crit*Omega_m);
				ewald_max = IS_PERIODIC+1;
				ewald_cut = ((REAL) ewald_max)-0.4; //cutoff radius for the ewald summation
				// Calculating the lookup table for the radial force calculation
				printf("Calculating the lookup table for the radial force calculation...\n");
				RADIAL_FORCE_TABLE = (REAL*)malloc(RADIAL_FORCE_TABLE_SIZE*sizeof(REAL)); //Allocating the lookup table
				if(IS_PERIODIC==1)
				{
					get_cylindrical_force_table(RADIAL_FORCE_TABLE, Rsim,L,RADIAL_FORCE_TABLE_SIZE,RADIAL_FORCE_ACCURACY);
					if(rank==0)
						printf("Cylindrical force table calculated.\nMagnitude of the gravitational pull correction of the finite Ewald cylinder at r = %.1f Mpc is |F(Lz=2.0Lsim)|/|F(Lz=inf)| = %.7f \n\n", Rsim, RADIAL_FORCE_TABLE[RADIAL_FORCE_TABLE_SIZE-1]);
				}
				else
				{
					get_cylindrical_force_table(RADIAL_FORCE_TABLE, Rsim, L*ewald_cut,RADIAL_FORCE_TABLE_SIZE,RADIAL_FORCE_ACCURACY);
					if(rank==0)
						printf("Cylindrical force table calculated.\nMagnitude of the gravitational pull correction of the finite Ewald cylinder at r = %.1f Mpc is |F(Lz=%.1fLsim)|/|F(Lz=inf)| = %.7f \n\n", Rsim, 2*ewald_cut, RADIAL_FORCE_TABLE[RADIAL_FORCE_TABLE_SIZE-1]);
				}
				CYLINDRICAL_LOOKUP_TABLE_CALCULATED = true;
			}
		#else
			// in spherical simmetrical simulations, the radial force is proportional to the mass in a unit sphere
			mass_in_unit_sphere = (REAL) (4.0*pi*rho_crit*Omega_m/3.0);
		#endif
		// Allocating memory for the radial BH force correction table on all threads
		if(!(RADIAL_BH_FORCE_TABLE = (REAL*)malloc(RADIAL_BH_FORCE_TABLE_SIZE*sizeof(REAL))))
		{
			fprintf(stderr, "MPI task %i: failed to allocate memory for RADIAL_BH_FORCE_TABLE.\n", rank);
			exit(-2);
		}
		if(!(RADIAL_BH_N_TABLE = (int*)malloc(RADIAL_BH_FORCE_TABLE_SIZE*sizeof(int))))
		{
			fprintf(stderr, "MPI task %i: failed to allocate memory for RADIAL_BH_N_TABLE.\n", rank);
			exit(-2);
		}
		// Initialize the allocated arrays
		for(i=0; i<RADIAL_BH_FORCE_TABLE_SIZE; i++)
		{
			RADIAL_BH_FORCE_TABLE[i] = 0.0;
			RADIAL_BH_N_TABLE[i] = 0;
		}
		// Calculating the forces between the particles using the BH force calculation method
		// Do this RADIAL_BH_FORCE_TABLE_ITERATION times, and average the results
		#if !defined(RANDOMIZE_BH)
			RADIAL_BH_FORCE_TABLE_ITERATION = 1; //If the BH force calculation is randomised, only one iteration is needed
			if(rank == 0)
				printf("Randomised BH force calculation is off. Only one iteration is needed.\n");
		#endif
		for(int i_iter=0; i_iter<RADIAL_BH_FORCE_TABLE_ITERATION; i_iter++)
		{
			if(rank == 0 && RADIAL_BH_FORCE_TABLE_ITERATION > 1)
			{
				printf("BH radial force correction iteration %d/%d\n------------------------------------------\n", i_iter+1, RADIAL_BH_FORCE_TABLE_ITERATION);
				fflush(stdout);
			}
			if(rank==0)
			{
				ID_MPI_min = 0;
				ID_MPI_max = (N%numtasks) + (rank+1)*(N/numtasks)-1;
				#if defined(PERIODIC_Z)
					//cylindrical periodic boundary conditions
					forces_periodic_z(x, F, ID_MPI_min, ID_MPI_max);
				#else
					//Spherical non-periodic [StePS] boundary conditions
					forces(x, F, ID_MPI_min, ID_MPI_max);
				#endif
			}
			else
			{
				//Slave threads calculate the forces on their own particles
				ID_MPI_min = (N%numtasks) + (rank)*(N/numtasks);
				ID_MPI_max = (N%numtasks) + (rank+1)*(N/numtasks)-1;
				#if defined(PERIODIC_Z)
					forces_periodic_z(x, F, ID_MPI_min, ID_MPI_max);
				#else
					forces(x, F, ID_MPI_min, ID_MPI_max);
				#endif
			}
			//Collecting all the forces to the rank 0 thread
			if(rank !=0)
			{
			#ifdef USE_SINGLE_PRECISION
				MPI_Send(F, 3*N_mpi_thread, MPI_FLOAT, 0, rank, MPI_COMM_WORLD);
			#else
				MPI_Send(F, 3*N_mpi_thread, MPI_DOUBLE, 0, rank, MPI_COMM_WORLD);
			#endif
			}
			else
			{
				if(numtasks > 1)
				{
					for(i=1; i<numtasks;i++)
					{
						BUFFER_start_ID = i*(N/numtasks)+(N%numtasks);
						#ifdef USE_SINGLE_PRECISION
							MPI_Recv(F_buffer, 3*(N/numtasks), MPI_FLOAT, i, i, MPI_COMM_WORLD, &Stat);
						#else
							MPI_Recv(F_buffer, 3*(N/numtasks), MPI_DOUBLE, i, i, MPI_COMM_WORLD, &Stat);
						#endif
						for(j=0; j<(N/numtasks); j++)
						{
							F[3*(BUFFER_start_ID+j)] = F_buffer[3*j];
							F[3*(BUFFER_start_ID+j)+1] = F_buffer[3*j+1];
							F[3*(BUFFER_start_ID+j)+2] = F_buffer[3*j+2];
						}
					}
				}
			}
			//Calculating the radial BH force correction from the force vectors
			if(rank == 0)
			{
				printf("Calculating the radial force correction from the estimated Octree forces...\n");
			}
			if(rank == 0)
			{
				get_radial_bh_force_correction_table(RADIAL_BH_FORCE_TABLE, RADIAL_BH_N_TABLE, RADIAL_BH_FORCE_TABLE_SIZE, F, x, N);
			}
		}
		//Normalizing the radial BH force correction table
		if(rank == 0)
		{
			for(i=0; i<RADIAL_BH_FORCE_TABLE_SIZE; i++)
			{
				if(RADIAL_BH_N_TABLE[i] > 0)
				{
					RADIAL_BH_FORCE_TABLE[i] /= (REAL) RADIAL_BH_N_TABLE[i]; //Normalizing the force table
				}
				else
				{
					RADIAL_BH_FORCE_TABLE[i] = 0.0; //If there are no particles in this bin, set the force to zero
					printf("Warning: No particles in shell %d, setting force to zero.\n", i);
				}
			}
			printf("Table values:\nBin\tR[Mpc]\t\tForce\t\tRADIAL_BH_N_TABLE\n-------------------------------------\n");
			for(i=0; i<RADIAL_BH_FORCE_TABLE_SIZE; i++)
			{
				printf("%i\t%f\t%f\t %i\n", i, ((double) i + 0.5) * (Rsim / (double) RADIAL_BH_FORCE_TABLE_SIZE), RADIAL_BH_FORCE_TABLE[i], RADIAL_BH_N_TABLE[i]);
			}
			fflush(stdout);
		}
		#ifdef SINGLE_PRECISION
		MPI_Bcast(RADIAL_BH_FORCE_TABLE, RADIAL_BH_FORCE_TABLE_SIZE, MPI_FLOAT, 0, MPI_COMM_WORLD);
		#else
		MPI_Bcast(RADIAL_BH_FORCE_TABLE, RADIAL_BH_FORCE_TABLE_SIZE, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		#endif
		if(rank == 0)
		{
			printf("Radial BH force correction table calculated.\n");
		}
		N_radial_bh_force_correction = N; //Number of particles used in the radial BH force correction
		USE_RADIAL_BH_CORRECTION = true; //The radial BH force table iteration is done
	}
	#endif
	if(rank == 0)
	{
		// Loading the initial conditions
		if(load_IC(IC_FILE, IC_FORMAT) != 0)
		{
			fprintf(stderr, "Error: failed to load the initial conditions from %s file.\nExiting.\n", IC_FILE);
			return (-1);
		}
		#ifdef USE_BH
		if(RADIAL_BH_FORCE_CORRECTION == 1)
		{
			if (N!=N_radial_bh_force_correction)
			{
				fprintf(stderr, "Error: the number of particles in the radial BH force correction (%i) does not match the number of particles in the initial conditions (%i).\nPlease check your input files. Exiting.\n", N_radial_bh_force_correction, N);
				return (-1);
			}
		}
		#endif
		if(REDSHIFT_CONE == 1 && COSMOLOGY != 1)
		{
			fprintf(stderr, "Error: you can not use redshift cone output format in non-cosmological simulations. \nExiting.\n");
			return (-2);
		}
		if(REDSHIFT_CONE == 1 && OUTPUT_TIME_VARIABLE != 1)
		{
			fprintf(stderr, "Error: you must use redshift as output time variable in redshift cone simulations. \nExiting.\n");
			return (-2);
		}
		if(REDSHIFT_CONE == 1)
		{
			//Allocating memory for the bool array
			IN_CONE = new bool[N];
			std::fill(IN_CONE, IN_CONE+N, false ); //setting every element to false
			#ifdef HAVE_HDF5
			HDF5_redshiftcone_firstshell = 1;
			N_redshiftcone = 0; //number of particles written out to the redshiftcone
			#endif
		}
		//Converting units, if needed
		if(H0_INDEPENDENT_UNITS != 0 && COSMOLOGY == 1)
		{
			if(H0==0.0)
			{
				fprintf(stderr, "Error: Hubble constant is zero while using H0 independent units. This must be a mistake.\nExiting.\n");
				return (-2);
			}
			REAL H0_dimless = H0*UNIT_V/100.0;
			for(i=0;i<N;i++)
			{
				for(j=0;j<3;j++)
				{
					x[3*i + j] /= H0_dimless; //converting coordinates
				}
				M[i] /= H0_dimless; //converting masses
			}
		}
		//Rescaling speeds. We are using the same convention that the Gadget uses: http://wwwmpa.mpa-garching.mpg.de/gadget/gadget-list/0113.html
		if(COSMOLOGY == 1 && COMOVING_INTEGRATION == 1)
		{
			for(i=0;i<N;i++)
			{
				v[3*i] = v[3*i]/sqrt(a_start)/UNIT_V;
				v[3*i+1] = v[3*i+1]/sqrt(a_start)/UNIT_V;
				v[3*i+2] = v[3*i+2]/sqrt(a_start)/UNIT_V;
			}
		}
		else if(COSMOLOGY == 1 && COMOVING_INTEGRATION == 0)
		{
			for(i=0;i<N;i++)
			{
				v[3*i] = v[3*i]/UNIT_V;
				v[3*i+1] = v[3*i+1]/UNIT_V;
				v[3*i+2] = v[3*i+2]/UNIT_V;
			}
		}
		if(numtasks > 1 && F_buffer_allocated == false)
		{
			if(!(F_buffer = (REAL*)malloc(3*(N/numtasks)*sizeof(REAL))))
			{
				fprintf(stderr, "MPI task %i: failed to allocate memory for F_buffer.\n", rank);
				exit(-2);
		  	}
			F_buffer_allocated = true; //F_buffer is allocated
		}
	}
	#ifdef USE_CUDA
	if(argc == 3)
	{
		n_GPU = atoi( argv[2] );
		if(rank == 0)
			printf("Using %i cuda capable GPU per MPI task.\n\n", n_GPU);
	}
	else
	{
		n_GPU = 1;
	}
	#endif
	//Bcasting the number of particles
	MPI_Bcast(&N,1,MPI_INT,0,MPI_COMM_WORLD);
	if(rank == 0)
		N_mpi_thread = (N/numtasks) + (N%numtasks);
	else
		N_mpi_thread = N/numtasks;
	if(rank != 0)
	{
		//Allocating memory for the particle datas on the rank != 0 MPI threads
		//The rank 0 thread already allocated the memory during the particle loading
		//Allocating memory for the coordinates
		if(!(x = (REAL*)malloc(3*N*sizeof(REAL))))
		{
			fprintf(stderr, "MPI task %i: failed to allocate memory for x.\n", rank);
			exit(-2);
		}
		//Allocating memory for the forces. There is no need to allocate for N forces. N/numtasks should be enough
		if(!(F = (REAL*)malloc((3*N_mpi_thread)*sizeof(REAL))))
		{
			fprintf(stderr, "MPI task %i: failed to allocate memory for F.\n", rank);
			exit(-2);
		}
		//Allocating memory for the masses
			if(!(M = (REAL*)malloc(N*sizeof(REAL))))
		{
			fprintf(stderr, "MPI task %i: failed to allocate memory for M.\n", rank);
			exit(-2);
		}
		//Allocating memory for the softening lengths
			if(!(SOFT_LENGTH = (REAL*)malloc(N*sizeof(REAL))))
		{
			fprintf(stderr, "MPI task %i: failed to allocate memory for SOFT_LENGTH.\n", rank);
			exit(-2);
		}
	}
	//Bcasting the ICs to the rank!=0 threads
#ifdef USE_SINGLE_PRECISION
	MPI_Bcast(x,3*N,MPI_FLOAT,0,MPI_COMM_WORLD);
  	MPI_Bcast(M,N,MPI_FLOAT,0,MPI_COMM_WORLD);
#else
	MPI_Bcast(x,3*N,MPI_DOUBLE,0,MPI_COMM_WORLD);
	MPI_Bcast(M,N,MPI_DOUBLE,0,MPI_COMM_WORLD);
#endif
#ifdef GLASS_MAKING
	//setting all velocities to zero
	int k;
	if(rank == 0)
	{
		printf("Glass making: setting all velocities to zero.\n\n");
		for(i=0; i<N; i++)
		{
			for(k=0; k<3; k++)
			{
				v[3*i+k] = 0.0;
			}
		}
	}
#endif
	//Critical density and particle masses
	if(COSMOLOGY == 1)
	{
	if(COMOVING_INTEGRATION == 1)
	{
		Omega_dm = Omega_m-Omega_b;
		Omega_k = 1.-Omega_m-Omega_lambda-Omega_r;
		rho_crit = 3.0*H0*H0/(8.0*pi);
		#if defined(PERIODIC_Z)
			if(CYLINDRICAL_LOOKUP_TABLE_CALCULATED == false)
			{
				// in cylindrically symmetric simulations, the radial force is proportional to the mass in a unit cylinder
				mass_in_unit_sphere = (REAL) (2.0*pi*rho_crit*Omega_m);
				ewald_max = IS_PERIODIC+1;
				ewald_cut = ((REAL) ewald_max)-0.4; //cutoff radius for the ewald summation
				// Calculating the lookup table for the radial force calculation
				printf("Calculating the lookup table for the radial force calculation...\n");
				RADIAL_FORCE_TABLE = (REAL*)malloc(RADIAL_FORCE_TABLE_SIZE*sizeof(REAL)); //Allocating the lookup table
				if(IS_PERIODIC==1)
				{
					get_cylindrical_force_table(RADIAL_FORCE_TABLE, Rsim,L,RADIAL_FORCE_TABLE_SIZE,RADIAL_FORCE_ACCURACY);
					if(rank==0)
						printf("Cylindrical force table calculated.\nMagnitude of the gravitational pull correction of the finite Ewald cylinder at r = %.1f Mpc is |F(Lz=2.0Lsim)|/|F(Lz=inf)| = %.7f \n\n", Rsim, RADIAL_FORCE_TABLE[RADIAL_FORCE_TABLE_SIZE-1]);
				}
				else
				{
					get_cylindrical_force_table(RADIAL_FORCE_TABLE, Rsim, L*ewald_cut,RADIAL_FORCE_TABLE_SIZE,RADIAL_FORCE_ACCURACY);
					if(rank==0)
						printf("Cylindrical force table calculated.\nMagnitude of the gravitational pull correction of the finite Ewald cylinder at r = %.1f Mpc is |F(Lz=%.1fLsim)|/|F(Lz=inf)| = %.7f \n\n", Rsim, 2*ewald_cut, RADIAL_FORCE_TABLE[RADIAL_FORCE_TABLE_SIZE-1]);
				}
				CYLINDRICAL_LOOKUP_TABLE_CALCULATED = true;
			}
		#else
			// in spherical simmetrical simulations, the radial force is proportional to the mass in a unit sphere
			mass_in_unit_sphere = (REAL) (4.0*pi*rho_crit*Omega_m/3.0);
		#endif
		M_tmp = Omega_m*rho_crit*pow(L, 3.0)/((REAL) N); //Assuming DM only case
		#if defined(PERIODIC) || defined(PERIODIC_Z)
		if(IC_FORMAT == 1)
		{
			#if defined(PERIODIC)
			M_tmp = Omega_m*rho_crit*pow(L, 3.0)/((REAL) N); //Assuming DM only case
			if(rank == 0)
			{
				printf("Every particle has the same mass in periodic cosmological simulations, if the input is in GADGET format.\nM=%.10f*10e+11M_sol\n", M_tmp);
			}
			//Every particle has the same mass in periodic cosmological simulations, if the IC is in GADGET format
			for(i=0; i<N; i++)
			{
				M[i] = M_tmp;
			}
			#endif
		}
		//Calculating the total mean desity of the simulation volume
		//in here we sum the total particle mass with Kahan summation
		REAL rho_mean_full_box = 0.0;
		REAL Kahan_compensation = 0.0;
		REAL Kahan_t, Kahan_y;
		for(i=0;i<N;i++)
		{
			Kahan_y = M[i] - Kahan_compensation;
			Kahan_t = rho_mean_full_box + Kahan_y;
			Kahan_compensation = (Kahan_t - rho_mean_full_box) - Kahan_y;
			rho_mean_full_box = Kahan_t;
		}
		#if defined(PERIODIC)
			rho_mean_full_box /= pow(L, 3.0); //dividing the total mass by the simulation volume
		#elif defined(PERIODIC_Z)
			rho_mean_full_box /= (pi*Rsim*Rsim*L); //dividing the total mass by the simulation volume
		#endif
		if(fabs(rho_mean_full_box/(rho_crit*Omega_m) - 1) > 1e-5)
		{
			 #if COSMOPARAM>=0 || !defined(COSMOPARAM)
			 if(fabs(rho_mean_full_box/(rho_crit*Omega_m) - 1) > 1e-2)
			 {
			 	fprintf(stderr, "Error: The particle masses are inconsistent with the cosmological parameters!\nrho_part/rho_cosm = %.6f\nExiting.\n", rho_mean_full_box/(rho_crit*Omega_m));
			 	return (-1);
			 }
			 else
			 {
			 	printf("Warning: The particle masses are inconsistent with the cosmological parameters set in the parameter file!\nrho_part/rho_cosm = %.6f\n\tRescaling the particle masses with this number.\n", rho_mean_full_box/(rho_crit*Omega_m));
				for(i=0;i<N;i++)
			 	{
			 		M[i] /= (rho_mean_full_box/(rho_crit*Omega_m));
			 	}				
			 }
			 #else
			 printf("Warning: The particle masses are inconsistent with the cosmological parameters set in the parameter file!\nrho_part/rho_cosm = %.6f\nSince the expansion history read from an external file, this is not necessarily an error.\nPlease make sure that the particle masses are set correctly in the initial condition file.\n\n", rho_mean_full_box/(rho_crit*Omega_m));
			 #endif
		}
		#endif
	}
	else
	{
		if(IS_PERIODIC>0)
		{
			if(rank == 0)
				fprintf(stderr, "Error: COSMOLOGY = 1, IS_PERIODOC>0 and COMOVING_INTEGRATION = 0!\nThis code can not handle non-comoving periodic cosmological simulations.\nExiting.\n");
			return (-1);
		}
		if(rank == 0)
		{
			#if COSMOPARAM==0 || !defined(COSMOPARAM)
			printf("COSMOLOGY = 1 and COMOVING_INTEGRATION = 0:\nThis run will be in non-comoving coodinates. As a consequence, this will be a fully Newtonian cosmological simulation.\nMake sure that you set the correct parameters at the IC making.\na_max is used as maximal time in Gy in the parameter file.\n\n");
			#else
			printf("ERROR: COSMOLOGY = 1 and COMOVING_INTEGRATION = 0 and using ");
				#if COSMOPARAM==-1
					printf("tabulated expansion history.\n");
				#elif COSMOPARAM==1
					printf("wCDM cosmology parametrization.\n");
				#elif COSMOPARAM==2
					printf("CPL dark energy equation of state.\n");
				#else
					printf("unkown cosmology parametrization.\n");
				#endif
			printf("This is not supported in StePS version %s. Exiting...\n", PROGRAM_VERSION);
			return (-1);
			#endif
		}
		Omega_dm = Omega_m - Omega_b;
		Omega_k = 1.-Omega_m-Omega_lambda-Omega_r;
		rho_crit = 3*H0*H0/(8*pi);
	}
	}
	else
	{
		if(rank == 0)
			printf("Running non-cosmological gravitational N-body simulation.\n");
	}
	//Searching the minimal mass particle
	if(rank == 0)
	{
		calculate_softening_length(SOFT_LENGTH, M, N); //Calculating the softening lengths for the particles
	}
#ifdef USE_SINGLE_PRECISION
	MPI_Bcast(&M_min,1,MPI_FLOAT,0,MPI_COMM_WORLD);
	MPI_Bcast(&rho_part,1,MPI_FLOAT,0,MPI_COMM_WORLD);
	MPI_Bcast(SOFT_LENGTH,N,MPI_FLOAT,0,MPI_COMM_WORLD);
#else
	MPI_Bcast(&M_min,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
	MPI_Bcast(&rho_part,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
	MPI_Bcast(SOFT_LENGTH,N,MPI_DOUBLE,0,MPI_COMM_WORLD);
#endif
	beta = ParticleRadi;
	a=a_start;//scalefactor
	t_next = 0.;
	T = 0.0;
	REAL Delta_T_out = 0;
	if(COSMOLOGY == 0)
		a=1;//scalefactor
	int out_z_index = 0;
	int delta_z_index = 1;
	//Calculating initial time on the task=0 MPI thread
	if(rank == 0)
	{
	if(COSMOLOGY == 1)
	{
		#if COSMOPARAM==-1
		//Reading the tabulated expansion history
		if(file_exist(EXPANSION_FILE)!=0)
		{
			read_expansion_history(EXPANSION_FILE);
		}
		else
		{
			fprintf(stderr, "Error: The %s expansion history file does not exist!\nExiting.\n", EXPANSION_FILE);
			return (-1);
		}
		#endif
		a = a_start;
		a_tmp = a;
		if(COMOVING_INTEGRATION == 1)
		{
			printf("a_start=%.9f\tz=%.9f\n", a, 1/a-1);
		}
		T = friedmann_solver_start(1,0,h_min*0.05,a_start);
		if(HAVE_OUT_LIST == 0)
		{
			if(OUTPUT_TIME_VARIABLE==0)
			{
				Delta_T_out = H_OUT/UNIT_T; //Output frequency in internal time units
				if(FIRST_T_OUT >= T) //Calculating first output time
				{
					t_next = FIRST_T_OUT/UNIT_T;
				}
				else
				{
					t_next = T+Delta_T_out;
				}
			}
			else
			{
				Delta_T_out = H_OUT; //Output frequency in redshift
				if(FIRST_T_OUT >= T) //Calculating first output redshift
				{
					t_next = FIRST_T_OUT;
				}
				else
				{
					t_next = FIRST_T_OUT - Delta_T_out;
				}
			}
		}
		else
		{
			if(OUTPUT_TIME_VARIABLE==1)
			{
				if(1.0/a-1.0 > out_list[0])
				{
					t_next = out_list[0];
				}
				else
				{
					out_z_index=0;
					while(out_list[out_z_index] >= 1.0/a-1.0)
					{
						out_z_index++;
						t_next = out_list[out_z_index];
						if(out_z_index >= out_list_size || out_list[out_z_index] < 1.0/a_max-1.0)
						{
							fprintf(stderr, "Error: No valid output redshift!\nExiting.\n");
							return (-2);
						}
					}
					printf("Next output redshift = %f (out_list[out_z_index=%i]=%f)\n",t_next,out_z_index,out_list[out_z_index]);
				}
			}
			else
			{
				if(T < out_list[0])
				{
					t_next = out_list[0];
				}
				else
				{
					i=0;
					while(out_list[i] <= T)
					{
						t_next = out_list[i];
						i++;
						if(i == out_list_size)
						{
							fprintf(stderr, "Error: No valid output time found in the OUT_LST file!\nExiting.\n");
							return (-2);
						}
					}
					printf("Next output time = %fGy (out_list[%i]=%fGy)\n",t_next,i,out_list[out_z_index]);
				}
			}
		}
		if(COMOVING_INTEGRATION == 1)
		{
		printf("Initial time:\t\tt_start = %.10f Gy\nInitial scalefactor:\ta_start = %.8f\nMaximal scalefactor:\ta_max   = %.8f\n\n", T*UNIT_T, a, a_max);
		}
		if(COMOVING_INTEGRATION == 0)
		{
			Hubble_param = 0;
			a_tmp = 0;
			a_max = a_max/UNIT_T;
			a = 1;
			printf("Initial time:\tt_start = %.10f Gy\nMaximal time:\tt_max   = %.8f Gy\n\n", T*UNIT_T, a_max*UNIT_T);
		}
	}
	else
	{
		a = 1;
		Hubble_param = 0;
		T = 0.0; //If we do not running cosmological simulations, the initial time will be 0.
		printf("t_start = %f\tt_max = %f\n", T, a_max);
		a_tmp = 0;
		Delta_T_out = H_OUT;
		if(HAVE_OUT_LIST==0)
		{
			t_next = T+Delta_T_out;
		}
		else
		{
			i = 0;
			while(out_list[i] < 0.0)
			{
				t_next=out_list[i];
				i++;
				if(i == out_list_size && out_list[i] < T)
				{
					fprintf(stderr, "Error: No valid output time found in the OUT_LST file!\nExiting.\n");
					return (-2);
				}
			}
		}
	}
	}
	//Bcasting the initial time and other variables
	MPI_Bcast(&t_next,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
	MPI_Bcast(&T,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
	double SIM_omp_start_time;
	//Timing
	SIM_omp_start_time = omp_get_wtime();
	//Timing
	if(rank == 0)
		printf("Initial force calculation...\n");
	//Initial force calculation
	if(rank==0)
	{
		ID_MPI_min = 0;
		ID_MPI_max = (N%numtasks) + (rank+1)*(N/numtasks)-1;
		#if defined(PERIODIC)
			//3-torus periodic boundary conditions
			forces_periodic(x, F, ID_MPI_min, ID_MPI_max);
		#elif defined(PERIODIC_Z)
			//cylindrical periodic boundary conditions
			forces_periodic_z(x, F, ID_MPI_min, ID_MPI_max);
		#else
			//Spherical non-periodic [StePS] boundary conditions
			forces(x, F, ID_MPI_min, ID_MPI_max);
		#endif
	}
	else
	{
		ID_MPI_min = (N%numtasks) + (rank)*(N/numtasks);
		ID_MPI_max = (N%numtasks) + (rank+1)*(N/numtasks)-1;
		#if defined(PERIODIC)
			forces_periodic(x, F, ID_MPI_min, ID_MPI_max);
		#elif defined(PERIODIC_Z)
			forces_periodic_z(x, F, ID_MPI_min, ID_MPI_max);
		#else
			forces(x, F, ID_MPI_min, ID_MPI_max);
		#endif
	}
	//if the force calculation is finished, the calculated forces should be collected into the rank=0 thread`s F matrix
	if(rank !=0)
	{
	#ifdef USE_SINGLE_PRECISION
		MPI_Send(F, 3*N_mpi_thread, MPI_FLOAT, 0, rank, MPI_COMM_WORLD);
	#else
		MPI_Send(F, 3*N_mpi_thread, MPI_DOUBLE, 0, rank, MPI_COMM_WORLD);
	#endif
	}
	else
	{
		if(numtasks > 1)
		{
			for(i=1; i<numtasks;i++)
			{
				BUFFER_start_ID = i*(N/numtasks)+(N%numtasks);
				#ifdef USE_SINGLE_PRECISION
					MPI_Recv(F_buffer, 3*(N/numtasks), MPI_FLOAT, i, i, MPI_COMM_WORLD, &Stat);
				#else
					MPI_Recv(F_buffer, 3*(N/numtasks), MPI_DOUBLE, i, i, MPI_COMM_WORLD, &Stat);
				#endif
				for(j=0; j<(N/numtasks); j++)
				{
					F[3*(BUFFER_start_ID+j)] = F_buffer[3*j];
					F[3*(BUFFER_start_ID+j)+1] = F_buffer[3*j+1];
					F[3*(BUFFER_start_ID+j)+2] = F_buffer[3*j+2];
				}
			}
		}
	}
	//The simulation is starting...
	//Calculating the initial Hubble parameter, using the Friedmann-equations
	if(COSMOLOGY == 1 && rank == 0)
	{
		Hubble_param = CALCULATE_Hubble_param(a);
		printf("Initial Hubble-parameter:\nH(z=%f) = %fkm/s/Mpc\n\n", 1.0/a-1.0, Hubble_param*UNIT_V);
	}
	if(COSMOLOGY == 0 || COMOVING_INTEGRATION == 0)
	{
		Hubble_param = 0;
	}
	if(rank == 0)
	{
		h = calculate_init_h();
		if(h>h_max)
    {
			if(COSMOLOGY == 1)
				printf("Initial timestep length %fMy is larger than h_max. Setting timestep length to %fMy.\n", h*UNIT_T*1000.0, h_max*UNIT_T*1000.0);
			else
			printf("Initial timestep length %f is larger than h_max. Setting timestep length to %f.\n", h, h_max);
			h=h_max;
    }
		else if(h<h_min)
		{
			if(COSMOLOGY == 1)
				printf("Initial timestep length %fMy is smaller than h_min. Setting timestep length to %fMy.\n", h*UNIT_T*1000.0, h_min*UNIT_T*1000.0);
			else
			printf("Initial timestep length %f is smaller than h_min. Setting timestep length to %f.\n", h, h_min);
			h = h_min;
		}
	}
	MPI_Bcast(&h,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
	if(rank == 0)
		printf("The simulation is starting...\n");
	REAL T_prev,Hubble_param_prev;
	T_prev = T;
	Hubble_param_prev = Hubble_param;
	//Main loop
	for(t=0; a_tmp<a_max; t++)
	{
		if(rank == 0)
		{
			printf("\n\n------------------------------------------------------------------------------------------------\n");
			if(COSMOLOGY == 1)
    		{
				if(COMOVING_INTEGRATION == 1)
				{
					if(h*UNIT_T >= 1.0)
					printf("Timestep %i, t=%.8fGy, h=%fGy, a=%.8f, H=%.8fkm/s/Mpc, z=%.8f:\n", t, T*UNIT_T, h*UNIT_T, a, Hubble_param*UNIT_V, 1.0/a-1.0);
					else
						printf("Timestep %i, t=%.8fGy, h=%fMy, a=%.8f, H=%.8fkm/s/Mpc, z=%.8f:\n", t, T*UNIT_T, h*UNIT_T*1000.0, a, Hubble_param*UNIT_V, 1.0/a-1.0);
					if(OUTPUT_TIME_VARIABLE == 0)
						printf("Next output time = %f Gy\n",t_next*UNIT_T );
					else if (OUTPUT_TIME_VARIABLE == 1)
						printf("Next output redshift = %f\n",t_next);
				}
				else
				{
					if(h*UNIT_T >= 1.0)
						printf("Timestep %i, t=%.8fGy, h=%fGy\n", t, T*UNIT_T, h*UNIT_T);
					else
						printf("Timestep %i, t=%.8fGy, h=%fMy\n", t, T*UNIT_T, h*UNIT_T*1000.0);
					if(OUTPUT_TIME_VARIABLE == 0)
						printf("Next output time = %f Gy\n",t_next*UNIT_T );
				}
      		}
			else
			{
            	printf("Timestep %i, t=%f, h=%f:\n", t, T, h);
				printf("Next output time = %f\n",t_next);
    		}
		}
		Hubble_param_prev = Hubble_param;
		T_prev = T;
		T = T+h;
		step(x, v, F);
		if(rank == 0)
		{
			Log_write();	//Writing logfile
			if(HAVE_OUT_LIST == 0)
			{
				if(OUTPUT_TIME_VARIABLE == 0)
				{
					if(T > t_next)
					{
						if(OUTPUT_FORMAT == 0)
							write_ascii_snapshot(x, v);
						#ifdef HAVE_HDF5
						if(OUTPUT_FORMAT == 2)
							write_hdf5_snapshot(x, v, M);
						#endif
						t_next+=Delta_T_out;
						printf("...done.\n");
					}
				}
				else
				{
					if( 1.0/a-1.0 < t_next)
					{
						if(OUTPUT_FORMAT == 0)
							write_ascii_snapshot(x, v);
						#ifdef HAVE_HDF5
						if(OUTPUT_FORMAT == 2)
							write_hdf5_snapshot(x, v, M);
						#endif
						t_next-=Delta_T_out;
						if(COSMOLOGY == 1)
						{
							printf("t = %f Gy\n\th=%f Gy\n", T*UNIT_T, h*UNIT_T);
						}
						else
						{
							printf("t = %f\n\terr_max = %e\th=%f\n", T, errmax, h);
						}
					}
				}
			}
			else
			{
				if(OUTPUT_TIME_VARIABLE == 1)
				{
					if( 1.0/a-1.0 < t_next)
					{
						if(REDSHIFT_CONE != 1)
						{
							if(OUTPUT_FORMAT == 0)
								write_ascii_snapshot(x, v);
							#ifdef HAVE_HDF5
							if(OUTPUT_FORMAT == 2)
								write_hdf5_snapshot(x, v, M);
							#endif
							out_z_index += delta_z_index;
							t_next = out_list[out_z_index];
						}
						else
						{
							if(a_tmp >= a_max)
							{
								CONE_ALL = 1;
								printf("Last timestep.\n");
								if(OUTPUT_FORMAT == 0)
									write_ascii_snapshot(x, v);
								#ifdef HAVE_HDF5
								if(OUTPUT_FORMAT == 2)
									write_hdf5_snapshot(x, v, M);
								#endif
							}
							write_redshift_cone(x, v, r_bin_limits, out_z_index, delta_z_index, CONE_ALL);
							if(1.0/a-1.0 <= out_list[out_z_index+delta_z_index])
							{
								if( (out_z_index+delta_z_index+8) < out_list_size)
									delta_z_index += 8;
								else
									CONE_ALL = 1;
							}
							if(CONE_ALL == 1)
							{
								t_next = 0.0;
							}
							else
							{
								out_z_index += delta_z_index;
								t_next = out_list[out_z_index];
							}
							if(MIN_REDSHIFT>t_next && CONE_ALL != 1)
							{
								CONE_ALL = 1;
								printf("Warning: The simulation reached the minimal z = %f redshift. After this point the z=0 coordinates will be written out with redshifts taken from the input file. This can cause inconsistencies, if this minimal redshift is not low enough.\n", MIN_REDSHIFT);
								t_next = 0.0;
							}
						}
					}
				}
				else
				{
					if(T >= t_next)
					{
						if(OUTPUT_FORMAT == 0)
							write_ascii_snapshot(x, v);
						#ifdef HAVE_HDF5
						if(OUTPUT_FORMAT == 2)
							write_hdf5_snapshot(x, v, M);
						#endif
						out_z_index += delta_z_index;
						t_next = out_list[out_z_index];
						if(COSMOLOGY == 1)
						{
							printf("t = %f Gy\n\th=%f Gy\n", T*UNIT_T, h*UNIT_T);
						}
						else
						{
							printf("t = %f\n\terr_max = %e\th=%f\n", T, errmax, h);
						}
					}
				}
			}
			h = (double) pow(2*ACC_PARAM/errmax, 0.5);
			if(h<h_min)
			{
				h=h_min;
			}
			else if(h>h_max)
			{
				h=h_max;
			}
			if((h+T > t_next) && OUTPUT_TIME_VARIABLE == 0)
			{
				h = t_next-T+(1E-9*h_min);
			}
		}
		MPI_Bcast(&h,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
		fflush(stdout);
		MPI_Barrier(MPI_COMM_WORLD);
		if(ForceError == true)
		{
			if(rank == 0)
				printf("\nFatal error has been detected in the force calculation.\n Exiting...\n");
			break;
		}
		if( TIME_LIMIT_IN_MINS != 0 && (omp_get_wtime()-SIM_omp_start_time)/60.0 >= TIME_LIMIT_IN_MINS)
		{
			if(rank == 0)
				printf("\nSimulation wall-clock time limit reached (%.1fmin >= %.1fmin). Stopping...\n", (omp_get_wtime()-SIM_omp_start_time)/60.0, TIME_LIMIT_IN_MINS);
			break;
		}
		MPI_Bcast(&a_tmp,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
	}
	if(OUTPUT_TIME_VARIABLE == 0 && rank == 0)
	{
		if(OUTPUT_FORMAT == 0)
			write_ascii_snapshot(x, v); //writing output
		#ifdef HAVE_HDF5
		if(OUTPUT_FORMAT == 2)
			write_hdf5_snapshot(x, v, M);
		#endif
	}
	if(rank == 0)
	{
		printf("\n\n------------------------------------------------------------------------------------------------\n");
		printf("The simulation ended. The final state:\n");
		if(COSMOLOGY == 1)
		{
			if(COMOVING_INTEGRATION == 1)
			{
				printf("Timestep %i, t=%.8fGy, h=%fMy, a=%.8f, H=%.8fkm/s/Mpc, z=%.8f\n", t, T*UNIT_T, h*UNIT_T*1000.0, a, Hubble_param*UNIT_V, 1.0/a-1.0);

				double a_end, b_end;
				a_end = (Hubble_param - Hubble_param_prev)/(a-a_prev);
				b_end = Hubble_param_prev-a_end*a_prev;
				double H_end = a_max*a_end+b_end;
				a_end = (T - T_prev)/(a-a_prev);
			        b_end = T_prev-a_end*a_prev;
				double T_end = a_max*a_end+b_end;
				printf("\nAt a = %f state, with linear interpolation:\n",a_max);
				printf("t=%.8fGy, a=%.8f, H=%.8fkm/s/Mpc\n\n", T_end*UNIT_T, a_max, H_end*UNIT_V);
			}
			else
			{
				printf("Timestep %i, t=%.8fGy, h=%fGy\n", t, T*UNIT_T, h*UNIT_T);
			}
		}
		else
		{
			printf("Timestep %i, t=%f, h=%f, a=%f:\n", t, T, h, a);
		}
		//Timing
		double SIM_omp_end_time = omp_get_wtime();
		//Timing
		printf("Wall-clock time of the simulation = %fs (=%fh)\n", SIM_omp_end_time-SIM_omp_start_time, (SIM_omp_end_time-SIM_omp_start_time)/3600.0);
		#ifdef USE_CUDA
		printf("Total GPU time = %fh\n", (SIM_omp_end_time-SIM_omp_start_time)*numtasks*n_GPU/3600.0);
		#else
		printf("Total CPU time = %fh\n", (SIM_omp_end_time-SIM_omp_start_time)*numtasks*omp_threads/3600.0);
		#endif
	}
	// done with MPI
	MPI_Finalize();
	return 0;
}

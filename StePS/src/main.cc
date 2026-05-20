/********************************************************************************/
/*  StePS - STEreographically Projected cosmological Simulations                */
/*    Copyright (C) 2017-2026 Gabor Racz, Balazs Pal                            */
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
#ifdef POINCARE_DODECAHEDRAL
#include "pds_group.h"
#endif
#ifdef HAVE_HDF5
#include <hdf5.h>
#endif

#ifdef USE_SINGLE_PRECISION
typedef float REAL;
#else
typedef double REAL;
#endif

int t,N;
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
int N_saved_ics = 0; //number of saved ICs, only used for naming the output files when testing the force accuracy.

int n_GPU; //number of cuda capable GPUs
int numtasks, rank; //Variables for MPI

MPI_Status Stat;
int BUFFER_start_ID;
REAL* F_buffer;

REAL x4, err, errmax;
REAL beta, ParticleRadi, rho_part, M_min;

#ifdef USE_BH
	REAL THETA;//value for the opening angle (used in BH forces)
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
int H0_INDEPENDENT_UNITS; //0: i/o in Mpc, Msol, etc. 1: i/o in Mpc/h, Msol/h, etc.
double *out_list; //Output redshits
double *r_bin_limits; //bin limints in Dc for redshift cone simulations
int out_list_size; //Number of output redshits
unsigned int N_snapshot; //number of written out snapshots
bool save_accelerations = false; //bool variable to decide whether to save accelerations, only true if SAVE_ACCELERATIONS is defined

//timing and workload-balance variables
double TIME_LIMIT_IN_MINS; //Simulation wall-clock time limit in minutes.
double *mpi_time_array; //array for storing the time spent in each MPI thread
int **mpi_particle_range; //2-index array for storing the particle ID ranges of the MPI threads. first index: mpi_task_id, second index: 0-start_id, 1-end_id, 2-npart
int ID_MPI_min, ID_MPI_max; //max and min ID of of calculated forces in the actual MPI thread
int N_mpi_thread; //Number of calculated forces in the actual MPI thread

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
#if defined(PERIODIC)
	//Variables only used in T^3 periodic simulations
	REAL *T3_EWALD_FORCE_TABLE; //Ewald force correction lookup table for T^3 topology
	char ewaldfilepath[0x100];
	int N_EWALD_FORCE_GRID; //size of the ewald force lookup table
#endif
#if defined(POINCARE_DODECAHEDRAL)
	//Variables only used in S^3/I* (Poincare Dodecahedral Space) simulations
	REAL *PDS_Q;                 //4D quaternion positions (4*N REAL values)
	REAL *PDS_EWALD_FORCE_TABLE; //1D Ewald correction table indexed by geodesic distance chi in [0,pi]
	int   N_PDS_EWALD_GRID;      //number of grid points in the PDS Ewald table
	REAL  PDS_R_CURV;            //curvature radius of S^3 in internal length units (Mpc)
	char  pds_ewaldfilepath[0x100];
#endif
#if defined(PERIODIC_Z)
	//Variables only used in S^1 x R^2 simulations
	int RADIAL_FORCE_TABLE_SIZE; //size of the lookup table for the radial force calculation
	int RADIAL_FORCE_ACCURACY; //number of points used in the integration for the lookup table
	REAL *RADIAL_FORCE_TABLE; //lookup table for the radial force calculation
	bool CYLINDRICAL_LOOKUP_TABLE_CALCULATED=false; //true, if the cylindrical lookup table is calculated
	char CYLINDRICAL_LOOKUP_TABLE_PATH[0x100]; //output file path for the radial cylindrical force lookup table
	#if defined(PERIODIC_Z_NOLOOKUP)
		//Direct periodic real-space summation variables
		int ewald_max; //number of images in the z direction
		REAL ewald_cut; //cutoff radius for the ewald summation
	#else
		//Ewald summation variables
		REAL *S1R2_EWALD_FORCE_TABLE; //Ewald force correction lookup table for S^1 x R^2 topology
		char ewaldfilepath[0x100];
		int Nz_EWALD_FORCE_GRID; //size of the ewald force lookup in the z direction
		int Nrho_EWALD_FORCE_GRID; //size of the ewald force lookup in the radial direction
	#endif
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
	#if !defined(PERIODIC)
		char RADIAL_BH_FORCE_CORRECTION_PATH[0x100]; //output file path for the radial BH force correction table
	#endif
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
void redistribute_workload(double *mpi_time_array, int numtasks, int N, int **mpi_particle_range);
double friedmann_solver_start(double a0, double t0, double h, double a_start);
double friedmann_solver_step(double a0, double h);
double CALCULATE_Hubble_param(double a);
double CALCULATE_decel_param(double a);
REAL kahan_sum(const REAL *array, size_t N);
//Functions used in MPI parallelisation
void BCAST_global_parameters();
void BCAST_MPI_particle_ranges();
#ifdef USE_BH
void get_radial_bh_force_correction_table(REAL *RADIAL_BH_FORCE_TABLE, int *RADIAL_BH_N_TABLE, int TABLE_SIZE, REAL *F, REAL *x, int N);
#endif
#ifdef PERIODIC
int ewald_space(REAL R, int ewald_index[][4]);
void calculate_t3_ewald_lookup_table(int Ngrid, REAL L, REAL alpha, int realspace_el, int recspace_el, int realspace_ewald_index[][4], int recspace_ewald_index[][4], REAL rel_cut, REAL rec_cut, REAL *T3_EWALD_FORCE_TABLE);
int save_t3_ewald_lookup_table(const char *filename, int Ngrid, const REAL *T3_EWALD_FORCE_TABLE);
int load_t3_ewald_lookup_table(const char *filename, int *Ngrid, REAL **table_out);
#elif defined(PERIODIC_Z) && !defined(PERIODIC_Z_NOLOOKUP)
//Functions for S^1 x R^2 Ewald summation
void calculate_S1R2ewald_correction_table(int Nrho, int Nz, REAL rho_max, REAL Lz, REAL alpha, int nmax, int mmax, REAL*& S1R2_EWALD_FORCE_TABLE);
#ifdef HAVE_HDF5
int load_s1r2_ewald_lookup_table(const char *filename, int *Nrho_grid, int *Nz_grid, REAL **table_out);
int save_s1r2_ewald_lookup_table(const char *filename, int Nrho_grid, int Nz_grid, const REAL *S1R2_EWALD_FORCE_TABLE);
#endif
#elif defined(POINCARE_DODECAHEDRAL)
//Functions for S^3/I* (Poincare Dodecahedral Space) simulations
void calculate_pds_ewald_lookup_table(int Ngrid, double R_curv, REAL *table);
double pds_ewald_interpolate(const REAL *table, int Ngrid, double R_curv, double chi);
void forces_pds(REAL *q, REAL *F, int ID_min, int ID_max);
#ifdef HAVE_HDF5
int save_pds_ewald_lookup_table(const char *filename, int Ngrid, double R_curv, const REAL *PDS_EWALD_FORCE_TABLE);
int load_pds_ewald_lookup_table(const char *filename, int *Ngrid, double *R_curv, REAL **table_out);
#endif
#endif
#if defined(PERIODIC_Z)
//Functions for direct periodic S^1 x R^2 real space force calculation
void get_cylindrical_force_table(REAL* FORCE_TABLE, REAL R, REAL Lz, int TABLE_SIZE, int RADIAL_FORCE_ACCURACY);
#endif

//helper functions
void set_REAL_array_to_zero(REAL *array, int N);

//Input/Output functions
int file_exist(char *file_name);
int dir_exist(char *dir_name);
int load_IC(char *IC_FILE, int IC_FORMAT);
int read_OUT_LST();
void write_redshift_cone(REAL *x, REAL *v, double *limits, int z_index, int delta_z_index, int ALL);
void write_ascii_snapshot(REAL* x, REAL *v);
void Log_write();
void save_function_to_ascii_table(char *filename, REAL x_min, REAL x_max, REAL deltax, REAL *values, int Ntable, char* header);
#ifdef HAVE_HDF5
int N_redshiftcone, HDF5_redshiftcone_firstshell;
//Functions for HDF5 I/O
void write_hdf5_snapshot(REAL *x, REAL *v, REAL *M, bool save_accelerations, REAL *F, bool IC_file);
void write_header_attributes_in_hdf5(hid_t handle);
#endif
#if COSMOPARAM==-1
void read_expansion_history(char* filename);
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
		printf("+-----------------------------------------------------------------------------------------------+\n|   _____ _       _____   _____ \t\t\t\t\t\t\t\t|\n|  / ____| |     |  __ \\ / ____|\t\t\t\t\t\t\t\t|\n| | (___ | |_ ___| |__) | (___  \t\t\t\t\t\t\t\t|\n|  \\___ \\| __/ _ \\  ___/ \\___ \\ \t\t\t\t\t\t\t\t|\n|  ____) | ||  __/ |     ____) |\t\t\t\t\t\t\t\t|\n| |_____/ \\__\\___|_|    |_____/ \t\t\t\t\t\t\t\t|\n|StePS %s\t\t\t\t\t\t\t\t\t\t\t|\n| (STEreographically Projected cosmological Simulations)\t\t\t\t\t|\n+-----------------------------------------------------------------------------------------------+\n| Copyright (C) 2017-2026 Gabor Racz et al.\t\t\t\t\t\t\t|\n|\tDepartment of Physics, University of Helsinki | Helsinki, Finland\t\t\t|\n|\tJet Propulsion Laboratory, California Institute of Technology | Pasadena, CA, USA\t|\n|\tDepartment of Physics of Complex Systems, Eotvos Lorand University | Budapest, Hungary\t|\n|\tDepartment of Physics & Astronomy, Johns Hopkins University | Baltimore, MD, USA\t|\n|\t\t\t\t\t\t\t\t\t\t\t\t|\n|", PROGRAM_VERSION);
		printf(" Build date: %s\t\t\t\t\t\t\t|\n|",  BUILD_DATE);
		printf(" Compiled with: %s", COMPILER_VERSION);
		unsigned long int I;
		for(I = 0; I<10-((sizeof(COMPILER_VERSION)-1)/8); I++)
			printf("\t");
		printf("|\n| Git branch: %s", GIT_BRANCH);
		for(I = 0; I<10-((sizeof(GIT_BRANCH)-1)/8); I++)
			printf("\t");
		printf("|\n| Git commit: %s", GIT_COMMIT_ID);
		for(I = 0; I<11-((sizeof(GIT_COMMIT_ID)-1)/8); I++)
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
	{
		printf("\tUsing CUDA capable GPUs for force calculation.\n");
	}
	#ifdef USE_BH
	if(rank == 0)
	{
		fprintf(stderr,"\nError: Barnes-Hut octree force calculation is not (yet) implemented on CUDA capable GPUs.\nPlease recompile StePS without the USE_BH or USE_CUDA option.\nExiting...\n");
	}
	return (-1);
	#endif
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
		printf("\tPeriodic boundary conditions. (T^3 topological manifold)\n");
	#elif defined(PERIODIC_Z)
		if(rank == 0)
			printf("\tPeriodic boundary conditions in the z direction. (S^1 x R^2 topological manifold)\n");
		#if defined(PERIODIC_Z_NOLOOKUP)
		if(rank == 0)
			printf("\t\tUsing direct periodic real-space summation in force calculation.\n");
		#else
		if(rank == 0)
		{
			#if !defined(PERIODIC_Z_RSPACELOOKUP)
			printf("\t\tUsing S^1 x R^2 Ewald lookup table in force calculation. (Tornberg 2015 method)\n");
			#else
			printf("\t\tUsing S^1 x R^2 real-space lookup table in force calculation.\n");
			#endif
			printf("\t\tEwald interpolation method: ");
			#if EWALD_INTERPOLATION_ORDER==0
				printf("NGP (Nearest Grid Point)\n");
			#elif EWALD_INTERPOLATION_ORDER==2
				printf("CIC (Cloud-in-Cell)\n");
			#elif EWALD_INTERPOLATION_ORDER==4
				printf("TSC (Triangular Shaped Cloud)\n");
			#else
				//this should never happen
				printf("Unknown method (defaulting to TSC)\n");
			#endif
		}
		#endif
	#else
	if(rank == 0)
		printf("\tNon-periodic boundary conditions. (R^3 topological manifold)\n");
	#endif
	// Warn if IS_PERIODIC in the parameter file is inconsistent with the compiled topology.
	// IS_PERIODIC > 0 in a non-periodic binary, or IS_PERIODIC == 0 in a periodic binary,
	// will silently use the wrong force kernel.
	if(rank == 0)
	{
		#if defined(PERIODIC)
		if(IS_PERIODIC == 0)
			printf("\nWARNING: IS_PERIODIC=0 in the parameter file, but this binary was compiled for T^3 (PERIODIC). The T^3 force kernel will be used.\n\n");
		#elif defined(PERIODIC_Z)
		if(IS_PERIODIC == 0)
			printf("\nWARNING: IS_PERIODIC=0 in the parameter file, but this binary was compiled for S^1 x R^2 (PERIODIC_Z). The cylindrical force kernel will be used.\n\n");
		#elif defined(POINCARE_DODECAHEDRAL)
		if(IS_PERIODIC == 0)
			printf("\nWARNING: IS_PERIODIC=0 in the parameter file, but this binary was compiled for S^3/I* (POINCARE_DODECAHEDRAL). The PDS force kernel will be used.\n\n");
		#else
		if(IS_PERIODIC > 0)
			printf("\nWARNING: IS_PERIODIC=%d in the parameter file, but this binary was compiled for open boundaries (R^3). Periodic forces will NOT be applied.\n\n", IS_PERIODIC);
		#endif
	}
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
	#ifdef SAVE_ACCELERATIONS
		#ifdef HAVE_HDF5
		save_accelerations = true;
		if(rank == 0)
			printf("\tCalculated accelerations will be saved to the HDF5 snapshots.\n");
		#else
		save_accelerations = false;
		if(rank == 0)
			printf("\tWarning: Calculated accelerations will not be saved. This is only possible with HDF5 output.\n");
		#endif
	#else
		save_accelerations = false;
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
	#if defined(GLASS_MAKING) && defined(USE_BH) && !defined(PERIODIC)
		if(rank == 0)
			printf("Warning: Using Barnes-Hut tree (Octree) algorithm during glass making in non-periodic\nsimulations can cause significant force calculation errors, especially in the radial\ndirection. Consider using direct summation for better glass quality.\n\n");
	#endif
	int i,j;
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
	fflush(stdout);
	BCAST_global_parameters();
	if(rank == 0)
	{
		//allocating memory for the MPI timing array with malloc, and setting the values to 1.0
		mpi_time_array = (double*) malloc(numtasks * sizeof(double));
		for(i=0; i<numtasks; i++)
		{
			mpi_time_array[i] = 1.0;
		}
		//allocating memory for the MPI particle range array with malloc
		mpi_particle_range = (int **) malloc(numtasks * sizeof(int *));
		for(i=0; i<numtasks; i++)
		{
			mpi_particle_range[i] = (int *) malloc(3 * sizeof(int));
			mpi_particle_range[i][0] = 0; //start ID
			mpi_particle_range[i][1] = 0; //end ID
			mpi_particle_range[i][2] = 0; //number of particles
		}
	}
	#ifdef PERIODIC
		if(IS_PERIODIC < 1 || IS_PERIODIC > 4)
		{
			if(rank == 0)
				fprintf(stderr, "Error: Bad boundary condition were set in the paramfile!\nThis executable are able to deal with periodic simulation only.\nExiting.\n");
			fflush(stdout);
			return (-2);
		}

		if(IS_PERIODIC>1)
		{
			if(rank == 0)
			{
				//variables only used in periodic ewald force calculation
				int number_of_ewald_real_space_indexes,number_of_ewald_reciprocal_space_indexes;
				int ewald_real_space_indexes[739][4];
				int ewald_reciprocal_space_indexes[11459][4];
				double EWALD_alpha; //Ewald alpha parameter
				REAL rel_cut, rec_cut;
				EWALD_alpha = 2.0/L;
				//rank 0 MPI thread is calculating the ewald lookup table
				char EwaldTableFile[] = "Ewald_table_lowres.hdf5";
				if(IS_PERIODIC==2)
				{
					printf("Ewald force calculation is on. (Ewald cut is 2.6*L in real and 8 in reciprocal space)\nCalculating Ewald lookup tables...\n");
					rel_cut = 2.6;
					rec_cut = 8.0;
					N_EWALD_FORCE_GRID = 63; //size of the ewald force lookup table (must be odd to include the center point and axes)
					
				}
				else if(IS_PERIODIC==3)
				{
					printf("Medium precision Ewald force calculation is on. (Ewald cut is 3.6*L in real and 10 in reciprocal space)\nCalculating Ewald lookup tables...\n\n");
					rel_cut = 3.6;
					rec_cut = 10.0;
					N_EWALD_FORCE_GRID = 127; //size of the ewald force lookup table (must be odd to include the center point and axes)
					strcpy(EwaldTableFile, "Ewald_table_medres.hdf5");
				}
				else
				{
					printf("High precision Ewald force calculation is on. (Ewald cut is 4.6*L in real and 12 in reciprocal space)\nCalculating Ewald lookup tables...\n\n");
					rel_cut = 4.6;
					rec_cut = 12.0;
					N_EWALD_FORCE_GRID = 255; //size of the ewald force lookup table (must be odd to include the center point and axes)
					strcpy(EwaldTableFile, "Ewald_table_higres.hdf5");
				}
				//Allocating memory for the ewald lookup table in the rank 0 MPI thread
				printf("MPI task %i: Allocating memory for the Ewald lookup table with %i^3 grid points...\n", rank, N_EWALD_FORCE_GRID);
				fflush(stdout);
				if(!(T3_EWALD_FORCE_TABLE = (REAL*)malloc((size_t)N_EWALD_FORCE_GRID*N_EWALD_FORCE_GRID*N_EWALD_FORCE_GRID*3*sizeof(REAL))))
				{
					fprintf(stderr, "MPI task %i: failed to allocate memory for The Ewald lookup table.\n", rank);
					exit(-2);
				}
				#ifdef HAVE_HDF5
					//if we have HDF5, we save/load the Ewald table in HDF5 format				
					if(snprintf(ewaldfilepath, sizeof(ewaldfilepath), "%s%s", OUT_DIR, EwaldTableFile) < 0)
					{
						fprintf(stderr, "Error: The name of the ewald table got truncated.\nAborting.\n");
						abort();
					}
					if(file_exist(ewaldfilepath) == 0)
					{
						printf("Ewald lookup table file (%s) not found.\nCalculating new lookup table...\n", ewaldfilepath);
						double EWALD_omp_start_time = omp_get_wtime(); //Timing
						number_of_ewald_real_space_indexes = ewald_space(rel_cut+1,ewald_real_space_indexes); //real space indexes
						number_of_ewald_reciprocal_space_indexes = ewald_space(rec_cut+2,ewald_reciprocal_space_indexes); //reciprocal space indexes
						printf("Ewald real space images: %i; Ewald reciprocal space images: %i\n", number_of_ewald_real_space_indexes, number_of_ewald_reciprocal_space_indexes);
						calculate_t3_ewald_lookup_table(N_EWALD_FORCE_GRID, L, EWALD_alpha, number_of_ewald_real_space_indexes, number_of_ewald_reciprocal_space_indexes, ewald_real_space_indexes, ewald_reciprocal_space_indexes, rel_cut, rec_cut, T3_EWALD_FORCE_TABLE);
						double EWALD_omp_end_time = omp_get_wtime(); //Timing
						printf("Ewald lookup table calculation finished. Wall-clock time = %fs.\n", EWALD_omp_end_time-EWALD_omp_start_time);
						printf("Ewald lookup table calculated.\nSaving into %s\n", ewaldfilepath);
						save_t3_ewald_lookup_table(ewaldfilepath, N_EWALD_FORCE_GRID, T3_EWALD_FORCE_TABLE);
					}
					else
					{
						printf("Ewald lookup table file (%s) found.\nLoading lookup table from file...\n", ewaldfilepath);
						double EWALD_omp_start_time = omp_get_wtime(); //Timing
						if(load_t3_ewald_lookup_table(ewaldfilepath, &N_EWALD_FORCE_GRID, &T3_EWALD_FORCE_TABLE)!=0)
						{
							fprintf(stderr, "Error: Failed to load the Ewald lookup table from file %s\nAborting.\n", ewaldfilepath);
							return(-2);
						}
						printf("Ewald lookup table loaded from file %s with grid size %i.\n", ewaldfilepath, N_EWALD_FORCE_GRID);
						double EWALD_omp_end_time = omp_get_wtime(); //Timing
						printf("Ewald lookup table loading finished. Wall-clock time = %fs.\n", EWALD_omp_end_time-EWALD_omp_start_time);
					}
				#else
					printf("HDF5 support not compiled in.\nCalculating new Ewald lookup table...\n");
					double EWALD_omp_start_time = omp_get_wtime(); //Timing
					number_of_ewald_real_space_indexes = ewald_space(rel_cut+1,ewald_real_space_indexes); //real space indexes
					number_of_ewald_reciprocal_space_indexes = ewald_space(rec_cut+2,ewald_reciprocal_space_indexes); //reciprocal space indexes
					calculate_t3_ewald_lookup_table(N_EWALD_FORCE_GRID, L, EWALD_alpha, number_of_ewald_real_space_indexes, number_of_ewald_reciprocal_space_indexes, ewald_real_space_indexes, ewald_reciprocal_space_indexes, rel_cut, rec_cut, T3_EWALD_FORCE_TABLE);
					double EWALD_omp_end_time = omp_get_wtime(); //Timing
					printf("Ewald lookup table calculation finished. Wall-clock time = %fs.\n", EWALD_omp_end_time-EWALD_omp_start_time);
				#endif
				#ifndef USE_CUDA
					printf("Ewald lookup table interpolation order: %i\n\n", EWALD_INTERPOLATION_ORDER);
				#else
					printf("Ewald lookup table interpolation order is always 4 on GPUs in T^3 topological manifolds.\n\n");
				#endif
				fflush(stdout);
			}
			//Bcasting the N_EWALD_FORCE_GRID variable to all MPI threads
			MPI_Bcast(&N_EWALD_FORCE_GRID, 1, MPI_INT, 0, MPI_COMM_WORLD);
			if(rank != 0)
			{
				//Allocating memory for the ewald lookup table in all other MPI threads
				T3_EWALD_FORCE_TABLE = (REAL*)malloc((size_t)N_EWALD_FORCE_GRID*N_EWALD_FORCE_GRID*N_EWALD_FORCE_GRID*3*sizeof(REAL));
			}
			//bcasting the ewald lookup table to all MPI threads
			#ifdef USE_SINGLE_PRECISION
				MPI_Bcast(T3_EWALD_FORCE_TABLE, N_EWALD_FORCE_GRID*N_EWALD_FORCE_GRID*N_EWALD_FORCE_GRID*3, MPI_FLOAT, 0, MPI_COMM_WORLD);
			#else
				MPI_Bcast(T3_EWALD_FORCE_TABLE, N_EWALD_FORCE_GRID*N_EWALD_FORCE_GRID*N_EWALD_FORCE_GRID*3, MPI_DOUBLE, 0, MPI_COMM_WORLD);
			#endif
			fflush(stdout);
		}
		else
		{
			if(rank == 0)
				printf("Quasi-periodic boundary conditions. (No Ewald force correction)\n");
			//To avoid errors, we set N_EWALD_FORCE_GRID to 1 and allocate a dummy ewald table with 0 forces
			N_EWALD_FORCE_GRID = 1;
			//Allocating memory for the dummy ewald lookup table in all MPI threads
			if(!(T3_EWALD_FORCE_TABLE = (REAL*)malloc((size_t)N_EWALD_FORCE_GRID*N_EWALD_FORCE_GRID*N_EWALD_FORCE_GRID*3*sizeof(REAL))))
			{
				fprintf(stderr, "MPI task %i: failed to allocate memory for The Ewald lookup table.\n", rank);
				exit(-2);
			}
			T3_EWALD_FORCE_TABLE[0]=0.0;
			T3_EWALD_FORCE_TABLE[1]=0.0;
			T3_EWALD_FORCE_TABLE[2]=0.0;
		}
	#elif defined(PERIODIC_Z)
		if(IS_PERIODIC < 1)
		{
			if(rank == 0)
				fprintf(stderr, "Error: Bad boundary conditions were set in the paramfile!\nThis executable is able to run semi-periodic and periodic simulations in z direction only.\nExiting.\n");
			fflush(stdout);
			return (-2);
		}
		#if defined(USE_BH) && defined(PERIODIC_Z_NOLOOKUP)
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
					fflush(stdout);
				}
			}
		#endif
		if(IS_PERIODIC> 1)
		{
			#if !defined(PERIODIC_Z_NOLOOKUP)
				//Building the periodic lookup table.
				if(rank == 0)
				{
					//rank 0 MPI thread is calculating the ewald lookup table
					char EwaldTableFile[] = "S1R2_Ewald_table_lowres.hdf5";
					#if !defined(PERIODIC_Z_RSPACELOOKUP)
					//variables only used in periodic ewald force calculation
					double EWALD_alpha; //Ewald alpha parameter
					int rel_cut, rec_cut; //real and reciprocal space cutoffs
					if(IS_PERIODIC==2)
					{
						printf("S^1 x R^2 Ewald force calculation is on. (Ewald cut is 5*L in real and 12 in reciprocal space)\nCalculating Ewald lookup tables...\n\n");
						rel_cut = 5;
						rec_cut = 12;
						EWALD_alpha = 0.71716533601505/L; //Optimal alpha based on numerical tests, if rel_cut=5L and rec_cut=12 (relative force error: 1.1e-7 [ideal for 32bit precision])
						Nz_EWALD_FORCE_GRID = 128; //size of the ewald force lookup table in the z direction
						Nrho_EWALD_FORCE_GRID = (int) floor( ( (REAL)Nz_EWALD_FORCE_GRID ) * EWALD_LOOKUP_TABLE_RADIAL_EXTENT_FACTOR * Rsim / L ); //size of the ewald force lookup table in the radial direction (EWALD_LOOKUP_TABLE_RADIAL_EXTENT_FACTORx Rsim);
					}
					else if(IS_PERIODIC==3)
					{
						printf("Medium precision S^1 x R^2 Ewald force calculation is on. (Ewald cut is 6*L in real and 13 in reciprocal space)\nCalculating Ewald lookup tables...\n\n");
						rel_cut = 6;
						rec_cut = 13;
						EWALD_alpha = 0.6635543550051043/L; //Optimal alpha based on numerical tests, if rel_cut=6L and rec_cut=13 (relative force error: 4.4e-9)
						Nz_EWALD_FORCE_GRID = 256; //size of the ewald force lookup table in the z direction
						Nrho_EWALD_FORCE_GRID = (int) floor( ( (REAL)Nz_EWALD_FORCE_GRID ) * EWALD_LOOKUP_TABLE_RADIAL_EXTENT_FACTOR * Rsim / L ); //size of the ewald force lookup table in the radial direction (EWALD_LOOKUP_TABLE_RADIAL_EXTENT_FACTORx Rsim);
						strcpy(EwaldTableFile, "S1R2_Ewald_table_medres.hdf5");
					}
					else
					{
						rel_cut = IS_PERIODIC+3;
						rec_cut = IS_PERIODIC+10;
						printf("High precision S^1 x R^2 Ewald force calculation is on. (Ewald cut is %i*L in real and %i in reciprocal space)\nCalculating Ewald lookup tables...\n\n", rel_cut, rec_cut);
						if(IS_PERIODIC==4)
						{
							EWALD_alpha = 0.6205537827349956/L; //Optimal alpha based on numerical tests, if rel_cut==7L and rec_cut==14 (relative force error: 1.8e-10)
						}
						else if(IS_PERIODIC==5)
						{
							EWALD_alpha = 0.5851341941700122/L; //Optimal alpha based on numerical tests, if rel_cut==8L and rec_cut==15 (relative force error: 1.1e-11)
						}
						else
						{
							EWALD_alpha = 0.5546606614185521/L; //Optimal alpha based on numerical tests, if rel_cut==9L and rec_cut==16 (relative force error: 4.2e-12)
							if(IS_PERIODIC>6)
							{
								printf("Warning: Using more than 6 periodic images in real and 17 in reciprocal space in Ewald summation.\nThe optimal Ewald parameters may have not be accurately determined.\nConsider using less periodic images for better performance and accuracy.\n\n");
							}
						}
						Nz_EWALD_FORCE_GRID = 512; //size of the ewald force lookup table in the z direction
						Nrho_EWALD_FORCE_GRID = (int) floor( ( (REAL)Nz_EWALD_FORCE_GRID ) * EWALD_LOOKUP_TABLE_RADIAL_EXTENT_FACTOR * Rsim / L ); //size of the ewald force lookup table in the radial direction (EWALD_LOOKUP_TABLE_RADIAL_EXTENT_FACTORx Rsim);
						strcpy(EwaldTableFile, "S1R2_Ewald_table_higres.hdf5");
					}
					#else
					//If TORNBERG2015 method is not defined, we use direct summation to build the periodic lookup table
					int rel_cut;
					if(IS_PERIODIC==2)
					{
						printf("Calculating S^1 x R^2 periodic lookup table... (7.2*10^3*L in real space)\n\n");
						rel_cut = 7200;
						Nz_EWALD_FORCE_GRID = 128; //size of the periodic force lookup table in the z direction
						Nrho_EWALD_FORCE_GRID = (int) floor( ( (REAL)Nz_EWALD_FORCE_GRID ) * EWALD_LOOKUP_TABLE_RADIAL_EXTENT_FACTOR * Rsim / L ); //size of the periodic force lookup table in the radial direction (EWALD_LOOKUP_TABLE_RADIAL_EXTENT_FACTORx Rsim);
					}
					else if(IS_PERIODIC==3)
					{
						printf("Calculating medium precision S^1 x R^2 periodic lookup table... (3.6*10^4*L in real space)\n\n");
						rel_cut = 36000;
						Nz_EWALD_FORCE_GRID = 256; //size of the periodic force lookup table in the z direction
						Nrho_EWALD_FORCE_GRID = (int) floor( ( (REAL)Nz_EWALD_FORCE_GRID ) * EWALD_LOOKUP_TABLE_RADIAL_EXTENT_FACTOR * Rsim / L ); //size of the periodic force lookup table in the radial direction (EWALD_LOOKUP_TABLE_RADIAL_EXTENT_FACTORx Rsim);
						strcpy(EwaldTableFile, "S1R2_Ewald_table_medres.hdf5");
					}
					else
					{
						printf("Calculating high precision S^1 x R^2 periodic lookup table... (1.8*10^5*L in real space)\n\n");
						rel_cut = 180000;
						Nz_EWALD_FORCE_GRID = 512; //size of the periodic force lookup table in the z direction
						Nrho_EWALD_FORCE_GRID = (int) floor( ( (REAL)Nz_EWALD_FORCE_GRID ) * EWALD_LOOKUP_TABLE_RADIAL_EXTENT_FACTOR * Rsim / L ); //size of the periodic force lookup table in the radial direction (EWALD_LOOKUP_TABLE_RADIAL_EXTENT_FACTORx Rsim);
						strcpy(EwaldTableFile, "S1R2_Ewald_table_higres.hdf5");
					}
					#endif
					//Allocating memory for the ewald lookup table in the rank 0 MPI thread
					printf("MPI task %i: Allocating memory for the Ewald lookup table with %i*%i=%i grid points...\n", rank, Nrho_EWALD_FORCE_GRID, Nz_EWALD_FORCE_GRID, Nrho_EWALD_FORCE_GRID*Nz_EWALD_FORCE_GRID);
					if(!(S1R2_EWALD_FORCE_TABLE = (REAL*)malloc((size_t)Nz_EWALD_FORCE_GRID*Nrho_EWALD_FORCE_GRID*2*sizeof(REAL))))
					{
						fprintf(stderr, "MPI task %i: failed to allocate memory for The Ewald lookup table.\n", rank);
						exit(-2);
					}
					#ifdef HAVE_HDF5
						//if we have HDF5, we save/load the Ewald table in HDF5 format				
						if(snprintf(ewaldfilepath, sizeof(ewaldfilepath), "%s%s", OUT_DIR, EwaldTableFile) < 0)
						{
							fprintf(stderr, "Error: The name of the ewald table got truncated.\nAborting.\n");
							abort();
						}
						if(file_exist(ewaldfilepath) == 0)
						{
							printf("Ewald lookup table file (%s) not found.\nCalculating new lookup table...\n", ewaldfilepath);
							double EWALD_omp_start_time = omp_get_wtime(); //Timing
							#if !defined(PERIODIC_Z_RSPACELOOKUP)
							calculate_S1R2ewald_correction_table(Nrho_EWALD_FORCE_GRID, Nz_EWALD_FORCE_GRID, EWALD_LOOKUP_TABLE_RADIAL_EXTENT_FACTOR*Rsim, L, EWALD_alpha, rel_cut, rec_cut, S1R2_EWALD_FORCE_TABLE);
							#else
							calculate_S1R2ewald_correction_table(Nrho_EWALD_FORCE_GRID, Nz_EWALD_FORCE_GRID, EWALD_LOOKUP_TABLE_RADIAL_EXTENT_FACTOR*Rsim, L, 0.0, rel_cut, 0, S1R2_EWALD_FORCE_TABLE);
							#endif
							double EWALD_omp_end_time = omp_get_wtime(); //Timing
							printf("Ewald lookup table calculation finished. Wall-clock time = %fs.\n", EWALD_omp_end_time-EWALD_omp_start_time);
							printf("Ewald lookup table calculated.\nSaving into %s\n", ewaldfilepath);
							save_s1r2_ewald_lookup_table(ewaldfilepath, Nrho_EWALD_FORCE_GRID, Nz_EWALD_FORCE_GRID, S1R2_EWALD_FORCE_TABLE);
						}
						else
						{
							printf("Ewald lookup table file (%s) found.\nLoading lookup table from file...\n", ewaldfilepath);
							double EWALD_omp_start_time = omp_get_wtime(); //Timing
							int Nrho_EWALD_FORCE_GRID_file, Nz_EWALD_FORCE_GRID_file;
							//loading the table
							if(load_s1r2_ewald_lookup_table(ewaldfilepath, &Nrho_EWALD_FORCE_GRID_file, &Nz_EWALD_FORCE_GRID_file, &S1R2_EWALD_FORCE_TABLE)!=0)
							{
								fprintf(stderr, "Error: Failed to load the Ewald lookup table from file %s\nAborting.\n", ewaldfilepath);
								return(-2);
							}
							printf("Ewald lookup table loaded from file %s with grid size %i x %i.\n", ewaldfilepath, Nrho_EWALD_FORCE_GRID_file, Nz_EWALD_FORCE_GRID_file);
							if(Nrho_EWALD_FORCE_GRID_file != Nrho_EWALD_FORCE_GRID || Nz_EWALD_FORCE_GRID_file != Nz_EWALD_FORCE_GRID)
							{
								fprintf(stderr, "Error: The loaded Ewald lookup table size (%i x %i) does not match the expected size (%i x %i).\nPlease delete the old lookup table file and run the simulation again.\nAborting.\n", Nrho_EWALD_FORCE_GRID_file, Nz_EWALD_FORCE_GRID_file, Nrho_EWALD_FORCE_GRID, Nz_EWALD_FORCE_GRID);
								return(-2);
							}
							double EWALD_omp_end_time = omp_get_wtime(); //Timing
							printf("Ewald lookup table loading finished. Wall-clock time = %fs.\n", EWALD_omp_end_time-EWALD_omp_start_time);
						}
					#else
						printf("HDF5 support not compiled in.\nCalculating new Ewald lookup table...\n");
						double EWALD_omp_start_time = omp_get_wtime(); //Timing
						#if !defined(PERIODIC_Z_RSPACELOOKUP)
						calculate_S1R2ewald_correction_table(Nrho_EWALD_FORCE_GRID, Nz_EWALD_FORCE_GRID, EWALD_LOOKUP_TABLE_RADIAL_EXTENT_FACTOR*Rsim, L, EWALD_alpha, rel_cut, rec_cut, S1R2_EWALD_FORCE_TABLE);
						#else
						calculate_S1R2ewald_correction_table(Nrho_EWALD_FORCE_GRID, Nz_EWALD_FORCE_GRID, EWALD_LOOKUP_TABLE_RADIAL_EXTENT_FACTOR*Rsim, L, 0.0, rel_cut, 0, S1R2_EWALD_FORCE_TABLE);
						#endif
						double EWALD_omp_end_time = omp_get_wtime(); //Timing
						printf("Ewald lookup table calculation finished. Wall-clock time = %fs.\n", EWALD_omp_end_time-EWALD_omp_start_time);
					#endif
					printf("Ewald lookup table interpolation order: %i\n\n", EWALD_INTERPOLATION_ORDER);
				}
				//Bcasting the Nz_EWALD_FORCE_GRID and Nrho_EWALD_FORCE_GRID variables to all MPI threads
				MPI_Bcast(&Nz_EWALD_FORCE_GRID, 1, MPI_INT, 0, MPI_COMM_WORLD);
				MPI_Bcast(&Nrho_EWALD_FORCE_GRID, 1, MPI_INT, 0, MPI_COMM_WORLD);
				if(rank != 0)
				{
					//Allocating memory for the ewald lookup table in all other MPI threads
					S1R2_EWALD_FORCE_TABLE = (REAL*)malloc((size_t)Nz_EWALD_FORCE_GRID*Nrho_EWALD_FORCE_GRID*2*sizeof(REAL));
				}
				//bcasting the ewald lookup table to all MPI threads
				#ifdef USE_SINGLE_PRECISION
					MPI_Bcast(S1R2_EWALD_FORCE_TABLE, Nz_EWALD_FORCE_GRID*Nrho_EWALD_FORCE_GRID*2, MPI_FLOAT, 0, MPI_COMM_WORLD);
				#else
					MPI_Bcast(S1R2_EWALD_FORCE_TABLE, Nz_EWALD_FORCE_GRID*Nrho_EWALD_FORCE_GRID*2, MPI_DOUBLE, 0, MPI_COMM_WORLD);
				#endif
			#endif
			fflush(stdout);
		}
		else
		{
			if(rank == 0)
			{
				printf("Warning: Quasi-periodic boundary conditions only in the z direction.\n         Using only one periodic image in this geometry can easily cause inaccurate forces.\n\n");
			}
			#if !defined(PERIODIC_Z_NOLOOKUP)
				//To avoid errors, we set Nz_EWALD_FORCE_GRID and Nrho_EWALD_FORCE_GRID to 1 and allocate a dummy ewald table with 0 forces
				Nz_EWALD_FORCE_GRID = 1;
				Nrho_EWALD_FORCE_GRID = 1;
				//Allocating memory for the dummy ewald lookup table in all MPI threads
				if(!(S1R2_EWALD_FORCE_TABLE = (REAL*)malloc((size_t)Nz_EWALD_FORCE_GRID*Nrho_EWALD_FORCE_GRID*2*sizeof(REAL))))
				{
					fprintf(stderr, "MPI task %i: failed to allocate memory for The Ewald lookup table.\n", rank);
					exit(-2);
				}
				S1R2_EWALD_FORCE_TABLE[0]=0.0;
				S1R2_EWALD_FORCE_TABLE[1]=0.0;
			#endif
		}
	#elif defined(POINCARE_DODECAHEDRAL)
		// Initialise the binary icosahedral group I* (120 unit quaternions).
		// pds_init() is idempotent; must be called before any pds_wrap / pds_in_domain.
		pds_init();
		if(IS_PERIODIC < 1)
		{
			if(rank == 0)
				fprintf(stderr, "Error: Bad boundary conditions were set in the paramfile!\nThis executable runs Poincare Dodecahedral Space (S^3/I*) simulations only.\nExiting.\n");
			fflush(stdout);
			return (-2);
		}
		if(IS_PERIODIC >= 2)
		{
			//Ewald correction table setup
			if(rank == 0)
			{
				if(IS_PERIODIC == 2)
				{
					printf("PDS (S^3/I*) Ewald force calculation is on.\n");
					N_PDS_EWALD_GRID = 1024;
				}
				else if(IS_PERIODIC == 3)
				{
					printf("Medium precision PDS (S^3/I*) Ewald force calculation is on.\n");
					N_PDS_EWALD_GRID = 4096;
				}
				else
				{
					printf("High precision PDS (S^3/I*) Ewald force calculation is on.\n");
					N_PDS_EWALD_GRID = 16384;
				}
				//Allocating memory for the Ewald lookup table
				printf("MPI task %i: Allocating memory for the PDS Ewald lookup table with %i grid points...\n", rank, N_PDS_EWALD_GRID);
				if(!(PDS_EWALD_FORCE_TABLE = (REAL*)malloc((size_t)N_PDS_EWALD_GRID*sizeof(REAL))))
				{
					fprintf(stderr, "MPI task %i: failed to allocate memory for the PDS Ewald lookup table.\n", rank);
					exit(-2);
				}
				#ifdef HAVE_HDF5
					char PDS_EwaldTableFile[] = "PDS_Ewald_table.hdf5";
					if(snprintf(pds_ewaldfilepath, sizeof(pds_ewaldfilepath), "%s%s", OUT_DIR, PDS_EwaldTableFile) < 0)
					{
						fprintf(stderr, "Error: The name of the PDS Ewald table got truncated.\nAborting.\n");
						abort();
					}
					if(file_exist(pds_ewaldfilepath) == 0)
					{
						printf("PDS Ewald lookup table file (%s) not found.\nCalculating new lookup table...\n", pds_ewaldfilepath);
						double EWALD_omp_start_time = omp_get_wtime();
						calculate_pds_ewald_lookup_table(N_PDS_EWALD_GRID, (double)PDS_R_CURV, PDS_EWALD_FORCE_TABLE);
						double EWALD_omp_end_time = omp_get_wtime();
						printf("PDS Ewald lookup table calculation finished. Wall-clock time = %fs.\n", EWALD_omp_end_time-EWALD_omp_start_time);
						printf("Saving PDS Ewald lookup table into %s\n", pds_ewaldfilepath);
						save_pds_ewald_lookup_table(pds_ewaldfilepath, N_PDS_EWALD_GRID, (double)PDS_R_CURV, PDS_EWALD_FORCE_TABLE);
					}
					else
					{
						printf("PDS Ewald lookup table file (%s) found.\nLoading lookup table from file...\n", pds_ewaldfilepath);
						double EWALD_omp_start_time = omp_get_wtime();
						double R_curv_from_file;
						if(load_pds_ewald_lookup_table(pds_ewaldfilepath, &N_PDS_EWALD_GRID, &R_curv_from_file, &PDS_EWALD_FORCE_TABLE) != 0)
						{
							fprintf(stderr, "Error: Failed to load the PDS Ewald lookup table from file %s\nAborting.\n", pds_ewaldfilepath);
							return(-2);
						}
						if(fabs(R_curv_from_file - (double)PDS_R_CURV) > 1e-6 * (double)PDS_R_CURV)
						{
							fprintf(stderr, "Error: PDS curvature radius mismatch: file has R_curv=%.6g Mpc, parameter file has R_curv=%.6g Mpc.\nPlease delete the old lookup table and run again.\nAborting.\n", R_curv_from_file, (double)PDS_R_CURV);
							return(-2);
						}
						double EWALD_omp_end_time = omp_get_wtime();
						printf("PDS Ewald lookup table loaded from file %s with %i grid points (R_curv=%.4g Mpc).\n", pds_ewaldfilepath, N_PDS_EWALD_GRID, R_curv_from_file);
						printf("PDS Ewald lookup table loading finished. Wall-clock time = %fs.\n", EWALD_omp_end_time-EWALD_omp_start_time);
					}
				#else
					printf("HDF5 support not compiled in.\nCalculating new PDS Ewald lookup table...\n");
					double EWALD_omp_start_time = omp_get_wtime();
					calculate_pds_ewald_lookup_table(N_PDS_EWALD_GRID, (double)PDS_R_CURV, PDS_EWALD_FORCE_TABLE);
					double EWALD_omp_end_time = omp_get_wtime();
					printf("PDS Ewald lookup table calculation finished. Wall-clock time = %fs.\n", EWALD_omp_end_time-EWALD_omp_start_time);
				#endif
				fflush(stdout);
			}
			//Bcast the grid size and R_curv to all MPI threads
			MPI_Bcast(&N_PDS_EWALD_GRID, 1, MPI_INT, 0, MPI_COMM_WORLD);
			MPI_Bcast(&PDS_R_CURV, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
			if(rank != 0)
			{
				PDS_EWALD_FORCE_TABLE = (REAL*)malloc((size_t)N_PDS_EWALD_GRID*sizeof(REAL));
			}
			#ifdef USE_SINGLE_PRECISION
				MPI_Bcast(PDS_EWALD_FORCE_TABLE, N_PDS_EWALD_GRID, MPI_FLOAT, 0, MPI_COMM_WORLD);
			#else
				MPI_Bcast(PDS_EWALD_FORCE_TABLE, N_PDS_EWALD_GRID, MPI_DOUBLE, 0, MPI_COMM_WORLD);
			#endif
			fflush(stdout);
		}
		else
		{
			if(rank == 0)
				printf("PDS (S^3/I*) nearest-image-only mode (no Ewald correction).\n");
			N_PDS_EWALD_GRID = 1;
			if(!(PDS_EWALD_FORCE_TABLE = (REAL*)malloc(sizeof(REAL))))
			{
				fprintf(stderr, "MPI task %i: failed to allocate memory for dummy PDS Ewald table.\n", rank);
				exit(-2);
			}
			PDS_EWALD_FORCE_TABLE[0] = 0.0;
		}
	#else
		if(IS_PERIODIC  != 0)
		{
			if(rank == 0)
				fprintf(stderr, "Error: Bad boundary conditions were set in the paramfile!\nThis executable is able to run non-periodic (R^3 spherical) simulations only.\nExiting.\n");
			fflush(stdout);
			return (-2);
		}
	#endif
	if(OUTPUT_TIME_VARIABLE != 0 && OUTPUT_TIME_VARIABLE !=1)
	{
		if(rank == 0)
			fprintf(stderr, "Error: bad OUTPUT time variable %i!\nExiting.\n", OUTPUT_TIME_VARIABLE);
		fflush(stdout);
		return (-2);
	}
	if(OUTPUT_TIME_VARIABLE == 1 && COSMOLOGY != 1)
	{
		if(rank == 0)
			fprintf(stderr, "Error: you can not use redshift output format in non-cosmological simulations. \nExiting.\n");
		fflush(stdout);
		return (-2);
	}
	if(H0 == 0.0 && COSMOLOGY == 1)
	{
    #if !defined(COSMOPARAM) || COSMOPARAM>=0
		if(rank == 0)
			fprintf(stderr, "Error: Hubble constant is set to zero in a cosmological simulation. This must be a mistake.\nExiting.\n");
		fflush(stdout);
		return (-2);
		#else
		if(rank == 0)
			printf("Warning: Hubble constant is set to zero in a cosmological simulation. \nSince the expansion history read from an external file, this is not necessarily an error.\nPlease make sure that the Hubble constant was set correctly during the initial condition generation.\n\n");
		#endif
	}
	if(rank == 0)
	{
		printf("Determining the output times...\n");
		fflush(stdout);
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
	fflush(stdout);
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
			ID_MPI_min = 0;
			ID_MPI_max = (N%numtasks) + (rank+1)*(N/numtasks)-1;
			mpi_particle_range[0][0] = ID_MPI_min;
			mpi_particle_range[0][1] = ID_MPI_max;
			mpi_particle_range[0][2] = N_mpi_thread;
			for(i=1; i<numtasks; i++)
			{
				mpi_particle_range[i][0] = (N%numtasks) + (i)*(N/numtasks);
				mpi_particle_range[i][1] = (N%numtasks) + (i+1)*(N/numtasks)-1;
				mpi_particle_range[i][2] = N/numtasks;
			}
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
		}
		//Bcasting the number of particles
		MPI_Bcast(&N,1,MPI_INT,0,MPI_COMM_WORLD);
		//Bcasting the particle ranges
		BCAST_MPI_particle_ranges();
		if(rank != 0)
		{
			//Allocating memory for the slave processes

			//Allocating memory for the coordinates
			if(!(x = (REAL*)malloc(3*N*sizeof(REAL))))
			{
				fprintf(stderr, "MPI task %i: failed to allocate memory for x.\n", rank);
				exit(-2);
			}
			//Allocating memory for the forces. There is no need to allocate for N forces. N_mpi_thread should be enough. Note that the forces will be re-allocated later.
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
		#if defined(PERIODIC_Z) && defined(PERIODIC_Z_NOLOOKUP)
			printf("Calculating the lookup table for the radial force calculation...\n");
			if(CYLINDRICAL_LOOKUP_TABLE_CALCULATED == false)
			{
				RADIAL_FORCE_TABLE = (REAL*)malloc(RADIAL_FORCE_TABLE_SIZE*sizeof(REAL)); //Allocating the lookup table
				// Calculating the radial force in a finite cylinder
				mass_in_unit_sphere = (REAL) (2.0*pi*rho_crit*Omega_m); // in cylindrically symmetric simulations, the radial force is proportional to the mass in a unit cylinder
				ewald_max = IS_PERIODIC+1;
				ewald_cut = ((REAL) ewald_max)-0.4; //cutoff radius for the direct real-space periodic summation
				if(IS_PERIODIC==1)
				{
					get_cylindrical_force_table(RADIAL_FORCE_TABLE, Rsim,0.5*L,RADIAL_FORCE_TABLE_SIZE,RADIAL_FORCE_ACCURACY);
					if(rank==0)
						printf("Cylindrical force table calculated.\nMagnitude of the gravitational pull correction of the finite Ewald cylinder at r = %.1f Mpc is |F(Lz=0.5Lsim)|/|F(Lz=inf)| = %.7f \n\n", Rsim, RADIAL_FORCE_TABLE[RADIAL_FORCE_TABLE_SIZE-1]);
				}
				else
				{
					get_cylindrical_force_table(RADIAL_FORCE_TABLE, Rsim, L*ewald_cut,RADIAL_FORCE_TABLE_SIZE,RADIAL_FORCE_ACCURACY);
					if(rank==0)
						printf("Cylindrical force table calculated.\nMagnitude of the gravitational pull correction of the finite Ewald cylinder at r = %.1f Mpc is |F(Lz=%.1fLsim)|/|F(Lz=inf)| = %.7f \n\n", Rsim, 2*ewald_cut, RADIAL_FORCE_TABLE[RADIAL_FORCE_TABLE_SIZE-1]);
				}
				CYLINDRICAL_LOOKUP_TABLE_CALCULATED = true;
				//saving the lookup table to a file
				char CYLINDRICAL_LOOKUP_TABLE_FILENAME[] = "radial_force_table_chang.dat";
				char CYLINDRICAL_LOOKUP_TABLE_HEADER[128];
				sprintf(CYLINDRICAL_LOOKUP_TABLE_HEADER, "# Radial force compensation table based on Chang (1988)\n# r[Mpc]    F_r(r,L=%.1f*Lsim)/F_r(L=inf)", 2*ewald_cut);
				if(snprintf(CYLINDRICAL_LOOKUP_TABLE_PATH, sizeof(CYLINDRICAL_LOOKUP_TABLE_PATH), "%s%s", OUT_DIR, CYLINDRICAL_LOOKUP_TABLE_FILENAME) < 0)
				{
					fprintf(stderr, "Error: The name of the ewald table got truncated.\nAborting.\n");
					abort();
				}
				printf("Saving the radial BH force correction table into %s\n\n", CYLINDRICAL_LOOKUP_TABLE_PATH);
				save_function_to_ascii_table(CYLINDRICAL_LOOKUP_TABLE_PATH, Rsim/(REAL)RADIAL_FORCE_TABLE_SIZE, Rsim, Rsim/(REAL)RADIAL_FORCE_TABLE_SIZE, RADIAL_FORCE_TABLE, RADIAL_FORCE_TABLE_SIZE, CYLINDRICAL_LOOKUP_TABLE_HEADER);
			}
		#elif defined(PERIODIC_Z)
			// in cylindrically symmetric (S^1xR^2) simulations, the radial force is proportional to the mass in a unit cylinder
			mass_in_unit_sphere = (REAL) (2.0*pi*rho_crit*Omega_m);
			if(IS_PERIODIC==1)
			{
				if(CYLINDRICAL_LOOKUP_TABLE_CALCULATED == false)
				{
					RADIAL_FORCE_TABLE = (REAL*)malloc(RADIAL_FORCE_TABLE_SIZE*sizeof(REAL)); //Allocating the lookup table
					printf("Calculating the lookup table for the radial force calculation...\n");
					get_cylindrical_force_table(RADIAL_FORCE_TABLE, Rsim,0.5*L,RADIAL_FORCE_TABLE_SIZE,RADIAL_FORCE_ACCURACY);
					if(rank==0)
						printf("Cylindrical force table calculated.\nMagnitude of the gravitational pull correction of the finite Ewald cylinder at r = %.1f Mpc is |F(Lz=0.5Lsim)|/|F(Lz=inf)| = %.7f \n\n", Rsim, RADIAL_FORCE_TABLE[RADIAL_FORCE_TABLE_SIZE-1]);
					CYLINDRICAL_LOOKUP_TABLE_CALCULATED = true;
				}
			}
			else
			{
				//setting up the lookup table with zero elements
				if(CYLINDRICAL_LOOKUP_TABLE_CALCULATED == false)
				{
					RADIAL_FORCE_TABLE = (REAL*)malloc(RADIAL_FORCE_TABLE_SIZE*sizeof(REAL)); //Allocating the lookup table
					set_REAL_array_to_zero(RADIAL_FORCE_TABLE, RADIAL_FORCE_TABLE_SIZE);
					CYLINDRICAL_LOOKUP_TABLE_CALCULATED = true;
				}
			}
		#else
			// in spherically symmetric (R^3) simulations, the radial force is proportional to the mass in a unit sphere
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
		if(rank == 0 && RADIAL_BH_FORCE_TABLE_ITERATION > 1)
		{
			printf("Randomised BH force calculation is on. The BH force table will be calculated for %d iterations and averaged to reduce the noise.\n\n", RADIAL_BH_FORCE_TABLE_ITERATION);
		}
		for(int i_iter=0; i_iter<RADIAL_BH_FORCE_TABLE_ITERATION; i_iter++)
		{
			double iter_start_time, iter_end_time; //Timing variables for the BH force table iteration
			iter_start_time = omp_get_wtime(); //Timing
			if(rank == 0 && RADIAL_BH_FORCE_TABLE_ITERATION > 1)
			{
				printf("BH radial force correction iteration %d/%d\n------------------------------------------\n", i_iter+1, RADIAL_BH_FORCE_TABLE_ITERATION);
				fflush(stdout);
			}
			
			//Bcasting the particle ranges
			BCAST_MPI_particle_ranges();
			if(rank!=0)
			{
				//Re-allocating the force array of all slave threads
				free(F);
				if(!(F = (REAL*)malloc((3*N_mpi_thread)*sizeof(REAL))))
				{
					fprintf(stderr, "MPI task %i: failed to allocate memory for F.\n", rank);
					exit(-2);
				}
			}

			double force_calc_start_time = omp_get_wtime(); //Timing the force calculation
			//Threads calculate the forces on their own particles
			#if defined(PERIODIC_Z)
				forces_periodic_z(x, F, ID_MPI_min, ID_MPI_max);
			#else
				forces(x, F, ID_MPI_min, ID_MPI_max);
			#endif

			double force_calc_end_time = omp_get_wtime();
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
						//the F_buffer should be re-allocated based on the mpi_particle_range[i][2] value.
						if(!(F_buffer = (REAL*)malloc(3*(mpi_particle_range[i][2])*sizeof(REAL))))
						{
							fprintf(stderr, "MPI task %i: failed to allocate memory for F_buffer.\n", rank);
							exit(-2);
						}
						BUFFER_start_ID = mpi_particle_range[i][0];
						#ifdef USE_SINGLE_PRECISION
							MPI_Recv(F_buffer, 3*mpi_particle_range[i][2], MPI_FLOAT, i, i, MPI_COMM_WORLD, &Stat);
						#else
							MPI_Recv(F_buffer, 3*mpi_particle_range[i][2], MPI_DOUBLE, i, i, MPI_COMM_WORLD, &Stat);
						#endif
						for(j=0; j<mpi_particle_range[i][2]; j++)
						{
							F[3*(BUFFER_start_ID+j)] = F_buffer[3*j];
							F[3*(BUFFER_start_ID+j)+1] = F_buffer[3*j+1];
							F[3*(BUFFER_start_ID+j)+2] = F_buffer[3*j+2];
						}
						free(F_buffer);
					}
				}
			}
			if(rank!=0)
			{
				//sending the time spent in the force calculation to the rank=0 thread (always in double precison)
				double force_calc_time = force_calc_end_time - force_calc_start_time;
				MPI_Send(&force_calc_time, 1, MPI_DOUBLE, 0, rank, MPI_COMM_WORLD);
			}
			else
			{
				double force_calc_time = force_calc_end_time - force_calc_start_time;
				mpi_time_array[0] = force_calc_time;
				//receiving the time spent in the force calculation from the slave threads, and storing it in the mpi_time_array
				if(numtasks > 1)
				{
					for(i=1; i<numtasks;i++)
					{
						MPI_Recv(&force_calc_time, 1, MPI_DOUBLE, i, i, MPI_COMM_WORLD, &Stat);
						mpi_time_array[i] = force_calc_time;
						//storing the longest time spent in the force_calc_time variable, to calculate the workload-balance later
						if(i==1 || force_calc_time > force_calc_time)
							force_calc_time = force_calc_time;
					}
					//Adding the time spent in the force calculation to the mpi_time_array, to calculate the total time spent in the force calculation later
					force_calc_time = 0.0;
					for(i=0; i<numtasks; i++)
					{
							force_calc_time += mpi_time_array[i];
					}
					//Printing the time spent in the force calculation and workload-balance for each MPI thread
					fflush(stdout);
					printf("\nForce calculation time for each MPI thread:\n");
					for(i=0; i<numtasks; i++)
					{
						printf("MPI task %i: %fs, workload balance: %f %%\n", i, mpi_time_array[i], (mpi_time_array[i])/force_calc_time * numtasks * 100.0);
					}
					//Re-calculating the workload of each thread based on the time spent in the force calculation, and re-distributing the particles for the next iteration if the workload balance is too bad (e.g. if one thread takes more than 10% longer than the average time)
					if(numtasks > 1)
					{
						redistribute_workload(mpi_time_array, numtasks, N, mpi_particle_range);
					}
					printf("\n");
				}
			}
			//Bcasting the mpi particle ranges to all threads for the next iteration
			BCAST_MPI_particle_ranges();
			if(rank!=0)
			{
				//Re-allocating the force array for the next iteration
				free(F);
				if(!(F = (REAL*)malloc((3*N_mpi_thread)*sizeof(REAL))))
				{
					fprintf(stderr, "MPI task %i: failed to allocate memory for F.\n", rank);
					exit(-2);
				}
			}
			fflush(stdout);
			//Calculating the radial BH force correction from the force vectors
			if(rank == 0)
			{
				printf("Calculating the radial force correction from the estimated Octree forces...\n");
				get_radial_bh_force_correction_table(RADIAL_BH_FORCE_TABLE, RADIAL_BH_N_TABLE, RADIAL_BH_FORCE_TABLE_SIZE, F, x, N);
				iter_end_time = omp_get_wtime(); //Timing
				printf("Radial BH force correction calculated for iteration %d/%d completed under %.2fs.\n\n", i_iter+1, RADIAL_BH_FORCE_TABLE_ITERATION, iter_end_time-iter_start_time);
				fflush(stdout);
				#ifdef SAVE_ACCELERATIONS
					printf("Saving the initial conditions with the calculated forces as a HDF5 snapshot for acceleration comparison...\n");
					write_hdf5_snapshot(x, v, M, true, F, true);
					printf("...done.\n");
				#endif
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
			char radial_bh_force_table_filename[] = "radial_bh_force_table.dat";
			char RADIAL_BH_FORCE_CORRECTION_HEADER[] = "# Radial BH force correction table\n# r[Mpc]    F_correction[internal units]";
			if(snprintf(RADIAL_BH_FORCE_CORRECTION_PATH, sizeof(RADIAL_BH_FORCE_CORRECTION_PATH), "%s%s", OUT_DIR, radial_bh_force_table_filename) < 0)
			{
				fprintf(stderr, "Error: The name of the ewald table got truncated.\nAborting.\n");
				abort();
			}
			printf("Saving the radial BH force correction table into %s\n\n", RADIAL_BH_FORCE_CORRECTION_PATH);
			save_function_to_ascii_table(RADIAL_BH_FORCE_CORRECTION_PATH, 0.5*Rsim/(REAL)RADIAL_BH_FORCE_TABLE_SIZE, Rsim, Rsim/(REAL)RADIAL_BH_FORCE_TABLE_SIZE, RADIAL_BH_FORCE_TABLE, RADIAL_BH_FORCE_TABLE_SIZE, RADIAL_BH_FORCE_CORRECTION_HEADER);
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
		if(rank != 0)
		{
			//freeing the memory used for the BH force table iteration
			//free(F);
			free(x);
			free(M);
			free(SOFT_LENGTH);
		}
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
		//Calculating the particle ranges for each MPI thread, and the number of particles in each thread 
		//Initially, the particles are distributed evenly among the threads, but this will be re-distributed later based on the workload balance of the threads
		N_mpi_thread = (N/numtasks) + (N%numtasks);
		ID_MPI_min = 0;
		ID_MPI_max = (N%numtasks) + (rank+1)*(N/numtasks)-1;
		mpi_particle_range[0][0] = ID_MPI_min;
		mpi_particle_range[0][1] = ID_MPI_max;
		mpi_particle_range[0][2] = N_mpi_thread;
		for(i=1; i<numtasks; i++)
		{
			mpi_particle_range[i][0] = (N%numtasks) + (i)*(N/numtasks);
			mpi_particle_range[i][1] = (N%numtasks) + (i+1)*(N/numtasks)-1;
			mpi_particle_range[i][2] = N/numtasks;
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
	//Bcasting the particle ranges
	BCAST_MPI_particle_ranges();
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
		//Allocating memory for the forces. There is no need to allocate for N forces. N_mpi_thread should be enough
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
		#ifdef POINCARE_DODECAHEDRAL
		//Allocating memory for the 4D quaternion positions
		if(!(PDS_Q = (REAL*)malloc(4*N*sizeof(REAL))))
		{
			fprintf(stderr, "MPI task %i: failed to allocate memory for PDS_Q.\n", rank);
			exit(-2);
		}
		#endif
	}
	//Bcasting the ICs to the rank!=0 threads
#ifdef USE_SINGLE_PRECISION
	MPI_Bcast(x,3*N,MPI_FLOAT,0,MPI_COMM_WORLD);
  	MPI_Bcast(M,N,MPI_FLOAT,0,MPI_COMM_WORLD);
	#ifdef POINCARE_DODECAHEDRAL
	MPI_Bcast(PDS_Q,4*N,MPI_FLOAT,0,MPI_COMM_WORLD);
	#endif
#else
	MPI_Bcast(x,3*N,MPI_DOUBLE,0,MPI_COMM_WORLD);
	MPI_Bcast(M,N,MPI_DOUBLE,0,MPI_COMM_WORLD);
	#ifdef POINCARE_DODECAHEDRAL
	MPI_Bcast(PDS_Q,4*N,MPI_DOUBLE,0,MPI_COMM_WORLD);
	#endif
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
		#if defined(PERIODIC_Z) && defined(PERIODIC_Z_NOLOOKUP)
			// Calculating the lookup table for the radial force calculation
			printf("Calculating the lookup table for the radial force calculation...\n");
			if(CYLINDRICAL_LOOKUP_TABLE_CALCULATED == false)
			{
				RADIAL_FORCE_TABLE = (REAL*)malloc(RADIAL_FORCE_TABLE_SIZE*sizeof(REAL)); //Allocating the lookup table
				mass_in_unit_sphere = (REAL) (2.0*pi*rho_crit*Omega_m); // in cylindrically symmetric simulations, the radial force is proportional to the mass in a unit cylinder
				ewald_max = IS_PERIODIC+1;
				ewald_cut = ((REAL) ewald_max)-0.4; //cutoff radius for the direct real-space periodic summation
				if(IS_PERIODIC==1)
				{
					get_cylindrical_force_table(RADIAL_FORCE_TABLE, Rsim,0.5*L,RADIAL_FORCE_TABLE_SIZE,RADIAL_FORCE_ACCURACY);
					if(rank==0)
						printf("Cylindrical force table calculated.\nMagnitude of the gravitational pull correction of the finite Ewald cylinder at r = %.1f Mpc is |F(Lz=0.5Lsim)|/|F(Lz=inf)| = %.7f \n\n", Rsim, RADIAL_FORCE_TABLE[RADIAL_FORCE_TABLE_SIZE-1]);
				}
				else
				{
					get_cylindrical_force_table(RADIAL_FORCE_TABLE, Rsim, L*ewald_cut,RADIAL_FORCE_TABLE_SIZE,RADIAL_FORCE_ACCURACY);
					if(rank==0)
						printf("Cylindrical force table calculated.\nMagnitude of the gravitational pull correction of the finite Ewald cylinder at r = %.1f Mpc is |F(Lz=%.1fLsim)|/|F(Lz=inf)| = %.7f \n\n", Rsim, 2*ewald_cut, RADIAL_FORCE_TABLE[RADIAL_FORCE_TABLE_SIZE-1]);
				}
				CYLINDRICAL_LOOKUP_TABLE_CALCULATED = true;
			}
		#elif defined(PERIODIC_Z)
			// in cylindrical symmetric (S^1xR^2) simulations, the radial force is proportional to the mass in a unit cylinder
			mass_in_unit_sphere = (REAL) (2.0*pi*rho_crit*Omega_m);
			if(IS_PERIODIC==1)
			{
				if(CYLINDRICAL_LOOKUP_TABLE_CALCULATED == false)
				{
					RADIAL_FORCE_TABLE = (REAL*)malloc(RADIAL_FORCE_TABLE_SIZE*sizeof(REAL)); //Allocating the lookup table
					printf("Calculating the lookup table for the radial force calculation...\n");
					get_cylindrical_force_table(RADIAL_FORCE_TABLE, Rsim,0.5*L,RADIAL_FORCE_TABLE_SIZE,RADIAL_FORCE_ACCURACY);
					if(rank==0)
						printf("Cylindrical force table calculated.\nMagnitude of the gravitational pull correction of the finite Ewald cylinder at r = %.1f Mpc is |F(Lz=0.5Lsim)|/|F(Lz=inf)| = %.7f \n\n", Rsim, RADIAL_FORCE_TABLE[RADIAL_FORCE_TABLE_SIZE-1]);
					CYLINDRICAL_LOOKUP_TABLE_CALCULATED = true;
				}
			}
			else
			{
				//setting up the lookup table with zero elements
				if(CYLINDRICAL_LOOKUP_TABLE_CALCULATED == false)
				{
					RADIAL_FORCE_TABLE = (REAL*)malloc(RADIAL_FORCE_TABLE_SIZE*sizeof(REAL)); //Allocating the lookup table
					set_REAL_array_to_zero(RADIAL_FORCE_TABLE, RADIAL_FORCE_TABLE_SIZE);
					CYLINDRICAL_LOOKUP_TABLE_CALCULATED = true;
				}
			}
		#else
			// in spherically symmetric (R^3) simulations, the radial force is proportional to the mass in a unit sphere
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
			rho_mean_full_box = kahan_sum(M, N);
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
					fprintf(stderr, "Error: The particle masses are inconsistent with the cosmological parameters:\nrho_part/rho_cosm = %.6f\nExiting.\n", rho_mean_full_box/(rho_crit*Omega_m));
					return (-1);
				}
				else
				{
					printf("Warning: The particle masses are inconsistent with the cosmological parameters set in the parameter file:\nrho_part/rho_cosm = %.6f\n\tRescaling the particle masses with this number.\n", rho_mean_full_box/(rho_crit*Omega_m));
					for(i=0;i<N;i++)
					{
						M[i] /= (rho_mean_full_box/(rho_crit*Omega_m));
					}				
				}
				#else
				printf("Warning: The particle masses are inconsistent with the cosmological parameters set in the parameter file:\nrho_part/rho_cosm = %.6f\nSince the expansion history read from an external file, this is not necessarily an error.\nPlease make sure that the particle masses are set correctly in the initial condition file.\n\n", rho_mean_full_box/(rho_crit*Omega_m));
				#endif
			}
		#else
			//Non-periodic cosmological simulations 
			REAL rho_mean_full_sphere;
			rho_mean_full_sphere = kahan_sum(M, N);
			rho_mean_full_sphere /= (4.0/3.0*pi*pow(Rsim, 3.0)); //dividing the total mass by the simulation volume
			if(fabs(rho_mean_full_sphere/(rho_crit*Omega_m) - 1) > 1e-5)
			{
				#if COSMOPARAM>=0 || !defined(COSMOPARAM)
				if(fabs(rho_mean_full_sphere/(rho_crit*Omega_m) - 1) > 1e-2)
				{
					fprintf(stderr, "Error: The particle masses are inconsistent with the cosmological parameters:\nrho_part/rho_cosm = %.6f\nExiting.\n", rho_mean_full_sphere/(rho_crit*Omega_m));
					return (-1);
				}
				else
				{
					printf("Warning: The particle masses are inconsistent with the cosmological parameters set in the parameter file:\nrho_part/rho_cosm = %.6f\n\tRescaling the particle masses with this number.\n", rho_mean_full_sphere/(rho_crit*Omega_m));
					for(i=0;i<N;i++)					{
						M[i] /= (rho_mean_full_sphere/(rho_crit*Omega_m));
					}				
				}
				#else
				printf("Warning: The particle masses are inconsistent with the cosmological parameters set in the parameter file!\nrho_part/rho_cosm = %.6f\nSince the expansion history read from an external file, this is not necessarily an error.\nPlease make sure that the particle masses are set correctly in the initial condition file.\n\n", rho_mean_full_sphere/(rho_crit*Omega_m));
				#endif
			}
			else
			{
				printf("The particle masses are consistent with the cosmological parameters set in the parameter file:\nrho_part/rho_cosm - 1 = %.6e\n\n", rho_mean_full_sphere/(rho_crit*Omega_m)-1.0);
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
				if(FIRST_T_OUT >= T*UNIT_T) //Calculating first output time
				{	
					t_next = FIRST_T_OUT/UNIT_T;
				}
				else
				{
					printf("Warning: FIRST_T_OUT is larger than the starting time! Setting the first output time to t_start+H_OUT Gy.\n");
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
	//Each MPI thread calculates the forces for its own particle range.
	double force_calc_start_time = omp_get_wtime(); //Timing the force calculation
	#if defined(PERIODIC)
		forces_periodic(x, F, ID_MPI_min, ID_MPI_max);
	#elif defined(PERIODIC_Z)
		forces_periodic_z(x, F, ID_MPI_min, ID_MPI_max);
	#elif defined(POINCARE_DODECAHEDRAL)
		forces_pds(PDS_Q, F, ID_MPI_min, ID_MPI_max);
	#else
		forces(x, F, ID_MPI_min, ID_MPI_max);
	#endif
	double force_calc_end_time = omp_get_wtime(); //Timing the force calculation

	//if the force calculation is finished, the calculated forces should be collected into the rank=0 thread`s F array
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
				//the F_buffer should be re-allocated based on the mpi_particle_range[i][2] value.
				if(!(F_buffer = (REAL*)malloc(3*(mpi_particle_range[i][2])*sizeof(REAL))))
				{
					fprintf(stderr, "MPI task %i: failed to allocate memory for F_buffer.\n", rank);
					exit(-2);
				}
				BUFFER_start_ID = mpi_particle_range[i][0];
				#ifdef USE_SINGLE_PRECISION
					MPI_Recv(F_buffer, 3*mpi_particle_range[i][2], MPI_FLOAT, i, i, MPI_COMM_WORLD, &Stat);
				#else
					MPI_Recv(F_buffer, 3*mpi_particle_range[i][2], MPI_DOUBLE, i, i, MPI_COMM_WORLD, &Stat);
				#endif
				for(j=0; j<mpi_particle_range[i][2]; j++)
				{
					F[3*(BUFFER_start_ID+j)] = F_buffer[3*j];
					F[3*(BUFFER_start_ID+j)+1] = F_buffer[3*j+1];
					F[3*(BUFFER_start_ID+j)+2] = F_buffer[3*j+2];
				}
				free(F_buffer);
			}
		}
	}
	//redistributing the workload based on the time spent in the initial force calculation
	if(rank!=0)
	{
		//sending the time spent in the force calculation to the rank=0 thread (always in double precison)
		double force_calc_time = force_calc_end_time - force_calc_start_time;
		MPI_Send(&force_calc_time, 1, MPI_DOUBLE, 0, rank, MPI_COMM_WORLD);
	}
	else
	{
		double force_calc_time = force_calc_end_time - force_calc_start_time;
		mpi_time_array[0] = force_calc_time;
		//receiving the time spent in the force calculation from the slave threads, and storing it in the mpi_time_array
		if(numtasks > 1)
		{
			for(i=1; i<numtasks;i++)
			{
				MPI_Recv(&force_calc_time, 1, MPI_DOUBLE, i, i, MPI_COMM_WORLD, &Stat);
				mpi_time_array[i] = force_calc_time;
				//storing the longest time spent in the force_calc_time variable, to calculate the workload-balance later
				if(i==1 || force_calc_time > force_calc_time)
					force_calc_time = force_calc_time;
			}
			//Adding the time spent in the force calculation to the mpi_time_array, to calculate the total time spent in the force calculation later
			force_calc_time = 0.0;
			for(i=0; i<numtasks; i++)
			{
					force_calc_time += mpi_time_array[i];
			}
			//Printing the time spent in the force calculation and workload-balance for each MPI thread
			fflush(stdout);
			printf("\nForce calculation time for each MPI thread:\n");
			for(i=0; i<numtasks; i++)
			{
				printf("MPI task %i: %fs, workload balance: %f %%\n", i, mpi_time_array[i], (mpi_time_array[i])/force_calc_time * numtasks * 100.0);
			}
			//Re-calculating the workload of each thread based on the time spent in the force calculation, and re-distributing the particles for the next iteration if the workload balance is too bad (e.g. if one thread takes more than 10% longer than the average time)
			if(numtasks > 1)
			{
				redistribute_workload(mpi_time_array, numtasks, N, mpi_particle_range);
			}
			printf("\n");
		}
	}
	BCAST_MPI_particle_ranges(); //Bcasting the particle ranges again after the workload re-distribution
	//Force vectors are collected. If SAVE_ACCELERATIONS is defined, we save the the IC with the calculated forces as a HDF5 snapshot. This can be used to compare the forces with other codes.
	#ifdef SAVE_ACCELERATIONS
	#ifdef HAVE_HDF5
	if(rank == 0)
	{
		printf("Saving the initial conditions with the calculated forces as a HDF5 snapshot for acceleration comparison...\n");
		write_hdf5_snapshot(x, v, M, true, F, true);
		printf("...done.\n");
	}
	#endif
	#endif
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
		if(rank!=0)
		{
			//Re-allocating the force array for the next iteration in all slave threads
			free(F);
			if(!(F = (REAL*)malloc((3*N_mpi_thread)*sizeof(REAL))))
			{
				fprintf(stderr, "MPI task %i: failed to allocate memory for F.\n", rank);
				exit(-2);
			}
		}
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
							write_hdf5_snapshot(x, v, M, save_accelerations, F, false);
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
							write_hdf5_snapshot(x, v, M, save_accelerations, F, false);
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
								write_hdf5_snapshot(x, v, M, save_accelerations, F, false);
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
									write_hdf5_snapshot(x, v, M, save_accelerations, F, false);
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
							write_hdf5_snapshot(x, v, M, save_accelerations, F, false);
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
			{
				printf("\nSimulation wall-clock time limit reached (%.1fmin >= %.1fmin). Stopping...\n", (omp_get_wtime()-SIM_omp_start_time)/60.0, TIME_LIMIT_IN_MINS);
				printf("Saving the current state of the simulation as the final output...\n");
				if(OUTPUT_FORMAT == 0)
					write_ascii_snapshot(x, v); //writing output
				#ifdef HAVE_HDF5
				if(OUTPUT_FORMAT == 2)
					write_hdf5_snapshot(x, v, M, save_accelerations, F, false); //writing output
				#endif
				printf("...done.\n");
			}
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
			write_hdf5_snapshot(x, v, M, save_accelerations, F, false); //writing output
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
				
				//Linear interpolation to calculate the time and Hubble parameter at the final scalefactor a_max, but only if the final scalefactor goes beyond this. (If the final scalefactor is reached.)
				if(a > a_max)
				{
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
					printf("\n\n");
				}
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

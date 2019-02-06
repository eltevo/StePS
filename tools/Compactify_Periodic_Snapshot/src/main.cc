#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <time.h>
#include "kdtree.h"
#include "global_variables.h"

double L; //Size of the simulation box
double SPHERE_DIAMETER; //diameter of the 4D sphere in Mpc used in the stereographic projection
double R_CUT;
int N_SIDE, R_GRID, N_glass;
char IC_FILE[1024]; //input file
char SphericalGlassFILE[512]; //input spherical glass file
char OUT_FILE[1024]; //output directory
int IC_FORMAT; // 0: ascii, 1:GADGET
int SPHERICAL_GLASS; // 0: using Healpix for IC generation 1: using spherical glassfile of IC generation
int FOR_COMOVING_INTEGRATION;
int RANDOM_SEED;
int RANDOM_ROTATION; //if 1 using random rotation, otherwise the code will not rotate randomly the HEALPix shells
int NUMBER_OF_INPUT_FILES; //Number of files that are written in parallel by the periodic IC generator
int TILEFAC;

double a_max; //maximal scalefactor

double* M; //Particle masses
double M_tmp;
unsigned long long int N; //Number of particles
unsigned long long int N_IC_tot; //total Number of particles in the input files
unsigned long int N_out; //Number of output particles
double** x; //particle coordinates and velocities
double** SphericalGlass; //xyz coordinates of the sperical glass
double** x_out; //Output particle coordinates and velocities
long int* COUNT;
double dist_unit_in_kpc;
double VOI[3]; //The (x,y,z) coordinates of the center of volume of interest(VOI)

double G; //Newtonian gravitational constant

//Cosmological parameters
double Omega_b,Omega_lambda,Omega_dm,Omega_r,Omega_k,Omega_m,H0,H0_start,Hubble_param;
double rho_crit; //Critical density
double a, a_start; //Scalefactor, scalefactor at the starting time, previous scalefactor
//kdtree variables
kdtree* tree;
int *index_list;


extern char __BUILD_DATE;

//Functions
void read_param(FILE *param_file);
void read_ic(FILE *ic_file, int N);
void kiiras(FILE *outfile, double** x, int N);
void stereographic_projection(double** x, double* M);
void finish_stereographic_projection(double** x_out);
void add_hubble_flow(double** x_out, unsigned long int N_out, double H0_start);
void Calculate_redshifts();
//kdtree using functions
void make_kdtree();
//Functions for reading GADGET2 format IC
int gadget_format_conversion(void);
int load_snapshot(char *fname, int files);
int allocate_memory(void);

//Function for reading ascii IC file
void read_ic(FILE *ic_file, int N)
{
	int i,j;
	x = (double**)malloc(N*sizeof(double*)); //Allocating memory
	for(i = 0; i < N; i++)
	{
                x[i] = (double*)malloc(6*sizeof(double));
	}
	M = (double*)malloc(N*sizeof(double));
	printf("\nReading IC from the %s file...\n", IC_FILE);
	for(i=0; i<N; i++) //reading
	{
		//Reading particle coordinates
        	for(j=0; j<6; j++)
        	{
			fscanf(ic_file, "%lf", & x[i][j]);
		}
		//Reading particle masses
		fscanf(ic_file, "%lf", & M[i]);
	}
	printf("...done.\n\n");
	fclose(ic_file);
	return;
}

//Function for reading spherical glass file
void read_spherical_glass(char *filename)
{
	int i;
	double theta, phi;
	printf("Reading spherical glass from the %s file...\n", filename);
	//Allocating memory
	SphericalGlass = (double**)malloc(N_glass*sizeof(double*));
	for(i=0; i<N_glass; i++)
		SphericalGlass[i] = (double*)malloc(3*sizeof(double));
	//Reading the file
	FILE *SPHERICALGLASS_FILE = fopen(filename, "r");
	for(i=0; i<N_glass; i++)
	{
		fscanf(SPHERICALGLASS_FILE, "%lf", &theta);
		fscanf(SPHERICALGLASS_FILE, "%lf", &phi);
		//Converting the spherical coordinates into cartesian
		SphericalGlass[i][0] = sin(theta)*cos(phi);
		SphericalGlass[i][1] = sin(theta)*sin(phi);
		SphericalGlass[i][2] = cos(theta);
	}
	fclose(SPHERICALGLASS_FILE);
	printf("...done.\n\n");
}

//Function for writing the output file
void kiiras(FILE *outfile, double** x, int N)
{
	int i,k;
	printf("Writing the output file...\n");
	for(i=0; i<N; i++)
        {
		if(COUNT[i]>0)
		{
                for(k=0; k<7; k++)
                {
                        fprintf(outfile, "%.16f\t",x[i][k]);
                }
                fprintf(outfile, "\n");
		}
        }
	fclose(outfile);
	printf("...done.\n");
	return;
}


int main(int argc, char *argv[])
{
	printf("-----------------------------------------------------------------------------------------------\nCompactify_Periodic_Snapshot v0.3.2.0\n Gabor Racz, 2017-2018\n\tDepartment of Physics of Complex Systems, Eotvos Lorand University | Budapest, Hungary\n\tDepartment of Physics & Astronomy, Johns Hopkins University | Baltimore, MD, USA\n\n");
	printf("Build date: %s\nCompiled with: %s\n-----------------------------------------------------------------------------------------------\n\n", BUILD_DATE, COMPILER_VERSION);
	printf("Attention!\n\tThis program can be used to generate initial conditions (ICs)\n\tfor StePS simulations, but this is highly not recommended.\n\tPlease use the StePS_IC.py script for making ICs.\n\tThe main purpose of this program is to generate spherical snapshots\n\tform a periodic snapshot only for comparison.\n\n");
	if( argc != 2)
        {
                fprintf(stderr, "Missing parameter file!\n");
                fprintf(stderr, "Call with: ./Compactify_Periodic_Snapshot  <parameter file>\n");
                return (-1);
        }
	unsigned long long int i;
	int j;
	dist_unit_in_kpc = 1.0;
	TILEFAC = 1;
	SPHERICAL_GLASS = 0;
	VOI[0] = VOI[1] = VOI[2] = -1.0;
	FILE *param_file = fopen(argv[1], "r");
	read_param(param_file);
	if(FOR_COMOVING_INTEGRATION != 1 && FOR_COMOVING_INTEGRATION != 0)
	{
		fprintf(stderr, "Error: FOR_COMOVING = %i !\nVariable FOR_COMOVING must be 1 or 0 in the parameter file.\nExiting.\n", FOR_COMOVING_INTEGRATION);
		return (-1);
	}
	if(IC_FORMAT != 0 && IC_FORMAT != 1)
        {
                fprintf(stderr, "Error: bad IC format!\nExiting.\n");
                return (-1);
        }
	printf("--------------------------------------------------------------------------------------------\n");
	N_glass = 12*N_SIDE*N_SIDE;
	if(SPHERICAL_GLASS != 0)
	{
		//reading the spherical glass file
		read_spherical_glass(SphericalGlassFILE);
		//after the glass file is readed, we build a kdtree, for the fast nearest neighbour search
		make_kdtree();
	}
        if(IC_FORMAT == 0)
        {
                FILE *ic_file = fopen(IC_FILE, "r");
                read_ic(ic_file, N);

		//Setting up the cosmological parameters
		Omega_m = Omega_b+Omega_dm;
		Omega_k = 1.-Omega_m-Omega_lambda-Omega_r;
		G = 1;
		rho_crit = 3*H0*H0/(8*pi*G);
		M_tmp = Omega_dm*rho_crit*pow(L, 3.0)/((double) N);
		for(i=0;i<N;i++)//in periodic cosmological simulations every particle has the same mass
		{
			M[i] = M_tmp;
		}

		//Allocating memory for the output particles
		N_out = 12*(unsigned long int)N_SIDE*(unsigned long int)N_SIDE*(unsigned long int)R_GRID;
		COUNT = (long int*)malloc(N_out*sizeof(long int));
		x_out = (double**) malloc(N_out*sizeof(double*));
		for(i=0;i<N_out;i++)
		{
			x_out[i] = (double*)malloc(7*sizeof(double)); //(x,y,z,vx,vy,vz,M)
		}
		for(i=0;i<N_out;i++)
		{
			for(j=0;j<7;j++)
			{
				x_out[i][j] = 0.0;
			}
			COUNT[i] = 0;
		}
		for(i=0;i<3;i++)
		{
			if(VOI[i]<0 || VOI[i]>L)
			{
				fprintf(stderr, "ERROR: (VOI_X, VOI_Y, VOI_Z) coordinate is not in the periodic box.\nExiting.\n");
				return(-1);
			}
		}

		//Making the stereographic projection
		stereographic_projection(x, M);
		finish_stereographic_projection(x_out);
		if(FOR_COMOVING_INTEGRATION == 0)
		{
			printf("Using non-comoving coordinates. Transforming the velocities with H_start=%f km/s/Mpc\n\n", H0_start*20.7386814448645);
			add_hubble_flow(x_out, N_out, H0_start);
		}

		//writing out the IC
		FILE *outfile = fopen(OUT_FILE, "w");
		kiiras(outfile, x_out, N_out);

        }
        if(IC_FORMAT == 1)
        {
                int files;
                printf("The IC file is in Gadget format.\nThe IC determines the box size.\n");
                files = 1;      /* number of files per snapshot */
                x = (double**)malloc(N*sizeof(double*)); //Allocating memory
                for(i = 0; i < N; i++)
                {
                        x[i] = (double*)malloc(6*sizeof(double));
                }
		M = (double*)malloc(N*sizeof(double));

		//Allocating memory for the output particles
		N_out = 12*(unsigned long int)N_SIDE*(unsigned long int)N_SIDE*(unsigned long int)R_GRID;
		COUNT = (long int*)malloc(N_out*sizeof(long int));
		x_out = (double**) malloc(N_out*sizeof(double*));
		for(i=0;i<N_out;i++)
		{
			x_out[i] = (double*)malloc(7*sizeof(double)); //(x,y,z,vx,vy,vz,M)
		}
		for(i=0;i<N_out;i++)
		{
			for(j=0;j<7;j++)
			{
				x_out[i][j] = 0.0;
			}
			COUNT[i] = 0;
		}

		//Setting up the cosmological parameters
		Omega_m = Omega_b+Omega_dm;
		Omega_k = 1.-Omega_m-Omega_lambda-Omega_r;
		G = 1;
		rho_crit = 3*H0*H0/(8*pi*G);

		char IC_FILE_J[0x100];
		char A[20];
		if(NUMBER_OF_INPUT_FILES>1)
		{
			for(j=0;j<NUMBER_OF_INPUT_FILES;j++)//Reading multiple files in GADGET format
			{
			sprintf(A, "%d", j);
			snprintf(IC_FILE_J, sizeof(IC_FILE_J), "%s.%s", IC_FILE, A);
			printf("Reading %s IC file...\n", IC_FILE_J);
			//Reading GADGET format ic
			load_snapshot(IC_FILE_J, files);
			gadget_format_conversion();
			M_tmp = Omega_dm*rho_crit*pow(L, 3.0)/((double) N_IC_tot);
			for(i=0;i<N;i++)//in periodic gadget simulations every particlehhave the same mass
			{
				M[i] = M_tmp;
			}
			for(i=0;i<3;i++)
			{
				if(VOI[i]<0 || VOI[i]>L)
				{
					fprintf(stderr, "ERROR: (VOI_X, VOI_Y, VOI_Z) coordinate is not in the periodic box.\nExiting.\n");
					return(-1);
				}
			}

			//Making the stereographic projection
			stereographic_projection(x, M);
			}
		}
		else
		{
			printf("Reading %s IC file...\n", IC_FILE);
			//Reading GADGET format ic
			load_snapshot(IC_FILE, files);
			gadget_format_conversion();
			M_tmp = Omega_dm*rho_crit*pow(L, 3.0)/((double) N_IC_tot);
			for(i=0;i<N;i++)//in periodic cosmological simulations every particle has the same mass
                        {
                                M[i] = M_tmp;
                        }
			for(i=0;i<3;i++)
			{
				if(VOI[i]<0 || VOI[i]>L)
				{
				fprintf(stderr, "ERROR: (VOI_X, VOI_Y, VOI_Z) coordinate is not in the periodic box.\nExiting.\n");
				return(-1);
				}
			}

			stereographic_projection(x, M);
		}
		finish_stereographic_projection(x_out);
		Calculate_redshifts();
		if(FOR_COMOVING_INTEGRATION == 0)
		{
			printf("According to the cosmological parameters:\nH_start=%.16fkm/s/Mpc\nThe code will use the H_start readed from the parameter file.\n", H0*sqrt(Omega_m*pow(a_start, -3.0) + Omega_lambda + Omega_k*pow(a_start, -2.0))*UNIT_V);
			printf("Using non-comoving coordinates. Transforming the velocities with H_start=%f km/s/Mpc\n\n", H0_start*UNIT_V);
			add_hubble_flow(x_out, N_out, H0_start);
		}
		printf("\na_start = %f\tz_start= %f\n", a_start, 1/a_start-1);
		//writing out the IC
		FILE *outfile = fopen(OUT_FILE, "w");
		kiiras(outfile, x_out, N_out);

	}
	return 0;
}

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <math.h>
#include <omp.h>
#include <time.h>
#include <algorithm>
#include "mpi.h"
#include "global_variables.h"

#ifdef USE_SINGLE_PRECISION
typedef float REAL;
#else
typedef double REAL;
#endif

int t,N,el,hl;
int e[2202][4];
int H[2202][4];
REAL SOFT_CONST[8];
REAL w[3];
double a_max;
REAL* x;
REAL* v;
REAL* F;
bool* IN_CONE;
double h, h_min, h_max, T, t_next, t_bigbang;
REAL mean_err;
double FIRST_T_OUT, H_OUT; //First output time, output frequency in Gy
double rho_crit; //Critical density
REAL mass_in_unit_sphere; //Mass in unit sphere

int n_GPU; //number of cuda capable GPUs
int numtasks, rank; //Variables for MPI
int N_mpi_thread; //Number of calculated forces in one MPI thread
int ID_MPI_min, ID_MPI_max; //max and min ID of of calculated forces in one MPI thread
MPI_Status Stat;
int BUFFER_start_ID;
REAL* F_buffer;

REAL x4, err, errmax;
REAL beta, ParticleRadi, rho_part, M_min;

int IS_PERIODIC, COSMOLOGY;
int COMOVING_INTEGRATION; //Comoving integration 0=no, 1=yes, used only when  COSMOLOGY=1
REAL L;
char IC_FILE[1024];
char OUT_DIR[1024];
char OUT_LST[1024]; //output redshift list file. only used when OUTPUT_FORMAT=1
extern char __BUILD_DATE;
int IC_FORMAT; // 0: ascii, 1:GADGET
int OUTPUT_FORMAT; // 0: time, 1: redshift
double MIN_REDSHIFT; //The minimal output redshift. Lower redshifts considered 0.
int REDSHIFT_CONE; // 0: standard output files 1: one output redshift cone file
double *out_list; //Output redshits
double *r_bin_limits; //bin limints in Dc for redshift cone simulations
int out_list_size; //Number of output redshits

double Omega_b,Omega_lambda,Omega_dm,Omega_r,Omega_k,Omega_m,H0,Hubble_param, Decel_param, delta_Hubble_param, Hubble_tmp; //Cosmologycal parameters

double epsilon=1;
double sigma=1;
REAL G;//Newtonian gravitational constant
REAL* M;//Particle mass
REAL M_tmp;
double a, a_start,a_prev,a_tmp;//Scalefactor, scalefactor at the starting time, previous scalefactor
double Omega_m_eff; //Effective Omega_m
double delta_a;

int RESTART; //Restarted simulation(0=no, 1=yes)
double T_RESTART; //Time of restart
double A_RESTART; //Scalefactor at the time of restart
double H_RESTART; //Hubble-parameter at the time of restart


//Functions for reading GADGET2 format IC
int gadget_format_conversion(void);
int load_snapshot(char *fname, int files);
int allocate_memory(void);
int reordering(void);


void read_ic(FILE *ic_file, int N);
void read_param(FILE *param_file);
int read_OUT_LST();
void step(REAL* x, REAL* v, REAL* F);
void kiiras(REAL* x, REAL* v);
void Log_write();
void forces(REAL* x, REAL* F, int ID_min, int ID_max);
void forces_periodic(REAL*x, REAL*F, int ID_min, int ID_max);
double friedmann_solver_start(double a0, double t0, double h, double Omega_lambda, double Omega_r, double Omega_m, double H0, double a_start);
double friedman_solver_step(double a0, double h, double Omega_lambda, double Omega_r, double Omega_m, double Omega_k, double H0);
int ewald_space(REAL R, int ewald_index[2102][4]);
double CALCULATE_decel_param(double a);
//Functions used in MPI parallelisation
void BCAST_global_parameters();

void read_ic(FILE *ic_file, int N)
{
int i,j;

x = (REAL*)malloc(3*N*sizeof(REAL)); //Allocating memory for the coordinates
v = (REAL*)malloc(3*N*sizeof(REAL)); //Allocating memory for the velocities
F = (REAL*)malloc(3*N*sizeof(REAL)); //Allocating memory for the forces
M = (REAL*)malloc(N*sizeof(REAL));


printf("\nReading IC from the %s file...\n", IC_FILE);
for(i=0; i<N; i++) //reading
{
	//Reading particle coordinates
	for(j=0; j<3; j++)
	{
		#ifdef USE_SINGLE_PRECISION
		fscanf(ic_file, "%f", &x[3*i + j]);
		#else
		fscanf(ic_file, "%lf", &x[3*i + j]);
		#endif

	}
	for(j=0; j<3; j++)
	{
		#ifdef USE_SINGLE_PRECISION
		fscanf(ic_file, "%f", &v[3*i + j]);
		#else
		fscanf(ic_file, "%lf", &v[3*i + j]);
		#endif
	}
	//Reading particle masses
	#ifdef USE_SINGLE_PRECISION
	fscanf(ic_file, "%f", & M[i]);
	#else
	fscanf(ic_file, "%lf", & M[i]);
	#endif

}
printf("...done.\n\n");
fclose(ic_file);
return;
}

int read_OUT_LST()
{
	FILE *infile = fopen(OUT_LST, "r");
	FILE *in_bin_file;
	char BIN_LIST[1038];
	snprintf(BIN_LIST, sizeof(BIN_LIST), "%s_rlimits", OUT_LST);
	char *buffer, *buffer1;
	char ch;
	int data[2]; //[0]: previous char; [1]: actual char
	int i, j, size;
	fseek(infile,0,SEEK_END);
	size = ftell(infile);
	fseek(infile,0,SEEK_SET);
	buffer = (char*)malloc((size+1)*sizeof(char));
	i=0;
	while((ch=fgetc(infile)) != EOF)
	{
		buffer[i] = ch;
		i++;
	}
	fclose(infile);
	data[0] = 0;
	data[1] = 0;
	size = 0;
	for(j=0; j<i+1; j++)
	{
		if(i!=0)
		{
			data[0] = data[1];
		}
		if(buffer[j] == '\t' || buffer[j] == ' ' || buffer[j] == '\n' || buffer[j] == '\0')
		{
			data[1] = 0;
		}
		else
		{
			data[1] = 1;
		}

		if(data[1] == 0 && data[0] == 1)
		{
			size++;
		}
	}
	out_list = (double*)malloc(size*sizeof(double));
	int offset;
	for(i=0; i<size; i++)
	{
		sscanf(buffer, "%lf%n", &out_list[i], &offset);
		buffer += offset;
	}
	std::sort(out_list, out_list+size, std::greater<double>());
	out_list_size = size;
	size = 0;
	if(REDSHIFT_CONE == 1)
	{
		//reading the limist of the comoving distance bins
		in_bin_file = fopen(BIN_LIST, "r");
		fseek(in_bin_file,0,SEEK_END);
		size = ftell(in_bin_file);
		fseek(in_bin_file,0,SEEK_SET);
		buffer1 = (char*)malloc((size+1)*sizeof(char));
		i=0;
		while((ch=fgetc(in_bin_file)) != EOF)
		{
			buffer1[i] = ch;
			i++;
		}
		fclose(in_bin_file);
		data[0] = 0;
		data[1] = 0;
		size = 0;
		for(j=0; j<i+1; j++)
		{
			if(i!=0)
			{
				data[0] = data[1];
			}
			if(buffer1[j] == '\t' || buffer1[j] == ' ' || buffer1[j] == '\n' || buffer1[j] == '\0')
			{
				data[1] = 0;
			}
			else
			{
				data[1] = 1;
			}

			if(data[1] == 0 && data[0] == 1)
			{
				size++;
			}
		}
		r_bin_limits = (double*)malloc(size*sizeof(double));
		for(i=0; i<size; i++)
		{
			sscanf(buffer1, "%lf%n", &r_bin_limits[i], &offset);
			buffer1 += offset;
		}
		std::sort(r_bin_limits, r_bin_limits+size, std::greater<double>());
		if(size - 1 != out_list_size)
		{
			fprintf(stderr, "Error: The number of redshift bins (=%i) and radial bins (=%i) are not equal!\n", size - 1, out_list_size);
			return (-1);
		}
	}
	printf("\n");
	return 0;	

}

void write_redshift_cone(REAL *x, REAL *v, double *limits, int z_index, int delta_z_index, int ALL)
{
	//Writing out the redshift cone
	char filename[0x400];
	int i, j;
	int COUNT=0;
	double COMOVING_DISTANCE, z_write;
	z_write = out_list[z_index];
	snprintf(filename, sizeof(filename), "%sredshift_cone.dat", OUT_DIR);
	if(ALL == 0)
		printf("Saving: z=%f:\t%fMpc<Dc<%fMpc bin of the\n%s redshift cone.\ndelta_z_index = %i\n",out_list[z_index], limits[z_index+1], limits[z_index-delta_z_index+1], filename, delta_z_index);
	else
		printf("Saving: z=%f:\tDc<%f\n%s\n redshift cone.\n", t_next, limits[z_index-delta_z_index+1], filename);
	FILE *redshiftcone_file = fopen(filename, "a");
	if(ALL == 0)
	{
		for(i=0; i<N; i++)
		{
			COMOVING_DISTANCE = sqrt(x[3*i]*x[3*i] + x[3*i+1]*x[3*i+1] +x[3*i+2]*x[3*i+2]);
			if(limits[z_index+1] <= COMOVING_DISTANCE && IN_CONE[i] == false )
			{
				for(j=0; j<3; j++)
				{
					fprintf(redshiftcone_file, "%.16f\t",x[3*i+j]);
				}
				for(j=0; j<3; j++)
				{
					fprintf(redshiftcone_file, "%.16f\t",v[3*i+j]);
				}
				fprintf(redshiftcone_file, "%.16f\t%.16f\t%.16f\t%i\n", M[i], COMOVING_DISTANCE, out_list[z_index], i);
				IN_CONE[i] = true;
				COUNT++;
			}
		}
	}
	else
	{
		for(i=0; i<N; i++)
		{
			COMOVING_DISTANCE = sqrt(x[3*i]*x[3*i] + x[3*i+1]*x[3*i+1] +x[3*i+2]*x[3*i+2]);
			if(IN_CONE[i] == false)
			{
				//searching for the proper redshift shell
				j=z_index;
				while(j++)
				{
					if(limits[j] <= COMOVING_DISTANCE)
					{
						z_write = out_list[j];
						break;
					}
				}
				for(j=0; j<3; j++)
				{
					fprintf(redshiftcone_file, "%.16f\t",x[3*i+j]);
				}
				for(j=0; j<3; j++)
				{
					fprintf(redshiftcone_file, "%.16f\t",v[3*i+j]);
				}
				fprintf(redshiftcone_file, "%.16f\t%.16f\t%.16f\t%i\n", M[i], COMOVING_DISTANCE, z_write, i);
				IN_CONE[i] = true;
				COUNT++;
			}
		}
	}
	fclose(redshiftcone_file);
	printf("%i particles were written out.\n", COUNT); 
	COUNT = 0;
	
}

void kiiras(REAL* x, REAL *v)
{
	int i,k;
	char A[20];
	if(COSMOLOGY == 1)
	{
		if(OUTPUT_FORMAT == 0)
		{
			sprintf(A, "%d", (int)(round(100*t_next*47.1482347621227)));
		}
		else
		{
			sprintf(A, "%d", (int)(round(1000*t_next)));
		}
	}
	else
	{
		sprintf(A, "%d", (int)(round(100*t_next)));
	}
	char filename[0x400];
	if(OUTPUT_FORMAT == 0)
	{
		snprintf(filename, sizeof(filename), "%st%s.dat", OUT_DIR, A);
	}
	else
	{
		snprintf(filename, sizeof(filename), "%sz%s.dat", OUT_DIR, A);
	}
	if(COSMOLOGY == 0)
	{
			printf("Saving: t= %f, file: \"%st%s.dat\" \n", t_next, OUT_DIR, A);
	}
	else
	{
		if(OUTPUT_FORMAT == 0)
		{
			printf("Saving: t= %f, file: \"%st%s.dat\" \n", t_next*47.1482347621227, OUT_DIR, A);
		}
		else
		{
			printf("Saving: z= %f, file: \"%sz%s.dat\" \n", t_next, OUT_DIR, A);
		}
	}
	FILE *coordinate_file;
	if(t < 1)
	{
		coordinate_file = fopen(filename, "w");
	}
	else
	{
		coordinate_file = fopen(filename, "a");
	}

	for(i=0; i<N; i++)
	{
		for(k=0; k<3; k++)
		{
			fprintf(coordinate_file, "%.16f\t",x[3*i+k]);
		}
		for(k=0; k<3; k++)
		{
			fprintf(coordinate_file, "%.16f\t",v[3*i+k]);
		}
		fprintf(coordinate_file, "%.16f\t",M[i]);
		fprintf(coordinate_file, "\n");
	}

	fclose(coordinate_file);
}

void Log_write() //Writing logfile
{
	FILE *LOGFILE;
	char A[] = "Logfile.dat";
	char filename[0x100];
	snprintf(filename, sizeof(filename), "%s%s", OUT_DIR, A);
	LOGFILE = fopen(filename, "a");
	fprintf(LOGFILE, "%.15f\t%e\t%e\t%.15f\t%.15f\t%.15f\t%.15f\t%.10f\n", T*47.1482347621227, errmax, h*47.1482347621227, a, a_max/a-1, Hubble_param*20.7386814448645, Decel_param, Omega_m_eff);
	fclose(LOGFILE);
}


int main(int argc, char *argv[])
{
	//initialize MPI
	MPI_Init(&argc,&argv);
	// get number of tasks
	MPI_Comm_size(MPI_COMM_WORLD,&numtasks);
	// get my rank
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);
	if(rank == 0)
	{
		printf("----------------------------------------------------------------------------------------------\nStePS v0.3.0.2\n (STEreographically Projected cosmological Simulations)\n\n Gabor Racz, 2017-2018\n\tDepartment of Physics of Complex Systems, Eotvos Lorand University | Budapest, Hungary\n\tDepartment of Physics & Astronomy, Johns Hopkins University | Baltimore, MD, USA\n\n");
		printf("Build date: %lu\n----------------------------------------------------------------------------------------------\n\n", (unsigned long) &__BUILD_DATE);
	}
	if(numtasks != 1 && rank == 0)
	{
		printf("Number of MPI tasks: %i\n", numtasks);
	}
	int i,j;
	int CONE_ALL=0;
	RESTART = 0;
	T_RESTART = 0;
	OUTPUT_FORMAT = 0;
	if( argc < 2 )
	{
		if(rank == 0)
		{
			fprintf(stderr, "Missing parameter file!\n");
			fprintf(stderr, "Call with: ./StePS  <parameter file>\n");
		}
		return (-1);
	}
	else if(argc > 3)
	{
		if(rank == 0)
		{
			fprintf(stderr, "Too many arguments!\n");
			fprintf(stderr, "Call with: ./StePS  <parameter file>\nor with: ./StePS_CUDA  <parameter file> \'i\', where \'i\' is the number of the cuda capable GPUs.\n");
		}
		return (-1);
	}
	//the rank=0 thread reads the paramfile, and bcast the variables to the other threads 
	if(rank == 0)
	{
		FILE *param_file = fopen(argv[1], "r");
		read_param(param_file);
	}
	BCAST_global_parameters();
	if(rank == 0)
		N_mpi_thread = (N/numtasks) + (N%numtasks);
	else
		N_mpi_thread = N/numtasks;
	if(argc == 3)
	{
		n_GPU = atoi( argv[2] );
		if(rank == 0)
			printf("Using %i cuda capable GPU per MPI task.\n", n_GPU);
	}
	else
	{
		n_GPU = 1;
	}
	if(IS_PERIODIC>1)
	{
		el = ewald_space(3.6,e);
		if(IS_PERIODIC>2)
		{
			hl = ewald_space(8.0,H);
		}
	}
	if(OUTPUT_FORMAT != 0 && OUTPUT_FORMAT !=1)
	{
		fprintf(stderr, "Error: bad OUTPUT format!\nExiting.\n");
		return (-2);
	}
	if(OUTPUT_FORMAT == 1 && COSMOLOGY != 1)
	{
		fprintf(stderr, "Error: you can not use redshift output format in non-cosmological simulations. \nExiting.\n");
		return (-2);
	}
	if(OUTPUT_FORMAT ==1)
	{
		if(rank == 0)
		{
			if(0 != read_OUT_LST())
			{
				fprintf(stderr, "Exiting.\n");
				return (-2);
			}
		}
	}
	if(rank == 0)
	{
		if(REDSHIFT_CONE == 1 && COSMOLOGY != 1)
		{
			fprintf(stderr, "Error: you can not use redshift cone output format in non-cosmological simulations. \nExiting.\n");
			return (-2);
		}
		if(REDSHIFT_CONE == 1 && OUTPUT_FORMAT != 1)
		{
			fprintf(stderr, "Error: you must use redshift output format in redshift cone simulations. \nExiting.\n");
			return (-2);
		}
		if(REDSHIFT_CONE == 1)
		{
			//Allocating memory for the bool array
			IN_CONE = new bool[N];
			std::fill(IN_CONE, IN_CONE+N, false ); //setting every element to false
		}
		if(IC_FORMAT != 0 && IC_FORMAT != 1)
		{
			fprintf(stderr, "Error: bad IC format!\nExiting.\n");
			return (-1);
		}
		if(IC_FORMAT == 0)
		{
			FILE *ic_file = fopen(IC_FILE, "r");
			read_ic(ic_file, N);
		}
		if(IC_FORMAT == 1)
		{
			int files;
			printf("The IC file is in Gadget format.\nThe IC determines the box size.\n");
			files = 1;      /* number of files per snapshot */
			x = (REAL*)malloc(3*N*sizeof(REAL)); //Allocating memory for the coordinates
			v = (REAL*)malloc(3*N*sizeof(REAL)); //Allocating memory for the velocities
			F = (REAL*)malloc(3*N*sizeof(REAL)); //Allocating memory for the forces
			M = (REAL*)malloc(N*sizeof(REAL));
			load_snapshot(IC_FILE, files);
			reordering();
			gadget_format_conversion();
		}
		//Rescaling speeds. If one uses Gadget format: http://wwwmpa.mpa-garching.mpg.de/gadget/gadget-list/0113.html
		if(RESTART == 0 && COSMOLOGY == 1 && COMOVING_INTEGRATION == 1)
		{
		for(i=0;i<N;i++)
		{
			v[3*i] = v[3*i]/sqrt(a_start);
			v[3*i+1] = v[3*i+1]/sqrt(a_start);
			v[3*i+2] = v[3*i+2]/sqrt(a_start);
		}
		}
		if(numtasks > 1)
		{
			F_buffer = (REAL*)malloc(3*(N/numtasks)*sizeof(REAL));
		}
	}
	else
	{
		//Allocating memory for the particle datas on the rank != 0 MPI threads
		x = (REAL*)malloc(3*N*sizeof(REAL)); //Allocating memory fo the coordinates
		v = (REAL*)malloc(3*N*sizeof(REAL)); //Allocating memory for the velocities
		F = (REAL*)malloc(3*N_mpi_thread*sizeof(REAL));//There is no need to allocate for N forces. N/numtasks should be enough
		M = (REAL*)malloc(N*sizeof(REAL));

	}
	//Bcasting the ICs to the rank!=0 threads
#ifdef USE_SINGLE_PRECISION
	MPI_Bcast(x,3*N,MPI_FLOAT,0,MPI_COMM_WORLD);
        MPI_Bcast(M,N,MPI_FLOAT,0,MPI_COMM_WORLD);
#else
	MPI_Bcast(x,3*N,MPI_DOUBLE,0,MPI_COMM_WORLD);
	MPI_Bcast(M,N,MPI_DOUBLE,0,MPI_COMM_WORLD);
#endif
	//Critical density, Gravitational constant and particle masses
	G = 1;
	if(COSMOLOGY == 1)
	{
	if(COMOVING_INTEGRATION == 1)
	{
		Omega_m = Omega_b+Omega_dm;
		Omega_k = 1.-Omega_m-Omega_lambda-Omega_r;
		rho_crit = 3*H0*H0/(8*pi*G);
		mass_in_unit_sphere = (REAL) (4.0*pi*rho_crit*Omega_m/3.0);
		M_tmp = Omega_dm*rho_crit*pow(L, 3.0)/((REAL) N);
		if(IS_PERIODIC>0)
		{
		if(rank == 0)
			printf("Every particle has the same mass in periodic cosmological simulations.\nM=%.10f*10e+11M_sol\n", M_tmp);
		for(i=0;i<N;i++)//Every particle has the same mass in periodic cosmological simulations
		{
			M[i] = M_tmp;
		}
		}
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
			printf("COSMOLOGY = 1 and COMOVING_INTEGRATION = 0:\nNon-comoving, full Newtonian cosmological simulation. If you want physical solution, you should set Omega_lambda to zero.\na_max is used as maximal time in Gy in the parameter file.\n\n");
		Omega_m = Omega_b+Omega_dm;
		Omega_k = 1.-Omega_m-Omega_lambda-Omega_r;
		rho_crit = 3*H0*H0/(8*pi*G);
	}
	}
	else
	{
		if(rank == 0)
			printf("Running classical gravitational N-body simulation.\n");
	}
	//Searching the minimal mass particle
	if(rank == 0)
	{
		M_min = M[0];
		for(i=0;i<N;i++)
		{
			if(M_min>M[i])
			{
				M_min = M[i];
			}
		}
		rho_part = M_min/(4.0*pi*pow(ParticleRadi, 3.0) / 3.0);
	}
#ifdef USE_SINGLE_PRECISION
	MPI_Bcast(&M_min,1,MPI_FLOAT,0,MPI_COMM_WORLD);
	MPI_Bcast(&rho_part,1,MPI_FLOAT,0,MPI_COMM_WORLD);
#else
	MPI_Bcast(&M_min,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
	MPI_Bcast(&rho_part,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
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
		a = a_start;
		a_tmp = a;
		if(COMOVING_INTEGRATION == 1)
		{
			printf("a_start/a_today=%.8f\tz=%.8f\n", a, 1/a-1);
		}
		if(RESTART == 0)
		{
			T = friedmann_solver_start(1,0,h_min*0.00031,Omega_lambda,Omega_r,Omega_m,H0,a_start);
		}
		else
		{
			T = T_RESTART/47.1482347621227; //if the simulation is restarted
			if(COMOVING_INTEGRATION == 1)
			{
			Hubble_param = H_RESTART;
			a = A_RESTART;
			recalculate_softening();
			}
		}
		Delta_T_out = H_OUT/47.1482347621227; //Output frequency
		if(OUTPUT_FORMAT == 0)
		{
			if(FIRST_T_OUT >= T) //Calculating first output time
			{
				t_next = FIRST_T_OUT/47.1482347621227;
			}
			else
			{
				t_next = T+Delta_T_out;
			}
		}
		else
		{
			if(1.0/a-1.0 > out_list[0])
			{
				t_next = out_list[0];
			}
			else
			{
				i=0;
				while(out_list[i] > 1.0/a-1.0)
				{
					t_next = out_list[i];
					i++;
					if(i == out_list_size && out_list[i] > 1.0/a-1.0)
					{
						fprintf(stderr, "Error: No valid output redshift!\nExiting.\n");
						return (-2);
					}
				}
			}
		}
		if(COMOVING_INTEGRATION == 1)
		{
		printf("Initial time:\tt_start = %.10fGy\nInitial scalefactor:\ta_start = %.8f\nMaximal scalefactor:\t%.8f\n\n", T*47.1482347621227, a, a_max);
		}
		if(COMOVING_INTEGRATION == 0)
		{
			Hubble_param = 0;
			a_tmp = 0;
			a_max = a_max/47.1482347621227;
			a = 1;
			printf("Initial time:\tt_start = %.10fGy\nMaximal time:\t%.8f\n\n", T*47.1482347621227, a_max*47.1482347621227);
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
		t_next = T+Delta_T_out;
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
		if(IS_PERIODIC < 2)
		{
			forces(x, F, ID_MPI_min, ID_MPI_max);
		}
		if(IS_PERIODIC == 2)
		{
			forces_periodic(x, F, ID_MPI_min, ID_MPI_max);
		}
	}
	else
	{
		ID_MPI_min = (N%numtasks) + (rank)*(N/numtasks);
		ID_MPI_max = (N%numtasks) + (rank+1)*(N/numtasks)-1;
		if(IS_PERIODIC < 2)
		{
			forces(x, F, ID_MPI_min, ID_MPI_max);
		}
		if(IS_PERIODIC == 2)
		{
			forces_periodic(x, F, ID_MPI_min, ID_MPI_max);
		}
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
	if(COSMOLOGY == 1)
	{
		Hubble_tmp = H0*sqrt(Omega_m*pow(a, -3)+Omega_r*pow(a, -4)+Omega_lambda+Omega_k*pow(a, -2));
		Hubble_param = H0*sqrt(Omega_m*pow(a, -3)+Omega_r*pow(a, -4)+Omega_lambda+Omega_k*pow(a, -2));
		if(rank == 0)
			printf("Initial Hubble-parameter from the cosmological parameters:\nH(z=%f) = %fkm/s/Mpc\n\n", 1.0/a-1.0, Hubble_param*20.7386814448645);
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
			h=h_max;
                }
	}
	MPI_Bcast(&h,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
	if(rank == 0)
		printf("The simulation is starting...\n");
	REAL T_prev,Hubble_param_prev;
	T_prev = T;
	Hubble_param_prev = Hubble_param;
	for(t=0; a_tmp<a_max; t++)
	{
		if(rank == 0)
		{
		printf("\n\n----------------------------------------------------------------------------------------------\n");
		if(COSMOLOGY == 1)
                {
			if(COMOVING_INTEGRATION == 1)
			{
                        printf("Timestep %i, t=%.8fGy, h=%fGy, a=%.8f, H=%.8fkm/s/Mpc, z=%.8f:\n", t, T*47.1482347621227, h*47.1482347621227, a, Hubble_param*20.7386814448645, 1.0/a-1.0);
			}
			else
			{
				printf("Timestep %i, t=%.8fGy, h=%fGy\n", t, T*47.1482347621227, h*47.1482347621227);
			}
                }
                else
                {
                        printf("Timestep %i, t=%f, h=%f:\n", t, T, h);
                }
		}
		Hubble_param_prev = Hubble_param;
		T_prev = T;
		T = T+h;
		step(x, v, F);
		if(rank == 0)
		{
			Log_write();	//Writing logfile
			if(OUTPUT_FORMAT == 0)
			{
				if(T > t_next)
				{
					kiiras(x, v);
					t_next=t_next+Delta_T_out;
					if(COSMOLOGY == 1)
					{
						printf("t = %f Gy\n\th=%f Gy\n", T*47.1482347621227, h*47.1482347621227);
					}
					else
					{
						printf("t = %f\n\terr_max = %e\th=%f\n", T, errmax, h);
					}
				}
			}
			else
			{
				if( 1.0/a-1.0 < t_next)
				{
					if(REDSHIFT_CONE != 1)
						kiiras(x, v);
					if(REDSHIFT_CONE == 1)
					{
						if(a_tmp >= a_max)
						{
							CONE_ALL = 1;
							printf("Last timestep.\n");
							kiiras(x, v);
						}
						write_redshift_cone(x, v, r_bin_limits, out_z_index, delta_z_index, CONE_ALL);
					}
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
			h = (double) pow(2*mean_err/errmax, 0.5);
			if(h<h_min)
			{
				h=h_min;
			}
			if(h>h_max)
			{
				h=h_max;
			}
		}
		MPI_Bcast(&h,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
	}
	if(OUTPUT_FORMAT == 0 && rank == 0)
	{
		kiiras(x, v); //writing output
	}
	if(rank == 0)
	{
		printf("\n\n----------------------------------------------------------------------------------------------\n");
		printf("The simulation ended. The final state:\n");
		if(COSMOLOGY == 1)
		{
			if(COMOVING_INTEGRATION == 1)
			{
				printf("Timestep %i, t=%.8fGy, h=%f, a=%.8f, H=%.8f, z=%.8f\n", t, T*47.1482347621227, h*47.1482347621227, a, Hubble_param*20.7386814448645, a_max/a-1.0);

				double a_end, b_end;
				a_end = (Hubble_param - Hubble_param_prev)/(a-a_prev);
				b_end = Hubble_param_prev-a_end*a_prev;
				double H_end = a_max*a_end+b_end;
				a_end = (T - T_prev)/(a-a_prev);
			        b_end = T_prev-a_end*a_prev;
				double T_end = a_max*a_end+b_end;
				printf("\nAt a = %f state, with linear interpolation:\n",a_max);
				printf("t=%.8fGy, a=%.8f, H=%.8fkm/s/Mpc\n\n", T_end*47.1482347621227, a_max, H_end*20.7386814448645);
			}
			else
			{
				printf("Timestep %i, t=%.8fGy, h=%f\n", t, T*47.1482347621227, h*47.1482347621227);
			}
		}
		else
		{
			printf("Timestep %i, t=%f, h=%f, a=%f:\n", t, T, h, a);
		}
		//Timing
		double SIM_omp_end_time = omp_get_wtime();
		//Timing
		printf("Wall-clock time of the simulation = %fs\n", SIM_omp_end_time-SIM_omp_start_time);
	}
	// done with MPI
	MPI_Finalize();
	return 0;
}

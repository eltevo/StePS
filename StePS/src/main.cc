#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <math.h>
#include <omp.h>
#include <time.h>
#include <algorithm>
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
REAL** x;
REAL** F;
double h, h_min, h_max, T, t_next, t_bigbang;
REAL mean_err;
double FIRST_T_OUT, H_OUT; //First output time, output frequency in Gy
double rho_crit; //Critical density
REAL mass_in_unit_sphere; //Mass in unit sphere

int GPU_ID; //ID of the GPU

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
double delta_a, a_prev1, a_prev2, h_prev;

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
void step(REAL** x, REAL** F);
void kiiras(REAL** x);
void Log_write(REAL** x);
void forces_old(REAL** x, REAL** F);
void forces_old_periodic(REAL**x, REAL**F);
void forces_EWALD(REAL** x, REAL** F);
double friedmann_solver_start(double a0, double t0, double h, double Omega_lambda, double Omega_r, double Omega_m, double H0, double a_start);
double friedman_solver_step(double a0, double h, double Omega_lambda, double Omega_r, double Omega_m, double Omega_k, double H0);
int ewald_space(REAL R, int ewald_index[2102][4]);
double CALCULATE_decel_param(double a, double a_prev1, double a_prev2, double h, double h_prev);


void read_ic(FILE *ic_file, int N)
{
int i,j;

x = (REAL**)malloc(N*sizeof(REAL*)); //Allocating memory
for(i = 0; i < N; i++)
	{
		x[i] = (REAL*)malloc(6*sizeof(REAL));
	}

F = (REAL**)malloc(N*sizeof(REAL*)); 
for(i = 0; i < N; i++)
{
	F[i] = (REAL*)malloc(3*sizeof(REAL));
}
M = (REAL*)malloc(N*sizeof(REAL));


printf("\nReading IC from the %s file...\n", IC_FILE);
for(i=0; i<N; i++) //reading
{
	//Reading particle coordinates
	for(j=0; j<6; j++)
	{
		#ifdef USE_SINGLE_PRECISION
		fscanf(ic_file, "%f", & x[i][j]);
		#else
		fscanf(ic_file, "%lf", & x[i][j]);
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
	printf("The readed output redshift list:\n");
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
	i=0;
	while(out_list[i]>MIN_REDSHIFT && i < out_list_size)
	{
			printf("Out_z[%i] = \t%lf\n", i, out_list[i]);
			i++;
	}
	printf("\n");
	return 0;	

}

void write_redshift_cone(REAL**x, double *limits, int z_index, int ALL)
{
	//Writing out the redshift cone
	char filename[0x400];
	int i, j;
	double COMOVING_DISTANCE, z_write;
	z_write = out_list[z_index];
	snprintf(filename, sizeof(filename), "%sredshift_cone.dat", OUT_DIR);
	if(ALL == 0)
		printf("Saving: z=%lf:\t%lfMpc<Dc<%lfMpc bin of the\n%s\n redshift cone.\n",out_list[z_index], limits[z_index+1], limits[z_index], filename);
	else
		printf("Saving: z=%lf:\tDc<%lf\n%s\n redshift cone.\n", t_next, limits[z_index], filename);
	FILE *redshiftcone_file = fopen(filename, "a");
	if(ALL == 0)
	{
		for(i=0; i<N; i++)
		{
			COMOVING_DISTANCE = sqrt(x[i][0]*x[i][0] + x[i][1]*x[i][1] +x[i][2]*x[i][2]);
			if(limits[z_index+1] < COMOVING_DISTANCE && COMOVING_DISTANCE < limits[z_index])
			{
				for(j=0; j<6; j++)
				{
					fprintf(redshiftcone_file, "%.16f\t",x[i][j]);
				}
				fprintf(redshiftcone_file, "%.16f\t%.16f\t%.16f\n", M[i], COMOVING_DISTANCE, out_list[z_index]);
			}
		}
	}
	else
	{
		for(i=0; i<N; i++)
		{
			COMOVING_DISTANCE = sqrt(x[i][0]*x[i][0] + x[i][1]*x[i][1] +x[i][2]*x[i][2]);
			if(COMOVING_DISTANCE < limits[z_index])
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
				for(j=0; j<6; j++)
				{
					fprintf(redshiftcone_file, "%.16f\t",x[i][j]);
				}
				fprintf(redshiftcone_file, "%.16f\t%.16f\t%lf\n", M[i], COMOVING_DISTANCE, z_write);
			}
		}
	}
	fclose(redshiftcone_file);
	
	
}

void kiiras(REAL** x)
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
		for(k=0; k<6; k++)
		{
			fprintf(coordinate_file, "%.16f\t",x[i][k]);
		}
		fprintf(coordinate_file, "%.16f\t",M[i]);
		fprintf(coordinate_file, "\n");
	}

	fclose(coordinate_file);
}

void Log_write(REAL** x) //Writing logfile
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
	printf("-------------------------------------------------------------------\nStePS v0.1.2.2\n (STEreographically Projected cosmological Simulations)\n\n Gabor Racz, 2017\n Department of Physics of Complex Systems, Eötvös Loránd University\n\n");
	printf("Build date: %zu\n-------------------------------------------------------------------\n\n", (unsigned long) &__BUILD_DATE);
	int i;
	int CONE_ALL=0;
	RESTART = 0;
	T_RESTART = 0;
	OUTPUT_FORMAT = 0;
	if( argc < 2)
	{
		fprintf(stderr, "Missing parameter file!\n");
		fprintf(stderr, "Call with: ./StePS  <parameter file>\n");
		return (-1);
	}
	else if(argc > 3)
	{
		fprintf(stderr, "Too many arguments!\n");
		fprintf(stderr, "Call with: ./StePS  <parameter file>\nor with: ./StePS_CUDA  <parameter file> \'i\', where \'i\' is the id of the GPU.\n");
		return (-1);
	} 
	FILE *param_file = fopen(argv[1], "r");
	read_param(param_file);
	if(argc == 3)
	{
		GPU_ID = std::stoi( argv[2] );
		printf("Using GPU %i\n", GPU_ID);
	}
	else
	{
		GPU_ID = 0;
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
		if(0 != read_OUT_LST())
		{
			fprintf(stderr, "Exiting.\n");
			return (-2);
		}
	}
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
		x = (REAL**)malloc(N*sizeof(REAL*)); //Allocating memory
		for(i = 0; i < N; i++)
		{
			x[i] = (REAL*)malloc(6*sizeof(REAL));
		}
		F = (REAL**)malloc(N*sizeof(REAL*));
		for(i = 0; i < N; i++)
		{
			F[i] = (REAL*)malloc(3*sizeof(REAL));
		}
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
		x[i][3] = x[i][3]/sqrt(a_start);
		x[i][4] = x[i][4]/sqrt(a_start);
		x[i][5] = x[i][5]/sqrt(a_start);
	}
	}
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
			fprintf(stderr, "Error: COSMOLOGY = 1, IS_PERIODOC>0 and COMOVING_INTEGRATION = 0!\nThis code can not handle non-comoving periodic cosmological simulations.\nExiting.\n");
			return (-1);
		}
		printf("COSMOLOGY = 1 and COMOVING_INTEGRATION = 0:\nNon-comoving, full Newtonian cosmological simulation. If you want physical solution, you should set Omega_lambda to zero.\na_max is used as maximal time in Gy in the parameter file.\n\n");
		Omega_m = Omega_b+Omega_dm;
		Omega_k = 1.-Omega_m-Omega_lambda-Omega_r;
		rho_crit = 3*H0*H0/(8*pi*G);
	}
	}
	else
	{
		printf("Running classical gravitational N-body simulation.\n");
		/*for(i=0;i<N;i++)//Every particle has its own mass
                {
			printf("M[%i] = %.10f\n", i, M[i]);
                }*/
	}
	//Searching the minimal mass particle
	M_min = M[0];
	for(i=0;i<N;i++)
	{
		if(M_min>M[i])
		{
			M_min = M[i];
		}
	}
	rho_part = M_min/(4.0*pi*pow(ParticleRadi, 3.0) / 3.0);
	T=0.0;
	beta = ParticleRadi;
	a=a_start;//scalefactor	
	recalculate_softening();
	t_next = 0.;
	T = 0;
	REAL Delta_T_out;
	if(COSMOLOGY == 0)
		a=1;//scalefactor
	int out_z_index = 0;
	//Calculating initial time
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
	//Timing
	REAL SIM_start_time = (REAL) clock () / (REAL) CLOCKS_PER_SEC;
	REAL SIM_omp_start_time = omp_get_wtime();
	//Timing

	//Initial force calculation
	if(IS_PERIODIC < 2)
	{
		forces_old(x, F);
	}
	if(IS_PERIODIC == 2)
	{
		forces_old_periodic(x, F);
	}
	
	//The simulation is starting...
	if(COSMOLOGY == 1 && COMOVING_INTEGRATION == 1)
	{
	a_prev1 = friedman_solver_step(a, -1*h, Omega_lambda, Omega_r, Omega_m, Omega_k, H0);
	a_prev2 = friedman_solver_step(a_prev1, -1*h, Omega_lambda, Omega_r, Omega_m, Omega_k, H0);
	}
	else
	{
	a_prev1 = a;
	a_prev2 = a;
	}
	h_prev = h;
	//Calculating the initial Hubble parameter, using the Friedmann-equations
	Hubble_tmp = H0*sqrt(Omega_m*pow(a, -3)+Omega_r*pow(a, -4)+Omega_lambda+Omega_k*pow(a, -2));
	Hubble_param = H0*sqrt(Omega_m*pow(a, -3)+Omega_r*pow(a, -4)+Omega_lambda+Omega_k*pow(a, -2));
	printf("Initial Hubble-parameter from the cosmological parameters:\nH(z=%lf) = %lfkm/s/Mpc\n\n", 1.0/a-1.0, Hubble_param*20.7386814448645);
	if(COSMOLOGY == 0 || COMOVING_INTEGRATION == 0)
	{
		Hubble_param = 0;
	}
	printf("The simulation is starting...\n");
	REAL T_prev,Hubble_param_prev;
	T_prev = T;
	Hubble_param_prev = Hubble_param;
	for(t=0; a_tmp<a_max; t++)
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
		Hubble_param_prev = Hubble_param;
		T_prev = T;
		T = T+h;
		step(x, F);
		Log_write(x);	//Writing logfile

		if(OUTPUT_FORMAT == 0)
		{
			if(T > t_next)
			{
				kiiras(x);
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
				kiiras(x);
				if(REDSHIFT_CONE == 1)
				{
					write_redshift_cone(x, r_bin_limits, out_z_index, CONE_ALL);
				}
				out_z_index++;
				t_next = out_list[out_z_index];
				if(MIN_REDSHIFT>t_next || (1.0/a-1.0 < t_next) )
				{
					CONE_ALL = 1;
					if((1.0/a-1.0 < t_next))
					{
						printf("z=%lf < z_next=%lf\n", 1.0/a-1.0, t_next);
						printf("Warning: the length of timestep is larger than the distance between output redshifts. After this point the z=0 coordinates will be written out with redshifts taken from the input file. This can cause inconsistencies, if the redshifts are not low enough.\n");
					}
					t_next = 0.0;
				}
				printf("z= %f\tz_bin= %lf\tt = %f Gy\n\th=%f Gy\n", 1/a-1.0,out_list[out_z_index-1], T*47.1482347621227, h*47.1482347621227);
			}
		}
		//Changing timestep length
		h_prev = h;
		if(h<h_max || h>h_min || mean_err/errmax>1)
		{
			h = (double) pow(2*mean_err*beta/errmax, 0.5);
		}

		if(h<h_min)
		{
			h=h_min;
		}
		if(h>h_max)
		{
			h=h_max;
		}
	}
	if(OUTPUT_FORMAT == 0)
	{
		kiiras(x); //writing output
	}
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
	printf("\nAt a = %lf state, with linear interpolation:\n",a_max);
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
	printf("Running time of the simulation:\n");
	//Timing
	REAL SIM_end_time = (REAL) clock () / (REAL) CLOCKS_PER_SEC;
	REAL SIM_omp_end_time = omp_get_wtime();
	//Timing
	printf("CPU time = %lfs\n", SIM_end_time-SIM_start_time);
	printf("RUN time = %lfs\n", SIM_omp_end_time-SIM_omp_start_time);
	return 0;
}

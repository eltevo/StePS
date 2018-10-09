#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <math.h>
#include <algorithm>
#include <sys/types.h>
#include <sys/stat.h>
#include "mpi.h"
#include "global_variables.h"

int file_exist(char *file_name)
{
	struct stat file_stat;
	stat(file_name, &file_stat);
	return S_ISREG(file_stat.st_mode);
}

int dir_exist(char *dir_name)
{
	struct stat dir_stat;
	stat(dir_name, &dir_stat);
	return S_ISDIR(dir_stat.st_mode);
}

int measure_N_part_from_ascii_snapshot(char * filename)
{
	int lines = 0;
	FILE *inputfile = fopen(filename, "r");
	while(EOF != (fscanf(inputfile,"%*[^\n]"), fscanf(inputfile,"%*c")))
		++lines;
	fclose(inputfile);
	printf("Number of particles\t\t%i\n\n", lines);
	return lines;
}

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
	//Reading the velocities
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
	if(file_exist(OUT_LST) == 0)
	{
		fprintf(stderr, "Error: The %s output list does not exist!\n", OUT_LST);
		return (-1);
	}
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
	if(OUTPUT_FORMAT == 1)
		out_list = (double*)malloc((size+1)*sizeof(double));
	else
		out_list = (double*)malloc((size)*sizeof(double));
	int offset;
	for(i=0; i<size; i++)
	{
		sscanf(buffer, "%lf%n", &out_list[i], &offset);
		buffer += offset;
	}
	out_list[size] = 1.0/a_max-1.0;
	if(OUTPUT_FORMAT == 1)
	{
		out_list[size] = 1.0/a_max-1.0;
		std::sort(out_list, out_list+size, std::greater<double>());
		out_list_size = size;
	}
	else
	{
		std::sort(out_list, out_list+size-1, std::less<double>());
		out_list_size = size-1;
		for(i=0; i<out_list_size; i++)
			out_list[i] /= UNIT_T; //converting input Gy to internal units
	}
	size = 0;
	if(REDSHIFT_CONE == 1)
	{
		//reading the limist of the comoving distance bins
		if(file_exist(BIN_LIST) == 0)
		{
			fprintf(stderr, "Error: The %s file does not exist!\nThis is used in redshift cone simulations, and the IC generator should have generated it.\n", BIN_LIST);
			return (-1);
		}
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
		printf("Saving: z=%f:\tDc<%fMpc\n%s\n redshift cone.\n", t_next, limits[z_index-delta_z_index+1], filename);
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
					fprintf(redshiftcone_file, "%.16f\t",v[3*i+j]*UNIT_V); //km/s output
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

void write_ascii_snapshot(REAL* x, REAL *v)
{
	int i,k;
	char A[20];
	if(COSMOLOGY == 1)
	{
		if(OUTPUT_FORMAT == 0)
		{
			sprintf(A, "%d", (int)(round(100*t_next*UNIT_T)));
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
			printf("Saving: t= %f, file: \"%st%s.dat\" \n", t_next*UNIT_T, OUT_DIR, A);
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
			//writing out zero speeds if glassmaking is on
			#ifdef GLASS_MAKING
			fprintf(coordinate_file, "%.16f\t", 0.0);
			#else
			fprintf(coordinate_file, "%.16f\t",v[3*i+k]*sqrt(a)*UNIT_V); //km/s * sqrt(a) output
			#endif
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
	fprintf(LOGFILE, "%.15f\t%e\t%e\t%.15f\t%.15f\t%.15f\t%.15f\t%.10f\n", T*UNIT_T, errmax, h*UNIT_T, a, 1.0/a-1.0, Hubble_param*UNIT_V, Decel_param, Omega_m_eff);
	fclose(LOGFILE);
}

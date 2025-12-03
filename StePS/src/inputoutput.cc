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

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cstring>
#include <math.h>
#include <algorithm>
#include <sys/types.h>
#include <sys/stat.h>
#include "mpi.h"
#include "global_variables.h"

#ifdef HAVE_HDF5
#include <hdf5.h>

void write_header_attributes_in_hdf5(hid_t handle);
#endif

//Functions for reading GADGET2 format IC
int gadget_format_conversion(bool allocate_memory);
int load_snapshot(char *fname, int files);
int allocate_memory(void);
int reordering(void);

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
	return lines;
}

#if COSMOPARAM==-1
void read_expansion_history(char* filename)
{
	int i,j;
	N_expansion_tab = measure_N_part_from_ascii_snapshot(filename);
	//Allocating memory for the tabulated expansion history
	expansion_tab =(double**) malloc(N_expansion_tab*sizeof(double*));
	for (i = 0; i < N_expansion_tab; i++)
		expansion_tab[i] = (double*)malloc(3*sizeof(double));
	//reading the data from the ASCII file
	FILE *exp_file = fopen(filename, "r");
	printf("\nReading the expansion history from the %s file...\n", filename);
	for(i=0; i<N_expansion_tab; i++)
	{
		//Reading particle coordinates
		for(j=0; j<3; j++)
		{
			fscanf(exp_file, "%lf", &expansion_tab[i][j]);
			if(N_expansion_tab<10)
				printf("%f\t", expansion_tab[i][j]);
			if(j==0)
			{
				expansion_tab[i][j] /= UNIT_T; //converting time from Gy to internal units.
			}
			else if(j==2)
			{
				expansion_tab[i][j] /= UNIT_V; //converting the Hubble parameter from km/s/Mpc to internal units.
			}
		}
		if(N_expansion_tab<10)
			printf("\n");
	}
	fclose(exp_file);
	return;
}
#endif

void read_ascii_ic(FILE *ic_file, int N, bool allocate_memory)
{
	// This function reads the initial conditions from an ASCII file.
	// Input parameters:
	// ic_file: pointer to the input file
	// N: number of particles
	// allocate_memory: if true, allocates memory for the particle data arrays. if false, assumes that the memory is already allocated.
	int i,j;
	if(allocate_memory)
	{
		//Allocating memory for the coordinates
		if(!(x = (REAL*)malloc(3*N*sizeof(REAL))))
		{
			fprintf(stderr, "MPI task %i: failed to allocate memory for x.\n", rank);
			exit(-2);
		}
		//Allocating memory for the velocities
		if(!(v = (REAL*)malloc(3*N*sizeof(REAL))))
		{
			fprintf(stderr, "MPI task %i: failed to allocate memory for v.\n", rank);
			exit(-2);
		}
		//Allocating memory for the forces
		if(!(F = (REAL*)malloc(3*N*sizeof(REAL))))
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
			if(N<10)
				printf("%f\t", x[3*i + j]);
		}
		//Reading the velocities
		for(j=0; j<3; j++)
		{
			#ifdef USE_SINGLE_PRECISION
			fscanf(ic_file, "%f", &v[3*i + j]);
			#else
			fscanf(ic_file, "%lf", &v[3*i + j]);
			#endif
			if(N<10)
				printf("%f\t", v[3*i + j]);
		}
		//Reading particle masses
		#ifdef USE_SINGLE_PRECISION
		fscanf(ic_file, "%f", & M[i]);
		#else
		fscanf(ic_file, "%lf", & M[i]);
		#endif
		if(N<10)
			printf("%f\n", M[i]);

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
	if(OUTPUT_TIME_VARIABLE == 1)
		out_list = (double*)malloc((size+1)*sizeof(double));
	else
		out_list = (double*)malloc((size)*sizeof(double));
	int offset;
	for(i=0; i<size; i++)
	{
		sscanf(buffer, "%lf%n", &out_list[i], &offset);
		buffer += offset;
	}
	if(OUTPUT_TIME_VARIABLE == 1)
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
	REAL H0_dimless;
	if(H0_INDEPENDENT_UNITS != 0)
	{
		 H0_dimless = H0*UNIT_V/100.0;
	}
	else
	{
		H0_dimless = 1.0;
	}
	if(OUTPUT_FORMAT == 0)
		if(snprintf(filename, sizeof(filename), "%sredshift_cone.dat", OUT_DIR) < 0)
		{
			fprintf(stderr, "Error: The output file name truncated.\nAborting.\n");
			abort();
		}
	#ifdef HAVE_HDF5
	if(OUTPUT_FORMAT == 2)
		if(snprintf(filename, sizeof(filename), "%sredshift_cone.hdf5", OUT_DIR) < 0)
		{
			fprintf(stderr, "Error: The output file name truncated.\nAborting.\n");
			abort();
		}
	#endif
	if(ALL == 0)
		printf("Saving: z=%f:\t%fMpc<Dc<%fMpc bin of the\n%s redshift cone.\ndelta_z_index = %i\n",out_list[z_index], limits[z_index+1], limits[z_index-delta_z_index+1], filename, delta_z_index);
	else
		printf("Saving: z=%f:\tDc<%fMpc\n%s\n redshift cone.\n", t_next, limits[z_index-delta_z_index+1], filename);
	if(OUTPUT_FORMAT == 0)
	{
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
						fprintf(redshiftcone_file, "%.16f\t",x[3*i+j]*H0_dimless);
					}
					for(j=0; j<3; j++)
					{
						fprintf(redshiftcone_file, "%.16f\t",v[3*i+j]*UNIT_V); //output units in km/s
					}
					fprintf(redshiftcone_file, "%.16f\t%.16f\t%.16f\t%i\n", M[i]*H0_dimless, COMOVING_DISTANCE, out_list[z_index], i);
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
						fprintf(redshiftcone_file, "%.16f\t",x[3*i+j]*H0_dimless);
					}
					for(j=0; j<3; j++)
					{
						fprintf(redshiftcone_file, "%.16f\t",v[3*i+j]*UNIT_V); //km/s output
					}
					fprintf(redshiftcone_file, "%.16f\t%.16f\t%.16f\t%i\n", M[i]*H0_dimless, COMOVING_DISTANCE*H0_dimless, z_write, i);
					IN_CONE[i] = true;
					COUNT++;
				}
			}
		}
		fclose(redshiftcone_file);
	}
	#ifdef HAVE_HDF5
	char buf[500];
	REAL bufvec[3];
	unsigned long long int *ID;
	if(!(ID = (unsigned long long int *)malloc(1*sizeof(unsigned long long int))))
	{
		fprintf(stderr, "MPI task %i: failed to allocate memory for ID.\n", rank);
		exit(-2);
	}
	REAL *bufscalar;
	bufscalar = (REAL *)malloc(1*sizeof(REAL));
	int hdf5_rank;
	if(OUTPUT_FORMAT == 2)
	{
		hid_t redshiftcone = 0;
		hid_t hdf5_grp[6]; //group for the data types (only DM are used in this version)
		hid_t headergrp = 0;
		hid_t dataspace_in_file = 0;
		hid_t dataset = 0;
		hid_t datatype = 0;
		hid_t dataspace_memory;
		hsize_t dims[2], count[2], start[2];
		if(HDF5_redshiftcone_firstshell == 1)
		{
			//this is the first shell of the HDF5 format redshift cone.
			//Creating and saving the empty redshiftcone file.
			redshiftcone = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
			//Output file created. Creating the header group, and write the header into the file
			headergrp = H5Gcreate(redshiftcone, "/Header", 0, H5P_DEFAULT,H5P_DEFAULT);
			//This code only uses DM particles, so type=1 (we do this for gadget compatibility)
			int type = 1;
			sprintf(buf, "/PartType%d", type);
			hdf5_grp[type] = H5Gcreate(redshiftcone, buf, 0, H5P_DEFAULT,H5P_DEFAULT);
			write_header_attributes_in_hdf5(headergrp);
			//header written.
			//Creating dataspace for the particle positions
			dims[0] = N;
			dims[1] = 3; //hdf5_rank = 2 [if dims[1] = 1: 1; else: 2]
			hdf5_rank = 2;
			#ifdef USE_SINGLE_PRECISION
				datatype = H5Tcopy(H5T_NATIVE_FLOAT); //coordinates saved as float
			#else
				datatype = H5Tcopy(H5T_NATIVE_DOUBLE); //coordinates saved as double
			#endif
			strcpy(buf, "Coordinates");
			dataspace_in_file = H5Screate_simple(hdf5_rank, dims, NULL);
			dataset = H5Dcreate(hdf5_grp[type], buf, datatype, dataspace_in_file, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
			H5Dclose(dataset);
			H5Sclose(dataspace_in_file);
			H5Tclose(datatype);
			//Creating dataspace for the particle velocities
			dims[0] = N;
			dims[1] = 3; //hdf5_rank = 2 [if dims[1] = 1: 1; else: 2]
			hdf5_rank = 2;
			#ifdef USE_SINGLE_PRECISION
				datatype = H5Tcopy(H5T_NATIVE_FLOAT); //Velocities saved as float
			#else
				datatype = H5Tcopy(H5T_NATIVE_DOUBLE); //Velocities saved as double
			#endif
			strcpy(buf, "Velocities");
			dataspace_in_file = H5Screate_simple(hdf5_rank, dims, NULL);
			dataset = H5Dcreate(hdf5_grp[type], buf, datatype, dataspace_in_file, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
			H5Dclose(dataset);
			H5Sclose(dataspace_in_file);
			H5Tclose(datatype);
			//Creating dataspace for the particle IDs
			dims[0] = N;
			dims[1] = 1; //hdf5_rank = 2 [if dims[1] = 1: 1; else: 2]
			hdf5_rank = 1;
			datatype = H5Tcopy(H5T_NATIVE_UINT64); //IDs are saved as unsigned 64 bit ints
			strcpy(buf, "ParticleIDs");
			dataspace_in_file = H5Screate_simple(hdf5_rank, dims, NULL);
			dataset = H5Dcreate(hdf5_grp[type], buf, datatype, dataspace_in_file, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
			H5Dclose(dataset);
			H5Sclose(dataspace_in_file);
			H5Tclose(datatype);
			//Creating dataspace for the particle Masses
			dims[0] = N;
			dims[1] = 1; //hdf5_rank = 2 [if dims[1] = 1: 1; else: 2]
			hdf5_rank = 1;
			#ifdef USE_SINGLE_PRECISION
				datatype = H5Tcopy(H5T_NATIVE_FLOAT); //Masses saved as float
			#else
				datatype = H5Tcopy(H5T_NATIVE_DOUBLE); //Masses saved as double
			#endif
			strcpy(buf, "Masses");
			dataspace_in_file = H5Screate_simple(hdf5_rank, dims, NULL);
			dataset = H5Dcreate(hdf5_grp[type], buf, datatype, dataspace_in_file, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
			H5Dclose(dataset);
			H5Sclose(dataspace_in_file);
			H5Tclose(datatype);
			//Creating dataspace for the particle Comoving distance
			dims[0] = N;
			dims[1] = 1; //hdf5_rank = 2 [if dims[1] = 1: 1; else: 2]
			hdf5_rank = 1;
			#ifdef USE_SINGLE_PRECISION
				datatype = H5Tcopy(H5T_NATIVE_FLOAT); //Masses saved as float
			#else
				datatype = H5Tcopy(H5T_NATIVE_DOUBLE); //Masses saved as double
			#endif
			strcpy(buf, "ComovingDistances");
			dataspace_in_file = H5Screate_simple(hdf5_rank, dims, NULL);
			dataset = H5Dcreate(hdf5_grp[type], buf, datatype, dataspace_in_file, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
			H5Dclose(dataset);
			H5Sclose(dataspace_in_file);
			H5Tclose(datatype);
			//Creating dataspace for the particle redshifts
			dims[0] = N;
			dims[1] = 1; //hdf5_rank = 2 [if dims[1] = 1: 1; else: 2]
			hdf5_rank = 1;
			#ifdef USE_SINGLE_PRECISION
				datatype = H5Tcopy(H5T_NATIVE_FLOAT); //Masses saved as float
			#else
				datatype = H5Tcopy(H5T_NATIVE_DOUBLE); //Masses saved as double
			#endif
			strcpy(buf, "Redshifts");
			dataspace_in_file = H5Screate_simple(hdf5_rank, dims, NULL);
			dataset = H5Dcreate(hdf5_grp[type], buf, datatype, dataspace_in_file, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
			H5Dclose(dataset);
			H5Sclose(dataspace_in_file);
			H5Tclose(datatype);

			H5Fclose(redshiftcone);
			HDF5_redshiftcone_firstshell = 0;
		}
		//Opening the redshiftcone.hdf5 file
		redshiftcone = H5Fopen(filename, H5F_ACC_RDWR, H5P_DEFAULT);
		if(ALL == 0)
		{
			for(i=0; i<N; i++)
			{
				COMOVING_DISTANCE = sqrt(x[3*i]*x[3*i] + x[3*i+1]*x[3*i+1] +x[3*i+2]*x[3*i+2]);
				if(limits[z_index+1] <= COMOVING_DISTANCE && IN_CONE[i] == false )
				{
					//writing out the i-th particle's coordiate vector
					dataset = H5Dopen2(redshiftcone, "/PartType1/Coordinates", H5P_DEFAULT);
					dataspace_in_file = H5Dget_space(dataset);
					datatype =  H5Dget_type(dataset);
					dims[0] = 1;
					dims[1] = 3; //hdf5_rank = 2 [if dims[1] = 1: 1; else: 2]
					hdf5_rank = 2;
					start[0] = N_redshiftcone;
					start[1] = 0;
					count[0] = 1;
					count[1] = 3;
					dataspace_memory = H5Screate_simple(hdf5_rank, dims, NULL);
					bufvec[0] = x[3*i]*H0_dimless;
					bufvec[1] = x[3*i+1]*H0_dimless;
					bufvec[2] = x[3*i+2]*H0_dimless;
					H5Sselect_hyperslab(dataspace_in_file, H5S_SELECT_SET, start, NULL, count, NULL);
					H5Dwrite(dataset, datatype, dataspace_memory, dataspace_in_file, H5P_DEFAULT, bufvec);
					H5Dclose(dataset);
					H5Sclose(dataspace_memory);
					H5Sclose(dataspace_in_file);
					H5Tclose(datatype);

					//writing out the i-th particle's velocity vector
					dataset = H5Dopen2(redshiftcone, "/PartType1/Velocities", H5P_DEFAULT);
					dataspace_in_file = H5Dget_space(dataset);
					datatype =  H5Dget_type(dataset);
					dims[0] = 1;
					dims[1] = 3; //hdf5_rank = 2 [if dims[1] = 1: 1; else: 2]
					hdf5_rank = 2;
					start[0] = N_redshiftcone;
					start[1] = 0;
					count[0] = 1;
					count[1] = 3;
					dataspace_memory = H5Screate_simple(hdf5_rank, dims, NULL);
					bufvec[0] = v[3*i]*(REAL)UNIT_V;
					bufvec[1] = v[3*i+1]*(REAL)UNIT_V;
					bufvec[2] = v[3*i+2]*(REAL)UNIT_V;
					H5Sselect_hyperslab(dataspace_in_file, H5S_SELECT_SET, start, NULL, count, NULL);
					H5Dwrite(dataset, datatype, dataspace_memory, dataspace_in_file, H5P_DEFAULT, bufvec);
					H5Dclose(dataset);
					H5Sclose(dataspace_memory);
					H5Sclose(dataspace_in_file);
					H5Tclose(datatype);

					//writing out the i-th particle's ID
					dataset = H5Dopen2(redshiftcone, "/PartType1/ParticleIDs", H5P_DEFAULT);
                                        dataspace_in_file = H5Dget_space(dataset);
                                        datatype =  H5Dget_type(dataset);
                                        dims[0] = 1;
                                        dims[1] = 1; //hdf5_rank = 1 [if dims[1] = 1: 1; else: 2]
                                        hdf5_rank = 1;
                                        start[0] = N_redshiftcone;
                                        start[1] = 0;
                                        count[0] = 1;
                                        count[1] = 1;
                                        dataspace_memory = H5Screate_simple(hdf5_rank, dims, NULL);
					ID[0] = i;
                                        H5Sselect_hyperslab(dataspace_in_file, H5S_SELECT_SET, start, NULL, count, NULL);
                                        H5Dwrite(dataset, datatype, dataspace_memory, dataspace_in_file, H5P_DEFAULT, ID);
                                        H5Dclose(dataset);
                                        H5Sclose(dataspace_memory);
                                        H5Sclose(dataspace_in_file);
                                        H5Tclose(datatype);

					//writing out the i-th particle's Mass
					dataset = H5Dopen2(redshiftcone, "/PartType1/Masses", H5P_DEFAULT);
					dataspace_in_file = H5Dget_space(dataset);
					datatype =  H5Dget_type(dataset);
					dims[0] = 1;
					dims[1] = 1; //hdf5_rank = 1 [if dims[1] = 1: 1; else: 2]
					hdf5_rank = 1;
					start[0] = N_redshiftcone;
					start[1] = 0;
					count[0] = 1;
					count[1] = 1;
					dataspace_memory = H5Screate_simple(hdf5_rank, dims, NULL);
					bufscalar[0] = M[i]*H0_dimless;
					H5Sselect_hyperslab(dataspace_in_file, H5S_SELECT_SET, start, NULL, count, NULL);
					H5Dwrite(dataset, datatype, dataspace_memory, dataspace_in_file, H5P_DEFAULT, bufscalar);
					H5Dclose(dataset);
					H5Sclose(dataspace_memory);
					H5Sclose(dataspace_in_file);
					H5Tclose(datatype);

					//writing out the i-th particle's comoving distance from the center
					dataset = H5Dopen2(redshiftcone, "/PartType1/ComovingDistances", H5P_DEFAULT);
					dataspace_in_file = H5Dget_space(dataset);
					datatype =  H5Dget_type(dataset);
					dims[0] = 1;
					dims[1] = 1; //hdf5_rank = 1 [if dims[1] = 1: 1; else: 2]
					hdf5_rank = 1;
					start[0] = N_redshiftcone;
					start[1] = 0;
					count[0] = 1;
					count[1] = 1;
					dataspace_memory = H5Screate_simple(hdf5_rank, dims, NULL);
					bufscalar[0] = COMOVING_DISTANCE*H0_dimless;
					H5Sselect_hyperslab(dataspace_in_file, H5S_SELECT_SET, start, NULL, count, NULL);
					H5Dwrite(dataset, datatype, dataspace_memory, dataspace_in_file, H5P_DEFAULT, bufscalar);
					H5Dclose(dataset);
					H5Sclose(dataspace_memory);
					H5Sclose(dataspace_in_file);
					H5Tclose(datatype);

					//writing out the i-th particle's redshift
					dataset = H5Dopen2(redshiftcone, "/PartType1/Redshifts", H5P_DEFAULT);
					dataspace_in_file = H5Dget_space(dataset);
					datatype =  H5Dget_type(dataset);
					dims[0] = 1;
					dims[1] = 1; //hdf5_rank = 1 [if dims[1] = 1: 1; else: 2]
					hdf5_rank = 1;
					start[0] = N_redshiftcone;
					start[1] = 0;
					count[0] = 1;
					count[1] = 1;
					dataspace_memory = H5Screate_simple(hdf5_rank, dims, NULL);
					bufscalar[0] = out_list[z_index];
					H5Sselect_hyperslab(dataspace_in_file, H5S_SELECT_SET, start, NULL, count, NULL);
					H5Dwrite(dataset, datatype, dataspace_memory, dataspace_in_file, H5P_DEFAULT, bufscalar);
					H5Dclose(dataset);
					H5Sclose(dataspace_memory);
					H5Sclose(dataspace_in_file);
					H5Tclose(datatype);

					IN_CONE[i] = true;
					N_redshiftcone++;
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
					//writing out the i-th particle's coordiate vector
					dataset = H5Dopen2(redshiftcone, "/PartType1/Coordinates", H5P_DEFAULT);
					dataspace_in_file = H5Dget_space(dataset);
					datatype =  H5Dget_type(dataset);
					dims[0] = 1;
					dims[1] = 3; //hdf5_rank = 2 [if dims[1] = 1: 1; else: 2]
					hdf5_rank = 2;
					start[0] = N_redshiftcone;
					start[1] = 0;
					count[0] = 1;
					count[1] = 3;
					dataspace_memory = H5Screate_simple(hdf5_rank, dims, NULL);
					bufvec[0] = x[3*i];
					bufvec[1] = x[3*i+1];
 					bufvec[2] = x[3*i+2];
					H5Sselect_hyperslab(dataspace_in_file, H5S_SELECT_SET, start, NULL, count, NULL);
					H5Dwrite(dataset, datatype, dataspace_memory, dataspace_in_file, H5P_DEFAULT, bufvec);
					H5Dclose(dataset);
					H5Sclose(dataspace_memory);
					H5Sclose(dataspace_in_file);
					H5Tclose(datatype);

					//writing out the i-th particle's velocity vector
					dataset = H5Dopen2(redshiftcone, "/PartType1/Velocities", H5P_DEFAULT);
					dataspace_in_file = H5Dget_space(dataset);
					datatype =  H5Dget_type(dataset);
					dims[0] = 1;
 					dims[1] = 3; //hdf5_rank = 2 [if dims[1] = 1: 1; else: 2]
					hdf5_rank = 2;
					start[0] = N_redshiftcone;
					start[1] = 0;
					count[0] = 1;
					count[1] = 3;
					dataspace_memory = H5Screate_simple(hdf5_rank, dims, NULL);
					bufvec[0] = v[3*i]*(REAL)UNIT_V;
					bufvec[1] = v[3*i+1]*(REAL)UNIT_V;
					bufvec[2] = v[3*i+2]*(REAL)UNIT_V;
					H5Sselect_hyperslab(dataspace_in_file, H5S_SELECT_SET, start, NULL, count, NULL);
					H5Dwrite(dataset, datatype, dataspace_memory, dataspace_in_file, H5P_DEFAULT, bufvec);
					H5Dclose(dataset);
					H5Sclose(dataspace_memory);
					H5Sclose(dataspace_in_file);
					H5Tclose(datatype);

					//writing out the i-th particle's ID
					dataset = H5Dopen2(redshiftcone, "/PartType1/ParticleIDs", H5P_DEFAULT);
					dataspace_in_file = H5Dget_space(dataset);
					datatype =  H5Dget_type(dataset);
					dims[0] = 1;
					dims[1] = 1; //hdf5_rank = 1 [if dims[1] = 1: 1; else: 2]
					hdf5_rank = 1;
					start[0] = N_redshiftcone;
					start[1] = 0;
					count[0] = 1;
					count[1] = 1;
					dataspace_memory = H5Screate_simple(hdf5_rank, dims, NULL);
					ID[0] = i;
					H5Sselect_hyperslab(dataspace_in_file, H5S_SELECT_SET, start, NULL, count, NULL);
					H5Dwrite(dataset, datatype, dataspace_memory, dataspace_in_file, H5P_DEFAULT, ID);
					H5Dclose(dataset);
					H5Sclose(dataspace_memory);
					H5Sclose(dataspace_in_file);
					H5Tclose(datatype);

					//writing out the i-th particle's Mass
					dataset = H5Dopen2(redshiftcone, "/PartType1/Masses", H5P_DEFAULT);
					dataspace_in_file = H5Dget_space(dataset);
					datatype =  H5Dget_type(dataset);
					dims[0] = 1;
					dims[1] = 1; //hdf5_rank = 1 [if dims[1] = 1: 1; else: 2]
					hdf5_rank = 1;
					start[0] = N_redshiftcone;
					start[1] = 0;
					count[0] = 1;
					count[1] = 1;
					dataspace_memory = H5Screate_simple(hdf5_rank, dims, NULL);
					bufscalar[0] = M[i];
					H5Sselect_hyperslab(dataspace_in_file, H5S_SELECT_SET, start, NULL, count, NULL);
					H5Dwrite(dataset, datatype, dataspace_memory, dataspace_in_file, H5P_DEFAULT, bufscalar);
					H5Dclose(dataset);
					H5Sclose(dataspace_memory);
					H5Sclose(dataspace_in_file);
					H5Tclose(datatype);

					//writing out the i-th particle's comoving distance from the center
					dataset = H5Dopen2(redshiftcone, "/PartType1/ComovingDistances", H5P_DEFAULT);
					dataspace_in_file = H5Dget_space(dataset);
					datatype =  H5Dget_type(dataset);
					dims[0] = 1;
					dims[1] = 1; //hdf5_rank = 1 [if dims[1] = 1: 1; else: 2]
					hdf5_rank = 1;
					start[0] = N_redshiftcone;
					start[1] = 0;
					count[0] = 1;
					count[1] = 1;
					dataspace_memory = H5Screate_simple(hdf5_rank, dims, NULL);
					bufscalar[0] = COMOVING_DISTANCE;
					H5Sselect_hyperslab(dataspace_in_file, H5S_SELECT_SET, start, NULL, count, NULL);
					H5Dwrite(dataset, datatype, dataspace_memory, dataspace_in_file, H5P_DEFAULT, bufscalar);
					H5Dclose(dataset);
					H5Sclose(dataspace_memory);
					H5Sclose(dataspace_in_file);
					H5Tclose(datatype);

					//writing out the i-th particle's redshift
					dataset = H5Dopen2(redshiftcone, "/PartType1/Redshifts", H5P_DEFAULT);
					dataspace_in_file = H5Dget_space(dataset);
					datatype =  H5Dget_type(dataset);
					dims[0] = 1;
					dims[1] = 1; //hdf5_rank = 1 [if dims[1] = 1: 1; else: 2]
					hdf5_rank = 1;
					start[0] = N_redshiftcone;
					start[1] = 0;
					count[0] = 1;
					count[1] = 1;
					dataspace_memory = H5Screate_simple(hdf5_rank, dims, NULL);
					bufscalar[0] = z_write;
					H5Sselect_hyperslab(dataspace_in_file, H5S_SELECT_SET, start, NULL, count, NULL);
					H5Dwrite(dataset, datatype, dataspace_memory, dataspace_in_file, H5P_DEFAULT, bufscalar);
					H5Dclose(dataset);
					H5Sclose(dataspace_memory);
					H5Sclose(dataspace_in_file);
					H5Tclose(datatype);

					IN_CONE[i] = true;
					N_redshiftcone++;
					COUNT++;
				}
			}
		}
		H5Fclose(redshiftcone);

	}
	#endif
	printf("%i particles were written out.\n", COUNT);
	COUNT = 0;

}

void write_ascii_snapshot(REAL* x, REAL *v)
{
	int i,k;
	char A[20];
	REAL H0_dimless;
	if(H0_INDEPENDENT_UNITS != 0)
	{
		 H0_dimless = H0*UNIT_V/100.0;
	}
	else
	{
		H0_dimless = 1.0;
	}
	if(COSMOLOGY == 1)
	{
		if(OUTPUT_TIME_VARIABLE == 0)
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
	if(OUTPUT_TIME_VARIABLE == 0)
	{
		if(snprintf(filename, sizeof(filename), "%st%s.dat", OUT_DIR, A) < 0)
		{
			fprintf(stderr, "Error: The output file name truncated.\nAborting.\n");
			abort();
		}
	}
	else
	{
		if(snprintf(filename, sizeof(filename), "%sz%s.dat", OUT_DIR, A) < 0)
		{
			fprintf(stderr, "Error: The output file name truncated.\nAborting.\n");
			abort();
		}
	}
	if(COSMOLOGY == 0)
	{
			printf("Saving the  \"%st%s.dat\" snapshot file... \nt = %.15f", OUT_DIR, A, T);
	}
	else
	{
		if(OUTPUT_TIME_VARIABLE == 0)
		{
			printf("Saving the \"%st%s.dat\" snapshot file... \nt = %.15fGy\nz = %.15f\n", OUT_DIR, A, T*UNIT_T, 1.0/a-1.0);
		}
		else
		{
			printf("Saving the \"%sz%s.dat\" snapshot file... \nt = %.15fGy\nz = %.15f\n", OUT_DIR, A, T*UNIT_T, 1.0/a-1.0);
		}
	}
	FILE *coordinate_file;
	coordinate_file = fopen(filename, "w");

	for(i=0; i<N; i++)
	{
		for(k=0; k<3; k++)
		{
			fprintf(coordinate_file, "%.16f\t",x[3*i+k]*H0_dimless);
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
		fprintf(coordinate_file, "%.16f\t",M[i]*H0_dimless);
		fprintf(coordinate_file, "\n");
	}

	fclose(coordinate_file);
}

void Log_write() //Writing logfile
{
	FILE *LOGFILE;
	char A[] = "Logfile.dat";
	char filename[0x100];
	if(snprintf(filename, sizeof(filename), "%s%s", OUT_DIR, A) < 0)
	{
		fprintf(stderr, "Error: The name of the logfile got truncated.\nAborting.\n");
		abort();
	}
	LOGFILE = fopen(filename, "a");
	fprintf(LOGFILE, "%.15f\t%e\t%e\t%.15f\t%.15f\t%.15f\t%.15f\t%.10f\n", T*UNIT_T, errmax, h*UNIT_T, a, 1.0/a-1.0, Hubble_param*UNIT_V, Decel_param, Omega_m_eff);
	fclose(LOGFILE);
}

#ifdef HAVE_HDF5

void read_hdf5_ic(char *ic_file, bool allocate_memory)
{
	//Reading initial conditions from an HDF5 file
	// Input parameters:
	// ic_file: the name of the HDF5 file containing the initial conditions
	// allocate_memory: if true, memory will be allocated for the particle data arrays. if false, the function will assume that the memory is already allocated.
	hid_t attr_id = 0;
	hid_t group = 0;
	hid_t dataset = 0;
	hid_t datatype = 0;
	hid_t dataspace_in_file = 0;
	hid_t IC = 0;
	int Nbuf[6];
	printf("Reading the %s IC file...\n", ic_file);
	IC = H5Fopen(ic_file, H5F_ACC_RDONLY, H5P_DEFAULT);
	//reading the total number of particles from the header
	group = H5Gopen2(IC,"Header", H5P_DEFAULT);
	attr_id = H5Aopen(group, "NumPart_ThisFile", H5P_DEFAULT);
	H5Aread(attr_id,  H5T_NATIVE_INT, Nbuf);
	N = Nbuf[1];
	printf("\tThe number of particles:\t%i\n", N);
	H5Aclose(attr_id);
	H5Gclose(group);
	if(allocate_memory)
	{
		//Allocating memory
		printf("\tAllocating memory for the particle data arrays...\n");
		//Allocating memory for the coordinates
		if(!(x = (REAL*)malloc(3*N*sizeof(REAL))))
		{
			fprintf(stderr, "MPI task %i: failed to allocate memory for x.\n", rank);
			exit(-2);
		}
		//Allocating memory for the velocities
		if(!(v = (REAL*)malloc(3*N*sizeof(REAL))))
		{
			fprintf(stderr, "MPI task %i: failed to allocate memory for v.\n", rank);
			exit(-2);
		}
		//Allocating memory for the forces
		if(!(F = (REAL*)malloc(3*N*sizeof(REAL))))
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
	//reading the particle coordinates
	printf("\tReading /PartType1/Coordinates\n");
	dataset = H5Dopen2(IC, "/PartType1/Coordinates", H5P_DEFAULT);
        dataspace_in_file = H5Dget_space(dataset);
	datatype =  H5Dget_type(dataset);
#ifdef USE_SINGLE_PRECISION
	if(H5Tequal(datatype, H5T_NATIVE_FLOAT))
	{
		H5Dread(dataset, datatype, H5S_ALL, H5S_ALL, H5P_DEFAULT, x);
	}
	else
	{
		printf("\t\tData stored in doubles.\n");
		double* buffer;
		if(!(buffer = (double*)malloc(3*N*sizeof(double))))
		{
			fprintf(stderr, "MPI task %i: failed to allocate memory for x_buffer.\n", rank);
			exit(-2);
		}
		H5Dread(dataset, datatype, H5S_ALL, H5S_ALL, H5P_DEFAULT, buffer);
		for (int i = 0; i < 3*N; i++)
		{
			x[i] = (REAL) buffer[i];
		}
		free(buffer);
	}
#else
	if(H5Tequal(datatype, H5T_NATIVE_DOUBLE))
	{
		H5Dread(dataset, datatype, H5S_ALL, H5S_ALL, H5P_DEFAULT, x);
	}
	else
	{
		printf("\t\tData stored in floats.\n");
		float* buffer;
		if(!(buffer = (float*)malloc(3*N*sizeof(float))))
		{
			fprintf(stderr, "MPI task %i: failed to allocate memory for x_buffer.\n", rank);
			exit(-2);
		}
		H5Dread(dataset, datatype, H5S_ALL, H5S_ALL, H5P_DEFAULT, buffer);
		for (int i = 0; i < 3*N; i++)
		{
			x[i] = (REAL) buffer[i];
		}
		free(buffer);
	}
#endif
	H5Tclose(datatype);
	H5Sclose(dataspace_in_file);
	H5Dclose(dataset);
	//reading the particle velocities
	printf("\tReading /PartType1/Velocities\n");
	dataset = H5Dopen2(IC, "/PartType1/Velocities", H5P_DEFAULT);
	dataspace_in_file = H5Dget_space(dataset);
	datatype =  H5Dget_type(dataset);
#ifdef USE_SINGLE_PRECISION
	if(H5Tequal(datatype, H5T_NATIVE_FLOAT))
	{
		H5Dread(dataset, datatype, H5S_ALL, H5S_ALL, H5P_DEFAULT, v);
	}
	else
	{
		printf("\t\tData stored in doubles.\n");
		double* buffer;
		if(!(buffer = (double*)malloc(3*N*sizeof(double))))
		{
			fprintf(stderr, "MPI task %i: failed to allocate memory for v_buffer.\n", rank);
			exit(-2);
		}
		H5Dread(dataset, datatype, H5S_ALL, H5S_ALL, H5P_DEFAULT, buffer);
		for (int i = 0; i < 3*N; i++)
		{
			v[i] = (REAL) buffer[i];
		}
		free(buffer);
	}
#else
	if(H5Tequal(datatype, H5T_NATIVE_DOUBLE))
	{
		H5Dread(dataset, datatype, H5S_ALL, H5S_ALL, H5P_DEFAULT, v);
	}
	else
	{
		printf("\t\tData stored in floats.\n");
		float* buffer;
		if(!(buffer = (float*)malloc(3*N*sizeof(float))))
		{
			fprintf(stderr, "MPI task %i: failed to allocate memory for v_buffer.\n", rank);
			exit(-2);
		}
		H5Dread(dataset, datatype, H5S_ALL, H5S_ALL, H5P_DEFAULT, buffer);
		for (int i = 0; i < 3*N; i++)
		{
			v[i] = (REAL) buffer[i];
		}
		free(buffer);
		}
#endif
	H5Tclose(datatype);
	H5Sclose(dataspace_in_file);
	H5Dclose(dataset);
	//reading the particle masses
	// checking if the masses are stored in the file
	if (H5Lexists(IC, "/PartType1/Masses", H5P_DEFAULT) <= 0)
	{
		printf("\tMasses not found in the IC file. Assuming constant mass resolution in the full simulation volume.\n");
		// the mass resolution is stored in the header as the second value in the attribute: "/Header/MassTable"
		group = H5Gopen2(IC,"Header", H5P_DEFAULT);
		attr_id = H5Aopen(group, "MassTable", H5P_DEFAULT);
		REAL mass_table[6];
		datatype =  H5Aget_type(attr_id);
#ifdef USE_SINGLE_PRECISION
		if(H5Tequal(datatype, H5T_NATIVE_FLOAT))
		{
			H5Aread(attr_id,  H5T_NATIVE_FLOAT, &mass_table);
		}
		else
		{
			printf("\t\tData stored in doubles.\n");
			double* buffer;
			if(!(buffer = (double*)malloc(6*sizeof(double))))
			{
				fprintf(stderr, "MPI task %i: failed to allocate memory for mass_table_buffer.\n", rank);
				exit(-2);
			}
			H5Aread(attr_id,  H5T_NATIVE_DOUBLE, buffer);
			for (int i = 0; i < 6; i++)
			{
				mass_table[i] = (REAL) buffer[i];
			}
			free(buffer);
		}
#else
		if(H5Tequal(datatype, H5T_NATIVE_DOUBLE))
		{
			H5Aread(attr_id,  H5T_NATIVE_DOUBLE, &mass_table);
		}
		else
		{
			printf("\t\tData stored in floats.\n");
			float* buffer;
			if(!(buffer = (float*)malloc(6*sizeof(float))))
			{
				fprintf(stderr, "MPI task %i: failed to allocate memory for mass_table_buffer.\n", rank);
				exit(-2);
			}
			H5Aread(attr_id,  H5T_NATIVE_FLOAT, buffer);
			for (int i = 0; i < 6; i++)
			{
				mass_table[i] = (REAL) buffer[i];
			}
			free(buffer);
		}
#endif
		H5Aclose(attr_id);
		H5Gclose(group);
		// mass_table[0] is the mass of the first particle type (gas), mass_table[1] is the mass of the second particle type (dark matter)
		printf("\tThe particle mass is set to %.7f * 10^10 Msol(/h)\n", mass_table[1]);
		// set the mass of all particles to the mass of the second particle type (dark matter)
		for(int i = 0; i < N; i++)
		{
			M[i] = mass_table[1]/10.0;
		}
		H5Fclose(IC);
		printf("...done\n\n");
		return;
	}

	printf("\tReading /PartType1/Masses\n");
	dataset = H5Dopen2(IC, "/PartType1/Masses", H5P_DEFAULT);
	dataspace_in_file = H5Dget_space(dataset);
	datatype =  H5Dget_type(dataset);
#ifdef USE_SINGLE_PRECISION
	if(H5Tequal(datatype, H5T_NATIVE_FLOAT))
	{
		H5Dread(dataset, datatype, H5S_ALL, H5S_ALL, H5P_DEFAULT, M);
	}
	else
	{
		printf("\t\tData stored in doubles.\n");
		double* buffer;
		if(!(buffer = (double*)malloc(N*sizeof(double))))
		{
			fprintf(stderr, "MPI task %i: failed to allocate memory for M_buffer.\n", rank);
			exit(-2);
		}
		H5Dread(dataset, datatype, H5S_ALL, H5S_ALL, H5P_DEFAULT, buffer);
		for (int i = 0; i < N; i++)
		{
			M[i] = (REAL) buffer[i];
		}
		free(buffer);
	}
#else
	if(H5Tequal(datatype, H5T_NATIVE_DOUBLE))
	{
		H5Dread(dataset, datatype, H5S_ALL, H5S_ALL, H5P_DEFAULT, M);
	}
	else
	{
		printf("\t\tData stored in floats.\n");
		float* buffer;
		if(!(buffer = (float*)malloc(N*sizeof(float))))
		{
			fprintf(stderr, "MPI task %i: failed to allocate memory for M_buffer.\n", rank);
			exit(-2);
		}
		H5Dread(dataset, datatype, H5S_ALL, H5S_ALL, H5P_DEFAULT, buffer);
		for (int i = 0; i < N; i++)
		{
			M[i] = (REAL) buffer[i];
		}
		free(buffer);
	}
#endif
	H5Tclose(datatype);
	H5Sclose(dataspace_in_file);
	H5Dclose(dataset);

	H5Fclose(IC);
	printf("...done\n\n");
	return;
}

void write_hdf5_snapshot(REAL* x, REAL *v, REAL *M)
{
	int i, hdf5_rank;
	char buf[500];
	//setting up the output filename
	char filename[0x400];
	if(snprintf(filename, sizeof(filename), "%ssnapshot_%04d.hdf5", OUT_DIR, N_snapshot)<0)
	{
		fprintf(stderr, "Error: The output file name truncated.\nAborting.\n");
		abort();
	}
	if(COSMOLOGY == 0)
	{
		printf("Saving the \"%s\" snapshot file...\nt=%.14f", filename, T);
	}
	else
	{
		printf("Saving: the \"%s\" snapshot file...\nt = %.15fGy\na = %.15f\n", filename, T*UNIT_T, a);
	}
	//Output filename set. Creating the output file
	hid_t snapshot = 0;
	hid_t hdf5_grp[6]; //group for the data types (only DM are used in this version)
	hid_t headergrp = 0;
	hid_t dataspace_in_file = 0;
	hid_t dataset = 0;
	hid_t datatype = 0;
	hid_t dataspace_memory;
	hsize_t dims[2], count[2], start[2];
	snapshot = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
	//Output file created. Creating the header group, and write the header into the file
	headergrp = H5Gcreate(snapshot, "/Header", 0, H5P_DEFAULT,H5P_DEFAULT);
	//This code only uses DM particles, so type=1 (we do this for gadget compatibility)
	int type = 1;
	sprintf(buf, "/PartType%d", type);
	hdf5_grp[type] = H5Gcreate(snapshot, buf, 0, H5P_DEFAULT,H5P_DEFAULT);
	write_header_attributes_in_hdf5(headergrp);
	//header written.

	//Writing out the particle positions
	dims[0] = N;
	dims[1] = 3; //hdf5_rank = 2 [if dims[1] = 1: 1; else: 2]
	hdf5_rank = 2;
	#ifdef USE_SINGLE_PRECISION
		datatype = H5Tcopy(H5T_NATIVE_FLOAT); //coordinates saved as float
	#else
		datatype = H5Tcopy(H5T_NATIVE_DOUBLE); //coordinates saved as double
	#endif
	strcpy(buf, "Coordinates");
	dataspace_in_file = H5Screate_simple(hdf5_rank, dims, NULL);
	dataset = H5Dcreate(hdf5_grp[type], buf, datatype, dataspace_in_file, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	start[0] = 0;
	start[1] = 0;
	count[0] = N;
	count[1] = 3;
	H5Sselect_hyperslab(dataspace_in_file, H5S_SELECT_SET, start, NULL, count, NULL);
	dataspace_memory = H5Screate_simple(hdf5_rank, dims, NULL);
	if(H0_INDEPENDENT_UNITS != 0 && COSMOLOGY == 1)
	{
		REAL H0_dimless = H0*UNIT_V/100.0;
		for(i=0;i<3*N;i++)
			x[i] *= H0_dimless;
		H5Dwrite(dataset, datatype, dataspace_memory, dataspace_in_file, H5P_DEFAULT, x);
		for(i=0;i<3*N;i++)
			x[i] /= H0_dimless;
	}
	else
	{
		H5Dwrite(dataset, datatype, dataspace_memory, dataspace_in_file, H5P_DEFAULT, x);
	}
	H5Sclose(dataspace_memory);
	H5Dclose(dataset);
	H5Sclose(dataspace_in_file);
	H5Tclose(datatype);

	//Writing out particle velocities
	dims[0] = N;
	dims[1] = 3; //hdf5_rank = 2 [if dims[1] = 1: 1; else: 2]
	hdf5_rank = 2;
	#ifdef USE_SINGLE_PRECISION
		datatype = H5Tcopy(H5T_NATIVE_FLOAT); //velocities saved as float
	#else
		datatype = H5Tcopy(H5T_NATIVE_DOUBLE); //velocities saved as double
	#endif
	strcpy(buf, "Velocities");
	dataspace_in_file = H5Screate_simple(hdf5_rank, dims, NULL);
	dataset = H5Dcreate(hdf5_grp[type], buf, datatype, dataspace_in_file, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	start[0] = 0;
	start[1] = 0;
	count[0] = N;
	count[1] = 3;
	H5Sselect_hyperslab(dataspace_in_file, H5S_SELECT_SET, start, NULL, count, NULL);
	dataspace_memory = H5Screate_simple(hdf5_rank, dims, NULL);
	REAL *Velocity_buf;
	if(!(Velocity_buf = (REAL *)malloc(3*N*sizeof(REAL))))
	{
		fprintf(stderr, "MPI task %i: failed to allocate memory for Velocity_buff.\n", rank);
		exit(-2);
	}
	for(i=0;i<3*N;i++)
			Velocity_buf[i] = v[i]*(REAL)sqrt(a)*(REAL)UNIT_V; //km/s * sqrt(a) output
	H5Dwrite(dataset, datatype, dataspace_memory, dataspace_in_file, H5P_DEFAULT, Velocity_buf);
	H5Sclose(dataspace_memory);
	H5Dclose(dataset);
	H5Sclose(dataspace_in_file);
	H5Tclose(datatype);

	//Writing out particle IDs
	dims[0] = N;
	dims[1] = 1; //hdf5_rank = 2 [if dims[1] = 1: 1; else: 2]
	hdf5_rank = 1;
	datatype = H5Tcopy(H5T_NATIVE_UINT64); //IDs are saved as unsigned 64 bit ints
	strcpy(buf, "ParticleIDs");
	dataspace_in_file = H5Screate_simple(hdf5_rank, dims, NULL);
	dataset = H5Dcreate(hdf5_grp[type], buf, datatype, dataspace_in_file, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	start[0] = 0;
	start[1] = 0;
	count[0] = N;
	count[1] = 1;
	H5Sselect_hyperslab(dataspace_in_file, H5S_SELECT_SET, start, NULL, count, NULL);
	dataspace_memory = H5Screate_simple(hdf5_rank, dims, NULL);
	unsigned long long int *ID;
	if(!(ID = (unsigned long long int *)malloc(N*sizeof(unsigned long long int))))
	{
		fprintf(stderr, "MPI task %i: failed to allocate memory for ID.\n", rank);
		exit(-2);
	}
	for(i=0;i<N;i++)
		ID[i] = i;
	H5Dwrite(dataset, datatype, dataspace_memory, dataspace_in_file, H5P_DEFAULT, ID);
	H5Sclose(dataspace_memory);
	H5Dclose(dataset);
	H5Sclose(dataspace_in_file);
	H5Tclose(datatype);

	//Writing out particle Masses
        dims[0] = N;
        dims[1] = 1; //hdf5_rank = 2 [if dims[1] = 1: 1; else: 2]
        hdf5_rank = 1;
	#ifdef USE_SINGLE_PRECISION
		datatype = H5Tcopy(H5T_NATIVE_FLOAT); //Masses saved as float
	#else
		datatype = H5Tcopy(H5T_NATIVE_DOUBLE); //Masses saved as double
	#endif
  strcpy(buf, "Masses");
  dataspace_in_file = H5Screate_simple(hdf5_rank, dims, NULL);
  dataset = H5Dcreate(hdf5_grp[type], buf, datatype, dataspace_in_file, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  start[0] = 0;
  start[1] = 0;
  count[0] = N;
  count[1] = 1;
  H5Sselect_hyperslab(dataspace_in_file, H5S_SELECT_SET, start, NULL, count, NULL);
  dataspace_memory = H5Screate_simple(hdf5_rank, dims, NULL);
	if(H0_INDEPENDENT_UNITS != 0 && COSMOLOGY == 1)
	{
		REAL H0_dimless = H0*UNIT_V/100.0;
		for(i=0;i<N;i++)
			M[i] *= H0_dimless;
		H5Dwrite(dataset, datatype, dataspace_memory, dataspace_in_file, H5P_DEFAULT, M);
		for(i=0;i<N;i++)
			M[i] /= H0_dimless;
	}
	else
	{
  	H5Dwrite(dataset, datatype, dataspace_memory, dataspace_in_file, H5P_DEFAULT, M);
	}
  H5Sclose(dataspace_memory);
  H5Dclose(dataset);
  H5Sclose(dataspace_in_file);
  H5Tclose(datatype);

	H5Gclose(hdf5_grp[1]);
	H5Gclose(headergrp);
	H5Fclose(snapshot);
	N_snapshot++;

}

void write_header_attributes_in_hdf5(hid_t handle)
{
	int i;
	double doublebuf;
	hsize_t adim[1] = { 6 };
	hid_t hdf5_dataspace, hdf5_attribute;

	//NumPart_ThisFile
	hdf5_dataspace = H5Screate(H5S_SIMPLE);
	H5Sset_extent_simple(hdf5_dataspace, 1, adim, NULL);
	hdf5_attribute = H5Acreate(handle, "NumPart_ThisFile", H5T_NATIVE_INT, hdf5_dataspace, H5P_DEFAULT,  H5P_DEFAULT);
	int npart[6];
	for(i=0; i<6; i++)
		npart[i] = 0;
	npart[1] = (int) N;
	H5Awrite(hdf5_attribute, H5T_NATIVE_UINT, npart);
	H5Aclose(hdf5_attribute);
	H5Sclose(hdf5_dataspace);

	//NumPart_Total
	hdf5_dataspace = H5Screate(H5S_SIMPLE);
	H5Sset_extent_simple(hdf5_dataspace, 1, adim, NULL);
	hdf5_attribute = H5Acreate(handle, "NumPart_Total", H5T_NATIVE_UINT, hdf5_dataspace, H5P_DEFAULT, H5P_DEFAULT);
	H5Awrite(hdf5_attribute, H5T_NATIVE_UINT, npart);
	H5Aclose(hdf5_attribute);
	H5Sclose(hdf5_dataspace);

	//NumPart_Total_HighWord
	hdf5_dataspace = H5Screate(H5S_SIMPLE);
	H5Sset_extent_simple(hdf5_dataspace, 1, adim, NULL);
	for(i=0; i<6; i++)
		npart[i] = 0;
	hdf5_attribute = H5Acreate(handle, "NumPart_Total_HighWord", H5T_NATIVE_UINT, hdf5_dataspace, H5P_DEFAULT, H5P_DEFAULT);
	H5Awrite(hdf5_attribute, H5T_NATIVE_UINT, npart);
	H5Aclose(hdf5_attribute);
	H5Sclose(hdf5_dataspace);

	//MassTable
	hdf5_dataspace = H5Screate(H5S_SIMPLE);
	H5Sset_extent_simple(hdf5_dataspace, 1, adim, NULL);
	double mass[6];
	for(i=0; i<6; i++)
		mass[i] = 0.0;
	//mass[1] = (double) M_min;
	hdf5_attribute = H5Acreate(handle, "MassTable", H5T_NATIVE_DOUBLE, hdf5_dataspace, H5P_DEFAULT, H5P_DEFAULT);
	H5Awrite(hdf5_attribute, H5T_NATIVE_DOUBLE, mass);
	H5Aclose(hdf5_attribute);
	H5Sclose(hdf5_dataspace);

	//Time (or scale factor)
	hdf5_dataspace = H5Screate(H5S_SCALAR);
	hdf5_attribute = H5Acreate(handle, "Time", H5T_NATIVE_DOUBLE, hdf5_dataspace, H5P_DEFAULT,  H5P_DEFAULT);
	if(COSMOLOGY == 0 || (OUTPUT_TIME_VARIABLE==0 && COMOVING_INTEGRATION==0))
		doublebuf = T*UNIT_T; //physical time
	else
		doublebuf = a; //scalefactor
	H5Awrite(hdf5_attribute, H5T_NATIVE_DOUBLE, &doublebuf);
	H5Aclose(hdf5_attribute);
	H5Sclose(hdf5_dataspace);

	//Redshift
	hdf5_dataspace = H5Screate(H5S_SCALAR);
	hdf5_attribute = H5Acreate(handle, "Redshift", H5T_NATIVE_DOUBLE, hdf5_dataspace, H5P_DEFAULT, H5P_DEFAULT);
	double redshift=(1.0/a-1.0);
	H5Awrite(hdf5_attribute, H5T_NATIVE_DOUBLE, &redshift);
	H5Aclose(hdf5_attribute);
	H5Sclose(hdf5_dataspace);

	//BoxSize
	hdf5_dataspace = H5Screate(H5S_SCALAR);
	hdf5_attribute = H5Acreate(handle, "BoxSize", H5T_NATIVE_DOUBLE, hdf5_dataspace, H5P_DEFAULT, H5P_DEFAULT);
	if(H0_INDEPENDENT_UNITS == 0)
	{
		H5Awrite(hdf5_attribute, H5T_NATIVE_DOUBLE, &L); //Linear simulation size
	}
	else
	{
		doublebuf = L*H0*UNIT_V/100.0; //Linear simulation size
		H5Awrite(hdf5_attribute, H5T_NATIVE_DOUBLE, &doublebuf); //Linear simulation size in Mpc/h
	}
	H5Aclose(hdf5_attribute);
	H5Sclose(hdf5_dataspace);

	//NumFilesPerSnapshot
	hdf5_dataspace = H5Screate(H5S_SCALAR);
	hdf5_attribute = H5Acreate(handle, "NumFilesPerSnapshot", H5T_NATIVE_INT, hdf5_dataspace, H5P_DEFAULT, H5P_DEFAULT);
	int numfiles = 1;
	H5Awrite(hdf5_attribute, H5T_NATIVE_INT, &numfiles);
	H5Aclose(hdf5_attribute);
	H5Sclose(hdf5_dataspace);

	//Omega0 (Omega_matter)
	hdf5_dataspace = H5Screate(H5S_SCALAR);
	hdf5_attribute = H5Acreate(handle, "Omega0", H5T_NATIVE_DOUBLE, hdf5_dataspace, H5P_DEFAULT, H5P_DEFAULT);
	H5Awrite(hdf5_attribute, H5T_NATIVE_DOUBLE, &Omega_m);
	H5Aclose(hdf5_attribute);
	H5Sclose(hdf5_dataspace);

	//OmegaLambda (Omega_dark_energy)
	hdf5_dataspace = H5Screate(H5S_SCALAR);
	hdf5_attribute = H5Acreate(handle, "OmegaLambda", H5T_NATIVE_DOUBLE, hdf5_dataspace, H5P_DEFAULT, H5P_DEFAULT);
	H5Awrite(hdf5_attribute, H5T_NATIVE_DOUBLE, &Omega_lambda);
	H5Aclose(hdf5_attribute);
	H5Sclose(hdf5_dataspace);

	//HubbleParam (H0 in 100km/s/Mpc units)
	hdf5_dataspace = H5Screate(H5S_SCALAR);
	hdf5_attribute = H5Acreate(handle, "HubbleParam", H5T_NATIVE_DOUBLE, hdf5_dataspace, H5P_DEFAULT, H5P_DEFAULT);
	redshift = H0*UNIT_V/100.0;
	H5Awrite(hdf5_attribute, H5T_NATIVE_DOUBLE, &redshift); //H0 - Hubble constant in 100km/s/Mpc units
	H5Aclose(hdf5_attribute);
	H5Sclose(hdf5_dataspace);

	//Parameters of cosmological models beyond LCDM (if applicable)
	#if COSMOPARAM==1 || COSMOPARAM==2
	hdf5_dataspace = H5Screate(H5S_SCALAR);
	hdf5_attribute = H5Acreate(handle, "DE_w0", H5T_NATIVE_DOUBLE, hdf5_dataspace, H5P_DEFAULT, H5P_DEFAULT);
	H5Awrite(hdf5_attribute, H5T_NATIVE_DOUBLE, &w0);
	H5Aclose(hdf5_attribute);
	H5Sclose(hdf5_dataspace);
	#endif

	#if COSMOPARAM==2
	hdf5_dataspace = H5Screate(H5S_SCALAR);
	hdf5_attribute = H5Acreate(handle, "DE_wa", H5T_NATIVE_DOUBLE, hdf5_dataspace, H5P_DEFAULT, H5P_DEFAULT);
	H5Awrite(hdf5_attribute, H5T_NATIVE_DOUBLE, &wa);
	H5Aclose(hdf5_attribute);
	H5Sclose(hdf5_dataspace);
	#endif

	//Flags for Compile time options, program name, and version info
	hdf5_dataspace = H5Screate(H5S_SCALAR);
	hid_t str_type = H5Tcopy(H5T_C_S1);
	H5Tset_size(str_type, strlen(PROGRAMNAME) + 1);
	H5Tset_cset(str_type, H5T_CSET_ASCII);

	hdf5_attribute = H5Acreate(handle, "ProgramName", str_type, hdf5_dataspace, H5P_DEFAULT, H5P_DEFAULT);
	H5Awrite(hdf5_attribute, str_type, (const void*)PROGRAMNAME);
	H5Aclose(hdf5_attribute);

	H5Tset_size(str_type, strlen(PROGRAM_VERSION) + 1);
	hdf5_attribute = H5Acreate(handle, "ProgramVersion", str_type, hdf5_dataspace, H5P_DEFAULT, H5P_DEFAULT);
	H5Awrite(hdf5_attribute, str_type, (const void*)PROGRAM_VERSION);
	H5Aclose(hdf5_attribute);

	H5Tset_size(str_type, strlen(BUILD_DATE) + 1);
	hdf5_attribute = H5Acreate(handle, "ProgramBuildDate", str_type, hdf5_dataspace, H5P_DEFAULT, H5P_DEFAULT);
	H5Awrite(hdf5_attribute, str_type, (const void*)BUILD_DATE);
	H5Aclose(hdf5_attribute);

	H5Sclose(hdf5_dataspace);
	H5Tclose(str_type);

	

	//Flags for GADGET compatibility
	hdf5_dataspace = H5Screate(H5S_SCALAR);
	hdf5_attribute = H5Acreate(handle, "Flag_Sfr", H5T_NATIVE_INT, hdf5_dataspace, H5P_DEFAULT, H5P_DEFAULT);
	int zero = 0;
	H5Awrite(hdf5_attribute, H5T_NATIVE_INT, &zero);
	H5Aclose(hdf5_attribute);
	H5Sclose(hdf5_dataspace);

	hdf5_dataspace = H5Screate(H5S_SCALAR);
	hdf5_attribute = H5Acreate(handle, "Flag_Cooling", H5T_NATIVE_INT, hdf5_dataspace, H5P_DEFAULT, H5P_DEFAULT);
	H5Awrite(hdf5_attribute, H5T_NATIVE_INT, &zero);
	H5Aclose(hdf5_attribute);
	H5Sclose(hdf5_dataspace);

	hdf5_dataspace = H5Screate(H5S_SCALAR);
	hdf5_attribute = H5Acreate(handle, "Flag_StellarAge", H5T_NATIVE_INT, hdf5_dataspace, H5P_DEFAULT, H5P_DEFAULT);
	H5Awrite(hdf5_attribute, H5T_NATIVE_INT, &zero);
	H5Aclose(hdf5_attribute);
	H5Sclose(hdf5_dataspace);

	hdf5_dataspace = H5Screate(H5S_SCALAR);
	hdf5_attribute = H5Acreate(handle, "Flag_Metals", H5T_NATIVE_INT, hdf5_dataspace, H5P_DEFAULT, H5P_DEFAULT);
	H5Awrite(hdf5_attribute, H5T_NATIVE_INT, &zero);
	H5Aclose(hdf5_attribute);
	H5Sclose(hdf5_dataspace);

	hdf5_dataspace = H5Screate(H5S_SCALAR);
	hdf5_attribute = H5Acreate(handle, "Flag_Feedback", H5T_NATIVE_INT, hdf5_dataspace, H5P_DEFAULT, H5P_DEFAULT);
	H5Awrite(hdf5_attribute, H5T_NATIVE_INT, &zero);
	H5Aclose(hdf5_attribute);
	H5Sclose(hdf5_dataspace);

	hdf5_dataspace = H5Screate(H5S_SIMPLE);
	H5Sset_extent_simple(hdf5_dataspace, 1, adim, NULL);
	hdf5_attribute = H5Acreate(handle, "Flag_Entropy_ICs", H5T_NATIVE_UINT, hdf5_dataspace, H5P_DEFAULT, H5P_DEFAULT);
	H5Awrite(hdf5_attribute, H5T_NATIVE_UINT, &zero);
	H5Aclose(hdf5_attribute);
	H5Sclose(hdf5_dataspace);
}
#endif


int load_IC(char *IC_FILE, int IC_FORMAT)
{
	//Loading initial conditions from a file
	// Input parameters:
	// IC_FILE: the name of the file containing the initial conditions
	// IC_FORMAT: the format of the initial conditions (0 - ASCII, 1-GADGET, 2 - HDF5)
	#ifndef HAVE_HDF5
	if(IC_FORMAT != 0 && IC_FORMAT != 1)
	{
		fprintf(stderr, "Error: bad IC format!\nExiting.\n");
		return (-1);
	}
	#else
	if(IC_FORMAT < 0 || IC_FORMAT > 2)
			{
					fprintf(stderr, "Error: bad IC format!\nExiting.\n");
					return (-1);
			}
	#endif
	if(IC_FORMAT == 0)
	{
		printf("\nThe IC file is in ASCII format.\n");
		if(file_exist(IC_FILE) == 0)
		{
			fprintf(stderr, "Error: The %s IC file does not exist!\nExiting.\n", IC_FILE);
			return (-1);
		}
		N = measure_N_part_from_ascii_snapshot(IC_FILE);
		FILE *ic_file = fopen(IC_FILE, "r");
		read_ascii_ic(ic_file, N, Allocate_memory);
		Allocate_memory = false; //Now the memory is already allocated for the particle data arrays.
	}
	if(IC_FORMAT == 1)
	{
		int files;
		printf("\nThe IC file is in Gadget format.\nThe IC determines the box size.\n");
		files = 1;      /* number of files per snapshot */
		if(file_exist(IC_FILE) == 0)
		{
			fprintf(stderr, "Error: The %s IC file does not exist!\nExiting.\n", IC_FILE);
			return (-1);
		}
		load_snapshot(IC_FILE, files);
		reordering();
		gadget_format_conversion(Allocate_memory);
		Allocate_memory = false; //Now the memory is already allocated for the particle data arrays.
	}
	#ifdef HAVE_HDF5
	if(IC_FORMAT == 2)
	{
		printf("\nThe IC is in HDF5 format\n");
		if(file_exist(IC_FILE) == 0)
		{
			fprintf(stderr, "Error: The %s IC file does not exist!\nExiting.\n", IC_FILE);
			return (-1);
		}
		read_hdf5_ic(IC_FILE, Allocate_memory);
		Allocate_memory = false; //Now the memory is already allocated for the particle data arrays.
	}
	#endif
	return 0;//Return 0 if the IC file was loaded successfully
}
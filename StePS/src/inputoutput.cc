/********************************************************************************/
/*  StePS - STEreographically Projected cosmological Simulations                */
/*    Copyright (C) 2017-2018 Gabor Racz                                        */
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
#include <algorithm>
#include <sys/types.h>
#include <sys/stat.h>
#include "mpi.h"
#include "global_variables.h"

#ifdef HAVE_HDF5
#include <hdf5.h>

void write_header_attributes_in_hdf5(hid_t handle);
#endif

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

void read_ascii_ic(FILE *ic_file, int N)
{
int i,j;

x = (REAL*)malloc(3*N*sizeof(REAL)); //Allocating memory for the coordinates
v = (REAL*)malloc(3*N*sizeof(REAL)); //Allocating memory for the velocities
F = (REAL*)malloc(3*N*sizeof(REAL)); //Allocating memory for the forces
M = (REAL*)malloc(N*sizeof(REAL)); //Allocating memory for the masses


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
	out_list[size] = 1.0/a_max-1.0;
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
	if(OUTPUT_FORMAT == 0)
		snprintf(filename, sizeof(filename), "%sredshift_cone.dat", OUT_DIR);
	#ifdef HAVE_HDF5
	if(OUTPUT_FORMAT == 2)
		snprintf(filename, sizeof(filename), "%sredshift_cone.hdf5", OUT_DIR);
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
	}
	#ifdef HAVE_HDF5
	char buf[500];
	REAL bufvec[3];
	unsigned long long int *ID;
	ID = (unsigned long long int *)malloc(1*sizeof(unsigned long long int));
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
					bufvec[0] = v[3*i];
					bufvec[1] = v[3*i+1];
					bufvec[2] = v[3*i+2];
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
					bufvec[0] = v[3*i];
					bufvec[1] = v[3*i+1];
					bufvec[2] = v[3*i+2];
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
		if(OUTPUT_TIME_VARIABLE == 0)
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

#ifdef HAVE_HDF5

void read_hdf5_ic(char *ic_file)
{
	hid_t attr_id = 0;
	hid_t group = 0;
	hid_t dataset = 0;
	hid_t datatype = 0;
	hid_t dataspace_in_file = 0;
	hid_t IC = 0;
	int *Nbuf;
	Nbuf = (int *)malloc(6*sizeof(int));
	printf("Reading the %s IC file...\n", ic_file);
	IC = H5Fopen(ic_file, H5F_ACC_RDONLY, H5P_DEFAULT);
	//reading the total number of particles from the header
	group = H5Gopen2(IC,"Header", H5P_DEFAULT);
	attr_id = H5Aopen(group, "NumPart_ThisFile", H5P_DEFAULT);
	H5Aread(attr_id,  H5T_NATIVE_INT, Nbuf);
	N = Nbuf[1];
	printf("The number of particles:\t%i\n", N);
	H5Aclose(attr_id);
	H5Gclose(group);
	//Allocating memory
	x = (REAL*)malloc(3*N*sizeof(REAL)); //Allocating memory for the coordinates
	v = (REAL*)malloc(3*N*sizeof(REAL)); //Allocating memory for the velocities
	F = (REAL*)malloc(3*N*sizeof(REAL)); //Allocating memory for the forces
	M = (REAL*)malloc(N*sizeof(REAL)); //Allocating memory for the masses
	//reading the particle coordinates
	dataset = H5Dopen2(IC, "/PartType1/Coordinates", H5P_DEFAULT);
        dataspace_in_file = H5Dget_space(dataset);
	datatype =  H5Dget_type(dataset);
	H5Dread(dataset, datatype, H5S_ALL, H5S_ALL, H5P_DEFAULT, x);
	H5Tclose(datatype);
	H5Sclose(dataspace_in_file);
	H5Dclose(dataset);
	//reading the particle velocities
	dataset = H5Dopen2(IC, "/PartType1/Velocities", H5P_DEFAULT);
	dataspace_in_file = H5Dget_space(dataset);
	datatype =  H5Dget_type(dataset);
	H5Dread(dataset, datatype, H5S_ALL, H5S_ALL, H5P_DEFAULT, v);
	H5Tclose(datatype);
	H5Sclose(dataspace_in_file);
	H5Dclose(dataset);	
	//reading the particle masses
	dataset = H5Dopen2(IC, "/PartType1/Masses", H5P_DEFAULT);
	dataspace_in_file = H5Dget_space(dataset);
	datatype =  H5Dget_type(dataset);
	H5Dread(dataset, datatype, H5S_ALL, H5S_ALL, H5P_DEFAULT, M);
	H5Tclose(datatype);
	H5Sclose(dataspace_in_file);
	H5Dclose(dataset);
	
	H5Fclose(IC);
	printf("...done\n\n");
}

void write_hdf5_snapshot(REAL* x, REAL *v, REAL *M)
{
	int i, hdf5_rank;
	char buf[500];
	//setting up the output filename
	char filename[0x400];
	snprintf(filename, sizeof(filename), "%ssnapshot_%04d.hdf5", OUT_DIR, N_snapshot);
	if(COSMOLOGY == 0)
	{
		printf("Saving: t= %f, file: \"%s\" \n", t_next, filename);
	}
	else
	{
		if(OUTPUT_TIME_VARIABLE == 0)
		{
			printf("Saving: t= %f, file: \"%s\" \n", t_next*UNIT_T, filename);
		}
		else
		{
			printf("Saving: z= %f, file: \"%s\" \n", t_next, filename);
		}
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
	H5Dwrite(dataset, datatype, dataspace_memory, dataspace_in_file, H5P_DEFAULT, x);
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
	Velocity_buf = (REAL *)malloc(3*N*sizeof(REAL));
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
	ID = (unsigned long long int *)malloc(N*sizeof(unsigned long long int));
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
        H5Dwrite(dataset, datatype, dataspace_memory, dataspace_in_file, H5P_DEFAULT, M);
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

	hdf5_dataspace = H5Screate(H5S_SIMPLE);
	H5Sset_extent_simple(hdf5_dataspace, 1, adim, NULL);
	hdf5_attribute = H5Acreate(handle, "NumPart_Total", H5T_NATIVE_UINT, hdf5_dataspace, H5P_DEFAULT, H5P_DEFAULT);
	H5Awrite(hdf5_attribute, H5T_NATIVE_UINT, npart);
	H5Aclose(hdf5_attribute);
	H5Sclose(hdf5_dataspace);

	hdf5_dataspace = H5Screate(H5S_SIMPLE);
	H5Sset_extent_simple(hdf5_dataspace, 1, adim, NULL);
	for(i=0; i<6; i++)
		npart[i] = 0;
	hdf5_attribute = H5Acreate(handle, "NumPart_Total_HighWord", H5T_NATIVE_UINT, hdf5_dataspace, H5P_DEFAULT, H5P_DEFAULT);
	H5Awrite(hdf5_attribute, H5T_NATIVE_UINT, npart);
	H5Aclose(hdf5_attribute);
	H5Sclose(hdf5_dataspace);


	hdf5_dataspace = H5Screate(H5S_SIMPLE);
	H5Sset_extent_simple(hdf5_dataspace, 1, adim, NULL);
	double mass[6];
	for(i=0; i<6; i++)
		mass[i] = 0;
	mass[1] = (double) M_min;
	hdf5_attribute = H5Acreate(handle, "MassTable", H5T_NATIVE_DOUBLE, hdf5_dataspace, H5P_DEFAULT, H5P_DEFAULT);
	H5Awrite(hdf5_attribute, H5T_NATIVE_DOUBLE, mass);
	H5Aclose(hdf5_attribute);
	H5Sclose(hdf5_dataspace);

	hdf5_dataspace = H5Screate(H5S_SCALAR);
	hdf5_attribute = H5Acreate(handle, "Time", H5T_NATIVE_DOUBLE, hdf5_dataspace, H5P_DEFAULT,  H5P_DEFAULT);
	if(COSMOLOGY == 0)
		doublebuf = T*UNIT_T; //physical time
	else
		doublebuf = a; //scalefactor
	H5Awrite(hdf5_attribute, H5T_NATIVE_DOUBLE, &doublebuf);
	H5Aclose(hdf5_attribute);
	H5Sclose(hdf5_dataspace);

	hdf5_dataspace = H5Screate(H5S_SCALAR);
	hdf5_attribute = H5Acreate(handle, "Redshift", H5T_NATIVE_DOUBLE, hdf5_dataspace, H5P_DEFAULT, H5P_DEFAULT);
	double redshift=(1.0/a-1.0);
	H5Awrite(hdf5_attribute, H5T_NATIVE_DOUBLE, &redshift);
	H5Aclose(hdf5_attribute);
	H5Sclose(hdf5_dataspace);

	hdf5_dataspace = H5Screate(H5S_SCALAR);
	hdf5_attribute = H5Acreate(handle, "BoxSize", H5T_NATIVE_DOUBLE, hdf5_dataspace, H5P_DEFAULT, H5P_DEFAULT);
	H5Awrite(hdf5_attribute, H5T_NATIVE_DOUBLE, &L); //Linear simulation size
	H5Aclose(hdf5_attribute);
	H5Sclose(hdf5_dataspace);

	hdf5_dataspace = H5Screate(H5S_SCALAR);
	hdf5_attribute = H5Acreate(handle, "NumFilesPerSnapshot", H5T_NATIVE_INT, hdf5_dataspace, H5P_DEFAULT, H5P_DEFAULT);
	int numfiles = 1;
	H5Awrite(hdf5_attribute, H5T_NATIVE_INT, &numfiles);
	H5Aclose(hdf5_attribute);
	H5Sclose(hdf5_dataspace);

	hdf5_dataspace = H5Screate(H5S_SCALAR);
	hdf5_attribute = H5Acreate(handle, "Omega0", H5T_NATIVE_DOUBLE, hdf5_dataspace, H5P_DEFAULT, H5P_DEFAULT);
	H5Awrite(hdf5_attribute, H5T_NATIVE_DOUBLE, &Omega_m);
	H5Aclose(hdf5_attribute);
	H5Sclose(hdf5_dataspace);

	hdf5_dataspace = H5Screate(H5S_SCALAR);
	hdf5_attribute = H5Acreate(handle, "OmegaLambda", H5T_NATIVE_DOUBLE, hdf5_dataspace, H5P_DEFAULT, H5P_DEFAULT);
	H5Awrite(hdf5_attribute, H5T_NATIVE_DOUBLE, &Omega_lambda);
	H5Aclose(hdf5_attribute);
	H5Sclose(hdf5_dataspace);

	hdf5_dataspace = H5Screate(H5S_SCALAR);
	hdf5_attribute = H5Acreate(handle, "HubbleParam", H5T_NATIVE_DOUBLE, hdf5_dataspace, H5P_DEFAULT, H5P_DEFAULT);
	redshift = H0*UNIT_V/100.0;
	H5Awrite(hdf5_attribute, H5T_NATIVE_DOUBLE, &redshift); //H0 - Hubble constant in 100km/s/Mpc units
	H5Aclose(hdf5_attribute);
	H5Sclose(hdf5_dataspace);

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

/********************************************************************************/
/*  StePS - STEreographically Projected cosmological Simulations                */
/*    Copyright (C) 2017-2022 Gabor Racz                                        */
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
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include "mpi.h"
#include "global_variables.h"

#define BUFF_SIZE 1024

#ifdef USE_SINGLE_PRECISION
typedef float REAL;
#else
typedef double REAL;
#endif

char in_file[BUFF_SIZE];

void BCAST_global_parameters();

void read_param(FILE *param_file);

void BCAST_global_parameters()
{
	//Cosmological parameters
	MPI_Bcast(&Omega_b,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
	MPI_Bcast(&Omega_lambda,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
	MPI_Bcast(&Omega_dm,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
	MPI_Bcast(&Omega_r,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
	MPI_Bcast(&H0,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
#if COSMOPARAM==1
	MPI_Bcast(&w0,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
#elif COSMOPARAM==2
	MPI_Bcast(&w0,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
	MPI_Bcast(&wa,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
#endif
	MPI_Bcast(&COSMOLOGY,1,MPI_INT,0,MPI_COMM_WORLD);
	//Simulation parameters
	//the rank != 0 tasks do not need all of the simulation parameters
	MPI_Bcast(&IS_PERIODIC,1,MPI_INT,0,MPI_COMM_WORLD);
	MPI_Bcast(&a_max,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
	MPI_Bcast(&a_start,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
	MPI_Bcast(&COMOVING_INTEGRATION,1,MPI_INT,0,MPI_COMM_WORLD);
	MPI_Bcast(&OUTPUT_TIME_VARIABLE,1,MPI_INT,0,MPI_COMM_WORLD);
	MPI_Bcast(&TIME_LIMIT_IN_MINS,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
#ifdef USE_SINGLE_PRECISION
	MPI_Bcast(&L,1,MPI_FLOAT,0,MPI_COMM_WORLD);
	MPI_Bcast(&ACC_PARAM,1,MPI_FLOAT,0,MPI_COMM_WORLD);
	MPI_Bcast(&ParticleRadi,1,MPI_FLOAT,0,MPI_COMM_WORLD);
#else
	MPI_Bcast(&L,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
	MPI_Bcast(&ACC_PARAM,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
	MPI_Bcast(&ParticleRadi,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
#endif
return;
}

void read_param(FILE *param_file)
{
int i;
char c[BUFF_SIZE];
char str01[] = "Omega_b";
char str02[] = "Omega_lambda";
char str03[] = "Omega_dm";
char str04[] = "Omega_r";
char str05[] = "HubbleConstant";
char str06[] = "IC_FILE";
char str07[] = "L_BOX";
char str08[] = "IS_PERIODIC";
char str09[] = "OUTPUT_FORMAT";
char str10[] = "OUT_DIR";
char str11[] = "a_max";
char str12[] = "SNAPSHOT_START_NUMBER";
char str13[] = "ACC_PARAM";
char str14[] = "STEP_MIN";
char str15[] = "COSMOLOGY";
char str17[] = "a_start";
char str18[] = "STEP_MAX";
char str19[] = "COMOVING_INTEGRATION";
char str20[] = "TIME_LIMIT_IN_MIN";
char str21[] = "FIRST_T_OUT";
char str22[] = "IC_FORMAT";
char str23[] = "OUTPUT_TIME_VARIABLE";
char str24[] = "OUT_LST";
char str25[] = "H_OUT";
char str26[] = "PARTICLE_RADII";
char str31[] = "MIN_REDSHIFT";
char str32[] = "REDSHIFT_CONE";
char str33[] = "H_INDEPENDENT_UNITS";
#if COSMOPARAM==1
char str34[] = "w0";
#elif COSMOPARAM==2
char str34[] = "w0";
char str35[] = "wa";
#elif COSMOPARAM==-1
char str34[] = "EXPANSION_FILE";
char str35[] = "INTERPOLATION_ORDER";
#endif

printf("Reading parameter file...\n");
while(!feof(param_file))
{
#ifdef USE_SINGLE_PRECISION
	fgets(c, BUFF_SIZE, param_file);
	if(strstr(c, str01) != NULL)
	{
		sscanf(c, "%*s\t%lf", &Omega_b);
	}
	if(strstr(c, str02) != NULL)
	{

		sscanf(c, "%*s\t%lf", &Omega_lambda);
	}
	if(strstr(c, str03) != NULL)
	{
		sscanf(c, "%*s\t%lf", &Omega_dm);
	}
	if(strstr(c, str04) != NULL)
	{
		sscanf(c, "%*s\t%lf", &Omega_r);
	}
	if(strstr(c, str05) != NULL)
	{
		sscanf(c, "%*s\t%lf", &H0);
		//We convert the km/s/Mpc to our unit system:
		H0 = H0/UNIT_V;
	}
	if(strstr(c, str06) != NULL)
	{
		sscanf(c, "%*s\t%s", IC_FILE);
	}
	if(strstr(c, str07) != NULL)
	{
		sscanf(c, "%*s\t%f", &L);
	}
	if(strstr(c, str08) != NULL)
	{
		sscanf(c, "%*s\t%i", &IS_PERIODIC);
		if(IS_PERIODIC > 3)
		{
			printf("Error: IS_PERIODIC > 3: No such boundary condition\n: IS_PERIODIC is set to 3");
			IS_PERIODIC = 3;
		}
	}

	if(strstr(c, str09) != NULL)
  {
    sscanf(c, "%*s\t%i", &OUTPUT_FORMAT);
#ifdef HAVE_HDF5
    if(OUTPUT_FORMAT != 0 && OUTPUT_FORMAT != 2)
    {
          printf("Error: Unkown output format %i. The supported formats are: 0 (ASCII) and 2 (HDF5).\nOUTPUT_FORMAT is set to 0 (ASCII).\n", OUTPUT_FORMAT);
          OUTPUT_FORMAT = 0;
    }
#else
		if(OUTPUT_FORMAT != 0)
		{
			printf("Error: Unkown output format %i. The only supported format is 0 (ASCII).\nOUTPUT_FORMAT is set to 0 (ASCII).\n", OUTPUT_FORMAT);
			OUTPUT_FORMAT = 0;
		}
#endif
  }
	if(strstr(c, str10) != NULL)
	{
		sscanf(c, "%*s\t%s", OUT_DIR);
		for(i=9; c[i] != '\n';i++)
    {
    }
		if(OUT_DIR[i-10]!='/')
    	OUT_DIR[i-9]='/';
	}
	if(strstr(c, str11) != NULL)
	{
		sscanf(c, "%*s\t%lf", &a_max);
	}
	if(strstr(c, str12) != NULL)
	{
		sscanf(c, "%*s\t%u", &N_snapshot);
	}
	if(strstr(c, str13) != NULL)
	{
		sscanf(c, "%*s\t%f", &ACC_PARAM);
	}
	if(strstr(c, str14) != NULL)
	{
		sscanf(c, "%*s\t%lf", &h_min);
	}
	if(strstr(c, str15) != NULL)
	{
		sscanf(c, "%*s\t%i", &COSMOLOGY);
	}
	if(strstr(c, str17) != NULL)
	{
		sscanf(c, "%*s\t%lf", &a_start);
		if(COSMOLOGY == 1)
			a_prev = a_start;
		else
			a_prev = 1;
	}
	if(strstr(c, str18) != NULL)
	{
		sscanf(c, "%*s\t%lf", &h_max);
	}
	if(strstr(c, str19) != NULL)
	{
		sscanf(c, "%*s\t%i", &COMOVING_INTEGRATION);
		if(COMOVING_INTEGRATION>1)
			COMOVING_INTEGRATION = 1;

		if(COMOVING_INTEGRATION<0)
			COMOVING_INTEGRATION = 0;
	}
	if(strstr(c, str20) != NULL)
        {
		sscanf(c, "%*s\t%lf", &TIME_LIMIT_IN_MINS);
	}
	if(strstr(c, str21) != NULL)
	{
		sscanf(c, "%*s\t%lf", &FIRST_T_OUT);
	}
	if(strstr(c, str22) != NULL)
	{
		sscanf(c, "%*s\t%i", &IC_FORMAT);
	}
	if(strstr(c, str23) != NULL)
  {
    sscanf(c, "%*s\t%i", &OUTPUT_TIME_VARIABLE);
  }
	if(strstr(c, str24) != NULL)
  {
		sscanf(c, "%*s\t%s", OUT_LST);
  }
	if(strstr(c, str25) != NULL)
	{
		sscanf(c, "%*s\t%lf", &H_OUT);
	}

	if(strstr(c, str26) != NULL)
	{
		sscanf(c, "%*s\t%f", &ParticleRadi);
	}
	if(strstr(c, str31) != NULL)
	{
		sscanf(c, "%*s\t%lf", &MIN_REDSHIFT);
	}
	if(strstr(c, str32) != NULL)
	{
		sscanf(c, "%*s\t%i", &REDSHIFT_CONE);
	}
	if(strstr(c, str33) != NULL)
	{
		sscanf(c, "%*s\t%i", &H0_INDEPENDENT_UNITS);
	}
	#if COSMOPARAM==1
	if(strstr(c, str34) != NULL)
	{
		sscanf(c, "%*s\t%lf", &w0);
	}
	#elif COSMOPARAM==2
	if(strstr(c, str34) != NULL)
	{
		sscanf(c, "%*s\t%lf", &w0);
	}
	if(strstr(c, str35) != NULL)
	{
		sscanf(c, "%*s\t%lf", &wa);
	}
	#elif COSMOPARAM==-1
	if(strstr(c, str34) != NULL)
	{
		sscanf(c, "%*s\t%s", EXPANSION_FILE);
	}
	if(strstr(c, str35) != NULL)
	{
		sscanf(c, "%*s\t%i", &INTERPOLATION_ORDER);
	}
	#endif
#else
	fgets(c, BUFF_SIZE, param_file);
  if(strstr(c, str01) != NULL)
  {
    sscanf(c, "%*s\t%lf", &Omega_b);
  }
  if(strstr(c, str02) != NULL)
  {
    sscanf(c, "%*s\t%lf", &Omega_lambda);
  }
  if(strstr(c, str03) != NULL)
  {
    sscanf(c, "%*s\t%lf", &Omega_dm);
  }
  if(strstr(c, str04) != NULL)
  {
    sscanf(c, "%*s\t%lf", &Omega_r);
  }
  if(strstr(c, str05) != NULL)
  {
    sscanf(c, "%*s\t%lf", &H0);
    //We convert the km/s/Mpc to our unit system:
    H0 = H0/UNIT_V;
  }
  if(strstr(c, str06) != NULL)
  {
    sscanf(c, "%*s\t%s", IC_FILE);
  }
  if(strstr(c, str07) != NULL)
  {
    sscanf(c, "%*s\t%lf", &L);
  }
  if(strstr(c, str08) != NULL)
  {
    sscanf(c, "%*s\t%i", &IS_PERIODIC);
    if(IS_PERIODIC > 3)
    {
      printf("Error: IS_PERIODIC > 3: No such boundary condition\n: IS_PERIODIC is set to 3");
      IS_PERIODIC = 3;
    }
  }
	if(strstr(c, str09) != NULL)
	{
		sscanf(c, "%*s\t%i", &OUTPUT_FORMAT);
		#ifdef HAVE_HDF5
		if(OUTPUT_FORMAT != 0 && OUTPUT_FORMAT != 2)
		{
			printf("Error: Unkown output format %i. The supported formats are: 0 (ASCII) and 2 (HDF5).\nOUTPUT_FORMAT is set to 0 (ASCII).\n", OUTPUT_FORMAT);
			OUTPUT_FORMAT = 0;
		}
		#else
		if(OUTPUT_FORMAT != 0)
		{
			printf("Error: Unkown output format %i. The only supported format is 0 (ASCII).\nOUTPUT_FORMAT is set to 0 (ASCII).\n", OUTPUT_FORMAT);
			OUTPUT_FORMAT = 0;
		}
		#endif
	}
	if(strstr(c, str10) != NULL)
  {
		sscanf(c, "%*s\t%s", OUT_DIR);
		for(i=9; c[i] != '\n';i++)
    {
    }
		if(OUT_DIR[i-10]!='/')
    	OUT_DIR[i-9]='/';
  }
  if(strstr(c, str11) != NULL)
  {
		sscanf(c, "%*s\t%lf", &a_max);
  }
	if(strstr(c, str12) != NULL)
	{
		sscanf(c, "%*s\t%u", &N_snapshot);
	}
  if(strstr(c, str13) != NULL)
  {
		sscanf(c, "%*s\t%lf", &ACC_PARAM);
  }
  if(strstr(c, str14) != NULL)
  {
		sscanf(c, "%*s\t%lf", &h_min);
  }
  if(strstr(c, str15) != NULL)
  {
    sscanf(c, "%*s\t%i", &COSMOLOGY);
  }
  if(strstr(c, str17) != NULL)
  {
		sscanf(c, "%*s\t%lf", &a_start);
    if(COSMOLOGY == 1)
      a_prev = a_start;
    else
      a_prev = 1;
  }
  if(strstr(c, str18) != NULL)
  {
    sscanf(c, "%*s\t%lf", &h_max);
  }
	if(strstr(c, str19) != NULL)
	{
		sscanf(c, "%*s\t%i", &COMOVING_INTEGRATION);
		if(COMOVING_INTEGRATION>1)
			COMOVING_INTEGRATION = 1;
		if(COMOVING_INTEGRATION<0)
			COMOVING_INTEGRATION = 0;
	}
	if(strstr(c, str20) != NULL)
  {
    sscanf(c, "%*s\t%lf", &TIME_LIMIT_IN_MINS);
  }
  if(strstr(c, str21) != NULL)
  {
    sscanf(c, "%*s\t%lf", &FIRST_T_OUT);
  }
  if(strstr(c, str22) != NULL)
  {
    sscanf(c, "%*s\t%i", &IC_FORMAT);
  }
	if(strstr(c, str23) != NULL)
  {
    sscanf(c, "%*s\t%i", &OUTPUT_TIME_VARIABLE);
  }
	if(strstr(c, str24) != NULL)
  {
      sscanf(c, "%*s\t%s", OUT_LST);
  }
  if(strstr(c, str25) != NULL)
  {
        sscanf(c, "%*s\t%lf", &H_OUT);
  }
	if(strstr(c, str26) != NULL)
  {
      sscanf(c, "%*s\t%lf", &ParticleRadi);
  }
	if(strstr(c, str31) != NULL)
	{
		sscanf(c, "%*s\t%lf", &MIN_REDSHIFT);
	}
	if(strstr(c, str32) != NULL)
	{
		sscanf(c, "%*s\t%i", &REDSHIFT_CONE);
	}
	if(strstr(c, str33) != NULL)
	{
		sscanf(c, "%*s\t%i", &H0_INDEPENDENT_UNITS);
	}
	#if COSMOPARAM==1
	if(strstr(c, str34) != NULL)
	{
		sscanf(c, "%*s\t%lf", &w0);
	}
	#elif COSMOPARAM==2
	if(strstr(c, str34) != NULL)
	{
		sscanf(c, "%*s\t%lf", &w0);
	}
	if(strstr(c, str35) != NULL)
	{
		sscanf(c, "%*s\t%lf", &wa);
	}
	#elif COSMOPARAM==-1
	if(strstr(c, str34) != NULL)
	{
		sscanf(c, "%*s\t%s", EXPANSION_FILE);
	}
	if(strstr(c, str35) != NULL)
	{
		sscanf(c, "%*s\t%i", &INTERPOLATION_ORDER);
	}
	#endif
#endif
}

printf("...done.\n\n");
fclose(param_file);
if(COSMOLOGY == 1)
{
	printf("Cosmological parameters:\n------------------------\nOmega_b\t\t%f\nOmega_lambda\t%f\nOmega_dm\t%f\nOmega_r\t\t%f\nOmega_m\t\t%f\nOmega_k\t\t%f\nH0\t\t%f(km/s)/Mpc\na_start\t\t%.14f\t(z_start = %f)\n",Omega_b, Omega_lambda, Omega_dm, Omega_r, Omega_b+Omega_dm, 1-Omega_b-Omega_lambda-Omega_dm-Omega_r, H0*UNIT_V, a_start, (1.0/a_start - 1.0));
	#if !defined(COSMOPARAM) || COSMOPARAM==0
	printf("\n");
	#elif COSMOPARAM==1
	printf("w0\t\t%f\n\n",w0);
	#elif COSMOPARAM==2
	printf("w0\t\t%f\nwa\t\t%f\n\n",w0,wa);
	#elif COSMOPARAM==-1
	printf("Expansion history file\t%s\n\n",EXPANSION_FILE);
	if (INTERPOLATION_ORDER<1 || INTERPOLATION_ORDER>3)
	{
		printf("Warning: Interpolation order is set to a non-defined value. Setting linear interpolation.\n");
		INTERPOLATION_ORDER = 1;
	}
	printf("Order of interpolation\t%i\n\n",INTERPOLATION_ORDER);
	#endif
	//Converting the Gy inputs into internal units
	h_min /= UNIT_T;
	h_max /= UNIT_T;
	printf("The parameters of the simulation:\n-------------------------\nBoundary condition\t\t%i\nLinear size\t\t\t%fMpc\nMaximal scale factor\t\t%f\nAccuracy parameter\t\t%.10f\nMinimal timestep length\t\t%.10fGy\nMaximal timestep length\t\t%.10fGy\nInitial conditions\t\t%s\nOutput directory\t\t%s\nComoving integration\t\t%i\n",IS_PERIODIC,L,a_max,ACC_PARAM,h_min*UNIT_T,h_max*UNIT_T,IC_FILE,OUT_DIR,COMOVING_INTEGRATION);
}
else
{
	printf("Non-cosmological simulation.\n");
	printf("The parameters of the simulation:\n-------------------------\nBoundary condition\t\t%i\nBox size\t\t\t%f\na_max\t\t\t\t%f\nParticleRadi\t\t\t%f\nAccuracy parameter\t\t%f\nMinimal timestep length\t\t%f\nInitial conditions\t\t%s\nOutput directory\t\t%s\n",IS_PERIODIC,L,a_max,ParticleRadi,ACC_PARAM,h_min,IC_FILE,OUT_DIR);
}
printf("Simulation wall-clock time limit\t%fh\n", TIME_LIMIT_IN_MINS/60.0);
if(COSMOLOGY==1)
	printf("H independent units\t\t%i\n", H0_INDEPENDENT_UNITS);
else
	H0_INDEPENDENT_UNITS=0; //Using H0 independent units in non-cosmological simulations makes no sense.
if(N_snapshot != 0)
{
	printf("Initial snapshot ID number:\t%u\n", N_snapshot);
}
printf("\n");
Hubble_param = 0.0;
return;
}

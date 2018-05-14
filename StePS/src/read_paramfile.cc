#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include "global_variables.h"

#define BUFF_SIZE 1024

#ifdef USE_SINGLE_PRECISION
typedef float REAL;
#else
typedef double REAL;
#endif

char in_file[BUFF_SIZE];

void read_param(FILE *param_file);


void read_param(FILE *param_file)
{
int i;
char c[BUFF_SIZE];
char str01[] = "Omega_b";
char str02[] = "Omega_lambda";
char str03[] = "Omega_dm";
char str04[] = "Omega_r";
char str05[] = "H0";
char str06[] = "IC_FILE";
char str07[] = "L_box";
char str08[] = "IS_PERIODIC";
char str09[] = "N_particle";
char str10[] = "OUT_DIR";
char str11[] = "a_max";
char str13[] = "mean_err";
char str14[] = "h_min";
char str15[] = "COSMOLOGY";
char str16[] = "Particle_mass";
char str17[] = "a_start";
char str18[] = "h_max";
char str19[] = "COMOVING_INTEGRATION";
char str21[] = "FIRST_T_OUT";
char str22[] = "IC_FORMAT";
char str23[] = "OUTPUT_FORMAT";
char str24[] = "OUT_LST";
char str25[] = "H_OUT";
char str26[] = "ParticleRadi";
char str27[] = "RESTART";
char str28[] = "T_restart";
char str29[] = "a_restart";
char str30[] = "H_restart";
char str31[] = "MIN_REDSHIFT";
char str32[] = "REDSHIFT_CONE";

printf("Reading parameter file...\n");
while(!feof(param_file))
{
#ifdef USE_SINGLE_PRECISION
	fgets(c, BUFF_SIZE, param_file);
	if(strstr(c, str01) != NULL)
	{
		sscanf(c, "%s\t%lf", str01, &Omega_b);
	}
	if(strstr(c, str02) != NULL)
	{

		sscanf(c, "%s\t%lf", str02, &Omega_lambda);
	}
	if(strstr(c, str03) != NULL)
	{
		sscanf(c, "%s\t%lf", str03, &Omega_dm);
	}
	if(strstr(c, str04) != NULL)
	{
		sscanf(c, "%s\t%lf", str04, &Omega_r);
	}
	if(strstr(c, str05) != NULL)
	{
		sscanf(c, "%s\t%lf", str05, &H0);
		//Converting km/s/Mpc to 1/Gy:
		H0 = H0*0.00102271216531915;
		////We convert the 1/Gy to our unit system:
		H0 = H0*47.1482347621227;
		
		
	}
	if(strstr(c, str06) != NULL)
	{
		for(i=9; c[i] != '\n';i++)
		{
			IC_FILE[i-9] = c[i];
		}
	}
	if(strstr(c, str07) != NULL)
	{
		sscanf(c, "%s\t%f", str07, &L);
	}
	if(strstr(c, str08) != NULL)
	{
		sscanf(c, "%s\t%i", str08, &IS_PERIODIC);
		if(IS_PERIODIC > 2)
		{
			printf("Error: IS_PERIODIC > 2: No such boundary condition\n: IS_PERIODIC is set to 2");
			IS_PERIODIC = 2;
		}
	}
	if(strstr(c, str09) != NULL)
	{
		sscanf(c, "%s\t%i", str09, &N);
	}
	if(strstr(c, str10) != NULL)
	{
		for(i=9; c[i] != '\n';i++)
		{
			OUT_DIR[i-9] = c[i];
		}
	}
	if(strstr(c, str11) != NULL)
	{
		sscanf(c, "%s\t%lf", str11, &a_max);
	}
	if(strstr(c, str13) != NULL)
	{
		sscanf(c, "%s\t%f", str13, &mean_err);
	}
	if(strstr(c, str14) != NULL)
	{
		sscanf(c, "%s\t%lf", str14, &h_min);
	}
	if(strstr(c, str15) != NULL)
	{
		sscanf(c, "%s\t%i", str15, &COSMOLOGY);
	}
	if(strstr(c, str16) != NULL)
	{
		sscanf(c, "%s\t%f", str16, &M_tmp);
	}
	if(strstr(c, str17) != NULL)
	{
		sscanf(c, "%s\t%lf", str17, &a_start);
		if(COSMOLOGY == 1)
			a_prev = a_start;
		else
			a_prev = 1;
	}
	if(strstr(c, str18) != NULL)
	{
		sscanf(c, "%s\t%lf", str18, &h_max);
	}
	if(strstr(c, str19) != NULL)
	{
		sscanf(c, "%s\t%i", str19, &COMOVING_INTEGRATION);
		if(COMOVING_INTEGRATION>1)
			COMOVING_INTEGRATION = 1;

		if(COMOVING_INTEGRATION<0)
			COMOVING_INTEGRATION = 0;
	}
	if(strstr(c, str21) != NULL)
	{
		sscanf(c, "%s\t%lf", str21, &FIRST_T_OUT);
	}
	if(strstr(c, str22) != NULL)
	{
		sscanf(c, "%s\t%i", str22, &IC_FORMAT);
	}
	if(strstr(c, str23) != NULL)
        {
                sscanf(c, "%s\t%i", str22, &OUTPUT_FORMAT);
        }
	if(strstr(c, str24) != NULL)
        {
                for(i=9; c[i] != '\n';i++)
                {
                        OUT_LST[i-9] = c[i];
                }
        }
	if(strstr(c, str25) != NULL)
	{
		sscanf(c, "%s\t\t%lf", str25, &H_OUT);
	}

	if(strstr(c, str26) != NULL)
	{
		sscanf(c, "%s\t%f", str26, &ParticleRadi);
	}
	if(strstr(c, str27) != NULL)
        {
                sscanf(c, "%s\t%i", str27, &RESTART);
        }
	if(strstr(c, str28) != NULL)
        {
                sscanf(c, "%s\t%lf", str28, &T_RESTART);
        }
	if(strstr(c, str29) != NULL)
        {
                sscanf(c, "%s\t%lf", str29, &A_RESTART);
        }
	if(strstr(c, str30) != NULL)
        {
                sscanf(c, "%s\t%lf", str30, &H_RESTART);
		H_RESTART = 0.0482190732645460*H_RESTART;
	}
	if(strstr(c, str31) != NULL)
	{
		sscanf(c, "%s\t%lf", str31, &MIN_REDSHIFT);
	}
	if(strstr(c, str32) != NULL)
	{
		sscanf(c, "%s\t%i", str32, &REDSHIFT_CONE);
	}
#else
	fgets(c, BUFF_SIZE, param_file);
        if(strstr(c, str01) != NULL)
        {
                sscanf(c, "%s\t%lf", str01, &Omega_b);
        }
        if(strstr(c, str02) != NULL)
        {

                sscanf(c, "%s\t%lf", str02, &Omega_lambda);
        }
        if(strstr(c, str03) != NULL)
        {
                sscanf(c, "%s\t%lf", str03, &Omega_dm);
        }
        if(strstr(c, str04) != NULL)
        {
                sscanf(c, "%s\t%lf", str04, &Omega_r);
        }
        if(strstr(c, str05) != NULL)
        {
                sscanf(c, "%s\t%lf", str05, &H0);
                //Converting km/s/Mpc to 1/Gy:
                H0 = H0*0.00102271216531915;
                ////We convert the 1/Gy to our unit system:
                H0 = H0*47.1482347621227;


        }
        if(strstr(c, str06) != NULL)
        {
                for(i=9; c[i] != '\n';i++)
                {
                        IC_FILE[i-9] = c[i];
                }
        }
        if(strstr(c, str07) != NULL)
        {
                sscanf(c, "%s\t%lf", str07, &L);
        }
        if(strstr(c, str08) != NULL)
        {
                sscanf(c, "%s\t%i", str08, &IS_PERIODIC);
                if(IS_PERIODIC > 2)
                {
                        printf("Error: IS_PERIODIC > 2: No such boundary condition\n: IS_PERIODIC is set to 2");
                        IS_PERIODIC = 2;
                }
        }
        if(strstr(c, str09) != NULL)
        {
                sscanf(c, "%s\t%i", str09, &N);
        }
	if(strstr(c, str10) != NULL)
        {
                for(i=9; c[i] != '\n';i++)
                {
                        OUT_DIR[i-9] = c[i];
                }
        }
        if(strstr(c, str11) != NULL)
        {
                sscanf(c, "%s\t%lf", str11, &a_max);
        }
        if(strstr(c, str13) != NULL)
        {
                sscanf(c, "%s\t%lf", str13, &mean_err);
        }
        if(strstr(c, str14) != NULL)
        {
                sscanf(c, "%s\t%lf", str14, &h_min);
        }
        if(strstr(c, str15) != NULL)
        {
                sscanf(c, "%s\t%i", str15, &COSMOLOGY);
        }
        if(strstr(c, str16) != NULL)
        {
                sscanf(c, "%s\t%lf", str16, &M_tmp);
        }
        if(strstr(c, str17) != NULL)
        {
                sscanf(c, "%s\t%lf", str17, &a_start);
                if(COSMOLOGY == 1)
                        a_prev = a_start;
                else
                        a_prev = 1;
        }
        if(strstr(c, str18) != NULL)
        {
                sscanf(c, "%s\t%lf", str18, &h_max);
        }
	if(strstr(c, str19) != NULL)
	{
		sscanf(c, "%s\t%i", str19, &COMOVING_INTEGRATION);
		if(COMOVING_INTEGRATION>1)
			COMOVING_INTEGRATION = 1;

		if(COMOVING_INTEGRATION<0)
			COMOVING_INTEGRATION = 0;
	}
        if(strstr(c, str21) != NULL)
        {
                sscanf(c, "%s\t%lf", str21, &FIRST_T_OUT);
        }
        if(strstr(c, str22) != NULL)
        {
                sscanf(c, "%s\t%i", str22, &IC_FORMAT);
        }
	if(strstr(c, str23) != NULL)
        {
                sscanf(c, "%s\t%i", str22, &OUTPUT_FORMAT);
        }
	if(strstr(c, str24) != NULL)
        {
                for(i=9; c[i] != '\n';i++)
                {
                        OUT_LST[i-9] = c[i];
                }
        }
        if(strstr(c, str25) != NULL)
        {
                sscanf(c, "%s\t\t%lf", str25, &H_OUT);
        }

        if(strstr(c, str26) != NULL)
        {
                sscanf(c, "%s\t%lf", str26, &ParticleRadi);
        }
        if(strstr(c, str27) != NULL)
        {
                sscanf(c, "%s\t%i", str27, &RESTART);
        }
	if(strstr(c, str28) != NULL)
        {
                sscanf(c, "%s\t%lf", str28, &T_RESTART);
        }
        if(strstr(c, str29) != NULL)
        {
                sscanf(c, "%s\t%lf", str29, &A_RESTART);
        }
        if(strstr(c, str30) != NULL)
        {
                sscanf(c, "%s\t%lf", str30, &H_RESTART);
                H_RESTART = 0.0482190732645460*H_RESTART;
        }
	if(strstr(c, str31) != NULL)
	{
		sscanf(c, "%s\t%lf", str31, &MIN_REDSHIFT);
	}
	if(strstr(c, str32) != NULL)
	{
		sscanf(c, "%s\t%i", str32, &REDSHIFT_CONE);
	}
#endif
}
	
printf("...done.\n");
fclose(param_file);
if(COSMOLOGY == 1)
{
	printf("Cosmological parameters:\n------------------------\nOmega_b\t\t%f\nOmega_lambda\t%f\nOmega_dm\t%f\nOmega_r\t\t%f\nOmega_m\t\t%f\nOmega_k\t\t%f\nH0\t\t%f(km/s)/Mpc\na_start\t\t%f\n",Omega_b, Omega_lambda, Omega_dm, Omega_r, Omega_b+Omega_dm, 1-Omega_b-Omega_lambda-Omega_dm-Omega_r, H0*20.7386814448645, a_start);
	printf("COMOVING_INTEGRATION\t%i\n\n", COMOVING_INTEGRATION);
}
else
{
	printf("Non cosmological simulation.\n---------------------------\nParticle masses:\t%f\n\n", M_tmp);
}
	printf("The parameters of the simulation:\n-------------------------\nBoundary condition\t\t%i\nBox size\t\t\t%fMpc\nNumber of particles\t\t%i\na_max\t\t\t\t%f\nPrescribed error\t\t%f\nMinimal timestep length\t\t%f\nInitial conditions\t\t%s\nOutput directory\t\t%s\n\n",IS_PERIODIC,L,N,a_max,mean_err,h_min,IC_FILE,OUT_DIR);
	Hubble_param = 0.0;
	if(COSMOLOGY == 0 || COMOVING_INTEGRATION == 0)
	{
		Hubble_param = 0;
	}
return;
}



#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include "kdtree.h"
#include "global_variables.h"

#define BUFF_SIZE 1024

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
char str08[] = "N_particle";
char str09[] = "OUT_FILE";
char str10[] = "a_max";
char str11[] = "Particle_mass";
char str12[] = "a_start";
char str13[] = "startH_0";
char str14[] = "IC_FORMAT";
char str15[] = "SPHERE_DIAMETER";
char str16[] = "R_CUT";
char str17[] = "N_SIDE";
char str18[] = "R_GRID";
char str19[] = "FOR_COMOVING";
char str20[] = "RANDOM_SEED";
char str21[] = "NUMBER_OF_INPUT_FILES";
char str22[] = "N_IC_tot";
char str23[] = "RANDOM_ROTATION";
char str24[] = "dist_unit_in_kpc";
char str25[] = "VOI_X";
char str26[] = "VOI_Y";
char str27[] = "VOI_Z";
char str28[] = "TileFac";
char str29[] = "SphericalGlassFILE";

printf("Reading parameter file...\n");
while(!feof(param_file))
{
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
                //Converting km/s/Mpc to our unit system:
                H0 = H0 /  UNIT_V;


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
                sscanf(c, "%s\t%llu", str08, &N);
        }
	if(strstr(c, str09) != NULL)
        {
                for(i=9; c[i] != '\n';i++)
                {
                        OUT_FILE[i-9] = c[i];
                }
        }
        if(strstr(c, str10) != NULL)
        {
                sscanf(c, "%s\t%lf", str10, &a_max);
        }
        if(strstr(c, str11) != NULL)
        {
                sscanf(c, "%s\t%lf", str11, &M_tmp);
        }
        if(strstr(c, str12) != NULL)
        {
                sscanf(c, "%s\t%lf", str12, &a_start);
        }
        if(strstr(c, str13) != NULL)
        {
                sscanf(c, "%s\t%lf", str13, &H0_start);
                //We convert the km/s/Mpc to our unit system
                H0_start = H0_start /  UNIT_V;
                Hubble_param = H0_start;
        }

        if(strstr(c, str14) != NULL)
        {
                sscanf(c, "%s\t%i", str14, &IC_FORMAT);
        }
	if(strstr(c, str15) != NULL)
        {
                sscanf(c, "%s\t%lf", str15, &SPHERE_DIAMETER);
        }
	if(strstr(c, str16) != NULL)
        {
                sscanf(c, "%s\t\t%lf", str16, &R_CUT);
        }
	if(strstr(c, str17) != NULL)
        {
                sscanf(c, "%s\t\t%i", str17, &N_SIDE);
        }
	if(strstr(c, str18) != NULL)
        {
                sscanf(c, "%s\t\t%i", str18, &R_GRID);
        }
	if(strstr(c, str19) != NULL)
        {
                sscanf(c, "%s\t\t%i", str19, &FOR_COMOVING_INTEGRATION);
        }
	if(strstr(c, str20) != NULL)
        {
                sscanf(c, "%s\t\t%i", str20, &RANDOM_SEED);
        }
	if(strstr(c, str21) != NULL)
	{
		sscanf(c, "%s\t\t%i", str21, &NUMBER_OF_INPUT_FILES);
	}
	if(strstr(c, str22) != NULL)
	{
		sscanf(c, "%s\t\t%llu", str22, &N_IC_tot);
	}
	if(strstr(c, str23) != NULL)
        {
                sscanf(c, "%s\t\t%i", str23, &RANDOM_ROTATION);
        }
	if(strstr(c, str24) != NULL)
        {

                sscanf(c, "%s\t%lf", str24, &dist_unit_in_kpc);
        }
	if(strstr(c, str25) != NULL)
        {

                sscanf(c, "%s\t%lf", str25, &VOI[0]);
        }
	if(strstr(c, str26) != NULL)
        {

                sscanf(c, "%s\t%lf", str26, &VOI[1]);
        }
	if(strstr(c, str27) != NULL)
        {

                sscanf(c, "%s\t%lf", str27, &VOI[2]);
        }
	if(strstr(c, str28) != NULL)
        {

                sscanf(c, "%s\t%i", str28, &TILEFAC);
		if(TILEFAC < 1)
		{
			TILEFAC = 1;
			printf("Error: TileFac should be larger than zero. TileFac set to %i.\n", TILEFAC);
		}
		else if((TILEFAC % 2) != 1)
		{
			TILEFAC++;
			printf("Error: TileFac should be an odd number. TileFac set to %i.\n", TILEFAC);
		}
        }
	if(strstr(c, str29) != NULL)
	{
		for(i=19; c[i] != '\n';i++)
		{
			SphericalGlassFILE[i-19] = c[i];
		}
		char None[512] = "None";
		if( strcmp(SphericalGlassFILE, None) != 0)
		{
			SPHERICAL_GLASS = 1;
		}
	}
}
printf("...done.\n\n");
fclose(param_file);
printf("The readed parameters:\n\n");
printf("Cosmological parameters:\n------------------------\nOmega_b\t\t%f\nOmega_lambda\t%f\nOmega_dm\t%f\nOmega_r\t\t%f\nOmega_m\t\t%f\nOmega_k\t\t%f\nH0\t\t%f(km/s)/Mpc\na_start\t\t%f\n\n",Omega_b, Omega_lambda, Omega_dm, Omega_r, Omega_b+Omega_dm, 1-Omega_b-Omega_lambda-Omega_dm-Omega_r, H0 * UNIT_V, a_start);
printf("Parameters of the IC file:\n--------------------------\n");
printf("Particle masses:\t\t%f\n", M_tmp);
printf("Box size\t\t\t%fMpc\nNumber of particles\t\t%llu\na_max\t\t\t\t%f\nInitial conditions\t\t%s\nOutput file\t\t\t%s\nSPHERE_DIAMETER\t\t\t%f\nR_CUT\t\t\t\t%f\nN_SIDE\t\t\t\t%i\nR_GRID\t\t\t\t%i\nFOR_COMOVING_INTEGRATION\t%i\nNUMBER_OF_INPUT_FILES\t\t%i\nN_IC_tot\t\t\t%llu\nTileFac\t\t\t\t%i\n",L,N,a_max,IC_FILE,OUT_FILE, SPHERE_DIAMETER, R_CUT, N_SIDE, R_GRID, FOR_COMOVING_INTEGRATION, NUMBER_OF_INPUT_FILES, N_IC_tot, TILEFAC);
if(SPHERICAL_GLASS == 0)
{
	printf("\n\nWARNING: The code will not use spherical glass. This can cause errors in the simulations, if the initial redshift is too low!\n\n");
}
else
{
	printf("Spherical Glass file\t\t%s\n",SphericalGlassFILE);
}
if(RANDOM_ROTATION == 1)
{
	printf("Random rotation is\t\ton.\nRANDOM_SEED\t\t\t%i\n",  RANDOM_SEED);
}
else
{
	printf("Random rotation is\t\toff\n");
	RANDOM_ROTATION = 0;
}
printf("The coordinates of the center of Volume-Of-Interest(VOI):\nVOI_x = %fMpc\tVOI_y = %fMpc\tVOI_z = %fMpc\n\n",VOI[0], VOI[1], VOI[2]);
return;
}

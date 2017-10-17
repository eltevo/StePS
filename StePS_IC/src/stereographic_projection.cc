#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <time.h>
#include <chealpix.h>
#include <chealpix.c>
#include "global_variables.h"

//projecting the 3D coordinates into the surface of a 4D sphere
void stereographic_projection(double** x, double* M)
{
	int i,j,chunk, index;
	long healpix_index, omega_index, N_SPHERE;
	double r, omega, theta, phi;
	double omega_CUT = 2*atan(R_CUT/SPHERE_DIAMETER);
	chunk = (N-1)/omp_get_max_threads();
	printf("Projecting and using HEALPix...\n");
	printf("Number of particles = %i\n", N);
	N_SPHERE = nside2npix((long)N_SIDE);
	printf("N_SIDE=\t\t\t %i\nNumber of HEALPix indexes=\t %i\n\n", N_SIDE, (int)N_SPHERE);

if(RANDOM_ROTATION == 1)
{
	//Setting up the random rotation in the (theta, phi) grid
	double** RAND_ROT_TABLE;
	double a_x, a_y, a_z, a_r, x_rot, y_rot, z_rot, k_v;
	RAND_ROT_TABLE = (double**)malloc(R_GRID*sizeof(double*));
	for(i=0;i<R_GRID;i++)
	{
		RAND_ROT_TABLE[i] = (double*)malloc(5*sizeof(double));
	}
	srand(RANDOM_SEED);
	for(i=0;i<R_GRID;i++)
	{
		a_x = ((double)rand()/(double)RAND_MAX);
		a_y = ((double)rand()/(double)RAND_MAX);
		a_z = ((double)rand()/(double)RAND_MAX);
		a_r = sqrt(a_x*a_x + a_y*a_y + a_z*a_z);
		RAND_ROT_TABLE[i][0] = a_x/a_r; //x component of the rotation axis
		RAND_ROT_TABLE[i][1] = a_y/a_r; //y component of the rotation axis
		RAND_ROT_TABLE[i][2] = a_z/a_r; //z component of the rotation axis
		RAND_ROT_TABLE[i][3] = pi*((double)rand()/(double)RAND_MAX)/(double)N_SIDE; //angle of rotation
		RAND_ROT_TABLE[i][4] = sin(RAND_ROT_TABLE[i][3]); //sin of the angle of rotation
		RAND_ROT_TABLE[i][3] = cos(RAND_ROT_TABLE[i][3]); //cos of the angle of rotation
	}

	
	#pragma omp parallel default(shared)  private(r, omega, theta, phi, healpix_index, omega_index, index, j, k_v, x_rot, y_rot, z_rot)
	{
	#pragma omp for schedule(dynamic,chunk)
	for(i=0;i<N;i++)
	{
		for(j = 0; j<3; j++)
		{
			x[i][j] = x[i][j]-L*0.5;
		}
		r = sqrt(x[i][0]*x[i][0] + x[i][1]*x[i][1] + x[i][2]*x[i][2]);
		omega = 2*atan(r/SPHERE_DIAMETER);
		
		omega_index= (long)floor((omega/omega_CUT) * (double) R_GRID);

		if(omega_index<(long)R_GRID)
		{
			//rotation using Rodrigues' rotation formula
			k_v = x[i][0]*RAND_ROT_TABLE[omega_index][0] + x[i][1]*RAND_ROT_TABLE[omega_index][1] + x[i][2]*RAND_ROT_TABLE[omega_index][2];
			x_rot = x[i][0]*RAND_ROT_TABLE[omega_index][3] + (RAND_ROT_TABLE[omega_index][1]*x[i][2] - RAND_ROT_TABLE[omega_index][2]*x[i][1])*RAND_ROT_TABLE[omega_index][4] + RAND_ROT_TABLE[omega_index][0]*k_v*(1-RAND_ROT_TABLE[omega_index][3]);
			y_rot = x[i][1]*RAND_ROT_TABLE[omega_index][3] + (RAND_ROT_TABLE[omega_index][2]*x[i][0] - RAND_ROT_TABLE[omega_index][0]*x[i][2])*RAND_ROT_TABLE[omega_index][4] + RAND_ROT_TABLE[omega_index][1]*k_v*(1-RAND_ROT_TABLE[omega_index][3]);
			z_rot = x[i][2]*RAND_ROT_TABLE[omega_index][3] + (RAND_ROT_TABLE[omega_index][0]*x[i][1] - RAND_ROT_TABLE[omega_index][1]*x[i][0])*RAND_ROT_TABLE[omega_index][4] + RAND_ROT_TABLE[omega_index][2]*k_v*(1-RAND_ROT_TABLE[omega_index][3]);
			theta = acos(z_rot/r);
			phi = atan2(y_rot,x_rot);

			//theta = acos(x[i][2]/r);
			//phi = atan2(x[i][1],x[i][0]);
			healpix_index=ang2pix_ring_z_phi((long)N_SIDE, cos(theta), phi);
			index = (((int)N_SPHERE)*(int)omega_index)+(int)healpix_index;
			for(j=0;j<6;j++)
			{
				//coordinates and velocities
				#pragma omp atomic
				x_out[index][j]+=x[i][j]*M[i];

			}
			//masses
			#pragma omp atomic
			x_out[index][6] += M[i];

			#pragma omp atomic
			COUNT[index] += 1;
		}
		
	}
	}
}
else
{
	#pragma omp parallel default(shared)  private(r, omega, theta, phi, healpix_index, omega_index, index, j)
	{
	#pragma omp for schedule(dynamic,chunk)
	for(i=0;i<N;i++)
	{
		for(j = 0; j<3; j++)
		{
			x[i][j] = x[i][j]-L*0.5;
		}
		r = sqrt(x[i][0]*x[i][0] + x[i][1]*x[i][1] + x[i][2]*x[i][2]);
		omega = 2*atan(r/SPHERE_DIAMETER);

		omega_index= (long)floor((omega/omega_CUT) * (double) R_GRID);
		if(omega_index<(long)R_GRID)
              	{
			theta = acos(x[i][2]/r);
			phi = atan2(x[i][1],x[i][0]);
			healpix_index=ang2pix_ring_z_phi((long)N_SIDE, cos(theta), phi);
			index = (((int)N_SPHERE)*(int)omega_index)+(int)healpix_index;
			for(j=0;j<6;j++)
			{
				//coordinates and velocities
				#pragma omp atomic
				x_out[index][j]+=x[i][j]*M[i];

			}
			//masses
			#pragma omp atomic
			x_out[index][6] += M[i];

			#pragma omp atomic
			COUNT[index] += 1;
		}
	}	
	}
}
return;
}

void finish_stereographic_projection(double** x_out)
{
	int i,j, chunk;
	//Finishing the calculation of the center of mass and the velocity in each cell
	chunk = (N-1)/omp_get_max_threads();
	#pragma omp parallel default(shared)
        {
        #pragma omp for schedule(dynamic,chunk) private(j)
        for(i=0;i<N_out;i++)
        {
		for(j=0;j<6;j++)
		{
			#pragma omp atomic
			x_out[i][j] /= x_out[i][6];
		}
	}
	}


	return;
}

void add_hubble_flow(double** x_out, int N_out, double H0_start)
{
	int i, j, chunk;
	double a_tmp = a_start/a_max;
	printf("Initial scalefactor: a_start = %lf\n", a_tmp);
	chunk = (N-1)/omp_get_max_threads();
	#pragma omp parallel default(shared)  private(j)
        {
        #pragma omp for schedule(dynamic,chunk)
        for(i=0;i<N_out;i++)
        {
		
		for(j=0;j<3;j++)
		{
			x_out[i][j] *= a_tmp;
			//Gadget format: http://wwwmpa.mpa-garching.mpg.de/gadget/gadget-list/0113.html
			x_out[i][j+3] *= sqrt(a_tmp); //converting the peculiar speeds 
			x_out[i][j+3] += x_out[i][j]*H0_start;
		}
	}
	}
return;
}

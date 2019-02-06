#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <time.h>
#include <chealpix.h>
#include <chealpix.c>
#include <kdtree.h>
#include "global_variables.h"

//building up the kdtree
void make_kdtree()
{
	printf("Creating the 3 dimensional k-d tree for the fast search in the spherical glass...\n");
	int i;
	tree = kd_create( 3 ); //create k-d free for 3D points
	index_list=(int*)malloc(N_glass*sizeof(int)); //allocating memory for the index list
	for(i=0;i<N_glass;i++)
	{
		index_list[i] = i;
		kd_insert3(tree, SphericalGlass[i][0],SphericalGlass[i][1],SphericalGlass[i][2], &index_list[i]);
	}
	printf("...done\n");
	
}

//calculating the distance between two vectors (in unitsphere, in radians)
double distance(double glass_x,double glass_y,double glass_z, double IC_x,double IC_y,double IC_z)
{
	//the dot product of two unit vector is the cosine of the angle between the two vectors
	//We assume here, that the two input vector is normalised 
	return acos(IC_x*glass_x + IC_y*glass_y + IC_z*glass_z);
}

//finding the nearest glass particle using the kdtree
int coord2index(double IC_x,double IC_y,double IC_z, double r)
{
	int i,index = -1;
	int *dat;
	double IC_vec[3], pos[3];
	double dist;
	double dist_min = pi;
	IC_vec[0] = IC_x/r;
	IC_vec[1] = IC_y/r;
	IC_vec[2] = IC_z/r;
	double radi = 1.5*sqrt(4.0/(double)N_glass);
	struct kdres *results;
	results = kd_nearest_range( tree, IC_vec, radi );
	while( !kd_res_end( results ) )
	{
		dat = (int *) kd_res_item( results, pos );
		i = (int) *dat;
		dist = distance(IC_vec[0],IC_vec[1],IC_vec[2], pos[0], pos[1], pos[2]);
		if(dist < dist_min)
		{
			dist_min = dist;
			index = i;
		}
		kd_res_next( results );
	}
	if(index == -1)
		printf("Warning: the k-d tree did not found any neighbour glass particle for the (%f,%f,%f) point!\n", IC_vec[0], IC_vec[1], IC_vec[2]);
	kd_res_free(results);
	return index;
}

//projecting the 3D coordinates into the surface of a 4D sphere
void stereographic_projection(double** x, double* M)
{
	unsigned long long int i;
	int j, index;
	long spherical_index, omega_index, N_SPHERE;
	double r, omega, theta, phi;
	double omega_CUT = 2*atan(R_CUT/SPHERE_DIAMETER);
	int shift_x, shift_y, shift_z;
        double x_actual[3];
	printf("Projecting and binning in the compact space...\n");
	printf("Number of particles = %llu\n", N);
	N_SPHERE = nside2npix((long)N_SIDE);
	printf("N_SIDE=\t\t\t %i\nNumber of spherical indexes=\t %i\n\n", N_SIDE, (int)N_SPHERE);

if(RANDOM_ROTATION == 1)
{
	//Setting up the random rotation in the (theta, phi) grid
	double** RAND_ROT_TABLE;
	double a_x, a_y, a_z, a_r, x_rot, y_rot, z_rot, k_v;
	RAND_ROT_TABLE = (double**)malloc(R_GRID*sizeof(double*));
	for(i=0;i<(unsigned long long int)R_GRID;i++)
	{
		RAND_ROT_TABLE[i] = (double*)malloc(5*sizeof(double));
	}
	srand(RANDOM_SEED);
	for(i=0;i<(unsigned long long int)R_GRID;i++)
	{
		a_x = ((double)rand()/(double)RAND_MAX);
		a_y = ((double)rand()/(double)RAND_MAX);
		a_z = ((double)rand()/(double)RAND_MAX);
		a_r = sqrt(a_x*a_x + a_y*a_y + a_z*a_z);
		RAND_ROT_TABLE[i][0] = a_x/a_r; //x component of the rotation axis
		RAND_ROT_TABLE[i][1] = a_y/a_r; //y component of the rotation axis
		RAND_ROT_TABLE[i][2] = a_z/a_r; //z component of the rotation axis
		RAND_ROT_TABLE[i][3] = pi*((double)rand()/(double)RAND_MAX - 0.5); //angle of rotation
		RAND_ROT_TABLE[i][4] = sin(RAND_ROT_TABLE[i][3]); //sin of the angle of rotation
		RAND_ROT_TABLE[i][3] = cos(RAND_ROT_TABLE[i][3]); //cos of the angle of rotation
	}

	for(i=0;i<N;i++)
	{
		for(j=0; j<3; j++)
			x_actual[j] = x[i][j];
		//Using multiple images
		for(shift_x = -1*(TILEFAC-1)/2; shift_x <= (TILEFAC-1)/2; shift_x++)
		{
		for(shift_y = -1*(TILEFAC-1)/2; shift_y <= (TILEFAC-1)/2; shift_y++)
		{
		for(shift_z = -1*(TILEFAC-1)/2; shift_z <= (TILEFAC-1)/2; shift_z++)
		{
			//shifting the particles
			x[i][0] = x_actual[0]-L*0.5 + ((double) shift_x) * L;
			x[i][1] = x_actual[1]-L*0.5 + ((double) shift_y) * L;
			x[i][2] = x_actual[2]-L*0.5 + ((double) shift_z) * L;
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
				if(SPHERICAL_GLASS == 0)
				{
					theta = atan2(sqrt(x_rot*x_rot+y_rot*y_rot),z_rot); //see vec2ang function
					phi = atan2(y_rot,x_rot);
					spherical_index=ang2pix_ring_z_phi((long)N_SIDE, cos(theta), phi);
				}
				else
				{
					spherical_index = coord2index(x_rot, y_rot, z_rot, r);
				}
				if( (int)spherical_index >= (int)N_SPHERE)
				{
					printf("Warning: Invalid spherical index!\nspherical_index = %i\t N_SPHERE = %i\n", (int)spherical_index, (int)N_SPHERE);
				}
				index = (((int)N_SPHERE)*(int)omega_index)+(int)spherical_index;
				for(j=0;j<6;j++)
				{
					//coordinates and velocities
					x_out[index][j]+=x[i][j]*M[i];
	
				}
				//masses
				x_out[index][6] += M[i];
				COUNT[index] += 1;
			}
		}
		}
		}
		
	}
}
else
{
	for(i=0;i<N;i++)
	{
		for(j=0; j<3; j++)
                        x_actual[j] = x[i][j];
		//Using multiple images
		for(shift_x = -1*(TILEFAC-1)/2; shift_x <= (TILEFAC-1)/2; shift_x++)
		{
		for(shift_y = -1*(TILEFAC-1)/2; shift_y <= (TILEFAC-1)/2; shift_y++)
		{
		for(shift_z = -1*(TILEFAC-1)/2; shift_z <= (TILEFAC-1)/2; shift_z++)
		{
			//shifting the particles
			x[i][0] = x_actual[0]-L*0.5 + ((double) shift_x) * L;
			x[i][1] = x_actual[1]-L*0.5 + ((double) shift_y) * L;
			x[i][2] = x_actual[2]-L*0.5 + ((double) shift_z) * L;

			r = sqrt(x[i][0]*x[i][0] + x[i][1]*x[i][1] + x[i][2]*x[i][2]);
			omega = 2*atan(r/SPHERE_DIAMETER);
	
			omega_index= (long)floor((omega/omega_CUT) * (double) R_GRID);
			if(omega_index<(long)R_GRID)
			{
				if(SPHERICAL_GLASS == 0)
				{
					theta = atan2(sqrt(x[i][0]*x[i][0] + x[i][1]*x[i][1]), x[i][2]); //see vec2ang function
					phi = atan2(x[i][1],x[i][0]);
					spherical_index=ang2pix_ring_z_phi((long)N_SIDE, cos(theta), phi);
				}
				else
				{
					spherical_index = coord2index(x[i][0],x[i][1],x[i][2], r);
				}
				if( (int)spherical_index >= (int)N_SPHERE)
                                {
					printf("Warning: Invalid spherical index!\nspherical_index = %i\t N_SPHERE = %i\n", (int)spherical_index, (int)N_SPHERE);
                                }
				index = (((int)N_SPHERE)*(int)omega_index)+(int)spherical_index;
				for(j=0;j<6;j++)
				{
					//coordinates and velocities
					x_out[index][j]+=x[i][j]*M[i];
	
				}
				//masses
				x_out[index][6] += M[i];
				COUNT[index] += 1;
			}
		}
		}
		}
	}	
}
return;
}

void finish_stereographic_projection(double** x_out)
{
	unsigned long long int i;
	int j, chunk;
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
			if(j>2)
			{
				#pragma omp atomic
				x_out[i][j] *= UNIT_V;
			}
		}
	}
	}


	return;
}

void add_hubble_flow(double** x_out, unsigned long int N_out, double H0_start)
{
	unsigned long long int i;
	int j, chunk;
	double a_tmp = a_start/a_max;
	printf("Initial scalefactor: a_start = %f\n", a_tmp);
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
			x_out[i][j+3] += x_out[i][j]*H0_start*UNIT_V; //adding the Hubble-flow
		}
	}
	}
return;
}

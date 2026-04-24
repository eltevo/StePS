/********************************************************************************/
/*  StePS - STEreographically Projected cosmological Simulations                */
/*    Copyright (C) 2017-2026 Gabor Racz, Balazs Pal, Viola Varga               */
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
#include <math.h>
#include <omp.h>
#include <time.h>
#include "mpi.h"
#include "global_variables.h"

#define RADIAL_BH_FORCE_ORDER 1 // Order of the radial BH force correction table interpolation

extern int e[2202][4];
extern REAL w[3];
extern int N, el;


#if defined(PERIODIC)
//Functions for T^3 Ewald force correction
void ewald_interpolate_D(int Ngrid, REAL L, const REAL *T3_EWALD_FORCE_TABLE, const REAL dx, const REAL dy, const REAL dz, REAL D[3], int order);
#endif

#if defined(PERIODIC_Z)
//Functions for S^1 x R^2 Ewald force correction
void ewald_interpolate_D( const REAL* T, int Nrho, int Nz, REAL rho_max, REAL Lz, REAL dx, REAL dy, REAL dz, REAL &Fx, REAL &Fy, REAL &Fz);
#endif

void recalculate_softening();

void recalculate_softening()
{
	beta = ParticleRadi;
	if(COSMOLOGY ==1)
	{
		rho_part = M_min/(4.0*pi*pow(beta, 3.0) / 3.0);
	}
}

REAL force_softening(REAL r, REAL beta)
{
	//This function calculates the softened force coefficient between two particles
	//Only cubic spline softening is implemented here. New softening types can be added later.
	//Input:
	//    * r - distance between the two particles
	//    * beta - softening length
	//Output:
	//    * wij - softened force coefficient (1/r^3 for non-softened force)
	REAL SOFT_CONST[5];
	REAL betap2 = beta*0.5;
	REAL wij;
	wij = 0;
	if(r >= beta)
	{
		wij = pow(r, -3);
	}
	else if(r > betap2 && r < beta)
	{
		SOFT_CONST[0] = -32.0/(3.0*pow(beta, 6));
		SOFT_CONST[1] = 38.4/pow(beta, 5);
		SOFT_CONST[2] = -48.0/pow(beta, 4);
		SOFT_CONST[3] = 64.0/(3.0*pow(beta, 3));
		SOFT_CONST[4] = -1.0/15.0;
		wij = SOFT_CONST[0]*pow(r, 3)+SOFT_CONST[1]*pow(r, 2)+SOFT_CONST[2]*r+SOFT_CONST[3]+SOFT_CONST[4]/pow(r, 3);
	}
	else
	{
		SOFT_CONST[0] = 32.0/pow(beta, 6);
		SOFT_CONST[1] = -38.4/pow(beta, 5);
		SOFT_CONST[2] = 32.0/(3.0*pow(beta, 3));

		wij = SOFT_CONST[0]*pow(r, 3)+SOFT_CONST[1]*pow(r, 2)+SOFT_CONST[2];
	}
	return wij;
}

#if defined(PERIODIC_Z) || defined(USE_BH)
//These interpolators are defined in the utils.cc file
// They are used to interpolate the force table values in radial BH or cylindrical force calculations
REAL linear_interpolation(REAL X, REAL X1, REAL Y1, REAL X2, REAL Y2);
REAL quadratic_interpolation(REAL X, REAL X1, REAL Y1, REAL X2, REAL Y2, REAL X3, REAL Y3);
REAL cubic_interpolation(REAL X, REAL X1, REAL Y1, REAL X2, REAL Y2, REAL X3, REAL Y3, REAL X4, REAL Y4);
#endif

#if defined(USE_BH)
// Oct-tree struct
typedef struct OctreeNode
{
	REAL cx, cy, cz;		  // center of the cube
	REAL nodesize;			      // length of the cube
	REAL mass;
	REAL com_x, com_y, com_z;  // center of mass
	int particle_index;		   // -1 if internal node
	struct OctreeNode *children[8];
} OctreeNode;


OctreeNode* create_node(REAL cx, REAL cy, REAL cz, REAL nodesize)
{
    OctreeNode *node = (OctreeNode*)malloc(sizeof(OctreeNode));
    node->cx = cx; node->cy = cy; node->cz = cz;
    node->nodesize = nodesize;
    node->mass = 0;
    node->com_x = node->com_y = node->com_z = 0;
    node->particle_index = -1;
    for (int i = 0; i < 8; i++) node->children[i] = NULL;
    return node;
}

int get_octant(OctreeNode *node, REAL *X, int i)
{
    int index = 0;
    if (X[3*i]     > node->cx) index |= 1;
    if (X[3*i + 1] > node->cy) index |= 2;
    if (X[3*i + 2] > node->cz) index |= 4;
    return index;
}

void insert_particle(OctreeNode *node, REAL *X, REAL *M, int i)
{
    if (node->mass == 0 && node->particle_index == -1)
    {
        node->particle_index = i;
        node->mass = M[i];
        node->com_x = X[3*i];
        node->com_y = X[3*i+1];
        node->com_z = X[3*i+2];
        return;
    }

    if (node->particle_index != -1)
    {
        int existing = node->particle_index;
        node->particle_index = -1;

        for (int j = 0; j < 8; j++)
        {
            REAL offset = node->nodesize / 4;
            REAL new_cx = node->cx + ((j & 1) ? offset : -offset);
            REAL new_cy = node->cy + ((j & 2) ? offset : -offset);
            REAL new_cz = node->cz + ((j & 4) ? offset : -offset);
            node->children[j] = create_node(new_cx, new_cy, new_cz, node->nodesize / 2);
        }

        int oct = get_octant(node, X, existing);
        insert_particle(node->children[oct], X, M, existing);
    }

    int oct = get_octant(node, X, i);
    insert_particle(node->children[oct], X, M, i);

    REAL total_mass = node->mass + M[i];
    node->com_x = (node->com_x * node->mass + X[3*i] * M[i]) / total_mass;
    node->com_y = (node->com_y * node->mass + X[3*i+1] * M[i]) / total_mass;
    node->com_z = (node->com_z * node->mass + X[3*i+2] * M[i]) / total_mass;
    node->mass = total_mass;
}
void free_node(OctreeNode *node)
{
	// Free the node and its children (and their children, recursively)
	if (node == NULL) return;
	for (int i = 0; i < 8; i++)
    {
		free_node(node->children[i]);
	}
	free(node);

}

REAL get_bin_center(REAL R, int bin_index, int RADIAL_BH_FORCE_TABLE_SIZE)
{
	// This function calculates the center of a bin for the BH radial force correction table.
	// It returns the center of the bin for a given bin index.
	return (((REAL) bin_index) +0.5 ) * (R / (REAL) RADIAL_BH_FORCE_TABLE_SIZE);
}

REAL get_BH_radial_force_correction(REAL r, REAL R, REAL *RADIAL_BH_FORCE_TABLE, int RADIAL_BH_FORCE_TABLE_SIZE, int ORDER)
{
    // This function calculates the radial force correction for BH force calculation.
    // It uses the radial force table to interpolate the force correction for a given radial distance r.
	REAL correction = 0.0;
	REAL bin_center1, bin_center2, bin_center3;
	int bin_index1, bin_index2, bin_index3;
	if (ORDER>2) return 0; // If the order is higher than 2, we cannot interpolate
    if (r < 0) return 0;
	if (r>R)
	{
		// If r is greater than the maximum radius, return the linearly extrapolated value
		bin_index1 = RADIAL_BH_FORCE_TABLE_SIZE - 2;
		bin_center1 = get_bin_center(R, bin_index1, RADIAL_BH_FORCE_TABLE_SIZE);
		bin_index2 = RADIAL_BH_FORCE_TABLE_SIZE - 1;
		bin_center2 = get_bin_center(R, bin_index2, RADIAL_BH_FORCE_TABLE_SIZE);
		correction = linear_interpolation(r, bin_center1, RADIAL_BH_FORCE_TABLE[bin_index1], bin_center2, RADIAL_BH_FORCE_TABLE[bin_index2]);
		return correction;
	}
	bin_index1 = (int)floor(r / (Rsim / (double) RADIAL_BH_FORCE_TABLE_SIZE));
	bin_center1 = get_bin_center(Rsim, bin_index1, RADIAL_BH_FORCE_TABLE_SIZE);
	if(ORDER==1)
	{
		// linear interpolation. We use the two nearest bins
		if(bin_center1 <= r)
		{
			if(bin_index1 < RADIAL_BH_FORCE_TABLE_SIZE - 1)
			{
				bin_index2 = bin_index1 + 1;
				bin_center2 = (((REAL) bin_index2) +0.5 ) * (Rsim / (REAL) RADIAL_BH_FORCE_TABLE_SIZE);
			}
			else
			{
				bin_index1 -= 1; // If we are at the last bin, we use the previous bin
				bin_center1 = (((REAL) bin_index1) +0.5 ) * (Rsim / (REAL) RADIAL_BH_FORCE_TABLE_SIZE);
				bin_index2 = bin_index1 + 1;
				bin_center2 = (((REAL) bin_index2) +0.5 ) * (Rsim / (REAL) RADIAL_BH_FORCE_TABLE_SIZE);
			}
			correction = linear_interpolation(r, bin_center1, RADIAL_BH_FORCE_TABLE[bin_index1], bin_center2, RADIAL_BH_FORCE_TABLE[bin_index2]);

		}
		else
		{
			if (bin_index1 > 1)
			{
				bin_index2 = bin_index1 - 1;
				bin_center2 = (((REAL) bin_index2) +0.5 ) * (Rsim / (REAL) RADIAL_BH_FORCE_TABLE_SIZE);
				correction = linear_interpolation(r, bin_center2, RADIAL_BH_FORCE_TABLE[bin_index2], bin_center1, RADIAL_BH_FORCE_TABLE[bin_index1]);
			}
			else
			{
				correction = linear_interpolation(r, 0.0, 0.0, bin_center1, RADIAL_BH_FORCE_TABLE[bin_index1]);
			}
		}

	}
	else if(ORDER==2)
	{
		//since we are working from three poins, and the central bin is known, this is much easier
		bin_index1 -= 1; //before thecentral bin
		if(bin_index1 < 0)
		{
			// the index is less than 0, so we use the first bin
			bin_index1 = 0;
		}
		else if(bin_index1 >= RADIAL_BH_FORCE_TABLE_SIZE - 3)
		{
			bin_index1 = RADIAL_BH_FORCE_TABLE_SIZE - 3;
		}
		bin_center1 = get_bin_center(Rsim, bin_index1, RADIAL_BH_FORCE_TABLE_SIZE);
		bin_index2 = bin_index1 + 1;
		bin_center2 = get_bin_center(Rsim, bin_index2, RADIAL_BH_FORCE_TABLE_SIZE);
		bin_index3 = bin_index1 + 2;
		bin_center3 = get_bin_center(Rsim, bin_index3, RADIAL_BH_FORCE_TABLE_SIZE);
		correction = quadratic_interpolation(r, bin_center1, RADIAL_BH_FORCE_TABLE[bin_index1], bin_center2, RADIAL_BH_FORCE_TABLE[bin_index2], bin_center3, RADIAL_BH_FORCE_TABLE[bin_index3]);
	}

    return correction;
}

#endif



#if defined(PERIODIC_Z)

//This function calculates the force table for cylindrical simulations, and it is defined in the utils.cc file
void get_cylindrical_force_table(REAL* FORCE_TABLE, REAL R, REAL Lz, int TABLE_SIZE, int RADIAL_FORCE_ACCURACY);

//Function to interpolate a force for a given r, based on the values stored in the force table
REAL get_cylindrical_force_correction(REAL r, REAL R, REAL *FORCE_TABLE, int TABLE_SIZE, int ORDER)
{
    REAL step = R / (REAL) TABLE_SIZE;
    int i = (int) floor(r / R * (TABLE_SIZE - 1)); 
    REAL correction = FORCE_TABLE[TABLE_SIZE - 1];

    //Interpolate given the order
    if(ORDER == 1)
    {
        if (i < TABLE_SIZE - 1)
        {
            correction = linear_interpolation(r, step * i, FORCE_TABLE[i], step * (i + 1), FORCE_TABLE[i + 1]);
        }
    }
    else if(ORDER == 2)
    {
        if (i < TABLE_SIZE - 2)
        {
            correction = quadratic_interpolation(r, step * i, FORCE_TABLE[i], step * (i + 1), FORCE_TABLE[i + 1], step * (i + 2), FORCE_TABLE[i + 2]);
        }
        else if (i == TABLE_SIZE - 2)
        {
            correction = quadratic_interpolation(r, step * (i - 1), FORCE_TABLE[i - 1], step * i, FORCE_TABLE[i], step * (i + 1), FORCE_TABLE[i + 1]);
        }
    }  
    else if(ORDER == 3)
    {
        if (i < TABLE_SIZE - 3)
        {
            correction = cubic_interpolation(r, step * i, FORCE_TABLE[i], step * (i + 1), FORCE_TABLE[i + 1], step * (i + 2), FORCE_TABLE[i + 2], step * (i + 3), FORCE_TABLE[i + 3]);
        }
        else if (i == TABLE_SIZE - 3)
        {
            correction = cubic_interpolation(r, step * (i - 1), FORCE_TABLE[i - 1], step * i, FORCE_TABLE[i], step * (i + 1), FORCE_TABLE[i + 1], step * (i + 2), FORCE_TABLE[i + 2]);
        }
    }  
    return correction;
}
#endif

#if !defined(PERIODIC) && !defined(PERIODIC_Z)
// Free StePS boundary conditions
#if defined(USE_BH)
// Barnes-Hut oct-tree force calculation

#ifdef RANDOMIZE_BH
void rotate_vectors(REAL* CoordArray, const REAL* Y, REAL ROT_RAD, int idmin, int idmax)
{
	// 3D rotation of all coordinates around the Y axis with ROT_RAD radians by using Rodrigues' rotation formula
    REAL cos_theta = cos(ROT_RAD);
    REAL sin_theta = sin(ROT_RAD);

	for(int i = idmin; i < idmax+1; ++i)
	{
		REAL X[3] = {CoordArray[3*i], CoordArray[3*i+1], CoordArray[3*i+2]};

		// Cross product Y x X
		REAL cross[3] = {
			Y[1] * X[2] - Y[2] * X[1],
			Y[2] * X[0] - Y[0] * X[2],
			Y[0] * X[1] - Y[1] * X[0]
		};

		// Dot product Y * X
		REAL dot = Y[0] * X[0] + Y[1] * X[1] + Y[2] * X[2];

		// Rodrigues' rotation formula
		REAL rotated[3];
		for (int i = 0; i < 3; ++i)
		{
			rotated[i] = X[i] * cos_theta + cross[i] * sin_theta + Y[i] * dot * (1.0 - cos_theta);
		}

		// Store result back to the original array
		CoordArray[3*i] = rotated[0];
		CoordArray[3*i+1] = rotated[1];
		CoordArray[3*i+2] = rotated[2];
	}
}

void random_unit_vector(double* vec)
{
	// Generate a random unit vector in 3D space
    double theta = ((double)rand() / RAND_MAX) * 2.0 * M_PI;
    double z = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
    double r = sqrt(1.0 - z * z);

    vec[0] = r * cos(theta);
    vec[1] = r * sin(theta);
    vec[2] = z;
}

#endif

void compute_BH_force(OctreeNode *node, REAL *X, int i, REAL *SOFT_LENGTH, REAL *fx, REAL *fy, REAL *fz)
{
    if (node == NULL || (node->mass == 0) || (node->particle_index == i)) return;

	REAL wij, beta;

    REAL dx = node->com_x - X[3*i];
    REAL dy = node->com_y - X[3*i+1];
    REAL dz = node->com_z - X[3*i+2];
    REAL dist = sqrt(dx*dx + dy*dy + dz*dz);

    if (node->particle_index != -1 || node->nodesize / dist < THETA)
    {
		beta = cbrt(node->mass / M_min)*ParticleRadi + SOFT_LENGTH[i];
		wij = node->mass * force_softening(dist, beta);
        *fx += wij * dx;
        *fy += wij * dy;
        *fz += wij * dz;
    }
    else
    {
        for (int j = 0; j < 8; j++)
        {
            compute_BH_force(node->children[j], X, i, SOFT_LENGTH, fx, fy, fz);
        }
    }
}

void forces(REAL* x, REAL* F, int ID_min, int ID_max) //Force calculation
{
    //timing
    double omp_start_time = omp_get_wtime();
    //timing
    REAL Fx_tmp, Fy_tmp, Fz_tmp, r_xyz;
    REAL DE = (REAL) H0*H0*Omega_lambda;
    int i, k, chunk;
	REAL domain_center[3];
	REAL RootNodeSize;
    //Building the octree
	// Identifying the most outer particle radius
	REAL radius_tmp, Max_radius = 0.0;
	for (int i = 0; i < N; i++)
    {
		radius_tmp = sqrt(pow(x[3*i], 2) + pow(x[3*i+1], 2) + pow(x[3*i+2], 2));
		if (radius_tmp > Max_radius)
        {
			Max_radius = radius_tmp;
		}
	}
	#ifdef RANDOMIZE_BH
	REAL rotation_axis[3];
	REAL rotation_angle;
    //generating the random shift vector (100% of the maximum radius)
	for(i=0; i<3; i++)
	{
		domain_center[i] = ((REAL)rand()/(REAL)RAND_MAX-0.5)*2.0*Max_radius;
	}
	RootNodeSize = 4.00004 * Max_radius; //size of the root node (2Dsim+epsilon)
	rotation_angle = (REAL)rand()/(REAL)RAND_MAX * pi;
	random_unit_vector(rotation_axis);
	printf("MPI task %i: Octree force calculation started with random %.3f RAD rotation along the\n\t    (%.3f, %.3f, %.3f) axis vector, and with random domain center (%.3f, %.3f, %.3f).\n", rank, rotation_angle, rotation_axis[0], rotation_axis[1], rotation_axis[2], domain_center[0], domain_center[1], domain_center[2]);
	// Rotate the coordinates
	rotate_vectors(x, rotation_axis, rotation_angle, 0, N-1);
	#else
	domain_center[0] = domain_center[1] = domain_center[2] = 0.0;
	RootNodeSize = 2.00002 * Max_radius; //size of the root node (Dsim+epsilon)
	printf("MPI task %i: Octree force calculation started.\n", rank);
	#endif
	if (numtasks > 1)
		printf("MPI task %i: ID_min = %i, ID_max = %i.\n", rank, ID_min, ID_max);
    OctreeNode *rootnode = create_node(domain_center[0], domain_center[1], domain_center[2], RootNodeSize); //centered at domain_center, size RootNodeSize
    for (int i = 0; i < N; i++)
    {
        // Insert particles into the octree
        insert_particle(rootnode, x, M, i);
    }
    for(i=0; i<N_mpi_thread; i++)
    {
            for(k=0; k<3; k++)
            {
                    F[3*i+k] = 0;
            }
    }
	chunk = (ID_max-ID_min)/omp_get_max_threads()/4;
	if(chunk < 1)
	{
		chunk = 1;
	}
	#pragma omp parallel default(shared)  private(i, Fx_tmp, Fy_tmp, Fz_tmp)
	{
	#pragma omp for schedule(dynamic,chunk)
	for(i=ID_min; i<ID_max+1; i++)
	{
		Fx_tmp = Fy_tmp = Fz_tmp = 0.0;
        compute_BH_force(rootnode, x, i, SOFT_LENGTH, &Fx_tmp, &Fy_tmp, &Fz_tmp);
        #pragma omp atomic
            F[3*(i-ID_min)] += Fx_tmp;
		#pragma omp atomic
            F[3*(i-ID_min)+1] += Fy_tmp;
		#pragma omp atomic
            F[3*(i-ID_min)+2] += Fz_tmp;
		if(COSMOLOGY == 1 && COMOVING_INTEGRATION == 1)//Adding the external force from the outside of the simulation volume, if we run non-periodic comoving cosmological simulation
		{
			F[3*(i-ID_min)] += mass_in_unit_sphere * x[3*i];
			F[3*(i-ID_min)+1] += mass_in_unit_sphere * x[3*i+1];
			F[3*(i-ID_min)+2] += mass_in_unit_sphere * x[3*i+2];
			if(USE_RADIAL_BH_CORRECTION == true)
			{
				r_xyz = sqrt(pow(x[3*i], 2) + pow(x[3*i+1], 2) + pow(x[3*i+2], 2));
				F[3*(i-ID_min)] -= x[3*i]*get_BH_radial_force_correction(r_xyz, Rsim, RADIAL_BH_FORCE_TABLE, RADIAL_BH_FORCE_TABLE_SIZE, RADIAL_BH_FORCE_ORDER)/r_xyz;
				F[3*(i-ID_min)+1] -= x[3*i+1]*get_BH_radial_force_correction(r_xyz, Rsim, RADIAL_BH_FORCE_TABLE, RADIAL_BH_FORCE_TABLE_SIZE, RADIAL_BH_FORCE_ORDER)/r_xyz;
				F[3*(i-ID_min)+2] -= x[3*i+2]*get_BH_radial_force_correction(r_xyz, Rsim, RADIAL_BH_FORCE_TABLE, RADIAL_BH_FORCE_TABLE_SIZE, RADIAL_BH_FORCE_ORDER)/r_xyz;
			}
		}
		else if(COSMOLOGY == 1 && COMOVING_INTEGRATION == 0)
		{
			F[3*(i-ID_min)] +=  DE * x[3*i];
			F[3*(i-ID_min)+1] += DE * x[3*i+1];
			F[3*(i-ID_min)+2] += DE * x[3*i+2];
		}
	}
    }
	free_node(rootnode);
	// Rotating the coordinates and forces back to the original orientation
	#ifdef RANDOMIZE_BH
	rotate_vectors(x, rotation_axis, -rotation_angle, 0, N-1);
	rotate_vectors(F, rotation_axis, -rotation_angle, 0, N_mpi_thread-1);
	#endif
	//timing
	double omp_end_time = omp_get_wtime();
	//timing
	printf("Octree force calculation finished on MPI task %i. Force calculation wall-clock time = %fs.\n", rank, omp_end_time-omp_start_time);
	return;
}

#else
// Direct summation force calculation
void forces(REAL* x, REAL* F, int ID_min, int ID_max) //Force calculation
{
	REAL Fx_tmp, Fy_tmp, Fz_tmp, beta_priv;
	REAL DE = (REAL) H0*H0*Omega_lambda;

	//timing
    double omp_start_time = omp_get_wtime();
    //timing

	int i, j, k, chunk;
	for(i=0; i<N_mpi_thread; i++)
	{
			for(k=0; k<3; k++)
			{
					F[3*i+k] = 0;
			}
	}
    REAL r, dx, dy, dz, wij;
	chunk = (ID_max-ID_min)/omp_get_max_threads();
	if(chunk < 1)
	{
		chunk = 1;
	}
	#pragma omp parallel default(shared)  private(dx, dy, dz, r, wij, j, i, Fx_tmp, Fy_tmp, Fz_tmp, beta_priv)
	{
	#pragma omp for schedule(dynamic,chunk)
	for(i=ID_min; i<ID_max+1; i++)
	{
		for(j=0; j<N; j++)
		{
			beta_priv = (SOFT_LENGTH[i]+SOFT_LENGTH[j]);
			//calculating particle distances
            dx=x[3*j]-x[3*i];
			dy=x[3*j+1]-x[3*i+1];
			dz=x[3*j+2]-x[3*i+2];
			r = sqrt(pow(dx, 2)+pow(dy, 2)+pow(dz, 2));
			wij = M[j]*force_softening(r, beta_priv);
			Fx_tmp = wij*(dx);
			Fy_tmp = wij*(dy);
			Fz_tmp = wij*(dz);
			#pragma omp atomic
                        F[3*(i-ID_min)] += Fx_tmp;
			#pragma omp atomic
                        F[3*(i-ID_min)+1] += Fy_tmp;
			#pragma omp atomic
                        F[3*(i-ID_min)+2] += Fz_tmp;
        }
		if(COSMOLOGY == 1 && COMOVING_INTEGRATION == 1)//Adding the external force from the outside of the simulation volume, if we run non-periodic comoving cosmological simulation
		{
			F[3*(i-ID_min)] += mass_in_unit_sphere * x[3*i];
			F[3*(i-ID_min)+1] += mass_in_unit_sphere * x[3*i+1];
			F[3*(i-ID_min)+2] += mass_in_unit_sphere * x[3*i+2];
		}
		else if(COSMOLOGY == 1 && COMOVING_INTEGRATION == 0)
		{
			F[3*(i-ID_min)] +=  DE * x[3*i];
			F[3*(i-ID_min)+1] += DE * x[3*i+1];
			F[3*(i-ID_min)+2] += DE * x[3*i+2];
		}
	}

        }
//timing
double omp_end_time = omp_get_wtime();
//timing
printf("Direct force calculation finished on MPI task %i. Force calculation wall-clock time = %fs.\n", rank, omp_end_time-omp_start_time);
return;
}
#endif
#endif

#ifdef PERIODIC
#if defined(USE_BH)
//Barnes-Hut oct-tree force calculation with multiple images
void compute_BH_force(OctreeNode *node, REAL *X, int i, REAL *SOFT_LENGTH, REAL boxsize, REAL *fx, REAL *fy, REAL *fz)
{
    if (node == NULL || (node->mass == 0) || (node->particle_index == i)) return;

	REAL wij, beta;
	REAL D[3];
    REAL dx = node->com_x - X[3*i];
    REAL dy = node->com_y - X[3*i+1];
    REAL dz = node->com_z - X[3*i+2];
	//in this case we use only the nearest image of the node, but correcting for periodicity with the interpolated Ewald force table
	if(fabs(dx)>0.5*boxsize)
		dx = dx-boxsize*dx/fabs(dx);
	if(fabs(dy)>0.5*boxsize)
		dy = dy-boxsize*dy/fabs(dy);
	if(fabs(dz)>0.5*boxsize)
		dz = dz-boxsize*dz/fabs(dz);
    REAL dist = sqrt(dx*dx + dy*dy + dz*dz);
    if (node->particle_index != -1 || node->nodesize / dist < THETA)
    {
		beta = cbrt(node->mass / M_min)*ParticleRadi + SOFT_LENGTH[i];
		wij = force_softening(dist, beta);
		//Adding the Ewald correction
		ewald_interpolate_D(N_EWALD_FORCE_GRID, L, T3_EWALD_FORCE_TABLE, dx, dy, dz, D, EWALD_INTERPOLATION_ORDER);
        *fx += node->mass *( wij * dx - D[0]);
        *fy += node->mass *( wij * dy - D[1]);
        *fz += node->mass *( wij * dz - D[2]);
    }
    else
    {
        for (int j = 0; j < 8; j++)
        {
            compute_BH_force(node->children[j], X, i, SOFT_LENGTH, boxsize, fx, fy, fz);
        }
    }
}

void compute_BH_QP_force(OctreeNode *node, REAL *X, int i, REAL *SOFT_LENGTH, REAL boxsize, REAL *fx, REAL *fy, REAL *fz)
{
	//quasi-periodic force calculation
    if (node == NULL || (node->mass == 0) || (node->particle_index == i)) return;

	REAL wij, beta;

    REAL dx = node->com_x - X[3*i];
    REAL dy = node->com_y - X[3*i+1];
    REAL dz = node->com_z - X[3*i+2];
    //in this case we use only the nearest image of the node
	if(fabs(dx)>0.5*boxsize)
		dx = dx-boxsize*dx/fabs(dx);
	if(fabs(dy)>0.5*boxsize)
		dy = dy-boxsize*dy/fabs(dy);
	if(fabs(dz)>0.5*boxsize)
		dz = dz-boxsize*dz/fabs(dz);
	REAL dist = sqrt(pow(dx, 2)+pow(dy, 2)+pow(dz, 2));
    if (node->particle_index != -1 || node->nodesize / dist < THETA)
    {
		beta = cbrt(node->mass / M_min)*ParticleRadi + SOFT_LENGTH[i];
		wij = node->mass * force_softening(dist, beta);
        *fx += wij * dx;
        *fy += wij * dy;
        *fz += wij * dz;
    }
    else
    {
        for (int j = 0; j < 8; j++)
        {
            compute_BH_QP_force(node->children[j], X, i, SOFT_LENGTH, boxsize, fx, fy, fz);
        }
    }
}

void forces_periodic(REAL*x, REAL*F, int ID_min, int ID_max) //force calculation with multiple images
{
	//timing
	double omp_start_time = omp_get_wtime();
	//timing
	REAL Fx_tmp, Fy_tmp, Fz_tmp;
	int i, k, chunk;
	#ifdef RANDOMIZE_BH
	//generating the random shift vector
	REAL random_shift[3];
	for(i=0; i<3; i++)
	{
		random_shift[i] = ((REAL)rand()/(REAL)RAND_MAX-0.5)*L;
	}
	printf("MPI task %i: Octree force calculation started with random shift vector (%.3f %.3f %.3f).\n", rank, random_shift[0], random_shift[1], random_shift[2]);
	// Shifting the particles by the random vector with periodic boundary conditions
	for(i=0; i<N; i++)
	{
		for(k=0; k<3; k++)
		{
			x[3*i+k] += random_shift[k];
			if(x[3*i+k]<0)
			{
				x[3*i+k] = x[3*i+k] + L;
			}
			else if(x[3*i+k]>=L)
			{
				x[3*i+k] = x[3*i+k] - L;
			}
		}
	}
	#else
	printf("MPI task %i: Octree force calculation started.\n", rank);
	#endif
	if (numtasks > 1)
		printf("MPI task %i: ID_min = %i, ID_max = %i.\n", rank, ID_min, ID_max);
	//Building the octree;
    OctreeNode *rootnode = create_node(0.50*L, 0.50*L, 0.50*L, L); //center of the simulation box, size L
    for (int i = 0; i < N; i++)
    {
        // Insert particles into the octree
        insert_particle(rootnode, x, M, i);
    }
    for(i=0; i<N_mpi_thread; i++)
    {
		for(k=0; k<3; k++)
		{
			F[3*i+k] = 0;
		}
    }
	chunk = (ID_max-ID_min)/(omp_get_max_threads())/8;
	if(chunk < 1)
	{
		chunk = 1;
	}
	if(IS_PERIODIC>=2)
	{
		// Ewald summation with multiple images
		#pragma omp parallel default(shared)  private(i, Fx_tmp, Fy_tmp, Fz_tmp)
        {
        	#pragma omp for schedule(dynamic,chunk)
	        for(i=ID_min; i<ID_max+1; i++)
			{
				Fx_tmp = Fy_tmp = Fz_tmp = 0.0;
				compute_BH_force(rootnode, x, i, SOFT_LENGTH, L, &Fx_tmp, &Fy_tmp, &Fz_tmp);
				#pragma omp atomic
					F[3*(i-ID_min)] += Fx_tmp;
				#pragma omp atomic
					F[3*(i-ID_min)+1] += Fy_tmp;
				#pragma omp atomic
					F[3*(i-ID_min)+2] += Fz_tmp;
			}
		}	
	}
	else
	{
		//quasi-periodic force calculation with multiple images
		#pragma omp parallel default(shared)  private(i, Fx_tmp, Fy_tmp, Fz_tmp)
        {
        	#pragma omp for schedule(dynamic,chunk)
	        for(i=ID_min; i<ID_max+1; i++)
			{
				Fx_tmp = Fy_tmp = Fz_tmp = 0.0;
				compute_BH_QP_force(rootnode, x, i, SOFT_LENGTH, L, &Fx_tmp, &Fy_tmp, &Fz_tmp);
				#pragma omp atomic
					F[3*(i-ID_min)] += Fx_tmp;
				#pragma omp atomic
					F[3*(i-ID_min)+1] += Fy_tmp;
				#pragma omp atomic
					F[3*(i-ID_min)+2] += Fz_tmp;
			}
		}	
	}
	free_node(rootnode);
	#ifdef RANDOMIZE_BH
	// Shifting back the particles to their original position with periodic boundary conditions
	for(i=0; i<N; i++)
	{
		for(k=0; k<3; k++)
		{
			x[3*i+k] -= random_shift[k];
			if(x[3*i+k]<0)
			{
				x[3*i+k] = x[3*i+k] + L;
			}
			else if(x[3*i+k]>=L)
			{
				x[3*i+k] = x[3*i+k] - L;
			}
		}
	}
	#endif
	//timing
	double omp_end_time = omp_get_wtime();
	//timing
	printf("Octree force calculation finished on MPI task %i. Force calculation wall-clock time = %fs.\n", rank, omp_end_time-omp_start_time);
	return;
}

#else
//Direct summation force calculation with multiple images
void forces_periodic(REAL*x, REAL*F, int ID_min, int ID_max) //force calculation with multiple images
{
	//timing
	double omp_start_time = omp_get_wtime();
	//timing
	REAL D[3];
	REAL Fx_tmp, Fy_tmp, Fz_tmp, beta_priv;
	int i, j, k, chunk;
	for(i=0; i<N_mpi_thread; i++)
	{
		for(k=0; k<3; k++)
		{
			F[3*i+k] = 0;
		}
	}
	REAL r, dx, dy, dz, wij;
	chunk = (ID_max-ID_min)/(omp_get_max_threads());
	if(chunk < 1)
	{
		chunk = 1;
	}
	if(IS_PERIODIC>=2)
	{
		#pragma omp parallel default(shared)  private(dx, dy, dz, r, wij, j, i, Fx_tmp, Fy_tmp, Fz_tmp, beta_priv, D)
        	{
        	#pragma omp for schedule(dynamic,chunk)
	        for(i=ID_min; i<ID_max+1; i++)
			{
				for(j=0; j<N; j++)
				{
					beta_priv = (SOFT_LENGTH[i]+SOFT_LENGTH[j]);
					//calculating particle distances
					dx=x[3*j]-x[3*i];
					dy=x[3*j+1]-x[3*i+1];
					dz=x[3*j+2]-x[3*i+2];
					//in this case we use only the nearest image
                    if(fabs(dx)>0.5*L)
                        dx = dx-L*dx/fabs(dx);
					if(fabs(dy)>0.5*L)
						dy = dy-L*dy/fabs(dy);
					if(fabs(dz)>0.5*L)
						dz = dz-L*dz/fabs(dz);
					r = sqrt(pow(dx, 2)+pow(dy, 2)+pow(dz, 2));
					wij = force_softening(r, beta_priv);
					//adding the contributions from all images
					ewald_interpolate_D(N_EWALD_FORCE_GRID, L, T3_EWALD_FORCE_TABLE, dx, dy, dz, D, EWALD_INTERPOLATION_ORDER);
					Fx_tmp = M[j] * (wij*(dx) - D[0]);
					Fy_tmp = M[j] * (wij*(dy) - D[1]);
					Fz_tmp = M[j] * (wij*(dz) - D[2]);
					#pragma omp atomic
					F[3*(i-ID_min)] += Fx_tmp;
					#pragma omp atomic
					F[3*(i-ID_min)+1] += Fy_tmp;
					#pragma omp atomic
					F[3*(i-ID_min)+2] += Fz_tmp;
				}
			}
		}
	}
	else
	{
		#pragma omp parallel default(shared)  private(dx, dy, dz, r, wij, j, i, Fx_tmp, Fy_tmp, Fz_tmp, beta_priv)
        	{
        	#pragma omp for schedule(dynamic,chunk)
	        	for(i=ID_min; i<ID_max+1; i++)
			{
				for(j=0; j<N; j++)
				{
					beta_priv = (SOFT_LENGTH[i]+SOFT_LENGTH[j]);
					//calculating particle distances
					dx=x[3*j]-x[3*i];
					dy=x[3*j+1]-x[3*i+1];
					dz=x[3*j+2]-x[3*i+2];
					//in this case we use only the nearest image
                    if(fabs(dx)>0.5*L)
                        dx = dx-L*dx/fabs(dx);
					if(fabs(dy)>0.5*L)
						dy = dy-L*dy/fabs(dy);
					if(fabs(dz)>0.5*L)
						dz = dz-L*dz/fabs(dz);
					r = sqrt(pow(dx, 2)+pow(dy, 2)+pow(dz, 2));
					wij = M[j] * force_softening(r, beta_priv);
					Fx_tmp = wij*(dx);
					Fy_tmp = wij*(dy);
					Fz_tmp = wij*(dz);
					#pragma omp atomic
					F[3*(i-ID_min)] += Fx_tmp;
					#pragma omp atomic
					F[3*(i-ID_min)+1] += Fy_tmp;
					#pragma omp atomic
					F[3*(i-ID_min)+2] += Fz_tmp;
				}
			}
		}	
	}
//timing
double omp_end_time = omp_get_wtime();
//timing
printf("Direct force calculation finished on MPI task %i. Force calculation wall-clock time = %fs.\n", rank, omp_end_time-omp_start_time);
return;
}
#endif
#endif

#ifdef PERIODIC_Z
#if defined(USE_BH)
//Barnes-Hut oct-tree force calculation with multiple images in the z direction only

#ifdef RANDOMIZE_BH
void rotate_vectors_2d(REAL* CoordArray, REAL ROT_RAD, int idmin, int idmax)
{
	//2D Rotation in the x-y plane with ROT_RAD radians.
	REAL cos_theta = cos(ROT_RAD);
	REAL sin_theta = sin(ROT_RAD);
	REAL rotated[2];
	for(int i = idmin; i < idmax+1; ++i)
	{
		rotated[0] = CoordArray[3*i]*cos_theta-CoordArray[3*i+1]*sin_theta;
		rotated[1] = CoordArray[3*i]*sin_theta+CoordArray[3*i+1]*cos_theta;
		// Store result back to the original array
		CoordArray[3*i] = rotated[0];
		CoordArray[3*i+1] = rotated[1];
	}
}
#endif

#ifdef PERIODIC_Z_NOLOOKUP
void compute_BH_rspace_force_z(OctreeNode *node, REAL *X, int i, REAL *SOFT_LENGTH, REAL COORD_X, REAL COORD_Y, REAL COORD_Z, REAL ewald_cut, REAL *fx, REAL *fy, REAL *fz)
{
	if (node == NULL || (node->mass == 0) || (node->particle_index == i)) return;

	REAL wij, beta;

	REAL dx = node->com_x - COORD_X;
	REAL dy = node->com_y - COORD_Y;
	REAL dz = (node->com_z - COORD_Z);
	REAL dist = sqrt(dx*dx + dy*dy + dz*dz);
	if (node->particle_index != -1 || (node->nodesize) / dist < THETA)
	{
		if (fabs(dz) > ewald_cut) return;
		beta = cbrt(node->mass / M_min)*ParticleRadi + SOFT_LENGTH[i];
		wij = node->mass * force_softening(dist, beta);
		*fx += wij * dx;
		*fy += wij * dy;
		*fz += wij * dz;
	}
	else
	{
		for (int j = 0; j < 8; j++)
		{
			compute_BH_rspace_force_z(node->children[j], X, i, SOFT_LENGTH, COORD_X, COORD_Y, COORD_Z, ewald_cut, fx, fy, fz);
		}
	}
}
#else
void compute_BH_force_z(OctreeNode *node, REAL *X, int i, REAL *SOFT_LENGTH, REAL boxsize, REAL *fx, REAL *fy, REAL *fz)
{
	if (node == NULL || (node->mass == 0) || (node->particle_index == i)) return;

	REAL wij, beta;
	REAL D[3];
	REAL dx = node->com_x - X[3*i];
	REAL dy = node->com_y - X[3*i+1];
	REAL dz = node->com_z - X[3*i+2];
	//in this case we use only the nearest image of the node
	if(fabs(dz)>0.5*boxsize)
		dz = dz-boxsize*dz/fabs(dz); //wrapping in the z direction only (no need to wrap after this)
	REAL dist = sqrt(pow(dx, 2)+pow(dy, 2)+pow(dz, 2));
	if (node->particle_index != -1 || (node->nodesize) / dist < THETA)
	{
		beta = cbrt(node->mass / M_min)*ParticleRadi + SOFT_LENGTH[i];
		wij = force_softening(dist, beta);
		//Adding the Ewald correction
		ewald_interpolate_D( S1R2_EWALD_FORCE_TABLE, Nrho_EWALD_FORCE_GRID, Nz_EWALD_FORCE_GRID, EWALD_LOOKUP_TABLE_RADIAL_EXTENT_FACTOR*Rsim, boxsize, dx, dy, dz, D[0], D[1], D[2]);

		*fx += node->mass *  (wij * dx - D[0]);
		*fy += node->mass *  (wij * dy - D[1]);
		*fz += node->mass *  (wij * dz - D[2]);
	}
	else
	{
		for (int j = 0; j < 8; j++)
		{
			compute_BH_force_z(node->children[j], X, i, SOFT_LENGTH, boxsize, fx, fy, fz);
		}
	}
}
#endif

void compute_BH_QP_force_z(OctreeNode *node, REAL *X, int i, REAL *SOFT_LENGTH, REAL boxsize, REAL *fx, REAL *fy, REAL *fz)
{
	//quasi-periodic force calculation in the z direction only
	if (node == NULL || (node->mass == 0) || (node->particle_index == i)) return;

	REAL wij, beta;

	REAL dx = node->com_x - X[3*i];
	REAL dy = node->com_y - X[3*i+1];
	REAL dz = (node->com_z - X[3*i+2]);
	//in this case we use only the nearest image of the node
	if(fabs(dz)>0.5*boxsize)
		dz = dz-boxsize*dz/fabs(dz);
	REAL dist = sqrt(pow(dx, 2)+pow(dy, 2)+pow(dz, 2));
	if (node->particle_index != -1 || (node->nodesize) / dist < THETA)
	{
		beta = cbrt(node->mass / M_min)*ParticleRadi + SOFT_LENGTH[i];
		wij = node->mass * force_softening(dist, beta);
		*fx += wij * dx;
		*fy += wij * dy;
		*fz += wij * dz;
	}
	else
	{
		for (int j = 0; j < 8; j++)
		{
			compute_BH_QP_force_z(node->children[j], X, i, SOFT_LENGTH, boxsize, fx, fy, fz);
		}
	}
}

void forces_periodic_z(REAL* x, REAL* F, int ID_min, int ID_max)
{
    //timing
    double omp_start_time = omp_get_wtime();
    //timing
    REAL Fx_tmp, Fy_tmp, Fz_tmp, r_xy, RootNodeSize, cylindrical_force_correction;
	#ifdef PERIODIC_Z_NOLOOKUP
		REAL RealSpaceCut;
	#endif
	REAL random_shift[3];
	REAL DE = (REAL) H0*H0*Omega_lambda;
    int i, k, chunk;
    for(i=0; i<N_mpi_thread; i++)
    {
        for(k=0; k<3; k++)
        {
            F[3*i+k] = 0;
        }
    }
    //Building the octree
	// Identifying the most outer particle radius
	REAL radius_tmp, Max_radius = 0.0;
	for (int i = 0; i < N; i++)
    {
		radius_tmp = sqrt(pow(x[3*i], 2) + pow(x[3*i+1], 2));
		if (radius_tmp > Max_radius)
        {
			Max_radius = radius_tmp;
		}
	}
	#ifdef RANDOMIZE_BH
	//randomly shifting the domain center and rotating the simulation volume
	REAL rotation_angle;
	rotation_angle = ((REAL)rand()/(REAL)RAND_MAX)*2.0*pi; //random rotation angle between 0 and 2*pi
	random_shift[0] = ((REAL)rand()/(REAL)RAND_MAX-0.5)*2*Max_radius; //shift in the x direction between -Rsim and Rsim
	random_shift[1] = ((REAL)rand()/(REAL)RAND_MAX-0.5)*2*Max_radius; //shift in the y direction between -Rsim and Rsim
	random_shift[2] = ((REAL)rand()/(REAL)RAND_MAX-0.5)*L; //shift in the z direction between -0.5*Lz and 0.5*Lz
	printf("MPI task %i: Octree force calculation started with random shift vector (%.3f %.3f %.3f)\n\t    and rotation angle %.3f RAD around the z axis.\n", rank, random_shift[0], random_shift[1], random_shift[2], rotation_angle);
	//First, we rotate the particles around the z axis by the random angle
	rotate_vectors_2d(x, rotation_angle, 0, N-1);
	if (4*Max_radius < L)
	{
		//if the maximum diameter is smaller than half of the box size, we use the periodicity length as the root node size
		RootNodeSize = L;
	}
	else
	{
		//if the maximum diameter is larger than half of the box size, we use the double of maximum diameter as the root node size
		RootNodeSize = 4.0*Max_radius; // 2+epsilon times of the maximal diameter
	}
	#else
	random_shift[0] = 0.0; //no shift in the x direction
	random_shift[1] = 0.0; //no shift in the y direction
	random_shift[2] = 0.0; //no shift in the z direction
	if(2*Max_radius < L)
	{
		//if the maximum diameter is smaller than half of the box size, we use the periodicity length as the root node size
		RootNodeSize = L;
	}
	else
	{
		//if the maximum diameter is larger than half of the box size, we use the double of maximum diameter as the root node size
		RootNodeSize = 2.0*Max_radius; // 2+epsilon times of the maximal diameter
	}
	printf("MPI task %i: Octree force calculation started.\n", rank);
	#endif
	if (numtasks > 1)
		printf("MPI task %i: ID_min = %i, ID_max = %i.\n", rank, ID_min, ID_max);
	//building the octree
	OctreeNode *rootnode = create_node(random_shift[0], random_shift[1], 0.50*RootNodeSize, RootNodeSize); //center of the simulation box, size 2*(Rsim+epsilon)
	for (int i = 0; i < N; i++)
    {
		#ifdef RANDOMIZE_BH
		//Shifting the particles by the random magnitude with periodic boundary conditions only in the z direction
		x[3*i+2] += random_shift[2];
		//Checking the periodic boundaries along the z axis
		if(x[3*i+2]<0)
		{
			x[3*i+2] = x[3*i+2] + L;
		}
		else if(x[3*i+2]>=L)
		{
			x[3*i+2] = x[3*i+2] - L;
		}
		#endif
        // Insert particles into the octree
        insert_particle(rootnode, x, M, i);
	}
    for(i=0; i<N_mpi_thread; i++)
    {
		for(k=0; k<3; k++)
		{
			F[3*i+k] = 0;
		}
    }
    chunk = (ID_max-ID_min)/(omp_get_max_threads())/8;
    if(chunk < 1)
    {
        chunk = 1;
    }
    if(IS_PERIODIC>=2) 
	{
		// Fully periodic in the z direction using Ewald summation or direct real-space summation with multiple images
		#ifdef PERIODIC_Z_NOLOOKUP
			RealSpaceCut = ewald_cut*L; // Real space cutoff radius
		#endif
        #pragma omp parallel default(shared)  private(i, Fx_tmp, Fy_tmp, Fz_tmp)
		#pragma omp for schedule(dynamic,chunk)
			for(i=ID_min; i<ID_max+1; i++)
			{
				#ifdef PERIODIC_Z_NOLOOKUP
					//using direct real-space summation with multiple images
					for(int m=-ewald_max; m<ewald_max+1; m++)
					{
						Fx_tmp = Fy_tmp = Fz_tmp = 0.0;
						compute_BH_rspace_force_z(rootnode, x, i, SOFT_LENGTH, x[3*i], x[3*i+1], x[3*i+2]+((REAL) m)*L, RealSpaceCut, &Fx_tmp, &Fy_tmp, &Fz_tmp);
						#pragma omp atomic
							F[3*(i-ID_min)] += Fx_tmp;
						#pragma omp atomic
							F[3*(i-ID_min)+1] += Fy_tmp;
						#pragma omp atomic
							F[3*(i-ID_min)+2] += Fz_tmp;
					}
				#else
					//using Ewald look-up table for the periodicity in the z direction
					Fx_tmp = Fy_tmp = Fz_tmp = 0.0;
					compute_BH_force_z(rootnode, x, i, SOFT_LENGTH, L, &Fx_tmp, &Fy_tmp, &Fz_tmp);
					#pragma omp atomic
						F[3*(i-ID_min)] += Fx_tmp;
					#pragma omp atomic
						F[3*(i-ID_min)+1] += Fy_tmp;
					#pragma omp atomic
						F[3*(i-ID_min)+2] += Fz_tmp;
				#endif
				//adding the external force from the outside of the simulation volume,
				//if we run a not fully periodic comoving cosmological simulation
				if(COSMOLOGY == 1 && COMOVING_INTEGRATION == 1)
				{
					r_xy = sqrt(pow(x[3*i], 2) + pow(x[3*i+1], 2));
					#ifdef PERIODIC_Z_NOLOOKUP
						cylindrical_force_correction = get_cylindrical_force_correction(r_xy, Rsim, RADIAL_FORCE_TABLE, RADIAL_FORCE_TABLE_SIZE, 1);
						F[3*(i-ID_min)] += mass_in_unit_sphere * x[3*i] * cylindrical_force_correction;
						F[3*(i-ID_min)+1] += mass_in_unit_sphere * x[3*i+1] * cylindrical_force_correction;
					#else
						F[3*(i-ID_min)] += mass_in_unit_sphere * x[3*i]; //-2\pi*G*rho_b*x
                        F[3*(i-ID_min)+1] += mass_in_unit_sphere * x[3*i+1]; //-2\pi*G*rho_b*y
					#endif
					if(USE_RADIAL_BH_CORRECTION == true)
					{
						F[3*(i-ID_min)] -= x[3*i]*get_BH_radial_force_correction(r_xy, Rsim, RADIAL_BH_FORCE_TABLE, RADIAL_BH_FORCE_TABLE_SIZE, RADIAL_BH_FORCE_ORDER)/r_xy;
						F[3*(i-ID_min)+1] -= x[3*i+1]*get_BH_radial_force_correction(r_xy, Rsim, RADIAL_BH_FORCE_TABLE, RADIAL_BH_FORCE_TABLE_SIZE, RADIAL_BH_FORCE_ORDER)/r_xy;
					}
				}
				else if(COSMOLOGY == 1 && COMOVING_INTEGRATION == 0)
				{
					F[3*(i-ID_min)] +=  DE * x[3*i];
					F[3*(i-ID_min)+1] += DE * x[3*i+1];
				} //non-comoving integration is not implemented for periodic_z (yet?)
			}
    }
    else
	{
		//quasi-periodic in the z direction only
        #pragma omp parallel default(shared)  private(i, Fx_tmp, Fy_tmp, Fz_tmp)
		#pragma omp for schedule(dynamic,chunk)
			for(i=ID_min; i<ID_max+1; i++)
			{
				Fx_tmp = Fy_tmp = Fz_tmp = 0.0;
				//using the nearest image in the z direction
				compute_BH_QP_force_z(rootnode, x, i, SOFT_LENGTH, L, &Fx_tmp, &Fy_tmp, &Fz_tmp);
				#pragma omp atomic
					F[3*(i-ID_min)] += Fx_tmp;
				#pragma omp atomic
					F[3*(i-ID_min)+1] += Fy_tmp;
				#pragma omp atomic
					F[3*(i-ID_min)+2] += Fz_tmp;
				//adding the external force from the outside of the simulation volume,
				//if we run a not fully periodic comoving cosmological simulation
				if(COSMOLOGY == 1 && COMOVING_INTEGRATION == 1)
				{
					r_xy = sqrt(pow(x[3*i], 2) + pow(x[3*i+1], 2));
					cylindrical_force_correction = get_cylindrical_force_correction(r_xy, Rsim, RADIAL_FORCE_TABLE, RADIAL_FORCE_TABLE_SIZE, 1);
					F[3*(i-ID_min)] += mass_in_unit_sphere * x[3*i] * cylindrical_force_correction;
					F[3*(i-ID_min)+1] += mass_in_unit_sphere * x[3*i+1] * cylindrical_force_correction;
					if(USE_RADIAL_BH_CORRECTION == true)
					{
						F[3*(i-ID_min)] -= x[3*i]*get_BH_radial_force_correction(r_xy, Rsim, RADIAL_BH_FORCE_TABLE, RADIAL_BH_FORCE_TABLE_SIZE, RADIAL_BH_FORCE_ORDER)/r_xy;
						F[3*(i-ID_min)+1] -= x[3*i+1]*get_BH_radial_force_correction(r_xy, Rsim, RADIAL_BH_FORCE_TABLE, RADIAL_BH_FORCE_TABLE_SIZE, RADIAL_BH_FORCE_ORDER)/r_xy;
					}
				}
				else if(COSMOLOGY == 1 && COMOVING_INTEGRATION == 0)
				{
					F[3*(i-ID_min)] +=  DE * x[3*i];
					F[3*(i-ID_min)+1] += DE * x[3*i+1];
				} //non-comoving integration is not implemented for periodic_z (yet?)
			}
    }
	free_node(rootnode);
	#ifdef RANDOMIZE_BH
	for (int i = 0; i < N; i++)
	{
		x[3*i+2] -= random_shift[2]; //shifting back to the original position with periodic boundary conditions
		//Checking the periodic boundaries along the z axis
		if(x[3*i+2]<0)
		{
			x[3*i+2] = x[3*i+2] + L;
		}
		else if(x[3*i+2]>=L)
		{
			x[3*i+2] = x[3*i+2] - L;
		}
    }
	//rotating back the the simulation volume to its original orientation
	rotate_vectors_2d(x, -rotation_angle, 0, N-1);
	rotate_vectors_2d(F, -rotation_angle, 0, N_mpi_thread-1);
	#endif
    //timing
    double omp_end_time = omp_get_wtime();
    //timing
    printf("Octree force calculation finished on MPI task %i. Force calculation wall-clock time = %fs.\n", rank, omp_end_time-omp_start_time);
    return;
}

#else
//Direct force calculation with multiple images only in the z direction
void forces_periodic_z(REAL* x, REAL* F, int ID_min, int ID_max)
{
    //timing
    double omp_start_time = omp_get_wtime();
    //timing
    REAL Fx_tmp, Fy_tmp, Fz_tmp, beta_priv,cylindrical_force_correction, r_xy;
	REAL DE = (REAL) H0*H0*Omega_lambda;
    #if !defined(PERIODIC_Z_NOLOOKUP)
	REAL D[3];
	#else
	int m;
	REAL dz_image;
	#endif
    int i, j, k, chunk;
    for(i=0; i<N_mpi_thread; i++)
    {
        for(k=0; k<3; k++)
        {
            F[3*i+k] = 0;
        }
    }
    REAL r, dx, dy, dz, wij;
    chunk = (ID_max-ID_min)/(omp_get_max_threads());
    if(chunk < 1)
    {
        chunk = 1;
    }
    if(IS_PERIODIC>=2)
	{
		#ifdef PERIODIC_Z_NOLOOKUP
		// Direct summation in real space with multiple images in the z direction
        #pragma omp parallel default(shared)  private(dx, dy, dz, r, wij, i, j, m, Fx_tmp, Fy_tmp, Fz_tmp, beta_priv, dz_image)
            #pragma omp for schedule(dynamic,chunk)
                for(i=ID_min; i<ID_max+1; i++)
				{
                    for(j=0; j<N; j++)
					{
                        Fx_tmp = 0;
                        Fy_tmp = 0;
                        Fz_tmp = 0;
                        beta_priv = (SOFT_LENGTH[i] + SOFT_LENGTH[j]);
                        //calculating particle distances inside the simulation volume
                        dx = x[3*j] - x[3*i];
                        dy = x[3*j+1] - x[3*i+1];
                        dz = x[3*j+2] - x[3*i+2];
                        //In here, we use multiple images but only in the z direction.
						//Summing over 2*ewald_max+1 images (7=3+1+3, if IS_PERIODIC==2) in the z direction
                        for(m=-ewald_max; m<ewald_max+1; m++)
                        {
							//calculating the distance in the z direction
							dz_image = dz+((REAL) m)*L;
                            r = sqrt(pow(dx, 2) + pow(dy, 2) + pow(dz_image, 2));
                            wij = 0;
                            if(fabs(dz_image) <= ewald_cut*L)
                            {
								//applying a cutoff at ewald_cut*L (2.6*L, if IS_PERIODIC==2)
                                wij = M[j] * force_softening(r, beta_priv);
                                Fx_tmp += wij*(dx);
                                Fy_tmp += wij*(dy);
                                Fz_tmp += wij*(dz_image);
                            }
                        }
                        #pragma omp atomic
                            F[3*(i-ID_min)] += Fx_tmp;
                        #pragma omp atomic
                            F[3*(i-ID_min)+1] += Fy_tmp;
                        #pragma omp atomic
                            F[3*(i-ID_min)+2] += Fz_tmp;
                    }
                    //adding the external force from the outside of the simulation volume,
                    //if we run non-periodic comoving cosmological simulation
                    //only include this in the X and Y directions
                    if(COSMOLOGY == 1 && COMOVING_INTEGRATION == 1)
                    {
						r_xy = sqrt(pow(x[3*i], 2) + pow(x[3*i+1], 2));
                        cylindrical_force_correction = get_cylindrical_force_correction(r_xy, Rsim, RADIAL_FORCE_TABLE, RADIAL_FORCE_TABLE_SIZE, 1);
                        F[3*(i-ID_min)] += mass_in_unit_sphere * x[3*i] * cylindrical_force_correction;
                        F[3*(i-ID_min)+1] += mass_in_unit_sphere * x[3*i+1] * cylindrical_force_correction;
                    }
                    else if(COSMOLOGY == 1 && COMOVING_INTEGRATION == 0)
                    {
                        F[3*(i-ID_min)] +=  DE * x[3*i];
                        F[3*(i-ID_min)+1] += DE * x[3*i+1];
                    } //non-comoving integration is not implemented for periodic_z (yet?)
                }
		#else
		// Using Ewald summation results from the lookup table
		#pragma omp parallel default(shared) private(dx, dy, dz, r, wij, j, i, Fx_tmp, Fy_tmp, Fz_tmp, D, beta_priv)
            #pragma omp for schedule(dynamic,chunk)
                for(i=ID_min; i<ID_max+1; i++)
				{
                    for(j=0; j<N; j++)
                    {
                        beta_priv = (SOFT_LENGTH[i] + SOFT_LENGTH[j]);
                        //calculating particle distances
                        dx = x[3*j] - x[3*i];
                        dy = x[3*j+1] - x[3*i+1];
                        dz = x[3*j+2] - x[3*i+2];
                        //in this case we use only the nearest image (and correct with Ewald table)
                        if(fabs(dz)>0.5*L)
						{
							dz = dz-L*dz/fabs(dz);
						}
                        r = sqrt(pow(dx, 2) + pow(dy, 2) + pow(dz, 2));
                        wij = force_softening(r, beta_priv);
						ewald_interpolate_D(S1R2_EWALD_FORCE_TABLE, Nrho_EWALD_FORCE_GRID, Nz_EWALD_FORCE_GRID, EWALD_LOOKUP_TABLE_RADIAL_EXTENT_FACTOR*Rsim, L, dx, dy, dz, D[0], D[1], D[2]);
                        Fx_tmp = M[j]*(wij*(dx) - D[0]);
                        Fy_tmp = M[j]*(wij*(dy) - D[1]);
                        Fz_tmp = M[j]*(wij*(dz) - D[2]);
                        #pragma omp atomic
                            F[3*(i-ID_min)] += Fx_tmp;
                        #pragma omp atomic
                            F[3*(i-ID_min)+1] += Fy_tmp;
                        #pragma omp atomic
                            F[3*(i-ID_min)+2] += Fz_tmp;
                    }
                    //adding the external force from the outside of the simulation volume,
                    //if we run non-periodic comoving cosmological simulation
                    //only include this in the X and Y directions
                    if(COSMOLOGY == 1 && COMOVING_INTEGRATION == 1)
                    {
                        F[3*(i-ID_min)] += mass_in_unit_sphere * x[3*i]; //-2\pi*G*rho_b*x
                        F[3*(i-ID_min)+1] += mass_in_unit_sphere * x[3*i+1];//-2\pi*G*rho_b*y
                    }
                    else if(COSMOLOGY == 1 && COMOVING_INTEGRATION == 0)
                    {
                        F[3*(i-ID_min)] +=  DE * x[3*i];
                        F[3*(i-ID_min)+1] += DE * x[3*i+1];
                    } //non-comoving integration is not implemented for periodic_z (yet?)
                }
		#endif

    }
    else
	{
		//Quasi-periodic force calculation in S^1 x R^2 topology
        #pragma omp parallel default(shared) private(dx, dy, dz, r, wij, j, i, Fx_tmp, Fy_tmp, Fz_tmp, beta_priv)
            #pragma omp for schedule(dynamic,chunk)
                for(i=ID_min; i<ID_max+1; i++)
				{
                    for(j=0; j<N; j++)
                    {
                        beta_priv = (SOFT_LENGTH[i] + SOFT_LENGTH[j]);
                        //calculating particle distances
                        dx = x[3*j] - x[3*i];
                        dy = x[3*j+1] - x[3*i+1];
                        dz = x[3*j+2] - x[3*i+2];
                        //in this case we use only the nearest image
                        if(fabs(dz)>0.5*L) { dz = dz-L*dz/fabs(dz); }
                        r = sqrt(pow(dx, 2) + pow(dy, 2) + pow(dz, 2));
                        wij = M[j] * force_softening(r, beta_priv);
                        Fx_tmp = wij*(dx);
                        Fy_tmp = wij*(dy);
                        Fz_tmp = wij*(dz);
                        #pragma omp atomic
                            F[3*(i-ID_min)] += Fx_tmp;
                        #pragma omp atomic
                            F[3*(i-ID_min)+1] += Fy_tmp;
                        #pragma omp atomic
                            F[3*(i-ID_min)+2] += Fz_tmp;
                    }
                    //adding the external force from the outside of the simulation volume,
                    //if we run non-periodic comoving cosmological simulation
                    //only include this in the X and Y directions
                    if(COSMOLOGY == 1 && COMOVING_INTEGRATION == 1)
                    {
						r_xy = sqrt(pow(x[3*i], 2) + pow(x[3*i+1], 2));
						cylindrical_force_correction = get_cylindrical_force_correction(r_xy, Rsim, RADIAL_FORCE_TABLE, RADIAL_FORCE_TABLE_SIZE, 1);
                        F[3*(i-ID_min)] += mass_in_unit_sphere * x[3*i] * cylindrical_force_correction;
                        F[3*(i-ID_min)+1] += mass_in_unit_sphere * x[3*i+1] * cylindrical_force_correction;
                    }
                    else if(COSMOLOGY == 1 && COMOVING_INTEGRATION == 0)
                    {
                        F[3*(i-ID_min)] +=  DE * x[3*i];
                        F[3*(i-ID_min)+1] += DE * x[3*i+1];
                    } //non-comoving integration is not implemented for periodic_z (yet?)
                }
    }
    //timing
    double omp_end_time = omp_get_wtime();
    //timing
    printf("Force calculation finished on MPI task %i. Force calculation wall-clock time = %fs.\n", rank, omp_end_time-omp_start_time);
    return;
}
#endif
#endif
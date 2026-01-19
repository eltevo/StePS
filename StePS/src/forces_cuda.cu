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
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define BLOCKSIZE 256


extern int e[2202][4];
extern REAL w[3];
extern int N, el;

#if !defined(PERIODIC) && !defined(PERIODIC_Z)
cudaError_t forces_cuda(REAL*x, REAL*F, int n_GPU, int ID_min, int ID_max);
#elif defined(PERIODIC)
int ewald_space(REAL R, int ewald_index[2102][4]);
cudaError_t forces_periodic_cuda(REAL*x, REAL*F, int n_GPU, int ID_min, int ID_max);
#elif defined(PERIODIC_Z)
cudaError_t forces_periodic_z_cuda(REAL*x, REAL*F, int n_GPU, int ID_min, int ID_max);

__device__ REAL cuda_linear_interpolation(REAL X, REAL X1, REAL Y1, REAL X2, REAL Y2)
{

    //helper function for linear interpolation
	//         Y2
	//       / |
	//      ?  |
	//    / |  |
	//  Y1  |  |
	//  |   |  |
	//--X1--X--X2
	REAL A=(Y2-Y1)/(X2-X1);
	REAL B=Y1-A*X1;
	return A*X+B;
}


//Function to interpolate a force for a given r, based on the values stored in the force table
__device__ REAL get_cylindrical_force_correction(REAL r, REAL R, const REAL *FORCE_TABLE, int TABLE_SIZE)
{
	//only linear interpolation is on GPUs, because it is faster than the CPU version
    REAL step = R / (REAL) TABLE_SIZE;
    int i = (int) floor(r / R * (TABLE_SIZE - 1)); 
    REAL correction = FORCE_TABLE[TABLE_SIZE - 1];
	if (i < TABLE_SIZE - 1)
	{
		correction = cuda_linear_interpolation(r, step * i, FORCE_TABLE[i], step * (i + 1), FORCE_TABLE[i + 1]);
	}
    return correction;
}
#endif

#if !defined(PERIODIC) && !defined(PERIODIC_Z)
void forces(REAL*x, REAL*F, int ID_min, int ID_max)
{
	forces_cuda(x, F, n_GPU, ID_min, ID_max);
	return;
}
#endif
#ifdef PERIODIC
__device__ static inline size_t lut_offset_cuda(int Ngrid, int ix, int iy, int iz, int comp)
{
    return ((size_t)((ix * Ngrid + iy) * Ngrid + iz) * 3u) + (size_t)comp;
}


__device__ int imodp(int i, int n)
{
	// positive modulo for periodic wrap
    int r = i % n;
    return (r < 0) ? (r + n) : r;
}
__device__ void map_to_centered_grid(REAL r, REAL L, int Ngrid, int *i0, REAL *fx)
{
	// Map a physical coordinate r in [-L/2,L/2] (or any real number) to the index of the *left* neighbor cell-center and the fractional offset fx in [0,1), given that centers lie at (i+0.5)*a - L/2.
    REAL grid_spacing  = L / (REAL)Ngrid;                 /* grid spacing */
    REAL u  = (r + L*(REAL)0.5) / grid_spacing - (REAL)0.5; /* continuous index @ centers */
    REAL uf = floor(u);
    *i0 = imodp((int)uf, Ngrid);
    *fx = (REAL)(u - uf); /* in [0,1) */
}

__device__ void get_cubic_weights(REAL t, REAL w[4])
{
    REAL t2 = t*t;
    REAL t3 = t*t2;
    w[0] = -0.5*t3 + t2 - 0.5*t;
    w[1] =  1.5*t3 - 2.5*t2 + 1.0;
    w[2] = -1.5*t3 + 2.0*t2 + 0.5*t;
    w[3] =  0.5*t3 - 0.5*t2;
}

__device__ void ewald_interpolate_cuda(int Ngrid, REAL L, const REAL *table, const REAL dx, const REAL dy, const REAL dz, REAL D[3])
{
    //Ewald interpolation in T^3 periodic torus topology using tri-cubic interpolation
    //Inputs:
    //    * Ngrid   :  Grid dimension
    //    * L       :  Box size
    //    * table   :  Force table
    //    * dx,dy,dz:  Relative particle position
    //    * D       :  Output force correction


    // Map to grid coordinates
    int ix0, iy0, iz0;
    REAL fx, fy, fz;
    map_to_centered_grid(dx, L, Ngrid, &ix0, &fx);
    map_to_centered_grid(dy, L, Ngrid, &iy0, &fy);
    map_to_centered_grid(dz, L, Ngrid, &iz0, &fz);

    REAL wx[4], wy[4], wz[4];
    get_cubic_weights(fx, wx);
    get_cubic_weights(fy, wy);
    get_cubic_weights(fz, wz);

    // Perform Tensor Product Summation
    REAL sum[3] = {0.0, 0.0, 0.0};
    
    // Offset to the starting neighbor index
    int offset = - 1;

    for (int i = 0; i < 4; i++) {
        int ix = imodp(ix0 + offset + i, Ngrid);
        REAL w_x = wx[i];

        for (int j = 0; j < 4; j++) {
            int iy = imodp(iy0 + offset + j, Ngrid);
            REAL w_xy = w_x * wy[j];

            for (int k = 0; k < 4; k++) {
                int iz = imodp(iz0 + offset + k, Ngrid);
                REAL w_xyz = w_xy * wz[k];

                // Fetch vector from table
                size_t idx = lut_offset_cuda(Ngrid, ix, iy, iz, 0);
                
                sum[0] += w_xyz * table[idx + 0];
                sum[1] += w_xyz * table[idx + 1];
                sum[2] += w_xyz * table[idx + 2];
            }
        }
    }

    D[0] = sum[0];
    D[1] = sum[1];
    D[2] = sum[2];
}

void forces_periodic(REAL*x, REAL*F, int ID_min, int ID_max)
{
	forces_periodic_cuda(x, F, n_GPU, ID_min, ID_max);
	return;
}
#endif
#ifdef PERIODIC_Z

__device__ static inline size_t lut2D_offset_cuda(int Nrho, int Nz, int ir, int iz, int comp)
{
    return ((size_t)ir * (size_t)Nz + (size_t)iz) * 2u + (size_t)comp;
}

__device__ __forceinline__ int wrap(int i, int Nz)
{
    int r = i % Nz;
    return (r < 0) ? (r + Nz) : r;
}

__device__ void ngp_interp_S1R2ewald_D_cuda(const REAL* T, const int Nrho, const int Nz, const REAL rho_max, const REAL Lz, const REAL rho, const REAL z, REAL D_rhoz[2])
{
	// Nearest Grid Point interpolation in S^1 x R^2 cylindrical periodic topology for Ewald force correction table
    // Clamp rho into [0, rho_max]
	REAL rho_clamped = rho;
    if (rho < (REAL)0) rho_clamped = (REAL)0;
    if (rho > rho_max) rho_clamped = rho_max;

    const REAL half = (REAL)0.5 * Lz;

    // Grid spacings
    const REAL drho = rho_max / (REAL)max(1, Nrho - 1);
    const REAL dz   = Lz      / (REAL)Nz;

    // Continuous indices using same cell-center convention as CIC/TSC
    const REAL ur = (drho > (REAL)0) ? (rho_clamped / drho) : (REAL)0;
    const REAL uz = (z + half) / dz - (REAL)0.5;

    // NGP = nearest integer index
    int ir = (int)floor(ur + (REAL)0.5);
    int iz = (int)floor(uz + (REAL)0.5);

    // rho index clamped
    if (ir < 0)          ir = 0;
    if (ir > Nrho - 1)   ir = Nrho - 1;

    // z index wrapped periodically
	iz = wrap(iz, Nz);

    // Compute base offset only once: ((ir*Nz) * 2)
    const size_t base = (size_t)ir * (size_t)Nz * 2u + (size_t)iz * 2u;

    // Load the two components directly
    D_rhoz[0] = T[base + 0];
    D_rhoz[1] = T[base + 1];
}

__device__ void cic_interp_S1R2ewald_D_cuda(const REAL* __restrict T,  const int Nrho, const int Nz, const REAL rho_max, const REAL Lz, const REAL rho, REAL z, REAL D_rhoz[2])
{
    // Bilinear CIC interpolation on (rho,z) grid with z periodic.
	REAL ur;
	REAL drho = rho_max / (REAL)max(1, Nrho-1);
    if (rho < (REAL)0)
	{
		// Clamping rho: rho = (REAL)0;
		ur=0;
	}
    else if (rho > rho_max)
	{
		// Clamping rho: rho = rho_max;
		ur = (REAL)(Nrho - 1);
	}
	else
	{
		ur = rho / drho;
	}
    int  ir0 = (int)floor(ur);
    REAL half = (REAL)0.5 * Lz; //half box length

    
    REAL dz   = Lz / (REAL)Nz;

    REAL fr  = ur - (REAL)ir0;
    if(ir0 < 0)
    {
        ir0 = 0;
        fr = 0;
    }
    if (ir0 > Nrho-2)
    {
        ir0 = max(0, Nrho-2);
        fr = (REAL)1;
    }
    int ir1 = ir0 + 1;

    REAL uz = (z + half) / dz - (REAL)0.5;
    int  iz0 = (int)floor(uz);
    REAL fz  = uz - (REAL)iz0;
    // wrap indices in z direction
    iz0 = wrap(iz0, Nz);
    const int iz1 = wrap(iz0 + 1, Nz);

    // weights
    REAL w00 = (REAL)1 - fr;
    w00 *= ((REAL)1 - fz);
    REAL w10 = fr;
    w10 *= ((REAL)1 - fz);
    REAL w01 = (REAL)1 - fr;
    w01 *= fz;
    REAL w11 = fr * fz;

    auto get = [&](int ir, int iz, int c) -> REAL {return T[lut2D_offset_cuda(Nrho, Nz, ir, iz, c)];};

    D_rhoz[0] = w00*get(ir0,iz0,0) + w10*get(ir1,iz0,0) + w01*get(ir0,iz1,0) + w11*get(ir1,iz1,0);
    D_rhoz[1] = w00*get(ir0,iz0,1) + w10*get(ir1,iz0,1) + w01*get(ir0,iz1,1) + w11*get(ir1,iz1,1);
}

__device__ void tsc_interp_S1R2ewald_D_cuda(const REAL* __restrict T, const int Nrho, const int Nz, const REAL rho_max, const REAL Lz, const REAL rho, const REAL z, REAL D_rhoz[2])
{
    // Triangular Shaped Cloud (TSC) interpolation on (rho,z) grid with z periodic.

    REAL ur;
	const REAL drho = rho_max / (REAL)max(1, Nrho - 1);
	// Continuous indices in rho
    if (rho < (REAL)0)
	{
		// Clamping rho: rho = (REAL)0;
		ur = (REAL)0;
	}
    else if(rho > rho_max)
	{
		// Clamping rho: rho = rho_max;
		ur = (REAL)(Nrho - 1);
	}
	else
	{
		ur = rho / drho;
	}

    const REAL half = (REAL)0.5 * Lz; //half box length
    const REAL dz   = Lz / (REAL)Nz;

    // Continuous indices in z
    const REAL uz = (z + half) / dz - (REAL)0.5; // centers, same convention as CIC

    // Centers (nearest grid points)
    const int jr = (int)floor(ur + (REAL)0.5);
    const int jz = (int)floor(uz + (REAL)0.5);

    // Fractional offsets relative to centers in [-0.5, 0.5)
    const REAL sr = ur - (REAL)jr;
    const REAL sz = uz - (REAL)jz;

    // 1D TSC weights in rho (left=jr-1, center=jr, right=jr+1)
    // These formulas assume sr in [-0.5, 0.5). They sum to 1.
    const REAL wrm = (REAL)0.5 * ((REAL)0.5 - sr) * ((REAL)0.5 - sr); // jr-1
    const REAL wrc = (REAL)0.75 - sr * sr;                             // jr
    const REAL wrp = (REAL)0.5 * ((REAL)0.5 + sr) * ((REAL)0.5 + sr);  // jr+1

    // 1D TSC weights in z (left=jz-1, center=jz, right=jz+1)
    // These formulas assume sz in [-0.5, 0.5). They sum to 1.
    const REAL wzm = (REAL)0.5 * ((REAL)0.5 - sz) * ((REAL)0.5 - sz);
    const REAL wzc = (REAL)0.75 - sz * sz;
    const REAL wzp = (REAL)0.5 * ((REAL)0.5 + sz) * ((REAL)0.5 + sz);

    // Neighbor indices (rho: clamped, z: wrapped)
    int ir0 = jr - 1;
    if (ir0 < 0)
    {
        ir0 = 0;
    }
    int ir1 = jr;
    if(ir1 < 0)
    {
        ir1 = 0;
    }
    if(ir1 > Nrho - 1)
    {
        ir1 = Nrho - 1;
    }
    int ir2 = jr + 1;
    if(ir2 > Nrho - 1)
    {
        ir2 = Nrho - 1;
    }

    const int iz0 = wrap(jz - 1, Nz);
    const int iz1 = wrap(jz, Nz);
    const int iz2 = wrap(jz + 1, Nz);

    
    // Precompute row bases: ((ir * Nz) * 2)
    const size_t base_ir0 = (size_t)ir0 * (size_t)Nz * 2u;
    const size_t base_ir1 = (size_t)ir1 * (size_t)Nz * 2u;
    const size_t base_ir2 = (size_t)ir2 * (size_t)Nz * 2u;

    // Precompute column offsets: (iz * 2)
    const size_t col_iz0 = (size_t)iz0 * 2u;
    const size_t col_iz1 = (size_t)iz1 * 2u;
    const size_t col_iz2 = (size_t)iz2 * 2u;

    // Row-wise z weights
    const REAL wz_row0 = wzm;
    const REAL wz_row1 = wzc;
    const REAL wz_row2 = wzp;

    D_rhoz[0] = (REAL)0.0;
    D_rhoz[1] = (REAL)0.0;

    // Accumulate D_rho and D_z over 3x3 stencil
    // Unrolling the loops for performance
    // I used the definition of lut2D_offset/lut2D_offset_cuda to compute the offsets directly. If that changes, this code must be updated accordingly.
    // Row iz0
    {
        const REAL wz = wz_row0;
        const REAL wr0 = wrm * wz, wr1 = wrc * wz, wr2 = wrp * wz;

        const size_t off00 = base_ir0 + col_iz0;
        const size_t off10 = base_ir1 + col_iz0;
        const size_t off20 = base_ir2 + col_iz0;

        D_rhoz[0] += wr0 * T[off00 + 0] + wr1 * T[off10 + 0] + wr2 * T[off20 + 0];
        D_rhoz[1] += wr0 * T[off00 + 1] + wr1 * T[off10 + 1] + wr2 * T[off20 + 1];
    }

    // Row iz1
    {
        const REAL wz = wz_row1;
        const REAL wr0 = wrm * wz, wr1 = wrc * wz, wr2 = wrp * wz;

        const size_t off01 = base_ir0 + col_iz1;
        const size_t off11 = base_ir1 + col_iz1;
        const size_t off21 = base_ir2 + col_iz1;

        D_rhoz[0] += wr0 * T[off01 + 0] + wr1 * T[off11 + 0] + wr2 * T[off21 + 0];
        D_rhoz[1] += wr0 * T[off01 + 1] + wr1 * T[off11 + 1] + wr2 * T[off21 + 1];
    }

    // Row iz2
    {
        const REAL wz = wz_row2;
        const REAL wr0 = wrm * wz, wr1 = wrc * wz, wr2 = wrp * wz;

        const size_t off02 = base_ir0 + col_iz2;
        const size_t off12 = base_ir1 + col_iz2;
        const size_t off22 = base_ir2 + col_iz2;

        D_rhoz[0] += wr0 * T[off02 + 0] + wr1 * T[off12 + 0] + wr2 * T[off22 + 0];
        D_rhoz[1] += wr0 * T[off02 + 1] + wr1 * T[off12 + 1] + wr2 * T[off22 + 1];
    }
}

__device__ void ewald_interpolate_D_cuda( const REAL* table, const int Nrho, const int Nz, const REAL rho_max, const REAL Lz, const REAL dx, const REAL dy, const REAL dz, REAL D[3])
{
	// This CUDA kernel calculates the force correction vector (per unit G*m_i*m_j) using the table stored in table for S^1 x R^2 cylindrical periodic topology
	// Inputs:
	//    * table    :  Ewald force correction table
	//    * Nrho     :  Number of radial grid points in the table
	//    * Nz       :  Number of axial grid points in the table
	//    * rho_max  :  Maximum radial distance in the table
	//    * Lz       :  Axial box size
	//    * dx,dy,dz :  Relative particle position
	// Outputs:
	//    * D :  Output force correction vector components
    REAL rho = sqrt(dx*dx + dy*dy); //radial distance

    REAL D_rhoz[2];
    #if EWALD_INTERPOLATION_ORDER == 0
		//NGP interpolation
		ngp_interp_S1R2ewald_D_cuda(table, Nrho, Nz, rho_max, Lz, rho, dz, D_rhoz);
    #elif EWALD_INTERPOLATION_ORDER == 2
		//CIC interpolation
        cic_interp_S1R2ewald_D_cuda(table, Nrho, Nz, rho_max, Lz, rho, dz, D_rhoz);
    #elif EWALD_INTERPOLATION_ORDER == 4
		//TSC interpolation
        tsc_interp_S1R2ewald_D_cuda(table, Nrho, Nz, rho_max, Lz, rho, dz, D_rhoz);
    #else
        //This should not happen, fallback to TSC
        tsc_interp_S1R2ewald_D_cuda(table, Nrho, Nz, rho_max, Lz, rho, dz, D_rhoz);
    #endif

    // cylindrical -> Cartesian
    REAL ex = (rho > 0) ? dx/rho : (REAL)0;
    REAL ey = (rho > 0) ? dy/rho : (REAL)0;
    D[0] = 	D_rhoz[0] * ex;
    D[1] = D_rhoz[0] * ey;
    D[2] = D_rhoz[1]; // D[2] ("z component") already axial
}

void forces_periodic_z(REAL*x, REAL*F, int ID_min, int ID_max)
{
    forces_periodic_z_cuda(x, F, n_GPU, ID_min, ID_max);
    return;
}
#endif

void recalculate_softening();

__device__ REAL force_softening_cuda(REAL r, REAL beta)
{

    //This CUDA kernel calculates the softened force coefficient between two particles
	//Only cubic spline softening is implemented here. New softening types can be added later.
	//Input:
	//    * r - distance between the two particles
	//    * beta - softening length
	//Output:
	//    * wij - softened force coefficient (1/r^3 for non-softened force)
	REAL betap2 = beta*0.5;
	REAL wij;
	wij = 0.0;
	if(r >= beta)
	{
		wij = pow(r, -3);
	}
	else if(r > betap2 && r < beta)
	{
		REAL SOFT_CONST0 = -32.0/(3.0*pow(beta, 6));
		REAL SOFT_CONST1 = 38.4/pow(beta, 5);
		REAL SOFT_CONST2 = -48.0/pow(beta, 4);
		REAL SOFT_CONST3 = 64.0/(3.0*pow(beta, 3));
		REAL SOFT_CONST4 = -1.0/15.0;
		wij = SOFT_CONST0*pow(r, 3)+SOFT_CONST1*pow(r, 2)+SOFT_CONST2*r+SOFT_CONST3+SOFT_CONST4/pow(r, 3);
	}
	else
	{
		REAL SOFT_CONST0 = 32.0/pow(beta, 6);
		REAL SOFT_CONST1 = -38.4/pow(beta, 5);
		REAL SOFT_CONST2 = 32.0/(3.0*pow(beta, 3));
		wij = SOFT_CONST0*pow(r, 3)+SOFT_CONST1*pow(r, 2)+SOFT_CONST2;
	}
	return wij;

}

#if !defined(PERIODIC) && !defined(PERIODIC_Z)
__global__ void ForceKernel(int n, const int N, const REAL *xx, const REAL *xy, const REAL *xz, REAL *F, const REAL* M, const REAL* SOFT_LENGTH, const REAL mass_in_unit_sphere, const REAL DE, const int COSMOLOGY, const int COMOVING_INTEGRATION, int ID_min, int ID_max)
{
	REAL Fx_tmp, Fy_tmp, Fz_tmp;
	REAL r, dx, dy, dz, wij, beta_priv;
	int i, j, id;
	id = blockIdx.x * blockDim.x + threadIdx.x;
	Fx_tmp = Fy_tmp = Fz_tmp = 0.0;
	for (i = (ID_min+id); i<=ID_max; i+=n)
		{
			for (j = 0; j<N; j++)
			{
				beta_priv = (SOFT_LENGTH[i]+SOFT_LENGTH[j]);
				//calculating particle distances
				dx = (xx[j] - xx[i]);
				dy = (xy[j] - xy[i]);
				dz = (xz[j] - xz[i]);
				r = sqrt(pow(dx, 2) + pow(dy, 2) + pow(dz, 2));
				wij = M[j] * force_softening_cuda(r, beta_priv);
				Fx_tmp += wij*(dx);
				Fy_tmp += wij*(dy);
				Fz_tmp += wij*(dz);

			}
			if(COSMOLOGY == 1 && COMOVING_INTEGRATION == 1)//Adding the external force from the outside of the simulation volume, if we run non-periodic comoving cosmological simulation
			{
				Fx_tmp += mass_in_unit_sphere * xx[i];
				Fy_tmp += mass_in_unit_sphere * xy[i];
				Fz_tmp += mass_in_unit_sphere * xz[i];
			}
			else if(COSMOLOGY == 1 && COMOVING_INTEGRATION == 0)
			{
				Fx_tmp += DE * xx[i];
				Fy_tmp += DE * xy[i];
				Fz_tmp += DE * xz[i];
			}
			F[3*(i-ID_min)] += Fx_tmp;
			F[3*(i-ID_min)+1] += Fy_tmp;
			F[3*(i-ID_min)+2] += Fz_tmp;
			Fx_tmp = Fy_tmp = Fz_tmp = 0.0;

		}
}
#endif

#ifdef PERIODIC
__global__ void ForceKernel_periodic(int n, int N, const REAL *xx, const REAL *xy, const REAL *xz, REAL *F, const int IS_PERIODIC, const REAL* M, const REAL* SOFT_LENGTH, const REAL L, const REAL *EwaldTable, int NewaldGrid, int ID_min, int ID_max)
{
	// This kernel calculates forces in periodic T^3 topology
	REAL Fx_tmp, Fy_tmp, Fz_tmp;
	REAL r, dx, dy, dz, wij, beta_priv;
	REAL D[3];
	int i, j, id;
	id = blockIdx.x * blockDim.x + threadIdx.x;
	Fx_tmp = Fy_tmp = Fz_tmp = 0.0;
	if (IS_PERIODIC == 1)
	{
		// Quasi-periodic case: only the nearest image is used
		for (i = (ID_min+id); i<=ID_max; i+=n)
		{
			for (j = 0; j<N; j++)
			{
				beta_priv = (SOFT_LENGTH[i]+SOFT_LENGTH[j]);
				//calculating particle distances
				dx = (xx[j] - xx[i]);
				dy = (xy[j] - xy[i]);
				dz = (xz[j] - xz[i]);
				//in this quasi-periodic caes, we use only the nearest image
				if (fabs(dx)>0.5*L)
					dx = dx - L*dx / fabs(dx);
				if (fabs(dy)>0.5*L)
					dy = dy - L*dy / fabs(dy);
				if (fabs(dz)>0.5*L)
					dz = dz - L*dz / fabs(dz);
				r = sqrt(pow(dx, 2) + pow(dy, 2) + pow(dz, 2));
                                wij = 0.0;
				wij = M[j] * force_softening_cuda(r, beta_priv);
				Fx_tmp += wij*(dx);
				Fy_tmp += wij*(dy);
				Fz_tmp += wij*(dz);

			}
			F[3*(i-ID_min)] += Fx_tmp;
			F[3*(i-ID_min)+1] += Fy_tmp;
			F[3*(i-ID_min)+2] += Fz_tmp;
			Fx_tmp = Fy_tmp = Fz_tmp = 0.0;
		}
	}
	else if (IS_PERIODIC >= 2)
	{
		// Full periodic case: multiple images are used (Ewald lookup table)
		for (i = (ID_min+id); i<=ID_max; i+=n)
		{
			for (j = 0; j<N; j++)
			{
				beta_priv = (SOFT_LENGTH[i]+SOFT_LENGTH[j]);
				//calculating particle distances
				dx = (xx[j] - xx[i]);
				dy = (xy[j] - xy[i]);
				dz = (xz[j] - xz[i]);
				//Getting the softened nearest image force
				if (fabs(dx)>0.5*L)
					dx = dx - L*dx / fabs(dx);
				if (fabs(dy)>0.5*L)
					dy = dy - L*dy / fabs(dy);
				if (fabs(dz)>0.5*L)
					dz = dz - L*dz / fabs(dz);
				r = sqrt(pow(dx, 2) + pow(dy, 2) + pow(dz, 2));
                                wij = 0.0;
				wij = force_softening_cuda(r, beta_priv);
				//Calculating the Ewald force correction based on the lookup table
				ewald_interpolate_cuda(NewaldGrid, L, EwaldTable, dx, dy, dz, D);
				Fx_tmp += M[j] * (wij*dx - D[0]);
				Fy_tmp += M[j] * (wij*dy - D[1]);
				Fz_tmp += M[j] * (wij*dz - D[2]);

			}
			F[3*(i-ID_min)] += Fx_tmp;
			F[3*(i-ID_min)+1] += Fy_tmp;
			F[3*(i-ID_min)+2] += Fz_tmp;
			Fx_tmp = Fy_tmp = Fz_tmp = 0.0;
		}
	}

}
#endif

#ifdef PERIODIC_Z

#ifdef PERIODIC_Z_NOLOOKUP
// Direct summation over multiple real-space images in periodic Z direction
__global__ void ForceKernel_periodic_z(int n, int N, const REAL *xx, const REAL *xy, const REAL *xz, REAL *F, const int IS_PERIODIC, const REAL* M, const REAL* SOFT_LENGTH, const REAL L, const REAL Rsim, const REAL mass_in_unit_sphere, const REAL* RADIAL_FORCE_TABLE, const REAL RADIAL_FORCE_TABLE_SIZE, const REAL DE, const int COSMOLOGY, const int COMOVING_INTEGRATION, int ID_min, int ID_max)
{
    REAL Fx_tmp, Fy_tmp, Fz_tmp;
    REAL r, dx, dy, dz, dz_ewald, wij, beta_priv, ewald_cut, r_xy, cylindrical_force_correction;
    int i, j, m, id, ewald_max;
    id = blockIdx.x * blockDim.x + threadIdx.x;
    Fx_tmp = Fy_tmp = Fz_tmp = 0.0;
	ewald_max = IS_PERIODIC+1;
	ewald_cut = (((REAL) ewald_max)-0.4)*L;
    if (IS_PERIODIC == 1)
    {
        for (i = (ID_min+id); i<=ID_max; i+=n)
        {
            for (j = 0; j<N; j++)
            {
                beta_priv = (SOFT_LENGTH[i]+SOFT_LENGTH[j]);
                // Calculating particle distances
                dx = (xx[j] - xx[i]);
                dy = (xy[j] - xy[i]);
                dz = (xz[j] - xz[i]);
                
                // In this case, we use only the nearest image in Z direction
                if (fabs(dz)>0.5*L)
                    dz = dz - L*dz / fabs(dz);
                    
                r = sqrt(pow(dx, 2) + pow(dy, 2) + pow(dz, 2));
                wij = M[j] * force_softening_cuda(r, beta_priv);
                
                Fx_tmp += wij*(dx);
                Fy_tmp += wij*(dy);
                Fz_tmp += wij*(dz);
            }
            
            // Adding the external force from the outside of the simulation volume
            // Only include this in the X and Y directions
            if(COSMOLOGY == 1 && COMOVING_INTEGRATION == 1)
            {
				r_xy = sqrt(pow(xx[i], 2) + pow(xy[i], 2));
				cylindrical_force_correction = get_cylindrical_force_correction(r_xy, Rsim, RADIAL_FORCE_TABLE, RADIAL_FORCE_TABLE_SIZE);
                Fx_tmp += mass_in_unit_sphere * xx[i] * cylindrical_force_correction;
                Fy_tmp += mass_in_unit_sphere * xy[i] * cylindrical_force_correction;
            }
            else if(COSMOLOGY == 1 && COMOVING_INTEGRATION == 0)
            {
                Fx_tmp += DE * xx[i];
                Fy_tmp += DE * xy[i];
            }
            
            F[3*(i-ID_min)] += Fx_tmp;
            F[3*(i-ID_min)+1] += Fy_tmp;
            F[3*(i-ID_min)+2] += Fz_tmp;
            Fx_tmp = Fy_tmp = Fz_tmp = 0.0;
        }
    }
    else
    {
		//  if IS_PERIODIC >= 2 we use multiple real-space images directly
        for (i = (ID_min+id); i<=ID_max; i+=n)
        {
            for (j = 0; j<N; j++)
            {
                beta_priv = (SOFT_LENGTH[i]+SOFT_LENGTH[j]);
                
                // Calculating particle distances inside the simulation box
                dx = (xx[j] - xx[i]);
                dy = (xy[j] - xy[i]);
                dz = (xz[j] - xz[i]);
                
                // In here we use multiple images, but only in the z direction
                for (m = -ewald_max; m < ewald_max+1; m++)
                {
					//calculating the distance in the z direction
					dz_ewald = dz+((REAL) m)*L;
                    r = sqrt(pow(dx, 2) + pow(dy, 2) + pow(dz_ewald, 2));
                    wij = 0.0;
                    
                    if (fabs(dz_ewald) < ewald_cut)
                    {
						wij = M[j] * force_softening_cuda(r, beta_priv);
                        Fx_tmp += wij*(dx);
                        Fy_tmp += wij*(dy);
                        Fz_tmp += wij*(dz_ewald);
                    }
                }
                
                F[3*(i-ID_min)] += Fx_tmp;
                F[3*(i-ID_min)+1] += Fy_tmp;
                F[3*(i-ID_min)+2] += Fz_tmp;
                Fx_tmp = Fy_tmp = Fz_tmp = 0.0;
            }
            
            // Adding the external force from the outside of the simulation volume
            // Only include this in the X and Y directions
            if(COSMOLOGY == 1 && COMOVING_INTEGRATION == 1)
            {
				r_xy = sqrt(pow(xx[i], 2) + pow(xy[i], 2));
				cylindrical_force_correction = get_cylindrical_force_correction(r_xy, Rsim, RADIAL_FORCE_TABLE, RADIAL_FORCE_TABLE_SIZE);
                F[3*(i-ID_min)] += mass_in_unit_sphere * xx[i] * cylindrical_force_correction;
                F[3*(i-ID_min)+1] += mass_in_unit_sphere * xy[i] * cylindrical_force_correction;
            }
            else if(COSMOLOGY == 1 && COMOVING_INTEGRATION == 0)
            {
                F[3*(i-ID_min)] += DE * xx[i];
                F[3*(i-ID_min)+1] += DE * xy[i];
            } //non-comoving integration is not implemented for periodic_z (yet?)
        }
    }
}
#else
//Using Ewald lookup table for periodic_z force calculation
__global__ void ForceKernel_periodic_z(int n, const int N, const REAL *xx, const REAL *xy, const REAL *xz, REAL *F, const int IS_PERIODIC, const REAL* M, const REAL* SOFT_LENGTH, const REAL L, const REAL Rsim, const REAL mass_in_unit_sphere, const REAL* RADIAL_FORCE_TABLE, const int RADIAL_FORCE_TABLE_SIZE, const REAL* EWALD_TABLE, const size_t EWALD_TABLE_SIZE_RHO, const size_t EWALD_TABLE_SIZE_Z, const REAL DE, const int COSMOLOGY, const int COMOVING_INTEGRATION, int ID_min, int ID_max)
{
    REAL Fx_tmp, Fy_tmp, Fz_tmp;
	REAL D[3];
    REAL r, dx, dy, dz, wij, beta_priv, r_xy, cylindrical_force_correction;
    int i, j, id;
    id = blockIdx.x * blockDim.x + threadIdx.x;
    Fx_tmp = Fy_tmp = Fz_tmp = 0.0;
    if (IS_PERIODIC == 1)
    {
		//Quasi-periodic case: only the nearest image is used
        for (i = (ID_min+id); i<=ID_max; i+=n)
        {
            for (j = 0; j<N; j++)
            {
                beta_priv = (SOFT_LENGTH[i]+SOFT_LENGTH[j]);
                // Calculating particle distances
                dx = (xx[j] - xx[i]);
                dy = (xy[j] - xy[i]);
                dz = (xz[j] - xz[i]);
                
                // In this case, we use only the nearest image in Z direction
                if (fabs(dz)>0.5*L)
                    dz = dz - L*dz / fabs(dz);
                    
                r = sqrt(pow(dx, 2) + pow(dy, 2) + pow(dz, 2));
                wij = M[j] * force_softening_cuda(r, beta_priv);
                Fx_tmp += wij*(dx);
                Fy_tmp += wij*(dy);
                Fz_tmp += wij*(dz);
            }
            
            // Adding the external force from the outside of the simulation volume
            // Only include this in the X and Y directions
            if(COSMOLOGY == 1 && COMOVING_INTEGRATION == 1)
            {
				r_xy = sqrt(pow(xx[i], 2) + pow(xy[i], 2));
				cylindrical_force_correction = get_cylindrical_force_correction(r_xy, Rsim, RADIAL_FORCE_TABLE, RADIAL_FORCE_TABLE_SIZE);
                Fx_tmp += mass_in_unit_sphere * xx[i] * cylindrical_force_correction;
                Fy_tmp += mass_in_unit_sphere * xy[i] * cylindrical_force_correction;
            }
            else if(COSMOLOGY == 1 && COMOVING_INTEGRATION == 0)
            {
                Fx_tmp += DE * xx[i];
                Fy_tmp += DE * xy[i];
            }
            
            F[3*(i-ID_min)] += Fx_tmp;
            F[3*(i-ID_min)+1] += Fy_tmp;
            F[3*(i-ID_min)+2] += Fz_tmp;
            Fx_tmp = Fy_tmp = Fz_tmp = 0.0;
        }
    }
    else
    {
		//  Full periodic case: using Ewald lookup table
        for (i = (ID_min+id); i<=ID_max; i+=n)
        {
            for (j = 0; j<N; j++)
            {
                beta_priv = (SOFT_LENGTH[i]+SOFT_LENGTH[j]);
                // Calculating particle distances
                dx = (xx[j] - xx[i]);
                dy = (xy[j] - xy[i]);
                dz = (xz[j] - xz[i]);
                
                // In this case, we only need the nearest image in Z direction
                if (fabs(dz)>0.5*L)
                    dz = dz - L*dz / fabs(dz);
                    
                r = sqrt(pow(dx, 2) + pow(dy, 2) + pow(dz, 2));
                wij = force_softening_cuda(r, beta_priv);
				ewald_interpolate_D_cuda(EWALD_TABLE, EWALD_TABLE_SIZE_RHO, EWALD_TABLE_SIZE_Z, EWALD_LOOKUP_TABLE_RADIAL_EXTENT_FACTOR*Rsim, L, dx, dy, dz, D);
                Fx_tmp += M[j] * (wij*(dx) - D[0]);
                Fy_tmp += M[j] * (wij*(dy) - D[1]);
                Fz_tmp += M[j] * (wij*(dz) - D[2]);
            }
            // Adding the external force from the outside of the simulation volume
            // Only include this in the X and Y directions
            if(COSMOLOGY == 1 && COMOVING_INTEGRATION == 1)
            {
				//REAL cylindrical_force_correction = 0.78;
                Fx_tmp += mass_in_unit_sphere * xx[i];
                Fy_tmp += mass_in_unit_sphere * xy[i];
            }
            else if(COSMOLOGY == 1 && COMOVING_INTEGRATION == 0)
            {
                Fx_tmp += DE * xx[i];
                Fy_tmp += DE * xy[i];
            }
            
            F[3*(i-ID_min)] += Fx_tmp;
            F[3*(i-ID_min)+1] += Fy_tmp;
            F[3*(i-ID_min)+2] += Fz_tmp;
            Fx_tmp = Fy_tmp = Fz_tmp = 0.0;
        }
    }
}
#endif
#endif

void recalculate_softening()
{
	beta = ParticleRadi;
	if(COSMOLOGY ==1)
	{
		rho_part = M_min/(4.0*pi*pow(beta, 3.0) / 3.0);
	}
}

#if !defined(PERIODIC) && !defined(PERIODIC_Z)
cudaError_t forces_cuda(REAL*x, REAL*F, int n_GPU, int ID_min, int ID_max) //Force calculation on GPU
{
	int i, j;
	int mprocessors;
	int GPU_ID, nthreads;
	int N_GPU, GPU_index_min; //number of particles in this GPU, the first particles index
	cudaError_t cudaStatus;
	cudaStatus = cudaSuccess;
	double omp_start_time, omp_end_time;
	REAL DE = (REAL) H0*H0*Omega_lambda;
	REAL *xx_tmp, *xy_tmp, *xz_tmp, *F_tmp;
	REAL *dev_xx= 0;
	REAL *dev_xy= 0;
	REAL *dev_xz= 0;
	REAL *dev_M = 0;
	REAL *dev_SOFT_LENGTH = 0; //v0.3.7.1
	REAL *dev_F = 0;

	// Get the number of CUDA devices.
	int numDevices;
	cudaGetDeviceCount(&numDevices);
	if(numDevices<n_GPU)
	{
		if(numDevices == 1)
			fprintf(stderr, "Error: MPI rank %i: Cannot allocate %i GPUs, because only one is available\n", rank, n_GPU);
		else
			fprintf(stderr, "Error: MPI rank %i: Cannot allocate %i GPUs, because only %i are available\n", rank, n_GPU, numDevices);
		n_GPU = numDevices;
		printf("Number of GPUs set to %i\n", n_GPU);
	}

	if(!(xx_tmp = (REAL*)malloc(N*sizeof(REAL))))
	{
		fprintf(stderr, "MPI task %i: failed to allocate memory for xx_tmp (for CUDA force canculation).\n", rank);
		exit(-2);
	}
	if(!(xy_tmp = (REAL*)malloc(N*sizeof(REAL))))
	{
		fprintf(stderr, "MPI task %i: failed to allocate memory for xy_tmp (for CUDA force canculation).\n", rank);
		exit(-2);
	}
	if(!(xz_tmp = (REAL*)malloc(N*sizeof(REAL))))
	{
		fprintf(stderr, "MPI task %i: failed to allocate memory for xz_tmp (for CUDA force canculation).\n", rank);
		exit(-2);
	}
	for(i = 0; i < N; i++)
	{
		xx_tmp[i] = x[3*i];
		xy_tmp[i] = x[3*i+1];
		xz_tmp[i] = x[3*i+2];
	}
	//timing
	omp_start_time = omp_get_wtime();
	//timing
	omp_set_dynamic(0);		// Explicitly disable dynamic teams
	omp_set_num_threads(n_GPU);	// Use n_GPU threads
#pragma omp parallel default(shared) private(GPU_ID, F_tmp, i, j, mprocessors, cudaStatus, N_GPU, GPU_index_min, nthreads, dev_xx, dev_xy, dev_xz, dev_M, dev_SOFT_LENGTH, dev_F)
{
		#pragma omp critical
		{
		nthreads = omp_get_num_threads();
		GPU_ID = omp_get_thread_num(); //thread ID = GPU_ID
		}
		if(GPU_ID == 0)
		{
			N_GPU = (ID_max-ID_min+1)/n_GPU+(ID_max-ID_min+1)%n_GPU;
			GPU_index_min = ID_min;
		}
		else
		{
			N_GPU = (ID_max-ID_min+1)/n_GPU;
			GPU_index_min = ID_min + (ID_max-ID_min+1)%n_GPU+N_GPU*GPU_ID;
		}
		if(!(F_tmp = (REAL*)malloc(3 * N_GPU * sizeof(REAL))))
		{
			fprintf(stderr, "MPI task %i: failed to allocate memory for F_tmp (for CUDA force canculation).\n", rank);
			exit(-2);
		}
		for(i=0; i < N_GPU; i++)
		{
			for(j=0; j<3; j++)
			F_tmp[3*i + j] = 0.0f;
		}
		//Checking for the GPU
		#pragma omp critical
		cudaDeviceGetAttribute(&mprocessors, cudaDevAttrMultiProcessorCount, GPU_ID);
		if(GPU_ID == 0)
		{

			printf("MPI task %i: GPU force calculation.\n Number of GPUs: %i\n Number of OMP threads: %i\n Number of threads per GPU: %i\n", rank, n_GPU, nthreads, 32*mprocessors*BLOCKSIZE);
		}
		#pragma omp critical
		cudaStatus = cudaSetDevice(GPU_ID); //selecting the GPU
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "MPI rank %i: GPU%i: cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?\n", rank, GPU_ID);
			ForceError = true;
			goto Error;
		}
		// Allocate GPU buffers for coordinate and mass vectors
		cudaStatus = cudaMalloc((void**)&dev_xx, N * sizeof(REAL));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "MPI rank %i: GPU%i: xx cudaMalloc failed!\n", rank, GPU_ID);
			ForceError = true;
			goto Error;
		}
		cudaStatus = cudaMalloc((void**)&dev_xy, N * sizeof(REAL));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "MPI rank %i: GPU%i: xy cudaMalloc failed!\n", rank, GPU_ID);
			ForceError = true;
			goto Error;
		}
		cudaStatus = cudaMalloc((void**)&dev_xz, N * sizeof(REAL));
		if (cudaStatus != cudaSuccess) {
                	fprintf(stderr, "MPI rank %i: GPU%i: xz cudaMalloc failed!\n", rank, GPU_ID);
			ForceError = true;
			goto Error;
		}
		cudaStatus = cudaMalloc((void**)&dev_M, N * sizeof(REAL));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "MPI rank %i: GPU%i: M cudaMalloc failed!\n", rank, GPU_ID);
			ForceError = true;
			goto Error;
		}
		// Allocate GPU buffers for the softening vector
		cudaStatus = cudaMalloc((void**)&dev_SOFT_LENGTH, N * sizeof(REAL)); //v0.3.7.1
                if (cudaStatus != cudaSuccess) {
                        fprintf(stderr, "MPI rank %i: GPU%i: SOFT_LENGTH cudaMalloc failed!\n", rank, GPU_ID);
			ForceError = true;
                        goto Error;
                }
		// Allocate GPU buffers for force vectors
		cudaStatus = cudaMalloc((void**)&dev_F, 3 * N_GPU * sizeof(REAL));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "MPI rank %i: GPU%i: F cudaMalloc failed!\n", rank, GPU_ID);
			ForceError = true;
			goto Error;
		}
		// Copy input vectors from host memory to GPU buffers.
		cudaStatus = cudaMemcpy(dev_xx, xx_tmp, N * sizeof(REAL), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "MPI rank %i: GPU%i: cudaMemcpy xx in failed!\n", rank, GPU_ID);
			ForceError = true;
			goto Error;
		}
		cudaStatus = cudaMemcpy(dev_xy, xy_tmp, N * sizeof(REAL), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "MPI rank %i: GPU%i: cudaMemcpy xy in failed!\n", rank, GPU_ID);
			ForceError = true;
			goto Error;
		}
		cudaStatus = cudaMemcpy(dev_xz, xz_tmp, N * sizeof(REAL), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "MPI rank %i: GPU%i: cudaMemcpy xz in failed!\n", rank, GPU_ID);
			ForceError = true;
			goto Error;
		}
		cudaStatus = cudaMemcpy(dev_M, M, N * sizeof(REAL), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "MPI rank %i: GPU%i: cudaMemcpy M in failed!\n", rank, GPU_ID);
			ForceError = true;
			goto Error;
		}
		cudaStatus = cudaMemcpy(dev_SOFT_LENGTH, SOFT_LENGTH, N * sizeof(REAL), cudaMemcpyHostToDevice); // v0.3.7.1
                if (cudaStatus != cudaSuccess) {
                        fprintf(stderr, "MPI rank %i: GPU%i: cudaMemcpy SOFT_LENGTH in failed!\n", rank, GPU_ID);
			ForceError = true;
                        goto Error;
                }
		cudaStatus = cudaMemcpy(dev_F, F_tmp, 3 * N_GPU * sizeof(REAL), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "MPI rank %i: GPU%i: cudaMemcpy F in failed!\n", rank, GPU_ID);
			ForceError = true;
			goto Error;
		}
		printf("MPI task %i: GPU%i: ID_min = %i\tID_max = %i\n", rank, GPU_ID, GPU_index_min, GPU_index_min+N_GPU-1);
		// Launch a kernel on the GPU
		ForceKernel<<<32*mprocessors, BLOCKSIZE>>>(32 * mprocessors * BLOCKSIZE, N, dev_xx, dev_xy, dev_xz, dev_F, dev_M, dev_SOFT_LENGTH, mass_in_unit_sphere, DE, COSMOLOGY, COMOVING_INTEGRATION, GPU_index_min, GPU_index_min+N_GPU-1);
		// Check for any errors launching the kernel
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "MPI rank %i: GPU%i: ForceKernel launch failed: %s\n", rank, GPU_ID, cudaGetErrorString(cudaStatus));
			ForceError = true;
			goto Error;
		}
		// cudaDeviceSynchronize waits for the kernel to finish, and returns
		// any errors encountered during the launch.
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "MPI rank %i: GPU%i: cudaDeviceSynchronize returned error code %d after launching ForceKernel!\n", rank, GPU_ID, cudaStatus);
			ForceError = true;
			goto Error;
		}
		// Copy output vector from GPU buffer to host memory.
		cudaStatus = cudaMemcpy(F_tmp, dev_F, 3 * N_GPU * sizeof(REAL), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "MPI %i: GPU%i: cudaMemcpy F out failed!\n", rank, GPU_ID);
			ForceError = true;
			goto Error;
		}
		if(GPU_ID == 0)
		{
			for (i = 0; i < N_GPU; i++)
			{
				for (j = 0; j < 3; j++)
				{
					F[3*i+j] = F_tmp[(3 * i) + j];
				}
			}
		}
		else
		{
			for (i = GPU_index_min; i < GPU_index_min + N_GPU; i++)
			{
				for (j = 0; j < 3; j++)
				{
					F[3*(i-ID_min)+j] = F_tmp[3 * (i-GPU_index_min) + j];
				}
			}
		}
	free(F_tmp);
	Error:
		cudaFree(dev_xx);
                cudaFree(dev_xy);
                cudaFree(dev_xz);
                cudaFree(dev_M);
                cudaFree(dev_F);
		cudaFree(dev_SOFT_LENGTH);
		cudaDeviceReset();

}
	free(xx_tmp);
	free(xy_tmp);
	free(xz_tmp);
	//timing
	omp_end_time = omp_get_wtime();
	//timing
	printf("Force calculation finished on MPI task %i. Force calculation wall-clock time = %fs.\n", rank, omp_end_time-omp_start_time);
	return cudaStatus;
}
#endif

#ifdef PERIODIC
cudaError_t forces_periodic_cuda(REAL*x, REAL*F, int n_GPU, int ID_min, int ID_max) //Force calculation with multiple images on GPU
{
	int i, j;
	int mprocessors;
	int GPU_ID, nthreads;
	int N_GPU, GPU_index_min; //number of particles in this GPU, the first particles index
	cudaError_t cudaStatus;
	cudaStatus = cudaSuccess;
	double omp_start_time, omp_end_time;
	REAL *xx_tmp, *xy_tmp, *xz_tmp, *F_tmp;
	REAL *dev_xx= 0;
	REAL *dev_xy= 0;
	REAL *dev_xz= 0;
	REAL *dev_M = 0;
	REAL *dev_SOFT_LENGTH = 0;
	REAL *dev_F = 0;
	REAL *dev_EwaldTable = 0;

	int numDevices;
	cudaGetDeviceCount(&numDevices);
	if(numDevices<n_GPU)
	{
		if(numDevices == 1)
			fprintf(stderr, "Error: MPI rank %i: Cannot allocate %i GPUs, because only one is available\n", rank, n_GPU);
		else
			fprintf(stderr, "Error: MPI rank %i: Cannot allocate %i GPUs, because only %i are available\n", rank, n_GPU, numDevices);
		n_GPU = numDevices;
		printf("Number of GPUs set to %i\n", n_GPU);
	}

	//Converting the Nx3 matrix to 3Nx1 vector.
	if(!(xx_tmp = (REAL*)malloc(N*sizeof(REAL))))
	{
		fprintf(stderr, "MPI task %i: failed to allocate memory for xx_tmp (for CUDA force canculation).\n", rank);
		exit(-2);
	}
	if(!(xy_tmp = (REAL*)malloc(N*sizeof(REAL))))
	{
		fprintf(stderr, "MPI task %i: failed to allocate memory for xx_tmp (for CUDA force canculation).\n", rank);
		exit(-2);
	}
	if(!(xz_tmp = (REAL*)malloc(N*sizeof(REAL))))
	{
		fprintf(stderr, "MPI task %i: failed to allocate memory for xx_tmp (for CUDA force canculation).\n", rank);
		exit(-2);
	}
	for(i = 0; i < N; i++)
	{
		xx_tmp[i] = x[3*i];
		xy_tmp[i] = x[3*i+1];
		xz_tmp[i] = x[3*i+2];
	}
	//timing
	omp_start_time = omp_get_wtime();
        //timing
	omp_set_dynamic(0);             // Explicitly disable dynamic teams
	omp_set_num_threads(n_GPU);     // Use n_GPU threads
#pragma omp parallel default(shared) private(GPU_ID, F_tmp, i, j, mprocessors, cudaStatus, N_GPU, GPU_index_min, nthreads, dev_xx, dev_xy, dev_xz, dev_M, dev_F, dev_SOFT_LENGTH)
{
		#pragma omp critical
		{
		nthreads = omp_get_num_threads();
		GPU_ID = omp_get_thread_num(); //thread ID = GPU_ID
		}
		if(GPU_ID == 0)
		{
			N_GPU = (ID_max-ID_min+1)/n_GPU+(ID_max-ID_min+1)%n_GPU;
			GPU_index_min = ID_min;
		}
		else
		{
			N_GPU = (ID_max-ID_min+1)/n_GPU;
			GPU_index_min = ID_min + (ID_max-ID_min+1)%n_GPU+N_GPU*GPU_ID;
		}
		F_tmp = (REAL*)malloc(3 * N_GPU * sizeof(REAL));
		for(i=0; i < N_GPU; i++)
		{
			for(j=0; j<3; j++)
				F_tmp[3*i + j] = 0.0f;
		}
		//Checking for the GPU
		#pragma omp critical
		cudaDeviceGetAttribute(&mprocessors, cudaDevAttrMultiProcessorCount, GPU_ID);
		if(GPU_ID == 0)
		{
			if(IS_PERIODIC == 1)
				printf("MPI task %i: GPU force calculation (quasi-periodic).\n Number of GPUs: %i\n Number of OMP threads: %i\n Number of threads per GPU: %i\n", rank, n_GPU, nthreads, 32*mprocessors*BLOCKSIZE);
			else
				printf("MPI task %i: GPU force calculation (fully periodic with Ewald correction).\n Number of GPUs: %i\n Number of OMP threads: %i\n Number of threads per GPU: %i\n", rank, n_GPU, nthreads, 32*mprocessors*BLOCKSIZE);
		}
		#pragma omp critical
		cudaStatus = cudaSetDevice(GPU_ID); //selecting GPU
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "MPI rank %i: GPU%i: cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?\n", rank, GPU_ID);
			ForceError = true;
			goto Error;
		}
		// Allocate GPU buffers for coordinate and mass vectors
		cudaStatus = cudaMalloc((void**)&dev_xx, N * sizeof(REAL));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "MPI rank %i: GPU%i: xx cudaMalloc failed!\n", rank, GPU_ID);
			ForceError = true;
			goto Error;
		}
		cudaStatus = cudaMalloc((void**)&dev_xy, N * sizeof(REAL));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "MPI rank %i: GPU%i: xy cudaMalloc failed!\n", rank, GPU_ID);
			ForceError = true;
			goto Error;
		}
		cudaStatus = cudaMalloc((void**)&dev_xz, N * sizeof(REAL));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "MPI rank %i: GPU%i: xz cudaMalloc failed!\n", rank, GPU_ID);
			ForceError = true;
			goto Error;
		}
		cudaStatus = cudaMalloc((void**)&dev_M, N * sizeof(REAL));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "MPI rank %i: GPU%i: M cudaMalloc failed!\n", rank, GPU_ID);
			ForceError = true;
			goto Error;
		}
		// Allocate GPU buffers for the softening vector
		cudaStatus = cudaMalloc((void**)&dev_SOFT_LENGTH, N * sizeof(REAL)); //v0.3.7.1
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "MPI rank %i: GPU%i: SOFT_LENGTH cudaMalloc failed!\n", rank, GPU_ID);
			ForceError = true;
			goto Error;
		}
		// Allocate GPU buffers for force vectors
		cudaStatus = cudaMalloc((void**)&dev_F, 3 * N_GPU * sizeof(REAL));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "MPI rank %i: GPU%i: F cudaMalloc failed!\n", rank, GPU_ID);
			ForceError = true;
			goto Error;
		}
		// Allocate GPU buffers for Ewald lookup table
		cudaStatus = cudaMalloc((void**)&dev_EwaldTable, N_EWALD_FORCE_GRID * N_EWALD_FORCE_GRID * N_EWALD_FORCE_GRID * 3 * sizeof(REAL));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "MPI rank %i: GPU%i: EwaldTable cudaMalloc failed!\n", rank, GPU_ID);
			ForceError = true;
			goto Error;
		}
		// Copy input vectors from host memory to GPU buffers.
		cudaStatus = cudaMemcpy(dev_xx, xx_tmp, N * sizeof(REAL), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "MPI rank %i: GPU%i: cudaMemcpy xx in failed!\n", rank, GPU_ID);
			ForceError = true;
			goto Error;
		}
		cudaStatus = cudaMemcpy(dev_xy, xy_tmp, N * sizeof(REAL), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "MPI rank %i: GPU%i: cudaMemcpy xy in failed!\n", rank, GPU_ID);
			ForceError = true;
			goto Error;
		}
		cudaStatus = cudaMemcpy(dev_xz, xz_tmp, N * sizeof(REAL), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "MPI rank %i: GPU%i: cudaMemcpy xz in failed!\n", rank, GPU_ID);
			ForceError = true;
			goto Error;
		}
		cudaStatus = cudaMemcpy(dev_M, M, N * sizeof(REAL), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "MPI rank %i: GPU%i: cudaMemcpy M in failed!\n", rank, GPU_ID);
			ForceError = true;
			goto Error;
		}
		cudaStatus = cudaMemcpy(dev_SOFT_LENGTH, SOFT_LENGTH, N * sizeof(REAL), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "MPI rank %i: GPU%i: cudaMemcpy SOFT_LENGTH in failed!\n", rank, GPU_ID);
			ForceError = true;
			goto Error;
		}
		cudaStatus = cudaMemcpy(dev_F, F_tmp, 3 * N_GPU * sizeof(REAL), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "MPI rank %i: GPU%i: cudaMemcpy F in failed!\n", rank, GPU_ID);
			ForceError = true;
			goto Error;
		}
		// Copy Ewald lookup table to GPU
		cudaStatus = cudaMemcpy(dev_EwaldTable, T3_EWALD_FORCE_TABLE, N_EWALD_FORCE_GRID * N_EWALD_FORCE_GRID * N_EWALD_FORCE_GRID * 3 * sizeof(REAL), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "MPI rank %i: GPU%i: cudaMemcpy EwaldTable in failed!\n", rank, GPU_ID);
			ForceError = true;
			goto Error;
		}
		printf("MPI task %i: GPU%i: ID_min = %i\tID_max = %i\n", rank, GPU_ID, GPU_index_min, GPU_index_min+N_GPU-1);
		// Launch a kernel on the GPU with one thread for each element.
		ForceKernel_periodic << <32*mprocessors, BLOCKSIZE>> >(32*mprocessors * BLOCKSIZE, N, dev_xx, dev_xy, dev_xz, dev_F, IS_PERIODIC, dev_M, dev_SOFT_LENGTH, L, dev_EwaldTable, N_EWALD_FORCE_GRID, GPU_index_min, GPU_index_min+N_GPU-1);

		// Check for any errors launching the kernel
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "MPI rank %i: GPU%i: ForceKernel_periodic launch failed: %s\n", rank, GPU_ID, cudaGetErrorString(cudaStatus));
			ForceError = true;
			goto Error;
		}
		// cudaDeviceSynchronize waits for the kernel to finish, and returns
		// any errors encountered during the launch.
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "MPI rank %i: GPU%i: cudaDeviceSynchronize returned error code %d after launching ForceKernel_periodic!\n", rank, GPU_ID, cudaStatus);
			ForceError = true;
			goto Error;
		}
		// Copy output vector from GPU buffer to host memory.
		cudaStatus = cudaMemcpy(F_tmp, dev_F, 3 * N_GPU * sizeof(REAL), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "MPI rank %i: GPU%i: cudaMemcpy out failed!\n", rank, GPU_ID);
			ForceError = true;
			goto Error;
		}
		if(GPU_ID == 0)
		{
			for (i = 0; i < N_GPU; i++)
			{
				for (j = 0; j < 3; j++)
				{
					F[3*i+j] = F_tmp[(3 * i) + j];
				}
			}
		}
		else
		{
			for (i = GPU_index_min; i < GPU_index_min + N_GPU; i++)
			{
				for (j = 0; j < 3; j++)
				{
					F[3*(i-ID_min)+j] = F_tmp[3 * (i-GPU_index_min) + j];
				}
			}
		}
		free(F_tmp);
	Error:
		cudaFree(dev_xx);
		cudaFree(dev_xy);
		cudaFree(dev_xz);
		cudaFree(dev_M);
		cudaFree(dev_F);
		cudaFree(dev_SOFT_LENGTH);
		//Free Ewald lookup table
		cudaDeviceReset();
}
	free(xx_tmp);
	free(xy_tmp);
	free(xz_tmp);
	//timing
	omp_end_time = omp_get_wtime();
	//timing
	printf("Force calculation finished on MPI task %i. Force calculation wall-clock time = %fs.\n", rank, omp_end_time-omp_start_time);
	return cudaStatus;
}
#endif

#ifdef PERIODIC_Z
cudaError_t forces_periodic_z_cuda(REAL*x, REAL*F, int n_GPU, int ID_min, int ID_max) //Force calculation with z-periodic boundaries on GPU
{
    int i, j;
    int mprocessors;
    int GPU_ID, nthreads;
    int N_GPU, GPU_index_min; //number of particles in this GPU, the first particles index
    cudaError_t cudaStatus;
    cudaStatus = cudaSuccess;
    double omp_start_time, omp_end_time;
    REAL DE = (REAL) H0*H0*Omega_lambda;
    REAL *xx_tmp, *xy_tmp, *xz_tmp, *F_tmp;
	REAL *dev_RADIAL_FORCE_TABLE = 0;
    REAL *dev_xx= 0;
    REAL *dev_xy= 0;
    REAL *dev_xz= 0;
    REAL *dev_M = 0;
    REAL *dev_SOFT_LENGTH = 0;
    REAL *dev_F = 0;

    // Get the number of CUDA devices
    int numDevices;
    cudaGetDeviceCount(&numDevices);
    if(numDevices<n_GPU)
    {
        if(numDevices == 1)
            fprintf(stderr, "Error: MPI rank %i: Cannot allocate %i GPUs, because only one is available\n", rank, n_GPU);
        else
            fprintf(stderr, "Error: MPI rank %i: Cannot allocate %i GPUs, because only %i are available\n", rank, n_GPU, numDevices);
        n_GPU = numDevices;
        printf("Number of GPUs set to %i\n", n_GPU);
    }

    // Converting the Nx3 matrix to 3 Nx1 vectors for more efficient GPU processing
    if(!(xx_tmp = (REAL*)malloc(N*sizeof(REAL))))
    {
        fprintf(stderr, "MPI task %i: failed to allocate memory for xx_tmp (for CUDA force calculation).\n", rank);
        exit(-2);
    }
    if(!(xy_tmp = (REAL*)malloc(N*sizeof(REAL))))
    {
        fprintf(stderr, "MPI task %i: failed to allocate memory for xy_tmp (for CUDA force calculation).\n", rank);
        exit(-2);
    }
    if(!(xz_tmp = (REAL*)malloc(N*sizeof(REAL))))
    {
        fprintf(stderr, "MPI task %i: failed to allocate memory for xz_tmp (for CUDA force calculation).\n", rank);
        exit(-2);
    }
    
    for(i = 0; i < N; i++)
    {
        xx_tmp[i] = x[3*i];
        xy_tmp[i] = x[3*i+1];
        xz_tmp[i] = x[3*i+2];
    }
    
    //timing
    omp_start_time = omp_get_wtime();
    //timing
    
    omp_set_dynamic(0);             // Explicitly disable dynamic teams
    omp_set_num_threads(n_GPU);     // Use n_GPU threads
    
#pragma omp parallel default(shared) private(GPU_ID, F_tmp, i, j, mprocessors, cudaStatus, N_GPU, GPU_index_min, nthreads, dev_xx, dev_xy, dev_xz, dev_M, dev_F, dev_SOFT_LENGTH)
    {
		#if !defined(PERIODIC_Z_NOLOOKUP)
			REAL *dev_S1R2_EwaldTable = 0;
		#endif
        #pragma omp critical
        {
            nthreads = omp_get_num_threads();
            GPU_ID = omp_get_thread_num(); // thread ID = GPU_ID
        }
        
        // Distribute work across GPUs
        if(GPU_ID == 0)
        {
            N_GPU = (ID_max-ID_min+1)/n_GPU+(ID_max-ID_min+1)%n_GPU;
            GPU_index_min = ID_min;
        }
        else
        {
            N_GPU = (ID_max-ID_min+1)/n_GPU;
            GPU_index_min = ID_min + (ID_max-ID_min+1)%n_GPU+N_GPU*GPU_ID;
        }
        
        // Allocate memory for local force calculations
        if(!(F_tmp = (REAL*)malloc(3 * N_GPU * sizeof(REAL))))
        {
            fprintf(stderr, "MPI task %i: failed to allocate memory for F_tmp (for CUDA force calculation).\n", rank);
            exit(-2);
        }
        
        for(i=0; i < N_GPU; i++)
        {
            for(j=0; j<3; j++)
                F_tmp[3*i + j] = 0.0f;
        }
        
        // Check GPU properties
        #pragma omp critical
        cudaDeviceGetAttribute(&mprocessors, cudaDevAttrMultiProcessorCount, GPU_ID);

        if(GPU_ID == 0)
        {
            printf("MPI task %i: GPU force calculation (z-periodic).\n Number of GPUs: %i\n Number of OMP threads: %i\n Number of threads per GPU: %i\n", 
                   rank, n_GPU, nthreads, 32*mprocessors*BLOCKSIZE);
        }
        
        // Select GPU device
        #pragma omp critical
        cudaStatus = cudaSetDevice(GPU_ID);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "MPI rank %i: GPU%i: cudaSetDevice failed! Do you have a CUDA-capable GPU installed?\n", rank, GPU_ID);
            ForceError = true;
            goto Error;
        }
        
        // Allocate GPU memory for all data
        cudaStatus = cudaMalloc((void**)&dev_xx, N * sizeof(REAL));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "MPI rank %i: GPU%i: xx cudaMalloc failed!\n", rank, GPU_ID);
            ForceError = true;
            goto Error;
        }
        
        cudaStatus = cudaMalloc((void**)&dev_xy, N * sizeof(REAL));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "MPI rank %i: GPU%i: xy cudaMalloc failed!\n", rank, GPU_ID);
            ForceError = true;
            goto Error;
        }
        
        cudaStatus = cudaMalloc((void**)&dev_xz, N * sizeof(REAL));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "MPI rank %i: GPU%i: xz cudaMalloc failed!\n", rank, GPU_ID);
            ForceError = true;
            goto Error;
        }
        
        cudaStatus = cudaMalloc((void**)&dev_M, N * sizeof(REAL));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "MPI rank %i: GPU%i: M cudaMalloc failed!\n", rank, GPU_ID);
            ForceError = true;
            goto Error;
        }
        
        // Allocate GPU buffers for the softening vector
        cudaStatus = cudaMalloc((void**)&dev_SOFT_LENGTH, N * sizeof(REAL));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "MPI rank %i: GPU%i: SOFT_LENGTH cudaMalloc failed!\n", rank, GPU_ID);
            ForceError = true;
            goto Error;
        }

		// Allocate GPU memory for the radial force lookup table
		cudaStatus = cudaMalloc((void**)&dev_RADIAL_FORCE_TABLE, RADIAL_FORCE_TABLE_SIZE * sizeof(REAL));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "MPI rank %i: GPU%i: SOFT_LENGTH cudaMalloc failed!\n", rank, GPU_ID);
			ForceError = true;
			goto Error;
		}
		#if !defined(PERIODIC_Z_NOLOOKUP)
			// Allocate GPU memory for S1R2 Ewald lookup table [2 components on a Nrho x Nz grid]
			cudaStatus = cudaMalloc((void**)&dev_S1R2_EwaldTable, 2 * Nrho_EWALD_FORCE_GRID * Nz_EWALD_FORCE_GRID * sizeof(REAL));
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "MPI rank %i: GPU%i: S1R2_EwaldTable cudaMalloc failed!\n", rank, GPU_ID);
				ForceError = true;
				goto Error;
			}
		#endif
        // Allocate GPU buffers for force vectors
        cudaStatus = cudaMalloc((void**)&dev_F, 3 * N_GPU * sizeof(REAL));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "MPI rank %i: GPU%i: F cudaMalloc failed!\n", rank, GPU_ID);
            ForceError = true;
            goto Error;
		}
        
        // Copy input vectors from host memory to GPU buffers.
        cudaStatus = cudaMemcpy(dev_xx, xx_tmp, N * sizeof(REAL), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "MPI rank %i: GPU%i: cudaMemcpy xx in failed!\n", rank, GPU_ID);
            ForceError = true;
            goto Error;
        }
        
        cudaStatus = cudaMemcpy(dev_xy, xy_tmp, N * sizeof(REAL), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "MPI rank %i: GPU%i: cudaMemcpy xy in failed!\n", rank, GPU_ID);
            ForceError = true;
            goto Error;
        }
        
        cudaStatus = cudaMemcpy(dev_xz, xz_tmp, N * sizeof(REAL), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "MPI rank %i: GPU%i: cudaMemcpy xz in failed!\n", rank, GPU_ID);
            ForceError = true;
            goto Error;
        }
        
        cudaStatus = cudaMemcpy(dev_M, M, N * sizeof(REAL), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "MPI rank %i: GPU%i: cudaMemcpy M in failed!\n", rank, GPU_ID);
            ForceError = true;
            goto Error;
        }
        
        cudaStatus = cudaMemcpy(dev_SOFT_LENGTH, SOFT_LENGTH, N * sizeof(REAL), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "MPI rank %i: GPU%i: cudaMemcpy SOFT_LENGTH in failed!\n", rank, GPU_ID);
            ForceError = true;
            goto Error;
        }

		// Copy radial force lookup table to GPU
		cudaStatus = cudaMemcpy(dev_RADIAL_FORCE_TABLE, RADIAL_FORCE_TABLE, RADIAL_FORCE_TABLE_SIZE * sizeof(REAL), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "MPI rank %i: GPU%i: cudaMemcpy SOFT_LENGTH in failed!\n", rank, GPU_ID);
			ForceError = true;
			goto Error;
		}
		#if !defined(PERIODIC_Z_NOLOOKUP)
			// Copy S1R2 Ewald lookup table to GPU [2 components on a Nrho x Nz grid]
			cudaStatus = cudaMemcpy(dev_S1R2_EwaldTable, S1R2_EWALD_FORCE_TABLE, 2 * Nrho_EWALD_FORCE_GRID * Nz_EWALD_FORCE_GRID * sizeof(REAL), cudaMemcpyHostToDevice);
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "MPI rank %i: GPU%i: cudaMemcpy S1R2_EwaldTable in failed!\n", rank, GPU_ID);
				ForceError = true;
				goto Error;
			}
		#endif
        
        cudaStatus = cudaMemcpy(dev_F, F_tmp, 3 * N_GPU * sizeof(REAL), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "MPI rank %i: GPU%i: cudaMemcpy F in failed!\n", rank, GPU_ID);
            ForceError = true;
            goto Error;
        }
        
        printf("MPI task %i: GPU%i: ID_min = %i\tID_max = %i\n", rank, GPU_ID, GPU_index_min, GPU_index_min+N_GPU-1);
        
        // Launch a kernel on the GPU
		#ifdef PERIODIC_Z_NOLOOKUP
        ForceKernel_periodic_z<<<32*mprocessors, BLOCKSIZE>>>(32*mprocessors*BLOCKSIZE, N, dev_xx, dev_xy, dev_xz, dev_F, 
                                                            IS_PERIODIC, dev_M, dev_SOFT_LENGTH, L, Rsim, 
                                                            mass_in_unit_sphere, dev_RADIAL_FORCE_TABLE, RADIAL_FORCE_TABLE_SIZE, DE, COSMOLOGY, COMOVING_INTEGRATION,
                                                            GPU_index_min, GPU_index_min+N_GPU-1);
        #else
		ForceKernel_periodic_z<<<32*mprocessors, BLOCKSIZE>>>(32*mprocessors*BLOCKSIZE, N, dev_xx, dev_xy, dev_xz, dev_F, 
															IS_PERIODIC, dev_M, dev_SOFT_LENGTH, L, Rsim, 
															mass_in_unit_sphere, dev_RADIAL_FORCE_TABLE, RADIAL_FORCE_TABLE_SIZE, dev_S1R2_EwaldTable, Nrho_EWALD_FORCE_GRID, Nz_EWALD_FORCE_GRID, DE, COSMOLOGY, COMOVING_INTEGRATION,
															GPU_index_min, GPU_index_min+N_GPU-1);
		#endif
        // Check for any errors launching the kernel
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "MPI rank %i: GPU%i: ForceKernel_periodic_z launch failed: %s\n", rank, GPU_ID, cudaGetErrorString(cudaStatus));
            ForceError = true;
            goto Error;
        }
        
        // cudaDeviceSynchronize waits for the kernel to finish, and returns
        // any errors encountered during the launch.
        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "MPI rank %i: GPU%i: cudaDeviceSynchronize returned error code %d after launching ForceKernel_periodic_z!\n", rank, GPU_ID, cudaStatus);
            ForceError = true;
            goto Error;
        }
        
        // Copy output vector from GPU buffer to host memory.
        cudaStatus = cudaMemcpy(F_tmp, dev_F, 3 * N_GPU * sizeof(REAL), cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "MPI rank %i: GPU%i: cudaMemcpy F out failed!\n", rank, GPU_ID);
            ForceError = true;
            goto Error;
        }
        
        // Copy forces back to the main array
        if(GPU_ID == 0)
        {
            for (i = 0; i < N_GPU; i++)
            {
                for (j = 0; j < 3; j++)
                {
                    F[3*i+j] = F_tmp[(3 * i) + j];
                }
            }
        }
        else
        {
            for (i = GPU_index_min; i < GPU_index_min + N_GPU; i++)
            {
                for (j = 0; j < 3; j++)
                {
                    F[3*(i-ID_min)+j] = F_tmp[3 * (i-GPU_index_min) + j];
                }
            }
        }
        
        free(F_tmp);
        
    Error:
        cudaFree(dev_xx);
        cudaFree(dev_xy);
        cudaFree(dev_xz);
        cudaFree(dev_M);
        cudaFree(dev_F);
        cudaFree(dev_SOFT_LENGTH);
		#ifdef PERIODIC_Z_NOLOOKUP
			cudaFree(dev_RADIAL_FORCE_TABLE);
		#else
			cudaFree(dev_S1R2_EwaldTable);
		#endif
        cudaDeviceReset();
    }
    
    free(xx_tmp);
    free(xy_tmp);
    free(xz_tmp);
    
    //timing
    omp_end_time = omp_get_wtime();
    //timing
    printf("Z-periodic force calculation finished on MPI task %i. Force calculation wall-clock time = %fs.\n", rank, omp_end_time-omp_start_time);
    return cudaStatus;
}
#endif
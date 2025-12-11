/********************************************************************************/
/*  StePS - STEreographically Projected cosmological Simulations				*/
/*    Copyright (C) 2017-2025 Gabor Racz										*/
/*																				*/
/*    This program is free software; you can redistribute it and/or modify		*/
/*    it under the terms of the GNU General Public License as published by		*/
/*    the Free Software Foundation; either version 2 of the License, or			*/
/*    (at your option) any later version.										*/
/*																				*/
/*    This program is distributed in the hope that it will be useful,			*/
/*    but WITHOUT ANY WARRANTY; without even the implied warranty of			*/
/*    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the				*/
/*    GNU General Public License for more details.								*/
/********************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "mpi.h"
#include "global_variables.h"


#ifdef USE_SINGLE_PRECISION
#define ERFC(x)  erfcf((x))
#define SIN(x)   sinf((x))
#define EXP(x)   expf((x))
#define SQRT(x)  sqrtf((x))
#else
#define ERFC(x)  erfc((x))
#define SIN(x)   sin((x))
#define EXP(x)   exp((x))
#define SQRT(x)  sqrt((x))
#endif

//Symmetry transformations of the T^3 manifold
// grid index mapping for a flat [Ngrid^3][3] force-table
static inline size_t lut_offset(int Ngrid, int ix, int iy, int iz, int comp)
{
    return ((size_t)((ix * Ngrid + iy) * Ngrid + iz) * 3u) + (size_t)comp;
}

// flip an index wrt the box center: x -> -x
static inline int flip_index(int idx, int Ngrid)
{
    return (Ngrid - 1) - idx;
}

// rotation descriptor: x' =  s[0]*component[perm[0]] ; same for y', z'
typedef struct { int perm[3]; int sign[3]; } Rot;

// Build the 24 proper rotations (Group O) or 48 full symmetries (Group Oh)
// To get Oh, pass an array of size 48 and set full_Oh = 1.
void build_Oh_rotations(Rot R[], int full_Oh)
{
    // 1. Define Permutations
    // Even perms (0,1,2 cyclic): Det = +1
    int perms_even[3][3] = { {0,1,2}, {1,2,0}, {2,0,1} };
    // Odd perms (swaps): Det = -1
    int perms_odd[3][3]  = { {0,2,1}, {2,1,0}, {1,0,2} };

    // 2. Define Sign Groups
    // Product = +1 (0 or 2 negatives)
    int signs_pos[4][3] = { {+1,+1,+1}, {+1,-1,-1}, {-1,+1,-1}, {-1,-1,+1} };
    // Product = -1 (1 or 3 negatives)
    int signs_neg[4][3] = { {-1,-1,-1}, {-1,+1,+1}, {+1,-1,+1}, {+1,+1,-1} };

    int n = 0;

    // --- Build Group O (24 Proper Rotations, Det=+1) ---
    
    // Case A: Even Perm (+1) * Pos Signs (+1) = +1
    for (int p = 0; p < 3; ++p) {
        for (int s = 0; s < 4; ++s) {
            for(int k=0; k<3; k++) {
                R[n].perm[k] = perms_even[p][k];
                R[n].sign[k] = signs_pos[s][k];
            }
            n++;
        }
    }

    // Case B: Odd Perm (-1) * Neg Signs (-1) = +1
    for (int p = 0; p < 3; ++p) {
        for (int s = 0; s < 4; ++s) {
            for(int k=0; k<3; k++) {
                R[n].perm[k] = perms_odd[p][k];
                R[n].sign[k] = signs_neg[s][k];
            }
            n++;
        }
    }

    // --- Extension to Group Oh (48 Elements) ---
    // Oh = O + (O * Inversion). Inversion flips all signs.
    if (full_Oh) {
        for (int i = 0; i < 24; ++i) {
            for(int k=0; k<3; k++) {
                R[n].perm[k] = R[i].perm[k];
                R[n].sign[k] = -R[i].sign[k]; // Apply inversion
            }
            n++;
        }
    }
}


//get the indices of the Ewald space up to radius R
int ewald_space(REAL R, int ewald_index[][4])
{
	int i,j,k;
	int l=-1;
	for(i=0;i<R;i++)
	{
		for(j=0;j<R;j++)
		{
			for(k=0; k<R; k++)
			{
				if((REAL)(i*i+j*j+k*k)< R*R)
				{
					l=l+1;
					ewald_index[l][0]=i;
					ewald_index[l][1]=j;
					ewald_index[l][2]=k;
					ewald_index[l][3] = i*i+j*j+k*k;
					if(i!=0)
					{
						l++;
						ewald_index[l][0]=-i;
						ewald_index[l][1]=j;
						ewald_index[l][2]=k;
						ewald_index[l][3] = i*i+j*j+k*k;

					}

					if(j!=0)
					{
						l++;
						ewald_index[l][0]=i;
						ewald_index[l][1]=-j;
						ewald_index[l][2]=k;
						ewald_index[l][3] = i*i+j*j+k*k;
					}

					if(k!=0)
					{
						l++;
						ewald_index[l][0]=i;
						ewald_index[l][1]=j;
						ewald_index[l][2]=-k;
						ewald_index[l][3] = i*i+j*j+k*k;
					}

					if(j!=0 && k!=0)
					{
						l++;
						ewald_index[l][0]=i;
						ewald_index[l][1]=-j;
						ewald_index[l][2]=-k;
						ewald_index[l][3] = i*i+j*j+k*k;
					}

					if(i!=0 && k!=0)
					{
						l++;
						ewald_index[l][0]=-i;
						ewald_index[l][1]=j;
						ewald_index[l][2]=-k;
						ewald_index[l][3] = i*i+j*j+k*k;
					}

					if(i!=0 && j!=0)
					{
						l++;
						ewald_index[l][0]=-i;
						ewald_index[l][1]=-j;
						ewald_index[l][2]=k;
						ewald_index[l][3] = i*i+j*j+k*k;
					}

					if(i!=0 && j!=0 && k!=0)
					{
						l++;
						ewald_index[l][0]=-i;
						ewald_index[l][1]=-j;
						ewald_index[l][2]=-k;
						ewald_index[l][3] = i*i+j*j+k*k;
					}


				}
			}
		}
	}
    return l;
}


//Ewald force (per unit G * m_i * m_j) at a single separation r
//Compute the Ewald force (real + reciprocal) and subtract nearest-image Newton to return the correction D(r).
static void ewald_force_correction( const REAL r[3], REAL L, REAL alpha, int realspace_el, int recspace_el, int real_idx[][4], int rec_idx[][4], REAL rel_cut, REAL rec_cut, REAL D[3])
{
    REAL F_real[3] = {0,0,0};
    REAL F_rec [3] = {0,0,0};

    const REAL V = L*L*L;
    const REAL two_alpha_over_sqrtpi = (REAL)(2.0) * alpha / (REAL)sqrt(pi);
    const REAL fourpi_over_V = (REAL)(4.0 * pi) / V;
    const REAL relcutL2 = (rel_cut * L) * (rel_cut * L);

    // real-space sum
    for (int m = 0; m <= realspace_el; ++m)
    {
        const int nx = real_idx[m][0];
        const int ny = real_idx[m][1];
        const int nz = real_idx[m][2];
        REAL Rv[3] = { r[0] + (REAL)nx * L,
                       r[1] + (REAL)ny * L,
                       r[2] + (REAL)nz * L };
        REAL R2 = Rv[0]*Rv[0] + Rv[1]*Rv[1] + Rv[2]*Rv[2];
        if ((R2 == (REAL)0) || (R2 > relcutL2)) continue; // skip self contribution
        REAL R  = SQRT(R2);
        REAL invR  = (REAL)1.0 / R;
        REAL invR3 = invR * invR * invR;

        REAL erfc_term = ERFC(alpha * R);
        REAL exp_term  = EXP(-(alpha*alpha) * R2);
        REAL bracket   = erfc_term + two_alpha_over_sqrtpi * R * exp_term;
        REAL coeff     = invR3 * bracket; // scalar multiply of unit-vector Rv/R

        F_real[0] += -Rv[0] * coeff;
        F_real[1] += -Rv[1] * coeff;
        F_real[2] += -Rv[2] * coeff;
    }

    // reciprocal-space sum
    for (int m = 0; m <= recspace_el; ++m)
    {
        const int h = rec_idx[m][0];
        const int k = rec_idx[m][1];
        const int l = rec_idx[m][2];
        if (h == 0 && k == 0 && l == 0) continue;

        REAL factor = (REAL)(2.0 * pi / L);
        REAL kv[3] = { factor * (REAL)h, factor * (REAL)k, factor * (REAL)l };
        REAL k2 = kv[0]*kv[0] + kv[1]*kv[1] + kv[2]*kv[2];
        if (k2 == (REAL)0 || k2 > rec_cut*rec_cut) continue;

        REAL damp   = EXP( - k2 / (REAL)(4.0 * alpha * alpha) );
        REAL coeff  = fourpi_over_V * damp / k2;
        REAL phase  = kv[0]*r[0] + kv[1]*r[1] + kv[2]*r[2];
        REAL s      = SIN(phase);

        F_rec[0] += -kv[0] * coeff * s;
        F_rec[1] += -kv[1] * coeff * s;
        F_rec[2] += -kv[2] * coeff * s;
    }

    // nearest-image Newtonian force (as in R^3 manifold)
    REAL r2 = r[0]*r[0] + r[1]*r[1] + r[2]*r[2];
    if (r2 > (REAL)0)
    {
        REAL rnorm = SQRT(r2);
        REAL invr3 = (REAL)1.0 / (rnorm*rnorm*rnorm);
        REAL F_newton[3] = { -r[0]*invr3, -r[1]*invr3, -r[2]*invr3 };

        // correction: D = (F_real + F_rec) - F_newton
        D[0] = (F_real[0] + F_rec[0]) - F_newton[0];
        D[1] = (F_real[1] + F_rec[1]) - F_newton[1];
        D[2] = (F_real[2] + F_rec[2]) - F_newton[2];
    }
    else
    {
        D[0] = D[1] = D[2] = (REAL)0;
    }
	return;
}

void calculate_t3_ewald_lookup_table(int Ngrid, REAL L, REAL alpha, int realspace_el, int recspace_el, int realspace_ewald_index[][4], int recspace_ewald_index[][4], REAL rel_cut, REAL rec_cut, REAL *T3_EWALD_FORCE_TABLE)
{
	//Making the lookup force table for the Ewald summation in T^3 periodic torus topology (https://articles.adsabs.harvard.edu/pdf/1991ApJS...75..231H)
	//since the periodic box have O_h symmetry, only 1/48 of the grid points are needed to calculate all the force corrections
	//inputs:
	// 	Ngrid: number of the grid points along one axis of the cubic grid
	//	L: box size
	//	alpha: Ewald damping parameter
	//	realspace_el: number of the real space images
	//	recspace_el: number of the reciprocal space images
	//	realspace_ewald_index: real space image indices
	//	recspace_ewald_index: reciprocal space image indices
	//output:
	//	T3_EWALD_FORCE_TABLE: the Ewald force correction lookup table length = Ngrid * Ngrid * Ngrid * 3
	int i,j,k;
    int chunk = Ngrid/16; //dynamic scheduling chunk size
	//clear table
    memset(T3_EWALD_FORCE_TABLE, 0, (size_t)Ngrid*Ngrid*Ngrid*3*sizeof(REAL));
	//build O symmetry rotations
	Rot R[24];
    build_Oh_rotations(R, 0); // only proper rotations (24 elements)
	//grid spacing; points are cell centers in [-L/2, L/2]
    const REAL grid_spacing = L / (REAL)Ngrid;
	//Iterate the “fundamental wedge”: x>=y>=z>=0. In index-space with cell centers, this corresponds to i>=j>=k, and i,j,k in [Ngrid/2 .. Ngrid-1].
	#pragma omp parallel default(shared)  private(i,j,k)
    {
        #pragma omp for schedule(dynamic,chunk)
        for (i = Ngrid/2; i < Ngrid; ++i)
        {
            for (j = Ngrid/2; j <= i; ++j)
            {
                for (k = Ngrid/2; k <= j; ++k)
                {
                    // physical coordinates of the separation (nearest-image)
                    REAL r[3] = {
                        ( (REAL)i + (REAL)0.5 ) * grid_spacing - L/(REAL)2.0,
                        ( (REAL)j + (REAL)0.5 ) * grid_spacing - L/(REAL)2.0,
                        ( (REAL)k + (REAL)0.5 ) * grid_spacing - L/(REAL)2.0
                    };

                    // correction D at the wedge point
                    REAL D_w[3];
                    ewald_force_correction(r, L, alpha, realspace_el, recspace_el, realspace_ewald_index, recspace_ewald_index, rel_cut, rec_cut, D_w);

                    // store at the wedge point
                    for (int c = 0; c < 3; ++c)
                    {
                        T3_EWALD_FORCE_TABLE[lut_offset(Ngrid, i, j, k, c)] = D_w[c];
                    }

                    // fill the whole grid by O_h symmetry
                    int src_idx[3] = { i, j, k };

                    // 24 proper rotations
                    for (int q = 0; q < 24; ++q)
                    {
                        int dst_idx[3];
                        REAL D_rot[3];

                        // apply permutation and sign to indices and force components
                        for (int d = 0; d < 3; ++d)
                        {
                            int src_comp = R[q].perm[d];
                            int idx_val  = src_idx[src_comp];
                            // sign flip -> index flip around center
                            if (R[q].sign[d] < 0) idx_val = flip_index(idx_val, Ngrid);
                            dst_idx[d] = idx_val;

                            D_rot[d] = R[q].sign[d] * D_w[src_comp];
                        }

                        // store rotated
                        for (int c = 0; c < 3; ++c)
                        {
                            T3_EWALD_FORCE_TABLE[lut_offset(Ngrid, dst_idx[0], dst_idx[1], dst_idx[2], c)] = D_rot[c];
                        }

                        // inversion (multiply position by -1), force changes sign: F(-r) = -F(r) )
                        int inv_idx[3] = {
                            flip_index(dst_idx[0], Ngrid),
                            flip_index(dst_idx[1], Ngrid),
                            flip_index(dst_idx[2], Ngrid)
                        };
                        REAL D_inv[3] = { -D_rot[0], -D_rot[1], -D_rot[2] };

                        for (int c = 0; c < 3; ++c)
                        {
                            T3_EWALD_FORCE_TABLE[lut_offset(Ngrid, inv_idx[0], inv_idx[1], inv_idx[2], c)] = D_inv[c];
                        }
                    }
                }
            }
        }
    }
	return;
}

#ifndef USE_CUDA
//Helper functions for the interpolation in the T^3 Ewald force correction table (CPU version)
// CUDA version of interpolation function are in forces_cuda.cu



// positive modulo for periodic wrap
static inline int imodp(int i, int n)
{
    int r = i % n;
    return (r < 0) ? (r + n) : r;
}

// Map a physical coordinate r in [-L/2,L/2] (or any real number) to the index of the *left* neighbor cell-center and the fractional offset fx in [0,1), given that centers lie at (i+0.5)*a - L/2.
static inline void map_to_centered_grid(REAL r, REAL L, int Ngrid, int *i0, REAL *fx)
{
    REAL grid_spacing  = L / (REAL)Ngrid;                 /* grid spacing */
    REAL u  = (r + L*(REAL)0.5) / grid_spacing - (REAL)0.5; /* continuous index @ centers */
    REAL uf = floor(u);
    *i0 = imodp((int)uf, Ngrid);
    *fx = (REAL)(u - uf); /* in [0,1) */
}

#if EWALD_INTERPOLATION_ORDER != 4
static void get_lagrange_weights(REAL fx, int stencil, REAL *w) 
{
    //Compute 1D Lagrange weights for a uniform grid.
    //Note that this is an unoptimized general implementation for testing purposes.
    //inputs:
    // * fx     :  Fractional position in [0, 1) between center i and i+1
    // * stencil:  Number of points (must be even: 2=linear, 4=cubic, etc.)
    // * w      :  Output array of size [stencil]
    //          Logic for stencil: 
    //              -> If stencil=2, we use points relative to i0: {0, 1}
    //              -> If stencil=4, we use points relative to i0: {-1, 0, 1, 2}
    //              -> Generally, start_offset = -(stencil/2 - 1)

    // The relative integer coordinates of the stencil points
    // e.g. for N=4: -1, 0, 1, 2
    int start = -(stencil / 2 - 1);

    for (int j = 0; j < stencil; j++)
    {
        w[j] = 1.0;
        REAL xj = (REAL)(start + j); // coordinate of the j-th support point
        // Lagrange basis polynomial product
        for (int k = 0; k < stencil; k++)
        {
            if (k == j) continue;
            REAL xk = (REAL)(start + k);
            w[j] *= (fx - xk) / (xj - xk);
        }
    }
}
#else
void get_cubic_weights(REAL t, REAL w[4])
{
    // Same as above but hardcoded for cubic (stencil=4) for efficiency
    REAL t2 = t*t;
    REAL t3 = t*t2;
    w[0] = -0.5*t3 + t2 - 0.5*t;        // Index i0-1
    w[1] =  1.5*t3 - 2.5*t2 + 1.0;      // Index i0
    w[2] = -1.5*t3 + 2.0*t2 + 0.5*t;    // Index i0+1
    w[3] =  0.5*t3 - 0.5*t2;            // Index i0+2
}
#endif

void ewald_interpolate_D(int Ngrid, REAL L, const REAL *table, const REAL dx, const REAL dy, const REAL dz, REAL D[3], int order)
{
    //Ewald interpolation in T^3 periodic torus topology using Lagrange interpolation of given order (even numbers only, 2=cic(tri-linear), 4=tri-cubic, etc.)
    //Inputs:
    //    * Ngrid   :  Grid dimension
    //    * L       :  Box size
    //    * table   :  Force table
    //    * dx,dy,dz:  Relative particle position
    //    * D       :  Output force correction
    //    * order   :  Stencil size (Use 4 for Cubic/standard, 2 for Linear)

    // Safety check: order must be even and positive
    if (order < 2 || order % 2 != 0) order = 2; // fallback to linear

    // Map to grid coordinates
    int ix0, iy0, iz0;
    REAL fx, fy, fz;
    map_to_centered_grid(dx, L, Ngrid, &ix0, &fx);
    map_to_centered_grid(dy, L, Ngrid, &iy0, &fy);
    map_to_centered_grid(dz, L, Ngrid, &iz0, &fz);

    // Compute weights for each dimension
    #if EWALD_INTERPOLATION_ORDER != 4
    // Max order 8 is usually sufficient
    REAL wx[8], wy[8], wz[8];
    int safe_order = (order > 8) ? 8 : order;
    
    //this can be optimized further by precomputing weights for common fx, fy, fz values
    get_lagrange_weights(fx, safe_order, wx);
    get_lagrange_weights(fy, safe_order, wy);
    get_lagrange_weights(fz, safe_order, wz);
    #else
    // Hardcoded cubic weights for efficiency
    REAL wx[4], wy[4], wz[4];
    get_cubic_weights(fx, wx);
    get_cubic_weights(fy, wy);
    get_cubic_weights(fz, wz);
    int safe_order = 4;
    #endif

    // Perform Tensor Product Summation
    REAL sum[3] = {0.0, 0.0, 0.0};
    
    // Offset to the starting neighbor index
    int offset = -(safe_order / 2 - 1);

    for (int i = 0; i < safe_order; i++) {
        int ix = imodp(ix0 + offset + i, Ngrid);
        REAL w_x = wx[i];

        for (int j = 0; j < safe_order; j++) {
            int iy = imodp(iy0 + offset + j, Ngrid);
            REAL w_xy = w_x * wy[j];

            for (int k = 0; k < safe_order; k++) {
                int iz = imodp(iz0 + offset + k, Ngrid);
                REAL w_xyz = w_xy * wz[k];

                // Fetch vector from table
                size_t idx = lut_offset(Ngrid, ix, iy, iz, 0);
                
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
#endif

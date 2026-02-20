/********************************************************************************/
/*  StePS - STEreographically Projected cosmological Simulations				*/
/*    Copyright (C) 2017-2026 Gabor Racz										*/
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
#ifdef PERIODIC_Z
#include <cmath>
#include <algorithm>
#include <cstring>
#define sqrtpi 1.7724538509055160272981674833411 //sqrt(pi)
#endif

#include "mpi.h"
#include "global_variables.h"


#ifdef USE_SINGLE_PRECISION
#define ERFC(x)  erfcf((x))
#define SIN(x)   sinf((x))
#define COS(x)  cosf(x)
#define EXP(x)   expf((x))
#define SQRT(x)  sqrtf((x))
#else
#define ERFC(x)  erfc((x))
#define SIN(x)   sin((x))
#define COS(x)  cos((x))
#define EXP(x)   exp((x))
#define SQRT(x)  sqrt((x))
#endif

#ifdef PERIODIC
// grid index mapping for a flat [Ngrid^3][3] force-table
static inline size_t lut_offset(int Ngrid, int ix, int iy, int iz, int comp)
{
    return ((size_t)((ix * Ngrid + iy) * Ngrid + iz) * 3u) + (size_t)comp;
}

//Symmetry transformations of the T^3 manifold
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


//get the indices of the T^3 Ewald space up to radius R
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
//end of PERIODIC (T^3) case

#elif defined(PERIODIC_Z)
// S^1 x R^2 Ewald functions

// References for S^1 x R^2 Ewald method:
//   Tornberg, "The Ewald sums for singly, doubly and triply periodic electrostatic systems"
//     (arXiv:1404.3534; Springer 2015)
//      --> 1P (=S^1xR^2) derivation and zero-mode term
//   Shamshirgar & Tornberg, "The Spectral Ewald method for singly periodic domains" (JCP 2017)
//      --> Another nice derivation of the 1P Ewald sums, and error estimates

// Fast approximation of the scaled complementary error function erfcx(x) = exp(x^2) * erfc(x)
// for x >= 0. Accuracy is approx 1e-15.
static inline double fast_erfcx(double x)
{
    if (x < 0)
        return 2.0 * exp(x * x) - fast_erfcx(-x);
    if (x == 0)
        return 1.0;
    if (x > 1e10)
        return 0.564189583547756286 / x; // Asymptotic 1/(x*sqrt(pi))

    double t = 1.0 / (1.0 + 0.3275911 * x);
    // Standard Handbook of Mathematical Functions (Abramowitz & Stegun) 
    // formula 7.1.26 (accuracy ~1.5e-7) - For 1e-15, use minimax:
    
    // Using a 5th-order approximation for high precision:
    double a1 =  0.254829592;
    double a2 = -0.284496736;
    double a3 =  1.421413741;
    double a4 = -1.453152027;
    double a5 =  1.061405429;
    
    // This is the common "Abramowitz & Stegun" high-precision fit
    return t * (a1 + t * (a2 + t * (a3 + t * (a4 + t * a5))));
}

// Helper to compute exp(x) * erfc(y) safely
inline double exp_erfc_product(double km, double rho, double alpha, bool is_term_p)
{
    // Safe computation of exp(+/-km*rho) * erfc( km/(2*alpha) +/- alpha*rho )
    // For large arguments, exp(km*rho) and erfc(arg) can overflow.
    // To avoid this, we use the limit of the scaled complementary error function erfcx for large arguments.
    // Identity: exp(km*rho) * erfc(arg) = exp(km*rho - arg*arg) * erfcx(arg)
    // For arg = km/(2*alpha) + alpha*rho, the exponent simplifies to -(km^2/(4*alpha^2) + alpha^2*rho^2)
    double kmptwoalpha = km / (2.0 * alpha);
    double ar = alpha * rho;
    double arg = is_term_p ? (kmptwoalpha + ar) : (kmptwoalpha - ar);
    
    // This exponent is mathematically: (km*rho - arg*arg) for termP
    // and (-km*rho - arg*arg) for termM. Both simplify to this:
    double common_exponent = -(kmptwoalpha * kmptwoalpha + ar * ar);

    if (arg > 5.0)
    {
        return exp(common_exponent) * fast_erfcx(arg);
    }
    else if (arg < -5.0)
    {
        // This only happens for termM when alpha*rho is large.
        // Identity: exp(-k*rho) * erfc(arg) = 2*exp(-k*rho) - exp(common_exponent)*erfcx(-arg)
        return 2.0 * exp(-km * rho) - exp(common_exponent) * fast_erfcx(-arg);
    }
    else
    {
        // Small argument: standard calculation is safe from overflow/NaN
        double shift = is_term_p ? (km * rho) : (-km * rho);
        return exp(shift) * erfc(arg);
    }
}


// Wrap dz to nearest image in [-Lz/2, Lz/2]
static inline REAL wrap_dz(REAL dz, REAL Lz)
{
    REAL half = (REAL)0.5 * Lz;
    // map dz into (-Lz/2, Lz/2]
    if (dz >  half)
        dz -= Lz * std::floor((dz + half)/Lz);
    if (dz <= -half)
        dz -= Lz * std::floor((dz - half)/Lz);
    return dz;
}


static inline size_t lut2D_offset(int Nrho, int Nz, int ir, int iz, int comp)
{
    return ((size_t)ir * (size_t)Nz + (size_t)iz) * 2u + (size_t)comp;
}


// S1xR^2 Ewald force (per unit G*m_i*m_j)
void S1R2ewald_force_pair(double dx, double dy, double dz, double Lz, double alpha, int nmax, int mmax, double &Fx, double &Fy, double &Fz)
{
    // nearest-image in z
    dz = wrap_dz(dz, Lz);
    double rho2 = dx*dx + dy*dy; // radial distance squared
    double dzn, r2, r, invr3;
    
    #if !defined(PERIODIC_Z_RSPACELOOKUP)
    // Ewald sum as in Tornberg 2015 / Shamshirgar & Tornberg 2017
    double rho = sqrt(rho2); // radial distance
    double Fxr=0, Fyr=0, Fzr=0; // real-space force components
    double ar, coeff;
    // real-space: sum over images along z
    for (int n = -nmax; n <= nmax; ++n)
    {
        dzn = dz + (double)n * Lz;
        r2 = rho2 + dzn*dzn;
        if (r2 < 1e-18) continue; 
        r = sqrt(r2);
        invr3 = 1.0 / (r2 * r);
        ar = alpha * r;
        coeff = (erfc(ar) + (2.0/sqrtpi) * ar * exp(-ar*ar)) * invr3;
        
        Fxr -= dx * coeff;
        Fyr -= dy * coeff;
        Fzr -= dzn * coeff;
    }

    // k-space: modes m != 0 and zero-mode (m=0) parts
    // Note: The k-space sum uses the singly‑periodic kernel obtained by performing the two-dimensional Fourier integrals analytically with the Gaussian filter;
    //       this yields the familiar erfc⁡-based expression (no numerical K_nu​ evaluation), and a zero‑mode derivative that is just 1/\rho−2\alpha^2\rho*e^{−\alpha^2\rho^2} (no need to evaluate E1​ explicitly)
    //       see derivations in Tornberg 2015 (Sec. 7–9)
    double Frk = 0; 
    double Fzk = 0;
    double invL = 1.0 / Lz;
    double km, termP, termM, B, dBdrho, cos_kz, sin_kz;
    for (int m = 1; m <= mmax; ++m)
    {
        km = (2.0 * pi * m) * invL;

        /*//This part can be numerically unstable for large arguments; use exp_erfc_product helper
        double s = km; // km is always positive here
        double a = s / (2.0 * alpha);
        double am = a - alpha * rho;
        double ap = a + alpha * rho;
        double esr = exp(s * rho);
        double emsr = exp(-s * rho);
        
        // Potential Kernel B
        double erfc_p = erfc(ap);
        double erfc_m = erfc(am);
        double B = esr * erfc_p + emsr * erfc_m;

        // Radial derivative dB/drho: 
        // Note: The Gaussian terms from d(erfc)/drho cancel out analytically!
        double dBdrho = s * (esr * erfc_p - emsr * erfc_m);*/

        // Use numerically stable exp_erfc_product
        termP = exp_erfc_product(km, rho, alpha, true); // exp(s*rho) * erfc(ap)
        termM = exp_erfc_product(km, rho, alpha, false); // exp(-s*rho) * erfc(am)
        // Potential Kernel B
        B = termP + termM;
        // Radial derivative dB/drho: 
        // Note: The Gaussian terms from d(erfc)/drho cancel out analytically!
        dBdrho = km * (termP - termM);

        cos_kz = cos(km * dz);
        sin_kz = sin(km * dz);

        // Prefactor for S1 is 2/L. 
        // For Gravity, we want the gradient to be attractive:
        // Fr = + dPhi/drho (since B is positive and decreasing)
        // Fz = + dPhi/dz
        Frk += (2.0 * invL) * cos_kz * dBdrho;
        Fzk -= (2.0 * invL) * km * sin_kz * B;
    }

    // zero-mode (m=0) contributes only to F_rho via the 2D free-space (logarithmic) term.
    // The derivative of the regularized 1D potential -1/L * [E1(a^2rho^2) + ln(a^2rho^2)]
    // gives an attractive radial force: F_rho = -2 / (L * rho) * (1 - exp(-alpha^2 * rho^2))
    if (rho > 1e-12)
    {
        double Fr0 = -(2.0 * invL / rho) * (1.0 - exp(-alpha*alpha*rho2));
        Frk += Fr0;
    }

    // Map radial k-space force to Cartesian
    double Fxk = (rho > 0) ? (Frk * dx / rho) : 0;
    double Fyk = (rho > 0) ? (Frk * dy / rho) : 0;

    Fx = Fxr + Fxk;
    Fy = Fyr + Fyk;
    Fz = Fzr + Fzk;
    #else
    //Direct real space summation with no k-space
    Fx = 0.0;
    Fy = 0.0;
    Fz = 0.0;
    if(mmax !=0)
    {
        printf("Warning: mmax is ignored in direct S1R2 periodic real-space summation!\n");
    }
    if(alpha !=0.0)
    {
        printf("Warning: alpha is ignored in direct S1R2 periodic real-space summation!\n");
    }
    //for numerical stability, we sum from the larger contribution to the smaller one
    //n=0 (nearest image term)
    r2 = rho2 + dz*dz;
    if (r2 >= 1e-18)
    {
        r = sqrt(r2);
        invr3 = 1.0 / (r2 * r);
        Fx -= dx * invr3;
        Fy -= dy * invr3;
        Fz -= dz * invr3;       
    }
    //n!=0 terms
    for (int n = 1; n <= nmax; ++n)
    {
        for(int sign = -1; sign <= 1; sign += 2)
        {
            dzn = dz + (double)(sign * n) * Lz;
            r2 = rho2 + dzn*dzn;
            if (r2 < 1e-18) continue; 
            r = sqrt(r2);
            invr3 = 1.0 / (r2 * r);
            
            Fx -= dx * invr3;
            Fy -= dy * invr3;
            Fz -= dzn * invr3;
        }
    }
    #endif
}

void calculate_S1R2ewald_correction_table(int Nrho, int Nz, REAL rho_max, REAL Lz, REAL alpha, int nmax, int mmax, REAL*& S1R2_EWALD_FORCE_TABLE)
{
    double drho = (double) rho_max / (double)std::max(1, Nrho-1);
    double dz   = (double) Lz / (double)Nz;

    for (int ir = 0; ir < Nrho; ++ir)
    {
        double rho = (double)ir * drho;
        for (int iz = 0; iz < Nz; ++iz)
        {
            // cell centers in z: [-Lz/2, Lz/2)
            double z = ((double)iz + (double)0.5) * dz - (double)0.5 * Lz;

            // Compute full periodic force via 1P Ewald
            double dx = rho;  // choose x=rho, y=0 -- SO(2) symmetry
            double dy = 0;
            double dz_sep = z;
            double Fx, Fy, Fz;
            S1R2ewald_force_pair(dx, dy, dz_sep, (double) Lz, (double) alpha, nmax, mmax, Fx, Fy, Fz);

            // Convert to (Frho, Fz): since dy=0 and dx=rho, Frho = Fx
            double Frho_periodic = Fx;
            double Fz_periodic   = Fz;

            // Nearest-image Newtonian (per unit G*m_i*m_j):
            double dz_wrapped = wrap_dz(z, Lz);
            double r2 = rho*rho + dz_wrapped*dz_wrapped;
            double Frho_newt = 0, Fz_newt = 0;
            if (r2 > (double)0)
            {
                double r  = SQRT(r2);
                double invr3 = (double)1/(r*r*r);
                Frho_newt = - (rho) * invr3;
                Fz_newt   = - (dz_wrapped) * invr3;
            }

            // Correction = periodic - Newton
            double Drho = Frho_periodic - Frho_newt;
            double Dz   = Fz_periodic   - Fz_newt;

            S1R2_EWALD_FORCE_TABLE[lut2D_offset(Nrho,Nz,ir,iz,0)] = (REAL) Drho;
            S1R2_EWALD_FORCE_TABLE[lut2D_offset(Nrho,Nz,ir,iz,1)] = (REAL) Dz;
        }
    }
}

#ifndef USE_CUDA
//Helper functions for the interpolation in the S^1xR^2 Ewald force correction table (CPU version)

void ngp_interp_S1R2ewald_D(const REAL* __restrict T, int Nrho, int Nz, REAL rho_max, REAL Lz, REAL rho, REAL z, REAL &D_rho, REAL &D_z)
{
    // Clamp rho into [0, rho_max]
    if (rho < (REAL)0) rho = (REAL)0;
    if (rho > rho_max) rho = rho_max;

    const REAL half = (REAL)0.5 * Lz;

    // Grid spacings
    const REAL drho = rho_max / (REAL)std::max(1, Nrho - 1);
    const REAL dz   = Lz      / (REAL)Nz;

    // Continuous indices using same cell-center convention as CIC/TSC
    const REAL ur = (drho > (REAL)0) ? (rho / drho) : (REAL)0;
    const REAL uz = (z + half) / dz - (REAL)0.5;

    // NGP = nearest integer index
    int ir = (int)std::floor(ur + (REAL)0.5);
    int iz = (int)std::floor(uz + (REAL)0.5);

    // rho index clamped
    if (ir < 0)          ir = 0;
    if (ir > Nrho - 1)   ir = Nrho - 1;

    // z index wrapped periodically
    {
        int r = iz % Nz;
        iz = (r < 0) ? r + Nz : r;
    }

    // Compute base offset only once: ((ir*Nz) * 2)
    const size_t base = (size_t)ir * (size_t)Nz * 2u + (size_t)iz * 2u;

    // Load the two components directly
    D_rho = T[base + 0];
    D_z   = T[base + 1];
}

void cic_interp_S1R2ewald_D(const REAL* __restrict T, int Nrho, int Nz, REAL rho_max, REAL Lz, REAL rho, REAL z, REAL &D_rho, REAL &D_z)
{
    // Bilinear CIC interpolation on (rho,z) grid with z periodic.
    //  inputs: 
    //          * rho in [0,rho_max],
    //          * z arbitrary (wrapped inside).
    //  outputs: (D_rho, D_z) per unit G*m_i*m_j.
    // clamp rho to [0, rho_max]
    rho = std::max((REAL)0, std::min(rho, rho_max));
    // no need to wrap z to [-Lz/2, Lz/2), since we assume an already wrapped input in z
    REAL half = (REAL)0.5 * Lz;
    //z = z - Lz * std::floor( (z + half)/Lz );

    REAL drho = rho_max / (REAL)std::max(1, Nrho-1);
    REAL dz   = Lz / (REAL)Nz;

    // continuous indices
    REAL ur = (drho > 0) ? (rho / drho) : 0;
    int  ir0 = (int)std::floor(ur);
    REAL fr  = ur - (REAL)ir0;
    if(ir0 < 0)
    {
        ir0 = 0;
        fr = 0;
    }
    if (ir0 > Nrho-2)
    {
        ir0 = std::max(0, Nrho-2);
        fr = (REAL)1;
    }
    int ir1 = ir0 + 1;

    REAL uz = (z + half) / dz - (REAL)0.5;
    int  iz0 = (int)std::floor(uz);
    REAL fz  = uz - (REAL)iz0;
    // wrap indices in z
    auto mod = [&](int i)->int { int r=i%Nz; return r<0 ? r+Nz : r; };
    iz0 = mod(iz0);
    int iz1 = mod(iz0 + 1);

    // weights
    REAL w00 = (REAL)1 - fr;
    w00 *= ((REAL)1 - fz);
    REAL w10 = fr;
    w10 *= ((REAL)1 - fz);
    REAL w01 = (REAL)1 - fr;
    w01 *= fz;
    REAL w11 = fr * fz;

    auto get = [&](int ir, int iz, int c) -> REAL {return T[lut2D_offset(Nrho, Nz, ir, iz, c)];};

    D_rho = w00*get(ir0,iz0,0) + w10*get(ir1,iz0,0) + w01*get(ir0,iz1,0) + w11*get(ir1,iz1,0);
    D_z   = w00*get(ir0,iz0,1) + w10*get(ir1,iz0,1) + w01*get(ir0,iz1,1) + w11*get(ir1,iz1,1);
}


void tsc_interp_S1R2ewald_D(const REAL* __restrict T, int Nrho, int Nz, REAL rho_max, REAL Lz, REAL rho, REAL z, REAL &D_rho, REAL &D_z)
{
    // Triangular Shaped Cloud (TSC) interpolation on (rho,z) grid with z periodic.
    //  inputs: 
    //          * rho in [0,rho_max],
    //          * z arbitrary (wrapped inside).
    //  outputs: (D_rho, D_z) per unit G*m_i*m_j.

    // Clamp rho
    if (rho < (REAL)0) rho = (REAL)0;
    if (rho > rho_max) rho = rho_max;

    const REAL half = (REAL)0.5 * Lz;
    const REAL drho = rho_max / (REAL)std::max(1, Nrho - 1);
    const REAL dz   = Lz / (REAL)Nz;

    // Continuous indices in rho and z
    const REAL ur = (drho > (REAL)0) ? (rho / drho) : (REAL)0;
    const REAL uz = (z + half) / dz - (REAL)0.5; // centers, same convention as CIC

    // Centers (nearest grid points)
    const int jr = (int)std::floor(ur + (REAL)0.5);
    const int jz = (int)std::floor(uz + (REAL)0.5);

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

    auto wrap = [&](int i)->int { int r = i % Nz; return (r < 0) ? (r + Nz) : r; };
    const int iz0 = wrap(jz - 1);
    const int iz1 = wrap(jz);
    const int iz2 = wrap(jz + 1);

    
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

    D_rho = (REAL)0.0;
    D_z   = (REAL)0.0;

    // Accumulate D_rho and D_z over 3x3 stencil
    // Unrolling the loops for performance
    // I used the definition of lut2D_offset to compute the offsets directly. If that changes, this code must be updated accordingly.
    // Row iz0
    {
        const REAL wz = wz_row0;
        const REAL wr0 = wrm * wz, wr1 = wrc * wz, wr2 = wrp * wz;

        const size_t off00 = base_ir0 + col_iz0;
        const size_t off10 = base_ir1 + col_iz0;
        const size_t off20 = base_ir2 + col_iz0;

        D_rho += wr0 * T[off00 + 0] + wr1 * T[off10 + 0] + wr2 * T[off20 + 0];
        D_z   += wr0 * T[off00 + 1] + wr1 * T[off10 + 1] + wr2 * T[off20 + 1];
    }

    // Row iz1
    {
        const REAL wz = wz_row1;
        const REAL wr0 = wrm * wz, wr1 = wrc * wz, wr2 = wrp * wz;

        const size_t off01 = base_ir0 + col_iz1;
        const size_t off11 = base_ir1 + col_iz1;
        const size_t off21 = base_ir2 + col_iz1;

        D_rho += wr0 * T[off01 + 0] + wr1 * T[off11 + 0] + wr2 * T[off21 + 0];
        D_z   += wr0 * T[off01 + 1] + wr1 * T[off11 + 1] + wr2 * T[off21 + 1];
    }

    // Row iz2
    {
        const REAL wz = wz_row2;
        const REAL wr0 = wrm * wz, wr1 = wrc * wz, wr2 = wrp * wz;

        const size_t off02 = base_ir0 + col_iz2;
        const size_t off12 = base_ir1 + col_iz2;
        const size_t off22 = base_ir2 + col_iz2;

        D_rho += wr0 * T[off02 + 0] + wr1 * T[off12 + 0] + wr2 * T[off22 + 0];
        D_z   += wr0 * T[off02 + 1] + wr1 * T[off12 + 1] + wr2 * T[off22 + 1];
    }
}

// Use the table to get force correction vector (per unit G*m_i*m_j)
void ewald_interpolate_D( const REAL* T, int Nrho, int Nz, REAL rho_max, REAL Lz, REAL dx, REAL dy, REAL dz, REAL &Dx, REAL &Dy, REAL &Dz)
{
    REAL rho = SQRT(dx*dx + dy*dy); //radial distance

    REAL Drho;
    #if EWALD_INTERPOLATION_ORDER == 0
        ngp_interp_S1R2ewald_D(T, Nrho, Nz, rho_max, Lz, rho, dz, Drho, Dz);
    #elif EWALD_INTERPOLATION_ORDER == 2
        cic_interp_S1R2ewald_D(T, Nrho, Nz, rho_max, Lz, rho, dz, Drho, Dz);
    #elif EWALD_INTERPOLATION_ORDER == 4
        tsc_interp_S1R2ewald_D(T, Nrho, Nz, rho_max, Lz, rho, dz, Drho, Dz);
    #else
        //This should not happen, fallback to TSC
        tsc_interp_S1R2ewald_D(T, Nrho, Nz, rho_max, Lz, rho, dz, Drho, Dz);
    #endif

    // cylindrical -> Cartesian
    REAL ex = (rho > 0) ? dx/rho : (REAL)0;
    REAL ey = (rho > 0) ? dy/rho : (REAL)0;
    Dx = Drho * ex;
    Dy = Drho * ey;
    // Dz already axial
}
#endif

#endif // PERIODIC_Z
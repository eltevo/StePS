#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <time.h>
#include "global_variables.h"

#ifdef USE_SINGLE_PRECISION
	typedef float REAL;
#else
	typedef double REAL;
#endif

extern int H[2202][4];
extern int e[2202][4];
extern REAL w[3];
extern REAL G;
extern int N, hl, el;


int ewald_space(REAL R, int ewald_index[2102][4]);

void recalculate_softening();

void recalculate_softening()
{
	beta = ParticleRadi;
	if(COSMOLOGY ==1)
	{
	if(COMOVING_INTEGRATION == 1)
	//{
	//	printf("Calculating smoothing length...\n"); 
	//	beta = ParticleRadi/(REAL)(a);
	//	rho_part = M_min/(4.0*pi*pow(beta, 3.0) / 3.0);
	//	printf("The new smoothing length:\tbeta = %lf\n", beta);
	//}
	//else
	//{
		beta = ParticleRadi;
		rho_part = M_min/(4.0*pi*pow(beta, 3.0) / 3.0);
	//}
	}
	/*SOFT_CONST[0] = 32.0/pow(2.0*beta, 6);
        SOFT_CONST[1] = -38.4/pow(2.0*beta, 5);
        SOFT_CONST[2] = 32.0/(3.0*pow(2.0*beta, 3));

        SOFT_CONST[3] = -32.0/(3.0*pow(2*beta, 6));
        SOFT_CONST[4] = 38.4/pow(2.0*beta, 5);
        SOFT_CONST[5] = -48.0/pow(2.0*beta, 4);
        SOFT_CONST[6] = 64.0/(3.0*pow(2.0*beta, 3));
        SOFT_CONST[7] = -1.0/15.0;*/
}

void forces(REAL**x, REAL**F, int ID_min, int ID_max) //Force calculation
{
//timing
REAL start_time = (REAL) clock () / (REAL) CLOCKS_PER_SEC;
REAL omp_start_time = omp_get_wtime();
//timing
REAL Fx_tmp, Fy_tmp, Fz_tmp, beta_priv;
REAL SOFT_CONST[5];

int i, j, k, chunk;
for(i=ID_min; i<ID_max+1; i++)
{
        for(k=0; k<3; k++)
        {
                F[i][k] = 0;
        }
}
        REAL r, betai, dx, dy, dz, wij;
	REAL const_beta = 3.0/rho_part/(4.0*pi);
	chunk = (ID_max-ID_min)/omp_get_max_threads();
	if(chunk < 1)
	{
		chunk = 1;
	}
	#pragma omp parallel default(shared)  private(dx, dy, dz, r, wij, j, i, Fx_tmp, Fy_tmp, Fz_tmp, SOFT_CONST, beta_priv, betai)
	{
	#pragma omp for schedule(dynamic,chunk)
        for(i=ID_min; i<ID_max+1; i++)
        {
		betai = cbrt(M[i]*const_beta);
                for(j=0; j<N; j++)
                {
			beta_priv = (betai+cbrt(M[j]*const_beta))/2;
			//printf("beta_priv[%i][%i] = %lf\n", i, j, (double)beta_priv);
			//calculating particle distances
                        dx=x[j][0]-x[i][0];
			dy=x[j][1]-x[i][1];
			dz=x[j][2]-x[i][2];
			//in this function we use only the nearest image
			if(IS_PERIODIC==1)
			{
				if(fabs(dx)>0.5*L)
                        		dx = dx-L*dx/fabs(dx);
                        	if(fabs(dy)>0.5*L)
                       			dy = dy-L*dy/fabs(dy);
                        	if(fabs(dz)>0.5*L)
                        		dz = dz-L*dz/fabs(dz);
			}

                        r = sqrt(pow(dx, 2)+pow(dy, 2)+pow(dz, 2));
			wij = 0;
			if(r <= beta_priv)
                        {
				SOFT_CONST[0] = 32.0/pow(2.0*beta_priv, 6);
				SOFT_CONST[1] = -38.4/pow(2.0*beta_priv, 5);
				SOFT_CONST[2] = 32.0/(3.0*pow(2.0*beta_priv, 3));

				wij = M[j]*(SOFT_CONST[0]*pow(r, 3)+SOFT_CONST[1]*pow(r, 2)+SOFT_CONST[2]);
                        }
			if(r > beta_priv && r <= 2*beta_priv)
			{
				SOFT_CONST[0] = -32.0/(3.0*pow(2.0*beta_priv, 6));
				SOFT_CONST[1] = 38.4/pow(2.0*beta_priv, 5);
				SOFT_CONST[2] = -48.0/pow(2.0*beta_priv, 4);
				SOFT_CONST[3] = 64.0/(3.0*pow(2.0*beta_priv, 3));
				SOFT_CONST[4] = -1.0/15.0;
				wij = M[j]*(SOFT_CONST[0]*pow(r, 3)+SOFT_CONST[1]*pow(r, 2)+SOFT_CONST[2]*r+SOFT_CONST[3]+SOFT_CONST[4]/pow(r, 3));
			}
			if(r > 2*beta_priv)
			{
				wij = M[j]/(pow(r, 3));
			}
			Fx_tmp = wij*(dx);
			Fy_tmp = wij*(dy);
			Fz_tmp = wij*(dz);
			//#pragma omp atomic
                        //F[j][0] -= Fx_tmp;
			#pragma omp atomic
                        F[i][0] += Fx_tmp;
			//#pragma omp atomic
                        //F[j][1] -= Fy_tmp;
			#pragma omp atomic
                        F[i][1] += Fy_tmp;
			//#pragma omp atomic
                        //F[j][2] -= Fz_tmp;
			#pragma omp atomic
                        F[i][2] += Fz_tmp;


                }
		if(COSMOLOGY == 1 && IS_PERIODIC == 0 && COMOVING_INTEGRATION == 1)//Adding the external force from the outside of the simulation volume, if we run non-periodic comoving cosmological simulation
		{
			F[i][0] += mass_in_unit_sphere * x[i][0];
			F[i][1] += mass_in_unit_sphere * x[i][1];
			F[i][2] += mass_in_unit_sphere * x[i][2];
		}
	}

        }
//timing
REAL end_time = (REAL) clock () / (REAL) CLOCKS_PER_SEC;
REAL omp_end_time = omp_get_wtime();
//timing
printf("Force calculation finished.\n");
printf("Force CPU time = %fs\n", end_time-start_time);
printf("Force RUN time = %fs\n", omp_end_time-omp_start_time);
return;
}

void forces_periodic(REAL**x, REAL**F, int ID_min, int ID_max) //force calculation with multiple images
{
//timing
REAL start_time = (REAL) clock () / (REAL) CLOCKS_PER_SEC;
REAL omp_start_time = omp_get_wtime();
//timing
REAL Fx_tmp, Fy_tmp, Fz_tmp, beta_priv;
REAL SOFT_CONST[5];

	int i, j, k, m, chunk;
	for(i=ID_min; i<ID_max+1; i++)
	{
		for(k=0; k<3; k++)
		{
			F[i][k] = 0;
		}
	}
	REAL r, betai, dx, dy, dz, wij;
	REAL const_beta = 3.0/rho_part/(4.0*pi);
	chunk = (ID_max-ID_min)/(omp_get_max_threads());
	if(chunk < 1)
	{
		chunk = 1;
	}
	#pragma omp parallel default(shared)  private(dx, dy, dz, r, wij, i, j, m, Fx_tmp, Fy_tmp, Fz_tmp, SOFT_CONST, beta_priv, betai)
	{
	#pragma omp for schedule(dynamic,chunk)
	for(i=ID_min; i<ID_max+1; i++)
	{
		betai = cbrt(M[i]*const_beta);
		for(j=0; j<N; j++)
		{
			Fx_tmp = 0;
			Fy_tmp = 0;
			Fz_tmp = 0;
			beta_priv = (betai+cbrt(M[j]*const_beta))/2;
			//calculating particle distances inside the simulation box
			dx=x[j][0]-x[i][0];
			dy=x[j][1]-x[i][1];
			dz=x[j][2]-x[i][2];
			//In here we use multiple images
			for(m=0;m<el;m++) 
			{
				r = sqrt(pow((dx-((REAL) e[m][0])*L), 2)+pow((dy-((REAL) e[m][1])*L), 2)+pow((dz-((REAL) e[m][2])*L), 2));
				wij = 0;
				if(r <= beta_priv)
				{
					SOFT_CONST[0] = 32.0/pow(2.0*beta_priv, 6);
        	                        SOFT_CONST[1] = -38.4/pow(2.0*beta_priv, 5);
	                                SOFT_CONST[2] = 32.0/(3.0*pow(2.0*beta_priv, 3));
					wij = M[j]*(SOFT_CONST[0]*pow(r, 3)+SOFT_CONST[1]*pow(r, 2)+SOFT_CONST[2]);
				}
				if(r > beta_priv && r <= 2*beta_priv)
				{
					SOFT_CONST[0] = -32.0/(3.0*pow(2*beta_priv, 6));
                        	        SOFT_CONST[1] = 38.4/pow(2.0*beta_priv, 5);
                	                SOFT_CONST[2] = -48.0/pow(2.0*beta_priv, 4);
        	                        SOFT_CONST[3] = 64.0/(3.0*pow(2.0*beta_priv, 3));
	                                SOFT_CONST[4] = -1.0/15.0;
					wij = M[j]*(SOFT_CONST[0]*pow(r, 3)+SOFT_CONST[1]*pow(r, 2)+SOFT_CONST[2]*r+SOFT_CONST[3]+SOFT_CONST[4]/pow(r, 3));
				}
				if(r > 2*beta_priv && r < 2.6*L)
				{
					wij = M[j]/(pow(r, 3));
				}
				
				if(wij != 0)
				{
					Fx_tmp += wij*(dx-((REAL) e[m][0])*L);
					Fy_tmp += wij*(dy-((REAL) e[m][1])*L);
					Fz_tmp += wij*(dz-((REAL) e[m][2])*L);
				}
			}
			#pragma omp atomic
                        F[i][0] += Fx_tmp;
			#pragma omp atomic
                        F[i][1] += Fy_tmp;
			#pragma omp atomic
                        F[i][2] += Fz_tmp;
		}
		}
	}
//timing
REAL end_time = (REAL) clock () / (REAL) CLOCKS_PER_SEC;
REAL omp_end_time = omp_get_wtime();
//timing
printf("Force calculation finished.\n");
printf("CPU time = %fs\n", end_time-start_time);
printf("RUN time = %fs\n", omp_end_time-omp_start_time);
return;
}



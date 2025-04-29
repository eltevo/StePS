/********************************************************************************/
/*  StePS - STEreographically Projected cosmological Simulations                */
/*    Copyright (C) 2017-2025 Gabor Racz, Balazs Pal                            */
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


extern int e[2202][4];
extern REAL w[3];
extern int N, el;


int ewald_space(REAL R, int ewald_index[2102][4]);

void recalculate_softening();

void recalculate_softening()
{
	beta = ParticleRadi;
	if(COSMOLOGY ==1)
	{
		rho_part = M_min/(4.0*pi*pow(beta, 3.0) / 3.0);
	}
}

#if !defined(PERIODIC) && !defined(PERIODIC_Z)
void forces(REAL* x, REAL* F, int ID_min, int ID_max) //Force calculation
{
//timing
double omp_start_time = omp_get_wtime();
//timing
REAL Fx_tmp, Fy_tmp, Fz_tmp, beta_priv, beta_privp2;
REAL SOFT_CONST[5];

REAL DE = (REAL) H0*H0*Omega_lambda;

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
	#pragma omp parallel default(shared)  private(dx, dy, dz, r, wij, j, i, Fx_tmp, Fy_tmp, Fz_tmp, SOFT_CONST, beta_priv, beta_privp2)
	{
	#pragma omp for schedule(dynamic,chunk)
	for(i=ID_min; i<ID_max+1; i++)
	{
		for(j=0; j<N; j++)
		{
			beta_priv = (SOFT_LENGTH[i]+SOFT_LENGTH[j]);
			beta_privp2 = beta_priv*0.5;
			//calculating particle distances
            dx=x[3*j]-x[3*i];
			dy=x[3*j+1]-x[3*i+1];
			dz=x[3*j+2]-x[3*i+2];
			r = sqrt(pow(dx, 2)+pow(dy, 2)+pow(dz, 2));
			wij = 0;
			if(r >= beta_priv)
			{
				wij = M[j]/(pow(r, 3));
			}
			else if(r > beta_privp2 && r < beta_priv)
                        {
				SOFT_CONST[0] = -32.0/(3.0*pow(beta_priv, 6));
				SOFT_CONST[1] = 38.4/pow(beta_priv, 5);
				SOFT_CONST[2] = -48.0/pow(beta_priv, 4);
				SOFT_CONST[3] = 64.0/(3.0*pow(beta_priv, 3));
				SOFT_CONST[4] = -1.0/15.0;
				wij = M[j]*(SOFT_CONST[0]*pow(r, 3)+SOFT_CONST[1]*pow(r, 2)+SOFT_CONST[2]*r+SOFT_CONST[3]+SOFT_CONST[4]/pow(r, 3));
			}
			else
			{
				SOFT_CONST[0] = 32.0/pow(beta_priv, 6);
				SOFT_CONST[1] = -38.4/pow(beta_priv, 5);
				SOFT_CONST[2] = 32.0/(3.0*pow(beta_priv, 3));

				wij = M[j]*(SOFT_CONST[0]*pow(r, 3)+SOFT_CONST[1]*pow(r, 2)+SOFT_CONST[2]);
			}
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
printf("Force calculation finished on MPI task %i. Force calculation wall-clock time = %fs.\n", rank, omp_end_time-omp_start_time);
return;
}
#endif

#ifdef PERIODIC
void forces_periodic(REAL*x, REAL*F, int ID_min, int ID_max) //force calculation with multiple images
{
//timing
double omp_start_time = omp_get_wtime();
//timing
REAL Fx_tmp, Fy_tmp, Fz_tmp, beta_priv, beta_privp2;
REAL SOFT_CONST[5];
	int i, j, k, m, chunk;
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
	#pragma omp parallel default(shared)  private(dx, dy, dz, r, wij, i, j, m, Fx_tmp, Fy_tmp, Fz_tmp, SOFT_CONST, beta_priv, beta_privp2)
	{
	#pragma omp for schedule(dynamic,chunk)
	for(i=ID_min; i<ID_max+1; i++)
	{
		for(j=0; j<N; j++)
		{
			Fx_tmp = 0;
			Fy_tmp = 0;
			Fz_tmp = 0;
			beta_priv = (SOFT_LENGTH[i]+SOFT_LENGTH[j]);
			beta_privp2 = beta_priv*0.5; 
			//calculating particle distances inside the simulation box
			dx=x[3*j]-x[3*i];
			dy=x[3*j+1]-x[3*i+1];
			dz=x[3*j+2]-x[3*i+2];
			//In here we use multiple images
			for(m=0;m<el;m++)
			{
				r = sqrt(pow((dx-((REAL) e[m][0])*L), 2)+pow((dy-((REAL) e[m][1])*L), 2)+pow((dz-((REAL) e[m][2])*L), 2));
				wij = 0;
				if(r >= beta_priv && r < 2.6*L)
				{
					wij = M[j]/(pow(r, 3));
				}
				else if(r > beta_privp2 && r < beta_priv)
				{
					SOFT_CONST[0] = -32.0/(3.0*pow(beta_priv, 6));
					SOFT_CONST[1] = 38.4/pow(beta_priv, 5);
					SOFT_CONST[2] = -48.0/pow(beta_priv, 4);
					SOFT_CONST[3] = 64.0/(3.0*pow(beta_priv, 3));
					SOFT_CONST[4] = -1.0/15.0;
					wij = M[j]*(SOFT_CONST[0]*pow(r, 3)+SOFT_CONST[1]*pow(r, 2)+SOFT_CONST[2]*r+SOFT_CONST[3]+SOFT_CONST[4]/pow(r, 3));
				}
				else if(r <= beta_privp2)
				{
					SOFT_CONST[0] = 32.0/pow(beta_priv, 6);
        	                        SOFT_CONST[1] = -38.4/pow(beta_priv, 5);
	                                SOFT_CONST[2] = 32.0/(3.0*pow(beta_priv, 3));
					wij = M[j]*(SOFT_CONST[0]*pow(r, 3)+SOFT_CONST[1]*pow(r, 2)+SOFT_CONST[2]);
				}
				if(wij != 0)
				{
					Fx_tmp += wij*(dx-((REAL) e[m][0])*L);
					Fy_tmp += wij*(dy-((REAL) e[m][1])*L);
					Fz_tmp += wij*(dz-((REAL) e[m][2])*L);
				}
			}
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
		#pragma omp parallel default(shared)  private(dx, dy, dz, r, wij, j, i, Fx_tmp, Fy_tmp, Fz_tmp, SOFT_CONST, beta_priv, beta_privp2)
        	{
        	#pragma omp for schedule(dynamic,chunk)
	        	for(i=ID_min; i<ID_max+1; i++)
			{
				for(j=0; j<N; j++)
				{
					beta_priv = (SOFT_LENGTH[i]+SOFT_LENGTH[j]);
					beta_privp2 = beta_priv*0.5;
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
					wij = 0;
					if(r >= beta_priv)
					{
						wij = M[j]/(pow(r, 3));
					}
					else if(r > beta_privp2 && r < beta_priv)
					{
						SOFT_CONST[0] = -32.0/(3.0*pow(beta_priv, 6));
						SOFT_CONST[1] = 38.4/pow(beta_priv, 5);
						SOFT_CONST[2] = -48.0/pow(beta_priv, 4);
						SOFT_CONST[3] = 64.0/(3.0*pow(beta_priv, 3));
						SOFT_CONST[4] = -1.0/15.0;
						wij = M[j]*(SOFT_CONST[0]*pow(r, 3)+SOFT_CONST[1]*pow(r, 2)+SOFT_CONST[2]*r+SOFT_CONST[3]+SOFT_CONST[4]/pow(r, 3));
					}
					else
					{
						SOFT_CONST[0] = 32.0/pow(beta_priv, 6);
						SOFT_CONST[1] = -38.4/pow(beta_priv, 5);
						SOFT_CONST[2] = 32.0/(3.0*pow(beta_priv, 3));
						wij = M[j]*(SOFT_CONST[0]*pow(r, 3)+SOFT_CONST[1]*pow(r, 2)+SOFT_CONST[2]);
					}
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
printf("Force calculation finished on MPI task %i. Force calculation wall-clock time = %fs.\n", rank, omp_end_time-omp_start_time);
return;
}
#endif

#ifdef PERIODIC_Z
//force calculation with multiple images only in the z direction
void forces_periodic_z(REAL* x, REAL* F, int ID_min, int ID_max)
{
    //timing
    double omp_start_time = omp_get_wtime();
    //timing
    REAL Fx_tmp, Fy_tmp, Fz_tmp, beta_priv, beta_privp2;
    REAL SOFT_CONST[5];
    
    int i, j, k, m, chunk;
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
    if(IS_PERIODIC>=2) {
        #pragma omp parallel default(shared)  private(dx, dy, dz, r, wij, i, j, m, Fx_tmp, Fy_tmp, Fz_tmp, SOFT_CONST, beta_priv, beta_privp2)
            #pragma omp for schedule(dynamic,chunk)
                for(i=ID_min; i<ID_max+1; i++) {
                    for(j=0; j<N; j++) {
                        Fx_tmp = 0;
                        Fy_tmp = 0;
                        Fz_tmp = 0;
                        beta_priv = (SOFT_LENGTH[i] + SOFT_LENGTH[j]);
                        beta_privp2 = beta_priv*0.5; 
                        //calculating particle distances inside the simulation box
                        dx = x[3*j] - x[3*i];
                        dy = x[3*j+1] - x[3*i+1];
                        dz = x[3*j+2] - x[3*i+2];
                        //In here we use multiple images, but only in the z direction
                        for(m=0; m<el; m++)
                        {
                            r = sqrt(pow(dx, 2) + pow(dy, 2) + pow((dz-((REAL) e[m][2])*L), 2));
                            wij = 0;
                            if(r >= beta_priv && r < 2.6*L)
                            {
                                wij = M[j]/(pow(r, 3));
                            }
                            else if(r > beta_privp2 && r < beta_priv)
                            {
                                SOFT_CONST[0] = -32.0/(3.0*pow(beta_priv, 6));
                                SOFT_CONST[1] = 38.4/pow(beta_priv, 5);
                                SOFT_CONST[2] = -48.0/pow(beta_priv, 4);
                                SOFT_CONST[3] = 64.0/(3.0*pow(beta_priv, 3));
                                SOFT_CONST[4] = -1.0/15.0;
                                wij = M[j]*(SOFT_CONST[0]*pow(r, 3) + SOFT_CONST[1]*pow(r, 2) + SOFT_CONST[2]*r + SOFT_CONST[3] + SOFT_CONST[4]/pow(r, 3));
                            }
                            else if(r <= beta_privp2)
                            {
                                SOFT_CONST[0] = 32.0/pow(beta_priv, 6);
                                SOFT_CONST[1] = -38.4/pow(beta_priv, 5);
                                SOFT_CONST[2] = 32.0/(3.0*pow(beta_priv, 3));
                                wij = M[j]*(SOFT_CONST[0]*pow(r, 3) + SOFT_CONST[1]*pow(r, 2) + SOFT_CONST[2]);
                            }
                            if(wij != 0)
                            {
                                Fx_tmp += wij*(dx);
                                Fy_tmp += wij*(dy);
                                Fz_tmp += wij*(dz-((REAL) e[m][2])*L);
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
                        F[3*(i-ID_min)] += mass_in_unit_sphere * x[3*i];
                        F[3*(i-ID_min)+1] += mass_in_unit_sphere * x[3*i+1];
                    }
                    /*else if(COSMOLOGY == 1 && COMOVING_INTEGRATION == 0)
                    {
                        F[3*(i-ID_min)] +=  DE * x[3*i];
                        F[3*(i-ID_min)+1] += DE * x[3*i+1];
                    }*/ //non-comoving integration is not implemented for periodic_z (yet?)
                }
    }
    else {
        #pragma omp parallel default(shared) private(dx, dy, dz, r, wij, j, i, Fx_tmp, Fy_tmp, Fz_tmp, SOFT_CONST, beta_priv, beta_privp2)
            #pragma omp for schedule(dynamic,chunk)
                for(i=ID_min; i<ID_max+1; i++) {
                    for(j=0; j<N; j++)
                    {
                        beta_priv = (SOFT_LENGTH[i] + SOFT_LENGTH[j]);
                        beta_privp2 = beta_priv*0.5;
                        //calculating particle distances
                        dx = x[3*j] - x[3*i];
                        dy = x[3*j+1] - x[3*i+1];
                        dz = x[3*j+2] - x[3*i+2];
                        //in this case we use only the nearest image
                        if(fabs(dz)>0.5*L) { dz = dz-L*dz/fabs(dz); }
                        r = sqrt(pow(dx, 2) + pow(dy, 2) + pow(dz, 2));
                        wij = 0;
                        if(r >= beta_priv)
                        {
                            wij = M[j]/(pow(r, 3));
                        }
                        else if(r > beta_privp2 && r < beta_priv)
                        {
                            SOFT_CONST[0] = -32.0/(3.0*pow(beta_priv, 6));
                            SOFT_CONST[1] = 38.4/pow(beta_priv, 5);
                            SOFT_CONST[2] = -48.0/pow(beta_priv, 4);
                            SOFT_CONST[3] = 64.0/(3.0*pow(beta_priv, 3));
                            SOFT_CONST[4] = -1.0/15.0;
                            wij = M[j]*(SOFT_CONST[0]*pow(r, 3) + SOFT_CONST[1]*pow(r, 2) + SOFT_CONST[2]*r + SOFT_CONST[3] + SOFT_CONST[4]/pow(r, 3));
                        }
                        else
                        {
                            SOFT_CONST[0] = 32.0/pow(beta_priv, 6);
                            SOFT_CONST[1] = -38.4/pow(beta_priv, 5);
                            SOFT_CONST[2] = 32.0/(3.0*pow(beta_priv, 3));
                            wij = M[j]*(SOFT_CONST[0]*pow(r, 3) + SOFT_CONST[1]*pow(r, 2) + SOFT_CONST[2]);
                        }
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
                        F[3*(i-ID_min)] += mass_in_unit_sphere * x[3*i];
                        F[3*(i-ID_min)+1] += mass_in_unit_sphere * x[3*i+1];
                    }
                    /*else if(COSMOLOGY == 1 && COMOVING_INTEGRATION == 0)
                    {
                        F[3*(i-ID_min)] +=  DE * x[3*i];
                        F[3*(i-ID_min)+1] += DE * x[3*i+1];
                    }*/ //non-comoving integration is not implemented for periodic_z (yet?)
                }
    }
    //timing
    double omp_end_time = omp_get_wtime();
    //timing
    printf("Force calculation finished on MPI task %i. Force calculation wall-clock time = %fs.\n", rank, omp_end_time-omp_start_time);
    return;
}
#endif
/********************************************************************************/
/*  StePS - STEreographically Projected cosmological Simulations                */
/*    Copyright (C) 2017-2018 Gabor Racz                                        */
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

//This file describe one simulation timestep
//We use KDK integrator for the N-body simulation

void forces(REAL* x, REAL* F, int ID_min, int ID_max);
void forces_periodic(REAL* x, REAL*F, int ID_min, int ID_max);

double calculate_init_h()
{
	//calculating the initial timesep length
	int i,k;
	errmax = 0;
	REAL ACCELERATION[3];
	#ifdef PERIODIC
	for(i=0;i<N;i++)
	{
		for(k=0;k<3;k++)
		{
			//If we are using periodic boundary conditions, the code move every "out-of-box" particle inside the box
			if(x[3*i+k]<0)
			{
				x[3*i+k] = x[3*i+k] + L;
			}
			if(x[3*i+k]>=L)
			{
				x[3*i+k] = x[3*i+k] - L;
			}
		}
	}
	#endif
	REAL const_beta = 3.0/rho_part/(4.0*pi);
	for(i=0; i<N; i++)
	{
		for(k=0;k<3;k++)
		{
			//calculating the maximal acceleration for the initial timestep
			ACCELERATION[k] = (G*F[3*i+k]*(REAL)(pow(a, -3.0)) - 2.0*(REAL)(Hubble_param)*v[3*i+k]);
		}
                err = sqrt(ACCELERATION[0]*ACCELERATION[0] + ACCELERATION[1]*ACCELERATION[1] + ACCELERATION[2]*ACCELERATION[2])/cbrt(M[i]*const_beta);
		if(err>errmax)
		{
			errmax = err;
		}
	}
	if (pow(2*ACC_PARAM/errmax, 0.5)*UNIT_T >= 1.0)
		printf("Initial timestep length calculated. h_start=%fGy\n",  (double) pow(2*ACC_PARAM/errmax, 0.5)*UNIT_T);
	else
		printf("Initial timestep length calculated. h_start=%fMy\n",  (double) pow(2*ACC_PARAM/errmax, 0.5)*UNIT_T*1000.0);
	return (double) pow(2*ACC_PARAM/errmax, 0.5);

}

void step(REAL* x, REAL* v, REAL* F)
{
	//Timing
	double step_omp_start_time = omp_get_wtime();
	//Timing
	int i, j, k;
	REAL disp;
	#ifdef GLASS_MAKING
	REAL dmax,dmean;
	dmax = 0.0;
	dmean = 0.0;
	#endif
	#ifdef USE_CUDA
		omp_set_dynamic(0);		// Explicitly disable dynamic teams
		omp_set_num_threads(n_GPU);	// Use n_GPU threads
	#endif
	REAL ACCELERATION[3];
	errmax = 0;
	if(rank == 0)
	{
		printf("KDK Leapfrog integration...\n");
		for(i=0; i<N; i++)
		{
			for(k=0; k<3; k++)
			{
				ACCELERATION[k] = (G*F[3*i+k]*(REAL)(pow(a, -3.0)) - 2.0*(REAL)(Hubble_param)*v[3*i+k]);
				//'Kick' operation (h/2)
				disp = ACCELERATION[k]*(REAL)(h/2.0);
				v[3*i+k] += disp;
				//'Drift' operation (h)
				disp = v[3*i+k]*(REAL)(h);
				x[3*i+k] = x[3*i+k] + v[3*i+k]*(REAL)(h);
			}
			#ifdef GLASS_MAKING
				disp = sqrt(pow(v[3*i]*(REAL)(h), 2) + pow(v[3*i+1]*(REAL)(h), 2) + pow(v[3*i+2]*(REAL)(h), 2));
				dmean +=  disp;
				if(dmax <= disp)
					dmax = disp;
			#endif
			}
		//If we are using periodic boundary conditions, the code move every "out-of-box" particle inside the box
		#ifdef PERIODIC
		for(i=0; i<N; i++)
		{
			for(k=0;k<3;k++)
			{
				if(x[3*i+k]<0)
					x[3*i+k] = x[3*i+k] + L;
				if(x[3*i+k]>=L)
					x[3*i+k] = x[3*i+k] - L;
			}
		}
		#endif
	}
	//Bcasting the particle coordinates
#ifdef USE_SINGLE_PRECISION
	MPI_Bcast(x,3*N,MPI_FLOAT,0,MPI_COMM_WORLD);
#else
	MPI_Bcast(x,3*N,MPI_DOUBLE,0,MPI_COMM_WORLD);
#endif
	//Force calculation
	if(rank == 0)
		printf("Calculating Forces...\n");
	#ifndef PERIODIC
		forces(x, F, ID_MPI_min, ID_MPI_max);
	#else
		forces_periodic(x, F, ID_MPI_min, ID_MPI_max);
	#endif
	//if the force calculation is finished, the calculated forces should be collected into the rank=0 thread`s F matrix
	if(rank !=0)
	{
#ifdef USE_SINGLE_PRECISION
		MPI_Send(F, 3*N_mpi_thread, MPI_FLOAT, 0, rank, MPI_COMM_WORLD);
#else
		MPI_Send(F, 3*N_mpi_thread, MPI_DOUBLE, 0, rank, MPI_COMM_WORLD);
#endif
	}
	else
	{
		if(numtasks > 1)
		{
			for(i=1; i<numtasks;i++)
			{
				BUFFER_start_ID = i*(N/numtasks)+(N%numtasks);
#ifdef USE_SINGLE_PRECISION
				MPI_Recv(F_buffer, 3*(N/numtasks), MPI_FLOAT, i, i, MPI_COMM_WORLD, &Stat);
#else
				MPI_Recv(F_buffer, 3*(N/numtasks), MPI_DOUBLE, i, i, MPI_COMM_WORLD, &Stat);
#endif
				for(j=0; j<(N/numtasks); j++)
				{
					F[3*(BUFFER_start_ID+j)] = F_buffer[3*j];
					F[3*(BUFFER_start_ID+j)+1] = F_buffer[3*j+1];
					F[3*(BUFFER_start_ID+j)+2] = F_buffer[3*j+2];
				}
			}
		}
	}
	//Stepping in scale factor and Hubble-parameter
	//if COSMOLOGY == 1, than we step in scalefactor, using the specified cosmological model
	if(COSMOLOGY == 1)
	{
		if(COMOVING_INTEGRATION == 1)
		{
			a = friedman_solver_step(a, h, Omega_lambda, Omega_r, Omega_m, Omega_k, H0);
			recalculate_softening();
			a_tmp = a;
			Hubble_param = H0*sqrt(Omega_m*pow(a, -3)+Omega_r*pow(a, -4)+Omega_lambda+Omega_k*pow(a, -2));
			Decel_param = CALCULATE_decel_param(a); //Deceleration parameter
			Omega_m_eff = Omega_m*pow(a, -3)*pow(H0/Hubble_param, 2);
		}
		else
		{
			a_tmp = T;
		}
	}
	else
	{
		//For non-cosmological simulation
		a_tmp = T;
	}
	if(rank == 0)
	{
		REAL const_beta = 3.0/rho_part/(4.0*pi);
		for(i=0; i<N; i++)
		{
			for(k=0; k<3; k++)
			{
				ACCELERATION[k] = (G*F[3*i+k]*(REAL)(pow(a, -3.0)) - 2.0*(REAL)(Hubble_param)*v[3*i+k]);
				//'Kick' operation (h/2)
				v[3*i+k] += ACCELERATION[k]*(REAL)(h/2.0);
			}
			err = sqrt(ACCELERATION[0]*ACCELERATION[0] + ACCELERATION[1]*ACCELERATION[1] + ACCELERATION[2]*ACCELERATION[2])/cbrt(M[i]*const_beta);
			if(err>errmax)
			{
				errmax = err;
			}
		}
		printf("KDK Leapfrog integration...done.\n");
		#ifdef GLASS_MAKING
		dmean = dmean/((REAL) N);
		if(dmax>1.0)
			printf("Glass making: A_max = %e\tdisp-mean=%fMpc\tdisp-maximum = %fMpc\n", errmax/pow(a, 2.0),dmean,dmax);
		else
			printf("Glass making: A_max = %e\tdisp-mean=%fkpc\tdisp-maximum = %fkpc\n", errmax/pow(a, 2.0),dmean*1000,dmax*1000);
		#endif
	}
	//Timing
	double step_omp_end_time = omp_get_wtime();
	//Timing
	if(rank == 0)
	{
		printf("Timestep wall-clock time = %fs\n", step_omp_end_time-step_omp_start_time);
	}

return;
}

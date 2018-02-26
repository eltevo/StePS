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

//This file describe one simulation timestep
//We use KDK integrator for the N-body simulation 

void forces(REAL** x, REAL** F, int ID_min, int ID_max);
void forces_periodic(REAL**x, REAL**F, int ID_min, int ID_max);

double calculate_init_h()
{
	//calculating the initial timesep length
	int i,k;
	errmax = 0;
	REAL ACCELERATION[3];
	if(IS_PERIODIC == 0)
	{
		for(i=0;i<N;i++)
		{
			for(k=0;k<3;k++)
			{
				//calculating the maximal acceleration for the initial timestep
				ACCELERATION[k] = (F[i][k]*(REAL)(pow(a, -3.0)) - 2.0*(REAL)(Hubble_param)*x[i][k+3]);
				err = sqrt(ACCELERATION[0]*ACCELERATION[0] + ACCELERATION[1]*ACCELERATION[1] + ACCELERATION[2]*ACCELERATION[2]);
				if(err>errmax)
				{
					errmax = err;
				}
			}
		}
	}
	else
	{
		for(i=0;i<N;i++)
		{
			for(k=0;k<3;k++)
			{
				//If we are using periodic boundary conditions, the code move every "out-of-box" particle inside the box
				if(x[i][k]<0)
				{
				x[i][k] = x[i][k] + L;
				}
				if(x[i][k]>=L)
				{
				x[i][k] = x[i][k] - L;
				}
				//calculating the maximal acceleration for the initial timestep
				ACCELERATION[k] = (F[i][k]*(REAL)(pow(a, -3.0)) - 2.0*(REAL)(Hubble_param)*x[i][k+3]);
				err = sqrt(ACCELERATION[0]*ACCELERATION[0] + ACCELERATION[1]*ACCELERATION[1] + ACCELERATION[2]*ACCELERATION[2]);
				if(err>errmax)
				{
					errmax = err;
				}
			}
		}
	}
	printf("Initial timestep length calculated.\nerrmax=%f\nh=%f\n", errmax, (double) pow(2*mean_err*beta/errmax, 0.5));
	return (double) pow(2*mean_err*beta/errmax, 0.5);
	
}

void step(REAL** x, REAL** F)
{
	//Timing
	REAL step_start_time = (REAL) clock () / (REAL) CLOCKS_PER_SEC;
	REAL step_omp_start_time = omp_get_wtime();
	//Timing
	int i, k;
	REAL ACCELERATION[3];
	errmax = 0;
	printf("KDK Leapfrog integration...\n");
	for(i=0; i<N; i++)
	{
		for(k=0; k<3; k++)
		{
			ACCELERATION[k] = (F[i][k]*(REAL)(pow(a, -3.0)) - 2.0*(REAL)(Hubble_param)*x[i][k+3]);
			x[i][k+3] = x[i][k+3] + ACCELERATION[k]*(REAL)(h/2.0);
			x[i][k] = x[i][k] + x[i][k+3]*(REAL)(h);
		}
	}
	//If we are using periodic boundary conditions, the code move every "out-of-box" particle inside the box
	if(IS_PERIODIC != 0)
	{
		for(i=0; i<N; i++)
		{
		for(k=0;k<3;k++)
		{
		if(x[i][k]<0)
		{
		x[i][k] = x[i][k] + L;
		}
		if(x[i][k]>=L)
		{
		x[i][k] = x[i][k] - L;
		}
		}
		}
	}
	//Force calculation
	printf("Calculating Forces...\n");
	if(IS_PERIODIC < 2)
	{
		forces(x, F, 0, N-1);
	}
	if(IS_PERIODIC == 2)
	{
		forces_periodic(x, F, 0, N-1);
	}
	//Stepping in scale factor and Hubble-parameter
	//if COSMOLOGY == 1, than we step in scalefactor, using the specified cosmological model
	if(COSMOLOGY == 1)
	{
		if(COMOVING_INTEGRATION == 1)
		{
			a_prev2 = a_prev1;
			a_prev1 = a;
			a = friedman_solver_step(a, h, Omega_lambda, Omega_r, Omega_m, Omega_k, H0);
			recalculate_softening();
			a_tmp = a;
			Hubble_param = H0*sqrt(Omega_m*pow(a, -3)+Omega_r*pow(a, -4)+Omega_lambda+Omega_k*pow(a, -2));
			Decel_param = CALCULATE_decel_param(a, a_prev1, a_prev2, h, h_prev); //Deceleration parameter
			Omega_m_eff = Omega_m*pow(a, -3)*pow(H0/Hubble_param, 2);
		}
		else
		{
			a_tmp = T;
		}
	}
	else
	{
		//For non-cosmological simulation, taking into account the T_max
		a_tmp = T;
	}
	for(i=0; i<N; i++)
	{
		for(k=0; k<3; k++)
		{
			ACCELERATION[k] = (F[i][k]*(REAL)(pow(a, -3.0)) - 2.0*(REAL)(Hubble_param)*x[i][k+3]);
			x[i][k+3] = x[i][k+3] + ACCELERATION[k]*(REAL)(h/2.0);
		}
			err = sqrt(ACCELERATION[0]*ACCELERATION[0] + ACCELERATION[1]*ACCELERATION[1] + ACCELERATION[2]*ACCELERATION[2]);
			if(err>errmax)
			{
				errmax = err;
			}
	}
	printf("KDK Leapfrog integration...done.\n");
	//Timing
	REAL step_end_time = (REAL) clock () / (REAL) CLOCKS_PER_SEC;
	REAL step_omp_end_time = omp_get_wtime();
	//Timing
	printf("Timestep CPU time = %fs\n", step_end_time-step_start_time);
	printf("Timestep RUN time = %fs\n", step_omp_end_time-step_omp_start_time);

return;
}

/*******************************************************************************/
/*  StePS - STEreographically Projected cosmological Simulations                */
/*    Copyright (C) 2017-2022 Gabor Racz                                        */
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

#ifndef PERIODIC
cudaError_t forces_cuda(REAL*x, REAL*F, int n_GPU, int ID_min, int ID_max);
#else
int ewald_space(REAL R, int ewald_index[2102][4]);
cudaError_t forces_periodic_cuda(REAL*x, REAL*F, int n_GPU, int ID_min, int ID_max);
#endif

#ifndef PERIODIC
void forces(REAL*x, REAL*F, int ID_min, int ID_max)
{
	forces_cuda(x, F, n_GPU, ID_min, ID_max);
	return;
}
#endif
#ifdef PERIODIC
void forces_periodic(REAL*x, REAL*F, int ID_min, int ID_max)
{
	forces_periodic_cuda(x, F, n_GPU, ID_min, ID_max);
	return;
}
#endif


void recalculate_softening();

#ifndef PERIODIC
__global__ void ForceKernel(int n, const int N, const REAL *xx, const REAL *xy, const REAL *xz, REAL *F, const REAL* M, const REAL* SOFT_LENGTH, const REAL mass_in_unit_sphere, const REAL DE, const int COSMOLOGY, const int COMOVING_INTEGRATION, int ID_min, int ID_max)
{
	REAL Fx_tmp, Fy_tmp, Fz_tmp;
	REAL r, dx, dy, dz, wij, beta_priv, beta_privp2;
	REAL SOFT_CONST[5];
	int i, j, id;
	id = blockIdx.x * blockDim.x + threadIdx.x;
	Fx_tmp = Fy_tmp = Fz_tmp = 0.0;
	for (i = (ID_min+id); i<=ID_max; i+=n)
		{
			for (j = 0; j<N; j++)
			{
				beta_priv = (SOFT_LENGTH[i]+SOFT_LENGTH[j]);
				beta_privp2 = beta_priv*0.5;
				//calculating particle distances
				dx = (xx[j] - xx[i]);
				dy = (xy[j] - xy[i]);
				dz = (xz[j] - xz[i]);
				r = sqrt(pow(dx, 2) + pow(dy, 2) + pow(dz, 2));
				if (r >= beta_priv)
				{
					wij = M[j] / (pow(r, 3));
				}
				else if (r > beta_privp2 && r < beta_priv)
				{
					SOFT_CONST[0] = -32.0/(3.0*pow(beta_priv, 6));
					SOFT_CONST[1] = 38.4/pow(beta_priv, 5);
					SOFT_CONST[2] = -48.0/pow(beta_priv, 4);
					SOFT_CONST[3] = 64.0/(3.0*pow(beta_priv, 3));
					SOFT_CONST[4] = -1.0/15.0;
					wij = M[j]*(SOFT_CONST[0] * pow(r, 3) + SOFT_CONST[1] * pow(r, 2) + SOFT_CONST[2] * r + SOFT_CONST[3] + SOFT_CONST[4] / pow(r, 3));
				}
				else
				{
					SOFT_CONST[0] = 32.0/pow(beta_priv, 6);
					SOFT_CONST[1] = -38.4/pow(beta_priv, 5);
					SOFT_CONST[2] = 32.0/(3.0*pow(beta_priv, 3));
					wij = M[j]*(SOFT_CONST[0] * pow(r, 3) + SOFT_CONST[1] * pow(r, 2) + SOFT_CONST[2]);
				}
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
__global__ void ForceKernel_periodic(int n, int N, const REAL *xx, const REAL *xy, const REAL *xz, REAL *F, const int IS_PERIODIC, const REAL* M, const REAL* SOFT_LENGTH, const REAL L, const int *e, int el, int ID_min, int ID_max)
{
	REAL Fx_tmp, Fy_tmp, Fz_tmp;
	REAL r, dx, dy, dz, wij, beta_priv, beta_privp2;
	REAL SOFT_CONST[5];
	int i, j, m, id;

	id = blockIdx.x * blockDim.x + threadIdx.x;
	Fx_tmp = Fy_tmp = Fz_tmp = 0;
	if (IS_PERIODIC == 1)
	{
		for (i = (ID_min+id); i<=ID_max; i+=n)
		{
			for (j = 0; j<N; j++)
			{
				beta_priv = (SOFT_LENGTH[i]+SOFT_LENGTH[j]);
				beta_privp2 = beta_priv*0.5;
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
				if (r >= beta_priv)
				{
					wij = M[j] / (pow(r, 3));
				}
				else if (r > beta_privp2 && r < beta_priv)
                                {
					SOFT_CONST[0] = -32.0/(3.0*pow(beta_priv, 6));
					SOFT_CONST[1] = 38.4/pow(beta_priv, 5);
					SOFT_CONST[2] = -48.0/pow(beta_priv, 4);
					SOFT_CONST[3] = 64.0/(3.0*pow(beta_priv, 3));
					SOFT_CONST[4] = -1.0/15.0;
					wij = M[j]*(SOFT_CONST[0] * pow(r, 3) + SOFT_CONST[1] * pow(r, 2) + SOFT_CONST[2] * r + SOFT_CONST[3] + SOFT_CONST[4] / pow(r, 3));
				}
				else
				{
					SOFT_CONST[0] = 32.0/pow(beta_priv, 6);
					SOFT_CONST[1] = -38.4/pow(beta_priv, 5);
					SOFT_CONST[2] = 32.0/(3.0*pow(beta_priv, 3));
					wij = M[j]*(SOFT_CONST[0] * pow(r, 3) + SOFT_CONST[1] * pow(r, 2) + SOFT_CONST[2]);
				}
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
		for (i = (ID_min+id); i<=ID_max; i=i+n)
		{
			for (j = 0; j<N; j++)
			{
				beta_priv = (SOFT_LENGTH[i]+SOFT_LENGTH[j]);
				beta_privp2 = beta_priv*0.5;
				//calculating particle distances
				dx = (xx[j] - xx[i]);
				dy = (xy[j] - xy[i]);
				dz = (xz[j] - xz[i]);
				//in this function we use multiple images
				for (m = 0; m < 3*el; m = m+3)
				{
					r = sqrt(pow((dx - ((REAL)e[m])*L), 2) + pow((dy - ((REAL)e[m+1])*L), 2) + pow((dz-((REAL)e[m+2])*L), 2));
					wij = 0.0;
					if (r >= beta_priv && r < 2.6*L)
					{
						wij = M[j] / (pow(r, 3));
					}
					else if (r > beta_privp2 && r <= beta_priv)
					{
						SOFT_CONST[0] = -32.0/(3.0*pow(beta_priv, 6));
						SOFT_CONST[1] = 38.4/pow(beta_priv, 5);
						SOFT_CONST[2] = -48.0/pow(beta_priv, 4);
						SOFT_CONST[3] = 64.0/(3.0*pow(beta_priv, 3));
						SOFT_CONST[4] = -1.0/15.0;
						wij = M[j]*(SOFT_CONST[0] * pow(r, 3) + SOFT_CONST[1] * pow(r, 2) + SOFT_CONST[2] * r + SOFT_CONST[3] + SOFT_CONST[4] / pow(r, 3));
					}
					else if (r <= beta_privp2)
					{
						SOFT_CONST[0] = 32.0/pow(beta_priv, 6);
						SOFT_CONST[1] = -38.4/pow(beta_priv, 5);
						SOFT_CONST[2] = 32.0/(3.0*pow(beta_priv, 3));
						wij = M[j]*(SOFT_CONST[0] * pow(r, 3) + SOFT_CONST[1] * pow(r, 2) + SOFT_CONST[2]);
					}
					if (wij != 0)
					{
						Fx_tmp += wij*(dx - ((REAL)e[m])*L);
						Fy_tmp += wij*(dy - ((REAL)e[m + 1])*L);
						Fz_tmp += wij*(dz - ((REAL)e[m + 2])*L);
					}
				}

			}
			F[3 * (i-ID_min)] += Fx_tmp;
			F[3 * (i-ID_min) + 1] += Fy_tmp;
			F[3 * (i-ID_min) + 2] += Fz_tmp;
			Fx_tmp = Fy_tmp = Fz_tmp = 0;
		}
	}

}
#endif

void recalculate_softening()
{
	beta = ParticleRadi;
	if(COSMOLOGY ==1)
	{
		rho_part = M_min/(4.0*pi*pow(beta, 3.0) / 3.0);
	}
}

#ifndef PERIODIC
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
	int *dev_e;
	int e_tmp[6606];

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
#pragma omp parallel default(shared) private(GPU_ID, F_tmp, i, j, mprocessors, cudaStatus, N_GPU, GPU_index_min, nthreads, dev_xx, dev_xy, dev_xz, dev_M, dev_F, dev_SOFT_LENGTH, dev_e)
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
			printf("MPI task %i: GPU force calculation.\n Number of GPUs: %i\n Number of OMP threads: %i\n Number of threads per GPU: %i\n", rank, n_GPU, nthreads, 32*mprocessors*BLOCKSIZE);
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
		// Allocate GPU buffers for e matrix
		cudaStatus = cudaMalloc((void**)&dev_e, 6606 * sizeof(int));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "MPI rank %i: GPU%i: e cudaMalloc failed!\n", rank, GPU_ID);
			ForceError = true;
			goto Error;
		}
		//Converting e matrix into a vector
		for (i = 0; i < 2202; i++)
		{
			for (j = 0; j < 3; j++)
			{
				e_tmp[3 * i + j] = e[i][j];
			}
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
		cudaStatus = cudaMemcpy(dev_e, e_tmp, 6606 * sizeof(int), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "MPI rank %i: GPU%i: cudaMemcpy e in failed!\n", rank, GPU_ID);
			ForceError = true;
			goto Error;
		}
		printf("MPI task %i: GPU%i: ID_min = %i\tID_max = %i\n", rank, GPU_ID, GPU_index_min, GPU_index_min+N_GPU-1);
		// Launch a kernel on the GPU with one thread for each element.
		ForceKernel_periodic << <32*mprocessors, BLOCKSIZE>> >(32*mprocessors * BLOCKSIZE, N, dev_xx, dev_xy, dev_xz, dev_F, IS_PERIODIC, dev_M, dev_SOFT_LENGTH, L, dev_e, el, GPU_index_min, GPU_index_min+N_GPU-1);

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
		cudaFree(dev_e);
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

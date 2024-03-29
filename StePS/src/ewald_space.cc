/********************************************************************************/
/*  StePS - STEreographically Projected cosmological Simulations		*/
/*    Copyright (C) 2017-2022 Gabor Racz					*/
/*										*/
/*    This program is free software; you can redistribute it and/or modify	*/
/*    it under the terms of the GNU General Public License as published by	*/
/*    the Free Software Foundation; either version 2 of the License, or		*/
/*    (at your option) any later version.					*/
/*										*/
/*    This program is distributed in the hope that it will be useful,		*/
/*    but WITHOUT ANY WARRANTY; without even the implied warranty of		*/
/*    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the		*/
/*    GNU General Public License for more details.				*/
/********************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "mpi.h"
#include "global_variables.h"

#ifdef USE_SINGLE_PRECISION
typedef float REAL;
#else
typedef double REAL;
#endif

int ewald_space(REAL R, int ewald_index[2102][4])
{
	int i,j,k;
	int l=-1;
	printf("MPI task %i: Making Ewald sphere...\n", rank);
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
printf("MPI task %i: ...Ewald sphere finished. The number of the simulation box copies:%i\n", rank, l);
return l;
}

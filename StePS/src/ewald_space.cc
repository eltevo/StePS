#include <stdio.h>
#include <stdlib.h>
#include <math.h>
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
	printf("Making Ewald sphere...\n");
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
printf("...Ewald sphere finished. The number of the simulation box copies:%i\n", l);
return l;
}

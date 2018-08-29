#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "kdtree.h"
#include "global_variables.h"

#ifdef USE_SINGLE_PRECISION
typedef float REAL;
#else
typedef double REAL;
#endif

//This file is based on the example GADGET2 file reader code in http://wwwmpa.mpa-garching.mpg.de/gadget/gadget-2.0.7.tar.gz

int allocate_memory(void);

struct io_header_1
{
  int npart[6];
  double mass[6];
  double time;
  double redshift;
  int flag_sfr;
  int flag_feedback;
  int npartTotal[6];
  int flag_cooling;
  int num_files;
  double BoxSize;
  double Omega0;
  double OmegaLambda;
  double HubbleParam;
  char fill[256 - 6 * 4 - 6 * 8 - 2 * 8 - 2 * 4 - 6 * 4 - 2 * 4 - 4 * 8];	/* fills to 256 Bytes */
} header1;



int NumPart, Ngas;

struct particle_data
{
  float Pos[3];
  float Vel[3];
  float Mass;
  int Type;

  float Rho, U, Temp, Ne;
} *P;


double Time, Redshift;



/* Here we load a snapshot file. It can be distributed
 * onto several files (for files>1).
 * The particles are brought back into the order
 * implied by their ID's.
 * A unit conversion routine is called to do unit
 * conversion, and to evaluate the gas temperature.
 */





int gadget_format_conversion(void)
{
	int k, i, j;
	printf("Number of particles = %i\n", NumPart);
	N=NumPart;//Set global variable N to the readed particlenumber;
	printf("Time = %f Redshift= %f\n", Time, (1/Time)-1);
	printf("Boxsize \t %f kpc/h\n", header1.BoxSize*dist_unit_in_kpc);
        L = (REAL) header1.BoxSize*dist_unit_in_kpc/1000.0/header1.HubbleParam;
	printf("Boxsize \t %f Mpc\n", L);
	i=0;
	printf("Converting particle positions...\n");
	for(k=1;k<NumPart+1;++k)
	{
		if(P[k].Type == 1)
		{
			//We do not use the h=H0/100km/s/Mpc factors
			for(j=0; j<3; j++)
			{
				x[i][j] = (REAL)P[k].Pos[j]*dist_unit_in_kpc/1000.0/header1.HubbleParam - VOI[j] + L/2.0;
				if(x[i][j] < 0)
				{
					x[i][j] += L;
				}
				else if(x[i][j]>L)
				{
					x[i][j] -= L;
				}
			}
			i++;
		}

	}
	printf("...done\n");

	printf("Converting particle velocities...\n");
	i=0;
	for(k=1;k<NumPart+1;++k)
	{
		if(P[k].Type == 1)
		{
			//Converting particle velocities (GADGET uses km/s)
			x[i][3] = (REAL)P[k].Vel[0] / UNIT_V;
			x[i][4] = (REAL)P[k].Vel[1] / UNIT_V;
			x[i][5] = (REAL)P[k].Vel[2] / UNIT_V;
			i++;
		}
	}
	printf("...done\n");
	//freeing memory
	P++;
	free(P);
	return 0;

}









/* this routine loads particle data from Gadget's default
 * binary file format. (A snapshot may be distributed
 * into multiple files.
 */
int load_snapshot(char *fname, int files)
{
  FILE *fd;
  char buf[200];
  int i, k, dummy, ntot_withmasses;
  int n, pc, pc_new, pc_sph;

#define SKIP fread(&dummy, sizeof(dummy), 1, fd);

  for(i = 0, pc = 1; i < files; i++, pc = pc_new)
    {
      if(files > 1)
	sprintf(buf, "%s.%d", fname, i);
      else
	sprintf(buf, "%s", fname);

      if(!(fd = fopen(buf, "r")))
	{
	  printf("can't open file `%s`\n", buf);
	  exit(0);
	}

      printf("reading `%s' ...\n", buf);
      fflush(stdout);

      fread(&dummy, sizeof(dummy), 1, fd);
      fread(&header1, sizeof(header1), 1, fd);
      fread(&dummy, sizeof(dummy), 1, fd);

      if(files == 1)
	{
	  for(k = 0, NumPart = 0, ntot_withmasses = 0; k < 6; k++)
	    NumPart += header1.npart[k];
	  Ngas = header1.npart[0];
	}
      else
	{
	  for(k = 0, NumPart = 0, ntot_withmasses = 0; k < 6; k++)
	    NumPart += header1.npartTotal[k];
	  Ngas = header1.npartTotal[0];
	}

      for(k = 0, ntot_withmasses = 0; k < 6; k++)
	{
	  if(header1.mass[k] == 0)
	    ntot_withmasses += header1.npart[k];
	}

      if(i == 0)
	allocate_memory();

      SKIP;
      for(k = 0, pc_new = pc; k < 6; k++)
	{
	  for(n = 0; n < header1.npart[k]; n++)
	    {
	      fread(&P[pc_new].Pos[0], sizeof(float), 3, fd);
	      pc_new++;
	    }
	}
      SKIP;

      SKIP;
      for(k = 0, pc_new = pc; k < 6; k++)
	{
	  for(n = 0; n < header1.npart[k]; n++)
	    {
	      fread(&P[pc_new].Vel[0], sizeof(float), 3, fd);
	      pc_new++;
	    }
	}
      SKIP;


      SKIP;
      for(k = 0, pc_new = pc; k < 6; k++)
	{
	  for(n = 0; n < header1.npart[k]; n++)
	    {
	      SKIP;//we do not read the Id
	      pc_new++;
	    }
	}
      SKIP;


      if(ntot_withmasses > 0)
	SKIP;
      for(k = 0, pc_new = pc; k < 6; k++)
	{
	  for(n = 0; n < header1.npart[k]; n++)
	    {
	      P[pc_new].Type = k;

	      if(header1.mass[k] == 0)
		fread(&P[pc_new].Mass, sizeof(float), 1, fd);
	      else
		P[pc_new].Mass = header1.mass[k];
	      pc_new++;
	    }
	}
      if(ntot_withmasses > 0)
	SKIP;


      if(header1.npart[0] > 0)
	{
	  SKIP;
	  for(n = 0, pc_sph = pc; n < header1.npart[0]; n++)
	    {
	      fread(&P[pc_sph].U, sizeof(float), 1, fd);
	      pc_sph++;
	    }
	  SKIP;

	  SKIP;
	  for(n = 0, pc_sph = pc; n < header1.npart[0]; n++)
	    {
	      fread(&P[pc_sph].Rho, sizeof(float), 1, fd);
	      pc_sph++;
	    }
	  SKIP;

	  if(header1.flag_cooling)
	    {
	      SKIP;
	      for(n = 0, pc_sph = pc; n < header1.npart[0]; n++)
		{
		  fread(&P[pc_sph].Ne, sizeof(float), 1, fd);
		  pc_sph++;
		}
	      SKIP;
	    }
	  else
	    for(n = 0, pc_sph = pc; n < header1.npart[0]; n++)
	      {
		P[pc_sph].Ne = 1.0;
		pc_sph++;
	      }
	}

      fclose(fd);
    }


  Time = header1.time;
  Redshift = header1.time;
  return 0;
}




/* this routine allocates the memory for the 
 * particle data.
 */
int allocate_memory(void)
{
  printf("allocating memory...\n");

  if(!(P = (struct particle_data *)malloc(NumPart * sizeof(struct particle_data))))
    {
      fprintf(stderr, "failed to allocate memory.\n");
      exit(0);
    }

  P--;				/* start with offset 1 */


  printf("allocating memory...done\n");
  return 0;
}


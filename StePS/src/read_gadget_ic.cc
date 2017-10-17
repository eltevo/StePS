#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
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

int *Id;

double Time, Redshift;



/* Here we load a snapshot file. It can be distributed
 * onto several files (for files>1).
 * The particles are brought back into the order
 * implied by their ID's.
 * A unit conversion routine is called to do unit
 * conversion, and to evaluate the gas temperature.
 */





/* here the particle data is at your disposal 
 */
int gadget_format_conversion(void)
{
	int k, i;
	printf("Nuber of particles = %i\n", NumPart);
	printf("Time = %f Redshift= %f\n", Time, (1/Time)-1);
	printf("\nCosmological parameters:\n------------------------\n");
	printf("Boxsize \t %f kpc/h\n", header1.BoxSize);
	//Gadget formátumú IC esetén a dobozméretet az IC határozza meg.
        L = (REAL) header1.BoxSize/1000.0/header1.HubbleParam;
	//M=header1.mass[1]/header1.HubbleParam/10;
	printf("Boxsize \t %f Mpc\n", L);
	printf("Omega0 \t\t %f \n", header1.Omega0);
	printf("OmegaLambda \t %f \n", header1.OmegaLambda);
	printf("HubbleParam \t %f \n\n", header1.HubbleParam);

	printf("Masstab(in 1.0e10 solar masses):\n--------------------------------\n");
	printf("Gas: \t\t\t%f\n", header1.mass[0]);
	printf("Dark Matter(HALO): \t%f\n", header1.mass[1]);
	printf("Disc: \t\t\t%f\n", header1.mass[2]);
	printf("Bulge: \t\t\t%f\n", header1.mass[3]);
	printf("Stars: \t\t\t%f\n", header1.mass[4]);
	printf("Bndry: \t\t\t%f\n\n", header1.mass[5]);
	
	i=0;
	printf("Converting particle positions...\n");
	for(k=1;k<NumPart+1;++k)
	{
		if(P[k].Type == 1)
		{
			//We do not use the h=H0/100km/s/Mpc factors
			x[i][0] = (REAL)P[k].Pos[0]/1000.0/header1.HubbleParam;
			x[i][1] = (REAL)P[k].Pos[1]/1000.0/header1.HubbleParam;
			x[i][2] = (REAL)P[k].Pos[2]/1000.0/header1.HubbleParam;
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
			x[i][3] = (REAL)P[k].Vel[0]*0.0482190732645461;
			x[i][4] = (REAL)P[k].Vel[1]*0.0482190732645461;
			x[i][5] = (REAL)P[k].Vel[2]*0.0482190732645461;
			i++;
		}
	}
	printf("...done\n");
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
	      fread(&Id[pc_new], sizeof(int), 1, fd);
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


  if(!(Id = (int *)malloc(NumPart * sizeof(int))))
    {
      fprintf(stderr, "failed to allocate memory.\n");
      exit(0);
    }

  Id--;				/* start with offset 1 */

  printf("allocating memory...done\n");
  return 0;
}




/* This routine brings the particles back into
 * the order of their ID's.
 * NOTE: The routine only works if the ID's cover
 * the range from 1 to NumPart !
 * In other cases, one has to use more general
 * sorting routines.
 */
int reordering(void)
{
  int i;
  int idsource, idsave, dest;
  struct particle_data psave, psource;


  printf("reordering....\n");

  for(i = 1; i <= NumPart; i++)
    {
      if(Id[i] != i)
	{
	  psource = P[i];
	  idsource = Id[i];
	  dest = Id[i];

	  do
	    {
	      psave = P[dest];
	      idsave = Id[dest];

	      P[dest] = psource;
	      Id[dest] = idsource;

	      if(dest == i)
		break;

	      psource = psave;
	      idsource = idsave;

	      dest = idsource;
	    }
	  while(1);
	}
    }

  printf("done.\n");

  Id++;
  free(Id);

  printf("space for particle ID freed\n");
  return 0;
}

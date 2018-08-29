#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <gsl/gsl_integration.h>
#include <time.h>
#include "kdtree.h"
#include "global_variables.h"

//calculating the comoving distance and redshift

double *DC;

struct E_params{double Omega_m; double Omega_r; double Omega_lambda;};

double E(double z, void * p)
{
	struct E_params * params = (struct E_params *)p;
	double Omega_m = (params->Omega_m);
	double Omega_r = (params->Omega_r);
	double Omega_lambda = (params->Omega_lambda);

	double omega_k = 1.0-Omega_m-Omega_lambda-Omega_r;
	if(omega_k > 1e-9)
		return 1.0/sqrt(Omega_m*pow((1+z), 3.0) + Omega_r*pow((1+z), 4.0) + omega_k*pow((1+z), 2.0) + Omega_lambda);
	else
		return 1.0/sqrt(Omega_m*pow((1+z), 3.0) + Omega_r*pow((1+z), 4.0) + Omega_lambda);
}

double comoving_distance(double z, double Omega_m, double Omega_r, double Omega_lambda)
{
	gsl_integration_workspace * w = gsl_integration_workspace_alloc (2000);
	double DC, DC_err;
	gsl_function F;
	struct E_params params = {Omega_m, Omega_r, Omega_lambda};
	F.function = &E;
	F.params = &params;
	
	gsl_integration_qags(&F, 0, z, 0, 1e-13, 2000, w, &DC, &DC_err);
	gsl_integration_workspace_free(w);

	return DC/H0*0.0482190732645461*299792.458; // H_D = c/H0 [Mpc]
}

void calculate_DC_array(double *z_list, double Omega_m, double Omega_r, double Omega_lambda)
{
	DC[0] = 0.0;
	for(long int i=1;i<R_GRID*50;i++)
	{
		DC[i] = comoving_distance(z_list[i], Omega_m, Omega_r, Omega_lambda);
	}
	return;
}

double z_DC(double Dcom, double *z_list, double* DC_array, long int array_size)
{
	long int i, index;
	i=0;
	index = 0;
	while(Dcom>DC_array[i])
	{
		++i;
		index = i;
		if(i > array_size)
		{
			fprintf(stderr, "Warning: Too big comoving radial distance!\n");
			index = array_size-1;
			break;
		}
	}
	//calculating z using linear interpolation
	double a_DC, b_DC;
	a_DC = (z_list[index]-z_list[index-1]) / (DC_array[index]-DC_array[index-1]);
	b_DC = z_list[index-1]-a_DC*DC_array[index-1];
	return (Dcom*a_DC+b_DC);
}

void Calculate_redshifts()
{
	printf("\nCalculating redshifts for the spherical sells...\n\n");
	printf("The cosmological parameters:\nH0 =\t%fkm/s/Mpc\nOmega_m =\t%f\nOmega_r =\t%f\nOmega_lambda =\t%f\nOmega_k =%f\n", H0*UNIT_V, Omega_m, Omega_r, Omega_lambda, 1.0-Omega_m-Omega_r-Omega_lambda);


	double z_max = 1.0/a_start-1.0;
	double *zlist;
	long int N_zlist = R_GRID*50;
	DC = (double*)malloc(N_zlist*sizeof(double));
	zlist = (double*)malloc(N_zlist*sizeof(double));
	long int i;
	//Searching the maximal meaningful redshift (from the maximal radial comoving distance)
	for(i = 0; i<N_zlist; i++)
	{
		zlist[i] = z_max*(double)i/N_zlist;
	}
	calculate_DC_array(zlist, Omega_m, Omega_r, Omega_lambda);
	i=0;
	if(DC[N_zlist-1]<R_CUT)
		fprintf(stderr, "Warning: Too big comoving radial distances for the initial redshift!\n");
	while(DC[i]<R_CUT && i < N_zlist-1)
	{
		i++;
		z_max = zlist[i];
	}
	//Rebinning, using the maximal meaningful redshift.
	for(i = 0; i<N_zlist; i++)
	{
		zlist[i] = z_max*(double)i/N_zlist;
		DC[i] = 0;
	}
	calculate_DC_array(zlist, Omega_m, Omega_r, Omega_lambda);
	printf("The used redshift and comoving distance range:\nz_min = %f\tz_max = %f\nR_min = %fMpc\tR_max = %fMpc\n", zlist[0], zlist[N_zlist-1], DC[0], DC[N_zlist-1]);

	//Calculating comoving distances, redshifts for the shells.
	double omega_CUT = 2*atan(R_CUT/SPHERE_DIAMETER);
	double *r_bin_limits, *r_bin_centers, *z_bin_centers;
	r_bin_limits = (double*)malloc((R_GRID+1)*sizeof(double));
	r_bin_centers = (double*)malloc((R_GRID)*sizeof(double));
	z_bin_centers = (double*)malloc((R_GRID)*sizeof(double));
	
	r_bin_limits[0] = 0;
	for(i=1;i<R_GRID+1;i++)
	{
		r_bin_limits[i] = SPHERE_DIAMETER*tan((double)i / (2*R_GRID) * omega_CUT);
		r_bin_centers[i-1] = (r_bin_limits[i]+r_bin_limits[i-1])/2.0;
		z_bin_centers[i-1] = z_DC(r_bin_centers[i-1], zlist, DC, N_zlist); 
	}
	//writing out the grid centers and limits:
	char OUT_Z_LIST_file[1030];
	char OUT_BIN_LIMITS_file[1038]; 
	snprintf(OUT_Z_LIST_file, sizeof(OUT_Z_LIST_file), "%s_zbins", OUT_FILE);
	snprintf(OUT_BIN_LIMITS_file, sizeof(OUT_BIN_LIMITS_file), "%s_zbins_rlimits", OUT_FILE);
	FILE *out_z_file=fopen(OUT_Z_LIST_file, "w");
	FILE *out_limits_file=fopen(OUT_BIN_LIMITS_file, "w");
	
	//Writing output radial grid centers: r[Mpc] z;
	for(i=0;i<R_GRID;i++)
	{
		fprintf(out_z_file,"%e\n", z_bin_centers[i]);
		fprintf(out_limits_file,"%.15f\n", r_bin_limits[i]);
	}
	fprintf(out_limits_file,"%.15f\n", r_bin_limits[R_GRID]);
	fclose(out_z_file);
	fclose(out_limits_file);
	return;
}

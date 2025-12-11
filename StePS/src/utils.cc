/********************************************************************************/
/*  StePS - STEreographically Projected cosmological Simulations                */
/*    Copyright (C) 2017-2025 Gabor Racz, Viola Varga                           */
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

//this file contains the helper functions for the StePS code

#include <stdio.h>
#include <math.h>
#include "mpi.h"
#include "global_variables.h"

//interpolator functions
REAL linear_interpolation(REAL X, REAL X1, REAL Y1, REAL X2, REAL Y2)
{
    //helper function for linear interpolation
    //         Y2
    //       / |
    //      ?  |
    //    / |  |
    //  Y1  |  |
    //  |   |  |
    //--X1--X--X2
    REAL A=(Y2-Y1)/(X2-X1);
    REAL B=Y1-A*X1;
    return A*X+B;
}

REAL quadratic_interpolation(REAL X, REAL X1, REAL Y1, REAL X2, REAL Y2, REAL X3, REAL Y3)
{
    //helper function for quadratic_interpolation
    REAL L1 = (X-X2)*(X-X3)/((X1-X2)*(X1-X3));
    REAL L2 = (X-X1)*(X-X3)/((X2-X1)*(X2-X3));
    REAL L3 = (X-X1)*(X-X2)/((X3-X1)*(X3-X2));
    return Y1*L1+Y2*L2+Y3*L3;
}

REAL cubic_interpolation(REAL X, REAL X1, REAL Y1, REAL X2, REAL Y2, REAL X3, REAL Y3, REAL X4, REAL Y4)
{
    //helper function for cubic interpolation
    REAL L1 = (X-X2)*(X-X3)*(X-X4)/((X1-X2)*(X1-X3)*(X1-X4));
    REAL L2 = (X-X1)*(X-X3)*(X-X4)/((X2-X1)*(X2-X3)*(X2-X4));
    REAL L3 = (X-X1)*(X-X2)*(X-X4)/((X3-X1)*(X3-X2)*(X3-X4));
    REAL L4 = (X-X1)*(X-X2)*(X-X3)/((X4-X1)*(X4-X2)*(X4-X3));
    return Y1*L1+Y2*L2+Y3*L3+Y4*L4;
}

// Softening lenghth calculation
void calculate_softening_length(REAL *SOFT_LENGTH, REAL *M, int N)
{
    int i;
    M_min = M[0];
    for(i=0;i<N;i++)
    {
        if(M_min>M[i])
        {
            M_min = M[i];
        }
    }
    rho_part = M_min/(4.0*pi*pow(ParticleRadi, 3.0) / 3.0);
    //Calculating the softening length for each particle:
    REAL const_beta = 3.0/rho_part/(4.0*pi);
    printf("Calculating the softening lengths...\n");
    printf("\tMmin = %f * 10^11 Msol\tMinimal Particle Radius=%fMpc\tParticle density=%f * 10^11 Msol/Mpc^3\n", M_min, ParticleRadi, rho_part);
    for(i=0;i<N;i++)
    {
        SOFT_LENGTH[i] = cbrt(M[i]*const_beta); //setting up the softening length for each particle
        if(N<10)
            printf("SOFT_LENGTH[%i] = %f\n", i, SOFT_LENGTH[i]);
    }
    printf("...done\n\n");
}

#if defined(USE_BH) && !defined(PERIODIC)
// Function to calculate the radial force correction for BH force calculation
void get_radial_bh_force_correction_table(REAL *RADIAL_BH_FORCE_TABLE, int *RADIAL_BH_N_TABLE, int TABLE_SIZE, REAL *F, REAL *x, int N)
{
    // This function calculates the radial force correction table for the spherical and cylindrical BH force calculation.
    // The table is calculated by averaging the forces in the radial direction, in TABLE_SIZE radial bins.
    int i;
    double r, F_radial;
    double *FORCE_TABLE_LOC; // Temporary array to hold the forces in the radial direction
    int *RADIAL_BH_N_TABLE_LOC; // Temporary array to hold the number of particles in each radial bin
    if(!(FORCE_TABLE_LOC = (double*)malloc(TABLE_SIZE*sizeof(double))))
    {
        fprintf(stderr, "Failed to allocate memory for FORCE_TABLE_LOC.\n");
        exit(-2);
    }
    if(!(RADIAL_BH_N_TABLE_LOC = (int*)malloc(TABLE_SIZE*sizeof(int))))
    {
        fprintf(stderr, "Failed to allocate memory for RADIAL_BH_N_TABLE_LOC.\n");
        exit(-2);
    }
    // Initialize the FORCE_TABLE_LOC and RADIAL_BH_N_TABLE_LOC arrays
    for(i=0; i<TABLE_SIZE; i++)
    {
        FORCE_TABLE_LOC[i] = 0.0;
        RADIAL_BH_N_TABLE_LOC[i] = 0;
    }
    // Calculate the radial forces and fill the FORCE_TABLE_LOC and RADIAL_BH_N_TABLE_LOC arrays
    printf("Calculating the radial force correction table...\n");
    for(i=0; i<N; i++)
    {
        // Calculate the radial distance of the particle from the center of the simulation volume
        #if !defined(PERIODIC_Z)
            // In non-periodic simulations, we calculate the radial distance in 3D
            r = (double) sqrt(pow(x[3*i], 2) + pow(x[3*i+1], 2) + pow(x[3*i+2], 2));
            F_radial = (double) (F[3*i] * x[3*i] + F[3*i+1] * x[3*i+1] + F[3*i+2] * x[3*i+2]) / r; // the radial component of the force
        #else
            // In cylindrical simulations, we calculate the radial distance in 2D
            r = (double) sqrt(pow(x[3*i], 2) + pow(x[3*i+1], 2));
            F_radial = (double) (F[3*i] * x[3*i] + F[3*i+1] * x[3*i+1]) / r; // the radial component of the force
        #endif
        // Find the bin index for the radial distance
        int bin_index = (int)floor(r / (Rsim / (double) TABLE_SIZE));
        if(bin_index >= TABLE_SIZE) 
            bin_index = TABLE_SIZE - 1; // Ensure we don't go out of bounds
        // Accumulate the force in the radial direction
        FORCE_TABLE_LOC[bin_index] += F_radial;
        RADIAL_BH_N_TABLE_LOC[bin_index]++; // Count the number of particles in this shell
    }
    //printing the results
    /*printf("Table values in this iteration:\nBin\tR[Mpc]\t\tForce\t\tRADIAL_BH_N_TABLE\n-------------------------------------\n");
    for(i=0; i<RADIAL_BH_FORCE_TABLE_SIZE; i++)
    {
        printf("%i\t%f\t%f\t %i\n", i, ((double) i + 0.5) * (Rsim / (double) RADIAL_BH_FORCE_TABLE_SIZE), FORCE_TABLE_LOC[i]/(double)RADIAL_BH_N_TABLE_LOC[i], RADIAL_BH_N_TABLE_LOC[i]);
    }*/
    for(i=0; i<TABLE_SIZE; i++)
    {
        RADIAL_BH_FORCE_TABLE[i] += (REAL) FORCE_TABLE_LOC[i]; // Adding the results to the output table
        RADIAL_BH_N_TABLE[i] += RADIAL_BH_N_TABLE_LOC[i]; // Adding the number of particles in this shell to the output table
    }
    free(FORCE_TABLE_LOC); // Free the temporary array
    free(RADIAL_BH_N_TABLE_LOC); // Free the temporary array
    printf("...Radial force correction table calculated.\n\n");
    return;
}
#endif

#if defined(PERIODIC_Z)
//Force table calculation for cylindrical simulations
void get_cylindrical_force_table(REAL* FORCE_TABLE, REAL R, REAL Lz, int TABLE_SIZE, int RADIAL_FORCE_ACCURACY)
{
    double int_R = (double) R;
    double int_Lz = (double) Lz;

    double step = int_R / (double) RADIAL_FORCE_ACCURACY;
    double Y, f1, f2, integrand_prev, integrand, total, a;
    FORCE_TABLE[0] = 0.0; //Set default value so that even if we cannot interpolate, the table doesn't have an empty value

    //Loop to iterate through each r, and get an integral (for the force) of accuracy RADIAL_FORCE_ACCURACY for each
    for (int i = 1; i < TABLE_SIZE; i++)
    {
        if(i==TABLE_SIZE-1)
        {
            a = int_R; //For the last element, we want to set a = R
        }
        else
        {
            a = int_R / (double) (TABLE_SIZE - 1) * i; //Calculate the current r value
        }
        total = 0;

        //Integrator loop using the trapezoidal method
        for (int j = 1; j <= RADIAL_FORCE_ACCURACY; j++)
        {
            if (a >= int_R)
            {
                a = int_R; //in case a > R, since we want the force for a > R to be that of a = R
                Y = (j - 1) * step;
                integrand_prev = -2 * log((pow((pow(int_Lz,2) + 2 * pow(int_R,2) + 2 * int_R * pow((pow(int_R,2) - pow(Y,2)), 0.5)), 0.5) + int_Lz) / (pow((pow(int_Lz,2) + 2 * pow(int_R,2) - 2 * int_R * pow((pow(int_R,2) - pow(Y,2)), 0.5)), 0.5) + int_Lz));
                
                Y = j * step;
                integrand = -2 * log((pow((pow(int_Lz,2) + 2 * pow(int_R,2) + 2 * int_R * pow((pow(int_R,2) - pow(Y,2)), 0.5)), 0.5) + int_Lz) / (pow((pow(int_Lz,2) + 2 * pow(int_R,2) - 2 * int_R * pow((pow(int_R,2) - pow(Y,2)), 0.5)), 0.5) + int_Lz));
            }
            else
            {
                Y = (j - 1) * step;
                f1 = -2 * log((pow((pow(int_Lz,2) + pow(int_R,2) + pow(a,2) + 2*a*pow((pow(int_R,2) - pow(Y,2)), 0.5)),0.5)+int_Lz) / (pow((pow(int_Lz,2) + pow(int_R,2) + pow(a,2) - 2*a*pow((pow(int_R,2) - pow(Y,2)), 0.5)),0.5)+int_Lz));
                f2 = log((pow(int_R,2) + pow(a,2) + 2*a*pow((pow(int_R,2) - pow(Y,2)),0.5))/(pow(int_R,2) + pow(a,2) - 2*a*pow((pow(int_R,2) - pow(Y,2)),0.5)));
                integrand_prev = f1 + f2;

                Y = j * step;
                f1 = -2 * log((pow((pow(int_Lz,2) + pow(int_R,2) + pow(a,2) + 2*a*pow((pow(int_R,2) - pow(Y,2)), 0.5)),0.5)+int_Lz) / (pow((pow(int_Lz,2) + pow(int_R,2) + pow(a,2) - 2*a*pow((pow(int_R,2) - pow(Y,2)), 0.5)),0.5) + int_Lz));
                f2 = log((pow(int_R,2) + pow(a,2) + 2*a*pow((pow(int_R,2) - pow(Y,2)),0.5))/(pow(int_R,2) + pow(a,2) - 2*a*pow((pow(int_R,2) - pow(Y,2)),0.5)));
                integrand = f1 + f2;
            }

            total += step * (integrand_prev + integrand) * 0.5;
        }

        //For each a, calculate the force and add them to the table
        if(a >= int_R)
        {
            FORCE_TABLE[i] = (REAL) (2 * (total + pi * int_R) / (2 * pi * int_R)); //In the last parantheses a = R because we want it to be constant after R
        }
        else
        {
            FORCE_TABLE[i] = (REAL) (2 * total / (2 * pi * a));
        }
    }

    //For the case a = 0, we interpolate
    if(TABLE_SIZE >= 3)
    {
        FORCE_TABLE[0] = (REAL) (linear_interpolation(0, 2 * int_R / (double) TABLE_SIZE, FORCE_TABLE[1], 4 * int_R / (double) TABLE_SIZE, FORCE_TABLE[2]));
    }
}
#endif

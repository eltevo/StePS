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
double linear_interpolation(double X, double X1, double Y1, double X2, double Y2)
{
    //helper function for linear interpolation
    //         Y2
    //       / |
    //      ?  |
    //    / |  |
    //  Y1  |  |
    //  |   |  |
    //--X1--X--X2
    double A=(Y2-Y1)/(X2-X1);
    double B=Y1-A*X1;
    return A*X+B;
}

double quadratic_interpolation(double X, double X1, double Y1, double X2, double Y2, double X3, double Y3)
{
    //helper function for quadratic_interpolation
    double L1 = (X-X2)*(X-X3)/((X1-X2)*(X1-X3));
    double L2 = (X-X1)*(X-X3)/((X2-X1)*(X2-X3));
    double L3 = (X-X1)*(X-X2)/((X3-X1)*(X3-X2));
    return Y1*L1+Y2*L2+Y3*L3;
}

double cubic_interpolation(double X, double X1, double Y1, double X2, double Y2, double X3, double Y3, double X4, double Y4)
{
    //helper function for cubic interpolation
    double L1 = (X-X2)*(X-X3)*(X-X4)/((X1-X2)*(X1-X3)*(X1-X4));
    double L2 = (X-X1)*(X-X3)*(X-X4)/((X2-X1)*(X2-X3)*(X2-X4));
    double L3 = (X-X1)*(X-X2)*(X-X4)/((X3-X1)*(X3-X2)*(X3-X4));
    double L4 = (X-X1)*(X-X2)*(X-X3)/((X4-X1)*(X4-X2)*(X4-X3));
    return Y1*L1+Y2*L2+Y3*L3+Y4*L4;
}

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
        a = int_R / (double) (TABLE_SIZE - 1) * i;
        total = 0;

        //Integrator loop using the trapezoidal method
        for (int j = 1; j <= RADIAL_FORCE_ACCURACY; j++)
        {
            if (a >= int_R)
            {
                a = int_R; //in case a > R, since we want the force for a > R to be that of a = R
                Y = (j - 1) * step;
                integrand_prev = -2 * log((pow((pow(int_Lz,2) + 2 * pow(int_R,2) + 2 * int_R * pow(abs(pow(int_R,2) - pow(Y,2)), 0.5)), 0.5) + int_Lz) / (pow((pow(int_Lz,2) + 2 * pow(int_R,2) - 2 * int_R * pow(abs(pow(int_R,2) - pow(Y,2)), 0.5)), 0.5) + int_Lz));
                
                Y = j * step;
                integrand = -2 * log((pow((pow(int_Lz,2) + 2 * pow(int_R,2) + 2 * int_R * pow(abs(pow(int_R,2) - pow(Y,2)), 0.5)), 0.5) + int_Lz) / (pow((pow(int_Lz,2) + 2 * pow(int_R,2) - 2 * int_R * pow(abs(pow(int_R,2) - pow(Y,2)), 0.5)), 0.5) + int_Lz));
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

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
#include <math.h>
#include "mpi.h"
#include "global_variables.h"

#ifdef USE_SINGLE_PRECISION
typedef float REAL;
#else
typedef double REAL;
#endif

//Friedman-equation integrator. We use 4th order Runge-Kutta integrator

double friedman_solver_step(double a0, double h, double Omega_lambda, double Omega_r, double Omega_m, double Omega_k, double H0);

double friedmann_solver_start(double a0, double t0, double h, double Omega_lambda, double Omega_r, double Omega_m, double H0, double a_start)
{
	//In this function we calculate the time of Big Bang, and the initial time for the simulation.
	printf("Calculating time for the initial scale factor...\n");
	printf("a_start=%f\na0=%f\n", a_start, a0);
	double b, b_tmp, t_cosm_tmp;
	double t_cosm = t0;
	double t_start;
	double t_start_err=1e-20;
	double h_var;
	b = a_start;
	printf("h=%eGy\n", h*UNIT_T);
	double Omega_k = 1.-Omega_m-Omega_lambda-Omega_r;
	//Solving the "da/dt = a*H0*sqrt(Omega_m*pow(a, -3)+Omega_r*pow(a, -4)+Omega_lambda)" differential equation
	b_tmp = b;
	while(0<b)
	{
		b_tmp = b;
		b = friedman_solver_step(b, -h, Omega_lambda, Omega_r, Omega_m, Omega_k, H0);
		t_cosm -= h;
	}
	t_bigbang=t_cosm+h; //rough estimation for t_bibgang.
	b = b_tmp;
	printf("First guess: %.12f Gy\n\n", -t_bigbang*UNIT_T);
	//Searching for t_start.
	h_var = -0.5*h;
	while(fabs(h_var)>t_start_err)
	{
		b_tmp = b;
		b = friedman_solver_step(b, h_var, Omega_lambda, Omega_r, Omega_m, Omega_k, H0);
		t_cosm_tmp = t_cosm;
		t_cosm=t_cosm+h_var;
		if(b<=0)
		{
			b = b_tmp;
			t_cosm = t_cosm_tmp;
			h_var=0.5*h_var;
		}
	}
	t_bigbang = t_cosm;
	//Setting t=0 to Big Bang:
	t_start =  -1.0 * t_bigbang;
	return t_start;
}

double friedman_solver_step(double a0, double h, double Omega_lambda, double Omega_r, double Omega_m, double Omega_k, double H0)
{
	int collapse;
	double b,k1,k2,k3,k4,K;
	double j,l,m,n;

	a_prev = a0;
	if(H0>0)
	{
		collapse=0;
	}
	else
	{
		collapse=1;
	}
	b = a0;
	if(fabs(Omega_k) < 1e-9)
	{
		j=Omega_m*pow(b, -3.0)+Omega_r*pow(b, -4.0)+Omega_lambda;
		k1 = b*H0*sqrt(Omega_m*pow(b, -3.0)+Omega_r*pow(b, -4.0)+Omega_lambda);
		l=Omega_m*pow((b+h*k1/2.0), -3.0)+Omega_r*pow((b+h*k1/2), -4.0)+Omega_lambda;
		k2 = (b+h*k1/2.0)*H0*sqrt(Omega_m*pow((b+h*k1/2.0), -3.0)+Omega_r*pow((b+h*k1/2), -4.0)+Omega_lambda);
		m=Omega_m*pow((b+h*k2/2.0), -3.0)+Omega_r*pow((b+h*k2/2), -4.0)+Omega_lambda;
		k3 = (b+h*k2/2.0)*H0*sqrt(Omega_m*pow((b+h*k2/2.0), -3.0)+Omega_r*pow((b+h*k2/2), -4.0)+Omega_lambda);
		n=Omega_m*pow((b+h*k3), -3.0)+Omega_r*pow((b+h*k3), -4.0)+Omega_lambda;
		k4 = (b+h*k3)*H0*sqrt(Omega_m*pow((b+h*k3), -3.0)+Omega_r*pow((b+h*k3), -4.0)+Omega_lambda);
		K = h*(k1+k2*2.0+k3*2.0+k4)/6.0;
		if(j<0||l<0||m<0||n<0)
			b=-1;
		else
			b += K;
	}
	else
	{
		j= Omega_m*pow(b, -3.0)+Omega_r*pow(b, -4.0)+Omega_lambda+Omega_k*pow(b, -2.0);
		k1 = b*H0*sqrt(fabs(j));
		l = Omega_m*pow((b+h*k1/2.0), -3.0)+Omega_r*pow((b+h*k1/2.0), -4.0)+Omega_lambda+Omega_k*pow((b+h*k1/2.0), -2.0);
		k2 = (b+h*k1/2.0)*H0*sqrt(fabs(l));
		m = Omega_m*pow((b+h*k2/2.0), -3.0)+Omega_r*pow((b+h*k2/2.0), -4.0)+Omega_lambda+Omega_k*pow((b+h*k2/2.0), -2.0);
		k3 = (b+h*k2/2.0)*H0*sqrt(fabs(m));
		n=Omega_m*pow((b+h*k3), -3.0)+Omega_r*pow((b+h*k3), -4.0)+Omega_lambda+Omega_k*pow((b+h*k3), -2.0);
		k4 = (b+h*k3)*H0*sqrt(fabs(n));
		if(j<0&&l<0&&m<0&&n<0)
		{
			collapse = 1;
		}

		if(collapse==0)
		{
			K = h*(k1+k2*2.0+k3*2.0+k4)/6.0;
		}
		else
		{
			K = -h*(k1+k2*2.0+k3*2.0+k4)/6.0;
		}
		b = b + K;
	}
	return b;
}


double CALCULATE_decel_param(double a)
{
	double Decel_param_out;
	double SUM_OMEGA_TMP = Omega_m*pow(a, -3.0) + Omega_r*pow(a, -4.0) + Omega_lambda + Omega_k*pow(a, -2.0);
	double Omega_m_tmp = Omega_m*pow(a, -3.0)/SUM_OMEGA_TMP;
	double Omega_r_tmp = Omega_r*pow(a, -4.0)/SUM_OMEGA_TMP;
	double Omega_lambda_tmp = Omega_lambda/SUM_OMEGA_TMP;
	double Omega_sum_tmp = Omega_m_tmp + Omega_r_tmp + Omega_lambda_tmp;
	Decel_param_out = Omega_sum_tmp/2.0 + Omega_r_tmp - 1.5*Omega_lambda_tmp;
	return Decel_param_out;
}

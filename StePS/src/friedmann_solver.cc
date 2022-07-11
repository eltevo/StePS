/********************************************************************************/
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
#include <math.h>
#include "mpi.h"
#include "global_variables.h"

#ifdef USE_SINGLE_PRECISION
typedef float REAL;
#else
typedef double REAL;
#endif

//Friedman-equation integrator. We use 4th order Runge-Kutta integrator

double friedmann_solver_step(double a0, double h);
double CALCULATE_Hubble_param(double a);

#if COSMOPARAM>=0 || !defined(COSMOPARAM)
double friedmann_solver_start(double a0, double t0, double h, double a_start)
{
	//In this function we calculate the time of Big Bang, and the initial time for the simulation.
	printf("Calculating time for the initial scale factor...\n");
	int i=0;
	double b, t_cosm_tmp;
	double b_tmp[2];
	double t1, t2;
	double b2perb1pow1A, Hubble_b, Omega_m_b, Omega_r_b;
	double t_cosm;
	double t_start_err=1e-10;
	int iteration_limit=20;
	double h_var;
	h_var = h;
	t_cosm = t0;
	t_cosm_tmp = t0+1.0;
	//Integrating the Friedmann equations
	while(fabs(t_cosm-t_cosm_tmp)>t_start_err && i<iteration_limit)
	{
		t_cosm_tmp = t_cosm;
		t_cosm = t0;
		b=a_start;
		b_tmp[0]=b;
		b_tmp[1]=b;
		while(0<b)
		{
			b_tmp[1]=b_tmp[0];
			b_tmp[0]=b;
			b = friedmann_solver_step(b, -h_var);
			t_cosm -= h_var;
		}
		//using extrapolation for calculating t_cosm(b=0)
		//We are using the last two positive scale factor value for this extrapolation
		//here we assume that b(t) = (H0 * (t-t_bigbang))^A, where
		//A=1/2 for radiation dominated universe, and A=2/3 for matter dominated universe
		//if b1 and b2 are the scale factor in t1 and t2 times respectively, than t_bigbang can be written as
		//t_bigbang = [(b2/b1)^(1/A) * t1 - t2]/[(b2/b1)^(1/A) - 1]
		Hubble_b = CALCULATE_Hubble_param(b_tmp[0]);
		Omega_m_b = Omega_m*pow(b_tmp[0], -3)*pow(H0/Hubble_b, 2);
		Omega_r_b = Omega_r*pow(b_tmp[0], -4)*pow(H0/Hubble_b, 2);
		if(Omega_m_b > Omega_r_b)
		{
			//Matter dominated universe at the last step of this integration.
			//This means that Omega_r were set to zero, or h_var is too large.
			b2perb1pow1A = pow(b_tmp[1]/b_tmp[0], 1.5);
		}
		else
		{
			//Radiation dominated age.
			b2perb1pow1A = pow(b_tmp[1]/b_tmp[0], 2.0);
		}
		t2=t_cosm + 2*h_var;
		t1=t_cosm + h_var;
		//printf(" t_lastpos=%ey, b_lastpos=%e, 1-Omb=%e", t1*UNIT_T*1e9, b_tmp[0], 1-Omega_m_b);
		t_cosm = (b2perb1pow1A*t1-t2)/(b2perb1pow1A-1);
		//printf(" t_cosm=%fy, err=%fy\n", t_cosm*UNIT_T*1e9, fabs(t_cosm_tmp-t_cosm)*UNIT_T*1e9);
		h_var *= 0.5;
		i++;
	}
	printf("...done. After %i iteration, the calculated initial time is %fy. (h_var=%fy,  err=%fy)\n\n", i, -1.0*t_cosm*UNIT_T*1e9, h_var*2.0*UNIT_T*1e9, (fabs(t_cosm-t_cosm_tmp))*UNIT_T*1e9);
	//Setting t=0 to Big Bang:
	return -1.0*t_cosm;
}
#endif

#if !defined(COSMOPARAM) || COSMOPARAM == 0
//COSMOPARAM == 0: standard LCDM parametrization
	double friedmann_solver_step(double a0, double h)
	{
		//4th order Runge-Kutta integrator
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

	double CALCULATE_Hubble_param(double a)
	{
		return H0*sqrt(Omega_m*pow(a, -3)+Omega_r*pow(a, -4)+Omega_lambda+Omega_k*pow(a, -2));
	}

	double CALCULATE_decel_param(double a)
	{
		double SUM_OMEGA_TMP = Omega_m*pow(a, -3.0) + Omega_r*pow(a, -4.0) + Omega_lambda + Omega_k*pow(a, -2.0);
		double Omega_m_tmp = Omega_m*pow(a, -3.0)/SUM_OMEGA_TMP;
		double Omega_r_tmp = Omega_r*pow(a, -4.0)/SUM_OMEGA_TMP;
		double Omega_lambda_tmp = Omega_lambda/SUM_OMEGA_TMP;
		return 0.5*Omega_m_tmp + Omega_r_tmp - Omega_lambda_tmp;
	}
#elif COSMOPARAM==1
	//COSMOPARAM == 1: wCDM dark energy parametrization
	double friedmann_solver_step(double a0, double h)
	{
		//4th order Runge-Kutta integrator
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
			j=Omega_m*pow(b, -3.0)+Omega_r*pow(b, -4.0)+Omega_lambda*pow(b, -3.0*(1.0+w0));
			k1 = b*H0*sqrt(Omega_m*pow(b, -3.0)+Omega_r*pow(b, -4.0)+Omega_lambda*pow(b, -3.0*(1.0+w0)));
			l=Omega_m*pow((b+h*k1/2.0), -3.0)+Omega_r*pow((b+h*k1/2), -4.0)+Omega_lambda*pow((b+h*k1/2.0), -3.0*(1.0+w0));
			k2 = (b+h*k1/2.0)*H0*sqrt(Omega_m*pow((b+h*k1/2.0), -3.0)+Omega_r*pow((b+h*k1/2), -4.0)+Omega_lambda*pow((b+h*k1/2.0), -3.0*(1.0+w0)));
			m=Omega_m*pow((b+h*k2/2.0), -3.0)+Omega_r*pow((b+h*k2/2), -4.0)+Omega_lambda*pow((b+h*k2/2.0), -3.0*(1.0+w0));
			k3 = (b+h*k2/2.0)*H0*sqrt(Omega_m*pow((b+h*k2/2.0), -3.0)+Omega_r*pow((b+h*k2/2), -4.0)+Omega_lambda*pow((b+h*k2/2.0), -3.0*(1.0+w0)));
			n=Omega_m*pow((b+h*k3), -3.0)+Omega_r*pow((b+h*k3), -4.0)+Omega_lambda*pow((b+h*k3), -3.0*(1.0+w0));
			k4 = (b+h*k3)*H0*sqrt(Omega_m*pow((b+h*k3), -3.0)+Omega_r*pow((b+h*k3), -4.0)+Omega_lambda*pow((b+h*k3), -3.0*(1.0+w0)));
			K = h*(k1+k2*2.0+k3*2.0+k4)/6.0;
			if(j<0||l<0||m<0||n<0)
				b=-1;
			else
				b += K;
		}
		else
		{
			j= Omega_m*pow(b, -3.0)+Omega_r*pow(b, -4.0)+Omega_lambda*pow(b, -3.0*(1.0+w0))+Omega_k*pow(b, -2.0);
			k1 = b*H0*sqrt(fabs(j));
			l = Omega_m*pow((b+h*k1/2.0), -3.0)+Omega_r*pow((b+h*k1/2.0), -4.0)+Omega_lambda*pow((b+h*k1/2.0), -3.0*(1.0+w0))+Omega_k*pow((b+h*k1/2.0), -2.0);
			k2 = (b+h*k1/2.0)*H0*sqrt(fabs(l));
			m = Omega_m*pow((b+h*k2/2.0), -3.0)+Omega_r*pow((b+h*k2/2.0), -4.0)+Omega_lambda*pow((b+h*k2/2.0), -3.0*(1.0+w0))+Omega_k*pow((b+h*k2/2.0), -2.0);
			k3 = (b+h*k2/2.0)*H0*sqrt(fabs(m));
			n=Omega_m*pow((b+h*k3), -3.0)+Omega_r*pow((b+h*k3), -4.0)+Omega_lambda*pow((b+h*k3), -3.0*(1.0+w0))+Omega_k*pow((b+h*k3), -2.0);
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

	double CALCULATE_Hubble_param(double a)
	{
		return H0*sqrt(Omega_m*pow(a, -3)+Omega_r*pow(a, -4)+Omega_lambda*pow(a, -3.0*(1.0+w0))+Omega_k*pow(a, -2));
	}

	double CALCULATE_decel_param(double a)
	{
		double SUM_OMEGA_TMP = Omega_m*pow(a, -3.0) + Omega_r*pow(a, -4.0) + Omega_lambda*pow(a, -3.0*(1.0+w0)) + Omega_k*pow(a, -2.0);
		double Omega_m_tmp = Omega_m*pow(a, -3.0)/SUM_OMEGA_TMP;
		double Omega_r_tmp = Omega_r*pow(a, -4.0)/SUM_OMEGA_TMP;
		double Omega_lambda_tmp = Omega_lambda*pow(a, -3.0*(1.0+w0))/SUM_OMEGA_TMP;
		return 0.5*Omega_m_tmp + Omega_r_tmp + 0.5*(1.0+3.0*w0)*Omega_lambda_tmp;
	}
#elif COSMOPARAM==2
	//w0waCDM dark energy parametrization
	//The dark energy equation of state uses the CPL form as described
	//by Chevallier, Polarski and Linder:
	//w(a) = w0 + wa*(1-a)
	double friedmann_solver_step(double a0, double h)
	{
		//4th order Runge-Kutta integrator
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
			j=Omega_m*pow(b, -3.0)+Omega_r*pow(b, -4.0)+Omega_lambda*pow(b, -3.0*(1.0+wa+w0))*exp(-3.0*wa*(1.0-b));
			k1 = b*H0*sqrt(Omega_m*pow(b, -3.0)+Omega_r*pow(b, -4.0)+Omega_lambda*pow(b, -3.0*(1.0+wa+w0))*exp(-3.0*wa*(1.0-b)));

			l=Omega_m*pow((b+h*k1/2.0), -3.0)+Omega_r*pow((b+h*k1/2), -4.0)+Omega_lambda*pow((b+h*k1/2.0), -3.0*(1.0+wa+w0))*exp(-3.0*wa*(1.0-(b+h*k1/2.0)));
			k2 = (b+h*k1/2.0)*H0*sqrt(Omega_m*pow((b+h*k1/2.0), -3.0)+Omega_r*pow((b+h*k1/2), -4.0)+Omega_lambda*pow((b+h*k1/2.0), -3.0*(1.0+wa+w0))*exp(-3.0*wa*(1.0-(b+h*k1/2.0))));

			m=Omega_m*pow((b+h*k2/2.0), -3.0)+Omega_r*pow((b+h*k2/2), -4.0)+Omega_lambda*pow((b+h*k2/2.0), -3.0*(1.0+wa+w0))*exp(-3.0*wa*(1.0-(b+h*k2/2.0)));
			k3 = (b+h*k2/2.0)*H0*sqrt(Omega_m*pow((b+h*k2/2.0), -3.0)+Omega_r*pow((b+h*k2/2), -4.0)+Omega_lambda*pow((b+h*k2/2.0), -3.0*(1.0+wa+w0))*exp(-3.0*wa*(1.0-(b+h*k2/2.0))));

			n=Omega_m*pow((b+h*k3), -3.0)+Omega_r*pow((b+h*k3), -4.0)+Omega_lambda*pow((b+h*k3), -3.0*(1.0+wa+w0))*exp(-3.0*wa*(1.0-(b+h*k3)));
			k4 = (b+h*k3)*H0*sqrt(Omega_m*pow((b+h*k3), -3.0)+Omega_r*pow((b+h*k3), -4.0)+Omega_lambda*pow((b+h*k3), -3.0*(1.0+wa+w0))*exp(-3.0*wa*(1.0-(b+h*k3))));
			K = h*(k1+k2*2.0+k3*2.0+k4)/6.0;
			if(j<0||l<0||m<0||n<0)
				b=-1;
			else
				b += K;
		}
		else
		{
			j= Omega_m*pow(b, -3.0)+Omega_r*pow(b, -4.0)+Omega_lambda*pow(b, -3.0*(1.0+wa+w0))*exp(-3.0*wa*(1.0-b))+Omega_k*pow(b, -2.0);
			k1 = b*H0*sqrt(fabs(j));

			l = Omega_m*pow((b+h*k1/2.0), -3.0)+Omega_r*pow((b+h*k1/2.0), -4.0)+Omega_lambda*pow((b+h*k1/2.0), -3.0*(1.0+wa+w0))*exp(-3.0*wa*(1.0-(b+h*k1/2.0)))+Omega_k*pow((b+h*k1/2.0), -2.0);
			k2 = (b+h*k1/2.0)*H0*sqrt(fabs(l));

			m = Omega_m*pow((b+h*k2/2.0), -3.0)+Omega_r*pow((b+h*k2/2.0), -4.0)+Omega_lambda*pow((b+h*k2/2.0), -3.0*(1.0+wa+w0))*exp(-3.0*wa*(1.0-(b+h*k2/2.0)))+Omega_k*pow((b+h*k2/2.0), -2.0);
			k3 = (b+h*k2/2.0)*H0*sqrt(fabs(m));

			n=Omega_m*pow((b+h*k3), -3.0)+Omega_r*pow((b+h*k3), -4.0)+Omega_lambda*pow((b+h*k3), -3.0*(1.0+wa+w0))*exp(-3.0*wa*(1.0-(b+h*k3)))+Omega_k*pow((b+h*k3), -2.0);
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

	double CALCULATE_Hubble_param(double a)
	{
		return H0*sqrt(Omega_m*pow(a, -3)+Omega_r*pow(a, -4)+Omega_lambda*pow(a, -3.0*(1.0+wa+w0))*exp(-3.0*wa*(1.0-a))+Omega_k*pow(a, -2));
	}

	double CALCULATE_decel_param(double a)
	{
		double SUM_OMEGA_TMP = Omega_m*pow(a, -3.0) + Omega_r*pow(a, -4.0) + Omega_lambda*pow(a, -3.0*(1.0+wa+w0))*exp(-3.0*wa*(1.0-a)) + Omega_k*pow(a, -2.0);
		double Omega_m_tmp = Omega_m*pow(a, -3.0)/SUM_OMEGA_TMP;
		double Omega_r_tmp = Omega_r*pow(a, -4.0)/SUM_OMEGA_TMP;
		double Omega_lambda_tmp = Omega_lambda*pow(a, -3.0*(1.0+wa+w0))*exp(-3.0*wa*(1.0-a))/SUM_OMEGA_TMP;
		return 0.5*Omega_m_tmp + Omega_r_tmp + 0.5*(1.0+3.0*(w0 + wa*(1.0-a)))*Omega_lambda_tmp;
	}
#elif COSMOPARAM==-1
	//using tabulated expansion history
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

	double friedmann_solver_step(double a0, double h)
	{
		int index = expansion_index-2;
		double T_prev = T-h;
		do{
			index++;
			if(index==N_expansion_tab)
			{
				index--;
				break;
			}
		}while(expansion_tab[index][0]<=T_prev);
		expansion_index = index;
		switch (INTERPOLATION_ORDER)
		{
			case 1:
				if(expansion_index<1)
					expansion_index=1;
				else if(expansion_index>=N_expansion_tab)
					expansion_index=N_expansion_tab-1;
				return linear_interpolation(T,expansion_tab[expansion_index-1][0],expansion_tab[expansion_index-1][1],expansion_tab[expansion_index][0],expansion_tab[expansion_index][1]);
			case 2:
				if(expansion_index<2)
					expansion_index=2;
				else if(expansion_index>=N_expansion_tab)
					expansion_index=N_expansion_tab-1;
				return quadratic_interpolation(T,expansion_tab[expansion_index-2][0],expansion_tab[expansion_index-2][1], expansion_tab[expansion_index-1][0],expansion_tab[expansion_index-1][1],expansion_tab[expansion_index][0],expansion_tab[expansion_index][1]);
			case 3:
				if(expansion_index<2)
					expansion_index=2;
				else if(expansion_index>=N_expansion_tab-1)
					expansion_index=N_expansion_tab-2;
				return cubic_interpolation(T,expansion_tab[expansion_index-2][0],expansion_tab[expansion_index-2][1], expansion_tab[expansion_index-1][0],expansion_tab[expansion_index-1][1],expansion_tab[expansion_index][0],expansion_tab[expansion_index][1],expansion_tab[expansion_index+1][0],expansion_tab[expansion_index+1][1]);
			default:
				return expansion_tab[expansion_index-1][1];
		}
	}

	double CALCULATE_Hubble_param(double a)
	{
		int index = expansion_index-2;
		do{
			index++;
			if(index==N_expansion_tab)
			{
				index--;
				break;
			}
		}while(expansion_tab[index][1]<=a);
		switch (INTERPOLATION_ORDER)
		{
			case 1:
				if(index<1)
					index=1;
				else if(index>=N_expansion_tab)
					index=N_expansion_tab-1;
				return linear_interpolation(a,expansion_tab[index-1][1],expansion_tab[index-1][2],expansion_tab[index][1],expansion_tab[index][2]);
			case 2:
				if(index<2)
					index=2;
				else if(index>=N_expansion_tab)
					index=N_expansion_tab-1;
				return quadratic_interpolation(a,expansion_tab[index-2][1],expansion_tab[index-2][2],expansion_tab[index-1][1],expansion_tab[index-1][2],expansion_tab[index][1],expansion_tab[index][2]);
			case 3:
				if(index<2)
					index=2;
				else if(index>=N_expansion_tab-2)
					index=N_expansion_tab-2;
				return cubic_interpolation(a,expansion_tab[index-2][1],expansion_tab[index-2][2],expansion_tab[index-1][1],expansion_tab[index-1][2],expansion_tab[index][1],expansion_tab[index][2],expansion_tab[index+1][1],expansion_tab[index+1][2]);
			default:
				return expansion_tab[expansion_index-1][2];
		}
	}

	double CALCULATE_decel_param(double a)
	{
		//numeric approximation of the deceleration parameter
		int index = expansion_index-2;
		do{
			index++;
			if(index==N_expansion_tab)
			{
				index--;
				break;
			}
		}while(expansion_tab[index][1]<=a);
		if(index<1)
			index=1;
		else if(index>=N_expansion_tab)
			index=N_expansion_tab-1;
		double dHdt = (expansion_tab[index][2]-expansion_tab[index-1][2])/((expansion_tab[index][0]-expansion_tab[index-1][0]));
		double H_tmp = CALCULATE_Hubble_param(a);
		return -1.0*(dHdt/H_tmp/H_tmp)-1.0;
	}

	double friedmann_solver_start(double a0, double t0, double h, double a_start)
	{
		printf("Calculating time for the initial scale factor...\n");
		//setting up the initial index
		expansion_index = 0;
		do{
			expansion_index++;
			printf("expansion_index: %i\t a_tabbed: %f\n", expansion_index,expansion_tab[expansion_index][1]);
		}while(expansion_tab[expansion_index][1]<=a_start);
		double t_start;
		switch (INTERPOLATION_ORDER)
		{
			case 1:
				if(expansion_index<1)
					expansion_index=1;
				t_start = linear_interpolation(a_start,expansion_tab[expansion_index-1][1],expansion_tab[expansion_index-1][0],expansion_tab[expansion_index][1],expansion_tab[expansion_index][0]);
			case 2:
				if(expansion_index<2)
					expansion_index=2;
				t_start = quadratic_interpolation(a_start,expansion_tab[expansion_index-2][1],expansion_tab[expansion_index-2][0],expansion_tab[expansion_index-1][1],expansion_tab[expansion_index-1][0],expansion_tab[expansion_index][1],expansion_tab[expansion_index][0]);
			case 3:
				if(expansion_index<2)
					expansion_index=2;
				t_start = cubic_interpolation(a_start,expansion_tab[expansion_index-2][1],expansion_tab[expansion_index-2][0],expansion_tab[expansion_index-1][1],expansion_tab[expansion_index-1][0],expansion_tab[expansion_index][1],expansion_tab[expansion_index][0],expansion_tab[expansion_index+1][1],expansion_tab[expansion_index+1][0]);
			default:
				t_start = expansion_tab[expansion_index-1][0];
		}
		printf("...done. After interpolating between %i and %i row, the calculated initial time is %fy.\n\n", expansion_index-1, expansion_index, t_start*UNIT_T*1e9);
		return t_start;
	}
#endif

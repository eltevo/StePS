/*  Unit tests for the intrinsic S^3 kinematics helpers in src/pds_group.h
 *  (Phase 1 geodesic integrator).  No simulation, no MPI, no HDF5.
 *
 *  Build & run:
 *      g++ -O2 -I../../src -o test_intrinsic_kinematics test_intrinsic_kinematics.cc -lm
 *      ./test_intrinsic_kinematics
 *
 *  These pin the invariants the integrator relies on:
 *    - exp map preserves |q| = 1, U . q = 0, and |U|
 *    - exp map reproduces the analytic great circle
 *    - composing exp maps equals one exp map of the summed time (geodesic flow)
 *    - parallel transport around a closed loop returns the original tangent
 *    - the stereographic velocity conversions round-trip
 *    - the factored conversions agree with pds_stereo_vel_transform()
 *    - isometries commute with the exp map (the wrap must not change trajectories)
 */
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "pds_group.h"

static int failures = 0;
static int checks   = 0;

static void check(int cond, const char *name, double measured, double tol)
{
    checks++;
    if(!cond) { failures++; printf("  [FAIL] %-52s %.3e > %.1e\n", name, measured, tol); }
    else      {             printf("  [PASS] %-52s %.3e\n",       name, measured); }
}

/* deterministic pseudo-random in [-1,1) */
static unsigned long long _s = 88172645463325252ULL;
static double rnd(void)
{
    _s ^= _s << 13; _s ^= _s >> 7; _s ^= _s << 17;
    return 2.0*((double)(_s >> 11) / 9007199254740992.0) - 1.0;
}

static void rand_state(double q[4], double U[4], double speed)
{
    double n = 0.0;
    for(int k = 0; k < 4; k++) { q[k] = rnd(); n += q[k]*q[k]; }
    n = sqrt(n); for(int k = 0; k < 4; k++) q[k] /= n;
    for(int k = 0; k < 4; k++) U[k] = rnd();
    pds_project_tangent(q, U);
    double m = sqrt(pds_dot4(U, U));
    for(int k = 0; k < 4; k++) U[k] *= speed/m;
}

int main(void)
{
    const double R = 3100.0;
    printf("Intrinsic S^3 kinematics (R = %.1f Mpc)\n", R);
    printf("---------------------------------------------------------------------\n");

    /* 1. exp map invariants over a wide range of step sizes */
    {
        double worst_norm = 0, worst_perp = 0, worst_speed = 0;
        for(int t = 0; t < 20000; t++)
        {
            double q[4], U[4], qn[4], Un[4];
            double speed = pow(10.0, -3.0 + 5.0*0.5*(rnd()+1.0));   /* 1e-3 .. 1e2 */
            rand_state(q, U, speed);
            double h = pow(10.0, -4.0 + 6.0*0.5*(rnd()+1.0));       /* 1e-4 .. 1e2 */
            pds_exp_map(q, U, h, R, qn, Un);

            double dn = fabs(sqrt(pds_dot4(qn, qn)) - 1.0);
            double dp = fabs(pds_dot4(qn, Un)) / speed;
            double ds = fabs(sqrt(pds_dot4(Un, Un)) - speed) / speed;
            if(dn > worst_norm)  worst_norm  = dn;
            if(dp > worst_perp)  worst_perp  = dp;
            if(ds > worst_speed) worst_speed = ds;
        }
        check(worst_norm  < 1e-14, "exp map keeps |q| = 1",        worst_norm,  1e-14);
        check(worst_perp  < 1e-14, "exp map keeps U . q = 0",      worst_perp,  1e-14);
        check(worst_speed < 1e-13, "exp map conserves |U|",        worst_speed, 1e-13);
    }

    /* 2. matches the analytic great circle */
    {
        double q[4] = {1,0,0,0}, U[4] = {0,1,0,0};
        double speed = 50.0;
        for(int k = 0; k < 4; k++) U[k] *= speed;
        double h = 7.3, qn[4], Un[4];
        pds_exp_map(q, U, h, R, qn, Un);
        double th = speed*h/R;
        double ref[4] = {cos(th), sin(th), 0, 0};
        double e = 0; for(int k = 0; k < 4; k++) e = fmax(e, fabs(qn[k]-ref[k]));
        check(e < 1e-15, "exp map reproduces the analytic great circle", e, 1e-15);
    }

    /* 3. geodesic flow: exp(h1) then exp(h2) == exp(h1+h2) */
    {
        double worst = 0;
        for(int t = 0; t < 5000; t++)
        {
            double q[4], U[4], a[4], Ua[4], b[4], Ub[4], c[4], Uc[4];
            rand_state(q, U, 1.0 + 40.0*0.5*(rnd()+1.0));
            double h1 = 0.5 + 3.0*0.5*(rnd()+1.0), h2 = 0.5 + 3.0*0.5*(rnd()+1.0);
            pds_exp_map(q, U, h1, R, a, Ua);
            pds_exp_map(a, Ua, h2, R, b, Ub);
            pds_exp_map(q, U, h1+h2, R, c, Uc);
            for(int k = 0; k < 4; k++) worst = fmax(worst, fabs(b[k]-c[k]));
        }
        check(worst < 1e-13, "exp(h1) o exp(h2) == exp(h1+h2)", worst, 1e-13);
    }

    /* 4. transport around a closed loop (full great circle) is the identity */
    {
        double worst = 0;
        for(int t = 0; t < 2000; t++)
        {
            double q[4], U[4], qn[4], Un[4];
            double speed = 1.0 + 40.0*0.5*(rnd()+1.0);
            rand_state(q, U, speed);
            double h_full = 2.0*M_PI*R/speed;             /* one full circuit */
            pds_exp_map(q, U, h_full, R, qn, Un);
            for(int k = 0; k < 4; k++)
                worst = fmax(worst, fmax(fabs(qn[k]-q[k]), fabs(Un[k]-U[k])/speed));
        }
        check(worst < 1e-10, "closed-loop transport is the identity", worst, 1e-10);
    }

    /* 5. stereographic velocity conversions round-trip */
    {
        double worst = 0;
        for(int t = 0; t < 20000; t++)
        {
            /* a point inside the fundamental domain, |x| <= 0.1584 R */
            double x[3], n = 0;
            for(int k = 0; k < 3; k++) { x[k] = rnd(); n += x[k]*x[k]; }
            n = sqrt(n);
            double rr = 0.1584*R*0.5*(rnd()+1.0);
            for(int k = 0; k < 3; k++) x[k] *= rr/n;
            double v[3] = {rnd()*300.0, rnd()*300.0, rnd()*300.0};

            double r2 = x[0]*x[0]+x[1]*x[1]+x[2]*x[2], D = R*R+r2;
            double q[4] = {(R*R-r2)/D, 2*R*x[0]/D, 2*R*x[1]/D, 2*R*x[2]/D};

            double U[4]; pds_tangent_from_stereo_vel(x, v, R, U);
            double perp = fabs(pds_dot4(q, U));
            double vb[3]; pds_stereo_vel_from_tangent(q, x, U, R, vb);
            for(int k = 0; k < 3; k++)
                worst = fmax(worst, fabs(vb[k]-v[k])/300.0);
            worst = fmax(worst, perp/300.0);
        }
        check(worst < 1e-13, "stereo velocity <-> tangent round-trip", worst, 1e-13);
    }

    /* 6. the factored helpers reproduce pds_stereo_vel_transform() */
    {
        double worst = 0;
        for(int t = 0; t < 5000; t++)
        {
            double x[3], n = 0;
            for(int k = 0; k < 3; k++) { x[k] = rnd(); n += x[k]*x[k]; }
            n = sqrt(n);
            double rr = 0.1584*R*0.5*(rnd()+1.0);
            for(int k = 0; k < 3; k++) x[k] *= rr/n;
            double r2 = x[0]*x[0]+x[1]*x[1]+x[2]*x[2], D = R*R+r2;
            double q_in[4] = {(R*R-r2)/D, 2*R*x[0]/D, 2*R*x[1]/D, 2*R*x[2]/D};
            double v[3] = {rnd()*300.0, rnd()*300.0, rnd()*300.0};

            /* pick a non-trivial group element */
            int g = 1 + (int)(((unsigned)(rnd()*1e6)) % 119u);
            double q_out[4]; pds_apply_group_element(g, q_in, q_out);
            if(1.0 + q_out[0] < 1e-6) continue;                  /* skip near-antipode */
            double s = 1.0/(1.0+q_out[0]);
            double x_out[3] = {R*q_out[1]*s, R*q_out[2]*s, R*q_out[3]*s};

            double v_ref[3] = {v[0], v[1], v[2]};
            pds_stereo_vel_transform(q_in, q_out, x, x_out, v_ref, R);

            /* same thing via the intrinsic route: to tangent, rotate, back */
            double U[4]; pds_tangent_from_stereo_vel(x, v, R, U);
            double gq[4]; pds_quat_conj(q_in, gq);
            double gb[4]; pds_quat_mult(q_out, gq, gb);
            double Ur[4]; pds_rotate_tangent(gb, U, Ur);
            double v_new[3]; pds_stereo_vel_from_tangent(q_out, x_out, Ur, R, v_new);

            for(int k = 0; k < 3; k++)
                worst = fmax(worst, fabs(v_new[k]-v_ref[k])/300.0);
        }
        check(worst < 1e-11, "factored helpers == pds_stereo_vel_transform", worst, 1e-11);
    }

    /* 7. isometries commute with the exp map: wrapping must not bend trajectories */
    {
        double worst = 0;
        for(int t = 0; t < 5000; t++)
        {
            double q[4], U[4];
            rand_state(q, U, 1.0 + 40.0*0.5*(rnd()+1.0));
            double h = 0.5 + 5.0*0.5*(rnd()+1.0);
            int g = 1 + (int)(((unsigned)(rnd()*1e6)) % 119u);

            /* evolve then rotate */
            double a[4], Ua[4], ra[4], rUa[4];
            pds_exp_map(q, U, h, R, a, Ua);
            pds_apply_group_element(g, a, ra);
            pds_rotate_tangent(PDS_I_STAR[g], Ua, rUa);

            /* rotate then evolve */
            double rq[4], rU[4], b[4], Ub[4];
            pds_apply_group_element(g, q, rq);
            pds_rotate_tangent(PDS_I_STAR[g], U, rU);
            pds_exp_map(rq, rU, h, R, b, Ub);

            for(int k = 0; k < 4; k++)
                worst = fmax(worst, fabs(ra[k]-b[k]));
        }
        check(worst < 1e-13, "isometries commute with the exp map", worst, 1e-13);
    }

    printf("---------------------------------------------------------------------\n");
    printf("%d/%d checks passed\n", checks-failures, checks);
    return failures ? 1 : 0;
}

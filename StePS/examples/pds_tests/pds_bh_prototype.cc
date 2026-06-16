/*****************************************************************************
 *  pds_bh_prototype.cc — CPU Barnes-Hut prototype for the PDS (S^3/I*) force
 *
 *  Step-1 feasibility prototype (see docs/PDS_guide.md "BH for PDS").
 *  It does NOT touch the StePS build; it is a standalone validator that
 *  measures the accuracy and speed of a Barnes-Hut tree force against the
 *  exact 120-image summation used by forces_pds()/ForceKernel_pds.
 *
 *  Algorithm (single tree walk per field particle, all 120 images per node):
 *    - Build an octree in the stereographic Cartesian coordinates x[] (the
 *      same coordinates StePS drifts in).  Inside the fundamental domain the
 *      stereographic map is conformal with a nearly constant scale factor
 *      (Omega varies < 4% across the domain), so the Euclidean opening-angle
 *      test is geometrically faithful to the S^3 geometry.
 *    - Walk the tree once with the standard nodesize/dist < THETA criterion
 *      (distance in stereo coords, i.e. the identity image).  When a node is
 *      accepted, add the contribution of ALL 120 I* images of the node's
 *      centre of mass with the background-compensated geodesic kernel.  A node
 *      compact enough to be a monopole for the (near) identity image is an
 *      even better monopole for the 119 (far) images, so one opening test and
 *      one set of node multipoles covers the whole image system.
 *      => cost ~ N * (accepted nodes ~ log N) * 120  vs  direct N * N * 120.
 *
 *  Build (inside the stepsic conda env):
 *    g++ -O3 -std=c++17 -fopenmp pds_bh_prototype.cc \
 *        -I$CONDA_PREFIX/include -L$CONDA_PREFIX/lib -lhdf5 \
 *        -Wl,-rpath,$CONDA_PREFIX/lib -o pds_bh_prototype
 *
 *  Run:
 *    ./pds_bh_prototype <snapshot.hdf5> [particle_radii] [n_sample]
 *
 *  Copyright (C) 2026 Gabor Racz, Istvan Csabai.  GPL v2 or later.
 *****************************************************************************/

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include <string>
#include <algorithm>
#include <random>
#include <hdf5.h>
#include <omp.h>

#include "../../src/pds_group.h"   /* I* group, geodesic distance, kernels */

/* ───────────────────────── data ───────────────────────── */
static int     N = 0;
static double  R_CURV = 3100.0;
static std::vector<double> X;       /* 3*N stereographic coordinates  */
static std::vector<double> Q;       /* 4*N unit quaternions           */
static std::vector<double> M;       /* N masses                       */
static std::vector<double> SOFT;    /* N softening lengths            */

/* ───────────────────── stereographic map ──────────────── */
static inline void stereo_to_quat(double cx, double cy, double cz, double q[4])
{
    double r2 = cx*cx + cy*cy + cz*cz;
    double d  = R_CURV*R_CURV + r2;
    q[0] = (R_CURV*R_CURV - r2) / d;
    q[1] = 2.0*R_CURV*cx / d;
    q[2] = 2.0*R_CURV*cy / d;
    q[3] = 2.0*R_CURV*cz / d;
}

/* ─────────────────────── octree ────────────────────────── */
struct Node {
    double cx, cy, cz, nodesize;
    double mass, com_x, com_y, com_z, soft;
    int    particle_index;          /* -1 if internal */
    Node*  child[8];
};

static Node* make_node(double cx, double cy, double cz, double s)
{
    Node* n = new Node();
    n->cx = cx; n->cy = cy; n->cz = cz; n->nodesize = s;
    n->mass = n->com_x = n->com_y = n->com_z = n->soft = 0.0;
    n->particle_index = -1;
    for(int i=0;i<8;i++) n->child[i] = nullptr;
    return n;
}

static inline int octant(const Node* nd, int i)
{
    int idx = 0;
    if(X[3*i]   > nd->cx) idx |= 1;
    if(X[3*i+1] > nd->cy) idx |= 2;
    if(X[3*i+2] > nd->cz) idx |= 4;
    return idx;
}

static void insert(Node* nd, int i)
{
    if(nd->mass == 0.0 && nd->particle_index == -1) {
        nd->particle_index = i;
        nd->mass  = M[i];
        nd->com_x = X[3*i]; nd->com_y = X[3*i+1]; nd->com_z = X[3*i+2];
        nd->soft  = SOFT[i];
        return;
    }
    if(nd->particle_index != -1) {        /* split an occupied leaf */
        int e = nd->particle_index;
        nd->particle_index = -1;
        for(int j=0;j<8;j++) {
            double o = nd->nodesize/4.0;
            nd->child[j] = make_node(nd->cx + ((j&1)?o:-o),
                                     nd->cy + ((j&2)?o:-o),
                                     nd->cz + ((j&4)?o:-o),
                                     nd->nodesize/2.0);
        }
        insert(nd->child[octant(nd,e)], e);
    }
    insert(nd->child[octant(nd,i)], i);
    double tm = nd->mass + M[i];
    nd->com_x = (nd->com_x*nd->mass + X[3*i]  *M[i]) / tm;
    nd->com_y = (nd->com_y*nd->mass + X[3*i+1]*M[i]) / tm;
    nd->com_z = (nd->com_z*nd->mass + X[3*i+2]*M[i]) / tm;
    nd->soft  = (nd->soft *nd->mass + SOFT[i] *M[i]) / tm;
    nd->mass  = tm;
}

/* ───────────────── exact 120-image force ───────────────── */
static void force_exact(int i, double F[3])
{
    F[0]=F[1]=F[2]=0.0;
    const double qi[4] = {Q[4*i],Q[4*i+1],Q[4*i+2],Q[4*i+3]};
    for(int j=0;j<N;j++) {
        const double qj[4] = {Q[4*j],Q[4*j+1],Q[4*j+2],Q[4*j+3]};
        double chi_soft = (SOFT[i]+SOFT[j]) / R_CURV;
        for(int g=0; g<PDS_N_ISTAR; g++) {
            double qg[4]; pds_apply_group_element(g, qj, qg);
            double chi = pds_chi(qi, qg);
            if(chi < 1e-12 || chi > M_PI - 1e-12) continue;
            double t[4]; pds_force_direction(qi, qg, t);
            double ce = (chi < chi_soft) ? chi_soft : chi;
            double fm = M[j] * pds_green_compensated(ce, R_CURV);
            F[0]+=fm*t[1]; F[1]+=fm*t[2]; F[2]+=fm*t[3];
        }
    }
}

/* ─────────────── Barnes-Hut 120-image force ────────────────────────────────
 *
 *  The opening test MUST be evaluated separately for each I* image g, in the
 *  S^3 geodesic metric.  A node that is far in the identity image can be
 *  *adjacent* to the field particle in an image that shares a dodecahedral face
 *  (the image IS the physical neighbour across the face); lumping such an image
 *  as a monopole is catastrophically wrong.  So we descend the tree once per
 *  image, using the geodesic distance chi(qi, g·q_C) of that image.  Genuinely
 *  far images (most of the 119) terminate at shallow levels, so the cost is far
 *  below 120 deep walks — only the identity and the ~12 face-adjacent images
 *  descend deep.
 *
 *  Node angular size on S^3: the stereographic map is conformal with scale
 *  Omega(r) = 2R^2/(R^2+r^2) (ds_geodesic = Omega·ds_stereo), so a cube of side
 *  `nodesize` at stereo radius r_C subtends a geodesic angle
 *      ang = Omega(r_C)·nodesize / R = 2R·nodesize/(R^2 + r_C^2).
 *  Isometries preserve it, so the imaged node has the same angular size.       */
static void force_bh_image(Node* nd, const double qi[4], double soft_i,
                           int g, double theta2, double F[3], long& nnodes)
{
    if(nd == nullptr || nd->mass == 0.0) return;

    double r2 = nd->com_x*nd->com_x + nd->com_y*nd->com_y + nd->com_z*nd->com_z;
    double qC[4]; stereo_to_quat(nd->com_x, nd->com_y, nd->com_z, qC);
    double qg[4]; pds_apply_group_element(g, qC, qg);
    double chi = pds_chi(qi, qg);
    double ang = 2.0*R_CURV*nd->nodesize / (R_CURV*R_CURV + r2);   /* geodesic angular size */

    if(nd->particle_index != -1 || ang*ang < theta2*chi*chi) {
        nnodes++;
        if(chi < 1e-12 || chi > M_PI - 1e-12) return;   /* identity self / antipode */
        double t[4]; pds_force_direction(qi, qg, t);
        double chi_soft = (soft_i + nd->soft) / R_CURV;
        double ce = (chi < chi_soft) ? chi_soft : chi;
        double fm = nd->mass * pds_green_compensated(ce, R_CURV);
        F[0]+=fm*t[1]; F[1]+=fm*t[2]; F[2]+=fm*t[3];
        return;
    }
    for(int j=0;j<8;j++) force_bh_image(nd->child[j], qi, soft_i, g, theta2, F, nnodes);
}

static void force_bh(Node* tree, int i, double theta2, double F[3], long& nnodes)
{
    F[0]=F[1]=F[2]=0.0;
    const double qi[4] = {Q[4*i],Q[4*i+1],Q[4*i+2],Q[4*i+3]};
    for(int g=0; g<PDS_N_ISTAR; g++)
        force_bh_image(tree, qi, SOFT[i], g, theta2, F, nnodes);
}

/* ────────────────────── HDF5 input ─────────────────────── */
static double* read_dset(hid_t file, const char* name, hsize_t* dims, int ndim)
{
    hid_t d = H5Dopen(file, name, H5P_DEFAULT);
    if(d < 0) { fprintf(stderr,"cannot open %s\n",name); exit(1); }
    hid_t s = H5Dget_space(d);
    H5Sget_simple_extent_dims(s, dims, nullptr);
    hsize_t tot = 1; for(int k=0;k<ndim;k++) tot *= dims[k];
    double* buf = (double*)malloc(tot*sizeof(double));
    H5Dread(d, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, buf);
    H5Sclose(s); H5Dclose(d);
    return buf;
}

int main(int argc, char** argv)
{
    if(argc < 2) {
        fprintf(stderr,"usage: %s <snapshot.hdf5> [particle_radii] [n_sample]\n", argv[0]);
        return 2;
    }
    const char* fname = argv[1];
    double particle_radii = (argc>2) ? atof(argv[2]) : 10.0;
    int    n_sample       = (argc>3) ? atoi(argv[3]) : 2000;

    pds_init();

    hid_t file = H5Fopen(fname, H5F_ACC_RDONLY, H5P_DEFAULT);
    if(file < 0) { fprintf(stderr,"cannot open %s\n",fname); return 1; }
    /* R_curv from header (fall back to 3100) */
    hid_t hdr = H5Gopen(file, "/Header", H5P_DEFAULT);
    if(hdr >= 0 && H5Aexists(hdr,"R_curvature_Mpc")>0) {
        hid_t a = H5Aopen(hdr,"R_curvature_Mpc",H5P_DEFAULT);
        H5Aread(a, H5T_NATIVE_DOUBLE, &R_CURV); H5Aclose(a);
    }
    if(hdr>=0) H5Gclose(hdr);

    hsize_t dc[2], dm[1];
    double* coord = read_dset(file, "/PartType1/Coordinates", dc, 2);
    double* mass  = read_dset(file, "/PartType1/Masses",      dm, 1);
    H5Fclose(file);
    N = (int)dc[0];

    X.resize(3*N); Q.resize(4*N); M.resize(N); SOFT.resize(N);
    for(int i=0;i<N;i++) {
        X[3*i]=coord[3*i]; X[3*i+1]=coord[3*i+1]; X[3*i+2]=coord[3*i+2];
        M[i]=mass[i];
        stereo_to_quat(X[3*i],X[3*i+1],X[3*i+2], &Q[4*i]);
    }
    free(coord); free(mass);

    /* softening exactly as utils.cc: SOFT[i] = ParticleRadi*(M[i]/M_min)^(1/3) */
    double M_min = M[0]; for(int i=1;i<N;i++) M_min = std::min(M_min, M[i]);
    for(int i=0;i<N;i++) SOFT[i] = particle_radii * cbrt(M[i]/M_min);

    /* domain extent / build octree */
    double Rmax=0.0;
    for(int i=0;i<N;i++) Rmax = std::max(Rmax, std::sqrt(X[3*i]*X[3*i]+X[3*i+1]*X[3*i+1]+X[3*i+2]*X[3*i+2]));
    double root = 2.00002*Rmax;
    Node* tree = make_node(0,0,0, root);
    double tbuild = omp_get_wtime();
    for(int i=0;i<N;i++) insert(tree, i);
    tbuild = omp_get_wtime() - tbuild;

    /* sample */
    n_sample = std::min(n_sample, N);
    std::vector<int> samp(N); for(int i=0;i<N;i++) samp[i]=i;
    std::mt19937 rng(12345); std::shuffle(samp.begin(), samp.end(), rng);
    samp.resize(n_sample);

    printf("PDS Barnes-Hut prototype\n");
    printf("  snapshot      : %s\n", fname);
    printf("  N             : %d\n", N);
    printf("  R_curv        : %.1f Mpc\n", R_CURV);
    printf("  particle_radii: %.4g Mpc   (M_min=%.4g)\n", particle_radii, M_min);
    printf("  sample        : %d field particles\n", n_sample);
    printf("  tree build    : %.3f s (%d leaves over the domain)\n\n", tbuild, N);

    /* ---- momentum-conservation audit mode (net force over ALL particles) ---- */
    if(argc > 4 && std::string(argv[4]) == "momentum") {
        printf("Momentum-conservation audit: net force  S = sum_i M_i a_i  over all %d particles.\n", N);
        printf("Reported as |S| / sum_i M_i|a_i|  (relative per-step momentum injection).\n");
        printf("A perfectly pairwise-antisymmetric (Newton-3rd-law) force gives 0; the\n");
        printf("compact-S^3 projected force is not exactly antisymmetric, so the EXACT value\n");
        printf("is the physical baseline that Barnes-Hut must not exceed.\n\n");

        auto netforce = [&](int theta_mode, double th2, double& rel, double& nfx, double& nfy, double& nfz){
            double sx=0, sy=0, sz=0, sabs=0;
            #pragma omp parallel for schedule(dynamic,16) reduction(+:sx,sy,sz,sabs)
            for(int i=0;i<N;i++){
                double Fb[3];
                if(theta_mode<0) { force_exact(i, Fb); }
                else { long nn=0; force_bh(tree, i, th2, Fb, nn); }
                sx += M[i]*Fb[0]; sy += M[i]*Fb[1]; sz += M[i]*Fb[2];
                sabs += M[i]*std::sqrt(Fb[0]*Fb[0]+Fb[1]*Fb[1]+Fb[2]*Fb[2]);
            }
            nfx=sx; nfy=sy; nfz=sz;
            rel = std::sqrt(sx*sx+sy*sy+sz*sz)/sabs;
        };

        double rel, nx, ny, nz;
        netforce(-1, 0.0, rel, nx, ny, nz);
        printf("  %-22s |S|/sum|Mf| = %.4e\n", "EXACT (baseline):", rel);
        double thetas_m[] = {0.7, 0.5, 0.35, 0.3, 0.25, 0.15};
        for(double th : thetas_m) {
            double r2,a,b,c; netforce(0, th*th, r2,a,b,c);
            printf("  Barnes-Hut theta=%.2f    |S|/sum|Mf| = %.4e   (%.2fx exact)\n",
                   th, r2, r2/rel);
        }
        return 0;
    }

    /* exact reference on the sample */
    std::vector<double> Fex(3*n_sample);
    double tex = omp_get_wtime();
    #pragma omp parallel for schedule(dynamic,16)
    for(int s=0;s<n_sample;s++) force_exact(samp[s], &Fex[3*s]);
    tex = omp_get_wtime() - tex;
    double tex_per = tex / n_sample;

    printf("  exact 120-image force: %.3f s for %d particles (%.3e s/particle)\n",
           tex, n_sample, tex_per);
    printf("  => full direct run would be ~%.3e s/step for all N (force only)\n\n",
           tex_per * N);

    double meanFex=0.0; for(int s=0;s<n_sample;s++)
        meanFex += std::sqrt(Fex[3*s]*Fex[3*s]+Fex[3*s+1]*Fex[3*s+1]+Fex[3*s+2]*Fex[3*s+2]);
    meanFex /= n_sample;
    printf("  mean |F_exact| = %.4e\n\n", meanFex);

    printf("  %-7s  %-11s  %-11s  %-11s  %-11s  %-9s  %-10s  %s\n",
           "theta","mean|dF/F|","med|dF/F|","99pct|dF/F|","mean|F_bh|","nodes/p","s/particle","speedup");
    double thetas[] = {1.0, 0.7, 0.5, 0.35, 0.25, 0.15, 0.05};
    for(double th : thetas) {
        double th2 = th*th;
        std::vector<double> relerr(n_sample);
        long nodes_tot = 0; double meanFbh=0.0;
        double tbh = omp_get_wtime();
        #pragma omp parallel for schedule(dynamic,16) reduction(+:nodes_tot,meanFbh)
        for(int s=0;s<n_sample;s++) {
            double Fb[3]; long nn=0;
            force_bh(tree, samp[s], th2, Fb, nn);
            nodes_tot += nn;
            double dx=Fb[0]-Fex[3*s], dy=Fb[1]-Fex[3*s+1], dz=Fb[2]-Fex[3*s+2];
            double fe=std::sqrt(Fex[3*s]*Fex[3*s]+Fex[3*s+1]*Fex[3*s+1]+Fex[3*s+2]*Fex[3*s+2]);
            meanFbh += std::sqrt(Fb[0]*Fb[0]+Fb[1]*Fb[1]+Fb[2]*Fb[2]);
            relerr[s] = (fe>0.0) ? std::sqrt(dx*dx+dy*dy+dz*dz)/fe : 0.0;
        }
        tbh = omp_get_wtime() - tbh;
        double tbh_per = tbh / n_sample;
        double mean=0.0; for(double e:relerr) mean+=e; mean/=n_sample;
        std::vector<double> sorted = relerr; std::sort(sorted.begin(), sorted.end());
        double med = sorted[n_sample/2];
        double p99 = sorted[(int)(0.99*n_sample)];
        printf("  %-7.2f  %-11.3e  %-11.3e  %-11.3e  %-11.3e  %-9.1f  %-10.3e  %.1fx\n",
               th, mean, med, p99, meanFbh/n_sample, (double)nodes_tot/n_sample, tbh_per, tex_per/tbh_per);
    }
    printf("\n  (speedup = exact / BH per-particle time at this N=%d; the BH\n"
           "   advantage grows ~ N/log N, so it widens at research-grade N.)\n", N);
    return 0;
}

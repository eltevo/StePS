#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "mpi.h"
#include "global_variables.h"

int N, rank = 0, numtasks = 1, n_GPU = 1, HOST_OMP_THREADS = 1, N_mpi_thread;
int IS_PERIODIC, COSMOLOGY, COMOVING_INTEGRATION;
REAL L = 10, Rsim = 4, THETA, M_min = 1, ParticleRadi = 0.1, mass_in_unit_sphere = 0.7;
REAL *M, *SOFT_LENGTH, *x, beta, rho_part;
double H0 = 0.3, Omega_lambda = 0.7;
bool ForceError = false, USE_RADIAL_BH_CORRECTION = true;
int RADIAL_BH_FORCE_TABLE_SIZE = 8;
REAL *RADIAL_BH_FORCE_TABLE;

#ifdef PERIODIC
REAL *T3_EWALD_FORCE_TABLE;
int N_EWALD_FORCE_GRID = 8;
void forces_periodic(REAL*, REAL*, int, int);
void cpu_forces_periodic(REAL*, REAL*, int, int);
#elif defined(PERIODIC_Z)
int RADIAL_FORCE_TABLE_SIZE = 8;
REAL *RADIAL_FORCE_TABLE;
#ifdef PERIODIC_Z_NOLOOKUP
int ewald_max = 3;
REAL ewald_cut = 2.6;
#else
REAL *S1R2_EWALD_FORCE_TABLE;
int Nrho_EWALD_FORCE_GRID = 8, Nz_EWALD_FORCE_GRID = 8;
#endif
void forces_periodic_z(REAL*, REAL*, int, int);
void cpu_forces_periodic_z(REAL*, REAL*, int, int);
#else
void forces(REAL*, REAL*, int, int);
void cpu_forces(REAL*, REAL*, int, int);
#endif
REAL force_softening(REAL, REAL);

int main()
{
    N = 96;
    std::vector<REAL> coords(3*N), mass(N), soft(N), original, cpu(3*N), gpu(3*N);
    M = mass.data(); SOFT_LENGTH = soft.data();
    srand(42);
    for (int i = 0; i < N; ++i) {
        for (int k = 0; k < 3; ++k) {
            double u = (double)rand()/RAND_MAX;
            #ifdef PERIODIC
            coords[3*i+k] = L*u;
            #elif defined(PERIODIC_Z)
            coords[3*i+k] = k == 2 ? L*u : (u-0.5)*5;
            #else
            coords[3*i+k] = (u-0.5)*5;
            #endif
        }
        mass[i] = 1 + (i % 5)*0.25;
        soft[i] = cbrt(mass[i])*ParticleRadi;
    }
    for (int i = 0; i < 3; ++i) {
        #ifdef PERIODIC
        coords[3*i] = L*0.5;
        #else
        coords[3*i] = 0;
        #endif
    }
    original = coords;
    x = coords.data();
    REAL radial[8], cylinder[8];
    for (int i = 0; i < 8; ++i) { radial[i] = 0.01*(i+1); cylinder[i] = 0.7+0.02*i; }
    RADIAL_BH_FORCE_TABLE = radial;
#ifdef PERIODIC
    std::vector<REAL> table(3*8*8*8);
    for (size_t i = 0; i < table.size(); ++i) table[i] = 0.001*sin(i*0.21);
    T3_EWALD_FORCE_TABLE = table.data();
#elif defined(PERIODIC_Z)
    RADIAL_FORCE_TABLE = cylinder;
    #ifndef PERIODIC_Z_NOLOOKUP
    std::vector<REAL> table(2*8*8);
    for (size_t i = 0; i < table.size(); ++i) table[i] = 0.001*sin(i*0.21);
    S1R2_EWALD_FORCE_TABLE = table.data();
    #endif
#endif
    const int first = 5, last = 82;
    N_mpi_thread = last-first+1;
    for (int mode = 1; mode <= 2; ++mode) {
        IS_PERIODIC = mode;
        for (int cosmology = 0; cosmology < 3; ++cosmology) {
            COSMOLOGY = cosmology != 0;
            COMOVING_INTEGRATION = cosmology == 1;
            for (int a = 0; a < 2; ++a) {
                THETA = a ? 0.4 : 0.00001;
                coords = original;
                srand(1234);
#ifdef PERIODIC
                cpu_forces_periodic(coords.data(), cpu.data(), first, last);
#elif defined(PERIODIC_Z)
                cpu_forces_periodic_z(coords.data(), cpu.data(), first, last);
#else
                cpu_forces(coords.data(), cpu.data(), first, last);
#endif
                coords = original;
                srand(1234);
#ifdef PERIODIC
                forces_periodic(coords.data(), gpu.data(), first, last);
#elif defined(PERIODIC_Z)
                forces_periodic_z(coords.data(), gpu.data(), first, last);
#else
                forces(coords.data(), gpu.data(), first, last);
#endif
                double maxerr = 0;
                for (int i = 0; i < 3*N_mpi_thread; ++i) {
                    if (!std::isfinite(cpu[i]) || !std::isfinite(gpu[i])) return 2;
                    maxerr = fmax(maxerr, fabs(cpu[i]-gpu[i]));
                }
                printf("mode=%d cosmology=%d theta=%g maxerr=%.12g\n",
                       mode, cosmology, (double)THETA, maxerr);
                #ifdef USE_SINGLE_PRECISION
                if (ForceError || maxerr > 1e-3 || coords != original) return 1;
                #else
                if (ForceError || maxerr > 1e-8 || coords != original) return 1;
                #endif
            }
        }
    }

    // Two particles closer than the Morton resolution must not be collapsed
    // into a single self-interacting leaf.
    REAL close_coords[] = {0.5, 1, 1, 0.50000001, 1, 1};
    REAL close_mass[] = {1, 2}, close_soft[] = {0.001, 0.001};
    REAL close_force[6];
    N = N_mpi_thread = 2;
    M = close_mass; SOFT_LENGTH = close_soft;
    ParticleRadi = 0.001;
    COSMOLOGY = 0; IS_PERIODIC = 1; THETA = 0.4;
    REAL distance = close_coords[3]-close_coords[0];
    REAL expected0 = close_mass[1]*force_softening(distance, cbrt(close_mass[1])*ParticleRadi + close_soft[0])*distance;
    REAL expected1 = -close_mass[0]*force_softening(distance, ParticleRadi + close_soft[1])*distance;
    srand(1234);
#ifdef PERIODIC
    forces_periodic(close_coords, close_force, 0, 1);
#elif defined(PERIODIC_Z)
    forces_periodic_z(close_coords, close_force, 0, 1);
#else
    forces(close_coords, close_force, 0, 1);
#endif
    for (int i = 0; i < 6; ++i)
        if (!std::isfinite(close_force[i])) return 2;
    if (fabs(close_force[0]-expected0) > 0.001 ||
        fabs(close_force[3]-expected1) > 0.001) return 1;
    return 0;
}

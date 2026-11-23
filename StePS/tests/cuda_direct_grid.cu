#include "../src/forces_cuda.cu"
#include <climits>
#include <cmath>
#include <vector>

#define CUDA_CHECK(call) do { \
    cudaError_t status = (call); \
    if (status != cudaSuccess) { \
        fprintf(stderr, "%s: %s\n", #call, cudaGetErrorString(status)); \
        return 1; \
    } \
} while (0)

int main()
{
    const int first = 5, count = 1101, size = first + count;
    if (force_grid_size(1, DIRECT_BLOCKSIZE) != 1 ||
        force_grid_size(INT_MAX, DIRECT_BLOCKSIZE) != 65535)
        return 1;

    std::vector<REAL> x(size), y(size), z(size), mass(size, 1), soft(size, 0.1);
    for (int i = 0; i < size; ++i) {
        x[i] = 0.2 + 0.002 * (i % 500);
        y[i] = 0.4 + 0.001 * (i % 300);
        z[i] = 0.7 + 0.001 * (i % 400);
    }
    x[0] = 4; y[0] = 4; z[0] = 4;
    x[1] = 7; y[1] = 7; z[1] = 7;

    REAL *dx, *dy, *dz, *dm, *ds, *out, *table, *radial;
    CUDA_CHECK(cudaMalloc(&dx, size*sizeof(REAL)));
    CUDA_CHECK(cudaMalloc(&dy, size*sizeof(REAL)));
    CUDA_CHECK(cudaMalloc(&dz, size*sizeof(REAL)));
    CUDA_CHECK(cudaMalloc(&dm, size*sizeof(REAL)));
    CUDA_CHECK(cudaMalloc(&ds, size*sizeof(REAL)));
    CUDA_CHECK(cudaMalloc(&out, 3*count*sizeof(REAL)));
    CUDA_CHECK(cudaMalloc(&table, 3*8*8*8*sizeof(REAL)));
    CUDA_CHECK(cudaMalloc(&radial, 8*sizeof(REAL)));
    CUDA_CHECK(cudaMemcpy(dx, x.data(), size*sizeof(REAL), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dy, y.data(), size*sizeof(REAL), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dz, z.data(), size*sizeof(REAL), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dm, mass.data(), size*sizeof(REAL), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ds, soft.data(), size*sizeof(REAL), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(table, 0, 3*8*8*8*sizeof(REAL)));
    CUDA_CHECK(cudaMemset(radial, 0, 8*sizeof(REAL)));

    std::vector<REAL> expected(3*count), actual(3*count);
    for (int mode = 1; mode <= 2; ++mode) {
        for (int pass = 0; pass < 2; ++pass) {
            CUDA_CHECK(cudaMemset(out, 0, 3*count*sizeof(REAL)));
            int blocks = pass ? 2 : force_grid_size(count, DIRECT_BLOCKSIZE);
#if defined(PERIODIC)
            ForceKernel_periodic<<<blocks, DIRECT_BLOCKSIZE>>>(2, dx, dy, dz, out, mode,
                dm, ds, 10, table, 8, first, first+count-1);
#elif defined(PERIODIC_Z) && defined(PERIODIC_Z_NOLOOKUP)
            ForceKernel_periodic_z<<<blocks, DIRECT_BLOCKSIZE>>>(2, dx, dy, dz, out, mode,
                dm, ds, 10, 4, 0, radial, 8, 0, 0, 0, first, first+count-1);
#elif defined(PERIODIC_Z)
            ForceKernel_periodic_z<<<blocks, DIRECT_BLOCKSIZE>>>(2, dx, dy, dz, out, mode,
                dm, ds, 10, 4, 0, radial, 8, table, 8, 8, 0, 0, 0, first, first+count-1);
#else
            ForceKernel<<<blocks, DIRECT_BLOCKSIZE>>>(2, dx, dy, dz, out, dm, ds,
                0, 0, 0, 0, first, first+count-1);
#endif
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(actual.data(), out, 3*count*sizeof(REAL), cudaMemcpyDeviceToHost));
            if (!pass) expected = actual;
            else for (int i = 0; i < 3*count; ++i) {
                if (!std::isfinite(actual[i]) || actual[i] != expected[i] || actual[i] == 0) {
                    fprintf(stderr, "mode=%d index=%d expected=%g got=%g\n",
                            mode, i, (double)expected[i], (double)actual[i]);
                    return 1;
                }
            }
        }
    }
    CUDA_CHECK(cudaFree(dx)); CUDA_CHECK(cudaFree(dy)); CUDA_CHECK(cudaFree(dz));
    CUDA_CHECK(cudaFree(dm)); CUDA_CHECK(cudaFree(ds)); CUDA_CHECK(cudaFree(out));
    CUDA_CHECK(cudaFree(table)); CUDA_CHECK(cudaFree(radial));
    puts("Direct CUDA grid-stride test passed.");
    return 0;
}

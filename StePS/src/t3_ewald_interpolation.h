#ifdef __CUDACC__
#define T3_HD __host__ __device__
#else
#define T3_HD
#endif

T3_HD static inline int t3_wrap_index(int i, int n)
{
    int wrapped = i % n;
    return wrapped < 0 ? wrapped + n : wrapped;
}

T3_HD static inline void t3_ewald_interpolate(int n, REAL box, const REAL *table, REAL dx, REAL dy, REAL dz, REAL result[3], int order)
{
    if (n == 1)
    {
        result[0] = table[0];
        result[1] = table[1];
        result[2] = table[2];
        return;
    }
    if (dx == 0 && dy == 0 && dz == 0)
    {
        result[0] = result[1] = result[2] = 0;
        return;
    }
    const REAL spacing = box / (REAL)n;
    const REAL coordinate[3] = {dx, dy, dz};
    int first[3], count;
    REAL weights[3][3];
    count = order == 0 ? 1 : (order == 2 ? 2 : 3);
    for (int axis = 0; axis < 3; ++axis)
    {
        REAL u = (coordinate[axis] + box*(REAL)0.5) / spacing - (REAL)0.5;
        if (order == 0)
        {
            first[axis] = (int)floor(u + (REAL)0.5);
            weights[axis][0] = 1;
        }
        else if (order == 2)
        {
            first[axis] = (int)floor(u);
            REAL t = u - (REAL)first[axis];
            weights[axis][0] = (REAL)1 - t;
            weights[axis][1] = t;
        }
        else
        {
            int nearest = (int)floor(u + (REAL)0.5);
            REAL t = u - (REAL)nearest;
            first[axis] = nearest - 1;
            weights[axis][0] = (REAL)0.5 * ((REAL)0.5 - t) * ((REAL)0.5 - t);
            weights[axis][1] = (REAL)0.75 - t*t;
            weights[axis][2] = (REAL)0.5 * ((REAL)0.5 + t) * ((REAL)0.5 + t);
        }
    }

    result[0] = result[1] = result[2] = 0;
    for (int i = 0; i < count; ++i)
        for (int j = 0; j < count; ++j)
            for (int k = 0; k < count; ++k)
            {
                int raw[3] = {first[0] + i, first[1] + j, first[2] + k};
                int wrapped[3] = {t3_wrap_index(raw[0], n), t3_wrap_index(raw[1], n),
                                  t3_wrap_index(raw[2], n)};
                size_t offset = (((size_t)wrapped[0]*n + wrapped[1])*n + wrapped[2])*3;
                REAL weight = weights[0][i]*weights[1][j]*weights[2][k];
                REAL value[3] = {table[offset], table[offset+1], table[offset+2]};
                if (raw[0] != wrapped[0] || raw[1] != wrapped[1] || raw[2] != wrapped[2])
                {
                    // The periodic force is continuous, but subtracting the nearest
                    // Newtonian force makes its correction jump across box faces.
                    REAL a[3], b[3], a2 = 0, b2 = 0;
                    for (int d = 0; d < 3; ++d)
                    {
                        a[d] = ((REAL)raw[d] + (REAL)0.5)*spacing - box*(REAL)0.5;
                        b[d] = ((REAL)wrapped[d] + (REAL)0.5)*spacing - box*(REAL)0.5;
                        a2 += a[d]*a[d];
                        b2 += b[d]*b[d];
                    }
                    REAL a3 = a2*sqrt(a2), b3 = b2*sqrt(b2);
                    for (int d = 0; d < 3; ++d)
                        value[d] += a[d]/a3 - b[d]/b3;
                }
                for (int d = 0; d < 3; ++d)
                    result[d] += weight*value[d];
            }
}

#undef T3_HD
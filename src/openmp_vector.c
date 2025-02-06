#include "openmp_vector.h"

void my_openmp_daxpy(int n, double alpha, const double *x, double *y) {
    #pragma omp for
    for (int i = 0; i < n; i++) {
        y[i] += alpha * x[i];
    }
    //#pragma omp barrier //追加
}

double my_openmp_ddot(int n, const double *x, const double *y) {
    double sum = 0.0;
    //#pragma omp threadprivate(sum)
    //#pragma omp for reduction(+:sum)
    #pragma omp for
    for (int i = 0; i < n; i++) {
        sum += x[i] * y[i];
    }
    //#pragma omp barrier //追加
    return sum;
}

void my_openmp_dscal(int n, double alpha, double *x) {
    #pragma omp for
    for (int i = 0; i < n; i++) {
        x[i] *= alpha;
    }
    //#pragma omp barrier //追加
}

void my_openmp_dcopy(int n, const double *x, double *y) {
    #pragma omp for
    for (int i = 0; i < n; i++) {
        y[i] = x[i];
    }
    //#pragma omp barrier //追加
}

void my_openmp_ddot_v2(int n, const double *x, const double *y, double *global_dot) {
    //#pragma omp single
    #pragma omp master
    {
        *global_dot = 0.0;
    }
    #pragma omp barrier //追加

    double sum = 0.0;
    #pragma omp for
    for (int i = 0; i < n; i++) {
        sum += x[i] * y[i];
    }
    #pragma omp barrier //追加

    #pragma omp atomic
    *global_dot += sum;
    #pragma omp barrier
}

void openmp_set_vector_zero(int vec_loc_size, double *vec) {
    #pragma omp for
    for (int l = 0; l < vec_loc_size; l++) {
        vec[l] = 0.0;
    }
    //#pragma omp barrier //追加
}

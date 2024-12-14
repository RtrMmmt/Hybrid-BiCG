#include "openmp_vector.h"

void my_openmp_daxpy(int n, double alpha, const double *x, double *y) {
    #pragma omp for
    for (int i = 0; i < n; i++) {
        y[i] += alpha * x[i];
    }
}

double my_openmp_ddot(int n, const double *x, const double *y) {
    double sum = 0.0;
    //#pragma omp threadprivate(sum)
    //#pragma omp for reduction(+:sum)
    #pragma omp for
    for (int i = 0; i < n; i++) {
        sum += x[i] * y[i];
    }
    return sum;
}

void my_openmp_dscal(int n, double alpha, double *x) {
    #pragma omp for
    for (int i = 0; i < n; i++) {
        x[i] *= alpha;
    }
}

void my_openmp_dcopy(int n, const double *x, double *y) {
    #pragma omp for
    for (int i = 0; i < n; i++) {
        y[i] = x[i];
    }
}

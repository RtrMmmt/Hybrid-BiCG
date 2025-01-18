#include "vector.h"

void my_daxpy(int n, double alpha, const double *x, double *y) {
    for (int i = 0; i < n; i++) {
        y[i] += alpha * x[i];
    }
}

double my_ddot(int n, const double *x, const double *y) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        sum += x[i] * y[i];
    }
    return sum;
}

void my_dscal(int n, double alpha, double *x) {
    for (int i = 0; i < n; i++) {
        x[i] *= alpha;
    }
}

void my_dcopy(int n, const double *x, double *y) {
    for (int i = 0; i < n; i++) {
        y[i] = x[i];
    }
}

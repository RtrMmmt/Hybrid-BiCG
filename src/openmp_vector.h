#ifndef OPENMP_VECTOR_H
#define OPENMP_VECTOR_H

#include <omp.h>

void    my_openmp_daxpy(int n, double alpha, const double *x, double *y);
double  my_openmp_ddot(int n, const double *x, const double *y);
void    my_openmp_dscal(int n, double alpha, double *x);
void    my_openmp_dcopy(int n, const double *x, double *y);
void    my_openmp_ddot_v2(int n, const double *x, const double *y, double *global_dot);

#endif // OPENMP_VECTOR_H

#ifndef VECTOR_H
#define VECTOR_H

void    my_daxpy(int n, double alpha, const double *x, double *y);
double  my_ddot(int n, const double *x, const double *y);
void    my_dscal(int n, double alpha, double *x);
void    my_dcopy(int n, const double *x, double *y);

#endif // VECTOR_H
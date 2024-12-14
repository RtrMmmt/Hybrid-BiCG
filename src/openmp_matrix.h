#ifndef OPENMP_MATRIX_H
#define OPENMP_MATRIX_H

#include "matrix.h"

#include <omp.h>

void MPI_openmp_csr_spmv_ovlap(CSR_Matrix *matrix_loc_diag, CSR_Matrix *matrix_loc_offd, INFO_Matrix *matrix_info, double *x_loc, double *x, double *y_loc);
void openmp_mult(CSR_Matrix *A_loc, double *x, double *y_loc);

#endif // OPENMP_MATRIX_H

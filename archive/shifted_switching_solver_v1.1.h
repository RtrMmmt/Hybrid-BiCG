#ifndef SHIFTED_SOLVER_H
#define SHIFTED_SOLVER_H

#include <math.h>
#include <time.h>
#include <stdbool.h>

#include "matrix.h"
#include "vector.h"
#include "openmp_matrix.h"
#include "openmp_vector.h"

int shifted_lopbicg(CSR_Matrix *A_loc_diag, CSR_Matrix *A_loc_offd, INFO_Matrix *A_info, double *x_loc_set, double *r_loc, double *sigma, int sigma_len, int seed);
int shifted_lopbicg_switching(CSR_Matrix *A_loc_diag, CSR_Matrix *A_loc_offd, INFO_Matrix *A_info, double *x_loc_set, double *r_loc, double *sigma, int sigma_len, int seed);

int shifted_lopbicg_matvec_ovlap(CSR_Matrix *A_loc_diag, CSR_Matrix *A_loc_offd, INFO_Matrix *A_info, double *x_loc_set, double *r_loc, double *sigma, int sigma_len, int seed);

int shifted_lopbicg_switching_noovlp(CSR_Matrix *A_loc_diag, CSR_Matrix *A_loc_offd, INFO_Matrix *A_info, double *x_loc_set, double *r_loc, double *sigma, int sigma_len, int seed);

#endif

#ifndef SOLVER_H
#define SOLVER_H

#include <math.h>
#include <time.h>
#include <stdbool.h>

#include "matrix.h"
#include "vector.h"
#include "openmp_matrix.h"
#include "openmp_vector.h"

#define EPS 1.0e-12   // 収束判定条件 
#define MAX_ITER 2000 // 最大反復回数 

#define MEASURE_TIME // 時間計測 
//#define MEASURE_SECTION_TIME // セクション時間計測
//#define DISPLAY_SECTION_TIME // 反復ごとのセクション時間表示

#define DISPLAY_ERROR  // 真の残差表示

//#define DISPLAY_SIGMA_RESIDUAL // 途中のsigma毎の残差表示 
#define OUT_ITER 1     // 残差の表示間隔 

#define SEED_SWITCHING

int shifted_lopbicg_dynamic(CSR_Matrix *A_loc_diag, CSR_Matrix *A_loc_offd, INFO_Matrix *A_info, double *x_loc_set, double *r_loc, double *sigma, int sigma_len, int seed);
int shifted_lopbicg_static(CSR_Matrix *A_loc_diag, CSR_Matrix *A_loc_offd, INFO_Matrix *A_info, double *x_loc_set, double *r_loc, double *sigma, int sigma_len, int seed);
int shifted_lopbicg_normal(CSR_Matrix *A_loc_diag, CSR_Matrix *A_loc_offd, INFO_Matrix *A_info, double *x_loc_set, double *r_loc, double *sigma, int sigma_len, int seed);

#endif

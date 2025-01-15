#include "openmp_matrix.h"

/******************************************************************************
 * @fn      MPI_openmp_csr_spmv_ovlap
 * @brief   Iallgathervでベクトルを集約している間に、対角ブロックとローカルのベクトルの積
 * 			を計算することでオーバーラップ
 * 			ベクトルの集約が終わったら、残りの非対角ブロックとベクトルの積を計算
 * 			今回の実装ではこれを採用
 ******************************************************************************/

void MPI_openmp_csr_spmv_ovlap(CSR_Matrix *matrix_loc_diag, CSR_Matrix *matrix_loc_offd, INFO_Matrix *matrix_info, double *x_loc, double *x, double *y_loc) {
	//int i;
	MPI_Request x_req;
	
    #pragma omp master
	{
    	MPI_Iallgatherv(x_loc, matrix_loc_diag->rows, MPI_DOUBLE, x, matrix_info->recvcounts, matrix_info->displs, MPI_DOUBLE, MPI_COMM_WORLD, &x_req);
	}

	#pragma omp for
    for (int i = 0; i < matrix_loc_diag->rows; i++) {
        y_loc[i] = 0.0;
    }

	openmp_mult(matrix_loc_diag, x_loc, y_loc);

    #pragma omp master
	{
    	MPI_Wait(&x_req, MPI_STATUS_IGNORE);
	}
	#pragma omp barrier

    openmp_mult(matrix_loc_offd, x, y_loc);
}

/******************************************************************************
 * @fn      openmp_mult
 * @brief   ローカルの行列とベクトルの積を計算
 ******************************************************************************/
/*
void openmp_mult(CSR_Matrix *A_loc, double *x, double *y_loc) {
	//unsigned int    j, end;
	//double          tempy;
	double          *val = A_loc->val;
	unsigned int    *col = A_loc->col;
	unsigned int    *ptr = A_loc->ptr;
	//end = 0;
	
	#pragma omp for //private(j, tempy, end)
	for (int i = 0; i < A_loc->rows; i++) {
		double tempy = 0.0;
		unsigned int j = ptr[i];
		unsigned int end = ptr[i + 1]; // ローカル変数として宣言

		for (; j < end; j++) {
			tempy += val[j] * x[col[j]];
		}

		y_loc[i] += tempy;
	}
}
*/

/*
void openmp_mult(CSR_Matrix *A_loc, double *x, double *y_loc) {
    unsigned int j, end;
    double tempy;
    double *val = A_loc->val;
    unsigned int *col = A_loc->col;
    unsigned int *ptr = A_loc->ptr;

    // スレッドごとの部分和を保持するローカル配列
    int n_threads = omp_get_max_threads();
    double *local_y = (double *)calloc(n_threads * A_loc->rows, sizeof(double));

    #pragma omp for //private(j, tempy, end)
    for (int i = 0; i < A_loc->rows; i++) {
        tempy = 0.0;
        j = ptr[i];
        end = ptr[i + 1];

        for (; j < end; j++) {
            tempy += val[j] * x[col[j]];
        }

        // スレッドローカルの配列に加算
        int thread_id = omp_get_thread_num();
        local_y[thread_id * A_loc->rows + i] += tempy;
    }

    // スレッド間で部分和を統合
    #pragma omp for
    for (int i = 0; i < A_loc->rows; i++) {
        for (int t = 0; t < n_threads; t++) {
            y_loc[i] += local_y[t * A_loc->rows + i];
        }
    }

    free(local_y);
}
*/

void openmp_mult(CSR_Matrix *A_loc, double *x, double *y_loc) {
    unsigned int j, end; // 外で変数を定義
    double tempy;         // 外で変数を定義
	end = 0;

    double *val = A_loc->val;
    unsigned int *col = A_loc->col;
    unsigned int *ptr = A_loc->ptr;

    #pragma omp for //private(j, end, tempy) // 各スレッドにローカルコピーを作成
    for (int i = 0; i < A_loc->rows; i++) {
        tempy = 0.0; // ループ内で初期化
        j = ptr[i];
        end = ptr[i + 1];

        for (; j < end; j++) {
            tempy += val[j] * x[col[j]];
        }

        y_loc[i] += tempy;
    }
}

#include "matrix.h"

//#define DYNAMIC_ROWS	/* 行ごとの非ゼロ要素数によって分割行数を決定 */

/******************************************************************************
 * @fn      coo_init_matrix, coo_free_matrix, coo_load_matrix
 * @brief   COO形式の行列の初期化、メモリ解放、matrix market形式のデータの読み込み
 ******************************************************************************/
void coo_init_matrix(COO_Matrix *m) {
	m->val = NULL;
	m->row = NULL;
	m->col = NULL;
	m->nz = m->rows = m->cols = 0;
}

void coo_free_matrix(COO_Matrix *m) {
	if (m->val != NULL) free(m->val);
	if (m->row != NULL) free(m->row);
	if (m->col != NULL) free(m->col);
	m->val = NULL;
	m->row = NULL;
	m->col = NULL;
	m->nz = m->rows = m->cols = 0;
}

int coo_load_matrix(char* filename, COO_Matrix *coo) {
	FILE 			*file;
	MM_typecode 	code;
	int 			m, n, nz, i, ival;
	unsigned int 	dc = 0;
	
	/* ファイルを開く */
	file = fopen(filename, "r");
	if (file == NULL) {
		fprintf(stderr, "ERROR: can't open file \"%s\"\n", filename);
		exit(EXIT_FAILURE);
	}
	
    /* codeにMatrix Marketフォーマットの情報を格納 */
	if (mm_read_banner(file, &code) != 0) {
		fprintf(stderr, "ERROR: Could not process Matrix Market banner.\n");
		exit(EXIT_FAILURE);
	}
	
	/* 対象としていない行列の場合エラー */
	if (!mm_is_matrix(code) || !mm_is_sparse(code) || !mm_is_coordinate(code)) {
		fprintf(stderr, "Sorry, this application does not support ");
		fprintf(stderr, "Market Market type: [%s]\n", mm_typecode_to_str(code));
		exit(EXIT_FAILURE);
	}
	
	/* 行列サイズを読み込む */
	if ((mm_read_mtx_crd_size(file, &m, &n, &nz)) != 0) {
		fprintf(stderr, "ERROR: Could not read matrix size.\n");
		exit(EXIT_FAILURE);
	}
	coo->rows = m;
	coo->cols = n;
	coo->nz = nz;
	
	/* メモリの確保 */
	coo->val = (double *) malloc(nz * sizeof(double));
	coo->row = (unsigned int *) malloc(nz * sizeof(unsigned int));
	coo->col = (unsigned int *) malloc(nz * sizeof(unsigned int));
	
	/* 行列の非ゼロ要素を読み込む */
	for(i = 0; i < nz; i++) {
		if (mm_is_pattern(code)) {
			if (fscanf(file, "%d %d\n", &(coo->row[i]), &(coo->col[i])) < 2) {
				fprintf(stderr, "ERROR: reading matrix data.\n");
				exit(EXIT_FAILURE);
			}
			coo->val[i] = 1.0;
		} else if (mm_is_real(code)) {
			if (fscanf(file, "%d %d %lg\n", &(coo->row[i]), &(coo->col[i]), &(coo->val[i])) < 3) {
				fprintf(stderr, "ERROR: reading matrix data.\n");
				exit(EXIT_FAILURE);
			}
		} else if (mm_is_integer(code)) {
			if (fscanf(file, "%d %d %d\n", &(coo->row[i]), &(coo->col[i]), &ival) < 3) {
				fprintf(stderr, "ERROR: reading matrix data.\n");
				exit(EXIT_FAILURE);
			}
			coo->val[i] = (double)ival;
		}
		coo->row[i]--;
		coo->col[i]--;
		if (coo->row[i] == coo->col[i]) ++dc;
	}
	
	fclose(file);
	
	return mm_is_symmetric(code);
}

/******************************************************************************
 * @fn     	coo_copy
 * @brief   COO形式のデータのコピー
 ******************************************************************************/

void coo_copy(COO_Matrix *in, COO_Matrix *out) {
	unsigned int i;
	out->val = (double *) malloc(in->nz * sizeof(double));
	out->row = (unsigned int *) malloc(in->nz * sizeof(unsigned int));
	out->col = (unsigned int *) malloc(in->nz * sizeof(unsigned int));
	out->nz = in->nz;
	out->rows = in->rows;
	out->cols = in->cols;
	
	for(i = 0; i < in->nz; i++) {
		out->val[i] = in->val[i]; 
	}
	for(i = 0; i < in->nz; i++) {
		out->row[i] = in->row[i]; 
	}
	for(i = 0; i < in->nz; i++) {
		out->col[i] = in->col[i]; 
	}
}

/******************************************************************************
 * @fn      coo_reorder_by_rows, coo_merge, coo_merge_sort
 * @brief   matrix market形式のデータから読み込んだ際に「行」で並び替え
 ******************************************************************************/
void coo_reorder_by_rows(COO_Matrix *m) {
    unsigned int 	*B = (unsigned int *)malloc(m->nz * sizeof(unsigned int));
    unsigned int 	*B2 = (unsigned int *)malloc(m->nz * sizeof(unsigned int));
    double 			*B3 = (double *)malloc(m->nz * sizeof(double));

    coo_merge_sort(m->row, m->col, m->val, B, B2, B3, 0, m->nz - 1);

    free(B); free(B2); free(B3);
}

void coo_merge(unsigned int *A, unsigned int *A2, double *A3, unsigned int *B, unsigned int *B2, double *B3, int left, int mid, int right) {
    int i = left, j = mid + 1, k = left;

    while (i <= mid && j <= right) {
        if (A[i] <= A[j]) {
            B[k]  = A[i];
            B2[k] = A2[i];
            B3[k] = A3[i];
            i++;
        } else {
            B[k]  = A[j];
            B2[k] = A2[j];
            B3[k] = A3[j];
            j++;
        }
        k++;
    }

    while (i <= mid) {
        B[k]  = A[i];
        B2[k] = A2[i];
        B3[k] = A3[i];
        i++;
        k++;
    }

    while (j <= right) {
        B[k]  = A[j];
        B2[k] = A2[j];
        B3[k] = A3[j];
        j++;
        k++;
    }

    for (i = left; i <= right; i++) {
        A[i]  = B[i];
        A2[i] = B2[i];
        A3[i] = B3[i];
    }
}

void coo_merge_sort(unsigned int *A, unsigned int *A2, double *A3, unsigned int *B, unsigned int *B2, double *B3, int left, int right) {
    if (left < right) {
        int mid = left + (right - left) / 2;
        coo_merge_sort(A, A2, A3, B, B2, B3, left, mid);
        coo_merge_sort(A, A2, A3, B, B2, B3, mid + 1, right);
        coo_merge(A, A2, A3, B, B2, B3, left, mid, right);
    }
}

/******************************************************************************
 * @fn      csr_init_matrix, csr_free_matrix, coo2csr, csr_load_matrix
 * @brief   CSR形式の行列の初期化、メモリ解放、COO形式からCSR形式へ変更
 ******************************************************************************/
void csr_init_matrix(CSR_Matrix *m) {
	m->val = NULL;
	m->col = NULL;
	m->ptr = NULL;
	m->nz  = m->rows = m->cols = 0;
}

void csr_free_matrix(CSR_Matrix *m) {
	if (m->val != NULL) free(m->val);
	if (m->col != NULL) free(m->col);
	if (m->ptr != NULL) free(m->ptr);
	m->val = NULL;
	m->col = NULL;
	m->ptr = NULL;
	m->nz  = m->rows = m->cols = 0;
}

void coo2csr(COO_Matrix *in, CSR_Matrix *out) {
	unsigned int 	i;
	unsigned int 	tot = 0;
	COO_Matrix 		coo;
	
	out->val = (double *) malloc(in->nz * sizeof(double));
	out->col = (unsigned int *) malloc(in->nz * sizeof(unsigned int));
	out->ptr = (unsigned int *) malloc(((in->rows)+1) * sizeof(unsigned int));
	out->nz  = in->nz;
	out->rows = in->rows;
	out->cols = in->cols;
	
	coo_copy(in, &coo);
	coo_reorder_by_rows(&coo);
	
	out->ptr[0] = tot;
	for(i = 0; i < coo.rows; i++) {
		while(tot < coo.nz && coo.row[tot] == i) {
			out->val[tot] = coo.val[tot];
			out->col[tot] = coo.col[tot];
			tot++;
		}
		out->ptr[i+1] = tot;
	}
	
	coo_free_matrix(&coo);
}

int csr_load_matrix(char* filename, CSR_Matrix *m) {
	int sym;
	COO_Matrix temp;
	coo_init_matrix(&temp);
	sym = coo_load_matrix(filename, &temp);
	coo2csr(&temp, m);
	coo_free_matrix(&temp);
	return sym;
}

/******************************************************************************
 * @fn      MPI_coo_load_matrix_block
 * @brief   行列を行方向に分割して、対角ブロックと非対角ブロックに分けてCOO形式で読み込む
 * 
 * 	|----|--------------|
 *  | d0 |	     od0  	| <- proc 0
 *  |----|----|---------|
 *  |od1 | d1 |	 od1	| <- proc 1
 *  |----|----|----|----|
 *  |  od2    |	d2 |od2	| <- proc 2
 *  |---------|----|----|
 *  |  od3         | d3 | <- proc 3
 *  |--------------|----|
 * 
 * @param   filename   		: matrix market形式のデータファイル
 * @param   matrix_loc_diag : 分割したローカルの対角ブロック(COO形式)
 * @param   matrix_loc_offd	: 分割したローカルの非対角ブロック(COO形式)
 * @param 	matrix_info		: 行列のサイズや分割サイズの情報
 * @sa
 * @detail  ローカルの行数はグローバルの行数をプロセス数で割ったもの
 * 			最初にローカルで保持するメモリのサイズを決めるため、1回データを全て読み取り非ゼ
 * 			ロ要素数をカウント　その後、確保したメモリにデータを格納するためもう一度データを
 * 			全て読み取る
 ******************************************************************************/
void MPI_coo_load_matrix_block(char *filename, COO_Matrix *matrix_loc_diag, COO_Matrix *matrix_loc_offd, INFO_Matrix *matrix_info) {
    int numprocs, myid;
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);

    int m, n, nz;

    FILE *file;
    MM_typecode code;

    file = fopen(filename, "r");	/* ファイルを開く */

	/* バナーと行列の情報を読み取る */
    if (mm_read_banner(file, &code) != 0) {
        fprintf(stderr, "ERROR: Could not process Matrix Market banner.\n");
        exit(EXIT_FAILURE);
    }
    if ((mm_read_mtx_crd_size(file, &m, &n, &nz)) != 0) {
        fprintf(stderr, "ERROR: Could not read matrix size.\n");
        exit(EXIT_FAILURE);
    }

    matrix_info->nz = nz;
    matrix_info->rows = m;
    matrix_info->cols = n;
    strncpy(matrix_info->code, code, sizeof(MM_typecode));

    int rows_per_proc = m / numprocs;
    int extra_rows = m % numprocs;
    int start_row = myid * rows_per_proc + (myid < extra_rows ? myid : extra_rows);
    int end_row = start_row + rows_per_proc + (myid < extra_rows ? 1 : 0);

    for (int proc = 0; proc < numprocs; proc++) {
        int proc_rows_per_proc = m / numprocs;
        int proc_extra_rows = m % numprocs;
        int proc_start_row = proc * proc_rows_per_proc + (proc < proc_extra_rows ? proc : proc_extra_rows);
        int proc_end_row = proc_start_row + proc_rows_per_proc + (proc < proc_extra_rows ? 1 : 0);
        
        matrix_info->recvcounts[proc] = proc_end_row - proc_start_row;
        matrix_info->displs[proc] = proc_start_row;
    }

    int row, col, ival;
    double val;

	int nz_loc_diag = 0, nz_loc_offd = 0;

	for (int i = 0; i < nz; i++) {
		if (mm_is_pattern(code)) {
			if (fscanf(file, "%d %d\n", &row, &col) < 2) {
				fprintf(stderr, "ERROR: reading matrix data.\n");
				exit(EXIT_FAILURE);
			}
		} else if (mm_is_real(code)) {
			if (fscanf(file, "%d %d %lg\n", &row, &col, &val) < 3) {
				fprintf(stderr, "ERROR: reading matrix data.\n");
				exit(EXIT_FAILURE);
			}
		} else if (mm_is_integer(code)) {
			if (fscanf(file, "%d %d %d\n", &row, &col, &ival) < 3) {
				fprintf(stderr, "ERROR: reading matrix data.\n");
				exit(EXIT_FAILURE);
			}
		}

		row--;  // 0-indexed に調整
		col--;  // 0-indexed に調整

		if (row >= start_row && row < end_row && col >= start_row && col < end_row) {
			nz_loc_diag++;
		} else if (row >= start_row && row < end_row) {
			nz_loc_offd++;
		}
	}

    matrix_loc_diag->rows = end_row - start_row;
    matrix_loc_diag->cols = end_row - start_row;
    matrix_loc_diag->nz = nz_loc_diag;
    matrix_loc_diag->val = (double *)malloc(nz_loc_diag * sizeof(double));
    matrix_loc_diag->row = (unsigned int *)malloc(nz_loc_diag * sizeof(unsigned int));
    matrix_loc_diag->col = (unsigned int *)malloc(nz_loc_diag * sizeof(unsigned int));

    matrix_loc_offd->rows = end_row - start_row;
    matrix_loc_offd->cols = n;
    matrix_loc_offd->nz = nz_loc_offd;
    matrix_loc_offd->val = (double *)malloc(nz_loc_offd * sizeof(double));
    matrix_loc_offd->row = (unsigned int *)malloc(nz_loc_offd * sizeof(unsigned int));
    matrix_loc_offd->col = (unsigned int *)malloc(nz_loc_offd * sizeof(unsigned int));

    fseek(file, 0, SEEK_SET);
    mm_read_banner(file, &code); // バナーを再度読み込み
    mm_read_mtx_crd_size(file, &m, &n, &nz); // 行列サイズを再度読み込み

    int diag_idx = 0, offd_idx = 0;
    for (int i = 0; i < nz; i++) {
        if (mm_is_pattern(code)) {
            if (fscanf(file, "%d %d\n", &row, &col) < 2) {
                fprintf(stderr, "ERROR: reading matrix data.\n");
                exit(EXIT_FAILURE);
            }
        } else if (mm_is_real(code)) {
            if (fscanf(file, "%d %d %lg\n", &row, &col, &val) < 3) {
                fprintf(stderr, "ERROR: reading matrix data.\n");
                exit(EXIT_FAILURE);
            }
        } else if (mm_is_integer(code)) {
            if (fscanf(file, "%d %d %d\n", &row, &col, &ival) < 3) {
                fprintf(stderr, "ERROR: reading matrix data.\n");
                exit(EXIT_FAILURE);
            }
        }
        row--; col--;
        if (row >= start_row && row < end_row) {
            if (col >= start_row && col < end_row) {
                matrix_loc_diag->row[diag_idx] = row - start_row;
                matrix_loc_diag->col[diag_idx] = col - start_row;
                matrix_loc_diag->val[diag_idx] = val;
                diag_idx++;
            } else {
                matrix_loc_offd->row[offd_idx] = row - start_row;
                matrix_loc_offd->col[offd_idx] = col;
                matrix_loc_offd->val[offd_idx] = val;
                offd_idx++;
            }
        }
    }

    fclose(file);
}

/******************************************************************************
 * @fn      MPI_csr_load_matrix_block
 * @brief   行列を対角ブロックと非対角ブロックに分割したCOO形式の行列をCSR形式に変換
 ******************************************************************************/
void MPI_csr_load_matrix_block(char *filename, CSR_Matrix *matrix_loc_diag, CSR_Matrix *matrix_loc_offd, INFO_Matrix *matrix_info) {
    COO_Matrix *temp_diag = (COO_Matrix *)malloc(sizeof(COO_Matrix));
    COO_Matrix *temp_offd = (COO_Matrix *)malloc(sizeof(COO_Matrix));
    
    coo_init_matrix(temp_diag);
    coo_init_matrix(temp_offd);

    MPI_coo_load_matrix_block(filename, temp_diag, temp_offd, matrix_info);

    csr_init_matrix(matrix_loc_diag);
    coo2csr(temp_diag, matrix_loc_diag);

    csr_init_matrix(matrix_loc_offd);
    coo2csr(temp_offd, matrix_loc_offd);

    coo_free_matrix(temp_diag); free(temp_diag); 
    coo_free_matrix(temp_offd); free(temp_offd);
}

/******************************************************************************
 * @fn      MPI_csr_spmv_ovlap
 * @brief   Iallgathervでベクトルを集約している間に、対角ブロックとローカルのベクトルの積
 * 			を計算することでオーバーラップ
 * 			ベクトルの集約が終わったら、残りの非対角ブロックとベクトルの積を計算
 * 			今回の実装ではこれを採用
 ******************************************************************************/
void MPI_csr_spmv_ovlap(CSR_Matrix *matrix_loc_diag, CSR_Matrix *matrix_loc_offd, INFO_Matrix *matrix_info, double *x_loc, double *x, double *y_loc) {
	int i;
	
	MPI_Request x_req;
	MPI_Iallgatherv(x_loc, matrix_loc_diag->rows, MPI_DOUBLE, x, matrix_info->recvcounts, matrix_info->displs, MPI_DOUBLE, MPI_COMM_WORLD, &x_req);

    for (i = 0; i < matrix_loc_diag->rows; i++) {
        y_loc[i] = 0.0;
    }
	mult(matrix_loc_diag, x_loc, y_loc);

	MPI_Wait(&x_req, MPI_STATUS_IGNORE);
	mult(matrix_loc_offd, x, y_loc);
}

/******************************************************************************
 * @fn      MPI_csr_spmv_async
 * @brief   IsendとIrecvを使い、各プロセスでベクトルを送受信する
 * 			その間に、対角ブロックとローカルのベクトルの積を計算
 * 			Waitsomeを使い、通信が終わったベクトルから非対角ブロックとの積を計算
 * 			Allgathervに比べて通信に時間がかかってしまうため今回は使わない(改善したい)
 ******************************************************************************/
void MPI_csr_spmv_async(CSR_Matrix *matrix_loc_diag, CSR_Matrix *matrix_loc_offd, INFO_Matrix *matrix_info, double *x_loc, double **x_recv, double *y_loc, int numsend, int myid, int *recv_procs) {
	int i, j, recvs_outstanding, completed, idx, recv_idx, start_idx, end_idx;
	int row, col_idx;
	
	MPI_Request req[numsend];
	MPI_Status stat[numsend];
	int indices[numsend];

	recvs_outstanding = numsend;

    for (i = 0; i < numsend; i++) {
        MPI_Isend(x_loc, matrix_info->recvcounts[myid], MPI_DOUBLE, recv_procs[i], 0, MPI_COMM_WORLD, &req[i]);
        MPI_Irecv(x_recv[i], matrix_info->recvcounts[recv_procs[i]], MPI_DOUBLE, recv_procs[i], 0, MPI_COMM_WORLD, &req[i]);
    }

    for (row = 0; row < matrix_loc_diag->rows; row++) {
        y_loc[row] = 0.0;
    }

	mult(matrix_loc_diag, x_loc, y_loc);

    while (recvs_outstanding > 0) {
        MPI_Waitsome(numsend, req, &completed, indices, stat);

        for (i = 0; i < completed; i++) {
            idx = stat[i].MPI_SOURCE;
            recv_idx = -1;

			if (idx > myid) {
				recv_idx = idx - 1;
			} else {
				recv_idx = idx;
			}

            if (recv_idx != -1) {
                start_idx = matrix_info->displs[recv_procs[recv_idx]];
				end_idx = start_idx + matrix_info->recvcounts[recv_procs[recv_idx]];
				mult_block(matrix_loc_offd, x_recv[recv_idx], y_loc, start_idx, end_idx);
                recvs_outstanding--;
            }
        }
    }
}

/******************************************************************************
 * @fn      mult
 * @brief   ローカルの行列とベクトルの積を計算
 ******************************************************************************/
void mult(CSR_Matrix *A_loc, double *x, double *y_loc) {
	unsigned int    i, j, end;
	double          tempy;
	double          *val = A_loc->val;
	unsigned int    *col = A_loc->col;
	unsigned int    *ptr = A_loc->ptr;
	end = 0;
	
	for(i = 0; i < A_loc->rows; i++) {
		tempy = 0.0;
		j = end;
		end = ptr[i+1];

		for( ; j < end; j++) {
			tempy += val[j] * x[col[j]];
		}
		y_loc[i] += tempy;
	}
}

/******************************************************************************
 * @fn      mult_block
 * @brief   ローカルの行列とローカルのベクトルの積を計算
 ******************************************************************************/
void mult_block(CSR_Matrix* A_loc, double* x_part, double* y_loc, int start_index, int end_index) {
    int row, idx, global_idx, local_idx;
	for (row = 0; row < A_loc->rows; row++) {
        for (idx = A_loc->ptr[row]; idx < A_loc->ptr[row + 1]; idx++) {
            global_idx = A_loc->col[idx];
            if (global_idx >= start_index && global_idx < end_index) {
                local_idx = global_idx - start_index;
                y_loc[row] += A_loc->val[idx] * x_part[local_idx];
            }
        }
    }
}


void csr_shift_diagonal(CSR_Matrix *A_diag, double sigma) {
    int diagonal_found;
    for (int i = 0; i < A_diag->rows; i++) {
        diagonal_found = 0;  // 各行の対角成分が見つかったかを確認するフラグ
        for (int j = A_diag->ptr[i]; j < A_diag->ptr[i + 1]; j++) {
            if (A_diag->col[j] == i) {
                A_diag->val[j] += sigma;
                diagonal_found = 1;  // 対角成分が見つかった
                break;  // 対角成分は一つだけのはずなので、ループを抜ける
            }
        }
        if (!diagonal_found) {  // 対角成分が見つからなかった場合
            fprintf(stderr, "Error: Diagonal element not found in row %d.\n", i);
            exit(EXIT_FAILURE);  // エラーメッセージを表示して終了
        }
    }
}
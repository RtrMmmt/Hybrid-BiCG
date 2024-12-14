// mpicc -O3 src/test.c src/matrix.c src/vector.c src/mmio.c src/openmp_matrix.c src/openmp_vector.c -I src -lm -fopenmp -L/usr/local/Cellar/libomp/19.1.5/lib -lomp

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <omp.h>

#include "vector.h"
#include "matrix.h"
#include "openmp_matrix.h"
#include "openmp_vector.h"

#define DISPLAY_NODE_INFO   /* ノード数とプロセス数の表示 */

int main(int argc, char *argv[]) {

    MPI_Init(&argc, &argv);

    int numprocs, myid, namelen;
    char proc_name[MPI_MAX_PROCESSOR_NAME];
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    MPI_Get_processor_name(proc_name, &namelen);

#ifdef DISPLAY_NODE_INFO
    /* ノード数とプロセス数をカウント */
    char *all_proc_names = NULL;
    if (myid == 0) {
        all_proc_names = (char *)malloc(numprocs * MPI_MAX_PROCESSOR_NAME * sizeof(char));
    }
    MPI_Gather(proc_name, MPI_MAX_PROCESSOR_NAME, MPI_CHAR, all_proc_names, MPI_MAX_PROCESSOR_NAME, MPI_CHAR, 0, MPI_COMM_WORLD);

    if (myid == 0) {
        int *proc_count_per_node = (int *)calloc(numprocs, sizeof(int));
        int unique_nodes = 0;

        for (int i = 0; i < numprocs; i++) {
            int found = 0;
            for (int j = 0; j < unique_nodes; j++) {
                if (strncmp(&all_proc_names[i * MPI_MAX_PROCESSOR_NAME], 
                            &all_proc_names[j * MPI_MAX_PROCESSOR_NAME], MPI_MAX_PROCESSOR_NAME) == 0) {
                    proc_count_per_node[j]++;
                    found = 1;
                    break;
                }
            }
            if (!found) {
                strncpy(&all_proc_names[unique_nodes * MPI_MAX_PROCESSOR_NAME], 
                        &all_proc_names[i * MPI_MAX_PROCESSOR_NAME], MPI_MAX_PROCESSOR_NAME);
                proc_count_per_node[unique_nodes]++;
                unique_nodes++;
            }
        }

        printf("Node: %d, Proc: %d\n", unique_nodes, numprocs);

        free(proc_count_per_node);
    }

    if (all_proc_names != NULL) {
        free(all_proc_names);
    }

    /* OpenMPスレッド数の確認 */
    int threads_per_process = 0;

    #pragma omp parallel
    {
        #pragma omp master
        threads_per_process = omp_get_num_threads();
    }

    /* 各プロセスで情報を表示 */
    printf("MPI Process %d on Node '%s': Threads per Process = %d\n", myid, proc_name, threads_per_process);
#endif

    double start_time, end_time, total_time;

    char *filename = argv[1];

    /* 行列の初期化 */
    INFO_Matrix A_info;
    A_info.recvcounts = (int *)malloc(numprocs * sizeof(int));
    A_info.displs = (int *)malloc(numprocs * sizeof(int));
	CSR_Matrix *A_loc_diag = (CSR_Matrix *)malloc(sizeof(CSR_Matrix));
    CSR_Matrix *A_loc_offd = (CSR_Matrix *)malloc(sizeof(CSR_Matrix));
	csr_init_matrix(A_loc_diag);
    csr_init_matrix(A_loc_offd);

    /* 行列の読み取り */
    start_time = MPI_Wtime();
    MPI_csr_load_matrix_block(filename, A_loc_diag, A_loc_offd, &A_info);
    end_time = MPI_Wtime();
    if (myid == 0) printf("IO time      : %e [sec.]\n", end_time - start_time);

    if (A_info.cols != A_info.rows) {
        printf("Error: matrix is not square.\n");
        exit(1);
    }

    /* ベクトルの初期化 */
    double *x_loc, *r_loc, *x, *r;
    int vec_size = A_info.rows;
    int vec_loc_size = A_loc_diag->rows;
    x_loc = (double *)malloc(vec_loc_size * sizeof(double));
    r_loc = (double *)malloc(vec_loc_size * sizeof(double));
    x = (double *)malloc(vec_size * sizeof(double));
    r = (double *)malloc(vec_size * sizeof(double));

    for (int i = 0; i < vec_loc_size; i++) {
        x_loc[i] = 1; /* 厳密解はすべて1 */
        r_loc[i] = 1; /* 初期残差はすべて2 */
    }

    double local_sum = 0.0;
    double openmp_norm = 0.0;

    start_time = MPI_Wtime();

#pragma omp parallel
{
    for (int i = 0; i < 100; i++) {
        openmp_norm = 0.0;
        MPI_openmp_csr_spmv_ovlap(A_loc_diag, A_loc_offd, &A_info, x_loc, x, r_loc);
        local_sum = my_openmp_ddot(vec_loc_size, r_loc, r_loc);
        #pragma omp atomic
        openmp_norm += local_sum;
        #pragma omp barrier
        #pragma omp master
        MPI_Allreduce(MPI_IN_PLACE, &openmp_norm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        #pragma omp barrier
    }
}

    end_time  = MPI_Wtime();

    if (myid == 0) {
        printf("OpenMP time: %e [sec.]\n", end_time - start_time);
    }

    double norm = 0.0;

    start_time = MPI_Wtime();

    for (int i = 0; i < 100; i++) {
        MPI_csr_spmv_ovlap(A_loc_diag, A_loc_offd, &A_info, x_loc, x, r_loc);
        norm = my_ddot(vec_loc_size, r_loc, r_loc);
        MPI_Allreduce(MPI_IN_PLACE, &norm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    }

    end_time  = MPI_Wtime();

    if (myid == 0) {
        printf("NO OpenMP time: %e [sec.]\n", end_time - start_time);
    }

    if (myid == 0) {
        printf("openmp_norm: %e, norm: %e\n", openmp_norm, norm);
    }


    //MPI_csr_spmv_ovlap(A_loc_diag, A_loc_offd, &A_info, x_loc, x, r_loc);

    //my_daxpy(vec_loc_size, 3, x_loc, r_loc);

/*
    start_time = MPI_Wtime();

    double global_sum = 0.0;

#pragma omp parallel
{

    
    for (int iter = 0; iter < 100; iter++) {
        MPI_Request global_sum_req;

        double local_sum = 0.0;

        MPI_openmp_csr_spmv_ovlap(A_loc_diag, A_loc_offd, &A_info, x_loc, x, r_loc);

        //my_daxpy(vec_loc_size, -1, x_loc, r_loc); 
        local_sum = my_openmp_ddot(vec_loc_size, r_loc, r_loc);
        #pragma omp atomic
        global_sum += local_sum;
        #pragma omp barrier
        #pragma omp master
        {
            MPI_Iallreduce(MPI_IN_PLACE, &global_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &global_sum_req);
        }
        #pragma omp master
        {
            MPI_Wait(&global_sum_req, MPI_STATUS_IGNORE);
        }
#pragma omp master
{
    if (myid == 0 && iter == 99) {
        printf("r_loc: [0]: %e [1]: %e [2]: %e\n", r_loc[0], r_loc[1], r_loc[2]);
        printf("iter: %d, local_sum: %e, global_sum: %e\n", iter, local_sum, global_sum);
    }
}
        #pragma omp barrier
        //double dot = my_openmp_ddot(vec_loc_size, x_loc, r_loc);
        my_openmp_dscal(vec_loc_size, 1, x_loc);
        my_openmp_dcopy(vec_loc_size, x_loc, r_loc);
        global_sum = 0.0;
    }
}

    end_time = MPI_Wtime();

/*
    if (myid == 0) {
        double sum = 0;
        for (int i = 0; i < vec_loc_size; i++) {
            sum += r_loc[i];
        }
        printf("vec_loc_size: %d\n", vec_loc_size);
        printf("sum of r_loc elements: %e\n", sum);
        printf("Calculation time: %e [sec.]\n", end_time - start_time);
    }
*/

/*
    start_time = MPI_Wtime();

    global_sum = 0.0;

    
    for (int iter = 0; iter < 100; iter++) {
        MPI_Request global_sum_req;

        double local_sum = 0.0;

        MPI_csr_spmv_ovlap(A_loc_diag, A_loc_offd, &A_info, x_loc, x, r_loc);

        //my_daxpy(vec_loc_size, -1, x_loc, r_loc);
        global_sum = my_ddot(vec_loc_size, r_loc, r_loc);
        MPI_Iallreduce(MPI_IN_PLACE, &global_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &global_sum_req);
        MPI_Wait(&global_sum_req, MPI_STATUS_IGNORE);

        if (myid == 0 && iter == 99) {
            printf("r_loc: [0]: %e [1]: %e [2]: %e\n", r_loc[0], r_loc[1], r_loc[2]);
            printf("iter: %d, local_sum: %e, global_sum: %e\n", iter, local_sum, global_sum);
        }
        
        //double dot = my_ddot(vec_loc_size, x_loc, r_loc);
        my_dscal(vec_loc_size, 1, x_loc);
        my_dcopy(vec_loc_size, x_loc, r_loc);
        global_sum = 0.0;
    }

    end_time = MPI_Wtime();
*/

	csr_free_matrix(A_loc_diag); free(A_loc_diag);
    csr_free_matrix(A_loc_offd); free(A_loc_offd);
    free(x_loc); free(r_loc); free(x); free(r);
    free(A_info.recvcounts);  free(A_info.displs);

	MPI_Finalize();
	return 0;
}

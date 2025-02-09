/*****************************************************************************
 * macでのコンパイルと実行コマンド
 * export LDFLAGS="-L/usr/local/opt/libomp/lib"
 * export CPPFLAGS="-I/usr/local/opt/libomp/include"
 * mpicc -O3 src/main_repeat.c src/solver_dynamic.c src/solver_static.c src/solver_mpi.c src/matrix.c src/vector.c src/mmio.c src/openmp_matrix.c src/openmp_vector.c -I src -lm -Xpreprocessor -fopenmp $CPPFLAGS $LDFLAGS -lomp
 * export OMP_NUM_THREADS=2
 * mpirun -np 4 ./a.out data/atmosmodd.mtx
 *****************************************************************************/

#include "solver.h"

#define DISPLAY_NODE_INFO   // ノード数とプロセス数の表示 
//#define DISPLAY_ERROR  // 相対誤差の表示 
//#define SOLVE_EACH_SIGMA  // 各システムでそれぞれ反復法を適用 

#define MIN_SIGMA_LENGTH 256
#define MAX_SIGMA_LENGTH 1024
#define SIGMA_LENGTH_STEP 2

int main(int argc, char *argv[]) {

    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    //MPI_Init(&argc, &argv);

    int numprocs, myid, namelen;
    char proc_name[MPI_MAX_PROCESSOR_NAME];
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    MPI_Get_processor_name(proc_name, &namelen);

#ifdef DISPLAY_NODE_INFO
    // ノード数とプロセス数をカウント 
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

    if (myid == 0) {
        int threads_per_process = 0;

        #pragma omp parallel
        {
            #pragma omp master
            threads_per_process = omp_get_num_threads();
        }

        printf("Threads/Proc %d\n", threads_per_process);
    }
#endif

    double start_time, end_time;

    char *filename = argv[1];

    // 行列の初期化 
    INFO_Matrix A_info;
    A_info.recvcounts = (int *)malloc(numprocs * sizeof(int));
    A_info.displs = (int *)malloc(numprocs * sizeof(int));
	CSR_Matrix *A_loc_diag = (CSR_Matrix *)malloc(sizeof(CSR_Matrix));
    CSR_Matrix *A_loc_offd = (CSR_Matrix *)malloc(sizeof(CSR_Matrix));
	csr_init_matrix(A_loc_diag);
    csr_init_matrix(A_loc_offd);

    // 行列の読み取り 
    start_time = MPI_Wtime();
    MPI_csr_load_matrix_block(filename, A_loc_diag, A_loc_offd, &A_info);
    end_time = MPI_Wtime();
    if (myid == 0) printf("IO time         : %e [sec.]\n", end_time - start_time);

    if (A_info.cols != A_info.rows) {
        printf("Error: matrix is not square.\n");
        exit(1);
    }

    double *x_loc_set, *r_loc;
    int vec_size = A_info.rows;
    int vec_loc_size = A_loc_diag->rows;
    r_loc = (double *)malloc(vec_loc_size * sizeof(double));

for (int sigma_len = MIN_SIGMA_LENGTH; sigma_len <= MAX_SIGMA_LENGTH; sigma_len *= SIGMA_LENGTH_STEP) {

    MPI_Barrier(MPI_COMM_WORLD);

    double sigma[sigma_len];
    int seed = sigma_len / 2 - 1;

    x_loc_set = (double *)malloc(vec_loc_size * sigma_len * sizeof(double));

    for (int i = 0; i < sigma_len; ++i) {
        sigma[i] = (i + 1) * (0.01 / sigma_len);
    }

    for (int i = 0; i < vec_loc_size; i++) {
        r_loc[i] = 1; // 右辺ベクトルはすべて1
    }

    for (int i = 0; i < vec_loc_size * sigma_len; i++) {
        x_loc_set[i] = 0; // 初期値はすべて0 
    }

    if (myid == 0) {
        printf("\n");
        printf("shift: %d, seed: %d\n", sigma_len, seed+1);
    }

    int total_iter;
    total_iter = shifted_lopbicg_dynamic(A_loc_diag, A_loc_offd, &A_info, x_loc_set, r_loc, sigma, sigma_len, seed);

    free(x_loc_set); 
}

	csr_free_matrix(A_loc_diag); free(A_loc_diag);
    csr_free_matrix(A_loc_offd); free(A_loc_offd);
    free(A_info.recvcounts);  free(A_info.displs);
    free(r_loc);

	MPI_Finalize();
	return 0;
}

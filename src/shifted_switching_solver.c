#include "shifted_switching_solver.h"

#define IDX(set, i, rows) ((set) + (i) * (rows))

#define EPS 1.0e-12   // 収束判定条件 
#define MAX_ITER 1000 // 最大反復回数 

#define MEASURE_TIME // 時間計測 
//#define MEASURE_SECTION_TIME // セクション時間計測
//#define DISPLAY_SECTION_TIME // セクション時間表示 

#define DISPLAY_RESULT // 結果表示 
//#define DISPLAY_RESIDUAL // 途中の残差表示 
//#define DISPLAY_SIGMA_RESIDUAL // 途中のsigma毎の残差表示 
#define OUT_ITER 1     // 残差の表示間隔 


int shifted_lopbicg(CSR_Matrix *A_loc_diag, CSR_Matrix *A_loc_offd, INFO_Matrix *A_info, double *x_loc_set, double *r_loc, double *sigma, int sigma_len, int seed) {

    int myid; MPI_Comm_rank(MPI_COMM_WORLD, &myid);

#if defined(MEASURE_TIME) || defined(MEASURE_SECTION_TIME)
    double start_time, end_time, total_time;
#endif

    if (A_info->cols != A_info->rows) {
        printf("Error: matrix is not square.\n");
        exit(1);
    }

    int vec_size = A_info->rows;
    int vec_loc_size = A_loc_diag->rows;

    int i, j;

    int k, max_iter, stop_count;
    double tol;
    double max_zeta_pi, abs_zeta_pi;

    double *r_old_loc, *r_hat_loc, *s_loc, *y_loc, *vec;

    double *p_loc_set;
    double *alpha_set, *beta_set, *omega_set, *eta_set, *zeta_set, *pi_old_set, *pi_new_set;
    double alpha_old, beta_old;

    //double dot_r, dot_zero, rTr, rTs, qTq, qTy, rTr_old;
    double global_dot_r, global_dot_zero, global_rTr, global_rTs, global_qTq, global_qTy, global_rTr_old;

    MPI_Request dot_r_req, rTr_req, rTs_req, qTq_req, qTy_req;

    bool *stop_flag;

    k = 0;
    tol = EPS;
    max_iter = MAX_ITER;
    stop_count = 0;

    r_old_loc   = (double *)malloc(vec_loc_size * sizeof(double));
    r_hat_loc   = (double *)malloc(vec_loc_size * sizeof(double));
    s_loc       = (double *)malloc(vec_loc_size * sizeof(double));
    y_loc       = (double *)malloc(vec_loc_size * sizeof(double));

    vec         = (double *)malloc(vec_size * sizeof(double));

    p_loc_set   = (double *)calloc(vec_loc_size * sigma_len, sizeof(double)); // 一応ゼロで初期化(下でもOK) 
    //p_loc_set   = (double *)malloc(vec_loc_size * sigma_len * sizeof(double));
    alpha_set   = (double *)malloc(sigma_len * sizeof(double));
    beta_set    = (double *)malloc(sigma_len * sizeof(double));
    omega_set   = (double *)malloc(sigma_len * sizeof(double));
    eta_set     = (double *)malloc(sigma_len * sizeof(double));
    zeta_set    = (double *)malloc(sigma_len * sizeof(double));
    pi_new_set  = (double *)malloc(sigma_len * sizeof(double));
    pi_old_set  = (double *)malloc(sigma_len * sizeof(double));

    stop_flag   = (bool *)calloc(sigma_len, sizeof(bool)); // Falseで初期化 

#ifdef MEASURE_SECTION_TIME
        double seed_time, shift_time;
        double section_start_time, section_end_time;
        seed_time = 0; shift_time = 0;
#endif

    global_rTr = my_ddot(vec_loc_size, r_loc, r_loc);      // (r#,r) 
    MPI_Iallreduce(MPI_IN_PLACE, &global_rTr, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &rTr_req);
    my_dcopy(vec_loc_size, r_loc, r_hat_loc);   // r# <- r = b 
    for (i = 0; i < sigma_len; i++) {
        my_dcopy(vec_loc_size, r_loc, &p_loc_set[i * vec_loc_size]);    // p[sigma] <- b 
        alpha_set[i]  = 1.0;  // alpha[sigma]  <- 1 
        beta_set[i]   = 0.0;  // beta[sigma]   <- 0         
        eta_set[i]    = 0.0;  // eta[sigma]    <- 0 
        pi_old_set[i] = 1.0;  // pi_old[sigma] <- 1 
        pi_new_set[i] = 1.0;  // pi_new[sigma] <- 1 
        zeta_set[i]   = 1.0;  // zeta[sigma]   <- 1 
    }
    my_dcopy(vec_loc_size, r_loc, &p_loc_set[seed * vec_loc_size]);
    MPI_Wait(&rTr_req, MPI_STATUS_IGNORE);

    global_dot_r = global_rTr;    // (r,r) 
    global_dot_zero = global_rTr; // (r#,r#) 
    max_zeta_pi = 1.0;   // max(|1/(zeta pi)|) 

#if defined(MEASURE_TIME) || defined(MEASURE_SECTION_TIME)
    start_time = MPI_Wtime();
#endif

#pragma omp parallel private(j)
{
    double local_dot_r, local_dot_zero, local_rTr, local_rTs, local_qTq, local_qTy, local_rTr_old;

    while (stop_count < sigma_len && k < max_iter) {

        // ===== グローバル変数の初期化 =====
        #pragma omp single
        {
            global_rTs = 0.0;
            global_qTq = 0.0;
            global_qTy = 0.0;
            global_dot_r = 0.0;
        }

        // ===== ベクトルのコピー =====
        my_openmp_dcopy(vec_loc_size, r_loc, r_old_loc);       // r_old <- r 
        my_openmp_dcopy(sigma_len, pi_new_set, pi_old_set);    // pi_old[sigma] <- pi_new[sigma] 

        // ===== alpha_old と beta_old の更新 =====
        #pragma omp single
        {
            alpha_old = alpha_set[seed];    // alpha_old <- alpha[seed] 
            beta_old = beta_set[seed];      // beta_old <- beta[seed] 
        }

        // ===== r# <- (A + sigma[seed] I) p[seed] =====
        MPI_openmp_csr_spmv_ovlap(A_loc_diag, A_loc_offd, A_info, &p_loc_set[seed * vec_loc_size], vec, s_loc);  // s <- (A + sigma[seed] I) p[seed]
        my_openmp_daxpy(vec_loc_size, sigma[seed], &p_loc_set[seed * vec_loc_size], s_loc);

        // ===== rTs = (r_hat, s) =====
        local_rTs = my_openmp_ddot(vec_loc_size, r_hat_loc, s_loc);
        #pragma omp atomic
        global_rTs += local_rTs;
        #pragma omp barrier

        #pragma omp master
        {
            MPI_Iallreduce(MPI_IN_PLACE, &global_rTs, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &rTs_req);  // rTs <- (r#,s) 
            MPI_Wait(&rTs_req, MPI_STATUS_IGNORE);
        }
        #pragma omp barrier

        // ===== alpha[seed] の更新 =====
        #pragma omp single
        {
            alpha_set[seed] = global_rTr / global_rTs;   // alpha[seed] <- (r#,r)/(r#,s) 
        }

        // ===== q <- r - alpha[seed] s =====
        my_openmp_daxpy(vec_loc_size, -alpha_set[seed], s_loc, r_loc);   // q <- r - alpha[seed] s (qはrを再利用する)

        // ===== y <- (A + sigma[seed] I) q =====
        MPI_openmp_csr_spmv_ovlap(A_loc_diag, A_loc_offd, A_info, r_loc, vec, y_loc);  // y <- (A + sigma[seed] I) q 
        my_openmp_daxpy(vec_loc_size, sigma[seed], r_loc, y_loc);

        // ===== (q,q) と (q,y) の計算 =====
        local_qTq = my_openmp_ddot(vec_loc_size, r_loc, r_loc);
        #pragma omp atomic
        global_qTq += local_qTq;
        #pragma omp barrier
        #pragma omp master
        {
            MPI_Iallreduce(MPI_IN_PLACE, &global_qTq, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &qTq_req);  // (q,q) 
        }
        local_qTy = my_openmp_ddot(vec_loc_size, r_loc, y_loc);
        #pragma omp atomic
        global_qTy += local_qTy;
        #pragma omp barrier
        #pragma omp master
        {
            MPI_Iallreduce(MPI_IN_PLACE, &global_qTy, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &qTy_req);  // (q,y) 
            MPI_Wait(&qTq_req, MPI_STATUS_IGNORE);
            MPI_Wait(&qTy_req, MPI_STATUS_IGNORE);
        }
        #pragma omp barrier

        // ===== omega[seed] の更新 =====
        #pragma omp single
        {
            omega_set[seed] = global_qTq / global_qTy;  // omega[seed] <- (q,q)/(q,y) 
        }

        // ===== x[seed] の更新 =====
        my_openmp_daxpy(vec_loc_size, alpha_set[seed], &p_loc_set[seed * vec_loc_size], &x_loc_set[seed * vec_loc_size]);     // x[seed] <- x[seed] + alpha[seed] p[seed] + omega[seed] q 
        my_openmp_daxpy(vec_loc_size, omega_set[seed], r_loc, &x_loc_set[seed * vec_loc_size]);

#ifdef MEASURE_SECTION_TIME
        #pragma omp master
        {
            section_start_time = MPI_Wtime();
        }
#endif

        // ===== シフト方程式 =====
        for (j = 0; j < sigma_len; j++) {
            if (j == seed) continue;   // seedはスキップ 
            if (stop_flag[j]) continue;  // 収束したものはスキップ 
            #pragma omp single
            {
                eta_set[j] = (beta_old / alpha_old) * alpha_set[seed] * eta_set[j] - (sigma[seed] - sigma[j]) * alpha_set[seed] * pi_old_set[j];
                // eta[sigma] = (beta_old / alpha_old) alpha[seed] eta[sigma] - (sigma[seed] - sigma[sigma]) alpha[seed] pi_old[sigma] 
                pi_new_set[j] = eta_set[j] + pi_old_set[j];    // pi_new[sigma] <- eta[sigma] + pi_old[sigma] 
                alpha_set[j] = (pi_old_set[j] / pi_new_set[j]) * alpha_set[seed];     // alpha[sigma] <- (pi_old[sigma] / pi_new[sigma]) alpha[seed] 
                omega_set[j] = omega_set[seed] / (1.0 - omega_set[seed] * (sigma[seed] - sigma[j]));      // omega[sigma] <- omega[0] / (1.0 + omega[0] * sigma) 
            }
            my_openmp_daxpy(vec_loc_size, omega_set[j] / (pi_new_set[j] * zeta_set[j]), r_loc, &x_loc_set[j * vec_loc_size]);     // x[sigma] <- x[sigma] + alpha[sigma] p[sigma] + omega[sigma] / (pi_new[sigma] zeta[sigma]) q 
            my_openmp_daxpy(vec_loc_size, alpha_set[j], &p_loc_set[j * vec_loc_size], &x_loc_set[j * vec_loc_size]);
            my_openmp_daxpy(vec_loc_size, omega_set[j] / (alpha_set[j] * zeta_set[j] * pi_new_set[j]), r_loc, &p_loc_set[j * vec_loc_size]);    // p[sigma] <- p[sigma] + omega[sigma] / (alpha[sigma] zeta[sigma]) (q / pi_new[sigma] - r_old / pi_old[sigma]) 
            my_openmp_daxpy(vec_loc_size, -omega_set[j] / (alpha_set[j] * zeta_set[j] * pi_old_set[j]), r_old_loc, &p_loc_set[j * vec_loc_size]);
            #pragma omp single
            {
                zeta_set[j] = (1.0 - omega_set[seed] * (sigma[seed] - sigma[j])) * zeta_set[j];      // zeta[sigma] <- (1.0 - omega[seed] (sigma[seed] - sigma[sigma])) * zeta[sigma] 
            }
        }

#ifdef MEASURE_SECTION_TIME
        #pragma omp master
        {
            section_end_time = MPI_Wtime();
            shift_time += section_end_time - section_start_time;
        }
#endif

        // ==== r の更新 ====
        my_openmp_daxpy(vec_loc_size, -omega_set[seed], y_loc, r_loc);            // r <- q - omega[seed] y 

        // ==== (r,r) の計算 ====
        local_dot_r = my_openmp_ddot(vec_loc_size, r_loc, r_loc);
        #pragma omp atomic
        global_dot_r += local_dot_r;
        #pragma omp barrier
        #pragma omp master
        {
            MPI_Iallreduce(MPI_IN_PLACE, &global_dot_r, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &dot_r_req);  // (r,r) 
            global_rTr_old = global_rTr;      // r_old <- (r#,r) 
        }

        // ==== (r#,r) の計算 ====
        local_rTr = my_openmp_ddot(vec_loc_size, r_hat_loc, r_loc);
        #pragma omp single
        {
            global_rTr = 0.0;
        }
        #pragma omp atomic
        global_rTr += local_rTr;
        #pragma omp barrier
        #pragma omp master
        {
            MPI_Iallreduce(MPI_IN_PLACE, &global_rTr, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &rTr_req);  // (r#,r) 
            MPI_Wait(&dot_r_req, MPI_STATUS_IGNORE);
            MPI_Wait(&rTr_req, MPI_STATUS_IGNORE);
        }
        #pragma omp barrier

        // ==== beta[seed] の更新 ====
        #pragma omp single
        {
            beta_set[seed] = (alpha_set[seed] / omega_set[seed]) * (global_rTr / global_rTr_old);   // beta[seed] <- (alpha[seed] / omega[seed]) ((r#,r)/(r#,r)) 
        }

        // ==== p[seed] の更新 ====
        my_openmp_dscal(vec_loc_size, beta_set[seed], &p_loc_set[seed * vec_loc_size]);     // p[seed] <- r + beta[seed] p[seed] - beta[seed] omega[seed] s 
        my_openmp_daxpy(vec_loc_size, 1.0, r_loc, &p_loc_set[seed * vec_loc_size]);
        my_openmp_daxpy(vec_loc_size, -beta_set[seed] * omega_set[seed], s_loc, &p_loc_set[seed * vec_loc_size]);

        // ==== シフト方程式 ====
        for (j = 0; j < sigma_len; j++) {
            if (j == seed) continue;
            if (stop_flag[j]) continue;
            #pragma omp single
            {
                beta_set[j] = (pi_old_set[j] / pi_new_set[j]) * (pi_old_set[j] / pi_new_set[j]) * beta_set[seed]; // beta[sigma] <- (pi_old[sigma] / pi_new[sigma])^2 beta[seed] 
            }
            my_openmp_dscal(vec_loc_size, beta_set[j], &p_loc_set[j * vec_loc_size]);      // p[sigma] <- 1 / (pi_new[sigma] zeta[sigma]) r + beta[sigma] p[sigma] 
            my_openmp_daxpy(vec_loc_size, 1.0 / (pi_new_set[j] * zeta_set[j]), r_loc, &p_loc_set[j * vec_loc_size]);
        }

        #pragma omp master
        {

#ifdef DISPLAY_SIGMA_RESIDUAL
            if (myid == 0 && k % OUT_ITER == 0) printf("Iter %d : ", k);
#endif

#ifdef MEASURE_SECTION_TIME
            section_start_time = MPI_Wtime();
#endif

            for (j = 0; j < sigma_len; j++) {
                if (stop_flag[j]) {
#ifdef DISPLAY_SIGMA_RESIDUAL
                    if (myid == 0 && k % OUT_ITER == 0) printf("------------ ");
#endif
                    continue;
                }
                if (j == seed) {
                    abs_zeta_pi = 1.0;
                } else {
                    abs_zeta_pi = fabs(1.0 / (zeta_set[j] * pi_new_set[j]));
                }
#ifdef DISPLAY_SIGMA_RESIDUAL
                if (myid == 0 && k % OUT_ITER == 0) printf("%e ", abs_zeta_pi * sqrt(global_dot_r / global_dot_zero));
#endif
                if (abs_zeta_pi * abs_zeta_pi * global_dot_r <= tol * tol * global_dot_zero) {
                    stop_flag[j] = true;
                    stop_count++;
                }
            }

#ifdef MEASURE_SECTION_TIME
            section_end_time = MPI_Wtime();
            shift_time += section_end_time - section_start_time;
#endif

#ifdef DISPLAY_SIGMA_RESIDUAL
            if (myid == 0 && k % OUT_ITER == 0) printf("\n");
#endif

            k++;

#ifdef DISPLAY_RESIDUAL
            if (myid == 0 && k % OUT_ITER == 0) {
                printf("Iteration: %d, Residual: %e\n", k, sqrt(global_dot_r / global_dot_zero));
            }
#endif

        }
        #pragma omp barrier
    }
}

#if defined(MEASURE_TIME) || defined(MEASURE_SECTION_TIME)
    end_time = MPI_Wtime();
    total_time = end_time - start_time;
#endif

#ifdef MEASURE_SECTION_TIME
    seed_time = total_time - shift_time;
#endif

    if (myid == 0) {
#ifdef DISPLAY_RESULT
        printf("Total iter   : %d\n", k - 1);
        printf("Final r      : %e\n", sqrt(global_dot_r / global_dot_zero));
        printf("x            : %e\n", x_loc_set[seed * vec_loc_size]);
#endif
#ifdef MEASURE_TIME
        printf("Total time   : %e [sec.] \n", total_time);
        printf("Avg time/iter: %e [sec.] \n", total_time / k);
#endif
#ifdef MEASURE_SECTION_TIME
        printf("Seed time    : %e [sec.]\n", seed_time);
        printf("Shift time   : %e [sec.]\n", shift_time);
#endif
    }

    free(r_old_loc); free(r_hat_loc); free(s_loc); free(y_loc);
    free(vec);
    free(p_loc_set); free(alpha_set); free(beta_set); free(omega_set); free(eta_set); free(zeta_set); free(pi_old_set); free(pi_new_set);
    free(stop_flag);

    return k;

}


int shifted_lopbicg_switching(CSR_Matrix *A_loc_diag, CSR_Matrix *A_loc_offd, INFO_Matrix *A_info, double *x_loc_set, double *r_loc, double *sigma, int sigma_len, int seed) {

    int myid; MPI_Comm_rank(MPI_COMM_WORLD, &myid);

#if defined(MEASURE_TIME) || defined(MEASURE_SECTION_TIME)
    double start_time, end_time, total_time;
#endif

    if (A_info->cols != A_info->rows) {
        printf("Error: matrix is not square.\n");
        exit(1);
    }

    int vec_size = A_info->rows;
    int vec_loc_size = A_loc_diag->rows;

    int i, j;

    int k, max_iter, stop_count;
    double tol;
    double max_zeta_pi, abs_zeta_pi;
    int max_sigma;

    double *r_old_loc, *r_hat_loc, *s_loc, *y_loc, *vec, *q_loc_copy;

    double *p_loc_set;
    double *alpha_set, *beta_set, *omega_set, *eta_set, *zeta_set;
    double alpha_old, beta_old;

    double *alpha_seed_archive, *beta_seed_archive, *omega_seed_archive;
    double *pi_archive_set;

    //double dot_r, dot_zero, rTr, rTs, qTq, qTy, rTr_old;
    double global_dot_r, global_dot_zero, global_rTr, global_rTs, global_qTq, global_qTy, global_rTr_old;

    MPI_Request dot_r_req, rTr_req, rTs_req, qTq_req, qTy_req, vec_req;

    bool *stop_flag;

    k = 1;
    tol = EPS;
    max_iter = MAX_ITER + 1;
    stop_count = 0;

    r_old_loc   = (double *)malloc(vec_loc_size * sizeof(double));
    r_hat_loc   = (double *)malloc(vec_loc_size * sizeof(double));
    s_loc       = (double *)malloc(vec_loc_size * sizeof(double));
    y_loc       = (double *)malloc(vec_loc_size * sizeof(double));
    q_loc_copy  = (double *)malloc(vec_loc_size * sizeof(double));

    vec         = (double *)malloc(vec_size * sizeof(double));

    p_loc_set   = (double *)calloc(vec_loc_size * sigma_len, sizeof(double)); // 一応ゼロで初期化(下でもOK) 
    //p_loc_set   = (double *)malloc(vec_loc_size * sigma_len * sizeof(double));
    alpha_set   = (double *)malloc(sigma_len * sizeof(double));
    beta_set    = (double *)malloc(sigma_len * sizeof(double));
    omega_set   = (double *)malloc(sigma_len * sizeof(double));
    eta_set     = (double *)malloc(sigma_len * sizeof(double));
    zeta_set    = (double *)malloc(sigma_len * sizeof(double));


    // seed switching で使うので履歴を保存する 
    alpha_seed_archive  = (double *)malloc(max_iter * sizeof(double));
    beta_seed_archive   = (double *)malloc(max_iter * sizeof(double));
    omega_seed_archive  = (double *)malloc(max_iter * sizeof(double));
    pi_archive_set      = (double *)malloc(max_iter * sigma_len * sizeof(double));

    stop_flag   = (bool *)calloc(sigma_len, sizeof(bool)); // Falseで初期化 

#ifdef MEASURE_SECTION_TIME
        double seed_time, shift_time, switch_time;
        double section_start_time, section_end_time;
        seed_time = 0; shift_time = 0; switch_time = 0;
#endif

    global_rTr = my_ddot(vec_loc_size, r_loc, r_loc);      // (r#,r) 
    MPI_Iallreduce(MPI_IN_PLACE, &global_rTr, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &rTr_req);
    my_dcopy(vec_loc_size, r_loc, r_hat_loc);   // r# <- r = b 
    for (i = 0; i < sigma_len; i++) {
        my_dcopy(vec_loc_size, r_loc, &p_loc_set[i * vec_loc_size]);    // p[sigma] <- b 
        alpha_set[i]  = 1.0;  // alpha[sigma]  <- 1 
        beta_set[i]   = 0.0;  // beta[sigma]   <- 0 
        eta_set[i]    = 0.0;  // eta[sigma]    <- 0 
        pi_archive_set[i * max_iter + 0] = 1.0;
        pi_archive_set[i * max_iter + 1] = 1.0;
        zeta_set[i]   = 1.0;  // zeta[sigma]   <- 1 
    }
    my_dcopy(vec_loc_size, r_loc, &p_loc_set[seed * vec_loc_size]);
    MPI_Allgatherv(&p_loc_set[seed * vec_loc_size], vec_loc_size, MPI_DOUBLE, vec, A_info->recvcounts, A_info->displs, MPI_DOUBLE, MPI_COMM_WORLD);
    MPI_Wait(&rTr_req, MPI_STATUS_IGNORE);

    global_dot_r = global_rTr;    // (r,r) 
    global_dot_zero = global_rTr; // (r#,r#) 
    max_zeta_pi = 1.0;   // max(|1/(zeta pi)|) 

    alpha_seed_archive[0] = 1.0;
    beta_seed_archive[0]  = 0.0;

#if defined(MEASURE_TIME) || defined(MEASURE_SECTION_TIME)
        start_time = MPI_Wtime();
#endif

#ifdef DISPLAY_SIGMA_RESIDUAL
        if (myid == 0) printf("Seed : %d\n", seed);
#endif

#pragma omp parallel private(j)  // スレッドの生成
{
    double local_dot_r, local_dot_zero, local_rTr, local_rTs, local_qTq, local_qTy, local_rTr_old;

    while (stop_count < sigma_len && k < max_iter) {

        // ===== グローバル変数の初期化 =====
        #pragma omp single
        {
            global_rTs = 0.0;
            global_qTq = 0.0;
            global_qTy = 0.0;
            global_dot_r = 0.0;
        }

        // ===== ベクトルのコピー =====
        my_openmp_dcopy(vec_loc_size, r_loc, r_old_loc);       // r_old <- r 

        // ===== r# <- (A + sigma[seed] I) p[seed] =====
        //MPI_csr_spmv_ovlap(A_loc_diag, A_loc_offd, A_info, &p_loc_set[seed * vec_loc_size], vec, s_loc);  // s <- (A + sigma[seed] I) p[seed] 

/*
        if (k != 0) {
            #pragma omp master
            {
                MPI_Iallgatherv(&p_loc_set[seed * vec_loc_size], vec_loc_size, MPI_DOUBLE, vec, A_info->recvcounts, A_info->displs, MPI_DOUBLE, MPI_COMM_WORLD, &vec_req);
            }
        }
*/


        // s_locの初期化
        #pragma omp for
        for (int l = 0; l < vec_loc_size; l++) {
            s_loc[l] = 0.0;
        }
        // 対角ブロックとローカルベクトルの積
        openmp_mult(A_loc_diag, &p_loc_set[seed * vec_loc_size], s_loc);

        // ベクトルの集約を待機
        if (k != 1) {
            #pragma omp master
            {
                MPI_Wait(&vec_req, MPI_STATUS_IGNORE);
            }
            #pragma omp barrier
        }

        // 非対角ブロックと集約ベクトルの積
        openmp_mult(A_loc_offd, vec, s_loc);

        my_openmp_daxpy(vec_loc_size, sigma[seed], &p_loc_set[seed * vec_loc_size], s_loc);

        // ===== rTs = (r_hat, s) =====
        local_rTs = my_openmp_ddot(vec_loc_size, r_hat_loc, s_loc);
        #pragma omp atomic
        global_rTs += local_rTs;
        #pragma omp barrier
        #pragma omp master
        {
            MPI_Allreduce(MPI_IN_PLACE, &global_rTs, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);  // rTs <- (r#,s) 
            //MPI_Iallreduce(MPI_IN_PLACE, &global_rTs, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &rTs_req);  // rTs <- (r#,s) 
            //MPI_Wait(&rTs_req, MPI_STATUS_IGNORE);
        }
        #pragma omp barrier

        // ===== alpha[seed] の更新 =====
        #pragma omp single
        {
            alpha_seed_archive[k] = global_rTr / global_rTs;   // alpha[seed] <- (r#,r)/(r#,s) 
        }

        // ===== q <- r - alpha[seed] s =====
        my_openmp_daxpy(vec_loc_size, -alpha_seed_archive[k], s_loc, r_loc);   // q <- r - alpha[seed] s 
        my_openmp_dcopy(vec_loc_size, r_loc, q_loc_copy); // q_copy <- q (q_copyにr_locをコピー　シード方程式を一つにまとめるため)

        // ===== y <- (A + sigma[seed] I) q =====
        //MPI_openmp_csr_spmv_ovlap(A_loc_diag, A_loc_offd, A_info, r_loc, vec, y_loc);  // y <- (A + sigma[seed] I) q 

        #pragma omp master
        {
            MPI_Iallgatherv(r_loc, vec_loc_size, MPI_DOUBLE, vec, A_info->recvcounts, A_info->displs, MPI_DOUBLE, MPI_COMM_WORLD, &vec_req);
        }

        // s_locの初期化
        #pragma omp for
        for (int l = 0; l < vec_loc_size; l++) {
            y_loc[l] = 0.0;
        }
        // 対角ブロックとローカルベクトルの積
        openmp_mult(A_loc_diag, r_loc, y_loc);

        // ベクトルの集約を待機
        #pragma omp master
        {
            MPI_Wait(&vec_req, MPI_STATUS_IGNORE);
        }
        #pragma omp barrier

        // 非対角ブロックと集約ベクトルの積
        openmp_mult(A_loc_offd, vec, y_loc);

        my_openmp_daxpy(vec_loc_size, sigma[seed], r_loc, y_loc);

        // ===== (q,q) と (q,y) の計算 =====
        local_qTq = my_openmp_ddot(vec_loc_size, r_loc, r_loc);
        #pragma omp atomic
        global_qTq += local_qTq;
        #pragma omp barrier
        #pragma omp master
        {
            MPI_Iallreduce(MPI_IN_PLACE, &global_qTq, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &qTq_req);  // (q,q) 
        }
        local_qTy = my_openmp_ddot(vec_loc_size, r_loc, y_loc);
        #pragma omp atomic
        global_qTy += local_qTy;
        #pragma omp barrier
        #pragma omp master
        {
            MPI_Iallreduce(MPI_IN_PLACE, &global_qTy, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &qTy_req);  // (q,y) 
            MPI_Wait(&qTq_req, MPI_STATUS_IGNORE);
            MPI_Wait(&qTy_req, MPI_STATUS_IGNORE);
        }
        #pragma omp barrier

        // ===== omega[seed] の更新 =====
        #pragma omp single
        {
            omega_seed_archive[k] = global_qTq / global_qTy;  // omega[seed] <- (q,q)/(q,y) 
        }

        // ===== x[seed] の更新 =====
        my_openmp_daxpy(vec_loc_size, alpha_seed_archive[k], &p_loc_set[seed * vec_loc_size], &x_loc_set[seed * vec_loc_size]);     // x[seed] <- x[seed] + alpha[seed] p[seed] + omega[seed] q 
        my_openmp_daxpy(vec_loc_size, omega_seed_archive[k], r_loc, &x_loc_set[seed * vec_loc_size]);

        // ==== r の更新 ====
        my_openmp_daxpy(vec_loc_size, -omega_seed_archive[k], y_loc, r_loc);            // r <- q - omega[seed] y 

        // ==== (r,r) の計算 ====
        local_dot_r = my_openmp_ddot(vec_loc_size, r_loc, r_loc);
        #pragma omp atomic
        global_dot_r += local_dot_r;
        #pragma omp barrier
        #pragma omp master
        {
            MPI_Iallreduce(MPI_IN_PLACE, &global_dot_r, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &dot_r_req);  // (r,r) 
            global_rTr_old = global_rTr;      // r_old <- (r#,r) 
        }

        // ==== (r#,r) の計算 ====
        local_rTr = my_openmp_ddot(vec_loc_size, r_hat_loc, r_loc);
        #pragma omp single
        {
            global_rTr = 0.0;
        }
        #pragma omp atomic
        global_rTr += local_rTr;
        #pragma omp barrier
        #pragma omp master
        {
            MPI_Iallreduce(MPI_IN_PLACE, &global_rTr, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &rTr_req);  // (r#,r) 
            MPI_Wait(&dot_r_req, MPI_STATUS_IGNORE);
            MPI_Wait(&rTr_req, MPI_STATUS_IGNORE);
        }
        #pragma omp barrier

        // ==== beta[seed] の更新 ====
        #pragma omp single
        {
            beta_seed_archive[k] = (alpha_seed_archive[k] / omega_seed_archive[k]) * (global_rTr / global_rTr_old);   // beta[seed] <- (alpha[seed] / omega[seed]) ((r#,r)/(r#,r)) 
        }

        // ==== p[seed] の更新 ====
        my_openmp_dscal(vec_loc_size, beta_seed_archive[k], &p_loc_set[seed * vec_loc_size]);     // p[seed] <- r + beta[seed] p[seed] - beta[seed] omega[seed] s 
        my_openmp_daxpy(vec_loc_size, 1.0, r_loc, &p_loc_set[seed * vec_loc_size]);
        my_openmp_daxpy(vec_loc_size, -beta_seed_archive[k] * omega_seed_archive[k], s_loc, &p_loc_set[seed * vec_loc_size]);

        // ==== 行列ベクトル積のためのベクトルqの集約を開始 ====
        #pragma omp master
        {
            MPI_Iallgatherv(&p_loc_set[seed * vec_loc_size], vec_loc_size, MPI_DOUBLE, vec, A_info->recvcounts, A_info->displs, MPI_DOUBLE, MPI_COMM_WORLD, &vec_req);
        }

        // ===== シフト方程式 =====

#ifdef MEASURE_SECTION_TIME
        #pragma omp master
        {
            section_start_time = MPI_Wtime();
        }
#endif

        for (j = 0; j < sigma_len; j++) {
            if (j == seed) continue;
            if (stop_flag[j]) continue;
            #pragma omp single
            {
                eta_set[j] = (beta_seed_archive[k - 1] / alpha_seed_archive[k - 1]) * alpha_seed_archive[k] * eta_set[j] - (sigma[seed] - sigma[j]) * alpha_seed_archive[k] * pi_archive_set[j * max_iter + (k - 1)];
                // eta[sigma] = (beta_old / alpha_old) alpha[seed] eta[sigma] - (sigma[seed] - sigma[sigma]) alpha[seed] pi_old[sigma] 
                pi_archive_set[j * max_iter + k] = eta_set[j] + pi_archive_set[j * max_iter + (k - 1)];    // pi_new[sigma] <- eta[sigma] + pi_old[sigma] 
                alpha_set[j] = (pi_archive_set[j * max_iter + (k - 1)] / pi_archive_set[j * max_iter + k]) * alpha_seed_archive[k];     // alpha[sigma] <- (pi_old[sigma] / pi_new[sigma]) alpha[seed] 
                omega_set[j] = omega_seed_archive[k] / (1.0 - omega_seed_archive[k] * (sigma[seed] - sigma[j]));      // omega[sigma] <- omega[0] / (1.0 + omega[0] * sigma) 
            }
            my_openmp_daxpy(vec_loc_size, omega_set[j] / (pi_archive_set[j * max_iter + k] * zeta_set[j]), q_loc_copy, &x_loc_set[j * vec_loc_size]);     // x[sigma] <- x[sigma] + alpha[sigma] p[sigma] + omega[sigma] / (pi_new[sigma] zeta[sigma]) q 
            my_openmp_daxpy(vec_loc_size, alpha_set[j], &p_loc_set[j * vec_loc_size], &x_loc_set[j * vec_loc_size]);
            my_openmp_daxpy(vec_loc_size, omega_set[j] / (alpha_set[j] * zeta_set[j] * pi_archive_set[j * max_iter + k]), q_loc_copy, &p_loc_set[j * vec_loc_size]);    // p[sigma] <- p[sigma] + omega[sigma] / (alpha[sigma] zeta[sigma]) (q / pi_new[sigma] - r_old / pi_old[sigma]) 
            my_openmp_daxpy(vec_loc_size, -omega_set[j] / (alpha_set[j] * zeta_set[j] * pi_archive_set[j * max_iter + (k - 1)]), r_old_loc, &p_loc_set[j * vec_loc_size]);
            #pragma omp single
            {
                zeta_set[j] = (1.0 - omega_seed_archive[k] * (sigma[seed] - sigma[j])) * zeta_set[j];      // zeta[sigma] <- (1.0 - omega[seed] (sigma[seed] - sigma[sigma])) * zeta[sigma] 
                beta_set[j] = (pi_archive_set[j * max_iter + (k - 1)] / pi_archive_set[j * max_iter + k]) * (pi_archive_set[j * max_iter + (k - 1)] / pi_archive_set[j * max_iter + k]) * beta_seed_archive[k]; // beta[sigma] <- (pi_old[sigma] / pi_new[sigma])^2 beta[seed] 
            }
            my_openmp_dscal(vec_loc_size, beta_set[j], &p_loc_set[j * vec_loc_size]);      // p[sigma] <- 1 / (pi_new[sigma] zeta[sigma]) r + beta[sigma] p[sigma] 
            my_openmp_daxpy(vec_loc_size, 1.0 / (pi_archive_set[j * max_iter + k] * zeta_set[j]), r_loc, &p_loc_set[j * vec_loc_size]);
        }

#ifdef MEASURE_SECTION_TIME
        #pragma omp master
        {
            section_end_time = MPI_Wtime();
            shift_time += section_end_time - section_start_time;
        }
#endif

        // ==== 収束判定 ====

        #pragma omp master
        {

#ifdef DISPLAY_SIGMA_RESIDUAL
            if (myid == 0 && k % OUT_ITER == 0) printf("Iter %d : ", k);
#endif

#ifdef MEASURE_SECTION_TIME
            section_start_time = MPI_Wtime();
#endif

            max_zeta_pi = 1.0;
            for (j = 0; j < sigma_len; j++) {
                if (stop_flag[j]) {
#ifdef DISPLAY_SIGMA_RESIDUAL
                    if (myid == 0 && k % OUT_ITER == 0) printf("------------ ");
#endif
                    continue;
                }
                if (j == seed) {
                    abs_zeta_pi = 1.0;
                } else {
                    abs_zeta_pi = fabs(1.0 / (zeta_set[j] * pi_archive_set[j * max_iter + k]));
                }
#ifdef DISPLAY_SIGMA_RESIDUAL
                if (myid == 0 && k % OUT_ITER == 0) printf("%e ", abs_zeta_pi * sqrt(global_dot_r / global_dot_zero));
#endif
                if (abs_zeta_pi * abs_zeta_pi * global_dot_r <= tol * tol * global_dot_zero) {
                    stop_flag[j] = true;
                    stop_count++;
                } else {
                    if (abs_zeta_pi > max_zeta_pi) {
                        max_zeta_pi = abs_zeta_pi;
                        max_sigma = j;
                    }
                }
            }

#ifdef MEASURE_SECTION_TIME
            section_end_time = MPI_Wtime();
            shift_time += section_end_time - section_start_time;
#endif

#ifdef DISPLAY_SIGMA_RESIDUAL
            if (myid == 0 && k % OUT_ITER == 0) printf("\n");
#endif

#ifdef MEASURE_SECTION_TIME
            section_start_time = MPI_Wtime();
#endif

            // seed switching 
            if (stop_flag[seed] && stop_count < sigma_len) {
#ifdef DISPLAY_SIGMA_RESIDUAL
                if (myid == 0) printf("Seed : %d\n", max_sigma);
#endif
                for (i = 1; i <= k; i++) {
                    alpha_seed_archive[i] = (pi_archive_set[max_sigma * max_iter + (i - 1)] / pi_archive_set[max_sigma * max_iter + i]) * alpha_seed_archive[i];
                    beta_seed_archive[i] = (pi_archive_set[max_sigma * max_iter + (i - 1)] / pi_archive_set[max_sigma * max_iter + i]) * (pi_archive_set[max_sigma * max_iter + (i - 1)] / pi_archive_set[max_sigma * max_iter + i]) * beta_seed_archive[i];
                    omega_seed_archive[i] = omega_seed_archive[i] / (1.0 - omega_seed_archive[i] * (sigma[seed] - sigma[max_sigma]));
                }
                my_dscal(vec_loc_size, 1.0 / (zeta_set[max_sigma] * pi_archive_set[max_sigma * max_iter + k]), r_loc);

                for (j = 0; j < sigma_len; j++) {
                    eta_set[j]    = 0.0;  // eta[sigma]    <- 0 
                    //pi_archive_set[j * max_iter + 0] = 1.0;
                    zeta_set[j]   = 1.0;  // zeta[sigma]   <- 1 
                }
                //alpha_seed_archive[0] = 1.0;
                //beta_seed_archive[0]  = 0.0;

                for (i = 1; i <= k; i++) {
                    for (j = 0; j < sigma_len; j++) {
                        if (stop_flag[j]) continue;
                        if (j == max_sigma) continue;
                        eta_set[j] = (beta_seed_archive[i - 1] / alpha_seed_archive[i - 1]) * alpha_seed_archive[i] * eta_set[j] - (sigma[max_sigma] - sigma[j]) * alpha_seed_archive[i] * pi_archive_set[j * max_iter + (i - 1)];
                        pi_archive_set[j * max_iter + i] = eta_set[j] + pi_archive_set[j * max_iter + (i - 1)];
                        zeta_set[j] = (1.0 - omega_seed_archive[i] * (sigma[max_sigma] - sigma[j])) * zeta_set[j];
                    }
                }

                seed = max_sigma;
            }

#ifdef MEASURE_SECTION_TIME
            section_end_time = MPI_Wtime();
            switch_time += section_end_time - section_start_time;
#endif

            k++;

#ifdef DISPLAY_RESIDUAL
            if (myid == 0 && k % OUT_ITER == 0) {
                printf("Iteration: %d, Residual: %e\n", k, sqrt(global_dot_r / global_dot_zero));
            }
#endif

        }
        #pragma omp barrier
    }
}

#if defined(MEASURE_TIME) || defined(MEASURE_SECTION_TIME)
    end_time = MPI_Wtime();
    total_time = end_time - start_time;
#endif

#ifdef MEASURE_SECTION_TIME
    seed_time = total_time - shift_time - switch_time;
#endif


    // ==== 結果表示 ====

    if (myid == 0) {
#ifdef DISPLAY_RESIDUAL
        printf("Total iter   : %d\n", k - 1);
        printf("Final r      : %e\n", sqrt(dot_r / dot_zero));
#endif
#ifdef MEASURE_TIME
        printf("Total time   : %e [sec.] \n", total_time);
        printf("Avg time/iter: %e [sec.] \n", total_time / k);
#endif
#ifdef MEASURE_SECTION_TIME
        printf("Seed time    : %e [sec.]\n", seed_time);
        printf("Shift time   : %e [sec.]\n", shift_time);
        printf("Switch time  : %e [sec.]\n", switch_time);
#endif
    }

    free(r_old_loc); free(r_hat_loc); free(s_loc); free(y_loc); free(q_loc_copy);
    free(vec);
    free(p_loc_set); free(alpha_set); free(beta_set); free(omega_set); free(eta_set); free(zeta_set);
    free(alpha_seed_archive); free(beta_seed_archive); free(omega_seed_archive); free(pi_archive_set);
    free(stop_flag);

    return k;

}


int shifted_lopbicg_matvec_ovlap(CSR_Matrix *A_loc_diag, CSR_Matrix *A_loc_offd, INFO_Matrix *A_info, double *x_loc_set, double *r_loc, double *sigma, int sigma_len, int seed) {

    int myid; MPI_Comm_rank(MPI_COMM_WORLD, &myid);

#if defined(MEASURE_TIME) || defined(MEASURE_SECTION_TIME)
    double start_time, end_time, total_time;
#endif

    if (A_info->cols != A_info->rows) {
        printf("Error: matrix is not square.\n");
        exit(1);
    }

    int vec_size = A_info->rows;
    int vec_loc_size = A_loc_diag->rows;

    int i, j;

    int k, max_iter, stop_count;
    double tol;
    double max_zeta_pi, abs_zeta_pi;

    double *r_old_loc, *r_hat_loc, *s_loc, *y_loc, *vec, *q_loc_copy;

    double *p_loc_set;
    double *alpha_set, *beta_set, *omega_set, *eta_set, *zeta_set, *pi_old_set, *pi_new_set;
    double alpha_old, beta_old;

    //double dot_r, dot_zero, rTr, rTs, qTq, qTy, rTr_old;
    double global_dot_r, global_dot_zero, global_rTr, global_rTs, global_qTq, global_qTy, global_rTr_old;

    MPI_Request dot_r_req, rTr_req, rTs_req, qTq_req, qTy_req, vec_req;

    bool *stop_flag;

    k = 0;
    tol = EPS;
    max_iter = MAX_ITER;
    stop_count = 0;

    r_old_loc   = (double *)malloc(vec_loc_size * sizeof(double));
    r_hat_loc   = (double *)malloc(vec_loc_size * sizeof(double));
    s_loc       = (double *)malloc(vec_loc_size * sizeof(double));
    y_loc       = (double *)malloc(vec_loc_size * sizeof(double));
    q_loc_copy  = (double *)malloc(vec_loc_size * sizeof(double));

    vec         = (double *)malloc(vec_size * sizeof(double));

    p_loc_set   = (double *)calloc(vec_loc_size * sigma_len, sizeof(double)); // 一応ゼロで初期化(下でもOK) 
    //p_loc_set   = (double *)malloc(vec_loc_size * sigma_len * sizeof(double));
    alpha_set   = (double *)malloc(sigma_len * sizeof(double));
    beta_set    = (double *)malloc(sigma_len * sizeof(double));
    omega_set   = (double *)malloc(sigma_len * sizeof(double));
    eta_set     = (double *)malloc(sigma_len * sizeof(double));
    zeta_set    = (double *)malloc(sigma_len * sizeof(double));
    pi_new_set  = (double *)malloc(sigma_len * sizeof(double));
    pi_old_set  = (double *)malloc(sigma_len * sizeof(double));

    stop_flag   = (bool *)calloc(sigma_len, sizeof(bool)); // Falseで初期化 

#ifdef MEASURE_SECTION_TIME
        double seed_time, shift_time;
        double section_start_time, section_end_time;
        seed_time = 0; shift_time = 0;
#endif

    global_rTr = my_ddot(vec_loc_size, r_loc, r_loc);      // (r#,r) 
    MPI_Iallreduce(MPI_IN_PLACE, &global_rTr, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &rTr_req);
    my_dcopy(vec_loc_size, r_loc, r_hat_loc);   // r# <- r = b 
    for (i = 0; i < sigma_len; i++) {
        my_dcopy(vec_loc_size, r_loc, &p_loc_set[i * vec_loc_size]);    // p[sigma] <- b 
        alpha_set[i]  = 1.0;  // alpha[sigma]  <- 1 
        beta_set[i]   = 0.0;  // beta[sigma]   <- 0         
        eta_set[i]    = 0.0;  // eta[sigma]    <- 0 
        pi_old_set[i] = 1.0;  // pi_old[sigma] <- 1 
        pi_new_set[i] = 1.0;  // pi_new[sigma] <- 1 
        zeta_set[i]   = 1.0;  // zeta[sigma]   <- 1 
    }
    my_dcopy(vec_loc_size, r_loc, &p_loc_set[seed * vec_loc_size]);
    MPI_Allgatherv(&p_loc_set[seed * vec_loc_size], vec_loc_size, MPI_DOUBLE, vec, A_info->recvcounts, A_info->displs, MPI_DOUBLE, MPI_COMM_WORLD);
    MPI_Wait(&rTr_req, MPI_STATUS_IGNORE);

    global_dot_r = global_rTr;    // (r,r) 
    global_dot_zero = global_rTr; // (r#,r#) 
    max_zeta_pi = 1.0;   // max(|1/(zeta pi)|) 

#if defined(MEASURE_TIME) || defined(MEASURE_SECTION_TIME)
    start_time = MPI_Wtime();
#endif

#pragma omp parallel private(j)  // スレッドの生成
{
    double local_dot_r, local_dot_zero, local_rTr, local_rTs, local_qTq, local_qTy, local_rTr_old;

    while (stop_count < sigma_len && k < max_iter) {

        // ===== グローバル変数の初期化 =====
        #pragma omp single
        {
            global_rTs = 0.0;
            global_qTq = 0.0;
            global_qTy = 0.0;
            global_dot_r = 0.0;
        }

        // ===== ベクトルのコピー =====
        my_openmp_dcopy(vec_loc_size, r_loc, r_old_loc);       // r_old <- r 
        my_openmp_dcopy(sigma_len, pi_new_set, pi_old_set);    // pi_old[sigma] <- pi_new[sigma] 

        // ===== alpha_old と beta_old の更新 =====
        #pragma omp single
        {
            alpha_old = alpha_set[seed];    // alpha_old <- alpha[seed] 
            beta_old = beta_set[seed];      // beta_old <- beta[seed] 
        }

        // ===== r# <- (A + sigma[seed] I) p[seed] =====
        //MPI_openmp_csr_spmv_ovlap(A_loc_diag, A_loc_offd, A_info, &p_loc_set[seed * vec_loc_size], vec, s_loc);  // s <- (A + sigma[seed] I) p[seed] 

/*
        if (k != 0) {
            #pragma omp master
            {
            MPI_Iallgatherv(&p_loc_set[seed * vec_loc_size], vec_loc_size, MPI_DOUBLE, vec, A_info->recvcounts, A_info->displs, MPI_DOUBLE, MPI_COMM_WORLD, &vec_req);
            }
        }
*/

        // s_locの初期化
        #pragma omp for
        for (int l = 0; l < vec_loc_size; l++) {
            s_loc[l] = 0.0;
        }
        // 対角ブロックとローカルベクトルの積
        openmp_mult(A_loc_diag, &p_loc_set[seed * vec_loc_size], s_loc);

        // ベクトルの集約を待機
        if (k != 0) {
            #pragma omp master
            {
                MPI_Wait(&vec_req, MPI_STATUS_IGNORE);
            }
            #pragma omp barrier
        }

        // 非対角ブロックと集約ベクトルの積
        openmp_mult(A_loc_offd, vec, s_loc);

        my_openmp_daxpy(vec_loc_size, sigma[seed], &p_loc_set[seed * vec_loc_size], s_loc);

        // ===== rTs = (r_hat, s) =====
        local_rTs = my_openmp_ddot(vec_loc_size, r_hat_loc, s_loc);
        #pragma omp atomic
        global_rTs += local_rTs;
        #pragma omp barrier
        #pragma omp master
        {
            MPI_Allreduce(MPI_IN_PLACE, &global_rTs, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);  // rTs <- (r#,s) 
            //MPI_Iallreduce(MPI_IN_PLACE, &global_rTs, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &rTs_req);  // rTs <- (r#,s) 
            //MPI_Wait(&rTs_req, MPI_STATUS_IGNORE);
        }
        #pragma omp barrier

        // ===== alpha[seed] の更新 =====
        #pragma omp single
        {
            alpha_set[seed] = global_rTr / global_rTs;   // alpha[seed] <- (r#,r)/(r#,s) 
        }

        // ===== q <- r - alpha[seed] s =====
        my_openmp_daxpy(vec_loc_size, -alpha_set[seed], s_loc, r_loc);   // q <- r - alpha[seed] s (qはrを再利用する)
        my_openmp_dcopy(vec_loc_size, r_loc, q_loc_copy); // q_copy <- q (q_copyにr_locをコピー　シード方程式を一つにまとめるため)

        // ===== y <- (A + sigma[seed] I) q =====
        MPI_openmp_csr_spmv_ovlap(A_loc_diag, A_loc_offd, A_info, r_loc, vec, y_loc);  // y <- (A + sigma[seed] I) q 
        my_openmp_daxpy(vec_loc_size, sigma[seed], r_loc, y_loc);

        // ===== (q,q) と (q,y) の計算 =====
        local_qTq = my_openmp_ddot(vec_loc_size, r_loc, r_loc);
        #pragma omp atomic
        global_qTq += local_qTq;
        #pragma omp barrier
        #pragma omp master
        {
            MPI_Iallreduce(MPI_IN_PLACE, &global_qTq, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &qTq_req);  // (q,q) 
        }
        local_qTy = my_openmp_ddot(vec_loc_size, r_loc, y_loc);
        #pragma omp atomic
        global_qTy += local_qTy;
        #pragma omp barrier
        #pragma omp master
        {
            MPI_Iallreduce(MPI_IN_PLACE, &global_qTy, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &qTy_req);  // (q,y) 
            MPI_Wait(&qTq_req, MPI_STATUS_IGNORE);
            MPI_Wait(&qTy_req, MPI_STATUS_IGNORE);
        }
        #pragma omp barrier

        // ===== omega[seed] の更新 =====
        #pragma omp single
        {
            omega_set[seed] = global_qTq / global_qTy;  // omega[seed] <- (q,q)/(q,y) 
        }

        // ===== x[seed] の更新 =====
        my_openmp_daxpy(vec_loc_size, alpha_set[seed], &p_loc_set[seed * vec_loc_size], &x_loc_set[seed * vec_loc_size]);     // x[seed] <- x[seed] + alpha[seed] p[seed] + omega[seed] q 
        my_openmp_daxpy(vec_loc_size, omega_set[seed], r_loc, &x_loc_set[seed * vec_loc_size]);

        // ==== r の更新 ====
        my_openmp_daxpy(vec_loc_size, -omega_set[seed], y_loc, r_loc);            // r <- q - omega[seed] y 

        // ==== (r,r) の計算 ====
        local_dot_r = my_openmp_ddot(vec_loc_size, r_loc, r_loc);
        #pragma omp atomic
        global_dot_r += local_dot_r;
        #pragma omp barrier
        #pragma omp master
        {
            MPI_Iallreduce(MPI_IN_PLACE, &global_dot_r, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &dot_r_req);  // (r,r) 
            global_rTr_old = global_rTr;      // r_old <- (r#,r) 
        }

        // ==== (r#,r) の計算 ====
        local_rTr = my_openmp_ddot(vec_loc_size, r_hat_loc, r_loc);
        #pragma omp single
        {
            global_rTr = 0.0;
        }
        #pragma omp atomic
        global_rTr += local_rTr;
        #pragma omp barrier
        #pragma omp master
        {
            MPI_Iallreduce(MPI_IN_PLACE, &global_rTr, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &rTr_req);  // (r#,r) 
            MPI_Wait(&dot_r_req, MPI_STATUS_IGNORE);
            MPI_Wait(&rTr_req, MPI_STATUS_IGNORE);
        }
        #pragma omp barrier

        // ==== beta[seed] の更新 ====
        #pragma omp single
        {
            beta_set[seed] = (alpha_set[seed] / omega_set[seed]) * (global_rTr / global_rTr_old);   // beta[seed] <- (alpha[seed] / omega[seed]) ((r#,r)/(r#,r)) 
        }

        // ==== p[seed] の更新 ====
        my_openmp_dscal(vec_loc_size, beta_set[seed], &p_loc_set[seed * vec_loc_size]);     // p[seed] <- r + beta[seed] p[seed] - beta[seed] omega[seed] s 
        my_openmp_daxpy(vec_loc_size, 1.0, r_loc, &p_loc_set[seed * vec_loc_size]);
        my_openmp_daxpy(vec_loc_size, -beta_set[seed] * omega_set[seed], s_loc, &p_loc_set[seed * vec_loc_size]);

        // ==== 行列ベクトル積のためのベクトルqの集約を開始 ====
        #pragma omp master
        {
            MPI_Iallgatherv(&p_loc_set[seed * vec_loc_size], vec_loc_size, MPI_DOUBLE, vec, A_info->recvcounts, A_info->displs, MPI_DOUBLE, MPI_COMM_WORLD, &vec_req);
        }


        // ===== シフト方程式 =====

#ifdef MEASURE_SECTION_TIME
        #pragma omp master
        {
            section_start_time = MPI_Wtime();
        }
#endif

        for (j = 0; j < sigma_len; j++) {
            if (j == seed) continue;   // seedはスキップ 
            if (stop_flag[j]) continue;  // 収束したものはスキップ 
            #pragma omp single
            {
                eta_set[j] = (beta_old / alpha_old) * alpha_set[seed] * eta_set[j] - (sigma[seed] - sigma[j]) * alpha_set[seed] * pi_old_set[j];
                // eta[sigma] = (beta_old / alpha_old) alpha[seed] eta[sigma] - (sigma[seed] - sigma[sigma]) alpha[seed] pi_old[sigma] 
                pi_new_set[j] = eta_set[j] + pi_old_set[j];    // pi_new[sigma] <- eta[sigma] + pi_old[sigma] 
                alpha_set[j] = (pi_old_set[j] / pi_new_set[j]) * alpha_set[seed];     // alpha[sigma] <- (pi_old[sigma] / pi_new[sigma]) alpha[seed] 
                omega_set[j] = omega_set[seed] / (1.0 - omega_set[seed] * (sigma[seed] - sigma[j]));      // omega[sigma] <- omega[0] / (1.0 + omega[0] * sigma) 
            }
            my_openmp_daxpy(vec_loc_size, omega_set[j] / (pi_new_set[j] * zeta_set[j]), q_loc_copy, &x_loc_set[j * vec_loc_size]);  // x[sigma] <- x[sigma] + alpha[sigma] p[sigma] + omega[sigma] / (pi_new[sigma] zeta[sigma]) q 
            my_openmp_daxpy(vec_loc_size, alpha_set[j], &p_loc_set[j * vec_loc_size], &x_loc_set[j * vec_loc_size]);
            my_openmp_daxpy(vec_loc_size, omega_set[j] / (alpha_set[j] * zeta_set[j] * pi_new_set[j]), q_loc_copy, &p_loc_set[j * vec_loc_size]); // p[sigma] <- p[sigma] + omega[sigma] / (alpha[sigma] zeta[sigma]) (q / pi_new[sigma] - r_old / pi_old[sigma]) 
            my_openmp_daxpy(vec_loc_size, -omega_set[j] / (alpha_set[j] * zeta_set[j] * pi_old_set[j]), r_old_loc, &p_loc_set[j * vec_loc_size]);
            #pragma omp single
            {
                zeta_set[j] = (1.0 - omega_set[seed] * (sigma[seed] - sigma[j])) * zeta_set[j];      // zeta[sigma] <- (1.0 - omega[seed] (sigma[seed] - sigma[sigma])) * zeta[sigma] 
                beta_set[j] = (pi_old_set[j] / pi_new_set[j]) * (pi_old_set[j] / pi_new_set[j]) * beta_set[seed]; // beta[sigma] <- (pi_old[sigma] / pi_new[sigma])^2 beta[seed] 
            }
            my_openmp_dscal(vec_loc_size, beta_set[j], &p_loc_set[j * vec_loc_size]);      // p[sigma] <- 1 / (pi_new[sigma] zeta[sigma]) r + beta[sigma] p[sigma] 
            my_openmp_daxpy(vec_loc_size, 1.0 / (pi_new_set[j] * zeta_set[j]), r_loc, &p_loc_set[j * vec_loc_size]);
        }

#ifdef MEASURE_SECTION_TIME
        #pragma omp master
        {
            section_end_time = MPI_Wtime();
            shift_time += section_end_time - section_start_time;
        }
#endif

        // ==== 収束判定 ====

        #pragma omp master
        {

#ifdef DISPLAY_SIGMA_RESIDUAL
            if (myid == 0 && k % OUT_ITER == 0) printf("Iter %d : ", k);
#endif

#ifdef MEASURE_SECTION_TIME
            section_start_time = MPI_Wtime();
#endif

            for (j = 0; j < sigma_len; j++) {
                if (stop_flag[j]) {
#ifdef DISPLAY_SIGMA_RESIDUAL
                    if (myid == 0 && k % OUT_ITER == 0) printf("------------ ");
#endif
                    continue;
                }
                if (j == seed) {
                    abs_zeta_pi = 1.0;
                } else {
                    abs_zeta_pi = fabs(1.0 / (zeta_set[j] * pi_new_set[j]));
                }
#ifdef DISPLAY_SIGMA_RESIDUAL
                if (myid == 0 && k % OUT_ITER == 0) printf("%e ", abs_zeta_pi * sqrt(global_dot_r / global_dot_zero));
#endif
                if (abs_zeta_pi * abs_zeta_pi * global_dot_r <= tol * tol * global_dot_zero) {
                    stop_flag[j] = true;
                    stop_count++;
                }
            }

#ifdef MEASURE_SECTION_TIME
            section_end_time = MPI_Wtime();
            shift_time += section_end_time - section_start_time;
#endif

#ifdef DISPLAY_SIGMA_RESIDUAL
            if (myid == 0 && k % OUT_ITER == 0) printf("\n");
#endif

            k++;

#ifdef DISPLAY_RESIDUAL
            if (myid == 0 && k % OUT_ITER == 0) {
                printf("Iteration: %d, Residual: %e\n", k, sqrt(global_dot_r / global_dot_zero));
            }
#endif

        }
        #pragma omp barrier
    }
}

#if defined(MEASURE_TIME) || defined(MEASURE_SECTION_TIME)
    end_time = MPI_Wtime();
    total_time = end_time - start_time;
#endif

#ifdef MEASURE_SECTION_TIME
    seed_time = total_time - shift_time;
#endif


    // ==== 結果表示 ====

    if (myid == 0) {
#ifdef DISPLAY_RESULT
        printf("Total iter   : %d\n", k - 1);
        printf("Final r      : %e\n", sqrt(global_dot_r / global_dot_zero));
#endif
#ifdef MEASURE_TIME
        printf("Total time   : %e [sec.] \n", total_time);
        printf("Avg time/iter: %e [sec.] \n", total_time / k);
#endif
#ifdef MEASURE_SECTION_TIME
        printf("Seed t/iter  : %e [sec.]\n", seed_time / k);
        printf("Shift t/iter : %e [sec.]\n", shift_time / k);
#endif
    }

    free(r_old_loc); free(r_hat_loc); free(s_loc); free(y_loc); free(q_loc_copy);
    free(vec);
    free(p_loc_set); free(alpha_set); free(beta_set); free(omega_set); free(eta_set); free(zeta_set); free(pi_old_set); free(pi_new_set);
    free(stop_flag);

    return k;

}


int shifted_lopbicg_switching_noovlp(CSR_Matrix *A_loc_diag, CSR_Matrix *A_loc_offd, INFO_Matrix *A_info, double *x_loc_set, double *r_loc, double *sigma, int sigma_len, int seed) {

    int myid; MPI_Comm_rank(MPI_COMM_WORLD, &myid);

#if defined(MEASURE_TIME) || defined(MEASURE_SECTION_TIME)
    double start_time, end_time, total_time;
#endif

    if (A_info->cols != A_info->rows) {
        printf("Error: matrix is not square.\n");
        exit(1);
    }

    int vec_size = A_info->rows;
    int vec_loc_size = A_loc_diag->rows;

    int i, j;

    int k, max_iter, stop_count;
    double tol;
    double max_zeta_pi, abs_zeta_pi;
    int max_sigma;

    double *r_old_loc, *r_hat_loc, *s_loc, *y_loc, *vec, *q_loc_copy;

    double *p_loc_set;
    double *alpha_set, *beta_set, *omega_set, *eta_set, *zeta_set;
    double alpha_old, beta_old;

    double *alpha_seed_archive, *beta_seed_archive, *omega_seed_archive;
    double *pi_archive_set;

    //double dot_r, dot_zero, rTr, rTs, qTq, qTy, rTr_old;
    double global_dot_r, global_dot_zero, global_rTr, global_rTs, global_qTq, global_qTy, global_rTr_old;

    MPI_Request dot_r_req, rTr_req, rTs_req, qTq_req, qTy_req, vec_req;

    bool *stop_flag;

    k = 1;
    tol = EPS;
    max_iter = MAX_ITER + 1;
    stop_count = 0;

    r_old_loc   = (double *)malloc(vec_loc_size * sizeof(double));
    r_hat_loc   = (double *)malloc(vec_loc_size * sizeof(double));
    s_loc       = (double *)malloc(vec_loc_size * sizeof(double));
    y_loc       = (double *)malloc(vec_loc_size * sizeof(double));
    q_loc_copy  = (double *)malloc(vec_loc_size * sizeof(double));

    vec         = (double *)malloc(vec_size * sizeof(double));

    p_loc_set   = (double *)calloc(vec_loc_size * sigma_len, sizeof(double)); // 一応ゼロで初期化(下でもOK) 
    //p_loc_set   = (double *)malloc(vec_loc_size * sigma_len * sizeof(double));
    alpha_set   = (double *)malloc(sigma_len * sizeof(double));
    beta_set    = (double *)malloc(sigma_len * sizeof(double));
    omega_set   = (double *)malloc(sigma_len * sizeof(double));
    eta_set     = (double *)malloc(sigma_len * sizeof(double));
    zeta_set    = (double *)malloc(sigma_len * sizeof(double));


    // seed switching で使うので履歴を保存する 
    alpha_seed_archive  = (double *)malloc(max_iter * sizeof(double));
    beta_seed_archive   = (double *)malloc(max_iter * sizeof(double));
    omega_seed_archive  = (double *)malloc(max_iter * sizeof(double));
    pi_archive_set      = (double *)malloc(max_iter * sigma_len * sizeof(double));

    stop_flag   = (bool *)calloc(sigma_len, sizeof(bool)); // Falseで初期化 

#ifdef MEASURE_SECTION_TIME
        double seed_time, shift_time, switch_time;
        double agv_1_time, mult_diag_1_time, mult_offd_1_time;
        double agv_2_time, mult_diag_2_time, mult_offd_2_time;
        double ared_time;
        double section_start_time, section_end_time;
        double seed_start_time, seed_end_time;

        double seed_iter_time, shift_iter_time;
        double agv_1_iter_time, mult_diag_1_iter_time, mult_offd_1_iter_time;
        double agv_2_iter_time, mult_diag_2_iter_time, mult_offd_2_iter_time;
        double ared_iter_time;

        seed_time = 0; shift_time = 0; switch_time = 0;
        agv_1_time = 0; mult_diag_1_time = 0; mult_offd_1_time = 0;
        agv_2_time = 0; mult_diag_2_time = 0; mult_offd_2_time = 0;
        ared_time = 0;
#endif

    global_rTr = my_ddot(vec_loc_size, r_loc, r_loc);      // (r#,r) 
    MPI_Iallreduce(MPI_IN_PLACE, &global_rTr, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &rTr_req);
    my_dcopy(vec_loc_size, r_loc, r_hat_loc);   // r# <- r = b 
    for (i = 0; i < sigma_len; i++) {
        my_dcopy(vec_loc_size, r_loc, &p_loc_set[i * vec_loc_size]);    // p[sigma] <- b 
        alpha_set[i]  = 1.0;  // alpha[sigma]  <- 1 
        beta_set[i]   = 0.0;  // beta[sigma]   <- 0 
        eta_set[i]    = 0.0;  // eta[sigma]    <- 0 
        pi_archive_set[i * max_iter + 0] = 1.0;
        pi_archive_set[i * max_iter + 1] = 1.0;
        zeta_set[i]   = 1.0;  // zeta[sigma]   <- 1 
    }
    my_dcopy(vec_loc_size, r_loc, &p_loc_set[seed * vec_loc_size]);
    MPI_Allgatherv(&p_loc_set[seed * vec_loc_size], vec_loc_size, MPI_DOUBLE, vec, A_info->recvcounts, A_info->displs, MPI_DOUBLE, MPI_COMM_WORLD);
    MPI_Wait(&rTr_req, MPI_STATUS_IGNORE);

    global_dot_r = global_rTr;    // (r,r) 
    global_dot_zero = global_rTr; // (r#,r#) 
    max_zeta_pi = 1.0;   // max(|1/(zeta pi)|) 

    alpha_seed_archive[0] = 1.0;
    beta_seed_archive[0]  = 0.0;

#if defined(MEASURE_TIME) || defined(MEASURE_SECTION_TIME)
        start_time = MPI_Wtime();
#endif

#ifdef DISPLAY_SIGMA_RESIDUAL
        if (myid == 0) printf("Seed : %d\n", seed);
#endif

#pragma omp parallel private(j)  // スレッドの生成
{
    double local_dot_r, local_dot_zero, local_rTr, local_rTs, local_qTq, local_qTy, local_rTr_old;

    while (stop_count < sigma_len && k < max_iter) {

#ifdef MEASURE_SECTION_TIME
        #pragma omp master
        {
            seed_start_time = MPI_Wtime();
        }
#endif

        // ===== グローバル変数の初期化 =====
        #pragma omp single
        {
            global_rTs = 0.0;
            global_qTq = 0.0;
            global_qTy = 0.0;
            global_dot_r = 0.0;
        }

        // ===== ベクトルのコピー =====
        my_openmp_dcopy(vec_loc_size, r_loc, r_old_loc);       // r_old <- r 

        // ===== r# <- (A + sigma[seed] I) p[seed] =====
        //MPI_csr_spmv_ovlap(A_loc_diag, A_loc_offd, A_info, &p_loc_set[seed * vec_loc_size], vec, s_loc);  // s <- (A + sigma[seed] I) p[seed] 

/*
#ifdef MEASURE_SECTION_TIME
        #pragma omp master
        {
            section_start_time = MPI_Wtime();
        }
#endif

        if (k != 0) {
            #pragma omp master
            {
                MPI_Iallgatherv(&p_loc_set[seed * vec_loc_size], vec_loc_size, MPI_DOUBLE, vec, A_info->recvcounts, A_info->displs, MPI_DOUBLE, MPI_COMM_WORLD, &vec_req);
                MPI_Wait(&vec_req, MPI_STATUS_IGNORE);
            }
        }

#ifdef MEASURE_SECTION_TIME
        #pragma omp master
        {
        section_end_time = MPI_Wtime();
        agv_1_time += section_end_time - section_start_time;
        agv_1_iter_time = section_end_time - section_start_time;
        }
#endif
*/

#ifdef MEASURE_SECTION_TIME
        #pragma omp master
        {
        section_start_time = MPI_Wtime();
        }
#endif

        // s_locの初期化
        #pragma omp for
        for (int l = 0; l < vec_loc_size; l++) {
            s_loc[l] = 0.0;
        }
        // 対角ブロックとローカルベクトルの積
        openmp_mult(A_loc_diag, &p_loc_set[seed * vec_loc_size], s_loc);

#ifdef MEASURE_SECTION_TIME
        #pragma omp master
        {
        section_end_time = MPI_Wtime();
        mult_diag_1_time += section_end_time - section_start_time;
        mult_diag_1_iter_time = section_end_time - section_start_time;
        }
#endif

#ifdef MEASURE_SECTION_TIME
        #pragma omp master
        {
            section_start_time = MPI_Wtime();
        }
#endif

        // ベクトルの集約を待機
        if (k != 1) {
            #pragma omp master
            {
                MPI_Wait(&vec_req, MPI_STATUS_IGNORE);
            }
            #pragma omp barrier
        }

#ifdef MEASURE_SECTION_TIME
        #pragma omp master
        {
        section_end_time = MPI_Wtime();
        agv_1_time += section_end_time - section_start_time;
        agv_1_iter_time = section_end_time - section_start_time;
        }
#endif

#ifdef MEASURE_SECTION_TIME
        #pragma omp master
        {
            section_start_time = MPI_Wtime();
        }
#endif

        // 非対角ブロックと集約ベクトルの積
        openmp_mult(A_loc_offd, vec, s_loc);

#ifdef MEASURE_SECTION_TIME
        #pragma omp master
        {
        section_end_time = MPI_Wtime();
        mult_offd_1_time += section_end_time - section_start_time;
        mult_offd_1_iter_time = section_end_time - section_start_time;
        }
#endif

        my_openmp_daxpy(vec_loc_size, sigma[seed], &p_loc_set[seed * vec_loc_size], s_loc);

        // ===== rTs = (r_hat, s) =====
        local_rTs = my_openmp_ddot(vec_loc_size, r_hat_loc, s_loc);
        #pragma omp atomic
        global_rTs += local_rTs;
        #pragma omp barrier

#ifdef MEASURE_SECTION_TIME
        #pragma omp master
        {
        section_start_time = MPI_Wtime();
        }
#endif

        #pragma omp master
        {
            MPI_Allreduce(MPI_IN_PLACE, &global_rTs, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);  // rTs <- (r#,s) 
            //MPI_Iallreduce(MPI_IN_PLACE, &global_rTs, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &rTs_req);  // rTs <- (r#,s) 
            //MPI_Wait(&rTs_req, MPI_STATUS_IGNORE);
        }
        #pragma omp barrier

#ifdef MEASURE_SECTION_TIME
        #pragma omp master
        {
        section_end_time = MPI_Wtime();
        ared_time += section_end_time - section_start_time;
        ared_iter_time = section_end_time - section_start_time;
        }
#endif

        // ===== alpha[seed] の更新 =====
        #pragma omp single
        {
            alpha_seed_archive[k] = global_rTr / global_rTs;   // alpha[seed] <- (r#,r)/(r#,s) 
        }

        // ===== q <- r - alpha[seed] s =====
        my_openmp_daxpy(vec_loc_size, -alpha_seed_archive[k], s_loc, r_loc);   // q <- r - alpha[seed] s 
        my_openmp_dcopy(vec_loc_size, r_loc, q_loc_copy); // q_copy <- q (q_copyにr_locをコピー　シード方程式を一つにまとめるため)

        // ===== y <- (A + sigma[seed] I) q =====
        //MPI_openmp_csr_spmv_ovlap(A_loc_diag, A_loc_offd, A_info, r_loc, vec, y_loc);  // y <- (A + sigma[seed] I) q 

/*
#ifdef MEASURE_SECTION_TIME
        #pragma omp master
        {
        section_start_time = MPI_Wtime();
        }
#endif
        #pragma omp master
        {
            MPI_Iallgatherv(r_loc, vec_loc_size, MPI_DOUBLE, vec, A_info->recvcounts, A_info->displs, MPI_DOUBLE, MPI_COMM_WORLD, &vec_req);
            MPI_Wait(&vec_req, MPI_STATUS_IGNORE);
        }

#ifdef MEASURE_SECTION_TIME
        #pragma omp master
        {
        section_end_time = MPI_Wtime();
        agv_2_time += section_end_time - section_start_time;
        agv_2_iter_time = section_end_time - section_start_time;
        }
#endif
*/

        #pragma omp master
        {
            MPI_Iallgatherv(r_loc, vec_loc_size, MPI_DOUBLE, vec, A_info->recvcounts, A_info->displs, MPI_DOUBLE, MPI_COMM_WORLD, &vec_req);
        }

#ifdef MEASURE_SECTION_TIME
        #pragma omp master
        {
            section_start_time = MPI_Wtime();
        }
#endif

        // s_locの初期化
        #pragma omp for
        for (int l = 0; l < vec_loc_size; l++) {
            y_loc[l] = 0.0;
        }
        // 対角ブロックとローカルベクトルの積
        openmp_mult(A_loc_diag, r_loc, y_loc);

#ifdef MEASURE_SECTION_TIME
        #pragma omp master
        {
        section_end_time = MPI_Wtime();
        mult_diag_2_time += section_end_time - section_start_time;
        mult_diag_2_iter_time = section_end_time - section_start_time;
        }
#endif

#ifdef MEASURE_SECTION_TIME
        #pragma omp master
        {
        section_start_time = MPI_Wtime();
        }
#endif

        // ベクトルの集約を待機
        #pragma omp master
        {
            MPI_Wait(&vec_req, MPI_STATUS_IGNORE);
        }
        #pragma omp barrier

#ifdef MEASURE_SECTION_TIME
        #pragma omp master
        {
        section_end_time = MPI_Wtime();
        agv_2_time += section_end_time - section_start_time;
        agv_2_iter_time = section_end_time - section_start_time;
        }
#endif

#ifdef MEASURE_SECTION_TIME
        #pragma omp master
        {
            section_start_time = MPI_Wtime();
        }
#endif

        // 非対角ブロックと集約ベクトルの積
        openmp_mult(A_loc_offd, vec, y_loc);

#ifdef MEASURE_SECTION_TIME
        #pragma omp master
        {
        section_end_time = MPI_Wtime();
        mult_offd_2_time += section_end_time - section_start_time;
        mult_offd_2_iter_time = section_end_time - section_start_time;
        }
#endif

        my_openmp_daxpy(vec_loc_size, sigma[seed], r_loc, y_loc);

        // ===== (q,q) と (q,y) の計算 =====
        local_qTq = my_openmp_ddot(vec_loc_size, r_loc, r_loc);
        #pragma omp atomic
        global_qTq += local_qTq;
        #pragma omp barrier
/*
        #pragma omp master
        {
            MPI_Iallreduce(MPI_IN_PLACE, &global_qTq, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &qTq_req);  // (q,q) 
        }
*/
        local_qTy = my_openmp_ddot(vec_loc_size, r_loc, y_loc);
        #pragma omp atomic
        global_qTy += local_qTy;
        #pragma omp barrier

#ifdef MEASURE_SECTION_TIME
        #pragma omp master
        {
        section_start_time = MPI_Wtime();
        }
#endif

        #pragma omp master
        {
            MPI_Iallreduce(MPI_IN_PLACE, &global_qTq, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &qTq_req);  // (q,q) 
            MPI_Iallreduce(MPI_IN_PLACE, &global_qTy, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &qTy_req);  // (q,y) 
            MPI_Wait(&qTq_req, MPI_STATUS_IGNORE);
            MPI_Wait(&qTy_req, MPI_STATUS_IGNORE);
        }
        #pragma omp barrier

#ifdef MEASURE_SECTION_TIME
        #pragma omp master
        {
        section_end_time = MPI_Wtime();
        ared_time += section_end_time - section_start_time;
        ared_iter_time += section_end_time - section_start_time;
        }
#endif

        // ===== omega[seed] の更新 =====
        #pragma omp single
        {
            omega_seed_archive[k] = global_qTq / global_qTy;  // omega[seed] <- (q,q)/(q,y) 
        }

        // ===== x[seed] の更新 =====
        my_openmp_daxpy(vec_loc_size, alpha_seed_archive[k], &p_loc_set[seed * vec_loc_size], &x_loc_set[seed * vec_loc_size]);     // x[seed] <- x[seed] + alpha[seed] p[seed] + omega[seed] q 
        my_openmp_daxpy(vec_loc_size, omega_seed_archive[k], r_loc, &x_loc_set[seed * vec_loc_size]);

        // ==== r の更新 ====
        my_openmp_daxpy(vec_loc_size, -omega_seed_archive[k], y_loc, r_loc);            // r <- q - omega[seed] y 

        // ==== (r,r) の計算 ====
        local_dot_r = my_openmp_ddot(vec_loc_size, r_loc, r_loc);
        #pragma omp atomic
        global_dot_r += local_dot_r;
        #pragma omp barrier
        #pragma omp master
        {
            //MPI_Iallreduce(MPI_IN_PLACE, &global_dot_r, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &dot_r_req);  // (r,r) 
            global_rTr_old = global_rTr;      // r_old <- (r#,r) 
        }

        // ==== (r#,r) の計算 ====
        local_rTr = my_openmp_ddot(vec_loc_size, r_hat_loc, r_loc);
        #pragma omp single
        {
            global_rTr = 0.0;
        }
        #pragma omp atomic
        global_rTr += local_rTr;
        #pragma omp barrier

#ifdef MEASURE_SECTION_TIME
        #pragma omp master
        {
        section_start_time = MPI_Wtime();
        }
#endif

        #pragma omp master
        {
            MPI_Iallreduce(MPI_IN_PLACE, &global_dot_r, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &dot_r_req);  // (r,r) 
            MPI_Iallreduce(MPI_IN_PLACE, &global_rTr, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &rTr_req);  // (r#,r) 
            MPI_Wait(&dot_r_req, MPI_STATUS_IGNORE);
            MPI_Wait(&rTr_req, MPI_STATUS_IGNORE);
        }
        #pragma omp barrier

#ifdef MEASURE_SECTION_TIME
        #pragma omp master
        {
        section_end_time = MPI_Wtime();
        ared_time += section_end_time - section_start_time;
        ared_iter_time += section_end_time - section_start_time;
        }
#endif

        // ==== beta[seed] の更新 ====
        #pragma omp single
        {
            beta_seed_archive[k] = (alpha_seed_archive[k] / omega_seed_archive[k]) * (global_rTr / global_rTr_old);   // beta[seed] <- (alpha[seed] / omega[seed]) ((r#,r)/(r#,r)) 
        }

        // ==== p[seed] の更新 ====
        my_openmp_dscal(vec_loc_size, beta_seed_archive[k], &p_loc_set[seed * vec_loc_size]);     // p[seed] <- r + beta[seed] p[seed] - beta[seed] omega[seed] s 
        my_openmp_daxpy(vec_loc_size, 1.0, r_loc, &p_loc_set[seed * vec_loc_size]);
        my_openmp_daxpy(vec_loc_size, -beta_seed_archive[k] * omega_seed_archive[k], s_loc, &p_loc_set[seed * vec_loc_size]);


        // ==== 行列ベクトル積のためのベクトルqの集約を開始 ====
        #pragma omp master
        {
            MPI_Iallgatherv(&p_loc_set[seed * vec_loc_size], vec_loc_size, MPI_DOUBLE, vec, A_info->recvcounts, A_info->displs, MPI_DOUBLE, MPI_COMM_WORLD, &vec_req);
        }


        // ===== シフト方程式 =====

#ifdef MEASURE_SECTION_TIME
        #pragma omp master
        {
        seed_end_time = MPI_Wtime();
        seed_time += seed_end_time - seed_start_time;
        seed_iter_time = seed_end_time - seed_start_time;
        section_start_time = MPI_Wtime();
        }
#endif

        for (j = 0; j < sigma_len; j++) {
            if (j == seed) continue;
            if (stop_flag[j]) continue;
            #pragma omp single
            {
                eta_set[j] = (beta_seed_archive[k - 1] / alpha_seed_archive[k - 1]) * alpha_seed_archive[k] * eta_set[j] - (sigma[seed] - sigma[j]) * alpha_seed_archive[k] * pi_archive_set[j * max_iter + (k - 1)];
                // eta[sigma] = (beta_old / alpha_old) alpha[seed] eta[sigma] - (sigma[seed] - sigma[sigma]) alpha[seed] pi_old[sigma] 
                pi_archive_set[j * max_iter + k] = eta_set[j] + pi_archive_set[j * max_iter + (k - 1)];    // pi_new[sigma] <- eta[sigma] + pi_old[sigma] 
                alpha_set[j] = (pi_archive_set[j * max_iter + (k - 1)] / pi_archive_set[j * max_iter + k]) * alpha_seed_archive[k];     // alpha[sigma] <- (pi_old[sigma] / pi_new[sigma]) alpha[seed] 
                omega_set[j] = omega_seed_archive[k] / (1.0 - omega_seed_archive[k] * (sigma[seed] - sigma[j]));      // omega[sigma] <- omega[0] / (1.0 + omega[0] * sigma) 
            }
            my_openmp_daxpy(vec_loc_size, omega_set[j] / (pi_archive_set[j * max_iter + k] * zeta_set[j]), q_loc_copy, &x_loc_set[j * vec_loc_size]);     // x[sigma] <- x[sigma] + alpha[sigma] p[sigma] + omega[sigma] / (pi_new[sigma] zeta[sigma]) q 
            my_openmp_daxpy(vec_loc_size, alpha_set[j], &p_loc_set[j * vec_loc_size], &x_loc_set[j * vec_loc_size]);
            my_openmp_daxpy(vec_loc_size, omega_set[j] / (alpha_set[j] * zeta_set[j] * pi_archive_set[j * max_iter + k]), q_loc_copy, &p_loc_set[j * vec_loc_size]);    // p[sigma] <- p[sigma] + omega[sigma] / (alpha[sigma] zeta[sigma]) (q / pi_new[sigma] - r_old / pi_old[sigma]) 
            my_openmp_daxpy(vec_loc_size, -omega_set[j] / (alpha_set[j] * zeta_set[j] * pi_archive_set[j * max_iter + (k - 1)]), r_old_loc, &p_loc_set[j * vec_loc_size]);
            #pragma omp single
            {
                zeta_set[j] = (1.0 - omega_seed_archive[k] * (sigma[seed] - sigma[j])) * zeta_set[j];      // zeta[sigma] <- (1.0 - omega[seed] (sigma[seed] - sigma[sigma])) * zeta[sigma] 
                beta_set[j] = (pi_archive_set[j * max_iter + (k - 1)] / pi_archive_set[j * max_iter + k]) * (pi_archive_set[j * max_iter + (k - 1)] / pi_archive_set[j * max_iter + k]) * beta_seed_archive[k]; // beta[sigma] <- (pi_old[sigma] / pi_new[sigma])^2 beta[seed] 
            }
            my_openmp_dscal(vec_loc_size, beta_set[j], &p_loc_set[j * vec_loc_size]);      // p[sigma] <- 1 / (pi_new[sigma] zeta[sigma]) r + beta[sigma] p[sigma] 
            my_openmp_daxpy(vec_loc_size, 1.0 / (pi_archive_set[j * max_iter + k] * zeta_set[j]), r_loc, &p_loc_set[j * vec_loc_size]);
        }

#ifdef MEASURE_SECTION_TIME
        #pragma omp master
        {
        section_end_time = MPI_Wtime();
        shift_time += section_end_time - section_start_time;
        shift_iter_time = section_end_time - section_start_time;
        }
#endif

#ifdef DISPLAY_SECTION_TIME
        #pragma omp master
        {
        if (myid == 0 && k == 1) {
            printf("iter, unsolved, seed, agv_1, mult_diag_1, mult_offd_1, agv_2, mult_diag_2, mult_offd_2, ared, shift\n");
        }

        if (myid == 0 && k % OUT_ITER == 0) {
            printf("%d, %d, %e, %e, %e, %e, %e, %e, %e, %e, %e\n", k, sigma_len - stop_count, seed_iter_time, agv_1_iter_time, mult_diag_1_iter_time, mult_offd_1_iter_time, agv_2_iter_time, mult_diag_2_iter_time, mult_offd_2_iter_time, ared_iter_time, shift_iter_time);
        }
        }
#endif

        // ==== 収束判定 ====

        #pragma omp master
        {

#ifdef DISPLAY_SIGMA_RESIDUAL
            if (myid == 0 && k % OUT_ITER == 0) printf("Iter %d : ", k);
#endif

#ifdef MEASURE_SECTION_TIME
            section_start_time = MPI_Wtime();
#endif

            max_zeta_pi = 1.0;
            for (j = 0; j < sigma_len; j++) {
                if (stop_flag[j]) {
#ifdef DISPLAY_SIGMA_RESIDUAL
                    if (myid == 0 && k % OUT_ITER == 0) printf("------------ ");
#endif
                    continue;
                }
                if (j == seed) {
                    abs_zeta_pi = 1.0;
                } else {
                    abs_zeta_pi = fabs(1.0 / (zeta_set[j] * pi_archive_set[j * max_iter + k]));
                }
#ifdef DISPLAY_SIGMA_RESIDUAL
                if (myid == 0 && k % OUT_ITER == 0) printf("%e ", abs_zeta_pi * sqrt(global_dot_r / global_dot_zero));
#endif
                if (abs_zeta_pi * abs_zeta_pi * global_dot_r <= tol * tol * global_dot_zero) {
                    stop_flag[j] = true;
                    stop_count++;
                } else {
                    if (abs_zeta_pi > max_zeta_pi) {
                        max_zeta_pi = abs_zeta_pi;
                        max_sigma = j;
                    }
                }
            }

#ifdef MEASURE_SECTION_TIME
            section_end_time = MPI_Wtime();
            shift_time += section_end_time - section_start_time;
#endif

#ifdef DISPLAY_SIGMA_RESIDUAL
            if (myid == 0 && k % OUT_ITER == 0) printf("\n");
#endif

#ifdef MEASURE_SECTION_TIME
            section_start_time = MPI_Wtime();
#endif

            // seed switching 
            if (stop_flag[seed] && stop_count < sigma_len) {
#ifdef DISPLAY_SIGMA_RESIDUAL
                if (myid == 0) printf("Seed : %d\n", max_sigma);
#endif
                for (i = 1; i <= k; i++) {
                    alpha_seed_archive[i] = (pi_archive_set[max_sigma * max_iter + (i - 1)] / pi_archive_set[max_sigma * max_iter + i]) * alpha_seed_archive[i];
                    beta_seed_archive[i] = (pi_archive_set[max_sigma * max_iter + (i - 1)] / pi_archive_set[max_sigma * max_iter + i]) * (pi_archive_set[max_sigma * max_iter + (i - 1)] / pi_archive_set[max_sigma * max_iter + i]) * beta_seed_archive[i];
                    omega_seed_archive[i] = omega_seed_archive[i] / (1.0 - omega_seed_archive[i] * (sigma[seed] - sigma[max_sigma]));
                }
                my_dscal(vec_loc_size, 1.0 / (zeta_set[max_sigma] * pi_archive_set[max_sigma * max_iter + k]), r_loc);

                for (j = 0; j < sigma_len; j++) {
                    eta_set[j]    = 0.0;  // eta[sigma]    <- 0 
                    //pi_archive_set[j * max_iter + 0] = 1.0;
                    zeta_set[j]   = 1.0;  // zeta[sigma]   <- 1 
                }
                //alpha_seed_archive[0] = 1.0;
                //beta_seed_archive[0]  = 0.0;

                for (i = 1; i <= k; i++) {
                    for (j = 0; j < sigma_len; j++) {
                        if (stop_flag[j]) continue;
                        if (j == max_sigma) continue;
                        eta_set[j] = (beta_seed_archive[i - 1] / alpha_seed_archive[i - 1]) * alpha_seed_archive[i] * eta_set[j] - (sigma[max_sigma] - sigma[j]) * alpha_seed_archive[i] * pi_archive_set[j * max_iter + (i - 1)];
                        pi_archive_set[j * max_iter + i] = eta_set[j] + pi_archive_set[j * max_iter + (i - 1)];
                        zeta_set[j] = (1.0 - omega_seed_archive[i] * (sigma[max_sigma] - sigma[j])) * zeta_set[j];
                    }
                }

                seed = max_sigma;
            }

#ifdef MEASURE_SECTION_TIME
            section_end_time = MPI_Wtime();
            switch_time += section_end_time - section_start_time;
#endif

            k++;

#ifdef DISPLAY_RESIDUAL
            if (myid == 0 && k % OUT_ITER == 0) {
                printf("Iteration: %d, Residual: %e\n", k, sqrt(global_dot_r / global_dot_zero));
            }
#endif

        }
        #pragma omp barrier
    }
}

#if defined(MEASURE_TIME) || defined(MEASURE_SECTION_TIME)
    end_time = MPI_Wtime();
    total_time = end_time - start_time;
#endif

#ifdef MEASURE_SECTION_TIME
    seed_time = total_time - shift_time - switch_time;
#endif


    // ==== 結果表示 ====

    if (myid == 0) {
#ifdef DISPLAY_RESIDUAL
        printf("Total iter   : %d\n", k - 1);
        printf("Final r      : %e\n", sqrt(dot_r / dot_zero));
#endif
#ifdef MEASURE_TIME
        printf("Total time   : %e [sec.] \n", total_time);
        //printf("Avg time/iter: %e [sec.] \n", total_time / k);
#endif
#ifdef MEASURE_SECTION_TIME
        printf("Seed time    : %e [sec.]\n", seed_time);
        printf(" 1 Agv time   : %e [sec.]\n", agv_1_time);
        printf(" 1 Mult_diag  : %e [sec.]\n", mult_diag_1_time);
        printf(" 1 Mult_offd  : %e [sec.]\n", mult_offd_1_time);
        printf(" 2 Agv time   : %e [sec.]\n", agv_2_time);
        printf(" 2 Mult_diag  : %e [sec.]\n", mult_diag_2_time);
        printf(" 2 Mult_offd  : %e [sec.]\n", mult_offd_2_time);
        printf(" Ared time    : %e [sec.]\n", ared_time);
        printf("Shift time   : %e [sec.]\n", shift_time);
        printf("Switch time  : %e [sec.]\n", switch_time);
#endif
    }

    free(r_old_loc); free(r_hat_loc); free(s_loc); free(y_loc); free(q_loc_copy);
    free(vec);
    free(p_loc_set); free(alpha_set); free(beta_set); free(omega_set); free(eta_set); free(zeta_set);
    free(alpha_seed_archive); free(beta_seed_archive); free(omega_seed_archive); free(pi_archive_set);
    free(stop_flag);

    return k;

}
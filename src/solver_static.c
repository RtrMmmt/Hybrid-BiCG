#include "solver.h"

int shifted_lopbicg_static(CSR_Matrix *A_loc_diag, CSR_Matrix *A_loc_offd, INFO_Matrix *A_info, double *x_loc_set, double *r_loc, double *sigma, int sigma_len, int seed) {

    int myid; MPI_Comm_rank(MPI_COMM_WORLD, &myid);

#if defined(MEASURE_TIME) || defined(MEASURE_SECTION_TIME)
    double start_time, end_time, total_time;
#endif

    if (A_info->cols != A_info->rows) {
        printf("Error: matrix is not square.\n");
        exit(1);
    }

    // ==== 変数の定義 ====
    int vec_size = A_info->rows;
    int vec_loc_size = A_loc_diag->rows;

    int j;

    int k, max_iter, stop_count;
    double tol;
    double max_zeta_pi, abs_zeta_pi;
    int max_sigma;

    double *r_old_loc, *r_hat_loc, *s_loc, *y_loc, *vec, *q_loc_copy;

    double *p_loc_set;
    double *alpha_set, *beta_set, *omega_set, *eta_set, *zeta_set;

    double *alpha_seed_archive, *beta_seed_archive, *omega_seed_archive;
    double *pi_archive_set;

    double global_dot_r, global_dot_zero, global_rTr, global_rTs, global_qTq, global_qTy, global_rTr_old;

    MPI_Request rTr_req;

    bool *stop_flag;

    k = 1;
    tol = EPS;
    max_iter = MAX_ITER + 1;
    stop_count = 0;

    // ==== メモリの確保 ====
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

#ifdef DISPLAY_ERROR
    double *ans_loc = (double *)malloc(vec_loc_size * sizeof(double));

    double *temp    = (double *)malloc(vec_loc_size * sizeof(double));
    for (int i = 0; i < vec_loc_size; i++) {
        temp[i] = 1.0;
    }
    MPI_csr_spmv_ovlap(A_loc_diag, A_loc_offd, A_info, temp, vec, ans_loc);
    my_daxpy(vec_loc_size, sigma[seed], temp, ans_loc);
    free(temp);

/*
    for (int i = 0; i < vec_loc_size; i++) {
        ans_loc[i] = 1; // 右辺ベクトルはすべて1
    }
*/
#endif

#ifdef MEASURE_SECTION_TIME
        double seed_time, shift_time, switch_time, agv_time;
        double seed_iter_time, matvec_iter_time, agv_iter_time;
        double section_start_time, section_end_time;
        double seed_start_time, seed_end_time;
        double agv_start_time, agv_end_time;
        seed_time = 0; shift_time = 0; switch_time = 0; agv_time = 0;
#endif

    // ==== 初期化 ====
    global_rTr = my_ddot(vec_loc_size, r_loc, r_loc);      // (r#,r) 
    MPI_Iallreduce(MPI_IN_PLACE, &global_rTr, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &rTr_req);
    my_dcopy(vec_loc_size, r_loc, r_hat_loc);   // r# <- r = b 
    for (int i = 0; i < sigma_len; i++) {
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

    double max_time;

    // ==== 反復計算 ====
#pragma omp parallel private(j)  // スレッドの生成
{
    double local_start_time, local_end_time, local_time;

    while (stop_count < sigma_len && k < max_iter) {

#ifdef MEASURE_SECTION_TIME
        #pragma omp master
        {
            seed_start_time = MPI_Wtime();
        }
#endif

        // ===== ベクトルのコピー =====
        my_openmp_dcopy(vec_loc_size, r_loc, r_old_loc);       // r_old <- r 

        // ===== r# <- (A + sigma[seed] I) p[seed] =====
/*
        openmp_set_vector_zero(vec_loc_size, s_loc);  // s_locの初期化
        openmp_mult(A_loc_diag, &p_loc_set[seed * vec_loc_size], s_loc);  // 対角ブロックとローカルベクトルの積
        #pragma omp barrier
        openmp_mult(A_loc_offd, vec, s_loc);  // 非対角ブロックと集約ベクトルの積
        #pragma omp barrier
*/
        #pragma omp master
        {
            MPI_csr_spmv_ovlap(A_loc_diag, A_loc_offd, A_info, &p_loc_set[seed * vec_loc_size], vec, s_loc);
        }
        #pragma omp barrier

        my_openmp_daxpy(vec_loc_size, sigma[seed], &p_loc_set[seed * vec_loc_size], s_loc);

        // ===== rTs = (r_hat, s) =====
        my_openmp_ddot_v2(vec_loc_size, r_hat_loc, s_loc, &global_rTs);
        #pragma omp master
        {
            MPI_Allreduce(MPI_IN_PLACE, &global_rTs, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);  // rTs <- (r#,s) 
        }
        #pragma omp barrier

        // ===== alpha[seed] の更新 =====
        //#pragma omp single
        #pragma omp master
        {
            alpha_seed_archive[k] = global_rTr / global_rTs;   // alpha[seed] <- (r#,r)/(r#,s) 
        }
        #pragma omp barrier

        // ===== q <- r - alpha[seed] s =====
        my_openmp_daxpy(vec_loc_size, -alpha_seed_archive[k], s_loc, r_loc);   // q <- r - alpha[seed] s 
        my_openmp_dcopy(vec_loc_size, r_loc, q_loc_copy); // q_copy <- q (q_copyにr_locをコピー　シード方程式を一つにまとめるため)

#ifdef MEASURE_SECTION_TIME
        #pragma omp master
        {
            section_start_time = MPI_Wtime();
        }
#endif

        // ===== y <- (A + sigma[seed] I) q =====
/*
        openmp_set_vector_zero(vec_loc_size, y_loc);  // y_locの初期化
        #pragma omp master
        {
            MPI_Allgatherv(r_loc, vec_loc_size, MPI_DOUBLE, vec, A_info->recvcounts, A_info->displs, MPI_DOUBLE, MPI_COMM_WORLD);
        }
        //openmp_mult(A_loc_diag, r_loc, y_loc);  // 対角ブロックとローカルベクトルの積
        openmp_mult_dynamic(A_loc_diag, r_loc, y_loc);
        #pragma omp barrier
        openmp_mult(A_loc_offd, vec, y_loc);  // 非対角ブロックと集約ベクトルの積
        #pragma omp barrier
*/
        #pragma omp master
        {
            MPI_csr_spmv_ovlap(A_loc_diag, A_loc_offd, A_info, r_loc, vec, y_loc);
        }
        #pragma omp barrier

        my_openmp_daxpy(vec_loc_size, sigma[seed], r_loc, y_loc);

#ifdef MEASURE_SECTION_TIME
        #pragma omp master
        {
            section_end_time = MPI_Wtime();
            matvec_iter_time = section_end_time - section_start_time;
        }
#endif

        // ===== (q,q) と (q,y) の計算 =====
/*
        my_openmp_ddot_v2(vec_loc_size, r_loc, r_loc, &global_qTq);
        my_openmp_ddot_v2(vec_loc_size, r_loc, y_loc, &global_qTy);
*/
        #pragma omp master
        {
            global_qTq = my_ddot(vec_loc_size, r_loc, r_loc);      // (r,r) 
            global_qTy = my_ddot(vec_loc_size, r_loc, y_loc);      // (r,y)
        }
        #pragma omp barrier

        #pragma omp master
        {
            MPI_Allreduce(MPI_IN_PLACE, &global_qTq, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);  // (q,q) 
            MPI_Allreduce(MPI_IN_PLACE, &global_qTy, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);  // (q,y) 
        }
        #pragma omp barrier

        // ===== omega[seed] の更新 =====
        //#pragma omp single
        #pragma omp master
        {
            omega_seed_archive[k] = global_qTq / global_qTy;  // omega[seed] <- (q,q)/(q,y) 
        }
        #pragma omp barrier

        // ===== x[seed] の更新 =====
        my_openmp_daxpy(vec_loc_size, alpha_seed_archive[k], &p_loc_set[seed * vec_loc_size], &x_loc_set[seed * vec_loc_size]);     // x[seed] <- x[seed] + alpha[seed] p[seed] + omega[seed] q 
        my_openmp_daxpy(vec_loc_size, omega_seed_archive[k], r_loc, &x_loc_set[seed * vec_loc_size]);

        // ==== r の更新 ====
        my_openmp_daxpy(vec_loc_size, -omega_seed_archive[k], y_loc, r_loc);            // r <- q - omega[seed] y 

        // ==== (r,r) と (r#,r) の計算 ====
        //#pragma omp single
        #pragma omp master
        {
            global_rTr_old = global_rTr; 
        }
        #pragma omp barrier
/*
        my_openmp_ddot_v2(vec_loc_size, r_loc, r_loc, &global_dot_r);
        my_openmp_ddot_v2(vec_loc_size, r_hat_loc, r_loc, &global_rTr);
*/
        #pragma omp master
        {
            global_dot_r = my_ddot(vec_loc_size, r_loc, r_loc);      // (r,r) 
            global_rTr = my_ddot(vec_loc_size, r_hat_loc, r_loc);    // (r#,r) 
        }
        #pragma omp barrier

        #pragma omp master
        {
            MPI_Allreduce(MPI_IN_PLACE, &global_dot_r, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);  // (r,r) 
            MPI_Allreduce(MPI_IN_PLACE, &global_rTr, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);  // (r#,r) 
        }
        #pragma omp barrier

        // ==== beta[seed] の更新 ====
        //#pragma omp single
        #pragma omp master
        {
            beta_seed_archive[k] = (alpha_seed_archive[k] / omega_seed_archive[k]) * (global_rTr / global_rTr_old);   // beta[seed] <- (alpha[seed] / omega[seed]) ((r#,r)/(r#,r)) 
        }
        #pragma omp barrier

        // ==== p[seed] の更新 ====
        my_openmp_dscal(vec_loc_size, beta_seed_archive[k], &p_loc_set[seed * vec_loc_size]);     // p[seed] <- r + beta[seed] p[seed] - beta[seed] omega[seed] s 
        my_openmp_daxpy(vec_loc_size, 1.0, r_loc, &p_loc_set[seed * vec_loc_size]);
        my_openmp_daxpy(vec_loc_size, -beta_seed_archive[k] * omega_seed_archive[k], s_loc, &p_loc_set[seed * vec_loc_size]);

#ifdef MEASURE_SECTION_TIME
        #pragma omp master
        {
            seed_end_time = MPI_Wtime();
            seed_iter_time = seed_end_time - seed_start_time;
            seed_time += seed_iter_time;
        }
#endif

#ifdef MEASURE_SECTION_TIME
        #pragma omp master
        {
            agv_start_time = MPI_Wtime();
        }
#endif

        // ==== 行列ベクトル積のための通信をオーバーラップ ====
/*
        #pragma omp master
        {
            if (global_dot_r > tol * tol * global_dot_zero) { // seed switching を行わない場合 <-> seed switching を行う場合はスイッチ後に Allgatherv
                MPI_Allgatherv(&p_loc_set[seed * vec_loc_size], vec_loc_size, MPI_DOUBLE, vec, A_info->recvcounts, A_info->displs, MPI_DOUBLE, MPI_COMM_WORLD);
            }
        }
        #pragma omp barrier
*/

#ifdef MEASURE_SECTION_TIME
        #pragma omp master
        {
            agv_end_time = MPI_Wtime();
            agv_iter_time = agv_end_time - agv_start_time;
            //agv_time += agv_iter_time;
            //section_start_time = MPI_Wtime();
        }

        #pragma omp barrier
        max_time = 0.0;
        local_start_time = MPI_Wtime();
#endif

        // ===== シフト方程式 =====
        #pragma omp for
        for (j = 0; j < sigma_len; j++) {
            if (j == seed) continue;
            if (stop_flag[j]) continue;
            eta_set[j] = (beta_seed_archive[k - 1] / alpha_seed_archive[k - 1]) * alpha_seed_archive[k] * eta_set[j] - (sigma[seed] - sigma[j]) * alpha_seed_archive[k] * pi_archive_set[j * max_iter + (k - 1)];
            // eta[sigma] = (beta_old / alpha_old) alpha[seed] eta[sigma] - (sigma[seed] - sigma[sigma]) alpha[seed] pi_old[sigma] 
            pi_archive_set[j * max_iter + k] = eta_set[j] + pi_archive_set[j * max_iter + (k - 1)];    // pi_new[sigma] <- eta[sigma] + pi_old[sigma] 
            alpha_set[j] = (pi_archive_set[j * max_iter + (k - 1)] / pi_archive_set[j * max_iter + k]) * alpha_seed_archive[k];     // alpha[sigma] <- (pi_old[sigma] / pi_new[sigma]) alpha[seed] 
            omega_set[j] = omega_seed_archive[k] / (1.0 - omega_seed_archive[k] * (sigma[seed] - sigma[j]));      // omega[sigma] <- omega[0] / (1.0 + omega[0] * sigma) 
            my_daxpy(vec_loc_size, omega_set[j] / (pi_archive_set[j * max_iter + k] * zeta_set[j]), q_loc_copy, &x_loc_set[j * vec_loc_size]);     // x[sigma] <- x[sigma] + alpha[sigma] p[sigma] + omega[sigma] / (pi_new[sigma] zeta[sigma]) q 
            my_daxpy(vec_loc_size, alpha_set[j], &p_loc_set[j * vec_loc_size], &x_loc_set[j * vec_loc_size]);
            my_daxpy(vec_loc_size, omega_set[j] / (alpha_set[j] * zeta_set[j] * pi_archive_set[j * max_iter + k]), q_loc_copy, &p_loc_set[j * vec_loc_size]);    // p[sigma] <- p[sigma] + omega[sigma] / (alpha[sigma] zeta[sigma]) (q / pi_new[sigma] - r_old / pi_old[sigma]) 
            my_daxpy(vec_loc_size, -omega_set[j] / (alpha_set[j] * zeta_set[j] * pi_archive_set[j * max_iter + (k - 1)]), r_old_loc, &p_loc_set[j * vec_loc_size]);
            zeta_set[j] = (1.0 - omega_seed_archive[k] * (sigma[seed] - sigma[j])) * zeta_set[j];      // zeta[sigma] <- (1.0 - omega[seed] (sigma[seed] - sigma[sigma])) * zeta[sigma] 
            beta_set[j] = (pi_archive_set[j * max_iter + (k - 1)] / pi_archive_set[j * max_iter + k]) * (pi_archive_set[j * max_iter + (k - 1)] / pi_archive_set[j * max_iter + k]) * beta_seed_archive[k]; // beta[sigma] <- (pi_old[sigma] / pi_new[sigma])^2 beta[seed] 
            my_dscal(vec_loc_size, beta_set[j], &p_loc_set[j * vec_loc_size]);      // p[sigma] <- 1 / (pi_new[sigma] zeta[sigma]) r + beta[sigma] p[sigma] 
            my_daxpy(vec_loc_size, 1.0 / (pi_archive_set[j * max_iter + k] * zeta_set[j]), r_loc, &p_loc_set[j * vec_loc_size]);
        }

        #pragma omp barrier

#ifdef MEASURE_SECTION_TIME
        local_end_time = MPI_Wtime();
        local_time = local_end_time - local_start_time;

        #pragma omp critical
        {
            if (local_time > max_time) {
                max_time = local_time;
            }
        }

        #pragma omp barrier
        #pragma omp master
        {
            shift_time += max_time;
        }
#endif

        // ==== 収束判定 ====
        //#pragma omp single
        #pragma omp master
        {

#ifdef DISPLAY_SIGMA_RESIDUAL
            if (myid == 0 && k % OUT_ITER == 0) printf("Iter %d : ", k);
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

#ifdef DISPLAY_SIGMA_RESIDUAL
            if (myid == 0 && k % OUT_ITER == 0) printf("\n");
#endif

#ifdef SEED_SWITCHING
            // seed switching 
            if (stop_flag[seed] && stop_count < sigma_len) {

#ifdef MEASURE_SECTION_TIME
                section_start_time = MPI_Wtime();
#endif

                for (int i = 1; i <= k; i++) {
                    alpha_seed_archive[i] = (pi_archive_set[max_sigma * max_iter + (i - 1)] / pi_archive_set[max_sigma * max_iter + i]) * alpha_seed_archive[i];
                    beta_seed_archive[i] = (pi_archive_set[max_sigma * max_iter + (i - 1)] / pi_archive_set[max_sigma * max_iter + i]) * (pi_archive_set[max_sigma * max_iter + (i - 1)] / pi_archive_set[max_sigma * max_iter + i]) * beta_seed_archive[i];
                    omega_seed_archive[i] = omega_seed_archive[i] / (1.0 - omega_seed_archive[i] * (sigma[seed] - sigma[max_sigma]));
                }
                my_dscal(vec_loc_size, 1.0 / (zeta_set[max_sigma] * pi_archive_set[max_sigma * max_iter + k]), r_loc);

                for (j = 0; j < sigma_len; j++) {
                    eta_set[j]    = 0.0;  // eta[sigma]    <- 0 
                    zeta_set[j]   = 1.0;  // zeta[sigma]   <- 1 
                }

                for (int i = 1; i <= k; i++) {
                    for (j = 0; j < sigma_len; j++) {
                        if (stop_flag[j]) continue;
                        if (j == max_sigma) continue;
                        eta_set[j] = (beta_seed_archive[i - 1] / alpha_seed_archive[i - 1]) * alpha_seed_archive[i] * eta_set[j] - (sigma[max_sigma] - sigma[j]) * alpha_seed_archive[i] * pi_archive_set[j * max_iter + (i - 1)];
                        pi_archive_set[j * max_iter + i] = eta_set[j] + pi_archive_set[j * max_iter + (i - 1)];
                        zeta_set[j] = (1.0 - omega_seed_archive[i] * (sigma[max_sigma] - sigma[j])) * zeta_set[j];
                    }
                }

                seed = max_sigma;

#ifdef MEASURE_SECTION_TIME
                section_end_time = MPI_Wtime();
                switch_time += section_end_time - section_start_time;
#endif

#ifdef MEASURE_SECTION_TIME
                section_start_time = MPI_Wtime();
#endif

                //MPI_Allgatherv(&p_loc_set[seed * vec_loc_size], vec_loc_size, MPI_DOUBLE, vec, A_info->recvcounts, A_info->displs, MPI_DOUBLE, MPI_COMM_WORLD);

#ifdef MEASURE_SECTION_TIME
                section_end_time = MPI_Wtime();
                agv_iter_time = section_end_time - section_start_time;
#endif
            }
#endif

#ifdef MEASURE_SECTION_TIME
            agv_time += agv_iter_time;
#endif

#ifdef DISPLAY_SECTION_TIME

            if (myid == 0 && k == 1) {
                printf("iter, unsolved, seed, matvec, agv, shift\n");
            }

            if (myid == 0 && k % OUT_ITER == 0) {
                printf("%d, %d, %e, %e, %e, %e\n", k, sigma_len - stop_count, seed_iter_time, matvec_iter_time, agv_iter_time, max_time);
            }
#endif

            k++;
        }
        #pragma omp barrier
    }
}
    // ==== 反復終了 ====

#if defined(MEASURE_TIME) || defined(MEASURE_SECTION_TIME)
    end_time = MPI_Wtime();
    total_time = end_time - start_time;
#endif

    // ==== 結果表示 ====

#ifdef DISPLAY_ERROR
    if (myid == 0) {
        printf("system #, sigma, relative error\n");
    }

    for (int i = 0; i < sigma_len; i++) {
        MPI_csr_spmv_ovlap(A_loc_diag, A_loc_offd, A_info, &x_loc_set[i * vec_loc_size], vec, r_loc);
        my_daxpy(vec_loc_size, sigma[i], &x_loc_set[i * vec_loc_size], r_loc);

        double diff;
        double local_diff_norm_2 = 0;
        double local_ans_norm_2 = 0;
        for (int j = 0; j < vec_loc_size; j++) {
            diff = r_loc[j] - ans_loc[j];
            local_diff_norm_2 += diff * diff;
            local_ans_norm_2 += ans_loc[j] * ans_loc[j];
        }
        double global_diff_norm_2, global_ans_norm_2;
        MPI_Allreduce(&local_diff_norm_2, &global_diff_norm_2, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&local_ans_norm_2, &global_ans_norm_2, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        double rerative_error = sqrt(global_diff_norm_2) / sqrt(global_ans_norm_2); //ノルムで相対誤差を計算
        if (myid == 0) {
            printf("%d, %e, %e\n", i+1, sigma[i], rerative_error);
        }
    }
    free(ans_loc);
#endif

    if (myid == 0) {
#ifdef MEASURE_TIME
        printf("iterations      : %d\n", k);
        printf("avg time/iter   : %e [sec.] \n", total_time / k);
        printf("total time      : %e [sec.] \n", total_time);
#endif
#ifdef MEASURE_SECTION_TIME
        printf("seed time       : %e [sec.]\n", seed_time);
        printf("agv time        : %e [sec.]\n", agv_time);
        printf("shift time      : %e [sec.]\n", shift_time);
        printf("switch time     : %e [sec.]\n", switch_time);
#endif
    }

    free(r_old_loc); free(r_hat_loc); free(s_loc); free(y_loc); free(q_loc_copy); free(vec);
    free(p_loc_set); free(alpha_set); free(beta_set); free(omega_set); free(eta_set); free(zeta_set);
    free(alpha_seed_archive); free(beta_seed_archive); free(omega_seed_archive); free(pi_archive_set);
    free(stop_flag);

    return k;

}

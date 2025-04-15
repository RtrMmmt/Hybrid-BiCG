#!/bin/bash
#PJM -g "jh240057o"
#PJM -L "rscgrp=debug-o"
#PJM -L "node=4"
#PJM --mpi "proc=16"
#PJM --omp "thread=12"
#PJM -L "elapse=30:00"
#PJM -o "output.out"
#PJM -j

# プロファイラ出力ディレクトリ（必要なら変更可）
PROFILE_DIR="./fipp_out"

# 古い出力が残っているとエラーになるため削除（または事前に空ディレクトリにする）
rm -rf $PROFILE_DIR

# fipp 実行：Elapsed Time + CPUパフォーマンス + MPIコストを測定
fipp -C -d $PROFILE_DIR \
    -Icpupa,mpi \
    -Hmode=all \
    -Sregion \
    -Wspawn \
    mpirun ./solver data/ss.mtx
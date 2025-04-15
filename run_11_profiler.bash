#!/bin/bash
#PJM -g "jh240057o"
#PJM -L "rscgrp=debug-o"
#PJM -L "node=4"
#PJM --mpi "proc=16"
#PJM --omp "thread=12"
#PJM -L "elapse=30:00"
#PJM -o "output.out"
#PJM -j

# 実行回数
NUM_RUNS=11

# 出力ベースディレクトリ
BASE_DIR="./pa"

# 実行ファイルと入力
EXEC="mpirun ./solver data/ss.mtx"
EVENT_BASE="pa"  # pa1, pa2, ...
FAPP_OPTIONS="-Icpupa,mpi -Wspawn"
FAPPPX_OPTIONS="-Icpupa,mpi -tcsv"

# ベースディレクトリがなければ作成
mkdir -p "$BASE_DIR"

for i in $(seq 1 $NUM_RUNS); do
    DIR="${BASE_DIR}/rep$i"
    EVENT="${EVENT_BASE}${i}"
    CSV_OUT="${BASE_DIR}/pa${i}.csv"

    # 古い出力が残っていれば削除
    rm -rf "$DIR"

    fapp -C -d "$DIR" \
         $FAPP_OPTIONS \
         -Hevent=${EVENT},mode=all \
         $EXEC

    fapppx -A -d "$DIR" $FAPPPX_OPTIONS -o "$CSV_OUT"
done
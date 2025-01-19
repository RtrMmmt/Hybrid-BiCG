#!/bin/bash
#PJM -g "b30304"
#PJM -L "rscgrp=ea"
#PJM -L "node=4"
#PJM --mpi "proc=8"
#PJM --omp "thread=20"
#PJM -L "elapse=30:00"
#PJM -o "output.out"
#PJM -j
mpirun ./solver data/ss.mtx
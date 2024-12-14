#!/bin/bash
#PJM -g "jh240057o"
#PJM -L "rscgrp=debug-o"
#PJM -L "node=2"
#PJM --mpi "proc=8"
#PJM --omp "thread=12"
#PJM -L "elapse=5:00"
#PJM -o "output.out"
#PJM -j
mpirun ./solver data/Transport.mtx
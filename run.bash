#!/bin/bash
#PJM -g "b30304"
#PJM -L "rscgrp=ea"
#PJM -L "node=4"
#PJM --mpi "proc=8"
#PJM -L "elapse=30:00"
#PJM -o "output.out"
#PJM -j

export OMP_NUM_THREADS=20
export I_MPI_PIN_DOMAIN=40
mpiexec.hydra -n ${PJM_MPI_PROC} ./solver data/5_ss.mtx
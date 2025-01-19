#!/bin/bash
#PJM -L "rscunit=fx"
#PJM -L "rscgrp=fx-debug"
#PJM -L "node=16"
#PJM --mpi "proc=64"
#PJM -L "elapse=60:00"
mpirun ./solver data/transport.mtx bicgstab
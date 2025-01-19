#!/bin/bash

#============ SBATCH Directives =======
#SBATCH -p jha
#SBATCH -t 0:30:0
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=56
#SBATCH --output=output.out
#SBATCH --error=output.out

#============ Shell Script ============
export OMP_NUM_THREADS=56
srun ./solver data/5_ss.mtx
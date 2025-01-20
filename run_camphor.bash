#!/bin/bash
#============ SBATCH Directives =======
#SBATCH -p jha
#SBATCH -t 0:30:0
#SBATCH --output=output.out
#SBATCH --error=output.out
#SBATCH --rsc p=4:t=28:c=28

#============ Shell Script ============

# MPIプログラムの実行
srun ./solver data/5_ss.mtx
#!/bin/bash
#============ SBATCH Directives =======
#SBATCH -p jha
#SBATCH -t 0:30:0
#SBATCH --output=output.out
#SBATCH --error=output.out
#SBATCH --rsc p=8:t=56:c=56

#============ Shell Script ============

# MPIプログラムの実行
srun ./solver data/5_ss.mtx
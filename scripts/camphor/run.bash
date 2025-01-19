#!/bin/bash
#============ SBATCH Directives =======
#SBATCH -p=jha
#SBATCH -t=00:30:00
#SBATCH --output=output.out
#SBATCH --error=output.out
#SBATCH --rsc p=4:t=8:c=8

#============ Shell Script ============

# MPIプログラムの実行
srun ./solver data/5_ss.mtx
#!/bin/bash
#============ SBATCH Directives =======
#SBATCH --account=b38307             # アカウント名 (ジョブグループ)
#SBATCH --partition=jha              # パーティション名 (リソースグループ)
#SBATCH --nodes=4                    # 使用するノード数
#SBATCH --ntasks=8                   # MPIプロセス数
#SBATCH --time=00:30:00              # 実行時間
#SBATCH --output=output.out          # 出力ファイル
#SBATCH --error=output.out           # エラーログを標準出力に統合
#SBATCH --rsc p=8:t=56:c=1      　　　# プロセスごとのリソース指定 (必要に応じて変更)

#============ Shell Script ============

# MPIプログラムの実行
srun ./solver data/5_ss.mtx
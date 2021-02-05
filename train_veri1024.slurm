#!/bin/bash

#SBATCH --job-name=GPU_HPL
#SBATCH --partition=dgx2
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mail-type=end
#SBATCH --mail-user=chenyongbiao0319@sjtu.edu.cn
#SBATCH --output=%j.out
#SBATCH --error=%j.err
module load miniconda3
source activate tenv
./train_veri1024.sh

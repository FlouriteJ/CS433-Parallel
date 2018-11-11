#!/bin/bash
#SBATCH --job-name=prac1a
#SBATCH --partition=k40
#SBATCH --gres=gpu:1
#SBATCH --mail-type=end
#SBATCH --mail-user=YOURMAIL@sjtu.edu.cn
#SBATCH --output=%j.out
#SBATCH --error=%j.err
module purge
module load gcc cuda
./simpleCUBLAS
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16GB
#SBATCH --time=1:00:00
#SBATCH --partition=gpu 
#SBATCH --output=gpujob.out
#SBATCH --gres=gpu:v100:1

module purge
module load nvidia-hpc-sdk
module load gcc/11.3.0
module load cuda/11.5.1

nvcc -o sparseGameOfLife sparseGameOfLife.cu
nvcc -o gameOfLife gameOfLife.cu

./sparseGameOfLife inputs/sparse.txt 1
./sparseGameOfLife inputs/sparse.txt 10
./sparseGameOfLife inputs/sparse.txt 100
./sparseGameOfLife inputs/sparse.txt 1000
./sparseGameOfLife inputs/sparse.txt 10000

./gameOfLife inputs/sparse.txt 1
./gameOfLife inputs/sparse.txt 10
./gameOfLife inputs/sparse.txt 100
./gameOfLife inputs/sparse.txt 1000
./gameOfLife inputs/sparse.txt 10000

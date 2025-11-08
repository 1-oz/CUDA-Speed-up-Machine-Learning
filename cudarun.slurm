#!/bin/bash

#SBATCH --job-name=MachineLearning
#SBATCH --partition=gpu
#SBATCH --output=CUmachinelearning_%j.log         
#SBATCH --error=error_%j.log
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --constraint=v100


nvcc -O3 -lcublas cuda.cu -o cuda -lz

srun ./cuda 1

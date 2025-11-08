#!/bin/bash

#SBATCH --job-name=MachineLearning
#SBATCH --partition=caslake
#SBATCH --output=machinelearning_%j.log         
#SBATCH --error=error_%j.log
#SBATCH --time=00:30:00
#SBATCH --nodes=1             
#SBATCH --ntasks-per-node=1                      
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=2000                        

export OMP_NUM_THREADS=32

icx -qopenmp -lopenblas -xHost -Ofast -o ml_cpu baseline.c -lm -lz
srun ./ml_cpu 0

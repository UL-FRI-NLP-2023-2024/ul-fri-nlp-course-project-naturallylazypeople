#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --job-name=onj
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --constraint=h100
#SBATCH --mem=64GB
#SBATCH --output=sample_out_h100_512_30.log
#SBATCH --time=02:00:00

module load CUDA/12.1.1
srun apptainer build --fakeroot image.sif image.def
srun apptainer exec --nv image.sif python main.py
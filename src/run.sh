#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --job-name=onj
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --output=sample_out.log

srun apptainer build --fakeroot image.sif image.def
srun apptainer exec --nv image.sif python main.py
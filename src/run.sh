#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --job-name=onj
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64GB
#SBATCH --output=sample_out_h100_512_30.log
#SBATCH --time=08:00:00

module load CUDA/12.1.1
srun apptainer build --fakeroot image.sif config/image.def
srun apptainer exec --nv image.sif python main.py
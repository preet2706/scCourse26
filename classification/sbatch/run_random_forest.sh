#!/bin/bash
#SBATCH --job-name=condclf-rf
#SBATCH --partition=htc-el8
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/shahp/scCourse26/classification/results/random_forest/slurm-%j.out

source ~/.bashrc
conda activate denbi
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd /scratch/shahp/scCourse26/classification
python train_random_forest.py --results-dir results/random_forest --n-jobs "$SLURM_CPUS_PER_TASK"

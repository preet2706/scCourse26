#!/bin/bash
#SBATCH --job-name=condclf-xgboost
#SBATCH --partition=htc-el8
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/shahp/scCourse26/classification/results/xgboost/slurm-%j.out

source ~/.bashrc
conda activate denbi
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd /scratch/shahp/scCourse26/classification
python train_xgboost.py --results-dir results/xgboost --n-jobs "$SLURM_CPUS_PER_TASK"

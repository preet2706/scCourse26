#!/bin/bash
#SBATCH --job-name=condclf-logreg
#SBATCH --partition=htc-el8
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/shahp/scCourse26/classification/results/logreg/slurm-%j.out
#SBATCH --error=/scratch/shahp/scCourse26/classification/results/logreg/slurm-%j.err

source ~/.bashrc
conda activate denbi
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd /scratch/shahp/scCourse26/classification
python train_logreg.py --results-dir results/logreg

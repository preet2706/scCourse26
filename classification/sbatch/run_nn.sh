#!/bin/bash
#SBATCH --job-name=condclf-nn
#SBATCH --partition=htc-el8
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/shahp/scCourse26/classification/results/nn/slurm-%j.out

source ~/.bashrc
conda activate denbi

# torch's numpy dependency needs the conda env's own libstdc++ ahead of the system one
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd /scratch/shahp/scCourse26/classification
python train_nn.py --results-dir results/nn --n-jobs "$SLURM_CPUS_PER_TASK"

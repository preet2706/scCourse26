#!/bin/bash
#SBATCH --job-name=pertpred-all
#SBATCH --partition=htc-el8
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/shahp/scCourse26/perturbation_prediction/results/slurm-%j.out

# The whole task is light enough to run interactively (all models train in seconds to ~1 min;
# the bootstrap CIs are the slowest step). This script just bundles the full pipeline.

source ~/.bashrc
conda activate denbi
set -e                                                       # after the profile sourcing, which can return nonzero
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"   # torch import fix (see project CLAUDE.md)
NCPU="${SLURM_CPUS_PER_TASK:-4}"                              # falls back when run as plain `bash`
export OMP_NUM_THREADS="$NCPU"

cd /scratch/shahp/scCourse26/perturbation_prediction

python data_prep.py
python train_baseline.py
python train_knn.py --features raw
python train_knn.py --features scalars
python train_mlp.py --features raw    --n-jobs "$NCPU"
python train_mlp.py --features scalars --n-jobs "$NCPU"
python compare_models.py
python build_notebook.py

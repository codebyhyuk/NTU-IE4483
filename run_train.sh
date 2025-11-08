#!/bin/bash -l
#SBATCH --job-name=resnet_train
#SBATCH --output=output.log
#SBATCH --error=output.log
#SBATCH --time=04:00:00
#SBATCH --gpus=1
#SBATCH --constraint=6000ada
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --chdir=/home/ee4483_02/IE4483_Project

# (If your cluster uses environment modules)
# module purge
# module load Miniforge3   # or Anaconda/Miniconda, cluster-specific

# Make 'conda activate' work in non-interactive shells
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ee4483      # <-- your env name

python -V
srun python train.py --config /home/ee4483_02/IE4483_Project/config/config.yaml

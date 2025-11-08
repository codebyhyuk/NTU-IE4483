#!/bin/bash -l
#SBATCH --job-name=resnet_train
#SBATCH --output=output.log
#SBATCH --error=error.log
#SBATCH --time=12:00:00
#SBATCH --gpus=1
#SBATCH --constraint=6000ada
#SBATCH --chdir=/home/ee4483_02/NTU-IE4483

# (If your cluster uses environment modules)
# module purge
# module load Miniforge3   # or Anaconda/Miniconda, cluster-specific

# Make 'conda activate' work in non-interactive shells
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ee4483      # <-- your env name

python -V
srun python train_pretrained_model.py
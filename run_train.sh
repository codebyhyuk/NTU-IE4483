#!/bin/bash -l
#SBATCH --job-name=resnet_train
#SBATCH --output=output.log
#SBATCH --error=error.log
#SBATCH --time=12:00:00
#SBATCH --gpus=1
#SBATCH --constraint=a5000
#SBATCH --chdir=/home/ee4483_02/NTU-IE4483

# (If your cluster uses environment modules)
# module purge
# module load Miniforge3   # or Anaconda/Miniconda, cluster-specific

# Make 'conda activate' work in non-interactive shells
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ee4483      # <-- your env name

python -V
srun python train.py \
	--dataset cifar10 \
	--data_dir /projects/448302 \
	--pretrained False \
	--lr 0.0001 \
	--epochs 50 \
	--batch_size 32 \
	--save_dir ./runs \
	--wandb enabled

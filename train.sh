#!/bin/bash

#SBATCH --job-name=planning_arit_bal
#SBATCH --account=lingo
#SBATCH --partition=lingo-h100
#SBATCH --qos=lingo-main
#SBATCH --time=24:00:00 # (hh:mm:ss)
#SBATCH --output=/data/scratch/lruis/planning_arit_bal.log  # CHANGE THIS
#SBATCH --error=/data/scratch/lruis/planning_arit_bal.err  # CHANGE THIS
#SBATCH --gpus=1
#SBATCH --mem=80GB

source $HOME/.local/bin/env
cd /data/scratch/lruis/planning_arithmetic
source planning_train_env/bin/activate

export PATH=/usr/local/cuda-12.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH

nvidia-smi
python train.py +experiment=scratch
#!/bin/bash
#SBATCH --job-name=tox21_edge
#SBATCH --output=logs/tox21_edge.out
#SBATCH --error=logs/tox21_edge.err
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=5GB
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --time=3:00:00
#SBATCH --account=cse576f25s001_class
#execute code

# (Optional) activate conda env
source ~/.bashrc
conda activate et_tox

# Restrict to GPU 0 and run training
CUDA_VISIBLE_DEVICES=0 python train.py \
    --conf train_yaml/tox21_edge.yaml

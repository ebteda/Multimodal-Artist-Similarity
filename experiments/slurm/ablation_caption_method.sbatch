#!/bin/bash
#SBATCH --job-name=thesis_caption
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/home/arahmatpour/thesis/logs/caption_%j.out
#SBATCH --error=/home/arahmatpour/thesis/logs/caption_%j.err

mkdir -p /home/arahmatpour/thesis/logs

module load miniforge/24.3.0-0
source activate $HOME/thesis/venv

echo "Job started at: $(date)"
echo "Running on: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo

python -u $HOME/thesis/code/ablation_caption_method.py

echo
echo "Job finished at: $(date)"

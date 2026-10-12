#!/bin/bash

#SBATCH --job-name=sft
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --mem=480G
#SBATCH --partition=standard-g
#SBATCH --time=0-12:00:00
#SBATCH --gpus-per-node=8
#SBATCH --account=<PROJECT_ID>

# LUMI example; load any environment that provides torch and transformers
module use /appl/local/csc/modulefiles/
module load pytorch/2.5

export HF_HOME="./hf_cache"

# distributed setup
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$((10000 + $SLURM_JOB_ID % 1000))
export WORLD_SIZE=$SLURM_NTASKS

# 32 GPUs with --tp 4 give 8 data-parallel groups and 4 gradient-accumulation steps for a batch of 32, e.g.
#   sbatch train.sh --data_path ../data/lima_english.jsonl --output_dir checkpoints/english_sft --max_epochs 1
srun --label python3 train.py "$@"

#!/bin/bash

#SBATCH --job-name=generate
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --partition=standard-g
#SBATCH --time=0-6:00:00
#SBATCH --gpus-per-node=1
#SBATCH --account=<PROJECT_ID>

# LUMI example; load any environment that provides torch and transformers
module use /appl/local/csc/modulefiles/
module load pytorch/2.5

export HF_HOME="./hf_cache"

# e.g. sbatch generate.sh --model_name ../rl/checkpoints/on-policy-RL/checkpoint_199 --dataset fluency_prompts --n_samples 16
srun python3 generate.py "$@"

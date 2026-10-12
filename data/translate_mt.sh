#!/bin/bash

#SBATCH --job-name=translate-mt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --mem=0
#SBATCH --partition=standard-g
#SBATCH --time=0-12:00:00
#SBATCH --gpus-per-node=mi250:8
#SBATCH --account=<PROJECT_ID>

# LUMI example; load any environment that provides torch, transformers, sentencepiece and nltk
module use /appl/local/csc/modulefiles/
module load pytorch/2.5

export WORLD_SIZE=$SLURM_NTASKS
export HF_HOME="./hf_cache"

set -euo pipefail

# one shard per task, e.g. sbatch translate_mt.sh --system nllb
srun python3 translate_mt.py "$@"

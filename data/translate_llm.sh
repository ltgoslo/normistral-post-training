#!/bin/bash

#SBATCH --job-name=translate-llm
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --partition=standard-g
#SBATCH --time=0-12:00:00
#SBATCH --gpus-per-node=mi250:8
#SBATCH --account=<PROJECT_ID>

# LUMI example; load any environment that provides torch, transformers and vllm
module use /appl/local/csc/modulefiles/
module load pytorch/2.5

export HF_HOME="./hf_cache"

set -euo pipefail

# e.g. sbatch translate_llm.sh --input no_robots_english.jsonl --output no_robots_nob_prompts.jsonl --roles user
python3 translate_llm.py "$@"

#!/bin/bash

#SBATCH --job-name=judge
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=56
#SBATCH --mem=480G
#SBATCH --partition=standard-g
#SBATCH --time=0-6:00:00
#SBATCH --gpus-per-node=8
#SBATCH --account=<PROJECT_ID>

# LUMI example; the environment needs vllm (the 70B judge runs tensor-parallel over 8 GPUs)
export EBU_USER_PREFIX=<EASYBUILD_PREFIX>
module purge
module load LUMI
module load PyTorch/2.6.0-rocm-6.2.4-python-3.12-singularity-20250404

export HF_HOME="./hf_cache"
export TRITON_CACHE_DIR="./triton_cache"

# e.g. sbatch judge.sh judge_pairwise.py --dataset norrewrite --baseline outputs/norrewrite_Llama-3.1-8B-Instruct.jsonl --predictions outputs/norrewrite_*.jsonl
#      sbatch judge.sh judge_norquad.py --inputs outputs/norquad_*.jsonl
python3 "$@"

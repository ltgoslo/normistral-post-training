#!/bin/bash

#SBATCH --job-name=on-policy-rl
#SBATCH --nodes=48
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --mem=480G
#SBATCH --partition=standard-g
#SBATCH --time=0-48:00:00
#SBATCH --gpus-per-node=8
#SBATCH --account=<PROJECT_ID>

# LUMI example; the environment needs torch, transformers, flash-attention and vllm
export EBU_USER_PREFIX=<EASYBUILD_PREFIX>
module purge
module load LUMI
module load PyTorch/2.6.0-rocm-6.2.4-python-3.12-singularity-20250404

export TRITON_CACHE_DIR="./triton_cache"
export HF_HOME="./hf_cache"

# distributed setup
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=9999
export WORLD_SIZE=$SLURM_NTASKS

export NCCL_MAX_P2P_NCHANNELS=2
export RCCL_MSCCL_ENABLE=0
export NCCL_PROTO=Simple

# default layout: 16 pipelines x (8 trainer + 8 generator + 8 judge GPUs) = 384 GPUs = 48 nodes, e.g.
#   sbatch run.sh --model_name ../sft/checkpoints/english_sft --judge_model_name mistralai/Mistral-Nemo-Instruct-2407
srun --label python3 rank_router.py "$@"

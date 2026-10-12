"""Entry point of the distributed on-policy RL training: one process per GPU, launched with `srun`.

The processes are assigned to roles in rank order: trainers, generators, then judges. Each of the
`--n_pipelines` pipelines consists of a trainer (tensor-parallel over --n_gpus_per_trainer GPUs),
a generator (one policy copy per GPU) and a judge (vLLM, tensor-parallel). Trainers and generators
communicate with torch.distributed; the judges live outside of the process group and exchange JSON
files with the trainers (see worker.py). With the default settings, a step samples
16 pipelines x 8 prompts x 8 responses and the policy is updated once per step with a batch of
128 prompts (Appendix C of the paper).
"""
import os
import argparse
import datetime
from socket import gethostname

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from huggingface_hub import login


def parse_config():
    parser = argparse.ArgumentParser()
    # models and data
    parser.add_argument("--model_name", type=str, default="../sft/checkpoints/english_sft", help="Initial policy (the English SFT checkpoint).")
    parser.add_argument("--ref_model_name", type=str, default=None, help="Reference policy of the KL term; defaults to --model_name.")
    parser.add_argument("--judge_model_name", type=str, default="mistralai/Mistral-Nemo-Instruct-2407")
    parser.add_argument("--judge_template", type=str, default="templates/judge_prompt.txt")
    parser.add_argument("--prompt_file", type=str, default="../data/no_robots_nob_prompts.jsonl", help="Conversations with translated prompts; their last (English) message is the gold response shown to the judge.")
    parser.add_argument("--output_dir", type=str, default="checkpoints/on-policy-RL")
    parser.add_argument("--hf_token", type=str, default=os.environ.get("HF_TOKEN"))
    parser.add_argument("--wandb_project", type=str, default="normistral-post-training")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_name", type=str, default="on-policy-RL")
    # parallelization
    parser.add_argument("--n_pipelines", type=int, default=16)
    parser.add_argument("--n_pipelines_per_trainer", type=int, default=1, help="Number of generator/judge pipelines served by one trainer.")
    parser.add_argument("--n_gpus_per_trainer", type=int, default=8)
    parser.add_argument("--n_gpus_per_generator", type=int, default=8)
    parser.add_argument("--n_gpus_per_judge", type=int, default=8)
    parser.add_argument("--n_judges", type=int, default=1, help="Number of judges per pipeline; their scores are averaged.")
    parser.add_argument("--tp", type=int, default=8, help="Tensor-parallel degree of the trainer; must equal --n_gpus_per_trainer.")
    # sampling
    parser.add_argument("--n_prompts_per_message", type=int, default=8, help="Prompts sampled by every generator per step.")
    parser.add_argument("--n_responses_per_prompt", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=51200)
    # optimization
    parser.add_argument("--global_batch_size", type=int, default=128, help="Prompts per optimizer step.")
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.99)
    parser.add_argument("--warmup_steps", type=int, default=10)
    parser.add_argument("--kl_coefficient", type=float, default=0.01)
    parser.add_argument("--entropy_coefficient", type=float, default=0.0, help="Weight of an optional soft entropy target; not used in the paper (0 disables it).")
    parser.add_argument("--target_entropy", type=float, default=1.0)
    parser.add_argument("--default_reward", type=float, default=3.0, help="Reward used when no score can be parsed from a judgment.")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--save_every", type=int, default=25)
    args = parser.parse_args()

    if args.ref_model_name is None:
        args.ref_model_name = args.model_name
    assert args.tp == args.n_gpus_per_trainer, "The trainer is tensor-parallel over all its GPUs"
    assert args.n_gpus_per_generator % args.n_gpus_per_trainer == 0

    return args


if __name__ == "__main__":
    config = parse_config()

    assert torch.cuda.is_available()

    config.world_size = int(os.environ["WORLD_SIZE"])
    config.global_rank = int(os.environ["SLURM_PROCID"])
    config.slurm_job_id = int(os.environ["SLURM_JOB_ID"])

    gpus_per_node = int(os.environ["SLURM_GPUS_ON_NODE"])
    assert gpus_per_node == torch.cuda.device_count()
    print(f"Hello from rank {config.global_rank} of {config.world_size} on {gethostname()} where there are {gpus_per_node} GPUs", flush=True)

    if config.global_rank == 0:
        os.makedirs(config.output_dir, exist_ok=True)

        import wandb
        wandb.init(name=config.wandb_name, config=config, project=config.wandb_project, entity=config.wandb_entity)

        if config.hf_token:
            login(token=config.hf_token)

    # --- assign a role to this rank ---
    n_trainer_ranks = config.n_gpus_per_trainer // config.n_pipelines_per_trainer * config.n_pipelines
    n_generator_ranks = config.n_gpus_per_generator * config.n_pipelines
    n_judge_ranks = config.n_gpus_per_judge * config.n_judges * config.n_pipelines
    assert config.world_size == n_trainer_ranks + n_generator_ranks + n_judge_ranks, \
        f"Expected {n_trainer_ranks + n_generator_ranks + n_judge_ranks} processes, got {config.world_size}"

    if config.global_rank < n_trainer_ranks:
        config.role = "trainer"
        config.pipeline_index = (config.global_rank // config.n_gpus_per_trainer) * config.n_pipelines_per_trainer
        config.worker_rank = config.global_rank % config.n_gpus_per_trainer
    elif config.global_rank < n_trainer_ranks + n_generator_ranks:
        config.role = "generator"
        config.pipeline_index = (config.global_rank - n_trainer_ranks) // config.n_gpus_per_generator
        config.worker_rank = (config.global_rank - n_trainer_ranks) % config.n_gpus_per_generator
    else:
        judge_rank = config.global_rank - n_trainer_ranks - n_generator_ranks
        config.role = "judge"
        config.judge_index = judge_rank // (config.n_gpus_per_judge * config.n_pipelines)
        config.pipeline_index = (judge_rank // config.n_gpus_per_judge) % config.n_pipelines
        config.worker_rank = judge_rank % config.n_gpus_per_judge

        # the judges do not participate in the process group, vLLM manages their GPUs itself
        config.node_rank = config.global_rank % gpus_per_node
        torch.cuda.set_device(config.node_rank)
        config.device = torch.device("cuda", config.node_rank)

        if config.worker_rank == 0:
            from worker_reward_judge import WorkerRewardJudge
            worker = WorkerRewardJudge(config)
        else:
            config.role = "idle"
            from worker_idle import WorkerIdle
            worker = WorkerIdle(config, use_barrier=False)

        worker.run()
        exit(0)

    dist.init_process_group(
        backend="nccl",
        rank=config.global_rank,
        world_size=config.world_size - n_judge_ranks,
        timeout=datetime.timedelta(minutes=60)
    )
    if config.global_rank == 0:
        print(f"Global process group initialized? {dist.is_initialized()}", flush=True)

    config.node_rank = config.global_rank % gpus_per_node
    torch.cuda.set_device(config.node_rank)
    config.device = torch.device("cuda", config.node_rank)

    print(f"Process {config.global_rank} is {config.role} {config.pipeline_index} / {config.n_pipelines}", flush=True)

    # --- global ranks of the trainer and generator GPUs of every pipeline ---
    config.worker_rank_ranges = {}
    for pipeline_index in range(config.n_pipelines):
        train_start = (pipeline_index // config.n_pipelines_per_trainer) * config.n_gpus_per_trainer
        gen_start = n_trainer_ranks + pipeline_index * config.n_gpus_per_generator
        config.worker_rank_ranges[pipeline_index] = {
            "trainer": list(range(train_start, train_start + config.n_gpus_per_trainer)),
            "generator": list(range(gen_start, gen_start + config.n_gpus_per_generator))
        }

    # --- process groups (every rank has to take part in the creation of every group) ---
    all_trainer_ranks = sorted({r for p in range(config.n_pipelines) for r in config.worker_rank_ranges[p]["trainer"]})
    all_generator_ranks = sorted({r for p in range(config.n_pipelines) for r in config.worker_rank_ranges[p]["generator"]})
    config.worker_type_groups = {
        "trainer": dist.new_group(ranks=all_trainer_ranks, backend="nccl", timeout=datetime.timedelta(minutes=60)),
        "generator": dist.new_group(ranks=all_generator_ranks, backend="nccl", timeout=datetime.timedelta(minutes=60)),
    }

    trainer_groups, generator_groups = [], []
    for pipeline_index in range(config.n_pipelines):
        trainer_groups.append(dist.new_group(ranks=config.worker_rank_ranges[pipeline_index]["trainer"], backend="nccl", timeout=datetime.timedelta(minutes=60)))
        generator_groups.append(dist.new_group(ranks=config.worker_rank_ranges[pipeline_index]["generator"], backend="nccl", timeout=datetime.timedelta(minutes=60)))
    config.trainer_pipeline_group = trainer_groups[config.pipeline_index]
    config.generator_pipeline_group = generator_groups[config.pipeline_index]

    config.my_pipeline_ranks = config.worker_rank_ranges[config.pipeline_index]

    # --- device mesh of the trainers (types x data-parallel x tensor-parallel) ---
    # the first "type" row holds the trainer GPUs, the remaining rows the generator GPUs
    n_types = 1 + (config.n_gpus_per_generator // config.n_gpus_per_trainer) * config.n_pipelines_per_trainer
    config.full_mesh = init_device_mesh(
        "cuda",
        mesh_shape=(n_types, config.n_pipelines // config.n_pipelines_per_trainer, config.tp),
        mesh_dim_names=("types", "dp", "tp")
    )
    config.train_submesh = config.full_mesh["dp", "tp"]

    if config.role == "trainer":
        from worker_trainer import WorkerTrainer
        worker = WorkerTrainer(config)
    elif config.role == "generator":
        from worker_generator import WorkerGenerator
        worker = WorkerGenerator(config)
    else:
        raise ValueError(f"Unknown role: {config.role}")

    worker.run()

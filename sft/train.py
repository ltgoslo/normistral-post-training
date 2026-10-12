"""Supervised finetuning with tensor parallelism and data parallelism (DDP with a ZeRO optimizer).

Used for the English SFT stage (LIMA, 1 epoch) and for the translated-SFT baseline (machine-translated
No Robots, 3 epochs). One process per GPU, launched with `srun` (see train.sh). The loss is computed
only on the assistant turns of each conversation.
"""
import os
import argparse
import datetime
import functools
from socket import gethostname
from contextlib import nullcontext

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.distributed._tensor import Replicate, DTensor
from torch.distributed.tensor.parallel import parallelize_module, ColwiseParallel, RowwiseParallel
from torch.distributed.tensor.parallel.ddp import _pre_dp_module_transform
from torch.distributed.device_mesh import init_device_mesh
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import warmup_then_constant_schedule, is_main_process, seed_everything
from dataset import Dataset
from stableadamw import StableAdamW

import warnings
warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def rgetattr(obj, attr):
    return functools.reduce(getattr, attr.split("."), obj)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="../data/lima_english.jsonl", type=str, help="JSONL file with conversations in the No Robots (`messages`) or LIMA (`conversations`) format.")
    parser.add_argument("--output_dir", default="checkpoints/english_sft", type=str, help="Directory where the final model is saved.")
    parser.add_argument("--model_path_or_name", default="ltg/normistral-11b-warm", type=str)
    parser.add_argument("--tokenizer_path_or_name", default=None, type=str, help="Defaults to --model_path_or_name.")
    parser.add_argument("--chat_template", default=os.path.join(SCRIPT_DIR, "chat_template.jinja"), type=str, help="Jinja chat template assigned to the tokenizer (Appendix E.1 of the paper); pass an empty string to keep the tokenizer's own template.")
    parser.add_argument("--seq_length", default=4096, type=int, help="Conversations longer than this many tokens are truncated.")
    parser.add_argument("--global_batch_size", default=32, type=int, help="Number of conversations per optimizer step.")
    parser.add_argument("--tp", default=4, type=int, help="Tensor-parallel degree.")
    parser.add_argument("--learning_rate", default=2e-6, type=float)
    parser.add_argument("--beta1", default=0.9, type=float)
    parser.add_argument("--beta2", default=0.99, type=float)
    parser.add_argument("--weight_decay", default=0.1, type=float)
    parser.add_argument("--max_epochs", default=1, type=int)
    parser.add_argument("--warmup_proportion", default=0.1, type=float, help="Proportion of training steps with linear learning-rate warm-up; the learning rate is constant afterwards.")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--wandb_project", default="normistral-post-training", type=str)
    parser.add_argument("--wandb_entity", default=None, type=str)
    parser.add_argument("--wandb_name", default=None, type=str, help="Defaults to the name of the output directory.")
    parser.add_argument("--no_wandb", action="store_true")
    args = parser.parse_args()

    if args.tokenizer_path_or_name is None:
        args.tokenizer_path_or_name = args.model_path_or_name
    if args.wandb_name is None:
        args.wandb_name = os.path.basename(os.path.normpath(args.output_dir))
    return args


def setup_training(args):
    assert torch.cuda.is_available()

    args.world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["SLURM_PROCID"])
    gpus_per_node = int(os.environ["SLURM_GPUS_ON_NODE"])
    assert gpus_per_node == torch.cuda.device_count()
    print(f"Hello from rank {rank} of {args.world_size} on {gethostname()} where there are {gpus_per_node} GPUs", flush=True)

    assert args.world_size % args.tp == 0, f"The number of GPUs ({args.world_size}) must be divisible by the tensor-parallel degree ({args.tp})"
    args.dp = args.world_size // args.tp
    assert args.global_batch_size % args.dp == 0, f"The global batch size ({args.global_batch_size}) must be divisible by the number of data-parallel groups ({args.dp})"
    args.gradient_accumulation = args.global_batch_size // args.dp

    seed_everything(args.seed)

    torch.distributed.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=args.world_size,
        timeout=datetime.timedelta(minutes=60)
    )

    local_rank = rank % gpus_per_node
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    args.use_wandb = is_main_process() and not args.no_wandb
    if is_main_process():
        os.makedirs(args.output_dir, exist_ok=True)
        print("All arguments:")
        for k, v in vars(args).items():
            print(f"{k}: {v}")
        print(flush=True)
    if args.use_wandb:
        import wandb
        wandb.init(name=args.wandb_name, config=args, project=args.wandb_project, entity=args.wandb_entity)

    return device


def load_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path_or_name)
    if args.chat_template:
        tokenizer.chat_template = open(args.chat_template).read()
    assert tokenizer.chat_template, "The tokenizer needs a chat template with {% generation %} markers around the assistant turns"
    return tokenizer


def load_dataloader(args, tokenizer):
    train_data = Dataset(args.data_path, tokenizer, args.rank, args.dp, seq_length=args.seq_length)
    print(f"Training on {len(train_data)} conversations per data-parallel rank", flush=True)

    # every item of a batch is one conversation; the batch is consumed with gradient accumulation
    train_dataloader = DataLoader(
        train_data,
        shuffle=True,
        batch_size=args.gradient_accumulation,
        num_workers=0,
        generator=torch.Generator().manual_seed(args.seed + args.rank),
        collate_fn=lambda x: x,
        drop_last=True,
    )
    print(f"Training on {len(train_dataloader)} batches per epoch", flush=True)

    return train_dataloader


def prepare_model_and_optimizer(args, device, tokenizer):
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path_or_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    if is_main_process():
        n_params = sum(p.numel() for p in model.parameters())
        print(model)
        print(f"NUMBER OF PARAMETERS: {n_params}\n", flush=True)
        if args.use_wandb:
            import wandb
            wandb.config.update({"n_params": n_params})
    model.to(device)

    mesh = init_device_mesh("cuda", mesh_shape=(args.dp, args.tp), mesh_dim_names=("dp", "tp"))
    args.rank = mesh.get_local_rank("dp")

    train_dataloader = load_dataloader(args, tokenizer)

    n_layers = model.config.num_hidden_layers
    model = parallelize_module(
        module=model,
        device_mesh=mesh["tp"],
        parallelize_plan={
            "model.embed_tokens": ColwiseParallel(output_layouts=Replicate()),
            **{f"model.layers.{i}.self_attn.q_proj": ColwiseParallel() for i in range(n_layers)},
            **{f"model.layers.{i}.self_attn.k_proj": ColwiseParallel() for i in range(n_layers)},
            **{f"model.layers.{i}.self_attn.v_proj": ColwiseParallel() for i in range(n_layers)},
            **{f"model.layers.{i}.self_attn.o_proj": RowwiseParallel() for i in range(n_layers)},
            **{f"model.layers.{i}.mlp.up_proj": ColwiseParallel() for i in range(n_layers)},
            **{f"model.layers.{i}.mlp.gate_proj": ColwiseParallel() for i in range(n_layers)},
            **{f"model.layers.{i}.mlp.down_proj": RowwiseParallel() for i in range(n_layers)},
        },
    )

    _pre_dp_module_transform(model)
    model = DDP(model, device_mesh=mesh["dp"])
    args.parameter_names = [n for n, _ in model.module.named_parameters()]
    torch.distributed.barrier()

    optimizer = ZeroRedundancyOptimizer(
        params=model.parameters(),
        optimizer_class=StableAdamW,
        process_group=mesh.get_group("dp"),
        lr=args.learning_rate,
        betas=(args.beta1, args.beta2),
        eps=1e-8,
        weight_decay=args.weight_decay,
        kahan_sum=True
    )

    total_steps = len(train_dataloader) * args.max_epochs
    scheduler = warmup_then_constant_schedule(optimizer, int(total_steps * args.warmup_proportion))

    return model, optimizer, scheduler, train_dataloader


def training_epoch(model, train_dataloader, optimizer, scheduler, args, device, epoch):
    model.train()
    model.zero_grad()

    for batch in tqdm(train_dataloader, desc=f"Epoch {epoch}", disable=not is_main_process()):
        # normalize the loss by the global number of target tokens in this optimizer step
        with torch.no_grad():
            local_n_labels = sum((output_ids != -100).sum() for _, output_ids in batch).float().to(device)
            global_n_labels = local_n_labels.clone()
            torch.distributed.all_reduce(global_n_labels, torch.distributed.ReduceOp.SUM)
            global_n_labels = global_n_labels / args.tp  # all tensor-parallel ranks of a group see the same batch
            coefficient = args.dp / global_n_labels

        total_loss, total_accuracy = 0.0, 0.0
        for i, (input_ids, output_ids) in enumerate(batch):
            input_ids = input_ids.to(device, non_blocking=True)
            output_ids = output_ids.to(device, non_blocking=True)

            with model.no_sync() if i < len(batch) - 1 else nullcontext():
                logits = model(input_ids=input_ids).logits.float()
                loss = F.cross_entropy(logits.flatten(0, 1), output_ids.flatten(0, 1), ignore_index=-100, reduction="sum")
                loss = loss * coefficient
                loss.backward()

                with torch.no_grad():
                    total_loss += loss.detach()
                    total_accuracy += coefficient * ((logits.argmax(-1) == output_ids) & (output_ids != -100)).float().sum()

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1_000_000_000)  # only measures the norm

        optimizer.step()
        model.zero_grad()
        scheduler.step()

        with torch.no_grad():
            message = torch.stack([total_loss, total_accuracy])
            torch.distributed.all_reduce(message, torch.distributed.ReduceOp.AVG)
            total_loss, total_accuracy = message.tolist()

        if args.use_wandb:
            import wandb
            wandb.log({
                "train/loss": total_loss,
                "train/accuracy": total_accuracy,
                "stats/learning_rate": optimizer.param_groups[0]["lr"],
                "stats/grad_norm": grad_norm,
                "stats/epoch": epoch
            })


def save(model, tokenizer, args):
    """Gathers the tensor-parallel shards and saves the model in the Hugging Face format."""
    state_dict = {}
    for n in args.parameter_names:
        p = rgetattr(model.module, n)
        if hasattr(p, "_st_info"):
            dtensor = DTensor.from_local(p, p._st_info.mesh, p._st_info.placements, run_check=False)
            state_dict[n] = dtensor.full_tensor().cpu()
        elif hasattr(p.data, "_local_tensor"):
            state_dict[n] = p.data.full_tensor().cpu()
        else:
            state_dict[n] = p.cpu()

    if is_main_process():
        torch.save(state_dict, f"{args.output_dir}/pytorch_model.bin")
        model.module.config.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        print(f"Model saved to {args.output_dir}", flush=True)


if __name__ == "__main__":
    args = parse_arguments()
    tokenizer = load_tokenizer(args)
    device = setup_training(args)
    model, optimizer, scheduler, train_dataloader = prepare_model_and_optimizer(args, device, tokenizer)

    for epoch in range(args.max_epochs):
        training_epoch(model, train_dataloader, optimizer, scheduler, args, device, epoch)

    save(model, tokenizer, args)

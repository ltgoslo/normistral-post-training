import os
import random

import torch
import torch.distributed as dist


def warmup_then_constant_schedule(optimizer, num_warmup_steps: int):
    """Linear warm-up followed by a constant learning rate."""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return 1.0

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def seed_everything(seed_value=42):
    os.environ["PYTHONHASHSEED"] = str(seed_value)
    random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)


def get_rank():
    if not dist.is_available() or not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_world_size():
    if not dist.is_available() or not dist.is_initialized():
        return 1
    return dist.get_world_size()


def is_main_process():
    return get_rank() == 0

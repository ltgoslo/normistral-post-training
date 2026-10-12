import time

import torch.distributed as dist

from worker import Worker


class WorkerIdle(Worker):
    """Placeholder worker for GPUs without a role; it only takes part in the global barriers."""

    def __init__(self, config, use_barrier=True):
        super().__init__(config)
        self.use_barrier = use_barrier

    def step(self):
        time.sleep(1)
        if self.use_barrier:
            for _ in range(4):
                dist.barrier()

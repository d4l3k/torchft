import os
from typing import Optional, TYPE_CHECKING
import sys
from unittest.mock import patch

from torch.nn import parallel
import torch
from torch import nn
from torch.distributed.algorithms.join import Joinable
import torch.distributed as dist
from torchft.process_group import ProcessGroup
from torchft.process_group import ProcessGroupGloo
from torchft.process_group import ProcessGroupDummy

if TYPE_CHECKING:
    from torchft.manager import Manager


class DistributedDataParallel(parallel.DistributedDataParallel):
    """
    This is a patched DistributedDataParallel implementation that makes it
    compatible with torchft.

    Important notes:
    * This requires states to be synced on step 0 using an external mechanism
      rather than an internal broadcast (torchft.Manager will do this).
    * Using non-basic features of the DDP may cause your model to catch fire as
      they haven't been tested with torchft.
    * This doesn't any sanity checks such as verifying parameter sizes are the
      same across workers.
    """

    def __init__(self, manager: "Manager", module: nn.Module, **args) -> None:
        # use a dummy PG to soak up the init all reduce, actual comms will go
        # through the comm_hook.
        pg = ProcessGroupDummy(0, 1)

        super().__init__(module, process_group=pg, **args)

        self.register_comm_hook(manager, self._comm_hook)

    @staticmethod
    def _comm_hook(
        state: "Manager", bucket: dist.GradBucket
    ) -> torch.futures.Future[torch.Tensor]:
        return state.allreduce_grad(bucket.buffer())


class PureDistributedDataParallel(nn.Module):
    """
    A pure Python reimplementation of the DDP wrapper.
    """

    def __init__(self, manager: "Manager", module: nn.Module):
        super().__init__()

        self.module = module

        def post_grad_hook(p):
            if p.grad is not None:
                manager.allreduce_grad(p.grad)

        for p in module.parameters():
            p.register_post_accumulate_grad_hook(post_grad_hook)

    def forward(self, *args: object) -> object:
        return self.module(*args)

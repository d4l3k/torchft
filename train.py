import sys
import logging

import torch
from torch import nn, optim

from torchft import Manager, ProcessGroupGloo, DistributedDataParallel, Optimizer

logging.basicConfig(level=logging.INFO)

device = "cpu"


def load_state_dict(state_dict):
    m.load_state_dict(state_dict["model"])
    optimizer.load_state_dict(state_dict["optim"])


def state_dict():
    return {
        "model": m.state_dict(),
        "optim": optimizer.state_dict(),
    }


manager = Manager(
    pg=ProcessGroupGloo(),
    min_replica_size=2,
    load_state_dict=load_state_dict,
    state_dict=state_dict,
)

m = nn.Linear(2, 3)
m = DistributedDataParallel(manager, m)
optimizer = Optimizer(manager, optim.AdamW(m.parameters()))

print(m)

for i in range(10000):
    batch = torch.rand(2, 2, device=device)

    # must be called at the beginning of each train loop
    # Quorum computation is triggered here but only needed in the backwards pass.
    optimizer.zero_grad()

    out = m(batch)
    loss = out.sum()

    # Gradient all reduce overlaps with the backwards pass.
    loss.backward()

    # must be called at the end of the train loop
    # This may not actually step the optimizer if an error occured during grad allreduce.
    optimizer.step()

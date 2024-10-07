import sys
import logging

import torch
from torch import nn, optim

from torchft import Manager, ReconfigPGGloo, DistributedDataParallel, Optimizer

logging.basicConfig(level=logging.INFO)

device = "cpu"

m = nn.Linear(2, 3)


manager = Manager(
    pg=ReconfigPGGloo(),
    load_state_dict=m.load_state_dict,
    state_dict=m.state_dict,
)

m = DistributedDataParallel(manager, m)
optimizer = Optimizer(manager, optim.AdamW(m.parameters()))

print(m)

for i in range(1000):
    batch = torch.rand(2, 2, device=device)

    # must be called at the beginning of each train loop
    optimizer.zero_grad()

    out = m(batch)
    loss = out.sum()

    loss.backward()

    for p in m.parameters():
        if p.grad is not None:
            manager.allreduce_grad(p.grad)

    # must be called at the end of the train loop
    optimizer.step()

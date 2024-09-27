import sys
import logging

import torch
from torch import nn, optim

# torchft import (TODO hack)
sys.path.append(".")
from python import Manager, ReconfigPGGloo

logging.basicConfig(level=logging.INFO)

device = "cpu"

m = nn.Linear(2, 3)

optimizer = optim.AdamW(m.parameters())

manager = Manager(ReconfigPGGloo(), None, None)

print(m)

for i in range(1000):
    manager.step()

    batch = torch.rand(2, 2, device=device)

    optimizer.zero_grad()

    out = m(batch)
    loss = out.sum()

    loss.backward()

    for p in m.parameters():
        if p.grad is not None:
            manager.allreduce_grad(p.grad)
    
    if manager.should_commit():
        optimizer.step()


    

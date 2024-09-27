import sys

sys.path.append(".")

from python import Manager

import torch
from torch import nn, optim

device = "cpu"

m = nn.Linear(2, 3)

optimizer = optim.AdamW(m.parameters())

manager = Manager(None, None)

print(m)

for i in range(1000):
    manager.step()

    batch = torch.rand(2, 2, device=device)

    optimizer.zero_grad()

    out = m(batch)
    loss = out.sum()

    loss.backward()

    for p in optimizer.parameters():
        if p.grad is not None:
            manager.allreduce_grad(p.grad)
    
    if manager.should_commit():
        optimizer.step()


    

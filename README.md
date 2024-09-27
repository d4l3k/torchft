# torch-ft
Prototype repo for PyTorch fault tolerance

This implements a lighthouse server that coordinates across the different
replica groups and then a per replica group manager and fault tolerance library
that can be used in a standard PyTorch training loop.

## Lighthouse

You can start a lighthouse server by running:

```sh
$ cargo run --bin lighthouse -- --min_replicas 2
```

## Manager

TODO: not implemented yet

```py
from torchft import Manager, NCCLBuilder

m = Model()
optim = ...

manager = Manager(
    rank=0,
    load_state_dict=m.load_state_dict,
    state_dict=m.state_dict,
    process_group_builder=NCCLBuilder(),
)

# TODO: maybe this is a bad idea?
# to save/load you should use the manager state dict as it correctly tracks step
# counts
manager.load_state_dict(manager.state_dict())

for batch in ...:
    manager.step()

    optim.zero_grad()

    loss = ...
    loss.backward()

    for p in optim.parameters():
        if p.grad is not None:
            manager.allreduce_grad(p.grad)
    
    if manager.should_commit():
        optim.step()
```
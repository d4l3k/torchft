# torch-ft
Prototype repo for PyTorch fault tolerance

This implements a lighthouse server that coordinates across the different
replica groups and then a per replica group manager and fault tolerance library
that can be used in a standard PyTorch training loop.

This allows for membership changes at the training step granularity which can
greatly improve efficiency by avoiding stop the world training on errors.

## Lighthouse

You can start a lighthouse server by running:

```sh
$ RUST_BACKTRACE=1 cargo run --bin lighthouse -- --min_replicas 1 --quorum_tick_ms 100 --join_timeout_ms 1000
```

## Example Training Loop

See [train.py](./train.py) for the full example.

Invoke with:

```sh
$ TORCHFT_MANAGER_PORT=29512 TORCH_LIGHTHOUSE=http://localhost:19510 torchrun --master_port 29501 --nnodes 1 --nproc_per_node 1 train.py
```

train.py:

```py
from torchft import Manager, ReconfigPGGloo

m = nn.Linear(2, 3)

optimizer = optim.AdamW(m.parameters())

manager = Manager(
    pg=ReconfigPGGloo(), 
    load_state_dict=m.load_state_dict, 
    state_dict=m.state_dict,
)

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
```

## Building Python Extension

This uses pyo3+maturin to build the package.

To install in editable mode w/ the Rust extensions you can use the normal pip install command:

```sh
$ pip install -e .
```

## Running Tests / Lint

```sh
$ cargo fmt
% cargo test
```

## License

Apache 2.0 -- see [LICENSE](./LICENSE) for more details.

Copyright (c) Tristan Rice 2024

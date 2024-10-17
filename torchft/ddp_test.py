from unittest import TestCase
from unittest.mock import create_autospec

import torch
from torch import nn
import torch.distributed as dist
from torch.futures import Future

from torchft.ddp import (
    PureDistributedDataParallel,
    DistributedDataParallel,
    HackedDistributedDataParallel,
)
from torchft.process_group import ProcessGroupGloo, ProcessGroupBabyGloo
from torchft.manager import Manager


class TestDDP(TestCase):
    def test_pure_ddp(self):
        manager = create_autospec(Manager)

        m = nn.Linear(3, 4)
        m = PureDistributedDataParallel(manager, m)

        inp = torch.rand(2, 3)
        out = m(inp)
        loss = out.mean()
        loss.backward()

        for p in m.parameters():
            self.assertIsNotNone(p.grad)

        self.assertEqual(manager.allreduce_grad.call_count, len(list(m.parameters())))

    def test_ddp(self):
        manager = create_autospec(Manager)

        call_count = 0

        def allreduce_grad(tensor: torch.Tensor) -> Future[torch.Tensor]:
            nonlocal call_count

            call_count += 1

            fut = Future()
            fut.set_result(tensor)
            return fut

        manager.allreduce_grad = allreduce_grad

        m = nn.Linear(3, 4)
        m = DistributedDataParallel(manager, m)

        inp = torch.rand(2, 3)
        out = m(inp)
        loss = out.mean()
        loss.backward()

        for p in m.parameters():
            self.assertIsNotNone(p.grad)

        self.assertGreaterEqual(call_count, 1)

    def test_ddp_hacked(self):
        store = dist.TCPStore(
            "127.0.0.1",
            port=0,
            is_master=True,
            wait_for_workers=False,
        )
        addr = f"127.0.0.1:{store.port}/foo"
        pg = ProcessGroupGloo()
        pg.configure(addr, 0, 1)

        m = nn.Linear(3, 4)
        m = HackedDistributedDataParallel(m, pg)

        inp = torch.rand(2, 3)
        out = m(inp)
        loss = out.mean()
        loss.backward()

        for p in m.parameters():
            self.assertIsNotNone(p.grad)

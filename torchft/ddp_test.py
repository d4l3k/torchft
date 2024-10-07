from unittest import TestCase
from unittest.mock import create_autospec

import torch
from torch import nn

from torchft.process_group import ReconfigPG, ReconfigPGDummy
from torchft.ddp import DistributedDataParallel


class TestDDP(TestCase):
    def test_ddp(self):
        pg = ReconfigPGDummy()

        m = nn.Linear(3, 4)
        m = DistributedDataParallel(m, pg)

        inp = torch.rand(2, 3)
        out = m(inp)
        loss = out.mean()
        loss.backward()

        for p in m.parameters():
            self.assertIsNotNone(p.grad)

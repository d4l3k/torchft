from unittest import TestCase

import torch
from torch.distributed import TCPStore, ReduceOp

from torchft.process_group import ProcessGroupBabyGloo, ProcessGroupGloo


class ProcessGroupTest(TestCase):
    def test_gloo(self) -> None:
        store = TCPStore(
            host_name="localhost", port=0, is_master=True, wait_for_workers=False
        )

        store_addr = f"localhost:{store.port}/prefix"
        pg = ProcessGroupGloo()
        pg.configure(store_addr, 0, 1)

        at = torch.tensor([2])

        a_work = pg.allreduce([at], ReduceOp.SUM)
        a_work.wait()

    def test_baby_gloo(self) -> None:
        store = TCPStore(
            host_name="localhost", port=0, is_master=True, wait_for_workers=False
        )

        store_addr = f"localhost:{store.port}/prefix"

        a = ProcessGroupBabyGloo()
        b = ProcessGroupBabyGloo()

        a.configure(store_addr, 0, 2)
        b.configure(store_addr, 1, 2)

        at = torch.tensor([1]).share_memory_()
        bt = torch.tensor([2]).share_memory_()

        a_work = a.allreduce([at], ReduceOp.SUM)
        b_work = b.allreduce([bt], ReduceOp.SUM)

        a_work.wait()
        b_work.wait()

        torch.testing.assert_close(at, bt)

        with self.assertRaisesRegex(AssertionError, "shared"):
            a_work = a.allreduce([torch.tensor(10)], ReduceOp.SUM)

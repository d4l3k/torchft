from abc import ABC
import logging
from typing import Type, List, Optional
from datetime import timedelta

from torch.futures import Future
from torch.distributed import (
    ProcessGroup as BaseProcessGroup,
    Store,
    TCPStore,
    PrefixStore,
    BroadcastOptions,
    ProcessGroupGloo as BaseProcessGroupGloo,
    ProcessGroupNCCL as BaseProcessGroupNCCL,
)
import torch.distributed as dist
from torch.distributed.distributed_c10d import Work
import torch
import torch.multiprocessing as mp

logger = logging.getLogger(__name__)


def _get(queue: mp.Queue, timeout) -> object:
    v = queue.get(timeout=timeout)
    if isinstance(v, Exception):
        raise v
    return v


def create_store(store_addr: str) -> Store:
    """
    Creates a PrefixStore(TCPStore(...)) from an address in the format:

    host:port/prefix

    Ex: localhost:1234/my/prefix
    """
    host, _, rest = store_addr.partition(":")
    port, _, prefix = rest.partition("/")

    store = TCPStore(
        host_name=host,
        port=int(port),
        is_master=False,
        wait_for_workers=False,
    )
    store = PrefixStore(prefix, store)
    return store


class ProcessGroup(BaseProcessGroup):
    def configure(self, store_addr: str, rank: int, world_size: int) -> None:
        raise NotImplementedError("not implemented")

    def allreduce(self, tensors: List[torch.Tensor], opts: object) -> Work:
        raise NotImplementedError("not implemented")

    def allgather(
        self,
        output_tensors: List[List[torch.Tensor]],
        input_tensor: List[torch.Tensor],
        opts: object,
    ) -> Work:
        raise NotImplementedError("not implemented")

    def broadcast(self, tensor_list: List[torch.Tensor], opts: object) -> Work:
        raise NotImplementedError("not implemented")

    def broadcast_one(self, tensor: torch.Tensor, root: int) -> Work:
        opts = BroadcastOptions()
        opts.rootRank = root
        return self.broadcast([tensor], opts)

    def size(self) -> int:
        raise NotImplementedError("not implemented")

    def getBackendName(self) -> str:
        raise NotImplementedError("not implemented")


class ProcessGroupGloo(ProcessGroup):
    """
    This is a wrapper around ProcessGroupGloo with a reconfiguration argument.
    """

    def __init__(self) -> None:
        super().__init__(0, 1)
        self._pg = None

    def configure(self, store_addr: str, rank: int, world_size: int) -> None:
        store = create_store(store_addr)

        # TODO: set lower timeout
        # pyre-fixme[16]: no attribute ProcessGroupGloo
        self._pg = BaseProcessGroupGloo(store, rank, world_size)

    def allreduce(self, tensors: List[torch.Tensor], opts: object) -> Work:
        return self._pg.allreduce(tensors, opts)

    def allgather(
        self,
        output_tensors: List[List[torch.Tensor]],
        input_tensor: List[torch.Tensor],
        opts: object,
    ) -> Work:
        return self._pg.allgather(output_tensors, input_tensor, opts)

    def broadcast(self, tensor_list: List[torch.Tensor], opts: object) -> Work:
        return self._pg.broadcast(tensor_list, opts)

    def size(self) -> int:
        return self._pg.size()

    def getBackendName(self) -> str:
        return "torchft-gloo"


class DummyWork(dist._Work):
    def __init__(self, result):
        super().__init__()
        self.result_ = result
        self.future_ = torch.futures.Future()
        self.future_.set_result(result)

    def wait(self, timeout):
        return True

    def get_future(self):
        return self.future_


class ProcessGroupDummy(ProcessGroup):
    """
    This PG only supports world_size of 1
    """

    def __init__(self, rank, world):
        super().__init__(rank, world)
        assert rank == 0
        assert world == 1

        self._rank = rank
        self._world = world
        self.wait_count = 0
        self.get_future_count = 0
        self._work = []

    def broadcast(self, tensor_list, opts):
        res = DummyWork(tensor_list)
        self._work.append(res)
        return res

    def allgather(self, output_tensors, input_tensor, opts):
        for o, i in zip(output_tensors[0], input_tensor):
            o.copy_(i)

        res = DummyWork(output_tensors)
        self._work.append(res)
        return res

    def allreduce(self, tensors, opts):
        res = DummyWork(tensors)
        self._work.append(res)
        return res

    def size(self):
        return self._world

    def getBackendName(self):
        return "torchft-dummy"


class BabyWork(Work):
    def __init__(self, tx: mp.Queue, rx: mp.Queue, op_id: int, timeout: float):
        super().__init__()

        self._tx = tx
        self._rx = rx
        self._op_id = op_id
        self._timeout = timeout

    def wait(self) -> bool:
        self._tx.put(("wait", self._op_id), timeout=self._timeout)
        assert _get(self._rx, self._timeout) == self._op_id
        return True


class ProcessGroupBaby(ProcessGroup):
    """
    This is a process group that runs the underlying process group in a
    subprocess. Since it's running in a subprocess all tensors need to be in
    shared memory or will be moved to shared memory. CUDA tensors are implicitly
    share able and don't need any changes.

    If the child process is killed while an operation is running CUDA tensors
    may leak in the current implementation.

    For the NCCL backend, extra memory will be used by the subprocesses CUDA
    context compared to running NCCL in the main process. This is typically
    around ~1GB.
    """

    PG_CLASS: Type[BaseProcessGroup]

    def __init__(self, timeout: float = 60.0) -> None:
        super().__init__(0, 1)

        self._world_size = -1

        self._p = None
        self._tx = None
        self._rx = None

        self._timeout = timeout

    @classmethod
    def _worker(
        cls, store_addr: str, rank: int, world_size: int, rx: mp.Queue, tx: mp.Queue
    ) -> None:
        try:
            store = create_store(store_addr)

            pg = cls.PG_CLASS(store, rank, world_size)

            work = {}
            next_op_id = 0

            while True:
                op = rx.get()
                cmd = op[0]
                if cmd == "allreduce":
                    work[next_op_id] = pg.allreduce(op[1], op[2])
                    tx.put(next_op_id)
                    next_op_id += 1
                elif cmd == "wait":
                    op_id = op[1]
                    work[op_id].wait()
                    del work[op_id]
                    tx.put(op_id)
                else:
                    raise ValueError(f"unknown cmd: {cmd}")
        except Exception as e:
            logger.exception("worker errored")
            tx.put(e)

    def configure(self, store_addr: str, rank: int, world_size: int) -> None:
        if self._p is not None:
            self._p.kill()

        self._world_size = world_size

        ctx = mp.get_context("spawn")
        self._tx = ctx.Queue()
        self._rx = ctx.Queue()

        self._p = ctx.Process(
            target=self._worker,
            args=(store_addr, rank, world_size, self._tx, self._rx),
            daemon=True,
        )
        self._p.start()

    def allreduce(self, tensors: List[torch.Tensor], opts: object) -> Work:
        assert isinstance(tensors, list), "input must be list"

        for tensor in tensors:
            if not tensor.is_shared():
                tensor.share_memory_()

        self._tx.put(("allreduce", tensors, opts), timeout=self._timeout)
        op_id = _get(self._rx, self._timeout)
        assert isinstance(op_id, int), f"invalid return {op_id}"
        return BabyWork(tx=self._tx, rx=self._rx, op_id=op_id, timeout=self._timeout)

    def size(self) -> int:
        return self._world_size


class ProcessGroupBabyGloo(ProcessGroupBaby):
    PG_CLASS = BaseProcessGroupGloo

    def getBackendName(self):
        return "torchft-baby-gloo"


class ProcessGroupBabyNCCL(ProcessGroupBaby):
    PG_CLASS = BaseProcessGroupGloo

    def getBackendName(self):
        return "torchft-baby-nccl"

from abc import ABC
import logging
from typing import Type, List

from torch.distributed import (
    ProcessGroup as BaseProcessGroup,
    Work,
    Store,
    TCPStore,
    PrefixStore,
    ProcessGroupGloo as BaseProcessGroupGloo,
)
import torch
import torch.multiprocessing as mp

# pyre-fixme[21]: no attribute ProcessGroupGloo
from torch.distributed import ProcessGroupGloo

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
    def configure(self, store_addr: str, rank: int, world_size: int) -> None: ...

    def allreduce(self, tensors: List[torch.Tensor], opts: object) -> Work: ...


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


class ProcessGroupDummy(ProcessGroup):
    def __init__(self) -> None:
        super().__init__(0, 1)

    def configure(self, store_addr: str, rank: int, world_size: int) -> None:
        pass

    def allreduce(self, tensors: List[torch.Tensor], opts: object) -> Work:
        # TODO: return work object
        pass


class BabyWork(Work):
    def __init__(self, tx: mp.Queue, rx: mp.Queue, op_id: int, timeout: float):
        super().__init__()

        self._tx = tx
        self._rx = rx
        self._op_id = op_id
        self._timeout = timeout

    def wait(self) -> None:
        self._tx.put(("wait", self._op_id), timeout=self._timeout)
        assert _get(self._rx, self._timeout) == self._op_id


class ProcessGroupBaby(ProcessGroup):
    PG_CLASS: Type[BaseProcessGroup]

    def __init__(self, timeout: float = 60.0) -> None:
        super().__init__(0, 1)

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
            assert tensor.is_shared(), "tensor must be in shared memory to be reduced"

        self._tx.put(("allreduce", tensors, opts), timeout=self._timeout)
        op_id = _get(self._rx, self._timeout)
        assert isinstance(op_id, int), f"invalid return {op_id}"
        return BabyWork(tx=self._tx, rx=self._rx, op_id=op_id, timeout=self._timeout)


class ProcessGroupBabyGloo(ProcessGroupBaby):
    PG_CLASS = BaseProcessGroupGloo

from abc import ABC

# pyre-fixme[21]: no attribute ProcessGroupGloo
from torch.distributed import ProcessGroupGloo, ProcessGroup, Work
import torch


class ReconfigPG(ProcessGroup):
    def configure(self, store, rank: int, world_size: int) -> None: ...

    def allreduce(self, tensor: torch.Tensor, opts: object) -> Work: ...


class ReconfigPGGloo(ReconfigPG):
    """
    This is a wrapper around ProcessGroupGloo with a reconfiguration argument.
    """

    def __init__(self) -> None:
        super().__init__(0, 1)
        self._pg = None

    def configure(self, store, rank: int, world_size: int) -> None:
        # TODO: set lower timeout
        # pyre-fixme[16]: no attribute ProcessGroupGloo
        self._pg = ProcessGroupGloo(store, rank, world_size)

    def allreduce(self, tensor: torch.Tensor, opts: object) -> None:
        return self._pg.allreduce(tensor)


class ReconfigPGDummy(ReconfigPG):
    def __init__(self) -> None:
        super().__init__(0, 1)

    def _get_backend_name() -> str:
        return "dummy"

    def configure(self, store, rank: int, world_size: int) -> None:
        pass

    def allreduce(self, tensor, opts: object) -> None:
        pass

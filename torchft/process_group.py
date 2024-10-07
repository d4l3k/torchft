from abc import ABC

# pyre-fixme[21]: no attribute ProcessGroupGloo
from torch.distributed import ProcessGroupGloo


class ReconfigPG(ABC):
    def configure(self, store, rank: int, world_size: int) -> None: ...

    def allreduce(self, tensor) -> None: ...


class ReconfigPGGloo(ReconfigPG):
    """
    This is a wrapper around ProcessGroupGloo with a reconfiguration argument.
    """

    def __init__(self) -> None:
        self._pg = None

    def configure(self, store, rank: int, world_size: int) -> None:
        # TODO: set lower timeout
        # pyre-fixme[16]: no attribute ProcessGroupGloo
        self._pg = ProcessGroupGloo(store, rank, world_size)

    def allreduce(self, tensor) -> None:
        work = self._pg.allreduce(tensor)
        work.wait()

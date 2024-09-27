from abc import ABC

from torch.distributed import ProcessGroupGloo


class ReconfigPG(ABC):
    def configure(self, store, rank: int, world_size: int) -> None: ...

    def allreduce(self, tensor) -> None: ...


class ReconfigPGGloo(ReconfigPG):
    """
    This is a wrapper around ProcessGroupGloo with a reconfiguration argument.
    """

    def __init__(self) -> None:
        pass

    def configure(self, store, rank: int, world_size: int) -> None:
        # TODO: set lower timeout
        self._pg = ProcessGroupGloo(store, rank, world_size)

    def allreduce(self, tensor) -> None:
        work = self._pg.allreduce(tensor)
        work.wait()

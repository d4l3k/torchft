from abc import ABC

from torch.distributed import ProcessGroup as BaseProcessGroup, Work
import torch


class ProcessGroup(BaseProcessGroup):
    def configure(self, store, rank: int, world_size: int) -> None: ...

    def allreduce(self, tensor: torch.Tensor, opts: object) -> Work: ...


class ProcessGroupGloo(ProcessGroup):
    """
    This is a wrapper around ProcessGroupGloo with a reconfiguration argument.
    """

    def __init__(self) -> None:
        super().__init__(0, 1)
        self._pg = None

    def configure(self, store, rank: int, world_size: int) -> None:
        # pyre-fixme[21]: no attribute ProcessGroupGloo
        from torch.distributed import ProcessGroupGloo

        # TODO: set lower timeout
        # pyre-fixme[16]: no attribute ProcessGroupGloo
        self._pg = ProcessGroupGloo(store, rank, world_size)

    def allreduce(self, tensor: torch.Tensor, opts: object) -> Work:
        return self._pg.allreduce(tensor)


class ProcessGroupDummy(ProcessGroup):
    def __init__(self) -> None:
        super().__init__(0, 1)

    def configure(self, store, rank: int, world_size: int) -> None:
        pass

    def allreduce(self, tensor, opts: object) -> Work:
        # TODO: return work object
        pass

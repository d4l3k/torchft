import os
import uuid
import socket
from typing import Dict
import time
import logging

from torch.distributed import TCPStore, PrefixStore
from torch.optim import Optimizer

# pyre-fixme[21]: can't find rust module
from torchft.torchft import Manager as _Manager, ManagerClient
from torchft.checkpointing import CheckpointServer

logger: logging.Logger = logging.getLogger(__name__)

MANAGER_ADDR_KEY: str = "manager_addr"
MANAGER_DEFAULT_PORT: int = int(os.environ.get("TORCHFT_MANAGER_PORT", 29511))


class Manager:
    """
    Manager manages the full fault tolerant training loop.

    NOTE: when saving periodic checkpoints you must save and restore the
    Manager's state_dict as well to avoid synchronization issues.
    """

    def __init__(
        self, pg, load_state_dict, state_dict, port: int = MANAGER_DEFAULT_PORT
    ) -> None:
        self._load_state_dict = load_state_dict
        self._state_dict = state_dict

        store_addr = os.environ["MASTER_ADDR"]
        store_port = int(os.environ["MASTER_PORT"])
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        self._rank = rank

        self._ckpt_server = CheckpointServer(state_dict)

        self._store = TCPStore(
            host_name=store_addr,
            port=store_port,
            is_master=False,
            wait_for_workers=False,
        )
        self._pg = pg

        if rank == 0:
            hostname = socket.gethostname()
            addr = f"http://{hostname}:{port}"
            bind = f"[::]:{port}"
            lighthouse_addr = os.environ["TORCHFT_LIGHTHOUSE"]

            replica_id = str(uuid.uuid4())
            # pyre-fixme[16]: can't find rust module
            self._manager = _Manager(
                replica_id=replica_id,
                lighthouse_addr=lighthouse_addr,
                address=addr,
                bind=bind,
                store_addr=f"{store_addr}:{store_port}",
                world_size=world_size,
            )

            self._store.set(MANAGER_ADDR_KEY, addr)

        addr = self._store.get(MANAGER_ADDR_KEY).decode("utf-8")
        # pyre-fixme[16]: can't find rust module
        self._client = ManagerClient(addr)

        self._step = 0
        self._quorum_id = -1
        self._errored = False
        self._healing = False

    def shutdown(self) -> None:
        self._ckpt_server.shutdown()

    def allreduce_grad(self, tensor) -> None:
        if self._errored:
            return
        try:
            handle = self._pg.allreduce(tensor, None)
            handle.wait()
            # TODO: rescale tensor according to num_max
        except Exception as e:
            logger.exception("got exception in all reduce -- skipping remaining")
            self._errored = True

    def step(self) -> None:
        self._step += 1
        self._errored = False
        self._ckpt_server.allow_checkpoint(self._step)

        # TODO: we should really be wrapping this whole section in a try-except
        # block to allow gracefully recovering from issues in process.

        # TODO: run this on a background thread pool

        # TODO: broadcast the weights iff step 0/1 to ensure initial model state
        # is in sync

        (
            quorum_id,
            replica_rank,
            replica_world,
            address,
            store_address,
            max_step,
            num_max,
        ) = self._client.quorum(
            rank=self._rank,
            step=self._step,
            checkpoint_server_addr=self._ckpt_server.address(),
        )

        if quorum_id != self._quorum_id:
            logger.info(f"reconfiguring for quorum_id {quorum_id}")
            # needs reconfig
            addr, _, port = store_address.rpartition(":")
            store = TCPStore(
                host_name=addr,
                port=int(port),
                is_master=False,
                wait_for_workers=False,
            )
            store = PrefixStore(f"torchft/{quorum_id}/{self._rank}", store)
            self._pg.configure(store, replica_rank, replica_world)
            self._quorum_id = quorum_id

        # TODO: on step 0 we need everyone to load a consistent checkpoint to
        # avoid initialization differences

        self._healing = self._step != max_step
        if self._healing:
            logger.info(f"detected behind step={self._step}, max_step={max_step}")

            logger.info(f"fetching checkpoint server address from {address}")
            # pyre-fixme[16]: can't find rust module
            primary_client = ManagerClient(address)
            checkpoint_server_address = primary_client.checkpoint_address(self._rank)

            state_dict = CheckpointServer.load_from_address(checkpoint_server_address)
            self._load_state_dict(state_dict)

            self._step = max_step

    def should_commit(self) -> bool:
        self._ckpt_server.disallow_checkpoint()

        # TODO: sync error condition
        if self._errored:
            return False
        return True

    def load_state_dict(self, state_dict: Dict[str, int]) -> None:
        self._step = state_dict["step"]

    def state_dict(self) -> Dict[str, int]:
        return {"step": self._step}

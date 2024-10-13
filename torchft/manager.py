import os
import uuid
import socket
from typing import Dict
import time
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta

import torch
from torch.distributed import TCPStore, PrefixStore, Work, ReduceOp
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
        self,
        pg,
        load_state_dict,
        state_dict,
        min_replica_size: int,
        port: int = MANAGER_DEFAULT_PORT,
        use_async_quorum: bool = True,
        timeout: timedelta = timedelta(seconds=60),
    ) -> None:
        self._load_state_dict = load_state_dict
        self._state_dict = state_dict
        self._use_async_quorum = use_async_quorum
        self._timeout = timeout

        store_addr = os.environ["MASTER_ADDR"]
        store_port = int(os.environ["MASTER_PORT"])
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        self._rank = rank
        self._min_replica_size = min_replica_size

        self._ckpt_server = CheckpointServer(state_dict)
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._quorum_future = None

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
        self._client = ManagerClient(addr, timeout=timeout)

        self._step = 0
        self._quorum_id = -1
        self._errored = False
        self._healing = False
        self._participating_replicas = -1
        self._pending_work: Tuple[Work, List[torch.Tensor]] = []

        # first step is 1
        self._should_step = True

    def shutdown(self) -> None:
        self._ckpt_server.shutdown()

    def allreduce_grad(self, grad: torch.Tensor) -> None:
        if self._errored:
            return

        self._quorum_future.result()

        if self._healing:
            assert self._use_async_quorum
            grad.zero_()

        try:
            # Run the allreduce async and save the work object so we can wait on
            # it later.
            work = self._pg.allreduce([grad], ReduceOp.SUM)
            self._pending_work.append((work, [grad]))
        except Exception as e:
            logger.exception("got exception in all reduce -- skipping remaining")
            self._errored = True

    def step(self) -> None:
        if self._should_step:
            self._step += 1
        self._errored = False
        self._healing = False
        self._ckpt_server.allow_checkpoint(self._step)

        # TODO: we should really be wrapping this whole section in a try-except
        # block to allow gracefully recovering from issues in PG setup and quorum.

        self._quorum_future = self._executor.submit(self._async_quorum)
        if not self._use_async_quorum:
            self._quorum_future.result()

            # we are forcing healing at the beginning so we're in a good state
            # and don't need to zero_grad
            self._healing = False

    def _async_quorum(self) -> None:
        (
            quorum_id,
            replica_rank,
            replica_world,
            address,
            store_address,
            max_step,
            num_max,
            heal,
        ) = self._client.quorum(
            rank=self._rank,
            step=self._step,
            checkpoint_server_addr=self._ckpt_server.address(),
        )
        self._participating_replicas = (
            num_max if self._use_async_quorum else replica_world
        )

        if quorum_id != self._quorum_id:
            logger.info(f"reconfiguring for quorum_id {quorum_id}")
            store_prefixed_addr = f"{store_address}/torchft/{quorum_id}/{self._rank}"
            self._pg.configure(store_prefixed_addr, replica_rank, replica_world)
            self._quorum_id = quorum_id

        # See manager.rs for healing conditions
        if heal:
            self._healing = True
            logger.info("healing required")

            logger.info(f"fetching checkpoint server address from {address}")
            # pyre-fixme[16]: can't find rust module
            primary_client = ManagerClient(address, timeout=self._timeout)
            checkpoint_server_address = primary_client.checkpoint_address(self._rank)

            state_dict = CheckpointServer.load_from_address(checkpoint_server_address)
            self._load_state_dict(state_dict)

            self._step = max_step

    def should_commit(self) -> bool:
        if not self._errored:
            for work, tensors in self._pending_work:
                try:
                    # TODO: increase timeout when waiting when healing
                    work.wait()
                except:
                    logger.exception(
                        "got exception in all reduce -- skipping remaining"
                    )
                    self._errored = True
                    break

                # On CUDA devices this operation will run async with the
                # should_commit call below thus overlapping with the communication
                # overhead.
                for tensor in tensors:
                    tensor /= self._participating_replicas

        self._pending_work = []

        logger.info("should_commit")

        enough_replicas = self._participating_replicas >= self._min_replica_size
        local_should_commit = enough_replicas and not self._errored
        should_commit = self._client.should_commit(
            self._rank, self._step, local_should_commit
        )
        logger.info(
            f"should_commit={should_commit} enough_replicas={enough_replicas}, errored={self._errored}"
        )

        self._ckpt_server.disallow_checkpoint()

        # decide whether we're in a healthy state to increase the step count
        self._should_step = should_commit

        return should_commit

    def load_state_dict(self, state_dict: Dict[str, int]) -> None:
        self._step = state_dict["step"]

    def state_dict(self) -> Dict[str, int]:
        return {"step": self._step}

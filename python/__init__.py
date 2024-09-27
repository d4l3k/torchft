import os
import uuid
import socket
from typing import Dict
import time

from torch.distributed import TCPStore

from torchft import Manager as _Manager, ManagerClient

MANAGER_ADDR_KEY: str = "manager_addr"

class Manager:
    def __init__(self, load_state_dict, state_dict, port: int = 29511) -> None:
        self.load_state_dict = load_state_dict
        self.state_dict = state_dict

        store_addr = os.environ["MASTER_ADDR"]
        store_port = int(os.environ["MASTER_PORT"])
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

        self._store = TCPStore(
            host_name=store_addr, 
            port=store_port, 
            is_master=False, 
            wait_for_workers=False,
        )

        if rank == 0:
            hostname = socket.gethostname()
            addr = f"http://{hostname}:{port}"
            bind = f"0.0.0.0:{port}"
            lighthouse_addr = os.environ["TORCH_LIGHTHOUSE"]

            replica_id = str(uuid.uuid4())
            self._manager = _Manager(
                replica_id=replica_id,
                lighthouse_addr=lighthouse_addr,
                address=addr,
                bind=bind,
                store_addr=f"{store_addr}:{store_port}",
                world_size=world_size,
            )

            self._store.set(MANAGER_ADDR_KEY, addr)

        time.sleep(10)

        addr = self._store.get(MANAGER_ADDR_KEY).decode("utf-8")
        self._client = ManagerClient(addr)

        self._step = 0

    def allreduce_grad(self, tensor) -> None:
        raise NotImplemented("allreduce_grad")

    def step(self) -> None:
        raise NotImplemented("step")

    def should_commit(self) -> bool:
        raise NotImplemented("should_commit")

    def load_state_dict(self, state_dict: Dict[str, int]) -> None:
        self._step = state_dict["step"]

    def state_dict(self) -> Dict[str, int]:
        return {
            "step": self._step
        }
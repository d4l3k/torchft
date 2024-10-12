from unittest import TestCase
from unittest.mock import patch, create_autospec, MagicMock

import torch
from torch.distributed import TCPStore

from torchft.torchft import ManagerClient
from torchft.manager import Manager, MANAGER_ADDR_KEY
from torchft.process_group import ProcessGroup


class TestManager(TestCase):
    def _create_manager(self) -> Manager:
        pg = create_autospec(ProcessGroup)
        self.store = TCPStore(
            host_name="localhost", port=0, is_master=True, wait_for_workers=False
        )
        self.store.set(MANAGER_ADDR_KEY, "dummy")
        with patch(
            "os.environ",
            {
                "MASTER_ADDR": "localhost",
                "MASTER_PORT": self.store.port,
                "RANK": "1",
                "WORLD_SIZE": "2",
            },
        ):
            self.load_state_dict = MagicMock()
            manager = Manager(
                pg=pg,
                load_state_dict=self.load_state_dict,
                state_dict=lambda: {},
            )
        return manager

    @patch("torchft.manager.ManagerClient", autospec=True)
    def test_manager(self, client_mock) -> None:
        manager = self._create_manager()
        self.assertEqual(client_mock.call_count, 1)

    @patch("torchft.manager.ManagerClient", autospec=True)
    def test_state_dict(self, client_mock) -> None:
        manager = self._create_manager()

        state_dict = manager.state_dict()
        self.assertEqual(state_dict, {"step": 0})

        manager.load_state_dict({"step": 1234})
        self.assertEqual(manager._step, 1234)

    @patch("torchft.manager.ManagerClient", autospec=True)
    def test_quorum_happy(self, client_mock) -> None:
        manager = self._create_manager()
        client_mock().should_commit = lambda rank, step, should_commit: should_commit

        client_mock().quorum.return_value = (
            123,  # quorum_id
            1,  # replica_rank
            2,  # replica_world
            "manager address",
            f"localhost:{self.store.port}",
            1,  # max_step
            2,  # num_max
            False,  # heal
        )

        self.assertEqual(manager._quorum_id, -1)
        self.assertEqual(manager._step, 0)

        manager.step()
        manager.allreduce_grad(torch.tensor([1.0]))
        self.assertEqual(len(manager._pending_work), 1)
        self.assertTrue(manager.should_commit())
        self.assertEqual(len(manager._pending_work), 0)

        self.assertEqual(manager._quorum_id, 123)
        self.assertEqual(manager._step, 1)
        self.assertEqual(manager._pg.allreduce.call_count, 1)

    @patch("torchft.manager.ManagerClient", autospec=True)
    def test_quorum_behind(self, client_mock) -> None:
        manager = self._create_manager()
        client_mock().should_commit = lambda rank, step, should_commit: should_commit

        client_mock().quorum.return_value = (
            123,  # quorum_id
            1,  # replica_rank
            2,  # replica_world
            "manager address",
            f"localhost:{self.store.port}",
            20,  # max_step
            2,  # num_max
            True,  # heal
        )
        # forceable increment checkpoint server to compute correct address
        manager._ckpt_server.allow_checkpoint(1)

        client_mock().checkpoint_address.return_value = manager._ckpt_server.address()

        self.assertEqual(manager._quorum_id, -1)
        self.assertEqual(manager._step, 0)

        manager.step()
        manager.allreduce_grad(torch.tensor([1.0]))
        self.assertTrue(manager.should_commit())

        self.assertEqual(manager._quorum_id, 123)
        self.assertEqual(manager._step, 20)
        self.assertEqual(manager._pg.allreduce.call_count, 1)

        self.assertEqual(self.load_state_dict.call_count, 1)

    @patch("torchft.manager.ManagerClient", autospec=True)
    def test_allreduce_error(self, client_mock) -> None:
        manager = self._create_manager()
        client_mock().should_commit = lambda rank, step, should_commit: should_commit

        client_mock().quorum.return_value = (
            123,  # quorum_id
            1,  # replica_rank
            2,  # replica_world
            "manager address",
            f"localhost:{self.store.port}",
            1,  # max_step
            2,  # num_max
            False,  # heal
        )

        self.assertEqual(manager._quorum_id, -1)
        self.assertEqual(manager._step, 0)

        manager.step()
        manager.allreduce_grad(torch.tensor([1.0]))
        self.assertEqual(manager._pg.allreduce.call_count, 1)

        # inject failure
        manager._pg.allreduce.side_effect = RuntimeError("injected failure")
        manager.allreduce_grad(torch.tensor([1.0]))
        # this should be skipped due to error
        manager.allreduce_grad(torch.tensor([1.0]))
        self.assertEqual(manager._pg.allreduce.call_count, 2)

        self.assertFalse(manager.should_commit())
        self.assertTrue(manager._errored)

        # recover on next step
        client_mock().quorum.return_value = (
            123,  # quorum_id
            1,  # replica_rank
            2,  # replica_world
            "manager address",
            f"localhost:{self.store.port}",
            2,  # max_step
            2,  # num_max
            False,  # heal
        )
        manager._pg.allreduce.side_effect = None

        manager.step()
        manager.allreduce_grad(torch.tensor([1.0]))
        self.assertTrue(manager.should_commit())

from unittest import TestCase
from unittest.mock import patch, create_autospec, MagicMock

import torch
from torch.distributed import TCPStore

from torchft.torchft import ManagerClient
from torchft.manager import Manager, MANAGER_ADDR_KEY
from torchft.process_group import ProcessGroup


class TestManager(TestCase):
    def _create_manager(
        self, use_async_quorum: bool = True, min_replica_size: int = 2
    ) -> Manager:
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
                min_replica_size=min_replica_size,
                load_state_dict=self.load_state_dict,
                state_dict=lambda: {},
                use_async_quorum=use_async_quorum,
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
    def test_quorum_heal_sync(self, client_mock) -> None:
        manager = self._create_manager(use_async_quorum=False)
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
        self.assertFalse(manager._healing)
        self.assertTrue(manager.should_commit())

        self.assertEqual(manager._quorum_id, 123)
        self.assertEqual(manager._step, 20)
        self.assertEqual(manager._pg.allreduce.call_count, 1)
        self.assertEqual(manager._pg.allreduce.return_value.wait.call_count, 1)

        self.assertEqual(self.load_state_dict.call_count, 1)

    @patch("torchft.manager.ManagerClient", autospec=True)
    def test_quorum_heal_async_not_enough_participants(self, client_mock) -> None:
        manager = self._create_manager(use_async_quorum=True, min_replica_size=2)
        client_mock().should_commit = lambda rank, step, should_commit: should_commit

        client_mock().quorum.return_value = (
            123,  # quorum_id
            1,  # replica_rank
            2,  # replica_world
            "manager address",
            f"localhost:{self.store.port}",
            20,  # max_step
            1,  # num_max
            True,  # heal
        )
        # forceable increment checkpoint server to compute correct address
        manager._ckpt_server.allow_checkpoint(1)

        client_mock().checkpoint_address.return_value = manager._ckpt_server.address()

        self.assertEqual(manager._quorum_id, -1)
        self.assertEqual(manager._step, 0)

        manager.step()
        manager._quorum_future.result()
        self.assertTrue(manager._healing)

        grad = torch.tensor([1.0])
        manager.allreduce_grad(grad)
        torch.testing.assert_close(grad, torch.zeros_like(grad))
        # don't commit since num_max < min_replica_size
        self.assertFalse(manager.should_commit())
        self.assertFalse(manager._should_step)

        self.assertEqual(manager._quorum_id, 123)
        self.assertEqual(manager._step, 20)
        self.assertEqual(manager._pg.allreduce.call_count, 1)
        self.assertEqual(manager._pg.allreduce.return_value.wait.call_count, 1)

        self.assertEqual(self.load_state_dict.call_count, 1)

        # failed to commit so no step
        manager.step()
        self.assertEqual(manager._step, 20)

    @patch("torchft.manager.ManagerClient", autospec=True)
    def test_quorum_heal_async_zero_grad(self, client_mock) -> None:
        manager = self._create_manager(use_async_quorum=True, min_replica_size=1)
        client_mock().should_commit = lambda rank, step, should_commit: should_commit

        client_mock().quorum.return_value = (
            123,  # quorum_id
            1,  # replica_rank
            2,  # replica_world
            "manager address",
            f"localhost:{self.store.port}",
            20,  # max_step
            1,  # num_max
            True,  # heal
        )
        # forceable increment checkpoint server to compute correct address
        manager._ckpt_server.allow_checkpoint(1)

        client_mock().checkpoint_address.return_value = manager._ckpt_server.address()

        self.assertEqual(manager._quorum_id, -1)
        self.assertEqual(manager._step, 0)

        manager.step()
        manager._quorum_future.result()
        self.assertTrue(manager._healing)

        grad = torch.tensor([1.0])
        manager.allreduce_grad(grad)
        torch.testing.assert_close(grad, torch.zeros_like(grad))
        # don't commit since num_max < min_replica_size
        self.assertTrue(manager.should_commit())
        self.assertTrue(manager._should_step)

        self.assertEqual(manager._quorum_id, 123)
        self.assertEqual(manager._step, 20)
        self.assertEqual(manager._pg.allreduce.call_count, 1)
        self.assertEqual(manager._pg.allreduce.return_value.wait.call_count, 1)

        self.assertEqual(self.load_state_dict.call_count, 1)

        manager.step()
        self.assertEqual(manager._step, 21)

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

        # inject failure when work queued
        manager._pg.allreduce.side_effect = RuntimeError("injected failure")
        manager.allreduce_grad(torch.tensor([1.0]))
        self.assertTrue(manager._errored)
        # this should be skipped due to error
        manager.allreduce_grad(torch.tensor([1.0]))
        self.assertEqual(manager._pg.allreduce.call_count, 2)
        self.assertEqual(manager._pg.allreduce.return_value.wait.call_count, 0)

        self.assertFalse(manager.should_commit())
        self.assertTrue(manager._errored)

        # cleanup
        manager._pg.allreduce.side_effect = None

        # inject failure when worked waited
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
        manager.step()
        manager._pg.allreduce.return_value.wait.side_effect = RuntimeError(
            "injected failure"
        )
        manager.allreduce_grad(torch.tensor([1.0]))
        self.assertFalse(manager._errored)
        self.assertFalse(manager.should_commit())
        self.assertTrue(manager._errored)
        self.assertEqual(manager._pg.allreduce.return_value.wait.call_count, 1)

        # cleanup
        manager._pg.allreduce.return_value.wait.side_effect = None

        # recover on next step
        client_mock().quorum.return_value = (
            123,  # quorum_id
            1,  # replica_rank
            2,  # replica_world
            "manager address",
            f"localhost:{self.store.port}",
            3,  # max_step
            2,  # num_max
            False,  # heal
        )

        manager.step()
        manager.allreduce_grad(torch.tensor([1.0]))
        self.assertTrue(manager.should_commit())

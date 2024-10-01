from torchft.checkpointing import CheckpointServer

from unittest import TestCase
from unittest.mock import MagicMock


class TestCheckpointing(TestCase):
    def test_checkpoint_server(self) -> None:
        expected = {"state": "dict"}
        state_dict_fn = MagicMock()
        state_dict_fn.return_value = expected
        server = CheckpointServer(state_dict=state_dict_fn)

        addr = server.address()

        out = CheckpointServer.load_from_address(addr)
        self.assertEqual(out, expected)

        server.shutdown()

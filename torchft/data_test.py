from unittest import TestCase

from torchft.data import DistributedSampler
from torch.utils.data import Dataset


class DummyDataset(Dataset):
    def __init__(self, length: int) -> None:
        self.length = length

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> int:
        return idx


class TestData(TestCase):
    def test_distributed_sampler(self) -> None:
        dataset = DummyDataset(1000)
        sampler = DistributedSampler(
            dataset,
            replica_group=1,
            num_replica_groups=2,
            rank=3,
            num_replicas=4,
        )
        self.assertEqual(sampler.global_rank, 3 + 1 * 4)
        self.assertEqual(sampler.global_world_size, 2 * 4)

        sampler_iter = iter(sampler)
        self.assertEqual(next(sampler_iter), 500)

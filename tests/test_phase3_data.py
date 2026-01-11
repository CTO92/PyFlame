"""
Tests for PyFlame Phase 3 Data Loading Module.

Tests Dataset, DataLoader, Samplers, and Transforms.
"""

import pytest
import random
import sys
import os

# Add Python module to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from pyflame.data.dataset import (
    Dataset,
    TensorDataset,
    Subset,
    ConcatDataset,
    random_split,
    MapDataset,
)
from pyflame.data.dataloader import DataLoader, default_collate
from pyflame.data.sampler import (
    SequentialSampler,
    RandomSampler,
    BatchSampler,
    SubsetRandomSampler,
    WeightedRandomSampler,
    DistributedSampler,
)
from pyflame.data.transforms import (
    Compose,
    Lambda,
    Normalize,
    ToTensor,
    RandomHorizontalFlip,
    RandomApply,
)


# =============================================================================
# Dataset Tests
# =============================================================================

class SimpleDataset(Dataset):
    """Simple dataset for testing."""

    def __init__(self, size: int = 100):
        self.size = size

    def __getitem__(self, index: int):
        return index * 2

    def __len__(self) -> int:
        return self.size


class TestDataset:
    """Test cases for Dataset classes."""

    def test_simple_dataset(self):
        dataset = SimpleDataset(10)
        assert len(dataset) == 10
        assert dataset[0] == 0
        assert dataset[5] == 10

    def test_tensor_dataset_single(self):
        try:
            import numpy as np
            data = np.arange(100).reshape(10, 10)
            dataset = TensorDataset(data)
            assert len(dataset) == 10
            sample = dataset[0]
            assert len(sample) == 1
        except ImportError:
            pytest.skip("NumPy not available")

    def test_tensor_dataset_multiple(self):
        try:
            import numpy as np
            features = np.random.randn(100, 10)
            labels = np.random.randint(0, 5, size=100)
            dataset = TensorDataset(features, labels)
            assert len(dataset) == 100
            x, y = dataset[0]
            assert len(x) == 10
        except ImportError:
            pytest.skip("NumPy not available")

    def test_subset(self):
        dataset = SimpleDataset(100)
        subset = Subset(dataset, range(10, 20))
        assert len(subset) == 10
        assert subset[0] == 20  # dataset[10] = 10 * 2 = 20

    def test_concat_dataset(self):
        ds1 = SimpleDataset(50)
        ds2 = SimpleDataset(30)
        combined = ConcatDataset([ds1, ds2])
        assert len(combined) == 80
        assert combined[0] == 0
        assert combined[50] == 0  # First item of ds2

    def test_concat_with_addition(self):
        ds1 = SimpleDataset(50)
        ds2 = SimpleDataset(30)
        combined = ds1 + ds2
        assert len(combined) == 80

    def test_random_split_lengths(self):
        dataset = SimpleDataset(100)
        splits = random_split(dataset, [70, 20, 10])
        assert len(splits) == 3
        assert len(splits[0]) == 70
        assert len(splits[1]) == 20
        assert len(splits[2]) == 10

    def test_random_split_fractions(self):
        dataset = SimpleDataset(100)
        splits = random_split(dataset, [0.8, 0.1, 0.1])
        assert len(splits) == 3
        assert len(splits[0]) == 80
        assert len(splits[1]) == 10
        assert len(splits[2]) == 10

    def test_map_dataset(self):
        dataset = SimpleDataset(10)
        mapped = MapDataset(dataset, lambda x: x + 1)
        assert mapped[0] == 1
        assert mapped[5] == 11


# =============================================================================
# Sampler Tests
# =============================================================================

class TestSamplers:
    """Test cases for Sampler classes."""

    def test_sequential_sampler(self):
        dataset = SimpleDataset(10)
        sampler = SequentialSampler(dataset)
        indices = list(sampler)
        assert indices == list(range(10))

    def test_random_sampler(self):
        dataset = SimpleDataset(10)
        sampler = RandomSampler(dataset, generator=random.Random(42))
        indices = list(sampler)
        assert len(indices) == 10
        assert set(indices) == set(range(10))

    def test_random_sampler_with_replacement(self):
        dataset = SimpleDataset(10)
        sampler = RandomSampler(dataset, replacement=True, num_samples=20)
        indices = list(sampler)
        assert len(indices) == 20

    def test_subset_random_sampler(self):
        indices = [2, 5, 7, 9]
        sampler = SubsetRandomSampler(indices)
        result = list(sampler)
        assert len(result) == 4
        assert set(result) == set(indices)

    def test_batch_sampler(self):
        dataset = SimpleDataset(10)
        sampler = SequentialSampler(dataset)
        batch_sampler = BatchSampler(sampler, batch_size=3, drop_last=False)
        batches = list(batch_sampler)
        assert len(batches) == 4
        assert batches[0] == [0, 1, 2]
        assert batches[-1] == [9]

    def test_batch_sampler_drop_last(self):
        dataset = SimpleDataset(10)
        sampler = SequentialSampler(dataset)
        batch_sampler = BatchSampler(sampler, batch_size=3, drop_last=True)
        batches = list(batch_sampler)
        assert len(batches) == 3

    def test_weighted_random_sampler(self):
        weights = [1.0, 2.0, 3.0, 4.0]
        sampler = WeightedRandomSampler(weights, num_samples=100, replacement=True)
        indices = list(sampler)
        assert len(indices) == 100
        # Higher weighted indices should appear more
        counts = [indices.count(i) for i in range(4)]
        # Index 3 (weight 4) should generally have more samples than index 0 (weight 1)
        assert counts[3] > counts[0] or counts[3] == counts[0]

    def test_distributed_sampler(self):
        dataset = SimpleDataset(100)
        sampler = DistributedSampler(dataset, num_replicas=4, rank=0)
        indices = list(sampler)
        assert len(indices) == 25  # 100 / 4


# =============================================================================
# DataLoader Tests
# =============================================================================

class TestDataLoader:
    """Test cases for DataLoader."""

    def test_dataloader_basic(self):
        dataset = SimpleDataset(10)
        loader = DataLoader(dataset, batch_size=3)
        batches = list(loader)
        assert len(batches) == 4

    def test_dataloader_drop_last(self):
        dataset = SimpleDataset(10)
        loader = DataLoader(dataset, batch_size=3, drop_last=True)
        batches = list(loader)
        assert len(batches) == 3

    def test_dataloader_shuffle(self):
        dataset = SimpleDataset(10)
        loader = DataLoader(dataset, batch_size=10, shuffle=True)
        batch1 = list(loader)[0]
        batch2 = list(loader)[0]
        # Note: batches might be the same by chance, but unlikely

    def test_dataloader_len(self):
        dataset = SimpleDataset(100)
        loader = DataLoader(dataset, batch_size=32)
        assert len(loader) == 4  # ceil(100/32)

    def test_dataloader_with_custom_sampler(self):
        dataset = SimpleDataset(100)
        sampler = SubsetRandomSampler(range(10))
        loader = DataLoader(dataset, batch_size=5, sampler=sampler)
        batches = list(loader)
        assert len(batches) == 2


# =============================================================================
# Collate Tests
# =============================================================================

class TestCollate:
    """Test cases for collate functions."""

    def test_collate_lists(self):
        batch = [[1, 2, 3], [4, 5, 6]]
        result = default_collate(batch)
        assert len(result) == 3

    def test_collate_dicts(self):
        batch = [
            {"a": 1, "b": 2},
            {"a": 3, "b": 4},
        ]
        result = default_collate(batch)
        assert "a" in result
        assert "b" in result

    def test_collate_tuples(self):
        batch = [(1, 2), (3, 4)]
        result = default_collate(batch)
        assert isinstance(result, tuple)
        assert len(result) == 2


# =============================================================================
# Transform Tests
# =============================================================================

class TestTransforms:
    """Test cases for data transforms."""

    def test_compose(self):
        transform = Compose([
            Lambda(lambda x: x * 2),
            Lambda(lambda x: x + 1),
        ])
        result = transform(5)
        assert result == 11

    def test_lambda(self):
        transform = Lambda(lambda x: x ** 2)
        assert transform(5) == 25

    def test_normalize(self):
        try:
            import numpy as np
            data = np.array([[1.0, 2.0], [3.0, 4.0]])
            transform = Normalize(mean=[0.0], std=[1.0])
            # Basic smoke test
            result = transform(data)
        except ImportError:
            pytest.skip("NumPy not available")

    def test_random_horizontal_flip(self):
        try:
            import numpy as np
            data = np.array([[1, 2, 3], [4, 5, 6]])
            transform = RandomHorizontalFlip(p=1.0)  # Always flip
            result = transform(data)
            # Check that it's flipped
        except ImportError:
            pytest.skip("NumPy not available")

    def test_random_apply(self):
        transform = RandomApply([Lambda(lambda x: x * 2)], p=0.0)
        assert transform(5) == 5  # Never applied

        transform = RandomApply([Lambda(lambda x: x * 2)], p=1.0)
        assert transform(5) == 10  # Always applied


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

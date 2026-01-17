"""
Samplers for PyFlame DataLoader.

Provides various sampling strategies for data loading.
"""

import random
from abc import ABC, abstractmethod
from typing import Iterator, List, Optional, Sized


class Sampler(ABC):
    """
    Base class for all samplers.

    Every sampler subclass must provide an __iter__ method that yields
    indices of samples and a __len__ method that returns the length.
    """

    @abstractmethod
    def __iter__(self) -> Iterator[int]:
        raise NotImplementedError

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError


class SequentialSampler(Sampler):
    """
    Samples elements sequentially, always in the same order.

    Args:
        data_source: Dataset to sample from.

    Example:
        >>> dataset = TensorDataset(pf.randn(100, 10))
        >>> sampler = SequentialSampler(dataset)
        >>> list(sampler)  # [0, 1, 2, ..., 99]
    """

    def __init__(self, data_source: Sized):
        self.data_source = data_source

    def __iter__(self) -> Iterator[int]:
        return iter(range(len(self.data_source)))

    def __len__(self) -> int:
        return len(self.data_source)


class RandomSampler(Sampler):
    """
    Samples elements randomly without replacement.

    Args:
        data_source: Dataset to sample from.
        replacement: If True, samples with replacement.
        num_samples: Number of samples to draw (default: len(data_source)).
        generator: Random generator for reproducibility.

    Example:
        >>> dataset = TensorDataset(pf.randn(100, 10))
        >>> sampler = RandomSampler(dataset)
        >>> indices = list(sampler)  # Shuffled [0-99]
    """

    def __init__(
        self,
        data_source: Sized,
        replacement: bool = False,
        num_samples: Optional[int] = None,
        generator: Optional[random.Random] = None,
        seed: Optional[int] = None,
    ):
        self.data_source = data_source
        self.replacement = replacement
        self._num_samples = num_samples
        self._seed = seed if seed is not None else random.randint(0, 2**31 - 1)
        self.generator = generator or random.Random(self._seed)
        self._epoch = 0

    @property
    def num_samples(self) -> int:
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples

    def __iter__(self) -> Iterator[int]:
        n = len(self.data_source)

        if self.replacement:
            # Sample with replacement
            for _ in range(self.num_samples):
                yield self.generator.randint(0, n - 1)
        else:
            # Shuffle indices
            indices = list(range(n))
            self.generator.shuffle(indices)
            yield from indices[: self.num_samples]

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int):
        """Set epoch for reproducible shuffling in distributed training."""
        self._epoch = epoch
        # Combine original seed with epoch for reproducible but varied shuffling
        self.generator.seed(self._seed + self._epoch)


class SubsetRandomSampler(Sampler):
    """
    Samples elements randomly from a given list of indices, without replacement.

    Args:
        indices: Sequence of indices to sample from.
        generator: Random generator for reproducibility.

    Example:
        >>> indices = [10, 20, 30, 40, 50]
        >>> sampler = SubsetRandomSampler(indices)
    """

    def __init__(
        self,
        indices: List[int],
        generator: Optional[random.Random] = None,
    ):
        self.indices = list(indices)
        self.generator = generator or random.Random()

    def __iter__(self) -> Iterator[int]:
        shuffled = self.indices.copy()
        self.generator.shuffle(shuffled)
        return iter(shuffled)

    def __len__(self) -> int:
        return len(self.indices)


class WeightedRandomSampler(Sampler):
    """
    Samples elements with given probabilities (weights).

    Args:
        weights: Sequence of weights (not necessarily summing to 1).
        num_samples: Number of samples to draw.
        replacement: If True, samples are drawn with replacement.
        generator: Random generator for reproducibility.

    Example:
        >>> # Sample more from class 0 (weight 2) than class 1 (weight 1)
        >>> weights = [2.0 if label == 0 else 1.0 for label in labels]
        >>> sampler = WeightedRandomSampler(weights, num_samples=100)
    """

    def __init__(
        self,
        weights: List[float],
        num_samples: int,
        replacement: bool = True,
        generator: Optional[random.Random] = None,
    ):
        if not replacement and num_samples > len(weights):
            raise ValueError(
                "num_samples must be <= len(weights) when sampling without replacement"
            )

        self.weights = list(weights)
        self.num_samples = num_samples
        self.replacement = replacement
        self.generator = generator or random.Random()

    def __iter__(self) -> Iterator[int]:
        # Normalize weights
        total = sum(self.weights)
        probs = [w / total for w in self.weights]

        if self.replacement:
            # Sample with replacement using weights
            indices = range(len(self.weights))
            for _ in range(self.num_samples):
                yield self.generator.choices(indices, weights=probs)[0]
        else:
            # Sample without replacement using set for O(1) membership check
            indices = list(range(len(self.weights)))
            remaining_probs = probs.copy()
            selected_set = set()  # Use set for O(1) lookup instead of list

            for _ in range(self.num_samples):
                # Renormalize remaining probabilities
                total_remaining = sum(
                    remaining_probs[i] for i in indices if i not in selected_set
                )
                if total_remaining <= 0:
                    break

                # Select one
                r = self.generator.random() * total_remaining
                cumsum = 0
                for i in indices:
                    if i in selected_set:
                        continue
                    cumsum += remaining_probs[i]
                    if cumsum >= r:
                        selected_set.add(i)
                        yield i
                        break

    def __len__(self) -> int:
        return self.num_samples


class BatchSampler(Sampler):
    """
    Wraps another sampler to yield batches of indices.

    Args:
        sampler: Base sampler.
        batch_size: Size of mini-batch.
        drop_last: If True, drop the last incomplete batch.

    Example:
        >>> sampler = SequentialSampler(dataset)
        >>> batch_sampler = BatchSampler(sampler, batch_size=32, drop_last=False)
        >>> for batch_indices in batch_sampler:
        ...     batch = [dataset[i] for i in batch_indices]
    """

    def __init__(
        self,
        sampler: Sampler,
        batch_size: int,
        drop_last: bool = False,
    ):
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self) -> Iterator[List[int]]:
        batch = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []

        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size


class DistributedSampler(Sampler):
    """
    Sampler for distributed training.

    Restricts data loading to a subset of the dataset exclusive to each process.

    Args:
        dataset: Dataset to sample from.
        num_replicas: Number of processes in distributed training.
        rank: Rank of the current process.
        shuffle: Whether to shuffle the indices.
        seed: Random seed for shuffling.
        drop_last: If True, drop samples to make evenly divisible.

    Example:
        >>> sampler = DistributedSampler(dataset, num_replicas=4, rank=0)
        >>> loader = DataLoader(dataset, sampler=sampler)
    """

    def __init__(
        self,
        dataset: Sized,
        num_replicas: int = 1,
        rank: int = 0,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ):
        if rank >= num_replicas or rank < 0:
            raise ValueError(f"Invalid rank {rank}, must be in [0, {num_replicas})")

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0

        # Calculate number of samples per replica
        total_size = len(dataset)
        if drop_last:
            self.num_samples = total_size // num_replicas
            self.total_size = self.num_samples * num_replicas
        else:
            self.num_samples = (total_size + num_replicas - 1) // num_replicas
            self.total_size = self.num_samples * num_replicas

    def __iter__(self) -> Iterator[int]:
        if self.shuffle:
            # Deterministic shuffling based on epoch
            g = random.Random(self.seed + self.epoch)
            indices = list(range(len(self.dataset)))
            g.shuffle(indices)
        else:
            indices = list(range(len(self.dataset)))

        # Pad to make evenly divisible
        if len(indices) < self.total_size:
            padding = self.total_size - len(indices)
            indices = indices + indices[:padding]

        # Subsample for this rank
        indices = indices[self.rank : self.total_size : self.num_replicas]

        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int):
        """Set epoch for reproducible shuffling."""
        self.epoch = epoch

"""
DataLoader for PyFlame.

Provides batching, shuffling, and parallel data loading.
"""

from typing import Any, Callable, Iterator, List, Optional, Sequence, Union
import random
from collections.abc import Mapping

from .dataset import Dataset, IterableDataset
from .sampler import (
    Sampler,
    RandomSampler,
    SequentialSampler,
    BatchSampler,
)


def default_collate(batch: List[Any]) -> Any:
    """
    Default collate function that stacks samples into a batch.

    Handles:
    - Tensors: Stack along new dimension
    - Numbers: Convert to tensor
    - Strings: Keep as list
    - Dicts: Recursively collate values
    - Tuples/Lists: Recursively collate elements

    Args:
        batch: List of samples from dataset.

    Returns:
        Batched data structure.
    """
    if len(batch) == 0:
        raise ValueError("Cannot collate empty batch")

    elem = batch[0]

    # Handle tensors (PyFlame tensors have a stack method)
    if hasattr(elem, "stack") or hasattr(elem, "__array__"):
        # Assume tensor-like, try to stack
        try:
            # For PyFlame tensors
            if hasattr(elem, "stack"):
                return elem.__class__.stack(batch, dim=0)
            # For numpy arrays
            import numpy as np
            return np.stack(batch, axis=0)
        except Exception:
            return batch

    # Handle numeric types
    elif isinstance(elem, (int, float)):
        # Convert to tensor
        try:
            import numpy as np
            return np.array(batch)
        except ImportError:
            return batch

    # Handle strings
    elif isinstance(elem, str):
        return batch

    # Handle mappings (dict-like)
    elif isinstance(elem, Mapping):
        return {key: default_collate([d[key] for d in batch]) for key in elem}

    # Handle tuples (preserve type)
    elif isinstance(elem, tuple) and hasattr(elem, "_fields"):  # namedtuple
        return type(elem)(*(default_collate(samples) for samples in zip(*batch)))

    elif isinstance(elem, tuple):
        return tuple(default_collate(samples) for samples in zip(*batch))

    # Handle sequences (lists)
    elif isinstance(elem, (list, Sequence)):
        return [default_collate(samples) for samples in zip(*batch)]

    # Fallback: return as-is
    return batch


class DataLoader:
    """
    Data loader for iterating over a dataset in batches.

    Combines a dataset and a sampler, and provides iteration over the dataset.

    Args:
        dataset: Dataset to load from.
        batch_size: Number of samples per batch.
        shuffle: Whether to shuffle data at each epoch.
        sampler: Custom sampler for drawing samples.
        batch_sampler: Custom batch sampler.
        num_workers: Number of worker processes (0 = main process).
        collate_fn: Function to merge samples into batch.
        pin_memory: Copy tensors to pinned memory.
        drop_last: Drop incomplete last batch if dataset size not divisible.
        timeout: Timeout for workers.
        prefetch_factor: Number of batches prefetched per worker.
        generator: Random number generator for shuffling.

    Example:
        >>> dataset = TensorDataset(features, labels)
        >>> loader = DataLoader(dataset, batch_size=32, shuffle=True)
        >>> for batch_x, batch_y in loader:
        ...     output = model(batch_x)
        ...     loss = criterion(output, batch_y)
    """

    def __init__(
        self,
        dataset: Union[Dataset, IterableDataset],
        batch_size: int = 1,
        shuffle: bool = False,
        sampler: Optional[Sampler] = None,
        batch_sampler: Optional[BatchSampler] = None,
        num_workers: int = 0,
        collate_fn: Optional[Callable] = None,
        pin_memory: bool = False,
        drop_last: bool = False,
        timeout: float = 0,
        prefetch_factor: int = 2,
        generator: Optional[random.Random] = None,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.timeout = timeout
        self.prefetch_factor = prefetch_factor
        self._generator = generator or random.Random()

        # Set collate function
        self.collate_fn = collate_fn or default_collate

        # Handle sampler configuration
        if batch_sampler is not None:
            if batch_size != 1 or shuffle or sampler is not None or drop_last:
                raise ValueError(
                    "batch_sampler is mutually exclusive with batch_size, "
                    "shuffle, sampler, and drop_last"
                )
            self.batch_sampler = batch_sampler
            self.sampler = None
        else:
            if sampler is not None:
                self.sampler = sampler
            elif shuffle:
                self.sampler = RandomSampler(dataset, generator=self._generator)
            else:
                self.sampler = SequentialSampler(dataset)

            self.batch_sampler = BatchSampler(
                self.sampler,
                batch_size,
                drop_last
            )

    def __iter__(self) -> Iterator:
        """Iterate over batches."""
        if isinstance(self.dataset, IterableDataset):
            return self._iter_iterable_dataset()
        return self._iter_map_dataset()

    def _iter_map_dataset(self) -> Iterator:
        """Iterate over a map-style dataset."""
        for batch_indices in self.batch_sampler:
            batch = [self.dataset[i] for i in batch_indices]
            yield self.collate_fn(batch)

    def _iter_iterable_dataset(self) -> Iterator:
        """Iterate over an iterable dataset."""
        batch = []
        for sample in self.dataset:
            batch.append(sample)
            if len(batch) == self.batch_size:
                yield self.collate_fn(batch)
                batch = []

        # Handle remaining samples
        if len(batch) > 0 and not self.drop_last:
            yield self.collate_fn(batch)

    def __len__(self) -> int:
        """Return number of batches."""
        if isinstance(self.dataset, IterableDataset):
            raise TypeError("IterableDataset has no len()")

        if self.drop_last:
            return len(self.dataset) // self.batch_size
        else:
            return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    @property
    def generator(self) -> random.Random:
        """Get the random generator."""
        return self._generator

    def set_epoch(self, epoch: int):
        """
        Set the epoch for shuffling.

        In distributed training, this ensures different shuffling per epoch.

        Args:
            epoch: Current epoch number.
        """
        if hasattr(self.sampler, "set_epoch"):
            self.sampler.set_epoch(epoch)


class DataLoaderIterator:
    """
    Iterator for DataLoader.

    Handles single-process and multi-process data loading.
    """

    def __init__(self, loader: DataLoader):
        self.loader = loader
        self.batch_sampler_iter = iter(loader.batch_sampler)
        self._num_yielded = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self._num_yielded >= len(self.loader):
            raise StopIteration

        batch_indices = next(self.batch_sampler_iter)
        batch = [self.loader.dataset[i] for i in batch_indices]
        self._num_yielded += 1
        return self.loader.collate_fn(batch)

    def __len__(self):
        return len(self.loader)

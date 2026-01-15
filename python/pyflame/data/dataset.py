"""
Dataset classes for PyFlame.

Provides base classes and utilities for creating datasets.
"""

import random
from abc import ABC, abstractmethod
from typing import Any, Callable, Iterator, List, Optional, Sequence, Tuple, Union


class Dataset(ABC):
    """
    Abstract base class for datasets.

    All datasets should subclass this and implement __getitem__ and __len__.

    Example:
        class MyDataset(Dataset):
            def __init__(self, data):
                self.data = data

            def __getitem__(self, index):
                return self.data[index]

            def __len__(self):
                return len(self.data)
    """

    @abstractmethod
    def __getitem__(self, index: int) -> Any:
        """Get item at index."""
        raise NotImplementedError

    @abstractmethod
    def __len__(self) -> int:
        """Return the total number of samples."""
        raise NotImplementedError

    def __add__(self, other: "Dataset") -> "ConcatDataset":
        """Concatenate datasets using + operator."""
        return ConcatDataset([self, other])


class TensorDataset(Dataset):
    """
    Dataset wrapping tensors.

    Each sample is retrieved by indexing tensors along the first dimension.

    Args:
        *tensors: Tensors with the same size in the first dimension.

    Example:
        >>> features = pf.randn(100, 10)
        >>> labels = pf.randint(0, 5, (100,))
        >>> dataset = TensorDataset(features, labels)
        >>> x, y = dataset[0]
    """

    def __init__(self, *tensors):
        if len(tensors) == 0:
            raise ValueError("At least one tensor required")

        # Check all tensors have same first dimension
        first_dim = len(tensors[0])
        for i, t in enumerate(tensors):
            if len(t) != first_dim:
                raise ValueError(
                    f"All tensors must have same length in first dimension. "
                    f"Tensor 0 has length {first_dim}, tensor {i} has length {len(t)}"
                )

        self.tensors = tensors

    def __getitem__(self, index: int) -> Tuple:
        return tuple(t[index] for t in self.tensors)

    def __len__(self) -> int:
        return len(self.tensors[0])


class IterableDataset(ABC):
    """
    An iterable dataset.

    Subclasses should implement __iter__ to return an iterator over samples.
    This is useful for streaming data that doesn't fit in memory.

    Example:
        class MyIterableDataset(IterableDataset):
            def __init__(self, file_path):
                self.file_path = file_path

            def __iter__(self):
                with open(self.file_path) as f:
                    for line in f:
                        yield process(line)
    """

    @abstractmethod
    def __iter__(self) -> Iterator:
        raise NotImplementedError


class Subset(Dataset):
    """
    Subset of a dataset at specified indices.

    Args:
        dataset: The original dataset.
        indices: Indices to select.

    Example:
        >>> dataset = TensorDataset(pf.randn(100, 10))
        >>> train_subset = Subset(dataset, range(80))
        >>> val_subset = Subset(dataset, range(80, 100))
    """

    def __init__(self, dataset: Dataset, indices: Sequence[int]):
        self.dataset = dataset
        self.indices = list(indices)

    def __getitem__(self, index: int) -> Any:
        if index < 0 or index >= len(self.indices):
            raise IndexError(
                f"Index {index} out of range for subset of size {len(self.indices)}"
            )
        return self.dataset[self.indices[index]]

    def __len__(self) -> int:
        return len(self.indices)


class ConcatDataset(Dataset):
    """
    Concatenation of multiple datasets.

    Args:
        datasets: List of datasets to concatenate.

    Example:
        >>> dataset1 = TensorDataset(pf.randn(50, 10))
        >>> dataset2 = TensorDataset(pf.randn(30, 10))
        >>> combined = ConcatDataset([dataset1, dataset2])
        >>> len(combined)  # 80
    """

    def __init__(self, datasets: List[Dataset]):
        if len(datasets) == 0:
            raise ValueError("datasets list cannot be empty")

        self.datasets = datasets
        self.cumulative_sizes = []

        cumsum = 0
        for d in datasets:
            cumsum += len(d)
            self.cumulative_sizes.append(cumsum)

    def __getitem__(self, index: int) -> Any:
        if index < 0 or index >= len(self):
            raise IndexError(f"Index {index} out of range")

        # Find which dataset contains this index
        dataset_idx = 0
        for i, size in enumerate(self.cumulative_sizes):
            if index < size:
                dataset_idx = i
                break

        # Compute index within that dataset
        if dataset_idx == 0:
            sample_idx = index
        else:
            sample_idx = index - self.cumulative_sizes[dataset_idx - 1]

        return self.datasets[dataset_idx][sample_idx]

    def __len__(self) -> int:
        return self.cumulative_sizes[-1] if self.cumulative_sizes else 0


def random_split(
    dataset: Dataset,
    lengths: List[Union[int, float]],
    generator: Optional[random.Random] = None,
) -> List[Subset]:
    """
    Randomly split a dataset into non-overlapping subsets.

    Args:
        dataset: Dataset to split.
        lengths: Lengths or fractions of splits. If floats, must sum to 1.0.
        generator: Random generator for reproducibility.

    Returns:
        List of Subset objects.

    Example:
        >>> dataset = TensorDataset(pf.randn(100, 10))
        >>> train, val, test = random_split(dataset, [0.8, 0.1, 0.1])
        >>> len(train), len(val), len(test)  # (80, 10, 10)
    """
    if generator is None:
        generator = random.Random()

    total_length = len(dataset)

    # Convert fractions to lengths
    if all(isinstance(frac, float) for frac in lengths):
        if abs(sum(lengths) - 1.0) > 1e-6:
            raise ValueError("Fractions must sum to 1.0")
        int_lengths = []
        remaining = total_length
        for _, frac in enumerate(lengths[:-1]):
            length = int(frac * total_length)
            int_lengths.append(length)
            remaining -= length
        int_lengths.append(remaining)
        lengths = int_lengths

    if sum(lengths) != total_length:
        raise ValueError(
            f"Sum of lengths ({sum(lengths)}) must equal dataset length ({total_length})"
        )

    # Shuffle indices
    indices = list(range(total_length))
    generator.shuffle(indices)

    # Create subsets
    subsets = []
    offset = 0
    for length in lengths:
        subset_indices = indices[offset : offset + length]
        subsets.append(Subset(dataset, subset_indices))
        offset += length

    return subsets


class ChainDataset(IterableDataset):
    """
    Chain multiple IterableDatasets together.

    Args:
        datasets: Iterable datasets to chain.

    Example:
        >>> ds1 = MyIterableDataset("file1.txt")
        >>> ds2 = MyIterableDataset("file2.txt")
        >>> combined = ChainDataset([ds1, ds2])
    """

    def __init__(self, datasets: List[IterableDataset]):
        self.datasets = datasets

    def __iter__(self) -> Iterator:
        for dataset in self.datasets:
            yield from dataset


class MapDataset(Dataset):
    """
    Apply a transform function to dataset samples.

    Args:
        dataset: Source dataset.
        transform: Function to apply to each sample.

    Example:
        >>> dataset = TensorDataset(pf.randn(100, 10))
        >>> normalized = MapDataset(dataset, lambda x: (x - x.mean()) / x.std())
    """

    def __init__(self, dataset: Dataset, transform: Callable[[Any], Any]):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index: int) -> Any:
        return self.transform(self.dataset[index])

    def __len__(self) -> int:
        return len(self.dataset)

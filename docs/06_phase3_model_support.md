# Phase 3: Model Support - Architecture & Implementation Plan

**PyFlame Version:** Pre-Release Alpha 1.0
**Phase:** 3 of 4
**Focus:** Model Support
**Status:** Implementation Complete

> **Implementation Status:** Phase 3 implementation is complete as of January 2026. All core components have been built including data loading infrastructure, model serialization, pre-built models (ResNet, Transformer/BERT), training utilities, metrics, and model hub/registry. See [Implementation Notes](#implementation-notes) for details.

---

## Table of Contents

1. [Phase 3 Overview](#1-phase-3-overview)
2. [Data Loading Infrastructure](#2-data-loading-infrastructure)
3. [Model Serialization](#3-model-serialization)
4. [Pre-built Model Architectures](#4-pre-built-model-architectures)
5. [Training Utilities](#5-training-utilities)
6. [Evaluation & Metrics](#6-evaluation--metrics)
7. [Model Registry & Pretrained Weights](#7-model-registry--pretrained-weights)
8. [CSL Backend Extensions](#8-csl-backend-extensions)
9. [Implementation Roadmap](#9-implementation-roadmap)
10. [Technical Decisions](#10-technical-decisions)
11. [Implementation Notes](#implementation-notes) - **NEW: Implementation complete**

---

## 1. Phase 3 Overview

### 1.1 Goals

Phase 3 builds upon Phase 2's ML primitives to provide complete model support infrastructure:

| Component | Description |
|-----------|-------------|
| **Data Loading** | Dataset abstraction, DataLoader, batch processing, transforms |
| **Serialization** | Save/load models to disk, checkpoint management |
| **Model Zoo** | Pre-built architectures (ResNet, Transformer, etc.) |
| **Training Utils** | Trainer class, callbacks, logging, early stopping |
| **Evaluation** | Metrics, evaluation loops, inference utilities |
| **Pretrained** | Model registry, weight downloading, fine-tuning support |

### 1.2 Dependencies on Phase 2

Phase 3 requires these Phase 2 components to be complete:

- [x] Automatic differentiation (autograd)
- [x] Neural network layers (Linear, Conv2d, Attention, Normalization)
- [x] Loss functions (CrossEntropy, MSE, etc.)
- [x] Optimizers (SGD, Adam, AdamW) with LR schedulers
- [x] Module system with state_dict() support
- [x] Python bindings for all ML primitives

### 1.3 Key Design Principle: PyTorch-Compatible API

To maximize adoption and ease migration, Phase 3 APIs will closely mirror PyTorch conventions while maintaining PyFlame's Cerebras-native optimizations.

---

## 2. Data Loading Infrastructure

### 2.1 Overview

The data loading system enables efficient batch processing for training and evaluation. For Cerebras WSE, data loading must account for the static graph compilation model.

### 2.2 Dataset Abstraction

#### 2.2.1 Base Dataset Class (C++)

```cpp
// include/pyflame/data/dataset.hpp

namespace pyflame::data {

/// Abstract base class for datasets
class Dataset {
public:
    virtual ~Dataset() = default;

    /// Get a single item by index
    /// Returns: tuple of (data, label) tensors
    virtual std::pair<Tensor, Tensor> get_item(int64_t index) const = 0;

    /// Total number of samples
    virtual int64_t size() const = 0;

    /// Alias for size()
    int64_t __len__() const { return size(); }
};

/// Dataset that wraps tensors (in-memory data)
class TensorDataset : public Dataset {
public:
    TensorDataset(Tensor data, Tensor labels);
    TensorDataset(std::vector<Tensor> tensors);

    std::pair<Tensor, Tensor> get_item(int64_t index) const override;
    int64_t size() const override;

private:
    std::vector<Tensor> tensors_;
};

/// Dataset that loads from disk lazily
class LazyDataset : public Dataset {
public:
    using LoadFn = std::function<std::pair<Tensor, Tensor>(int64_t)>;

    LazyDataset(int64_t size, LoadFn load_fn);

    std::pair<Tensor, Tensor> get_item(int64_t index) const override;
    int64_t size() const override { return size_; }

private:
    int64_t size_;
    LoadFn load_fn_;
};

/// Subset of a dataset
class Subset : public Dataset {
public:
    Subset(std::shared_ptr<Dataset> dataset, std::vector<int64_t> indices);

    std::pair<Tensor, Tensor> get_item(int64_t index) const override;
    int64_t size() const override { return indices_.size(); }

private:
    std::shared_ptr<Dataset> dataset_;
    std::vector<int64_t> indices_;
};

/// Concatenate multiple datasets
class ConcatDataset : public Dataset {
public:
    ConcatDataset(std::vector<std::shared_ptr<Dataset>> datasets);

    std::pair<Tensor, Tensor> get_item(int64_t index) const override;
    int64_t size() const override;

private:
    std::vector<std::shared_ptr<Dataset>> datasets_;
    std::vector<int64_t> cumulative_sizes_;
};

}  // namespace pyflame::data
```

#### 2.2.2 Python Dataset Interface

```python
# python/pyflame/data/dataset.py

from abc import ABC, abstractmethod
from typing import Tuple, Any, List, Optional
import pyflame as pf

class Dataset(ABC):
    """Abstract base class for all datasets."""

    @abstractmethod
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """Get item by index. Returns (data, label) tuple."""
        pass

    @abstractmethod
    def __len__(self) -> int:
        """Return total number of samples."""
        pass


class TensorDataset(Dataset):
    """Dataset wrapping tensors."""

    def __init__(self, *tensors: pf.Tensor):
        """
        Args:
            *tensors: Tensors with same first dimension (batch size)
        """
        assert all(t.shape[0] == tensors[0].shape[0] for t in tensors)
        self.tensors = tensors

    def __getitem__(self, index: int) -> Tuple[pf.Tensor, ...]:
        return tuple(t[index] for t in self.tensors)

    def __len__(self) -> int:
        return self.tensors[0].shape[0]


class MapDataset(Dataset):
    """Dataset that applies a transform to another dataset."""

    def __init__(self, dataset: Dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index: int):
        item = self.dataset[index]
        return self.transform(item)

    def __len__(self) -> int:
        return len(self.dataset)
```

### 2.3 DataLoader

#### 2.3.1 Core DataLoader Design

The DataLoader handles batching, shuffling, and parallel data loading. For Cerebras, batch sizes are typically fixed at compile time.

```cpp
// include/pyflame/data/dataloader.hpp

namespace pyflame::data {

struct DataLoaderOptions {
    int64_t batch_size = 1;
    bool shuffle = false;
    bool drop_last = false;          // Drop incomplete final batch
    int num_workers = 0;             // 0 = main thread loading
    bool pin_memory = false;         // For CPU->device transfer optimization
    uint64_t seed = 0;               // Random seed for shuffling (0 = random)
    std::optional<int64_t> prefetch_factor = std::nullopt;  // Batches to prefetch
};

class DataLoader {
public:
    DataLoader(std::shared_ptr<Dataset> dataset, DataLoaderOptions options = {});

    /// Iterator interface
    class Iterator {
    public:
        using value_type = std::pair<Tensor, Tensor>;  // (batch_data, batch_labels)

        Iterator(DataLoader* loader, int64_t batch_idx);

        value_type operator*() const;
        Iterator& operator++();
        bool operator!=(const Iterator& other) const;

    private:
        DataLoader* loader_;
        int64_t batch_idx_;
    };

    Iterator begin();
    Iterator end();

    /// Number of batches
    int64_t num_batches() const;

    /// Fetch a specific batch
    std::pair<Tensor, Tensor> get_batch(int64_t batch_idx) const;

    /// Reset epoch (reshuffle if enabled)
    void reset_epoch();

    /// Get current batch indices (for reproducibility)
    const std::vector<int64_t>& get_indices() const { return indices_; }

private:
    std::shared_ptr<Dataset> dataset_;
    DataLoaderOptions options_;
    std::vector<int64_t> indices_;     // Shuffled indices
    mutable std::mt19937 rng_;

    void shuffle_indices();
    std::pair<Tensor, Tensor> collate_batch(int64_t start_idx, int64_t end_idx) const;
};

}  // namespace pyflame::data
```

#### 2.3.2 Python DataLoader

```python
# python/pyflame/data/dataloader.py

from typing import Optional, Iterator, Tuple
import pyflame as pf
from .dataset import Dataset

class DataLoader:
    """Combines dataset with sampler and provides batched iteration."""

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 1,
        shuffle: bool = False,
        drop_last: bool = False,
        num_workers: int = 0,
        collate_fn = None,
        pin_memory: bool = False,
        seed: Optional[int] = None,
    ):
        """
        Args:
            dataset: Dataset to load from
            batch_size: Number of samples per batch
            shuffle: Whether to shuffle indices each epoch
            drop_last: Drop last incomplete batch if dataset size not divisible
            num_workers: Number of worker processes for loading (0 = main process)
            collate_fn: Function to merge samples into a batch
            pin_memory: Copy tensors to pinned memory for faster device transfer
            seed: Random seed for shuffling reproducibility
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.num_workers = num_workers
        self.collate_fn = collate_fn or self._default_collate
        self.pin_memory = pin_memory
        self.seed = seed

        self._indices = list(range(len(dataset)))
        self._epoch = 0

    def __iter__(self) -> Iterator[Tuple[pf.Tensor, ...]]:
        """Iterate over batches."""
        if self.shuffle:
            self._shuffle_indices()

        batch = []
        for idx in self._indices:
            batch.append(self.dataset[idx])
            if len(batch) == self.batch_size:
                yield self.collate_fn(batch)
                batch = []

        if batch and not self.drop_last:
            yield self.collate_fn(batch)

        self._epoch += 1

    def __len__(self) -> int:
        """Number of batches per epoch."""
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    def _shuffle_indices(self):
        """Shuffle dataset indices."""
        import random
        if self.seed is not None:
            random.seed(self.seed + self._epoch)
        random.shuffle(self._indices)

    @staticmethod
    def _default_collate(batch):
        """Stack samples into batch tensors."""
        # batch is list of (data, label) tuples
        data = pf.stack([item[0] for item in batch], dim=0)
        labels = pf.stack([item[1] for item in batch], dim=0)
        return data, labels
```

### 2.4 Data Transforms

```python
# python/pyflame/data/transforms.py

import pyflame as pf
from typing import Callable, List, Tuple

class Compose:
    """Compose multiple transforms."""

    def __init__(self, transforms: List[Callable]):
        self.transforms = transforms

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x


class ToTensor:
    """Convert numpy array or PIL Image to Tensor."""

    def __call__(self, x):
        import numpy as np
        if isinstance(x, np.ndarray):
            return pf.from_numpy(x.astype(np.float32))
        # Handle PIL Image
        try:
            import PIL.Image
            if isinstance(x, PIL.Image.Image):
                return pf.from_numpy(np.array(x).astype(np.float32))
        except ImportError:
            pass
        return x


class Normalize:
    """Normalize tensor with mean and std."""

    def __init__(self, mean: Tuple[float, ...], std: Tuple[float, ...]):
        self.mean = mean
        self.std = std

    def __call__(self, x: pf.Tensor) -> pf.Tensor:
        # x shape: [C, H, W] or [H, W]
        mean = pf.tensor(self.mean).reshape(-1, 1, 1)
        std = pf.tensor(self.std).reshape(-1, 1, 1)
        return (x - mean) / std


class RandomHorizontalFlip:
    """Randomly flip tensor horizontally."""

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, x: pf.Tensor) -> pf.Tensor:
        import random
        if random.random() < self.p:
            return x.flip(-1)  # Flip last dimension
        return x


class RandomCrop:
    """Randomly crop tensor to given size."""

    def __init__(self, size: Tuple[int, int], padding: int = 0):
        self.size = size
        self.padding = padding

    def __call__(self, x: pf.Tensor) -> pf.Tensor:
        import random
        h, w = x.shape[-2:]
        new_h, new_w = self.size

        if self.padding > 0:
            x = pf.pad(x, [self.padding] * 4)
            h += 2 * self.padding
            w += 2 * self.padding

        top = random.randint(0, h - new_h)
        left = random.randint(0, w - new_w)

        return x[..., top:top+new_h, left:left+new_w]


class Resize:
    """Resize tensor to given size."""

    def __init__(self, size: Tuple[int, int], mode: str = 'bilinear'):
        self.size = size
        self.mode = mode

    def __call__(self, x: pf.Tensor) -> pf.Tensor:
        return pf.nn.functional.interpolate(
            x.unsqueeze(0),
            size=self.size,
            mode=self.mode
        ).squeeze(0)
```

### 2.5 Built-in Datasets

```python
# python/pyflame/data/datasets/__init__.py

from .mnist import MNIST, FashionMNIST
from .cifar import CIFAR10, CIFAR100
from .imagenet import ImageNet
from .text import TextDataset, LanguageModelingDataset


# python/pyflame/data/datasets/mnist.py

import os
import gzip
import struct
import numpy as np
from ..dataset import Dataset
from ..transforms import Compose, ToTensor, Normalize

class MNIST(Dataset):
    """MNIST handwritten digits dataset."""

    URLS = {
        'train_images': 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
        'train_labels': 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
        'test_images': 'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
        'test_labels': 'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz',
    }

    def __init__(
        self,
        root: str,
        train: bool = True,
        download: bool = False,
        transform = None,
        target_transform = None,
    ):
        """
        Args:
            root: Root directory for dataset files
            train: If True, load training set; else test set
            download: If True, download dataset if not present
            transform: Transform to apply to images
            target_transform: Transform to apply to labels
        """
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        if download:
            self._download()

        self.data, self.targets = self._load_data()

    def __getitem__(self, index: int):
        img = self.data[index]
        target = self.targets[index]

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)

    def _download(self):
        """Download dataset if not present."""
        os.makedirs(self.root, exist_ok=True)
        # Implementation: download and extract files
        pass

    def _load_data(self):
        """Load data from disk."""
        prefix = 'train' if self.train else 't10k'
        # Load images and labels from IDX files
        pass
```

### 2.6 Cerebras-Specific Data Considerations

For Cerebras WSE compilation, data shapes must be known at compile time:

```python
# python/pyflame/data/cerebras_utils.py

class StaticShapeDataLoader(DataLoader):
    """DataLoader that guarantees fixed batch sizes for Cerebras compilation.

    Unlike standard DataLoader, this always returns exactly batch_size samples,
    padding the last batch if necessary.
    """

    def __init__(self, dataset, batch_size, pad_value=0.0, **kwargs):
        kwargs['drop_last'] = False  # Never drop, we'll pad instead
        super().__init__(dataset, batch_size, **kwargs)
        self.pad_value = pad_value

    def _default_collate(self, batch):
        """Pad batch to fixed size if needed."""
        actual_size = len(batch)
        if actual_size < self.batch_size:
            # Pad with zeros
            pad_count = self.batch_size - actual_size
            sample = batch[0]
            padding = [(pf.full_like(sample[i], self.pad_value) for i in range(len(sample)))
                       for _ in range(pad_count)]
            batch.extend(padding)

        return super()._default_collate(batch)


def prepare_for_cerebras(dataset, batch_size, mesh_layout=None):
    """Prepare dataset for Cerebras WSE execution.

    Returns a DataLoader with fixed batch sizes and optional mesh layout hints.
    """
    loader = StaticShapeDataLoader(dataset, batch_size)

    if mesh_layout is not None:
        # Add metadata for CSL code generation
        loader.mesh_layout = mesh_layout

    return loader
```

---

## 3. Model Serialization

### 3.1 Overview

Model serialization enables saving and loading model weights, optimizer states, and training checkpoints. PyFlame will support multiple formats for interoperability.

### 3.2 File Format Strategy

| Format | Use Case | Pros | Cons |
|--------|----------|------|------|
| **PyFlame Native (.pf)** | Default format | Fast, full fidelity | PyFlame-only |
| **SafeTensors (.safetensors)** | Interop with HuggingFace | Safe, fast, cross-platform | Limited metadata |
| **NumPy (.npz)** | Simple export | Universal | No optimizer state |
| **ONNX (.onnx)** | Inference export | Wide deployment support | Complex, inference only |

### 3.3 Core Serialization API (C++)

```cpp
// include/pyflame/utils/serialize.hpp

namespace pyflame::utils {

/// Serialization format enum
enum class SerializeFormat {
    PYFLAME_NATIVE,  // .pf
    SAFETENSORS,     // .safetensors
    NUMPY,           // .npz
    ONNX             // .onnx (export only)
};

/// Options for serialization
struct SerializeOptions {
    SerializeFormat format = SerializeFormat::PYFLAME_NATIVE;
    bool compress = true;           // Enable compression
    int compression_level = 6;      // 1-9 for zlib
    bool include_metadata = true;   // Include version info, creation time, etc.
};

/// Save state dict to file
void save(
    const std::map<std::string, Tensor>& state_dict,
    const std::string& path,
    SerializeOptions options = {}
);

/// Load state dict from file
std::map<std::string, Tensor> load(
    const std::string& path,
    SerializeOptions options = {}
);

/// Save entire training checkpoint
struct Checkpoint {
    std::map<std::string, Tensor> model_state;
    std::map<std::string, Tensor> optimizer_state;
    int64_t epoch;
    int64_t global_step;
    float best_metric;
    std::map<std::string, std::string> metadata;
};

void save_checkpoint(const Checkpoint& checkpoint, const std::string& path);
Checkpoint load_checkpoint(const std::string& path);

/// ONNX export for deployment
void export_onnx(
    const nn::Module& model,
    const std::vector<Tensor>& example_inputs,
    const std::string& path,
    int opset_version = 13
);

}  // namespace pyflame::utils
```

### 3.4 PyFlame Native Format Specification

```
PyFlame Model File (.pf) Format Specification
=============================================

Header (64 bytes):
  - Magic number: "PYFLAME\0" (8 bytes)
  - Version: uint32 (4 bytes)
  - Flags: uint32 (4 bytes)
    - Bit 0: Compressed
    - Bit 1: Contains optimizer state
    - Bit 2: Contains metadata
  - Num tensors: uint64 (8 bytes)
  - Metadata offset: uint64 (8 bytes)
  - Data offset: uint64 (8 bytes)
  - Reserved: (24 bytes)

Tensor Index (variable):
  For each tensor:
    - Name length: uint32 (4 bytes)
    - Name: char[name_length]
    - DType: uint8 (1 byte)
    - Ndim: uint8 (1 byte)
    - Shape: int64[ndim]
    - Data offset: uint64 (8 bytes)
    - Data size: uint64 (8 bytes)

Data Section:
  - Raw tensor data (potentially compressed)

Metadata Section (JSON):
  {
    "pyflame_version": "0.1.0",
    "created_at": "2026-01-11T12:00:00Z",
    "model_class": "ResNet18",
    "training_info": { ... }
  }
```

### 3.5 Python Serialization API

```python
# python/pyflame/utils/serialization.py

import pyflame as pf
from typing import Dict, Any, Optional, Union
from pathlib import Path

def save(
    obj: Union[Dict[str, pf.Tensor], 'pf.nn.Module'],
    path: Union[str, Path],
    format: str = 'auto',
    **kwargs
):
    """Save model or state dict to file.

    Args:
        obj: Model or state dict to save
        path: Destination file path
        format: 'auto', 'pyflame', 'safetensors', 'numpy'
        **kwargs: Additional format-specific options
    """
    path = Path(path)

    # Extract state dict if Module
    if hasattr(obj, 'state_dict'):
        state_dict = obj.state_dict()
    else:
        state_dict = obj

    # Auto-detect format from extension
    if format == 'auto':
        format = _detect_format(path)

    # Save using appropriate backend
    _save_dispatch[format](state_dict, path, **kwargs)


def load(
    path: Union[str, Path],
    map_location: Optional[str] = None,
    **kwargs
) -> Dict[str, pf.Tensor]:
    """Load state dict from file.

    Args:
        path: Source file path
        map_location: Device/layout to map tensors to
        **kwargs: Additional format-specific options

    Returns:
        State dictionary mapping names to tensors
    """
    path = Path(path)
    format = _detect_format(path)
    state_dict = _load_dispatch[format](path, **kwargs)

    if map_location:
        state_dict = {k: v.to(map_location) for k, v in state_dict.items()}

    return state_dict


class Checkpoint:
    """Training checkpoint with model, optimizer, and training state."""

    def __init__(
        self,
        model_state: Dict[str, pf.Tensor],
        optimizer_state: Optional[Dict[str, pf.Tensor]] = None,
        epoch: int = 0,
        global_step: int = 0,
        best_metric: float = float('inf'),
        **metadata
    ):
        self.model_state = model_state
        self.optimizer_state = optimizer_state or {}
        self.epoch = epoch
        self.global_step = global_step
        self.best_metric = best_metric
        self.metadata = metadata

    def save(self, path: Union[str, Path]):
        """Save checkpoint to file."""
        pf.utils.save_checkpoint(self, str(path))

    @classmethod
    def load(cls, path: Union[str, Path]) -> 'Checkpoint':
        """Load checkpoint from file."""
        return pf.utils.load_checkpoint(str(path))


class CheckpointManager:
    """Manages multiple checkpoints with automatic cleanup."""

    def __init__(
        self,
        directory: Union[str, Path],
        max_to_keep: int = 5,
        keep_best: bool = True,
    ):
        """
        Args:
            directory: Directory to store checkpoints
            max_to_keep: Maximum number of checkpoints to retain
            keep_best: Always keep the best checkpoint regardless of max_to_keep
        """
        self.directory = Path(directory)
        self.max_to_keep = max_to_keep
        self.keep_best = keep_best
        self._checkpoints = []
        self._best_checkpoint = None
        self._best_metric = float('inf')

        self.directory.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        checkpoint: Checkpoint,
        metric: Optional[float] = None,
        filename: Optional[str] = None
    ) -> Path:
        """Save checkpoint and manage retention."""
        if filename is None:
            filename = f"checkpoint_epoch{checkpoint.epoch}_step{checkpoint.global_step}.pf"

        path = self.directory / filename
        checkpoint.save(path)

        self._checkpoints.append(path)

        # Track best
        if metric is not None and metric < self._best_metric:
            self._best_metric = metric
            self._best_checkpoint = path

        # Cleanup old checkpoints
        self._cleanup()

        return path

    def _cleanup(self):
        """Remove old checkpoints exceeding max_to_keep."""
        while len(self._checkpoints) > self.max_to_keep:
            oldest = self._checkpoints.pop(0)
            if oldest != self._best_checkpoint:
                oldest.unlink(missing_ok=True)
```

---

## 4. Pre-built Model Architectures

### 4.1 Overview

PyFlame will provide pre-built implementations of common model architectures optimized for Cerebras WSE.

### 4.2 Model Architecture Categories

| Category | Models | Priority |
|----------|--------|----------|
| **CNNs** | ResNet, VGG, EfficientNet, MobileNet | High |
| **Transformers** | BERT, GPT-2, ViT, T5 | High |
| **RNNs** | LSTM, GRU | Medium |
| **Detection** | YOLO, SSD | Medium |
| **Segmentation** | U-Net, DeepLab | Low |

### 4.3 Model Implementation Pattern

```cpp
// include/pyflame/models/resnet.hpp

namespace pyflame::models {

/// Basic residual block
class BasicBlock : public nn::Module {
public:
    BasicBlock(int64_t in_planes, int64_t planes, int64_t stride = 1);
    Tensor forward(const Tensor& x) override;

    static constexpr int expansion = 1;

private:
    nn::Conv2d conv1_, conv2_;
    nn::BatchNorm2d bn1_, bn2_;
    nn::Sequential downsample_;
    int64_t stride_;
};

/// Bottleneck block for deeper ResNets
class Bottleneck : public nn::Module {
public:
    Bottleneck(int64_t in_planes, int64_t planes, int64_t stride = 1);
    Tensor forward(const Tensor& x) override;

    static constexpr int expansion = 4;

private:
    nn::Conv2d conv1_, conv2_, conv3_;
    nn::BatchNorm2d bn1_, bn2_, bn3_;
    nn::Sequential downsample_;
    int64_t stride_;
};

/// ResNet model
template<typename Block>
class ResNet : public nn::Module {
public:
    ResNet(
        std::vector<int64_t> layers,
        int64_t num_classes = 1000,
        bool zero_init_residual = false,
        int64_t groups = 1,
        int64_t width_per_group = 64
    );

    Tensor forward(const Tensor& x) override;

    /// Feature extraction (without final FC)
    Tensor forward_features(const Tensor& x);

private:
    nn::Conv2d conv1_;
    nn::BatchNorm2d bn1_;
    nn::Sequential layer1_, layer2_, layer3_, layer4_;
    nn::AdaptiveAvgPool2d avgpool_;
    nn::Linear fc_;

    int64_t in_planes_;
    int64_t groups_;
    int64_t base_width_;

    nn::Sequential make_layer(int64_t planes, int64_t blocks, int64_t stride = 1);
};

/// Convenience factory functions
std::shared_ptr<ResNet<BasicBlock>> resnet18(int64_t num_classes = 1000, bool pretrained = false);
std::shared_ptr<ResNet<BasicBlock>> resnet34(int64_t num_classes = 1000, bool pretrained = false);
std::shared_ptr<ResNet<Bottleneck>> resnet50(int64_t num_classes = 1000, bool pretrained = false);
std::shared_ptr<ResNet<Bottleneck>> resnet101(int64_t num_classes = 1000, bool pretrained = false);
std::shared_ptr<ResNet<Bottleneck>> resnet152(int64_t num_classes = 1000, bool pretrained = false);

}  // namespace pyflame::models
```

### 4.4 Transformer Architecture

```cpp
// include/pyflame/models/transformer.hpp

namespace pyflame::models {

/// Transformer configuration
struct TransformerConfig {
    int64_t vocab_size = 30522;
    int64_t hidden_size = 768;
    int64_t num_hidden_layers = 12;
    int64_t num_attention_heads = 12;
    int64_t intermediate_size = 3072;
    float hidden_dropout_prob = 0.1f;
    float attention_probs_dropout_prob = 0.1f;
    int64_t max_position_embeddings = 512;
    int64_t type_vocab_size = 2;
    float layer_norm_eps = 1e-12f;
    int64_t pad_token_id = 0;

    static TransformerConfig from_pretrained(const std::string& model_name);
    void to_json(const std::string& path) const;
    static TransformerConfig from_json(const std::string& path);
};

/// Transformer encoder layer
class TransformerEncoderLayer : public nn::Module {
public:
    TransformerEncoderLayer(const TransformerConfig& config);
    Tensor forward(const Tensor& hidden_states, const Tensor& attention_mask = {}) override;

private:
    nn::MultiheadAttention self_attention_;
    nn::LayerNorm norm1_, norm2_;
    nn::Linear fc1_, fc2_;
    nn::Dropout dropout_;
};

/// Full transformer encoder
class TransformerEncoder : public nn::Module {
public:
    TransformerEncoder(const TransformerConfig& config);
    Tensor forward(const Tensor& hidden_states, const Tensor& attention_mask = {}) override;

private:
    nn::ModuleList layers_;
    nn::LayerNorm final_norm_;
};

/// BERT-style model
class BertModel : public nn::Module {
public:
    BertModel(const TransformerConfig& config);

    /// Forward pass
    /// Returns: (last_hidden_state, pooler_output)
    std::pair<Tensor, Tensor> forward(
        const Tensor& input_ids,
        const Tensor& attention_mask = {},
        const Tensor& token_type_ids = {}
    );

    static std::shared_ptr<BertModel> from_pretrained(const std::string& model_name);

private:
    nn::Embedding word_embeddings_;
    nn::Embedding position_embeddings_;
    nn::Embedding token_type_embeddings_;
    nn::LayerNorm embed_norm_;
    nn::Dropout embed_dropout_;
    TransformerEncoder encoder_;
    nn::Linear pooler_;

    TransformerConfig config_;
};

/// GPT-2 style decoder
class GPT2Model : public nn::Module {
public:
    GPT2Model(const TransformerConfig& config);

    Tensor forward(
        const Tensor& input_ids,
        const Tensor& attention_mask = {},
        const Tensor& past_key_values = {}  // For incremental decoding
    );

    /// Generate text autoregressively
    Tensor generate(
        const Tensor& input_ids,
        int64_t max_length,
        float temperature = 1.0f,
        int64_t top_k = 50,
        float top_p = 0.95f
    );

    static std::shared_ptr<GPT2Model> from_pretrained(const std::string& model_name);

private:
    nn::Embedding wte_;  // Token embeddings
    nn::Embedding wpe_;  // Position embeddings
    nn::ModuleList layers_;
    nn::LayerNorm ln_f_;

    TransformerConfig config_;
};

/// Vision Transformer (ViT)
class ViTModel : public nn::Module {
public:
    ViTModel(
        int64_t image_size = 224,
        int64_t patch_size = 16,
        int64_t num_classes = 1000,
        int64_t hidden_size = 768,
        int64_t num_layers = 12,
        int64_t num_heads = 12,
        float dropout = 0.0f
    );

    Tensor forward(const Tensor& pixel_values) override;

    static std::shared_ptr<ViTModel> from_pretrained(const std::string& model_name);

private:
    nn::Conv2d patch_embed_;  // Linear projection via conv
    nn::Parameter cls_token_;
    nn::Parameter pos_embed_;
    nn::Dropout dropout_;
    TransformerEncoder encoder_;
    nn::LayerNorm norm_;
    nn::Linear head_;

    int64_t num_patches_;
};

}  // namespace pyflame::models
```

### 4.5 Python Model API

```python
# python/pyflame/models/__init__.py

from .resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from .vgg import vgg11, vgg13, vgg16, vgg19
from .bert import BertModel, BertConfig, BertForSequenceClassification
from .gpt2 import GPT2Model, GPT2Config, GPT2LMHeadModel
from .vit import ViTModel, ViTConfig, ViTForImageClassification


# python/pyflame/models/resnet.py

import pyflame as pf
import pyflame.nn as nn
from typing import Optional, List, Type

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes * self.expansion,
                         kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion)
            )

    def forward(self, x):
        out = pf.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = pf.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block: Type[BasicBlock],
        num_blocks: List[int],
        num_classes: int = 1000
    ):
        super().__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = pf.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.shape[0], -1)
        out = self.fc(out)
        return out


def resnet18(num_classes: int = 1000, pretrained: bool = False) -> ResNet:
    """Constructs a ResNet-18 model."""
    model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes)
    if pretrained:
        state_dict = pf.hub.load_state_dict('pyflame/resnet18')
        model.load_state_dict(state_dict)
    return model
```

---

## 5. Training Utilities

### 5.1 Trainer Class

```python
# python/pyflame/training/trainer.py

import pyflame as pf
import pyflame.nn as nn
import pyflame.optim as optim
from typing import Optional, Dict, Any, Callable, List
from dataclasses import dataclass, field

@dataclass
class TrainingArguments:
    """Arguments for training configuration."""

    # Output
    output_dir: str = "./output"
    overwrite_output_dir: bool = False

    # Training hyperparameters
    num_epochs: int = 3
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    learning_rate: float = 5e-5
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0

    # Optimizer
    optimizer: str = "adamw"
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8

    # LR Scheduler
    lr_scheduler_type: str = "linear"
    warmup_ratio: float = 0.0
    warmup_steps: int = 0

    # Logging
    logging_dir: Optional[str] = None
    logging_steps: int = 500
    logging_first_step: bool = False

    # Checkpointing
    save_strategy: str = "epoch"  # "no", "epoch", "steps"
    save_steps: int = 500
    save_total_limit: Optional[int] = None
    load_best_model_at_end: bool = False

    # Evaluation
    evaluation_strategy: str = "no"  # "no", "epoch", "steps"
    eval_steps: int = 500
    metric_for_best_model: str = "loss"
    greater_is_better: bool = False

    # Early stopping
    early_stopping_patience: Optional[int] = None
    early_stopping_threshold: float = 0.0

    # Cerebras-specific
    mesh_layout: Optional[pf.MeshLayout] = None
    compile_only: bool = False


class Trainer:
    """High-level training API."""

    def __init__(
        self,
        model: nn.Module,
        args: TrainingArguments,
        train_dataset = None,
        eval_dataset = None,
        compute_metrics: Optional[Callable] = None,
        callbacks: Optional[List['TrainerCallback']] = None,
        optimizers: tuple = (None, None),  # (optimizer, scheduler)
    ):
        self.model = model
        self.args = args
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.compute_metrics = compute_metrics
        self.callbacks = callbacks or []
        self.optimizer, self.lr_scheduler = optimizers

        self.state = TrainerState()
        self._setup_training()

    def _setup_training(self):
        """Initialize optimizer, scheduler, and other training components."""
        if self.optimizer is None:
            self.optimizer = self._create_optimizer()
        if self.lr_scheduler is None:
            self.lr_scheduler = self._create_scheduler()

    def _create_optimizer(self):
        """Create optimizer based on args."""
        params = self.model.parameters()

        if self.args.optimizer == "adamw":
            return optim.AdamW(
                params,
                lr=self.args.learning_rate,
                betas=(self.args.adam_beta1, self.args.adam_beta2),
                eps=self.args.adam_epsilon,
                weight_decay=self.args.weight_decay,
            )
        elif self.args.optimizer == "sgd":
            return optim.SGD(
                params,
                lr=self.args.learning_rate,
                weight_decay=self.args.weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.args.optimizer}")

    def train(self, resume_from_checkpoint: Optional[str] = None):
        """Main training loop."""
        # Resume from checkpoint if specified
        if resume_from_checkpoint:
            self._load_checkpoint(resume_from_checkpoint)

        # Create data loaders
        train_dataloader = self._get_train_dataloader()

        # Calculate total steps
        total_steps = len(train_dataloader) * self.args.num_epochs
        self.state.max_steps = total_steps

        # Callback: on_train_begin
        self._call_callbacks("on_train_begin")

        for epoch in range(self.state.epoch, self.args.num_epochs):
            self.state.epoch = epoch
            self._call_callbacks("on_epoch_begin")

            self.model.train()
            epoch_loss = 0.0

            for step, batch in enumerate(train_dataloader):
                self.state.global_step += 1

                # Forward pass
                loss = self._training_step(batch)
                epoch_loss += loss.item()

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()

                # Gradient clipping
                if self.args.max_grad_norm > 0:
                    pf.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.args.max_grad_norm
                    )

                # Optimizer step
                self.optimizer.step()
                self.lr_scheduler.step()

                # Logging
                if self.state.global_step % self.args.logging_steps == 0:
                    self._log({"loss": loss.item(), "lr": self.lr_scheduler.get_last_lr()})

                # Evaluation
                if self._should_evaluate():
                    metrics = self.evaluate()
                    self._call_callbacks("on_evaluate", metrics)

                # Checkpointing
                if self._should_save():
                    self._save_checkpoint()

                self._call_callbacks("on_step_end")

            # End of epoch
            self._call_callbacks("on_epoch_end")

            # Epoch-level evaluation
            if self.args.evaluation_strategy == "epoch":
                metrics = self.evaluate()
                self._call_callbacks("on_evaluate", metrics)

        self._call_callbacks("on_train_end")
        return self.state

    def evaluate(self, eval_dataset=None):
        """Evaluate the model."""
        eval_dataset = eval_dataset or self.eval_dataset
        if eval_dataset is None:
            return {}

        self.model.eval()
        eval_dataloader = self._get_eval_dataloader(eval_dataset)

        total_loss = 0.0
        all_preds = []
        all_labels = []

        with pf.no_grad():
            for batch in eval_dataloader:
                loss, logits, labels = self._evaluation_step(batch)
                total_loss += loss.item()
                all_preds.append(logits)
                all_labels.append(labels)

        metrics = {"eval_loss": total_loss / len(eval_dataloader)}

        if self.compute_metrics is not None:
            preds = pf.cat(all_preds, dim=0)
            labels = pf.cat(all_labels, dim=0)
            metrics.update(self.compute_metrics(preds, labels))

        return metrics

    def _training_step(self, batch):
        """Single training step. Override for custom behavior."""
        inputs, labels = batch
        outputs = self.model(inputs)
        loss = self.loss_fn(outputs, labels)
        return loss

    def _evaluation_step(self, batch):
        """Single evaluation step."""
        inputs, labels = batch
        outputs = self.model(inputs)
        loss = self.loss_fn(outputs, labels)
        return loss, outputs, labels


@dataclass
class TrainerState:
    """Tracks training state."""
    epoch: int = 0
    global_step: int = 0
    max_steps: int = 0
    best_metric: float = float('inf')
    best_model_checkpoint: Optional[str] = None


class TrainerCallback:
    """Base class for trainer callbacks."""

    def on_train_begin(self, trainer: Trainer, **kwargs):
        pass

    def on_train_end(self, trainer: Trainer, **kwargs):
        pass

    def on_epoch_begin(self, trainer: Trainer, **kwargs):
        pass

    def on_epoch_end(self, trainer: Trainer, **kwargs):
        pass

    def on_step_end(self, trainer: Trainer, **kwargs):
        pass

    def on_evaluate(self, trainer: Trainer, metrics: Dict[str, float], **kwargs):
        pass


class EarlyStoppingCallback(TrainerCallback):
    """Stop training when metric stops improving."""

    def __init__(self, patience: int = 3, threshold: float = 0.0):
        self.patience = patience
        self.threshold = threshold
        self.best_metric = float('inf')
        self.patience_counter = 0

    def on_evaluate(self, trainer: Trainer, metrics: Dict[str, float], **kwargs):
        metric = metrics.get(trainer.args.metric_for_best_model, metrics.get("eval_loss"))
        if metric < self.best_metric - self.threshold:
            self.best_metric = metric
            self.patience_counter = 0
        else:
            self.patience_counter += 1
            if self.patience_counter >= self.patience:
                trainer.state.should_training_stop = True
```

### 5.2 Gradient Utilities

```python
# python/pyflame/nn/utils.py

import pyflame as pf
from typing import Iterable, Optional

def clip_grad_norm_(
    parameters: Iterable[pf.Tensor],
    max_norm: float,
    norm_type: float = 2.0,
    error_if_nonfinite: bool = False
) -> pf.Tensor:
    """Clips gradient norm of parameters.

    Args:
        parameters: Iterable of Tensors with gradients
        max_norm: Max norm of gradients
        norm_type: Type of norm to use (default: 2.0 for L2)
        error_if_nonfinite: Raise error if gradient norm is nan/inf

    Returns:
        Total norm of gradients
    """
    parameters = list(parameters)
    grads = [p.grad for p in parameters if p.grad is not None]

    if len(grads) == 0:
        return pf.tensor(0.0)

    # Compute total norm
    if norm_type == float('inf'):
        total_norm = max(g.abs().max() for g in grads)
    else:
        total_norm = pf.stack([g.norm(norm_type) for g in grads]).norm(norm_type)

    if error_if_nonfinite and (total_norm.isnan() or total_norm.isinf()):
        raise RuntimeError(f"Gradient norm is {total_norm}")

    # Clip
    clip_coef = max_norm / (total_norm + 1e-6)
    clip_coef_clamped = pf.clamp(clip_coef, max=1.0)

    for g in grads:
        g.mul_(clip_coef_clamped)

    return total_norm


def clip_grad_value_(parameters: Iterable[pf.Tensor], clip_value: float):
    """Clips gradient values to [-clip_value, clip_value]."""
    for p in parameters:
        if p.grad is not None:
            p.grad.clamp_(-clip_value, clip_value)
```

---

## 6. Evaluation & Metrics

### 6.1 Metrics Module

```python
# python/pyflame/metrics/__init__.py

from .classification import accuracy, precision, recall, f1_score, confusion_matrix
from .regression import mse, mae, r2_score
from .nlp import perplexity, bleu_score
from .aggregator import MetricAggregator


# python/pyflame/metrics/classification.py

import pyflame as pf
from typing import Optional

def accuracy(
    preds: pf.Tensor,
    targets: pf.Tensor,
    top_k: int = 1
) -> pf.Tensor:
    """Compute classification accuracy.

    Args:
        preds: Predictions [N, C] logits or [N] class indices
        targets: Ground truth [N] class indices
        top_k: Consider prediction correct if target is in top-k

    Returns:
        Scalar accuracy tensor
    """
    if preds.ndim == 2:
        # Logits - get top-k predictions
        if top_k == 1:
            pred_classes = preds.argmax(dim=1)
            correct = (pred_classes == targets).float()
        else:
            _, top_k_preds = preds.topk(top_k, dim=1)
            correct = (top_k_preds == targets.unsqueeze(1)).any(dim=1).float()
    else:
        # Already class indices
        correct = (preds == targets).float()

    return correct.mean()


def precision(
    preds: pf.Tensor,
    targets: pf.Tensor,
    num_classes: Optional[int] = None,
    average: str = 'macro'
) -> pf.Tensor:
    """Compute precision score.

    Args:
        preds: Predictions
        targets: Ground truth
        num_classes: Number of classes (inferred if None)
        average: 'micro', 'macro', 'weighted', or 'none'
    """
    if preds.ndim == 2:
        preds = preds.argmax(dim=1)

    if num_classes is None:
        num_classes = max(preds.max().item(), targets.max().item()) + 1

    precisions = []
    supports = []

    for c in range(num_classes):
        pred_positive = (preds == c)
        true_positive = pred_positive & (targets == c)

        tp = true_positive.sum().float()
        pp = pred_positive.sum().float()

        prec = tp / (pp + 1e-10)
        precisions.append(prec)
        supports.append((targets == c).sum().float())

    precisions = pf.stack(precisions)
    supports = pf.stack(supports)

    if average == 'micro':
        return (precisions * supports).sum() / supports.sum()
    elif average == 'macro':
        return precisions.mean()
    elif average == 'weighted':
        return (precisions * supports).sum() / supports.sum()
    else:  # 'none'
        return precisions


def confusion_matrix(
    preds: pf.Tensor,
    targets: pf.Tensor,
    num_classes: Optional[int] = None
) -> pf.Tensor:
    """Compute confusion matrix.

    Args:
        preds: Predictions [N]
        targets: Ground truth [N]
        num_classes: Number of classes

    Returns:
        Confusion matrix [num_classes, num_classes]
    """
    if preds.ndim == 2:
        preds = preds.argmax(dim=1)

    if num_classes is None:
        num_classes = max(preds.max().item(), targets.max().item()) + 1

    cm = pf.zeros([num_classes, num_classes], dtype=pf.int64)

    for t, p in zip(targets.tolist(), preds.tolist()):
        cm[t, p] += 1

    return cm


# python/pyflame/metrics/aggregator.py

class MetricAggregator:
    """Aggregate metrics across batches."""

    def __init__(self):
        self.metrics = {}
        self.counts = {}

    def update(self, metrics_dict: dict, count: int = 1):
        """Add batch metrics."""
        for key, value in metrics_dict.items():
            if key not in self.metrics:
                self.metrics[key] = 0.0
                self.counts[key] = 0

            if isinstance(value, pf.Tensor):
                value = value.item()

            self.metrics[key] += value * count
            self.counts[key] += count

    def compute(self) -> dict:
        """Compute averaged metrics."""
        return {
            key: self.metrics[key] / self.counts[key]
            for key in self.metrics
        }

    def reset(self):
        """Reset all metrics."""
        self.metrics.clear()
        self.counts.clear()
```

---

## 7. Model Registry & Pretrained Weights

### 7.1 Hub Architecture

```python
# python/pyflame/hub/__init__.py

from .registry import ModelRegistry, register_model
from .download import download_file, load_state_dict
from .cache import get_cache_dir, clear_cache

# Global registry instance
_registry = ModelRegistry()


def list_models(filter: str = None) -> list:
    """List available models."""
    return _registry.list_models(filter)


def load(model_name: str, **kwargs):
    """Load a model from the hub."""
    return _registry.load(model_name, **kwargs)


def load_state_dict(model_name: str, map_location=None):
    """Load pretrained weights."""
    url = _registry.get_weights_url(model_name)
    return download_and_load(url, map_location)


# python/pyflame/hub/registry.py

from typing import Dict, Callable, Optional, List
import re

class ModelRegistry:
    """Registry for models and pretrained weights."""

    def __init__(self):
        self._models: Dict[str, Callable] = {}
        self._weights: Dict[str, str] = {}
        self._configs: Dict[str, dict] = {}

    def register(
        self,
        name: str,
        model_fn: Callable,
        weights_url: Optional[str] = None,
        config: Optional[dict] = None
    ):
        """Register a model."""
        self._models[name] = model_fn
        if weights_url:
            self._weights[name] = weights_url
        if config:
            self._configs[name] = config

    def list_models(self, filter: str = None) -> List[str]:
        """List registered models, optionally filtered by pattern."""
        models = list(self._models.keys())
        if filter:
            pattern = re.compile(filter, re.IGNORECASE)
            models = [m for m in models if pattern.search(m)]
        return sorted(models)

    def load(self, name: str, pretrained: bool = True, **kwargs):
        """Load a model by name."""
        if name not in self._models:
            raise ValueError(f"Unknown model: {name}. Available: {self.list_models()}")

        model = self._models[name](**kwargs)

        if pretrained and name in self._weights:
            state_dict = self._load_weights(name)
            model.load_state_dict(state_dict)

        return model

    def get_weights_url(self, name: str) -> str:
        """Get URL for pretrained weights."""
        if name not in self._weights:
            raise ValueError(f"No pretrained weights for: {name}")
        return self._weights[name]


def register_model(
    name: str,
    weights_url: Optional[str] = None,
    config: Optional[dict] = None
):
    """Decorator to register a model."""
    def decorator(fn):
        _registry.register(name, fn, weights_url, config)
        return fn
    return decorator


# python/pyflame/hub/download.py

import os
import hashlib
import urllib.request
from pathlib import Path
from typing import Optional
import pyflame as pf

CACHE_DIR = Path.home() / ".cache" / "pyflame" / "hub"


def get_cache_dir() -> Path:
    """Get the cache directory."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR


def download_file(
    url: str,
    filename: Optional[str] = None,
    cache_dir: Optional[Path] = None,
    force: bool = False,
    progress: bool = True
) -> Path:
    """Download a file with caching.

    Args:
        url: URL to download from
        filename: Local filename (derived from URL if None)
        cache_dir: Directory to cache files
        force: Re-download even if cached
        progress: Show download progress

    Returns:
        Path to downloaded file
    """
    cache_dir = cache_dir or get_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)

    if filename is None:
        filename = url.split("/")[-1]

    local_path = cache_dir / filename

    if local_path.exists() and not force:
        return local_path

    # Download with progress
    print(f"Downloading {url}")
    with urllib.request.urlopen(url) as response:
        total_size = int(response.headers.get('content-length', 0))

        with open(local_path, 'wb') as f:
            downloaded = 0
            chunk_size = 8192

            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)

                if progress and total_size > 0:
                    percent = (downloaded / total_size) * 100
                    print(f"\rProgress: {percent:.1f}%", end="", flush=True)

    print()  # Newline after progress
    return local_path


def load_state_dict_from_url(
    url: str,
    map_location = None,
    progress: bool = True,
    **kwargs
):
    """Download and load state dict from URL."""
    local_path = download_file(url, progress=progress)
    return pf.load(local_path, map_location=map_location)
```

### 7.2 Pretrained Weights Registry

```python
# python/pyflame/hub/pretrained.py

"""Registry of pretrained model weights."""

PRETRAINED_WEIGHTS = {
    # Vision Models
    "resnet18": {
        "url": "https://pyflame-hub.storage.googleapis.com/models/resnet18.pf",
        "sha256": "abc123...",
        "num_classes": 1000,
        "input_size": (224, 224),
    },
    "resnet50": {
        "url": "https://pyflame-hub.storage.googleapis.com/models/resnet50.pf",
        "sha256": "def456...",
        "num_classes": 1000,
        "input_size": (224, 224),
    },
    "vit-base-patch16-224": {
        "url": "https://pyflame-hub.storage.googleapis.com/models/vit_base_patch16_224.pf",
        "sha256": "ghi789...",
        "num_classes": 1000,
        "input_size": (224, 224),
    },

    # Language Models
    "bert-base-uncased": {
        "url": "https://pyflame-hub.storage.googleapis.com/models/bert_base_uncased.pf",
        "sha256": "jkl012...",
        "vocab_size": 30522,
        "hidden_size": 768,
    },
    "gpt2": {
        "url": "https://pyflame-hub.storage.googleapis.com/models/gpt2.pf",
        "sha256": "mno345...",
        "vocab_size": 50257,
        "hidden_size": 768,
    },
}


def get_pretrained_config(model_name: str) -> dict:
    """Get configuration for a pretrained model."""
    if model_name not in PRETRAINED_WEIGHTS:
        raise ValueError(f"Unknown pretrained model: {model_name}")
    return PRETRAINED_WEIGHTS[model_name]
```

---

## 8. CSL Backend Extensions

### 8.1 Model-Level Optimizations

For full model compilation, the CSL backend needs additional optimizations:

```cpp
// include/pyflame/backend/model_compiler.hpp

namespace pyflame::backend {

/// Compile an entire model for Cerebras WSE
class ModelCompiler {
public:
    struct CompileOptions {
        MeshLayout default_layout = MeshLayout::Grid(16, 16);
        bool fuse_operations = true;
        bool optimize_memory = true;
        bool enable_checkpointing = false;  // Gradient checkpointing
        int max_pe_memory_mb = 48;          // Per-PE memory limit
    };

    /// Compile model to CSL
    CSLProgram compile(
        const nn::Module& model,
        const std::vector<ir::TensorSpec>& input_specs,
        CompileOptions options = {}
    );

    /// Estimate memory usage
    MemoryEstimate estimate_memory(
        const nn::Module& model,
        const std::vector<ir::TensorSpec>& input_specs
    );

private:
    /// Optimize computation graph for WSE
    ir::Graph optimize_for_wse(const ir::Graph& graph, const CompileOptions& options);

    /// Compute optimal PE placement
    PEPlacement compute_placement(const ir::Graph& graph, const MeshLayout& layout);

    /// Generate wavelet routing
    WaveletRouting compute_routing(const ir::Graph& graph, const PEPlacement& placement);
};

/// Memory estimation result
struct MemoryEstimate {
    size_t total_params_bytes;
    size_t total_activations_bytes;
    size_t total_gradients_bytes;
    size_t peak_memory_bytes;
    bool fits_in_sram;
    std::map<std::string, size_t> per_layer_memory;
};

}  // namespace pyflame::backend
```

### 8.2 New CSL Templates

```cpp
// Templates for common model patterns

// Attention mechanism optimized for WSE
const char* CSL_ATTENTION_TEMPLATE = R"(
// Multi-head attention kernel
// Distributes heads across PE rows, sequence across PE columns

const HEAD_DIM = %HEAD_DIM%;
const NUM_HEADS = %NUM_HEADS%;
const SEQ_LEN = %SEQ_LEN%;

task compute_attention(pe_row: u16, pe_col: u16) void {
    // Each PE handles one attention head for a sequence segment
    const head_idx = pe_row;
    const seq_start = pe_col * CHUNK_SIZE;

    // Load Q, K, V for this head/segment
    var Q: [CHUNK_SIZE][HEAD_DIM]f32 = load_query(head_idx, seq_start);
    var K: [CHUNK_SIZE][HEAD_DIM]f32 = load_key(head_idx, seq_start);
    var V: [CHUNK_SIZE][HEAD_DIM]f32 = load_value(head_idx, seq_start);

    // Compute attention scores: Q @ K^T
    var scores: [CHUNK_SIZE][CHUNK_SIZE]f32;
    @builtin.matmul(Q, K, scores, .transpose_b = true);

    // Scale by sqrt(d_k)
    const scale = 1.0 / @sqrt(@as(f32, HEAD_DIM));
    @builtin.scale(scores, scale);

    // Softmax (with wavelet communication for global normalization)
    softmax_with_reduce(scores, pe_col);

    // Apply attention: scores @ V
    var output: [CHUNK_SIZE][HEAD_DIM]f32;
    @builtin.matmul(scores, V, output);

    // Send output via wavelet to aggregator
    @wavelet.send(output_color, output);
}
)";

// Residual connection pattern
const char* CSL_RESIDUAL_TEMPLATE = R"(
// Fused residual add + layer norm
task residual_layernorm(input: []f32, residual: []f32, output: []f32) void {
    // Add residual
    @builtin.add(input, residual, output);

    // Compute mean and variance (with reduction across PEs)
    const mean = compute_mean_with_reduce(output);
    const var = compute_var_with_reduce(output, mean);

    // Normalize
    const eps = 1e-5;
    const inv_std = 1.0 / @sqrt(var + eps);
    @builtin.affine(output, inv_std, -mean * inv_std, output);

    // Apply learned parameters
    @builtin.mul(output, gamma, output);
    @builtin.add(output, beta, output);
}
)";
```

---

## 9. Implementation Roadmap

### 9.1 Milestones

| Milestone | Deliverable | Duration |
|-----------|-------------|----------|
| **M3.1** | Dataset & DataLoader infrastructure | Weeks 1-3 |
| **M3.2** | Data transforms & built-in datasets (MNIST, CIFAR) | Weeks 4-5 |
| **M3.3** | Model serialization (save/load) | Weeks 6-7 |
| **M3.4** | Checkpoint management | Weeks 8-9 |
| **M3.5** | ResNet implementation | Weeks 10-11 |
| **M3.6** | Transformer/BERT implementation | Weeks 12-14 |
| **M3.7** | GPT-2 implementation | Weeks 15-16 |
| **M3.8** | Vision Transformer (ViT) | Weeks 17-18 |
| **M3.9** | Trainer class & callbacks | Weeks 19-21 |
| **M3.10** | Metrics & evaluation utilities | Weeks 22-23 |
| **M3.11** | Model hub & pretrained weights | Weeks 24-26 |

### 9.2 Dependencies

```
                    Data Loading (M3.1-M3.2)
                           │
           ┌───────────────┼───────────────┐
           ▼               ▼               ▼
    Serialization     Model Zoo      Training Utils
      (M3.3-M3.4)    (M3.5-M3.8)      (M3.9-M3.10)
           │               │               │
           └───────────────┼───────────────┘
                           ▼
                    Model Hub (M3.11)
                           │
                           ▼
                  CSL Backend Extensions
```

### 9.3 Testing Strategy

1. **Unit Tests**: Each component tested independently
2. **Integration Tests**: Full training pipelines on MNIST/CIFAR
3. **Model Equivalence**: Compare outputs to PyTorch reference
4. **Performance Benchmarks**: Training throughput, memory usage
5. **Cerebras Simulation**: All models tested on simulator

---

## 10. Technical Decisions

### 10.1 Decisions Made

| Decision | Choice | Rationale |
|----------|--------|-----------|
| File Format | PyFlame Native + SafeTensors | Balance of performance and interop |
| Hub Storage | Cloud storage (GCS/S3) | Scalable, CDN-cacheable |
| Trainer API | HuggingFace-style | Familiar to practitioners |
| Pretrained Source | Convert from PyTorch | Leverage existing ecosystem |
| Data Loading | Python-side with C++ collation | Flexibility + performance |

### 10.2 Open Questions

1. **Distributed Data Loading**: How to handle multi-node data parallelism?
2. **Mixed Precision**: FP16/BF16 training support timeline?
3. **Custom Model Upload**: Allow users to upload to PyFlame Hub?
4. **ONNX Import**: Support importing ONNX models from other frameworks?

### 10.3 Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Large model memory overflow | High | High | Gradient checkpointing, model parallelism |
| Pretrained weight compatibility | Medium | Medium | Extensive conversion testing |
| Training convergence differences | Low | High | Numerical validation against PyTorch |
| Hub infrastructure costs | Medium | Low | Start with essential models only |

---

## Implementation Notes

> **Completed: January 2026**

### Components Implemented

#### C++ Components

| Component | File | Description |
|-----------|------|-------------|
| **ResNet Models** | `include/pyflame/models/resnet.hpp` | BasicBlock, Bottleneck, ResNet, ResNeXt, WideResNet |
| **ResNet Implementation** | `src/models/resnet.cpp` | Full forward pass implementation with factory functions |
| **Transformer Models** | `include/pyflame/models/transformer.hpp` | MultiHeadAttention, TransformerEncoder/Decoder, BERT |
| **Transformer Implementation** | `src/models/transformer.cpp` | Attention mechanisms, positional encoding, BERT embeddings |
| **Python Bindings** | `src/python/bindings_phase3.cpp` | pybind11 bindings for all Phase 3 C++ components |

#### Python Components

| Component | Path | Description |
|-----------|------|-------------|
| **Data Module** | `python/pyflame/data/` | Dataset, DataLoader, Samplers, Transforms |
| **Training Module** | `python/pyflame/training/` | Trainer class, Callbacks (EarlyStopping, ModelCheckpoint, etc.) |
| **Metrics Module** | `python/pyflame/metrics/` | Classification metrics (Accuracy, F1, AUROC), Regression metrics (MSE, R2) |
| **Model Hub** | `python/pyflame/hub/` | Model registry, pretrained weight management, cache utilities |

### Key Implementation Files

```
python/pyflame/
├── data/
│   ├── __init__.py          # Module exports
│   ├── dataset.py           # Dataset, TensorDataset, Subset, ConcatDataset, MapDataset
│   ├── dataloader.py        # DataLoader with batching and shuffling
│   ├── sampler.py           # Sequential, Random, Weighted, Distributed samplers
│   └── transforms.py        # Compose, Normalize, image transforms, augmentations
├── training/
│   ├── __init__.py
│   ├── trainer.py           # Trainer class with TrainerConfig and TrainerState
│   └── callbacks.py         # EarlyStopping, ModelCheckpoint, LearningRateScheduler, etc.
├── metrics/
│   ├── __init__.py
│   ├── base.py              # Metric base class, MetricCollection
│   ├── classification.py    # Accuracy, Precision, Recall, F1, AUROC, ConfusionMatrix
│   └── regression.py        # MSE, RMSE, MAE, R2, MAPE
└── hub/
    ├── __init__.py
    ├── registry.py          # ModelRegistry, @register_model decorator
    └── pretrained.py        # Weight downloading, caching, loading

include/pyflame/models/
├── resnet.hpp               # ResNet architecture definitions
└── transformer.hpp          # Transformer and BERT architecture definitions

src/models/
├── resnet.cpp               # ResNet implementation
└── transformer.cpp          # Transformer implementation

src/python/
└── bindings_phase3.cpp      # Python bindings for Phase 3
```

### Test Coverage

| Test File | Coverage |
|-----------|----------|
| `tests/test_phase3_data.py` | Dataset, DataLoader, Samplers, Transforms |
| `tests/test_phase3_metrics.py` | Classification and regression metrics |
| `tests/test_phase3_training.py` | Trainer, Callbacks with mock components |

### API Compatibility

All implementations follow PyTorch-compatible APIs for ease of migration:

- **Dataset/DataLoader**: Compatible with `torch.utils.data` patterns
- **Transforms**: Compatible with `torchvision.transforms` API
- **Trainer**: Inspired by HuggingFace Transformers Trainer
- **Metrics**: Compatible with torchmetrics patterns

### Future Enhancements

The following are planned for future iterations:

1. **Multi-worker DataLoader**: Parallel data loading with multiple processes
2. **Distributed Training**: Support for multi-node training
3. **Mixed Precision**: FP16/BF16 training support
4. **Additional Models**: VGG, EfficientNet, GPT-2, ViT
5. **Built-in Datasets**: MNIST, CIFAR-10, ImageNet loaders
6. **ONNX Export**: Model export for deployment

---

## Appendix A: Python Package Structure (Phase 3)

```
python/pyflame/
├── __init__.py
├── data/
│   ├── __init__.py
│   ├── dataset.py
│   ├── dataloader.py
│   ├── sampler.py
│   ├── transforms.py
│   └── datasets/
│       ├── __init__.py
│       ├── mnist.py
│       ├── cifar.py
│       ├── imagenet.py
│       └── text.py
├── models/
│   ├── __init__.py
│   ├── resnet.py
│   ├── vgg.py
│   ├── bert.py
│   ├── gpt2.py
│   ├── vit.py
│   └── utils.py
├── training/
│   ├── __init__.py
│   ├── trainer.py
│   ├── callbacks.py
│   └── arguments.py
├── metrics/
│   ├── __init__.py
│   ├── classification.py
│   ├── regression.py
│   └── aggregator.py
├── hub/
│   ├── __init__.py
│   ├── registry.py
│   ├── download.py
│   └── pretrained.py
└── utils/
    ├── __init__.py
    ├── serialization.py
    └── checkpoint.py
```

---

## Appendix B: Example End-to-End Training Script

```python
#!/usr/bin/env python
"""Train ResNet-18 on CIFAR-10 with PyFlame."""

import pyflame as pf
import pyflame.nn as nn
import pyflame.optim as optim
from pyflame.data import DataLoader
from pyflame.data.datasets import CIFAR10
from pyflame.data.transforms import Compose, ToTensor, Normalize, RandomHorizontalFlip
from pyflame.models import resnet18
from pyflame.training import Trainer, TrainingArguments
from pyflame.metrics import accuracy

# Data transforms
train_transform = Compose([
    RandomHorizontalFlip(),
    ToTensor(),
    Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
])

test_transform = Compose([
    ToTensor(),
    Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
])

# Load datasets
train_dataset = CIFAR10(root="./data", train=True, download=True, transform=train_transform)
test_dataset = CIFAR10(root="./data", train=False, download=True, transform=test_transform)

# Create model (10 classes for CIFAR-10)
model = resnet18(num_classes=10, pretrained=False)

# Define metrics computation
def compute_metrics(preds, labels):
    return {
        "accuracy": accuracy(preds, labels).item(),
        "top5_accuracy": accuracy(preds, labels, top_k=5).item(),
    }

# Training arguments
args = TrainingArguments(
    output_dir="./cifar10_resnet18",
    num_epochs=100,
    per_device_train_batch_size=128,
    per_device_eval_batch_size=256,
    learning_rate=0.1,
    weight_decay=5e-4,
    optimizer="sgd",
    lr_scheduler_type="cosine",
    warmup_epochs=5,
    logging_steps=100,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
)

# Create trainer
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

# Train!
trainer.train()

# Save final model
pf.save(model.state_dict(), "./cifar10_resnet18/final_model.pf")

print("Training complete!")
print(f"Best accuracy: {trainer.state.best_metric:.2%}")
```

---

*Document Version: 1.0*
*Last Updated: January 11, 2026*

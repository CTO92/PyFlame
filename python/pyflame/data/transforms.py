"""
Data transforms for PyFlame.

Provides composable transforms for data preprocessing and augmentation.
"""

import random
from abc import ABC, abstractmethod
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union


class Transform(ABC):
    """Base class for all transforms."""

    @abstractmethod
    def __call__(self, data: Any) -> Any:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class Compose(Transform):
    """
    Compose multiple transforms together.

    Args:
        transforms: List of transforms to compose.

    Example:
        >>> transform = Compose([
        ...     Normalize(mean=[0.485], std=[0.229]),
        ...     ToTensor(),
        ... ])
        >>> output = transform(input)
    """

    def __init__(self, transforms: List[Transform]):
        self.transforms = transforms

    def __call__(self, data: Any) -> Any:
        for t in self.transforms:
            data = t(data)
        return data

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += f"\n    {t}"
        format_string += "\n)"
        return format_string


class Lambda(Transform):
    """
    Apply a user-defined function as a transform.

    Args:
        lambd: Lambda/function to apply.

    Example:
        >>> transform = Lambda(lambda x: x * 2)
    """

    def __init__(self, lambd: Callable):
        self.lambd = lambd

    def __call__(self, data: Any) -> Any:
        return self.lambd(data)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


# =============================================================================
# Tensor Transforms
# =============================================================================


class ToTensor(Transform):
    """
    Convert data to PyFlame tensor.

    Handles numpy arrays, lists, and PIL images.
    """

    def __call__(self, data: Any) -> Any:
        # Import here to avoid circular imports
        try:
            import pyflame as pf

            if hasattr(data, "__array__"):
                # Numpy array or similar
                return pf.from_numpy(data)
            elif isinstance(data, (list, tuple)):
                return pf.tensor(data)
            elif isinstance(data, (int, float)):
                # Scalar - create 0-dimensional tensor
                return pf.tensor(data)
            else:
                # Fallback: wrap in list
                return pf.tensor([data])
        except ImportError:
            # Fallback: try numpy
            import numpy as np

            return np.array(data)


class Normalize(Transform):
    """
    Normalize a tensor with mean and standard deviation.

    output = (input - mean) / std

    Args:
        mean: Sequence of means for each channel.
        std: Sequence of standard deviations for each channel.
        inplace: Whether to perform normalization in-place.

    Example:
        >>> # ImageNet normalization
        >>> normalize = Normalize(
        ...     mean=[0.485, 0.456, 0.406],
        ...     std=[0.229, 0.224, 0.225]
        ... )
    """

    def __init__(
        self,
        mean: Sequence[float],
        std: Sequence[float],
        inplace: bool = False,
    ):
        self.mean = list(mean)
        self.std = list(std)
        self.inplace = inplace

    def __call__(self, tensor: Any) -> Any:
        if not self.inplace:
            tensor = tensor.clone() if hasattr(tensor, "clone") else tensor.copy()

        # Normalize each channel
        # Assumes tensor shape is [..., C, H, W] or [..., C]
        for i, (m, s) in enumerate(zip(self.mean, self.std)):
            if hasattr(tensor, "select"):
                # PyFlame tensor
                tensor[..., i, :, :] = (tensor[..., i, :, :] - m) / s
            else:
                # Numpy
                tensor[..., i] = (tensor[..., i] - m) / s

        return tensor

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"


class StandardScaler(Transform):
    """
    Standardize features by removing the mean and scaling to unit variance.

    Computes mean and std from the data itself.

    Args:
        dim: Dimension(s) along which to compute mean/std.
    """

    def __init__(self, dim: Optional[Union[int, Tuple[int, ...]]] = None):
        self.dim = dim
        self.mean_ = None
        self.std_ = None

    def fit(self, data: Any) -> "StandardScaler":
        """Compute mean and std from data."""
        if hasattr(data, "mean"):
            self.mean_ = data.mean(dim=self.dim, keepdim=True)
            self.std_ = data.std(dim=self.dim, keepdim=True)
        else:
            import numpy as np

            self.mean_ = np.mean(data, axis=self.dim, keepdims=True)
            self.std_ = np.std(data, axis=self.dim, keepdims=True)
        return self

    def __call__(self, data: Any) -> Any:
        if self.mean_ is None:
            self.fit(data)
        return (data - self.mean_) / (self.std_ + 1e-8)


class MinMaxScaler(Transform):
    """
    Scale features to a given range (default [0, 1]).

    Args:
        feature_range: Desired range of transformed data.
    """

    def __init__(self, feature_range: Tuple[float, float] = (0.0, 1.0)):
        self.feature_range = feature_range
        self.min_ = None
        self.max_ = None

    def fit(self, data: Any) -> "MinMaxScaler":
        """Compute min and max from data."""
        if hasattr(data, "min"):
            self.min_ = data.min()
            self.max_ = data.max()
        else:
            import numpy as np

            self.min_ = np.min(data)
            self.max_ = np.max(data)
        return self

    def __call__(self, data: Any) -> Any:
        if self.min_ is None:
            self.fit(data)

        # Scale to [0, 1]
        data_scaled = (data - self.min_) / (self.max_ - self.min_ + 1e-8)

        # Scale to feature_range
        min_val, max_val = self.feature_range
        return data_scaled * (max_val - min_val) + min_val


# =============================================================================
# Image Transforms
# =============================================================================


class Resize(Transform):
    """
    Resize image to given size.

    Args:
        size: Desired output size. If int, smaller edge is matched.
        interpolation: Interpolation mode.
    """

    def __init__(
        self,
        size: Union[int, Tuple[int, int]],
        interpolation: str = "bilinear",
    ):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img: Any) -> Any:
        # This would use image processing library
        # Placeholder implementation
        return img

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={self.size})"


class CenterCrop(Transform):
    """
    Crop the center of the image.

    Args:
        size: Desired output size (height, width).
    """

    def __init__(self, size: Union[int, Tuple[int, int]]):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size

    def __call__(self, img: Any) -> Any:
        # Placeholder implementation
        return img

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={self.size})"


class RandomCrop(Transform):
    """
    Crop a random portion of the image.

    Args:
        size: Desired output size (height, width).
        padding: Optional padding on each border.
    """

    def __init__(
        self,
        size: Union[int, Tuple[int, int]],
        padding: Optional[Union[int, Tuple[int, ...]]] = None,
    ):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size
        self.padding = padding

    def __call__(self, img: Any) -> Any:
        # Placeholder implementation
        return img


class RandomHorizontalFlip(Transform):
    """
    Randomly flip image horizontally.

    Args:
        p: Probability of flipping.
    """

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, img: Any) -> Any:
        if random.random() < self.p:
            # Flip horizontally
            if hasattr(img, "flip"):
                return img.flip(-1)
            else:
                return img[..., ::-1]
        return img


class RandomVerticalFlip(Transform):
    """
    Randomly flip image vertically.

    Args:
        p: Probability of flipping.
    """

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, img: Any) -> Any:
        if random.random() < self.p:
            if hasattr(img, "flip"):
                return img.flip(-2)
            else:
                return img[..., ::-1, :]
        return img


class RandomRotation(Transform):
    """
    Rotate image by a random angle.

    Args:
        degrees: Range of degrees for rotation.
    """

    def __init__(self, degrees: Union[float, Tuple[float, float]]):
        if isinstance(degrees, (int, float)):
            self.degrees = (-degrees, degrees)
        else:
            self.degrees = degrees

    def __call__(self, img: Any) -> Any:
        _angle = random.uniform(*self.degrees)  # noqa: F841
        # Placeholder: would apply rotation using _angle
        return img


class ColorJitter(Transform):
    """
    Randomly change brightness, contrast, saturation, and hue.

    Args:
        brightness: How much to jitter brightness.
        contrast: How much to jitter contrast.
        saturation: How much to jitter saturation.
        hue: How much to jitter hue.
    """

    def __init__(
        self,
        brightness: float = 0,
        contrast: float = 0,
        saturation: float = 0,
        hue: float = 0,
    ):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def __call__(self, img: Any) -> Any:
        # Placeholder implementation
        return img


class RandomErasing(Transform):
    """
    Randomly erase a rectangular region in an image.

    Args:
        p: Probability of performing erasing.
        scale: Range of proportion of erased area.
        ratio: Range of aspect ratio of erased area.
        value: Erasing value (0 for black).
    """

    def __init__(
        self,
        p: float = 0.5,
        scale: Tuple[float, float] = (0.02, 0.33),
        ratio: Tuple[float, float] = (0.3, 3.3),
        value: float = 0,
    ):
        self.p = p
        self.scale = scale
        self.ratio = ratio
        self.value = value

    def __call__(self, img: Any) -> Any:
        if random.random() >= self.p:
            return img

        # Placeholder implementation
        return img


# =============================================================================
# Text Transforms
# =============================================================================


class Tokenize(Transform):
    """
    Tokenize text into tokens.

    Args:
        tokenizer: Tokenizer to use (e.g., from transformers library).
        max_length: Maximum sequence length.
        padding: Padding strategy.
        truncation: Whether to truncate.
    """

    def __init__(
        self,
        tokenizer: Any,
        max_length: int = 512,
        padding: str = "max_length",
        truncation: bool = True,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = padding
        self.truncation = truncation

    def __call__(self, text: str) -> dict:
        return self.tokenizer(
            text,
            max_length=self.max_length,
            padding=self.padding,
            truncation=self.truncation,
            return_tensors="np",
        )


class LowerCase(Transform):
    """Convert text to lowercase."""

    def __call__(self, text: str) -> str:
        return text.lower()


class StripWhitespace(Transform):
    """Strip leading and trailing whitespace."""

    def __call__(self, text: str) -> str:
        return text.strip()


# =============================================================================
# Audio Transforms
# =============================================================================


class Spectrogram(Transform):
    """
    Compute spectrogram from waveform.

    Args:
        n_fft: FFT size.
        hop_length: Hop length between frames.
        win_length: Window length.
    """

    def __init__(
        self,
        n_fft: int = 400,
        hop_length: Optional[int] = None,
        win_length: Optional[int] = None,
    ):
        self.n_fft = n_fft
        self.hop_length = hop_length or n_fft // 2
        self.win_length = win_length or n_fft

    def __call__(self, waveform: Any) -> Any:
        # Placeholder: would compute STFT
        return waveform


class MelSpectrogram(Transform):
    """
    Compute mel spectrogram from waveform.

    Args:
        sample_rate: Sample rate of audio.
        n_fft: FFT size.
        n_mels: Number of mel filterbanks.
        hop_length: Hop length between frames.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 400,
        n_mels: int = 80,
        hop_length: Optional[int] = None,
    ):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.hop_length = hop_length or n_fft // 2

    def __call__(self, waveform: Any) -> Any:
        # Placeholder: would compute mel spectrogram
        return waveform


# =============================================================================
# Random Apply
# =============================================================================


class RandomApply(Transform):
    """
    Apply transforms with a given probability.

    Args:
        transforms: List of transforms to apply.
        p: Probability of applying.
    """

    def __init__(self, transforms: List[Transform], p: float = 0.5):
        self.transforms = transforms
        self.p = p

    def __call__(self, data: Any) -> Any:
        if random.random() < self.p:
            for t in self.transforms:
                data = t(data)
        return data


class RandomChoice(Transform):
    """
    Apply one of the transforms randomly.

    Args:
        transforms: List of transforms to choose from.
    """

    def __init__(self, transforms: List[Transform]):
        self.transforms = transforms

    def __call__(self, data: Any) -> Any:
        t = random.choice(self.transforms)
        return t(data)


class RandomOrder(Transform):
    """
    Apply transforms in random order.

    Args:
        transforms: List of transforms to apply.
    """

    def __init__(self, transforms: List[Transform]):
        self.transforms = transforms

    def __call__(self, data: Any) -> Any:
        order = list(range(len(self.transforms)))
        random.shuffle(order)
        for i in order:
            data = self.transforms[i](data)
        return data

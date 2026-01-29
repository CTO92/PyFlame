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

    Supports numpy arrays with shapes:
    - (H, W) - grayscale
    - (H, W, C) - HWC format (default for numpy/PIL)
    - (C, H, W) - CHW format (common in deep learning)

    Uses cv2 or PIL if available, falls back to pure numpy bilinear interpolation.

    Args:
        size: Desired output size. If int, smaller edge is matched while
            preserving aspect ratio. If tuple (height, width), exact size.
        interpolation: Interpolation mode - "nearest", "bilinear", "bicubic".
            Note: "bicubic" requires cv2 or PIL, falls back to bilinear with numpy.

    Example:
        >>> transform = Resize(224)
        >>> resized = transform(image)  # Smaller edge becomes 224, aspect preserved
        >>> transform = Resize((224, 224))
        >>> resized = transform(image)  # Exact 224x224 output
    """

    def __init__(
        self,
        size: Union[int, Tuple[int, int]],
        interpolation: str = "bilinear",
    ):
        self.size = size
        self.interpolation = interpolation
        if interpolation not in ("nearest", "bilinear", "bicubic", "lanczos"):
            raise ValueError(
                f"Unsupported interpolation mode: {interpolation}. "
                "Supported: nearest, bilinear, bicubic, lanczos"
            )

    def _get_image_size(self, img: Any) -> Tuple[int, int, bool]:
        """Get height, width and whether image is in CHW format."""
        import numpy as np

        if not isinstance(img, np.ndarray):
            img = np.asarray(img)

        if img.ndim == 2:
            return img.shape[0], img.shape[1], False
        elif img.ndim == 3:
            if img.shape[0] in (1, 3, 4) and img.shape[0] < img.shape[1]:
                return img.shape[1], img.shape[2], True
            else:
                return img.shape[0], img.shape[1], False
        else:
            raise ValueError(f"Unsupported image dimensions: {img.ndim}")

    def _compute_output_size(self, input_h: int, input_w: int) -> Tuple[int, int]:
        """Compute output size, preserving aspect ratio if size is int."""
        if isinstance(self.size, int):
            if input_h < input_w:
                new_h = self.size
                new_w = int(input_w * self.size / input_h)
            else:
                new_w = self.size
                new_h = int(input_h * self.size / input_w)
            return new_h, new_w
        return self.size

    def _resize_numpy_nearest(self, img: Any, new_h: int, new_w: int) -> Any:
        """Nearest neighbor interpolation using numpy."""
        import numpy as np

        h, w = img.shape[:2]
        row_indices = (np.arange(new_h) * h / new_h).astype(int)
        col_indices = (np.arange(new_w) * w / new_w).astype(int)

        # Clip to valid range
        row_indices = np.clip(row_indices, 0, h - 1)
        col_indices = np.clip(col_indices, 0, w - 1)

        if img.ndim == 2:
            return img[row_indices][:, col_indices]
        else:
            return img[row_indices][:, col_indices, :]

    def _resize_numpy_bilinear(self, img: Any, new_h: int, new_w: int) -> Any:
        """Bilinear interpolation using numpy."""
        import numpy as np

        h, w = img.shape[:2]
        dtype = img.dtype

        # Create coordinate grids
        y_coords = np.linspace(0, h - 1, new_h)
        x_coords = np.linspace(0, w - 1, new_w)

        # Get integer and fractional parts
        y0 = np.floor(y_coords).astype(int)
        x0 = np.floor(x_coords).astype(int)
        y1 = np.minimum(y0 + 1, h - 1)
        x1 = np.minimum(x0 + 1, w - 1)

        y_frac = y_coords - y0
        x_frac = x_coords - x0

        # Reshape for broadcasting
        y0 = y0.reshape(-1, 1)
        y1 = y1.reshape(-1, 1)
        y_frac = y_frac.reshape(-1, 1)

        if img.ndim == 2:
            # Bilinear interpolation for 2D
            top = img[y0, x0] * (1 - x_frac) + img[y0, x1] * x_frac
            bottom = img[y1, x0] * (1 - x_frac) + img[y1, x1] * x_frac
            result = top * (1 - y_frac) + bottom * y_frac
        else:
            # Bilinear interpolation for 3D (HWC)
            y_frac = y_frac.reshape(-1, 1, 1)
            top = img[y0.flatten()][:, x0, :] * (1 - x_frac).reshape(1, -1, 1) + img[
                y0.flatten()
            ][:, x1, :] * x_frac.reshape(1, -1, 1)
            bottom = img[y1.flatten()][:, x0, :] * (1 - x_frac).reshape(1, -1, 1) + img[
                y1.flatten()
            ][:, x1, :] * x_frac.reshape(1, -1, 1)
            result = top * (1 - y_frac) + bottom * y_frac

        # Preserve dtype
        if np.issubdtype(dtype, np.integer):
            result = np.clip(result, 0, np.iinfo(dtype).max).astype(dtype)
        else:
            result = result.astype(dtype)

        return result

    def _resize_cv2(self, img: Any, new_h: int, new_w: int) -> Any:
        """Resize using OpenCV."""
        import cv2

        interp_map = {
            "nearest": cv2.INTER_NEAREST,
            "bilinear": cv2.INTER_LINEAR,
            "bicubic": cv2.INTER_CUBIC,
            "lanczos": cv2.INTER_LANCZOS4,
        }
        return cv2.resize(
            img, (new_w, new_h), interpolation=interp_map[self.interpolation]
        )

    def _resize_pil(self, img: Any, new_h: int, new_w: int) -> Any:
        """Resize using PIL."""
        import numpy as np
        from PIL import Image

        interp_map = {
            "nearest": Image.NEAREST,
            "bilinear": Image.BILINEAR,
            "bicubic": Image.BICUBIC,
            "lanczos": Image.LANCZOS,
        }

        if img.ndim == 2:
            pil_img = Image.fromarray(img)
        else:
            pil_img = Image.fromarray(img)

        resized = pil_img.resize((new_w, new_h), interp_map[self.interpolation])
        return np.array(resized)

    def __call__(self, img: Any) -> Any:
        import numpy as np

        if not isinstance(img, np.ndarray):
            img = np.asarray(img)

        h, w, is_chw = self._get_image_size(img)
        new_h, new_w = self._compute_output_size(h, w)

        # If CHW, transpose to HWC for processing
        if is_chw:
            img = np.transpose(img, (1, 2, 0))

        # Try cv2 first (fastest)
        try:
            result = self._resize_cv2(img, new_h, new_w)
        except ImportError:
            # Try PIL
            try:
                result = self._resize_pil(img, new_h, new_w)
            except ImportError:
                # Fallback to numpy
                if self.interpolation == "nearest":
                    result = self._resize_numpy_nearest(img, new_h, new_w)
                else:
                    # Use bilinear for all other modes in numpy fallback
                    result = self._resize_numpy_bilinear(img, new_h, new_w)

        # Transpose back if needed
        if is_chw:
            result = np.transpose(result, (2, 0, 1))

        return result

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={self.size}, interpolation='{self.interpolation}')"


class CenterCrop(Transform):
    """
    Crop the center of the image.

    Supports numpy arrays with shapes:
    - (H, W) - grayscale
    - (H, W, C) - HWC format (default for numpy/PIL)
    - (C, H, W) - CHW format (common in deep learning)

    Args:
        size: Desired output size (height, width) or single int for square crop.

    Example:
        >>> transform = CenterCrop(224)
        >>> cropped = transform(image)  # Returns 224x224 center crop
    """

    def __init__(self, size: Union[int, Tuple[int, int]]):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = tuple(size)

    def _get_image_size(self, img: Any) -> Tuple[int, int, bool]:
        """Get height, width and whether image is in CHW format."""
        import numpy as np

        if not isinstance(img, np.ndarray):
            img = np.asarray(img)

        if img.ndim == 2:
            # Grayscale (H, W)
            return img.shape[0], img.shape[1], False
        elif img.ndim == 3:
            # Check if CHW or HWC
            if img.shape[0] in (1, 3, 4) and img.shape[0] < img.shape[1]:
                # Likely CHW format
                return img.shape[1], img.shape[2], True
            else:
                # HWC format
                return img.shape[0], img.shape[1], False
        else:
            raise ValueError(f"Unsupported image dimensions: {img.ndim}")

    def __call__(self, img: Any) -> Any:
        import numpy as np

        if not isinstance(img, np.ndarray):
            img = np.asarray(img)

        h, w, is_chw = self._get_image_size(img)
        crop_h, crop_w = self.size

        if crop_h > h or crop_w > w:
            raise ValueError(
                f"Crop size {self.size} is larger than image size ({h}, {w})"
            )

        # Calculate center crop coordinates
        top = (h - crop_h) // 2
        left = (w - crop_w) // 2

        if img.ndim == 2:
            return img[top : top + crop_h, left : left + crop_w]
        elif is_chw:
            return img[:, top : top + crop_h, left : left + crop_w]
        else:
            return img[top : top + crop_h, left : left + crop_w, :]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={self.size})"


class RandomCrop(Transform):
    """
    Crop a random portion of the image.

    Supports numpy arrays with shapes:
    - (H, W) - grayscale
    - (H, W, C) - HWC format (default for numpy/PIL)
    - (C, H, W) - CHW format (common in deep learning)

    Args:
        size: Desired output size (height, width) or single int for square crop.
        padding: Optional padding on each border. Can be:
            - int: Same padding on all sides
            - (pad_h, pad_w): Symmetric padding for height and width
            - (left, top, right, bottom): Different padding for each side
        pad_value: Value to fill padding with (default 0).

    Example:
        >>> transform = RandomCrop(224, padding=10)
        >>> cropped = transform(image)  # Random 224x224 crop with 10px padding
    """

    def __init__(
        self,
        size: Union[int, Tuple[int, int]],
        padding: Optional[Union[int, Tuple[int, ...]]] = None,
        pad_value: float = 0,
    ):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = tuple(size)
        self.padding = padding
        self.pad_value = pad_value

    def _get_image_size(self, img: Any) -> Tuple[int, int, bool]:
        """Get height, width and whether image is in CHW format."""
        import numpy as np

        if not isinstance(img, np.ndarray):
            img = np.asarray(img)

        if img.ndim == 2:
            return img.shape[0], img.shape[1], False
        elif img.ndim == 3:
            if img.shape[0] in (1, 3, 4) and img.shape[0] < img.shape[1]:
                return img.shape[1], img.shape[2], True
            else:
                return img.shape[0], img.shape[1], False
        else:
            raise ValueError(f"Unsupported image dimensions: {img.ndim}")

    def _apply_padding(self, img: Any, is_chw: bool) -> Any:
        """Apply padding to image."""
        import numpy as np

        if self.padding is None:
            return img

        if isinstance(self.padding, int):
            pad_left = pad_right = pad_top = pad_bottom = self.padding
        elif len(self.padding) == 2:
            pad_top = pad_bottom = self.padding[0]
            pad_left = pad_right = self.padding[1]
        elif len(self.padding) == 4:
            pad_left, pad_top, pad_right, pad_bottom = self.padding
        else:
            raise ValueError(f"Invalid padding: {self.padding}")

        if img.ndim == 2:
            pad_width = ((pad_top, pad_bottom), (pad_left, pad_right))
        elif is_chw:
            pad_width = ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right))
        else:
            pad_width = ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0))

        return np.pad(img, pad_width, mode="constant", constant_values=self.pad_value)

    def __call__(self, img: Any) -> Any:
        import numpy as np

        if not isinstance(img, np.ndarray):
            img = np.asarray(img)

        # Get original dimensions
        _, _, is_chw = self._get_image_size(img)

        # Apply padding
        img = self._apply_padding(img, is_chw)

        # Get padded dimensions
        h, w, _ = self._get_image_size(img)
        crop_h, crop_w = self.size

        if crop_h > h or crop_w > w:
            raise ValueError(
                f"Crop size {self.size} is larger than padded image size ({h}, {w})"
            )

        # Random top-left corner
        top = random.randint(0, h - crop_h)
        left = random.randint(0, w - crop_w)

        if img.ndim == 2:
            return img[top : top + crop_h, left : left + crop_w]
        elif is_chw:
            return img[:, top : top + crop_h, left : left + crop_w]
        else:
            return img[top : top + crop_h, left : left + crop_w, :]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={self.size}, padding={self.padding})"


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

    Supports numpy arrays with shapes:
    - (H, W) - grayscale
    - (H, W, C) - HWC format (default for numpy/PIL)
    - (C, H, W) - CHW format (common in deep learning)

    Uses cv2 or PIL if available, falls back to pure numpy implementation.

    Args:
        degrees: Range of degrees for rotation. If single number, range is
            (-degrees, +degrees).
        expand: If True, expand output to fit entire rotated image.
            If False (default), keep original image size.
        center: Center of rotation. If None, uses image center.
        fill: Fill value for areas outside the rotated image.

    Example:
        >>> transform = RandomRotation(30)
        >>> rotated = transform(image)  # Random rotation between -30 and +30 degrees
    """

    def __init__(
        self,
        degrees: Union[float, Tuple[float, float]],
        expand: bool = False,
        center: Optional[Tuple[float, float]] = None,
        fill: float = 0,
    ):
        if isinstance(degrees, (int, float)):
            self.degrees = (-float(degrees), float(degrees))
        else:
            self.degrees = (float(degrees[0]), float(degrees[1]))
        self.expand = expand
        self.center = center
        self.fill = fill

    def _get_image_size(self, img: Any) -> Tuple[int, int, bool]:
        """Get height, width and whether image is in CHW format."""
        import numpy as np

        if not isinstance(img, np.ndarray):
            img = np.asarray(img)

        if img.ndim == 2:
            return img.shape[0], img.shape[1], False
        elif img.ndim == 3:
            if img.shape[0] in (1, 3, 4) and img.shape[0] < img.shape[1]:
                return img.shape[1], img.shape[2], True
            else:
                return img.shape[0], img.shape[1], False
        else:
            raise ValueError(f"Unsupported image dimensions: {img.ndim}")

    def _rotate_cv2(self, img: Any, angle: float, h: int, w: int) -> Any:
        """Rotate using OpenCV."""
        import cv2

        center = self.center if self.center else (w / 2, h / 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

        if self.expand:
            # Calculate new bounding box
            cos_a = abs(matrix[0, 0])
            sin_a = abs(matrix[0, 1])
            new_w = int(h * sin_a + w * cos_a)
            new_h = int(h * cos_a + w * sin_a)
            # Adjust matrix for new center
            matrix[0, 2] += (new_w - w) / 2
            matrix[1, 2] += (new_h - h) / 2
            out_size = (new_w, new_h)
        else:
            out_size = (w, h)

        return cv2.warpAffine(
            img,
            matrix,
            out_size,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=self.fill,
        )

    def _rotate_pil(self, img: Any, angle: float) -> Any:
        """Rotate using PIL."""
        import numpy as np
        from PIL import Image

        pil_img = Image.fromarray(img)
        rotated = pil_img.rotate(
            angle,
            expand=self.expand,
            center=self.center,
            fillcolor=int(self.fill) if img.dtype == np.uint8 else self.fill,
        )
        return np.array(rotated)

    def _rotate_numpy(self, img: Any, angle: float, h: int, w: int) -> Any:
        """Rotate using pure numpy (bilinear interpolation)."""
        import numpy as np

        angle_rad = np.deg2rad(-angle)  # Negative for correct rotation direction
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)

        # Center of rotation
        if self.center:
            cx, cy = self.center
        else:
            cx, cy = w / 2, h / 2

        if self.expand:
            # Calculate new dimensions
            new_w = int(abs(h * sin_a) + abs(w * cos_a))
            new_h = int(abs(h * cos_a) + abs(w * sin_a))
            new_cx, new_cy = new_w / 2, new_h / 2
        else:
            new_w, new_h = w, h
            new_cx, new_cy = cx, cy

        # Create output array
        if img.ndim == 2:
            result = np.full((new_h, new_w), self.fill, dtype=img.dtype)
        else:
            result = np.full((new_h, new_w, img.shape[2]), self.fill, dtype=img.dtype)

        # Create coordinate grid for output
        y_out, x_out = np.meshgrid(np.arange(new_h), np.arange(new_w), indexing="ij")

        # Transform to source coordinates (inverse rotation)
        x_src = cos_a * (x_out - new_cx) + sin_a * (y_out - new_cy) + cx
        y_src = -sin_a * (x_out - new_cx) + cos_a * (y_out - new_cy) + cy

        # Bilinear interpolation
        x0 = np.floor(x_src).astype(int)
        y0 = np.floor(y_src).astype(int)
        x1 = x0 + 1
        y1 = y0 + 1

        # Create mask for valid coordinates
        valid = (x0 >= 0) & (x1 < w) & (y0 >= 0) & (y1 < h)

        # Clip coordinates for indexing
        x0_c = np.clip(x0, 0, w - 1)
        x1_c = np.clip(x1, 0, w - 1)
        y0_c = np.clip(y0, 0, h - 1)
        y1_c = np.clip(y1, 0, h - 1)

        # Interpolation weights
        wx = x_src - x0
        wy = y_src - y0

        if img.ndim == 2:
            # Bilinear interpolation for grayscale
            val = (
                img[y0_c, x0_c] * (1 - wx) * (1 - wy)
                + img[y0_c, x1_c] * wx * (1 - wy)
                + img[y1_c, x0_c] * (1 - wx) * wy
                + img[y1_c, x1_c] * wx * wy
            )
            result[valid] = val[valid]
        else:
            # Bilinear interpolation for color
            wx = wx[..., np.newaxis]
            wy = wy[..., np.newaxis]
            val = (
                img[y0_c, x0_c] * (1 - wx) * (1 - wy)
                + img[y0_c, x1_c] * wx * (1 - wy)
                + img[y1_c, x0_c] * (1 - wx) * wy
                + img[y1_c, x1_c] * wx * wy
            )
            result[valid] = val[valid]

        return result.astype(img.dtype)

    def __call__(self, img: Any) -> Any:
        import numpy as np

        if not isinstance(img, np.ndarray):
            img = np.asarray(img)

        # Get random angle
        angle = random.uniform(self.degrees[0], self.degrees[1])

        h, w, is_chw = self._get_image_size(img)

        # If CHW, transpose to HWC for processing
        if is_chw:
            img = np.transpose(img, (1, 2, 0))

        # Try cv2 first (fastest)
        try:
            result = self._rotate_cv2(img, angle, h, w)
        except ImportError:
            # Try PIL
            try:
                result = self._rotate_pil(img, angle)
            except ImportError:
                # Fallback to numpy
                result = self._rotate_numpy(img, angle, h, w)

        # Transpose back if needed
        if is_chw:
            result = np.transpose(result, (2, 0, 1))

        return result

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(degrees={self.degrees}, expand={self.expand})"
        )


class ColorJitter(Transform):
    """
    Randomly change brightness, contrast, saturation, and hue.

    Supports numpy arrays with shapes:
    - (H, W, 3) - RGB image in HWC format
    - (3, H, W) - RGB image in CHW format

    Note: Grayscale images only support brightness and contrast adjustments.

    Args:
        brightness: How much to jitter brightness. If float, uniformly sample
            from [max(0, 1-brightness), 1+brightness]. If tuple (min, max),
            sample from that range.
        contrast: How much to jitter contrast. Same format as brightness.
        saturation: How much to jitter saturation. Same format as brightness.
        hue: How much to jitter hue. If float, sample from [-hue, hue].
            Should be in range [0, 0.5]. If tuple (min, max), sample from range.

    Example:
        >>> transform = ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
        >>> jittered = transform(image)
    """

    def __init__(
        self,
        brightness: Union[float, Tuple[float, float]] = 0,
        contrast: Union[float, Tuple[float, float]] = 0,
        saturation: Union[float, Tuple[float, float]] = 0,
        hue: Union[float, Tuple[float, float]] = 0,
    ):
        self.brightness = self._check_input(brightness, "brightness")
        self.contrast = self._check_input(contrast, "contrast")
        self.saturation = self._check_input(saturation, "saturation")
        self.hue = self._check_input(hue, "hue", center=0, bound=(-0.5, 0.5))

    def _check_input(
        self,
        value: Union[float, Tuple[float, float]],
        name: str,
        center: float = 1.0,
        bound: Tuple[float, float] = (0, float("inf")),
    ) -> Optional[Tuple[float, float]]:
        """Validate and convert input to range tuple."""
        if isinstance(value, (int, float)):
            if value < 0:
                raise ValueError(f"{name} must be non-negative, got {value}")
            if value == 0:
                return None
            if name == "hue":
                return (-value, value)
            return (max(bound[0], center - value), center + value)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            return (float(value[0]), float(value[1]))
        else:
            raise TypeError(f"{name} must be a float or tuple of 2 floats")

    def _get_image_size(self, img: Any) -> Tuple[int, int, bool]:
        """Get height, width and whether image is in CHW format."""
        import numpy as np

        if not isinstance(img, np.ndarray):
            img = np.asarray(img)

        if img.ndim == 2:
            return img.shape[0], img.shape[1], False
        elif img.ndim == 3:
            if img.shape[0] in (1, 3, 4) and img.shape[0] < img.shape[1]:
                return img.shape[1], img.shape[2], True
            else:
                return img.shape[0], img.shape[1], False
        else:
            raise ValueError(f"Unsupported image dimensions: {img.ndim}")

    def _rgb_to_hsv(self, rgb: Any) -> Any:
        """Convert RGB image to HSV. Input should be float in [0, 1]."""
        import numpy as np

        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]

        max_c = np.maximum(np.maximum(r, g), b)
        min_c = np.minimum(np.minimum(r, g), b)
        diff = max_c - min_c

        # Hue calculation
        h = np.zeros_like(max_c)
        mask = diff != 0

        # Red is max
        idx = mask & (max_c == r)
        h[idx] = ((g[idx] - b[idx]) / diff[idx]) % 6

        # Green is max
        idx = mask & (max_c == g)
        h[idx] = (b[idx] - r[idx]) / diff[idx] + 2

        # Blue is max
        idx = mask & (max_c == b)
        h[idx] = (r[idx] - g[idx]) / diff[idx] + 4

        h = h / 6.0  # Normalize to [0, 1]

        # Saturation
        s = np.where(max_c != 0, diff / max_c, 0)

        # Value
        v = max_c

        return np.stack([h, s, v], axis=-1)

    def _hsv_to_rgb(self, hsv: Any) -> Any:
        """Convert HSV image to RGB. Input H in [0, 1], S and V in [0, 1]."""
        import numpy as np

        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]

        h = h * 6.0  # Scale to [0, 6]
        i = np.floor(h).astype(int)
        f = h - i
        p = v * (1 - s)
        q = v * (1 - s * f)
        t = v * (1 - s * (1 - f))

        i = i % 6

        # Build RGB channels based on hue sector
        r = np.choose(i, [v, q, p, p, t, v])
        g = np.choose(i, [t, v, v, q, p, p])
        b = np.choose(i, [p, p, t, v, v, q])

        return np.stack([r, g, b], axis=-1)

    def _adjust_brightness(self, img: Any, factor: float) -> Any:
        """Multiply pixel values by factor."""
        import numpy as np

        return np.clip(img * factor, 0, 1)

    def _adjust_contrast(self, img: Any, factor: float) -> Any:
        """Adjust contrast around mean."""
        import numpy as np

        if img.ndim == 2:
            mean = img.mean()
        else:
            # Use luminance for RGB
            mean = (
                0.2989 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]
            ).mean()

        return np.clip((img - mean) * factor + mean, 0, 1)

    def _adjust_saturation(self, img: Any, factor: float) -> Any:
        """Adjust saturation in HSV space."""
        import numpy as np

        if img.ndim == 2:
            return img  # No saturation for grayscale

        hsv = self._rgb_to_hsv(img)
        hsv[..., 1] = np.clip(hsv[..., 1] * factor, 0, 1)
        return self._hsv_to_rgb(hsv)

    def _adjust_hue(self, img: Any, factor: float) -> Any:
        """Shift hue in HSV space."""
        if img.ndim == 2:
            return img  # No hue for grayscale

        hsv = self._rgb_to_hsv(img)
        hsv[..., 0] = (hsv[..., 0] + factor) % 1.0  # Hue wraps around
        return self._hsv_to_rgb(hsv)

    def __call__(self, img: Any) -> Any:
        import numpy as np

        if not isinstance(img, np.ndarray):
            img = np.asarray(img)

        _, _, is_chw = self._get_image_size(img)

        # If CHW, transpose to HWC for processing
        if is_chw:
            img = np.transpose(img, (1, 2, 0))

        # Convert to float [0, 1] for processing
        orig_dtype = img.dtype
        if np.issubdtype(orig_dtype, np.integer):
            max_val = np.iinfo(orig_dtype).max
            img = img.astype(np.float32) / max_val
        else:
            img = img.astype(np.float32)
            max_val = 1.0

        # Build list of transforms to apply
        transforms = []

        if self.brightness is not None:
            factor = random.uniform(self.brightness[0], self.brightness[1])
            transforms.append(lambda x, f=factor: self._adjust_brightness(x, f))

        if self.contrast is not None:
            factor = random.uniform(self.contrast[0], self.contrast[1])
            transforms.append(lambda x, f=factor: self._adjust_contrast(x, f))

        if self.saturation is not None:
            factor = random.uniform(self.saturation[0], self.saturation[1])
            transforms.append(lambda x, f=factor: self._adjust_saturation(x, f))

        if self.hue is not None:
            factor = random.uniform(self.hue[0], self.hue[1])
            transforms.append(lambda x, f=factor: self._adjust_hue(x, f))

        # Apply transforms in random order
        random.shuffle(transforms)
        for t in transforms:
            img = t(img)

        # Convert back to original dtype
        if np.issubdtype(orig_dtype, np.integer):
            img = np.clip(img * max_val, 0, max_val).astype(orig_dtype)
        else:
            img = img.astype(orig_dtype)

        # Transpose back if needed
        if is_chw:
            img = np.transpose(img, (2, 0, 1))

        return img

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"brightness={self.brightness}, contrast={self.contrast}, "
            f"saturation={self.saturation}, hue={self.hue})"
        )


class RandomErasing(Transform):
    """
    Randomly erase a rectangular region in an image.

    This is a data augmentation technique that helps models become more robust
    by randomly occluding parts of the input image during training.

    Supports numpy arrays with shapes:
    - (H, W) - grayscale
    - (H, W, C) - HWC format (default for numpy/PIL)
    - (C, H, W) - CHW format (common in deep learning)

    Args:
        p: Probability of performing erasing.
        scale: Range of proportion of erased area relative to image area.
        ratio: Range of aspect ratio of erased area.
        value: Erasing value. Can be:
            - float/int: Fill with constant value
            - "random": Fill with random values
            - tuple of 3 floats: Fill with RGB color (for color images)

    Example:
        >>> transform = RandomErasing(p=0.5, scale=(0.02, 0.33))
        >>> augmented = transform(image)
    """

    def __init__(
        self,
        p: float = 0.5,
        scale: Tuple[float, float] = (0.02, 0.33),
        ratio: Tuple[float, float] = (0.3, 3.3),
        value: Union[float, str, Tuple[float, ...]] = 0,
    ):
        if not 0 <= p <= 1:
            raise ValueError(f"p must be between 0 and 1, got {p}")
        if scale[0] > scale[1]:
            raise ValueError(f"scale[0] must be <= scale[1], got {scale}")
        if ratio[0] > ratio[1]:
            raise ValueError(f"ratio[0] must be <= ratio[1], got {ratio}")

        self.p = p
        self.scale = scale
        self.ratio = ratio
        self.value = value

    def _get_image_size(self, img: Any) -> Tuple[int, int, bool]:
        """Get height, width and whether image is in CHW format."""
        import numpy as np

        if not isinstance(img, np.ndarray):
            img = np.asarray(img)

        if img.ndim == 2:
            return img.shape[0], img.shape[1], False
        elif img.ndim == 3:
            if img.shape[0] in (1, 3, 4) and img.shape[0] < img.shape[1]:
                return img.shape[1], img.shape[2], True
            else:
                return img.shape[0], img.shape[1], False
        else:
            raise ValueError(f"Unsupported image dimensions: {img.ndim}")

    def _get_erase_params(
        self, img_h: int, img_w: int, max_attempts: int = 10
    ) -> Optional[Tuple[int, int, int, int]]:
        """Calculate parameters for erasing region.

        Returns:
            Tuple of (top, left, height, width) or None if no valid region found.
        """
        import math

        img_area = img_h * img_w

        for _ in range(max_attempts):
            # Sample area and aspect ratio
            erase_area = random.uniform(self.scale[0], self.scale[1]) * img_area
            aspect_ratio = random.uniform(self.ratio[0], self.ratio[1])

            # Calculate dimensions
            erase_h = int(round(math.sqrt(erase_area * aspect_ratio)))
            erase_w = int(round(math.sqrt(erase_area / aspect_ratio)))

            # Check if dimensions are valid
            if erase_h < img_h and erase_w < img_w and erase_h > 0 and erase_w > 0:
                top = random.randint(0, img_h - erase_h)
                left = random.randint(0, img_w - erase_w)
                return top, left, erase_h, erase_w

        return None

    def __call__(self, img: Any) -> Any:
        import numpy as np

        # Check probability
        if random.random() >= self.p:
            return img

        if not isinstance(img, np.ndarray):
            img = np.asarray(img)

        h, w, is_chw = self._get_image_size(img)

        # Get erasing parameters
        params = self._get_erase_params(h, w)
        if params is None:
            return img

        top, left, erase_h, erase_w = params

        # Make a copy to avoid modifying original
        img = img.copy()

        # Determine fill value
        if self.value == "random":
            if img.ndim == 2:
                fill = np.random.uniform(0, 1, size=(erase_h, erase_w))
            elif is_chw:
                fill = np.random.uniform(0, 1, size=(img.shape[0], erase_h, erase_w))
            else:
                fill = np.random.uniform(0, 1, size=(erase_h, erase_w, img.shape[2]))

            # Match dtype
            if np.issubdtype(img.dtype, np.integer):
                max_val = np.iinfo(img.dtype).max
                fill = (fill * max_val).astype(img.dtype)
            else:
                fill = fill.astype(img.dtype)
        elif isinstance(self.value, (tuple, list)):
            # RGB fill value
            if is_chw:
                fill = np.array(self.value, dtype=img.dtype).reshape(-1, 1, 1)
                fill = np.broadcast_to(fill, (len(self.value), erase_h, erase_w)).copy()
            else:
                fill = np.array(self.value, dtype=img.dtype).reshape(1, 1, -1)
                fill = np.broadcast_to(fill, (erase_h, erase_w, len(self.value))).copy()
        else:
            # Scalar fill value
            fill = self.value

        # Apply erasing
        if img.ndim == 2:
            img[top : top + erase_h, left : left + erase_w] = fill
        elif is_chw:
            img[:, top : top + erase_h, left : left + erase_w] = fill
        else:
            img[top : top + erase_h, left : left + erase_w, :] = fill

        return img

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(p={self.p}, scale={self.scale}, "
            f"ratio={self.ratio}, value={self.value})"
        )


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
    Compute spectrogram from waveform using Short-Time Fourier Transform (STFT).

    Converts a 1D audio waveform into a 2D time-frequency representation.
    Uses pure numpy implementation, with optional scipy backend for better performance.

    Args:
        n_fft: FFT window size. Determines frequency resolution.
        hop_length: Number of samples between successive frames.
            Default is n_fft // 2 (50% overlap).
        win_length: Window length. Default is n_fft.
        window: Window function - "hann", "hamming", "blackman", or "ones".
        center: If True, pad signal on both sides so frames are centered.
        pad_mode: Padding mode when center=True - "reflect", "constant", "edge".
        power: Exponent for magnitude spectrogram.
            1.0 for magnitude, 2.0 for power spectrogram. None for complex.
        normalized: If True, normalize by window sum.

    Example:
        >>> transform = Spectrogram(n_fft=512, hop_length=256)
        >>> spec = transform(waveform)  # Shape: (n_fft // 2 + 1, num_frames)
    """

    def __init__(
        self,
        n_fft: int = 400,
        hop_length: Optional[int] = None,
        win_length: Optional[int] = None,
        window: str = "hann",
        center: bool = True,
        pad_mode: str = "reflect",
        power: Optional[float] = 2.0,
        normalized: bool = False,
    ):
        self.n_fft = n_fft
        self.hop_length = hop_length or n_fft // 2
        self.win_length = win_length or n_fft
        self.window = window
        self.center = center
        self.pad_mode = pad_mode
        self.power = power
        self.normalized = normalized

        if self.win_length > self.n_fft:
            raise ValueError(
                f"win_length ({self.win_length}) must be <= n_fft ({self.n_fft})"
            )

    def _get_window(self) -> Any:
        """Create window function."""
        import numpy as np

        if self.window == "hann":
            win = np.hanning(self.win_length)
        elif self.window == "hamming":
            win = np.hamming(self.win_length)
        elif self.window == "blackman":
            win = np.blackman(self.win_length)
        elif self.window == "ones":
            win = np.ones(self.win_length)
        else:
            raise ValueError(f"Unknown window type: {self.window}")

        # Pad window to n_fft if needed
        if self.win_length < self.n_fft:
            left_pad = (self.n_fft - self.win_length) // 2
            right_pad = self.n_fft - self.win_length - left_pad
            win = np.pad(win, (left_pad, right_pad), mode="constant")

        return win

    def _stft_numpy(self, waveform: Any) -> Any:
        """Compute STFT using pure numpy."""
        import numpy as np

        # Center padding
        if self.center:
            pad_length = self.n_fft // 2
            waveform = np.pad(waveform, (pad_length, pad_length), mode=self.pad_mode)

        # Calculate number of frames
        num_samples = len(waveform)
        num_frames = 1 + (num_samples - self.n_fft) // self.hop_length

        if num_frames <= 0:
            raise ValueError(
                f"Signal too short ({num_samples} samples) for n_fft={self.n_fft}"
            )

        # Get window
        window = self._get_window()

        # Create output array for complex STFT
        n_freq = self.n_fft // 2 + 1
        stft = np.zeros((n_freq, num_frames), dtype=np.complex128)

        # Compute STFT frame by frame
        for i in range(num_frames):
            start = i * self.hop_length
            frame = waveform[start : start + self.n_fft] * window
            stft[:, i] = np.fft.rfft(frame, n=self.n_fft)

        return stft

    def _stft_scipy(self, waveform: Any) -> Any:
        """Compute STFT using scipy (if available)."""
        import numpy as np
        from scipy import signal

        # Get window
        if self.window == "ones":
            window = np.ones(self.win_length)
        else:
            window = signal.get_window(self.window, self.win_length)

        # Pad window if needed
        if self.win_length < self.n_fft:
            left_pad = (self.n_fft - self.win_length) // 2
            right_pad = self.n_fft - self.win_length - left_pad
            window = np.pad(window, (left_pad, right_pad), mode="constant")

        # Use scipy's stft
        _, _, stft = signal.stft(
            waveform,
            nperseg=self.n_fft,
            noverlap=self.n_fft - self.hop_length,
            window=window,
            boundary="zeros" if self.center else None,
            padded=self.center,
        )

        return stft

    def __call__(self, waveform: Any) -> Any:
        import numpy as np

        if not isinstance(waveform, np.ndarray):
            waveform = np.asarray(waveform)

        # Ensure 1D
        if waveform.ndim != 1:
            raise ValueError(f"Expected 1D waveform, got shape {waveform.shape}")

        # Convert to float
        if not np.issubdtype(waveform.dtype, np.floating):
            waveform = waveform.astype(np.float64)

        # Try scipy first (more optimized)
        try:
            stft = self._stft_scipy(waveform)
        except ImportError:
            stft = self._stft_numpy(waveform)

        # Apply normalization
        if self.normalized:
            window = self._get_window()
            stft = stft / window.sum()

        # Convert to magnitude/power spectrogram
        if self.power is not None:
            spectrogram = np.abs(stft) ** self.power
            return spectrogram.astype(np.float32)

        return stft

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(n_fft={self.n_fft}, "
            f"hop_length={self.hop_length}, win_length={self.win_length}, "
            f"window='{self.window}', power={self.power})"
        )


class MelSpectrogram(Transform):
    """
    Compute mel spectrogram from waveform.

    Combines STFT with a mel filterbank to produce a mel-scaled spectrogram,
    which better represents human auditory perception.

    Uses pure numpy implementation with optional librosa backend for
    better performance.

    Args:
        sample_rate: Sample rate of audio in Hz.
        n_fft: FFT window size.
        n_mels: Number of mel filterbanks.
        hop_length: Number of samples between successive frames.
        win_length: Window length. Default is n_fft.
        window: Window function - "hann", "hamming", "blackman", or "ones".
        center: If True, pad signal on both sides so frames are centered.
        f_min: Minimum frequency for mel filterbank.
        f_max: Maximum frequency for mel filterbank. Default is sample_rate / 2.
        power: Exponent for magnitude spectrogram (1.0 or 2.0).
        norm: Filterbank normalization - None or "slaney".
        log: If True, return log mel spectrogram.
        log_offset: Small offset added before log to avoid log(0).

    Example:
        >>> transform = MelSpectrogram(sample_rate=16000, n_mels=80)
        >>> mel_spec = transform(waveform)  # Shape: (n_mels, num_frames)
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 400,
        n_mels: int = 80,
        hop_length: Optional[int] = None,
        win_length: Optional[int] = None,
        window: str = "hann",
        center: bool = True,
        f_min: float = 0.0,
        f_max: Optional[float] = None,
        power: float = 2.0,
        norm: Optional[str] = "slaney",
        log: bool = False,
        log_offset: float = 1e-6,
    ):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.hop_length = hop_length or n_fft // 2
        self.win_length = win_length or n_fft
        self.window = window
        self.center = center
        self.f_min = f_min
        self.f_max = f_max or sample_rate / 2.0
        self.power = power
        self.norm = norm
        self.log = log
        self.log_offset = log_offset

        # Cache the mel filterbank
        self._mel_filterbank: Optional[Any] = None

    def _hz_to_mel(self, hz: Any) -> Any:
        """Convert Hz to mel scale (HTK formula)."""
        import numpy as np

        return 2595.0 * np.log10(1.0 + hz / 700.0)

    def _mel_to_hz(self, mel: Any) -> Any:
        """Convert mel to Hz scale (HTK formula)."""
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    def _create_mel_filterbank(self) -> Any:
        """Create mel filterbank matrix."""
        import numpy as np

        # Number of FFT bins
        n_fft_bins = self.n_fft // 2 + 1

        # Frequency points
        fft_freqs = np.linspace(0, self.sample_rate / 2, n_fft_bins)

        # Mel points - evenly spaced in mel scale
        mel_min = self._hz_to_mel(self.f_min)
        mel_max = self._hz_to_mel(self.f_max)
        mel_points = np.linspace(mel_min, mel_max, self.n_mels + 2)
        hz_points = self._mel_to_hz(mel_points)

        # Create filterbank
        filterbank = np.zeros((self.n_mels, n_fft_bins))

        for i in range(self.n_mels):
            # Triangle filter defined by three points
            low = hz_points[i]
            center = hz_points[i + 1]
            high = hz_points[i + 2]

            # Rising slope (from low to center)
            rising = (fft_freqs >= low) & (fft_freqs <= center)
            if center > low:
                filterbank[i, rising] = (fft_freqs[rising] - low) / (center - low)

            # Falling slope (from center to high)
            falling = (fft_freqs >= center) & (fft_freqs <= high)
            if high > center:
                filterbank[i, falling] = (high - fft_freqs[falling]) / (high - center)

        # Apply normalization
        if self.norm == "slaney":
            # Slaney-style normalization: divide by bandwidth
            enorm = 2.0 / (hz_points[2 : self.n_mels + 2] - hz_points[: self.n_mels])
            filterbank *= enorm[:, np.newaxis]

        return filterbank.astype(np.float32)

    def _get_mel_filterbank(self) -> Any:
        """Get cached mel filterbank or create new one."""
        if self._mel_filterbank is None:
            self._mel_filterbank = self._create_mel_filterbank()
        return self._mel_filterbank

    def _compute_mel_spectrogram_librosa(self, waveform: Any) -> Any:
        """Compute mel spectrogram using librosa (if available)."""
        import librosa

        mel_spec = librosa.feature.melspectrogram(
            y=waveform,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=self.center,
            n_mels=self.n_mels,
            fmin=self.f_min,
            fmax=self.f_max,
            power=self.power,
            norm=self.norm,
        )
        return mel_spec

    def __call__(self, waveform: Any) -> Any:
        import numpy as np

        if not isinstance(waveform, np.ndarray):
            waveform = np.asarray(waveform)

        # Ensure 1D
        if waveform.ndim != 1:
            raise ValueError(f"Expected 1D waveform, got shape {waveform.shape}")

        # Convert to float
        if not np.issubdtype(waveform.dtype, np.floating):
            waveform = waveform.astype(np.float64)

        # Try librosa first (most optimized for audio)
        try:
            mel_spec = self._compute_mel_spectrogram_librosa(waveform)
        except ImportError:
            # Fallback: compute spectrogram then apply mel filterbank
            spectrogram_transform = Spectrogram(
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                window=self.window,
                center=self.center,
                power=self.power,
            )
            spec = spectrogram_transform(waveform)

            # Apply mel filterbank: (n_mels, n_fft_bins) @ (n_fft_bins, num_frames)
            mel_filterbank = self._get_mel_filterbank()
            mel_spec = np.dot(mel_filterbank, spec)

        # Apply log if requested
        if self.log:
            mel_spec = np.log(mel_spec + self.log_offset)

        return mel_spec.astype(np.float32)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(sample_rate={self.sample_rate}, "
            f"n_fft={self.n_fft}, n_mels={self.n_mels}, "
            f"hop_length={self.hop_length}, f_min={self.f_min}, f_max={self.f_max})"
        )


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

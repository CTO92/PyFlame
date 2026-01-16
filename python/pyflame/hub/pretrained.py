"""
Pretrained weight management for PyFlame.

Provides functionality for downloading, caching, and loading pretrained weights.
"""

import hashlib
import logging
import os
import pickle
import shutil
import ssl
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Network security settings
DOWNLOAD_TIMEOUT = 300  # 5 minutes max for downloads
CONNECTION_TIMEOUT = 30  # 30 seconds for initial connection


# Default cache directory
DEFAULT_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "pyflame")


@dataclass
class WeightInfo:
    """Information about pretrained weights."""

    name: str
    model_name: str
    url: str
    sha256: str
    size_mb: float
    description: str = ""
    license: str = ""
    dataset: str = ""
    metrics: Dict[str, float] = None

    def __post_init__(self):
        if self.metrics is None:
            self.metrics = {}


# Registry of available pretrained weights
_weight_registry: Dict[str, Dict[str, WeightInfo]] = {
    "resnet18": {
        "imagenet1k-v1": WeightInfo(
            name="imagenet1k-v1",
            model_name="resnet18",
            url="https://pyflame.example.com/weights/resnet18_imagenet1k_v1.pf",
            sha256="abc123...",
            size_mb=44.7,
            description="ResNet-18 trained on ImageNet-1K",
            license="Apache-2.0",
            dataset="ImageNet-1K",
            metrics={"top1_accuracy": 69.76, "top5_accuracy": 89.08},
        ),
    },
    "resnet50": {
        "imagenet1k-v1": WeightInfo(
            name="imagenet1k-v1",
            model_name="resnet50",
            url="https://pyflame.example.com/weights/resnet50_imagenet1k_v1.pf",
            sha256="def456...",
            size_mb=97.8,
            description="ResNet-50 trained on ImageNet-1K",
            license="Apache-2.0",
            dataset="ImageNet-1K",
            metrics={"top1_accuracy": 76.13, "top5_accuracy": 92.86},
        ),
        "imagenet1k-v2": WeightInfo(
            name="imagenet1k-v2",
            model_name="resnet50",
            url="https://pyflame.example.com/weights/resnet50_imagenet1k_v2.pf",
            sha256="ghi789...",
            size_mb=97.8,
            description="ResNet-50 trained on ImageNet-1K (improved training recipe)",
            license="Apache-2.0",
            dataset="ImageNet-1K",
            metrics={"top1_accuracy": 80.86, "top5_accuracy": 95.43},
        ),
    },
    "bert-base-uncased": {
        "default": WeightInfo(
            name="default",
            model_name="bert-base-uncased",
            url="https://pyflame.example.com/weights/bert_base_uncased.pf",
            sha256="jkl012...",
            size_mb=438.0,
            description="BERT base uncased pretrained on English Wikipedia + BookCorpus",
            license="Apache-2.0",
            dataset="Wikipedia + BookCorpus",
        ),
    },
}


def get_cache_dir() -> str:
    """Get the cache directory for pretrained weights."""
    cache_dir = os.environ.get("PYFLAME_CACHE_DIR", DEFAULT_CACHE_DIR)
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def get_weight_path(model_name: str, weight_name: str = "default") -> str:
    """
    Get the local path for pretrained weights.

    Downloads weights if not cached.

    Args:
        model_name: Name of the model.
        weight_name: Name of the weight variant.

    Returns:
        Path to the weight file.

    Example:
        >>> path = get_weight_path("resnet50", "imagenet1k-v1")
    """
    # Check if already cached
    cache_dir = get_cache_dir()
    weight_dir = os.path.join(cache_dir, "weights", model_name)
    weight_path = os.path.join(weight_dir, f"{weight_name}.pf")

    if os.path.exists(weight_path):
        return weight_path

    # Need to download
    return download_weights(model_name, weight_name)


def download_weights(
    model_name: str,
    weight_name: str = "default",
    force: bool = False,
) -> str:
    """
    Download pretrained weights.

    Args:
        model_name: Name of the model.
        weight_name: Name of the weight variant.
        force: If True, re-download even if cached.

    Returns:
        Path to the downloaded weight file.

    Example:
        >>> path = download_weights("resnet50", "imagenet1k-v1")
    """
    # Get weight info
    if model_name not in _weight_registry:
        raise ValueError(f"No pretrained weights available for model '{model_name}'")

    weights = _weight_registry[model_name]
    if weight_name not in weights:
        available = ", ".join(weights.keys())
        raise ValueError(
            f"Weight '{weight_name}' not found for model '{model_name}'. "
            f"Available: {available}"
        )

    info = weights[weight_name]

    # Setup paths
    cache_dir = get_cache_dir()
    weight_dir = os.path.join(cache_dir, "weights", model_name)
    os.makedirs(weight_dir, exist_ok=True)
    weight_path = os.path.join(weight_dir, f"{weight_name}.pf")

    # Check if already exists
    if os.path.exists(weight_path) and not force:
        print(f"Using cached weights: {weight_path}")
        return weight_path

    # Download
    print(f"Downloading {model_name}/{weight_name} ({info.size_mb:.1f} MB)...")

    try:
        _download_file(info.url, weight_path, expected_sha256=info.sha256)
    except Exception as e:
        # Clean up partial download
        if os.path.exists(weight_path):
            os.remove(weight_path)
        raise RuntimeError(f"Failed to download weights: {e}")

    print(f"Downloaded to: {weight_path}")
    return weight_path


def _download_file(
    url: str,
    path: str,
    expected_sha256: Optional[str] = None,
    chunk_size: int = 8192,
    timeout: int = DOWNLOAD_TIMEOUT,
) -> None:
    """Download a file with progress, SSL verification, and checksum verification.

    Args:
        url: URL to download from (must be HTTPS for security)
        path: Local path to save file
        expected_sha256: Expected SHA256 hash for verification
        chunk_size: Download chunk size in bytes
        timeout: Total download timeout in seconds
    """
    temp_path = path + ".download"

    # Security: Require HTTPS for downloads
    if not url.startswith("https://"):
        raise ValueError(
            f"Only HTTPS URLs are allowed for security. Got: {url[:50]}..."
        )

    # Create SSL context with proper certificate verification
    ssl_context = ssl.create_default_context()
    # Ensure certificate verification is enabled (default, but explicit)
    ssl_context.check_hostname = True
    ssl_context.verify_mode = ssl.CERT_REQUIRED

    try:
        request = urllib.request.Request(
            url,
            headers={
                "User-Agent": "PyFlame/1.0",  # Identify ourselves
            },
        )

        with urllib.request.urlopen(
            request,
            timeout=timeout,
            context=ssl_context,
        ) as response:
            total_size = int(response.headers.get("content-length", 0))
            downloaded = 0

            with open(temp_path, "wb") as f:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)

                    # Progress
                    if total_size > 0:
                        pct = 100 * downloaded / total_size
                        print(f"\rProgress: {pct:.1f}%", end="", flush=True)

            print()  # Newline after progress

    except ssl.SSLError as e:
        logger.error(f"SSL certificate verification failed for {url}: {e}")
        raise RuntimeError(
            "SSL certificate verification failed. "
            "The server's certificate may be invalid or expired."
        )
    except urllib.error.URLError as e:
        logger.error(f"Network error downloading {url}: {e}")
        raise RuntimeError(f"Network error: {e}")
    except TimeoutError:
        logger.error(f"Download timed out after {timeout} seconds: {url}")
        raise RuntimeError(
            f"Download timed out after {timeout} seconds. "
            f"Try again or check your network connection."
        )

    # Verify checksum
    if expected_sha256:
        actual_sha256 = _compute_sha256(temp_path)
        if actual_sha256 != expected_sha256:
            # Security logging: checksum mismatch could indicate tampering
            logger.warning(
                f"SECURITY: Checksum verification failed for downloaded file. "
                f"Expected {expected_sha256[:16]}..., got {actual_sha256[:16]}... "
                f"This may indicate file corruption or tampering."
            )
            os.remove(temp_path)
            raise RuntimeError(
                f"Checksum mismatch. Expected {expected_sha256}, got {actual_sha256}"
            )

    # Move to final path
    shutil.move(temp_path, path)


def _compute_sha256(path: str) -> str:
    """Compute SHA256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def load_pretrained(
    model: Any,
    model_name: str,
    weight_name: str = "default",
    strict: bool = True,
) -> None:
    """
    Load pretrained weights into a model.

    Args:
        model: Model instance to load weights into.
        model_name: Name of the model.
        weight_name: Name of the weight variant.
        strict: If True, raise error on missing/unexpected keys.

    Example:
        >>> model = ResNet50()
        >>> load_pretrained(model, "resnet50", "imagenet1k-v1")
    """
    # Get weight path (downloads if needed)
    weight_path = get_weight_path(model_name, weight_name)

    # Load weights
    try:
        # Try PyFlame native format first
        state_dict = _load_state_dict(weight_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load weights from {weight_path}: {e}")

    # Load into model
    if hasattr(model, "load_state_dict"):
        missing, unexpected = _load_state_dict_into_model(
            model, state_dict, strict=strict
        )

        if missing and strict:
            raise RuntimeError(f"Missing keys: {missing}")
        if unexpected and strict:
            raise RuntimeError(f"Unexpected keys: {unexpected}")

        if missing:
            print(f"Warning: Missing keys: {missing}")
        if unexpected:
            print(f"Warning: Unexpected keys: {unexpected}")
    else:
        raise TypeError("Model does not support state_dict loading")


class RestrictedUnpickler(pickle.Unpickler):
    """Restricted unpickler that only allows explicitly whitelisted classes.

    Security Note: This unpickler uses a strict allowlist approach. Only
    explicitly listed (module, class) pairs are allowed. Module prefix
    matching is NOT used to prevent unsafe classes from slipping through.
    """

    # Classes that are explicitly allowed (strict allowlist)
    # Format: (module, class_name)
    SAFE_CLASSES = frozenset(
        {
            # NumPy core types
            ("numpy", "ndarray"),
            ("numpy", "dtype"),
            ("numpy", "float32"),
            ("numpy", "float64"),
            ("numpy", "float16"),
            ("numpy", "int32"),
            ("numpy", "int64"),
            ("numpy", "int16"),
            ("numpy", "int8"),
            ("numpy", "uint8"),
            ("numpy", "bool_"),
            ("numpy.core.multiarray", "_reconstruct"),
            ("numpy.core.multiarray", "scalar"),
            ("numpy._core.multiarray", "_reconstruct"),
            ("numpy._core.multiarray", "scalar"),
            # Python builtins (data types only, no code execution)
            ("builtins", "dict"),
            ("builtins", "list"),
            ("builtins", "tuple"),
            ("builtins", "set"),
            ("builtins", "frozenset"),
            ("builtins", "bytes"),
            ("builtins", "bytearray"),
            ("builtins", "str"),
            ("builtins", "int"),
            ("builtins", "float"),
            ("builtins", "bool"),
            ("builtins", "complex"),
            ("builtins", "slice"),
            ("builtins", "range"),
            # Collections
            ("collections", "OrderedDict"),
            ("collections", "defaultdict"),
            # Copy module (for deepcopy support)
            ("copy", "_reconstructor"),
            # Functools (for partial, but NOT arbitrary callables)
            # NOTE: functools.partial is intentionally NOT included as it can
            # wrap arbitrary callables
        }
    )

    # Dangerous classes that should NEVER be allowed
    # These are checked separately to provide clear error messages
    DANGEROUS_CLASSES = frozenset(
        {
            ("builtins", "eval"),
            ("builtins", "exec"),
            ("builtins", "compile"),
            ("builtins", "open"),
            ("builtins", "input"),
            ("builtins", "__import__"),
            ("os", "system"),
            ("os", "popen"),
            ("subprocess", "Popen"),
            ("subprocess", "call"),
            ("subprocess", "run"),
            ("pickle", "loads"),
            ("codecs", "encode"),
            ("codecs", "decode"),
        }
    )

    def find_class(self, module: str, name: str):
        """Only allow explicitly whitelisted classes to be unpickled."""
        # Check for explicitly dangerous classes first
        if (module, name) in self.DANGEROUS_CLASSES:
            logger.error(
                f"SECURITY ALERT: Blocked dangerous pickle class: {module}.{name}. "
                f"This is a strong indicator of a malicious model file."
            )
            raise pickle.UnpicklingError(
                f"BLOCKED: Dangerous class '{module}.{name}' in model file. "
                f"This file may be malicious and should not be trusted."
            )

        # Check explicit class allowlist (strict - no module prefix matching)
        if (module, name) in self.SAFE_CLASSES:
            return super().find_class(module, name)

        # Security logging: log all blocked unpickle attempts
        logger.warning(
            f"SECURITY: Blocked unpickle of non-whitelisted class: {module}.{name}. "
            f"This may indicate a malicious model file or incompatible format."
        )
        raise pickle.UnpicklingError(
            f"Class '{module}.{name}' is not in the allowed list. "
            f"For security, only explicitly whitelisted classes can be unpickled. "
            f"Consider using SafeTensors format instead."
        )


def _load_state_dict(path: str) -> Dict[str, Any]:
    """Load a state dict from file with security restrictions.

    Uses a restricted unpickler to prevent arbitrary code execution
    from malicious model files.
    """
    import pickle

    with open(path, "rb") as f:
        try:
            return RestrictedUnpickler(f).load()
        except pickle.UnpicklingError as e:
            # Security logging: failed unpickle indicates potential malicious file
            logger.error(
                f"SECURITY: Failed to unpickle model file '{path}': {e}. "
                f"The file may be corrupted or contain malicious content."
            )
            raise RuntimeError(
                f"Failed to load model file '{path}': {e}. "
                f"The file may be corrupted or contain unsafe content."
            )


def _load_state_dict_into_model(
    model: Any,
    state_dict: Dict[str, Any],
    strict: bool = True,
) -> tuple:
    """Load state dict into model, returning missing and unexpected keys."""
    model_state = model.state_dict()

    missing = []
    unexpected = []

    for key in model_state:
        if key not in state_dict:
            missing.append(key)

    for key in state_dict:
        if key not in model_state:
            unexpected.append(key)

    # Load matching keys
    model.load_state_dict(state_dict, strict=strict)

    return missing, unexpected


def list_pretrained(model_name: Optional[str] = None) -> Dict[str, List[str]]:
    """
    List available pretrained weights.

    Args:
        model_name: If provided, list weights for specific model.

    Returns:
        Dict mapping model names to list of weight names.

    Example:
        >>> all_weights = list_pretrained()
        >>> resnet_weights = list_pretrained("resnet50")
    """
    if model_name is not None:
        if model_name not in _weight_registry:
            return {}
        return {model_name: list(_weight_registry[model_name].keys())}

    return {model: list(weights.keys()) for model, weights in _weight_registry.items()}


def weight_info(model_name: str, weight_name: str = "default") -> WeightInfo:
    """
    Get information about pretrained weights.

    Args:
        model_name: Name of the model.
        weight_name: Name of the weight variant.

    Returns:
        WeightInfo object.

    Example:
        >>> info = weight_info("resnet50", "imagenet1k-v1")
        >>> print(f"Accuracy: {info.metrics['top1_accuracy']}")
    """
    if model_name not in _weight_registry:
        raise ValueError(f"No weights for model '{model_name}'")

    weights = _weight_registry[model_name]
    if weight_name not in weights:
        raise ValueError(f"Weight '{weight_name}' not found for '{model_name}'")

    return weights[weight_name]


def clear_cache(model_name: Optional[str] = None) -> None:
    """
    Clear cached weights.

    Args:
        model_name: If provided, only clear weights for this model.

    Example:
        >>> clear_cache("resnet50")  # Clear resnet50 weights
        >>> clear_cache()  # Clear all weights
    """
    cache_dir = get_cache_dir()
    weight_dir = os.path.join(cache_dir, "weights")

    if model_name:
        target_dir = os.path.join(weight_dir, model_name)
        if os.path.exists(target_dir):
            shutil.rmtree(target_dir)
            print(f"Cleared cache for {model_name}")
    else:
        if os.path.exists(weight_dir):
            shutil.rmtree(weight_dir)
            print("Cleared all weight cache")


def cache_info() -> Dict[str, Any]:
    """
    Get information about the cache.

    Returns:
        Dict with cache directory, size, and cached models.

    Example:
        >>> info = cache_info()
        >>> print(f"Cache size: {info['size_mb']:.1f} MB")
    """
    cache_dir = get_cache_dir()
    weight_dir = os.path.join(cache_dir, "weights")

    info = {
        "cache_dir": cache_dir,
        "size_mb": 0.0,
        "cached_models": {},
    }

    if not os.path.exists(weight_dir):
        return info

    total_size = 0
    for model_name in os.listdir(weight_dir):
        model_dir = os.path.join(weight_dir, model_name)
        if os.path.isdir(model_dir):
            weights = []
            for weight_file in os.listdir(model_dir):
                if weight_file.endswith(".pf"):
                    weight_path = os.path.join(model_dir, weight_file)
                    size = os.path.getsize(weight_path)
                    total_size += size
                    weights.append(
                        {
                            "name": weight_file[:-3],  # Remove .pf
                            "size_mb": size / (1024 * 1024),
                        }
                    )
            info["cached_models"][model_name] = weights

    info["size_mb"] = total_size / (1024 * 1024)
    return info

"""
PyFlame Model Hub

Provides model registry and pretrained model loading:
- Model registry for organizing model architectures
- Pretrained weight downloading and caching
- Model cards with metadata
"""

from .registry import (
    ModelRegistry,
    register_model,
    get_model,
    list_models,
    model_info,
)
from .pretrained import (
    load_pretrained,
    download_weights,
    list_pretrained,
    get_weight_path,
)

__all__ = [
    # Registry
    "ModelRegistry",
    "register_model",
    "get_model",
    "list_models",
    "model_info",
    # Pretrained
    "load_pretrained",
    "download_weights",
    "list_pretrained",
    "get_weight_path",
]

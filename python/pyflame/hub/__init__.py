"""
PyFlame Model Hub

Provides model registry and pretrained model loading:
- Model registry for organizing model architectures
- Pretrained weight downloading and caching
- Model cards with metadata
"""

from .pretrained import (
    download_weights,
    get_weight_path,
    list_pretrained,
    load_pretrained,
)
from .registry import (
    ModelRegistry,
    get_model,
    list_models,
    model_info,
    register_model,
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

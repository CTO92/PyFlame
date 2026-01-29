"""
Benchmark models for PyFlame.

Provides standard models for benchmarking and comparison.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class BenchmarkModelInfo:
    """Information about a benchmark model.

    Attributes:
        name: Model name
        description: Model description
        input_shape: Default input shape
        num_params: Number of parameters
        flops: Estimated FLOPs
        category: Model category
    """

    name: str
    description: str
    input_shape: List[int]
    num_params: int = 0
    flops: int = 0
    category: str = "general"


# Registry of benchmark models
_BENCHMARK_MODELS: Dict[str, Dict[str, Any]] = {
    "resnet18": {
        "info": BenchmarkModelInfo(
            name="resnet18",
            description="ResNet-18 (18 layers)",
            input_shape=[3, 224, 224],
            num_params=11_700_000,
            flops=1_800_000_000,
            category="vision",
        ),
        "factory": "_create_resnet18",
    },
    "resnet50": {
        "info": BenchmarkModelInfo(
            name="resnet50",
            description="ResNet-50 (50 layers)",
            input_shape=[3, 224, 224],
            num_params=25_600_000,
            flops=4_100_000_000,
            category="vision",
        ),
        "factory": "_create_resnet50",
    },
    "bert_base": {
        "info": BenchmarkModelInfo(
            name="bert_base",
            description="BERT Base (12 layers, 768 hidden)",
            input_shape=[512],
            num_params=110_000_000,
            flops=22_000_000_000,
            category="nlp",
        ),
        "factory": "_create_bert_base",
    },
    "mlp_small": {
        "info": BenchmarkModelInfo(
            name="mlp_small",
            description="Small MLP (3 layers, 256 hidden)",
            input_shape=[128],
            num_params=100_000,
            flops=200_000,
            category="general",
        ),
        "factory": "_create_mlp_small",
    },
    "mlp_large": {
        "info": BenchmarkModelInfo(
            name="mlp_large",
            description="Large MLP (5 layers, 1024 hidden)",
            input_shape=[512],
            num_params=5_000_000,
            flops=10_000_000,
            category="general",
        ),
        "factory": "_create_mlp_large",
    },
    "conv_small": {
        "info": BenchmarkModelInfo(
            name="conv_small",
            description="Small ConvNet (3 conv layers)",
            input_shape=[3, 32, 32],
            num_params=50_000,
            flops=10_000_000,
            category="vision",
        ),
        "factory": "_create_conv_small",
    },
    "transformer_encoder": {
        "info": BenchmarkModelInfo(
            name="transformer_encoder",
            description="Transformer Encoder (6 layers, 512 dim)",
            input_shape=[128, 512],
            num_params=20_000_000,
            flops=5_000_000_000,
            category="nlp",
        ),
        "factory": "_create_transformer_encoder",
    },
}


def list_benchmark_models(category: Optional[str] = None) -> List[BenchmarkModelInfo]:
    """List available benchmark models.

    Args:
        category: Filter by category (e.g., "vision", "nlp")

    Returns:
        List of model information
    """
    models = []
    for model_data in _BENCHMARK_MODELS.values():
        info = model_data["info"]
        if category is None or info.category == category:
            models.append(info)
    return models


def get_benchmark_model(
    name: str,
    num_classes: int = 1000,
    **kwargs,
) -> Any:
    """Get a benchmark model by name.

    Args:
        name: Model name
        num_classes: Number of output classes
        **kwargs: Additional model arguments

    Returns:
        Instantiated model

    Example:
        >>> model = get_benchmark_model("resnet50")
        >>> model = get_benchmark_model("bert_base", num_classes=2)
    """
    if name not in _BENCHMARK_MODELS:
        available = ", ".join(_BENCHMARK_MODELS.keys())
        raise ValueError(f"Unknown benchmark model: {name}. Available: {available}")

    factory_name = _BENCHMARK_MODELS[name]["factory"]
    factory = globals().get(factory_name)

    if factory is None:
        raise RuntimeError(f"Factory {factory_name} not found for model {name}")

    return factory(num_classes=num_classes, **kwargs)


def get_benchmark_model_info(name: str) -> BenchmarkModelInfo:
    """Get information about a benchmark model.

    Args:
        name: Model name

    Returns:
        Model information
    """
    if name not in _BENCHMARK_MODELS:
        raise ValueError(f"Unknown benchmark model: {name}")

    return _BENCHMARK_MODELS[name]["info"]


# Model factory functions
def _create_resnet18(num_classes: int = 1000, **kwargs):
    """Create ResNet-18 model."""
    try:
        from pyflame.models import resnet18

        return resnet18(num_classes=num_classes, **kwargs)
    except ImportError:
        return _create_mock_model("resnet18", [3, 224, 224], num_classes)


def _create_resnet50(num_classes: int = 1000, **kwargs):
    """Create ResNet-50 model."""
    try:
        from pyflame.models import resnet50

        return resnet50(num_classes=num_classes, **kwargs)
    except ImportError:
        return _create_mock_model("resnet50", [3, 224, 224], num_classes)


def _create_bert_base(num_classes: int = 2, **kwargs):
    """Create BERT Base model."""
    try:
        from pyflame.models import BertConfig, BertForSequenceClassification

        config = BertConfig(vocab_size=30522, num_classes=num_classes)
        return BertForSequenceClassification(config)
    except ImportError:
        return _create_mock_model("bert_base", [512], num_classes)


def _create_mlp_small(num_classes: int = 10, **kwargs):
    """Create small MLP model."""
    try:
        import pyflame.nn as nn

        return nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )
    except ImportError:
        return _create_mock_model("mlp_small", [128], num_classes)


def _create_mlp_large(num_classes: int = 10, **kwargs):
    """Create large MLP model."""
    try:
        import pyflame.nn as nn

        return nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes),
        )
    except ImportError:
        return _create_mock_model("mlp_large", [512], num_classes)


def _create_conv_small(num_classes: int = 10, **kwargs):
    """Create small ConvNet model."""
    try:
        import pyflame.nn as nn

        return nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, num_classes),
        )
    except ImportError:
        return _create_mock_model("conv_small", [3, 32, 32], num_classes)


def _create_transformer_encoder(num_classes: int = 10, **kwargs):
    """Create Transformer Encoder model."""
    try:
        import pyflame.nn as nn
        from pyflame.models import TransformerEncoder, TransformerEncoderLayer

        encoder_layer = TransformerEncoderLayer(d_model=512, nhead=8)
        encoder = TransformerEncoder(encoder_layer, num_layers=6)

        class TransformerClassifier(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = encoder
                self.classifier = nn.Linear(512, num_classes)

            def forward(self, x):
                x = self.encoder(x)
                x = x.mean(dim=1)  # Global average pooling
                return self.classifier(x)

        return TransformerClassifier()
    except ImportError:
        return _create_mock_model("transformer_encoder", [128, 512], num_classes)


def _create_mock_model(name: str, input_shape: List[int], num_classes: int):
    """Create a mock model for when PyFlame modules are not available.

    This is useful for testing the benchmark infrastructure.
    """
    import numpy as np

    class MockModel:
        def __init__(self):
            self.name = name
            self.input_shape = input_shape
            self.num_classes = num_classes

        def __call__(self, x):
            # Return random output with correct shape
            if hasattr(x, "shape"):
                batch_size = x.shape[0]
            else:
                batch_size = 1
            return np.random.randn(batch_size, self.num_classes).astype(np.float32)

        def eval(self):
            pass

        def train(self):
            pass

    return MockModel()


def register_benchmark_model(
    name: str,
    factory: callable,
    info: BenchmarkModelInfo,
):
    """Register a custom benchmark model.

    Args:
        name: Model name
        factory: Factory function to create model
        info: Model information

    Example:
        >>> def my_factory(num_classes=10):
        ...     return MyModel(num_classes)
        >>> register_benchmark_model(
        ...     "my_model",
        ...     my_factory,
        ...     BenchmarkModelInfo(
        ...         name="my_model",
        ...         description="My custom model",
        ...         input_shape=[3, 224, 224],
        ...     )
        ... )
    """
    # Register factory function
    factory_name = f"_create_{name}"
    globals()[factory_name] = factory

    # Register model info
    _BENCHMARK_MODELS[name] = {
        "info": info,
        "factory": factory_name,
    }

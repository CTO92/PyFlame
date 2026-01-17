"""
Model Registry for PyFlame.

Provides a centralized registry for model architectures.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union


@dataclass
class ModelInfo:
    """Information about a registered model."""

    name: str
    constructor: Callable
    description: str = ""
    paper: str = ""
    paper_url: str = ""
    default_config: Dict[str, Any] = field(default_factory=dict)
    pretrained_weights: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)


class ModelRegistry:
    """
    Registry for model architectures.

    Allows registering, discovering, and instantiating models by name.

    Example:
        >>> registry = ModelRegistry()
        >>> @registry.register("my_model")
        ... def create_my_model(**kwargs):
        ...     return MyModel(**kwargs)
        >>>
        >>> model = registry.get("my_model", num_classes=10)
    """

    def __init__(self):
        self._models: Dict[str, ModelInfo] = {}
        self._aliases: Dict[str, str] = {}

    def register(
        self,
        name: str,
        description: str = "",
        paper: str = "",
        paper_url: str = "",
        default_config: Optional[Dict[str, Any]] = None,
        pretrained_weights: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        aliases: Optional[List[str]] = None,
    ) -> Callable:
        """
        Decorator to register a model constructor.

        Args:
            name: Unique name for the model.
            description: Human-readable description.
            paper: Paper title.
            paper_url: URL to the paper.
            default_config: Default configuration values.
            pretrained_weights: Available pretrained weight names.
            tags: Tags for categorization.
            aliases: Alternative names for the model.

        Returns:
            Decorator function.

        Example:
            >>> @registry.register("resnet50", tags=["vision", "classification"])
            ... def resnet50(**kwargs):
            ...     return ResNet50(**kwargs)
        """

        def decorator(constructor: Callable) -> Callable:
            info = ModelInfo(
                name=name,
                constructor=constructor,
                description=description,
                paper=paper,
                paper_url=paper_url,
                default_config=default_config or {},
                pretrained_weights=pretrained_weights or [],
                tags=tags or [],
            )
            self._models[name] = info

            # Register aliases
            if aliases:
                for alias in aliases:
                    self._aliases[alias] = name

            return constructor

        return decorator

    def get(self, name: str, pretrained: Union[bool, str] = False, **kwargs) -> Any:
        """
        Get a model instance by name.

        Args:
            name: Model name or alias.
            pretrained: If True, load default weights. If string, load specific weights.
            **kwargs: Arguments to pass to model constructor.

        Returns:
            Model instance.

        Example:
            >>> model = registry.get("resnet50", pretrained=True, num_classes=100)
        """
        # Resolve alias (handle chained aliases with cycle detection)
        visited = set()
        while name in self._aliases:
            if name in visited:
                raise ValueError(f"Circular alias detected for '{name}'")
            visited.add(name)
            name = self._aliases[name]

        if name not in self._models:
            available = ", ".join(self.list())
            raise KeyError(f"Model '{name}' not found. Available: {available}")

        info = self._models[name]

        # Merge default config with provided kwargs
        config = {**info.default_config, **kwargs}

        # Create model
        model = info.constructor(**config)

        # Load pretrained weights if requested
        if pretrained:
            weight_name = pretrained if isinstance(pretrained, str) else "default"
            self._load_pretrained(model, name, weight_name)

        return model

    def _load_pretrained(self, model: Any, model_name: str, weight_name: str) -> None:
        """Load pretrained weights into model."""
        from .pretrained import load_pretrained

        load_pretrained(model, model_name, weight_name)

    def list(self, tag: Optional[str] = None) -> List[str]:
        """
        List all registered models.

        Args:
            tag: If provided, filter by tag.

        Returns:
            List of model names.
        """
        if tag is None:
            return sorted(self._models.keys())

        return sorted(name for name, info in self._models.items() if tag in info.tags)

    def info(self, name: str) -> ModelInfo:
        """
        Get information about a model.

        Args:
            name: Model name or alias.

        Returns:
            ModelInfo object.
        """
        if name in self._aliases:
            name = self._aliases[name]

        if name not in self._models:
            raise KeyError(f"Model '{name}' not found")

        return self._models[name]

    def search(self, query: str) -> List[str]:
        """
        Search for models by name or description.

        Args:
            query: Search query.

        Returns:
            List of matching model names.
        """
        query = query.lower()
        results = []

        for name, info in self._models.items():
            if query in name.lower() or query in info.description.lower():
                results.append(name)

        return sorted(results)


# Global registry instance
_global_registry = ModelRegistry()


def register_model(
    name: str,
    description: str = "",
    paper: str = "",
    paper_url: str = "",
    default_config: Optional[Dict[str, Any]] = None,
    pretrained_weights: Optional[List[str]] = None,
    tags: Optional[List[str]] = None,
    aliases: Optional[List[str]] = None,
) -> Callable:
    """
    Decorator to register a model in the global registry.

    Example:
        >>> @register_model("my_vit", tags=["vision", "transformer"])
        ... def create_vit(**kwargs):
        ...     return ViT(**kwargs)
    """
    return _global_registry.register(
        name=name,
        description=description,
        paper=paper,
        paper_url=paper_url,
        default_config=default_config,
        pretrained_weights=pretrained_weights,
        tags=tags,
        aliases=aliases,
    )


def get_model(name: str, pretrained: Union[bool, str] = False, **kwargs) -> Any:
    """
    Get a model from the global registry.

    Args:
        name: Model name.
        pretrained: Whether to load pretrained weights.
        **kwargs: Model configuration.

    Returns:
        Model instance.

    Example:
        >>> model = get_model("resnet50", pretrained=True, num_classes=10)
    """
    return _global_registry.get(name, pretrained=pretrained, **kwargs)


def list_models(tag: Optional[str] = None) -> List[str]:
    """
    List all registered models.

    Args:
        tag: Optional tag to filter by.

    Returns:
        List of model names.

    Example:
        >>> vision_models = list_models(tag="vision")
    """
    return _global_registry.list(tag=tag)


def model_info(name: str) -> ModelInfo:
    """
    Get information about a registered model.

    Args:
        name: Model name.

    Returns:
        ModelInfo object with model details.

    Example:
        >>> info = model_info("resnet50")
        >>> print(info.description)
    """
    return _global_registry.info(name)


# =============================================================================
# Register Built-in Models
# =============================================================================


@register_model(
    name="resnet18",
    description="ResNet-18 model for image classification",
    paper="Deep Residual Learning for Image Recognition",
    paper_url="https://arxiv.org/abs/1512.03385",
    default_config={"num_classes": 1000},
    pretrained_weights=["imagenet1k-v1", "imagenet1k-v2"],
    tags=["vision", "classification", "resnet"],
)
def _resnet18(**kwargs):
    # Placeholder - would import from models module
    pass


@register_model(
    name="resnet50",
    description="ResNet-50 model for image classification",
    paper="Deep Residual Learning for Image Recognition",
    paper_url="https://arxiv.org/abs/1512.03385",
    default_config={"num_classes": 1000},
    pretrained_weights=["imagenet1k-v1", "imagenet1k-v2"],
    tags=["vision", "classification", "resnet"],
)
def _resnet50(**kwargs):
    pass


@register_model(
    name="bert-base-uncased",
    description="BERT base model (uncased)",
    paper="BERT: Pre-training of Deep Bidirectional Transformers",
    paper_url="https://arxiv.org/abs/1810.04805",
    default_config={
        "vocab_size": 30522,
        "hidden_size": 768,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
    },
    pretrained_weights=["default"],
    tags=["nlp", "transformer", "bert"],
    aliases=["bert-base", "bert"],
)
def _bert_base(**kwargs):
    pass


@register_model(
    name="vit-base-patch16-224",
    description="Vision Transformer base with patch size 16 and 224x224 input",
    paper="An Image is Worth 16x16 Words",
    paper_url="https://arxiv.org/abs/2010.11929",
    default_config={
        "image_size": 224,
        "patch_size": 16,
        "num_classes": 1000,
    },
    pretrained_weights=["imagenet1k", "imagenet21k"],
    tags=["vision", "transformer", "vit"],
    aliases=["vit-base", "vit"],
)
def _vit_base(**kwargs):
    pass


@register_model(
    name="gpt2",
    description="GPT-2 language model",
    paper="Language Models are Unsupervised Multitask Learners",
    paper_url="https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf",
    default_config={
        "vocab_size": 50257,
        "n_positions": 1024,
        "n_embd": 768,
        "n_layer": 12,
        "n_head": 12,
    },
    pretrained_weights=["default"],
    tags=["nlp", "transformer", "gpt", "generative"],
)
def _gpt2(**kwargs):
    pass

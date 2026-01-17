"""
Base classes for PyFlame metrics.

Provides the Metric base class and MetricCollection.
"""

import copy
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union


class Metric(ABC):
    """
    Base class for all metrics.

    Metrics maintain internal state for computing values over multiple batches.
    Call `update()` to add new predictions/targets, and `compute()` to get the result.
    Call `reset()` to clear the state.

    Example:
        >>> accuracy = Accuracy()
        >>> for preds, targets in dataloader:
        ...     accuracy.update(preds, targets)
        >>> result = accuracy.compute()
        >>> accuracy.reset()
    """

    def __init__(self):
        self._state: Dict[str, Any] = {}
        self._defaults: Dict[str, Any] = {}

    def add_state(
        self,
        name: str,
        default: Any,
        dist_reduce_fx: Optional[str] = None,
    ) -> None:
        """
        Add a state variable to the metric.

        Args:
            name: Name of the state variable.
            default: Default value for the state.
            dist_reduce_fx: Reduction function for distributed ("sum", "mean", "cat").
        """
        self._defaults[name] = default
        self._state[name] = copy.deepcopy(default)

    @abstractmethod
    def update(self, *args, **kwargs) -> None:
        """
        Update the metric state with new predictions and targets.

        Override this method in subclasses.
        """
        raise NotImplementedError

    @abstractmethod
    def compute(self) -> Any:
        """
        Compute the metric value from the current state.

        Override this method in subclasses.
        """
        raise NotImplementedError

    def reset(self) -> None:
        """Reset the metric state to defaults."""
        for name, default in self._defaults.items():
            self._state[name] = copy.deepcopy(default)

    def __call__(self, *args, **kwargs) -> Any:
        """
        Update and compute the metric in one call.

        For accumulating over batches, use update() separately.
        """
        self.update(*args, **kwargs)
        return self.compute()

    def clone(self) -> "Metric":
        """Create a copy of this metric."""
        return copy.deepcopy(self)

    def to(self, device: Any) -> "Metric":
        """Move metric states to device (placeholder for tensor-backed states)."""
        return self

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class MetricCollection(Metric):
    """
    Collection of metrics to compute together.

    Args:
        metrics: Dict or list of metrics.
        prefix: Optional prefix for metric names.
        postfix: Optional postfix for metric names.

    Example:
        >>> metrics = MetricCollection({
        ...     "accuracy": Accuracy(),
        ...     "f1": F1Score(),
        ... })
        >>> metrics.update(preds, targets)
        >>> results = metrics.compute()  # {"accuracy": 0.95, "f1": 0.93}
    """

    def __init__(
        self,
        metrics: Union[Dict[str, Metric], List[Metric]],
        prefix: Optional[str] = None,
        postfix: Optional[str] = None,
    ):
        super().__init__()

        if isinstance(metrics, list):
            # Convert list to dict using class names
            metrics = {m.__class__.__name__.lower(): m for m in metrics}

        self.metrics = metrics
        self.prefix = prefix or ""
        self.postfix = postfix or ""

    def update(self, *args, **kwargs) -> None:
        """Update all metrics."""
        for metric in self.metrics.values():
            metric.update(*args, **kwargs)

    def compute(self) -> Dict[str, Any]:
        """Compute all metrics and return as dict."""
        results = {}
        for name, metric in self.metrics.items():
            key = f"{self.prefix}{name}{self.postfix}"
            results[key] = metric.compute()
        return results

    def reset(self) -> None:
        """Reset all metrics."""
        for metric in self.metrics.values():
            metric.reset()

    def __getitem__(self, key: str) -> Metric:
        return self.metrics[key]

    def __iter__(self):
        return iter(self.metrics)

    def keys(self):
        return self.metrics.keys()

    def values(self):
        return self.metrics.values()

    def items(self):
        return self.metrics.items()

    def clone(self) -> "MetricCollection":
        """Create a copy of this collection."""
        return MetricCollection(
            {name: metric.clone() for name, metric in self.metrics.items()},
            prefix=self.prefix,
            postfix=self.postfix,
        )


class StatScores(Metric):
    """
    Base class for metrics that compute TP, TN, FP, FN.
    """

    def __init__(
        self,
        num_classes: Optional[int] = None,
        threshold: float = 0.5,
        average: str = "micro",  # "micro", "macro", "weighted", "none"
    ):
        super().__init__()
        self.num_classes = num_classes
        self.threshold = threshold
        self.average = average

        self.add_state("tp", 0)
        self.add_state("fp", 0)
        self.add_state("tn", 0)
        self.add_state("fn", 0)

        if num_classes and average != "micro":
            self.add_state("tp_per_class", [0] * num_classes)
            self.add_state("fp_per_class", [0] * num_classes)
            self.add_state("fn_per_class", [0] * num_classes)

    def _to_numpy(self, x: Any):
        """Convert tensor to numpy array."""
        if hasattr(x, "numpy"):
            return x.numpy()
        elif hasattr(x, "cpu"):
            return x.cpu().numpy()
        return x

    def _compute_stats(self, preds, target):
        """Compute confusion matrix statistics."""
        preds = self._to_numpy(preds)
        target = self._to_numpy(target)

        # Flatten if needed
        preds = preds.flatten()
        target = target.flatten()

        # Apply threshold for probabilities
        import numpy as np

        if np.issubdtype(preds.dtype, np.floating):
            preds = (preds >= self.threshold).astype(int)

        # Binary case
        tp = ((preds == 1) & (target == 1)).sum()
        fp = ((preds == 1) & (target == 0)).sum()
        tn = ((preds == 0) & (target == 0)).sum()
        fn = ((preds == 0) & (target == 1)).sum()

        return int(tp), int(fp), int(tn), int(fn)

    def update(self, preds, target) -> None:
        tp, fp, tn, fn = self._compute_stats(preds, target)
        self._state["tp"] += tp
        self._state["fp"] += fp
        self._state["tn"] += tn
        self._state["fn"] += fn

    def compute(self):
        raise NotImplementedError("Subclasses must implement compute()")

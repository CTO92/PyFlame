"""
PyFlame Metrics Module

Provides evaluation metrics for machine learning:
- Classification metrics (accuracy, precision, recall, F1)
- Regression metrics (MSE, MAE, R2)
- Ranking metrics (AUC, AP)
- NLP metrics (BLEU, perplexity)
"""

from .base import Metric, MetricCollection
from .classification import (
    AUROC,
    Accuracy,
    AveragePrecision,
    ConfusionMatrix,
    F1Score,
    Precision,
    Recall,
)
from .regression import (
    MeanAbsoluteError,
    MeanSquaredError,
    R2Score,
    RootMeanSquaredError,
)

__all__ = [
    # Base
    "Metric",
    "MetricCollection",
    # Classification
    "Accuracy",
    "Precision",
    "Recall",
    "F1Score",
    "ConfusionMatrix",
    "AUROC",
    "AveragePrecision",
    # Regression
    "MeanSquaredError",
    "MeanAbsoluteError",
    "RootMeanSquaredError",
    "R2Score",
]

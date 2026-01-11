"""
PyFlame Metrics Module

Provides evaluation metrics for machine learning:
- Classification metrics (accuracy, precision, recall, F1)
- Regression metrics (MSE, MAE, R2)
- Ranking metrics (AUC, AP)
- NLP metrics (BLEU, perplexity)
"""

from .classification import (
    Accuracy,
    Precision,
    Recall,
    F1Score,
    ConfusionMatrix,
    AUROC,
    AveragePrecision,
)
from .regression import (
    MeanSquaredError,
    MeanAbsoluteError,
    RootMeanSquaredError,
    R2Score,
)
from .base import Metric, MetricCollection

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

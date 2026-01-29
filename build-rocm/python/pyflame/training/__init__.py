"""
PyFlame Training Module

Provides training utilities:
- Trainer class for training loops
- Callbacks for custom behavior
- Learning rate schedulers
- Early stopping
"""

from .callbacks import (
    Callback,
    CallbackList,
    CSVLogger,
    EarlyStopping,
    LearningRateScheduler,
    ModelCheckpoint,
    ProgressBar,
    TensorBoardLogger,
)
from .trainer import Trainer, TrainerConfig

__all__ = [
    "Trainer",
    "TrainerConfig",
    "Callback",
    "CallbackList",
    "EarlyStopping",
    "ModelCheckpoint",
    "LearningRateScheduler",
    "ProgressBar",
    "TensorBoardLogger",
    "CSVLogger",
]

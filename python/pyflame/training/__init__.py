"""
PyFlame Training Module

Provides training utilities:
- Trainer class for training loops
- Callbacks for custom behavior
- Learning rate schedulers
- Early stopping
"""

from .trainer import Trainer, TrainerConfig
from .callbacks import (
    Callback,
    CallbackList,
    EarlyStopping,
    ModelCheckpoint,
    LearningRateScheduler,
    ProgressBar,
    TensorBoardLogger,
    CSVLogger,
)

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

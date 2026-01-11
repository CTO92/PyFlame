"""
Callbacks for PyFlame Trainer.

Provides hooks into the training loop for custom behavior.
"""

from abc import ABC
from typing import Any, Callable, Dict, List, Optional, Union
import os
import time
import json


class Callback(ABC):
    """
    Base class for all callbacks.

    Callbacks allow customizing behavior at various points during training.
    Override methods to add custom functionality.
    """

    def on_fit_start(self, trainer: Any) -> None:
        """Called when fit begins."""
        pass

    def on_fit_end(self, trainer: Any) -> None:
        """Called when fit ends."""
        pass

    def on_epoch_start(self, trainer: Any) -> None:
        """Called at the start of each epoch."""
        pass

    def on_epoch_end(self, trainer: Any) -> None:
        """Called at the end of each epoch."""
        pass

    def on_train_epoch_start(self, trainer: Any) -> None:
        """Called at the start of training epoch."""
        pass

    def on_train_epoch_end(self, trainer: Any) -> None:
        """Called at the end of training epoch."""
        pass

    def on_train_batch_start(self, trainer: Any, batch: Any, batch_idx: int) -> None:
        """Called before processing a training batch."""
        pass

    def on_train_batch_end(self, trainer: Any, batch: Any, batch_idx: int) -> None:
        """Called after processing a training batch."""
        pass

    def on_validation_start(self, trainer: Any) -> None:
        """Called when validation begins."""
        pass

    def on_validation_end(self, trainer: Any) -> None:
        """Called when validation ends."""
        pass

    def on_validation_batch_start(self, trainer: Any, batch: Any, batch_idx: int) -> None:
        """Called before processing a validation batch."""
        pass

    def on_validation_batch_end(self, trainer: Any, batch: Any, batch_idx: int) -> None:
        """Called after processing a validation batch."""
        pass


class CallbackList:
    """Container for multiple callbacks."""

    def __init__(self, callbacks: Optional[List[Callback]] = None):
        self.callbacks = callbacks or []

    def append(self, callback: Callback) -> None:
        self.callbacks.append(callback)

    def __iter__(self):
        return iter(self.callbacks)

    # Delegate all callback methods
    def on_fit_start(self, trainer: Any) -> None:
        for cb in self.callbacks:
            cb.on_fit_start(trainer)

    def on_fit_end(self, trainer: Any) -> None:
        for cb in self.callbacks:
            cb.on_fit_end(trainer)

    def on_epoch_start(self, trainer: Any) -> None:
        for cb in self.callbacks:
            cb.on_epoch_start(trainer)

    def on_epoch_end(self, trainer: Any) -> None:
        for cb in self.callbacks:
            cb.on_epoch_end(trainer)

    def on_train_epoch_start(self, trainer: Any) -> None:
        for cb in self.callbacks:
            cb.on_train_epoch_start(trainer)

    def on_train_epoch_end(self, trainer: Any) -> None:
        for cb in self.callbacks:
            cb.on_train_epoch_end(trainer)

    def on_train_batch_start(self, trainer: Any, batch: Any, batch_idx: int) -> None:
        for cb in self.callbacks:
            cb.on_train_batch_start(trainer, batch, batch_idx)

    def on_train_batch_end(self, trainer: Any, batch: Any, batch_idx: int) -> None:
        for cb in self.callbacks:
            cb.on_train_batch_end(trainer, batch, batch_idx)

    def on_validation_start(self, trainer: Any) -> None:
        for cb in self.callbacks:
            cb.on_validation_start(trainer)

    def on_validation_end(self, trainer: Any) -> None:
        for cb in self.callbacks:
            cb.on_validation_end(trainer)

    def on_validation_batch_start(self, trainer: Any, batch: Any, batch_idx: int) -> None:
        for cb in self.callbacks:
            cb.on_validation_batch_start(trainer, batch, batch_idx)

    def on_validation_batch_end(self, trainer: Any, batch: Any, batch_idx: int) -> None:
        for cb in self.callbacks:
            cb.on_validation_batch_end(trainer, batch, batch_idx)


class EarlyStopping(Callback):
    """
    Stop training when a metric stops improving.

    Args:
        monitor: Metric to monitor (e.g., "val_loss").
        patience: Number of epochs with no improvement before stopping.
        min_delta: Minimum change to qualify as an improvement.
        mode: One of "min" or "max".
        verbose: Whether to print messages.

    Example:
        >>> early_stop = EarlyStopping(
        ...     monitor="val_loss",
        ...     patience=3,
        ...     mode="min",
        ... )
        >>> trainer = Trainer(callbacks=[early_stop])
    """

    def __init__(
        self,
        monitor: str = "val_loss",
        patience: int = 3,
        min_delta: float = 0.0,
        mode: str = "min",
        verbose: bool = True,
    ):
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose

        self.wait = 0
        self.best_value = float("inf") if mode == "min" else float("-inf")
        self.stopped_epoch = 0

    def _is_improvement(self, current: float) -> bool:
        if self.mode == "min":
            return current < self.best_value - self.min_delta
        else:
            return current > self.best_value + self.min_delta

    def on_epoch_end(self, trainer: Any) -> None:
        current = trainer.state.logs.get(self.monitor)

        if current is None:
            return

        if self._is_improvement(current):
            self.best_value = current
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                trainer.state.should_stop = True
                self.stopped_epoch = trainer.state.epoch
                if self.verbose:
                    print(
                        f"\nEarly stopping triggered at epoch {self.stopped_epoch}. "
                        f"Best {self.monitor}: {self.best_value:.6f}"
                    )


class ModelCheckpoint(Callback):
    """
    Save model checkpoints during training.

    Args:
        dirpath: Directory to save checkpoints.
        filename: Checkpoint filename template.
        monitor: Metric to monitor for best model.
        mode: One of "min" or "max".
        save_top_k: Number of best models to keep.
        save_last: Whether to save a "last.pt" checkpoint.
        verbose: Whether to print messages.

    Example:
        >>> checkpoint = ModelCheckpoint(
        ...     dirpath="./checkpoints",
        ...     monitor="val_loss",
        ...     save_top_k=3,
        ... )
    """

    def __init__(
        self,
        dirpath: str = "./checkpoints",
        filename: str = "model-{epoch:02d}-{val_loss:.4f}",
        monitor: str = "val_loss",
        mode: str = "min",
        save_top_k: int = 3,
        save_last: bool = True,
        verbose: bool = True,
    ):
        self.dirpath = dirpath
        self.filename = filename
        self.monitor = monitor
        self.mode = mode
        self.save_top_k = save_top_k
        self.save_last = save_last
        self.verbose = verbose

        self.best_k_models: List[tuple] = []  # (score, path)

        os.makedirs(dirpath, exist_ok=True)

    def _is_better(self, current: float, best: float) -> bool:
        if self.mode == "min":
            return current < best
        return current > best

    def on_epoch_end(self, trainer: Any) -> None:
        current = trainer.state.logs.get(self.monitor)

        # Generate filename
        filepath = os.path.join(
            self.dirpath,
            self._format_filename(trainer)
        )

        # Save last checkpoint
        if self.save_last:
            last_path = os.path.join(self.dirpath, "last.pt")
            trainer.save_checkpoint(last_path)

        if current is None:
            return

        # Check if this is a top-k model
        should_save = False

        if len(self.best_k_models) < self.save_top_k:
            should_save = True
        else:
            # Check if better than worst in top-k
            worst_score, worst_path = self.best_k_models[-1]
            if self._is_better(current, worst_score):
                should_save = True
                # Remove worst model
                if os.path.exists(worst_path):
                    os.remove(worst_path)
                self.best_k_models.pop()

        if should_save:
            # Save checkpoint
            trainer.save_checkpoint(filepath)

            # Add to top-k
            self.best_k_models.append((current, filepath))

            # Sort by score
            reverse = self.mode == "max"
            self.best_k_models.sort(key=lambda x: x[0], reverse=reverse)

            # Update best metric
            if len(self.best_k_models) > 0:
                trainer.state.best_metric = self.best_k_models[0][0]
                trainer.state.best_model_path = self.best_k_models[0][1]

            if self.verbose:
                print(f"\nSaved checkpoint: {filepath}")

    def _format_filename(self, trainer: Any) -> str:
        """Format filename with current metrics."""
        # Simple formatting
        filename = self.filename
        filename = filename.replace("{epoch:02d}", f"{trainer.state.epoch:02d}")

        for key, value in trainer.state.logs.items():
            placeholder = "{" + key + ":.4f}"
            if placeholder in filename:
                filename = filename.replace(placeholder, f"{value:.4f}")

        return filename + ".pt"


class LearningRateScheduler(Callback):
    """
    Adjust learning rate during training.

    Args:
        scheduler: LR scheduler instance or callable.
        monitor: Metric to monitor (for ReduceLROnPlateau).
        interval: When to step: "epoch" or "step".

    Example:
        >>> scheduler = pf.optim.lr_scheduler.StepLR(optimizer, step_size=10)
        >>> lr_callback = LearningRateScheduler(scheduler)
    """

    def __init__(
        self,
        scheduler: Any,
        monitor: str = "val_loss",
        interval: str = "epoch",
    ):
        self.scheduler = scheduler
        self.monitor = monitor
        self.interval = interval

    def on_epoch_end(self, trainer: Any) -> None:
        if self.interval != "epoch":
            return

        # Handle ReduceLROnPlateau
        if hasattr(self.scheduler, "step"):
            metric = trainer.state.logs.get(self.monitor)
            if metric is not None:
                try:
                    self.scheduler.step(metric)
                except TypeError:
                    self.scheduler.step()
            else:
                self.scheduler.step()

    def on_train_batch_end(self, trainer: Any, batch: Any, batch_idx: int) -> None:
        if self.interval != "step":
            return

        if hasattr(self.scheduler, "step"):
            self.scheduler.step()


class ProgressBar(Callback):
    """
    Display progress bar during training.

    Args:
        refresh_rate: How often to update the progress bar (in batches).
    """

    def __init__(self, refresh_rate: int = 1):
        self.refresh_rate = refresh_rate
        self.epoch_start_time = None
        self.train_batch_count = 0

    def on_epoch_start(self, trainer: Any) -> None:
        self.epoch_start_time = time.time()
        self.train_batch_count = 0
        print(f"\nEpoch {trainer.state.epoch + 1}/{trainer.config.max_epochs}")

    def on_train_batch_end(self, trainer: Any, batch: Any, batch_idx: int) -> None:
        self.train_batch_count += 1

        if batch_idx % self.refresh_rate == 0:
            # Calculate progress
            try:
                total_batches = len(trainer.train_loader)
                progress = (batch_idx + 1) / total_batches
                bar_length = 30
                filled = int(bar_length * progress)
                bar = "=" * filled + ">" + "." * (bar_length - filled - 1)

                # Get current loss
                loss = trainer.state.logs.get("train_loss", 0.0)

                # Print progress
                print(
                    f"\r[{bar}] {batch_idx + 1}/{total_batches} - "
                    f"loss: {loss:.4f}",
                    end=""
                )
            except TypeError:
                # If len() not supported
                print(f"\rBatch {batch_idx + 1} - loss: {trainer.state.logs.get('train_loss', 0.0):.4f}", end="")

    def on_epoch_end(self, trainer: Any) -> None:
        if self.epoch_start_time:
            elapsed = time.time() - self.epoch_start_time
            metrics_str = " - ".join(
                f"{k}: {v:.4f}" for k, v in trainer.state.logs.items()
                if isinstance(v, (int, float))
            )
            print(f"\nEpoch completed in {elapsed:.1f}s - {metrics_str}")


class TensorBoardLogger(Callback):
    """
    Log metrics to TensorBoard.

    Args:
        log_dir: Directory for TensorBoard logs.
        name: Experiment name.
        log_graph: Whether to log the model graph.
    """

    def __init__(
        self,
        log_dir: str = "./logs",
        name: str = "experiment",
        log_graph: bool = False,
    ):
        self.log_dir = os.path.join(log_dir, name)
        self.log_graph = log_graph
        self.writer = None

    def on_fit_start(self, trainer: Any) -> None:
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(self.log_dir)
        except ImportError:
            print("TensorBoard not available. Install with: pip install tensorboard")
            self.writer = None

    def on_train_batch_end(self, trainer: Any, batch: Any, batch_idx: int) -> None:
        if self.writer is None:
            return

        step = trainer.state.global_step
        if "train_loss" in trainer.state.logs:
            self.writer.add_scalar("Loss/train", trainer.state.logs["train_loss"], step)

    def on_epoch_end(self, trainer: Any) -> None:
        if self.writer is None:
            return

        epoch = trainer.state.epoch

        for key, value in trainer.state.logs.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(f"Epoch/{key}", value, epoch)

    def on_fit_end(self, trainer: Any) -> None:
        if self.writer:
            self.writer.close()


class CSVLogger(Callback):
    """
    Log metrics to CSV file.

    Args:
        save_dir: Directory to save CSV file.
        name: Filename (without extension).
        version: Version string for the experiment.
    """

    def __init__(
        self,
        save_dir: str = "./logs",
        name: str = "metrics",
        version: Optional[str] = None,
    ):
        self.save_dir = save_dir
        self.name = name
        self.version = version or ""
        self.filepath = os.path.join(save_dir, f"{name}_{version}.csv" if version else f"{name}.csv")
        self.metrics_history: List[Dict] = []

        os.makedirs(save_dir, exist_ok=True)

    def on_epoch_end(self, trainer: Any) -> None:
        # Collect metrics
        row = {
            "epoch": trainer.state.epoch,
            "global_step": trainer.state.global_step,
        }
        row.update(trainer.state.logs)
        self.metrics_history.append(row)

        # Write to CSV
        self._save_csv()

    def _save_csv(self) -> None:
        if not self.metrics_history:
            return

        # Get all keys
        keys = set()
        for row in self.metrics_history:
            keys.update(row.keys())
        keys = sorted(keys)

        # Write CSV
        with open(self.filepath, "w") as f:
            f.write(",".join(keys) + "\n")
            for row in self.metrics_history:
                values = [str(row.get(k, "")) for k in keys]
                f.write(",".join(values) + "\n")


class GradientAccumulationScheduler(Callback):
    """
    Change gradient accumulation factor during training.

    Args:
        scheduling: Dict mapping epochs to accumulation factors.

    Example:
        >>> # Double accumulation after epoch 5
        >>> scheduler = GradientAccumulationScheduler({5: 2, 10: 4})
    """

    def __init__(self, scheduling: Dict[int, int]):
        self.scheduling = scheduling

    def on_epoch_start(self, trainer: Any) -> None:
        epoch = trainer.state.epoch
        if epoch in self.scheduling:
            trainer.config.gradient_accumulation_steps = self.scheduling[epoch]


class StochasticWeightAveraging(Callback):
    """
    Stochastic Weight Averaging callback.

    Averages weights over the last portion of training for better generalization.

    Args:
        swa_start: Epoch to start SWA.
        swa_lr: Learning rate for SWA phase.
        anneal_epochs: Number of epochs to anneal to SWA LR.
    """

    def __init__(
        self,
        swa_start: int = 10,
        swa_lr: float = 0.05,
        anneal_epochs: int = 5,
    ):
        self.swa_start = swa_start
        self.swa_lr = swa_lr
        self.anneal_epochs = anneal_epochs
        self.swa_model = None
        self.n_averaged = 0

    def on_epoch_end(self, trainer: Any) -> None:
        if trainer.state.epoch < self.swa_start:
            return

        # Initialize SWA model
        if self.swa_model is None:
            # Deep copy model weights
            if hasattr(trainer.model, "state_dict"):
                import copy
                self.swa_model = copy.deepcopy(trainer.model.state_dict())
                self.n_averaged = 1
        else:
            # Update running average
            self.n_averaged += 1
            if hasattr(trainer.model, "state_dict"):
                model_state = trainer.model.state_dict()
                for key in self.swa_model:
                    self.swa_model[key] = (
                        self.swa_model[key] * (self.n_averaged - 1) + model_state[key]
                    ) / self.n_averaged

    def on_fit_end(self, trainer: Any) -> None:
        # Apply averaged weights
        if self.swa_model is not None and hasattr(trainer.model, "load_state_dict"):
            trainer.model.load_state_dict(self.swa_model)

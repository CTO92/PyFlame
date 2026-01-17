"""
Trainer class for PyFlame.

Provides a high-level training loop with support for:
- Multi-epoch training
- Validation
- Callbacks
- Gradient accumulation
- Mixed precision training
- Distributed training
"""

import logging
import os
import pickle
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from .callbacks import Callback, CallbackList

logger = logging.getLogger(__name__)


class RestrictedUnpickler(pickle.Unpickler):
    """Restricted unpickler that only allows explicitly whitelisted classes.

    Security Note: This unpickler uses a strict allowlist approach for
    checkpoint loading. Only explicitly listed (module, class) pairs are
    allowed. Module prefix matching is NOT used.
    """

    # Classes that are explicitly allowed (strict allowlist)
    SAFE_CLASSES = frozenset(
        {
            # NumPy core types
            ("numpy", "ndarray"),
            ("numpy", "dtype"),
            ("numpy", "float32"),
            ("numpy", "float64"),
            ("numpy", "float16"),
            ("numpy", "int32"),
            ("numpy", "int64"),
            ("numpy", "int16"),
            ("numpy", "int8"),
            ("numpy", "uint8"),
            ("numpy", "bool_"),
            ("numpy.core.multiarray", "_reconstruct"),
            ("numpy.core.multiarray", "scalar"),
            ("numpy._core.multiarray", "_reconstruct"),
            ("numpy._core.multiarray", "scalar"),
            # Python builtins (data types only)
            ("builtins", "dict"),
            ("builtins", "list"),
            ("builtins", "tuple"),
            ("builtins", "set"),
            ("builtins", "frozenset"),
            ("builtins", "bytes"),
            ("builtins", "bytearray"),
            ("builtins", "str"),
            ("builtins", "int"),
            ("builtins", "float"),
            ("builtins", "bool"),
            ("builtins", "complex"),
            ("builtins", "slice"),
            ("builtins", "range"),
            # Collections
            ("collections", "OrderedDict"),
            ("collections", "defaultdict"),
            # Copy module
            ("copy", "_reconstructor"),
        }
    )

    # Dangerous classes that should NEVER be allowed
    DANGEROUS_CLASSES = frozenset(
        {
            ("builtins", "eval"),
            ("builtins", "exec"),
            ("builtins", "compile"),
            ("builtins", "open"),
            ("builtins", "__import__"),
            ("os", "system"),
            ("os", "popen"),
            ("subprocess", "Popen"),
            ("subprocess", "call"),
            ("subprocess", "run"),
        }
    )

    def find_class(self, module: str, name: str):
        # Check for explicitly dangerous classes first
        if (module, name) in self.DANGEROUS_CLASSES:
            logger.error(
                f"SECURITY ALERT: Blocked dangerous class in checkpoint: {module}.{name}"
            )
            raise pickle.UnpicklingError(
                f"BLOCKED: Dangerous class '{module}.{name}' in checkpoint. "
                f"This checkpoint may be malicious."
            )

        # Check explicit class allowlist (strict - no module prefix matching)
        if (module, name) in self.SAFE_CLASSES:
            return super().find_class(module, name)

        # Block all non-whitelisted classes
        logger.warning(
            f"SECURITY: Blocked non-whitelisted class in checkpoint: {module}.{name}"
        )
        raise pickle.UnpicklingError(
            f"Class '{module}.{name}' is not in the allowed list for checkpoints. "
            f"Re-save the checkpoint using trainer.save_checkpoint() for safe format."
        )


@dataclass
class TrainerConfig:
    """Configuration for Trainer."""

    # Training
    max_epochs: int = 10
    max_steps: Optional[int] = None
    gradient_accumulation_steps: int = 1
    gradient_clip_val: Optional[float] = None
    gradient_clip_algorithm: str = "norm"  # "norm" or "value"

    # Validation
    val_check_interval: Union[int, float] = (
        1.0  # 1.0 = every epoch, 100 = every 100 steps
    )
    limit_val_batches: Optional[int] = None

    # Logging
    log_every_n_steps: int = 50
    enable_progress_bar: bool = True

    # Checkpointing
    save_dir: str = "./checkpoints"
    save_every_n_epochs: Optional[int] = 1
    save_top_k: int = 3

    # Performance
    precision: str = "32"  # "32", "16", "bf16", "16-mixed", "bf16-mixed"
    accelerator: str = "auto"  # "auto", "cpu", "gpu", "cerebras"
    devices: Union[int, str, List[int]] = "auto"

    # Reproducibility
    seed: Optional[int] = None
    deterministic: bool = False

    # Advanced
    detect_anomaly: bool = False
    fast_dev_run: bool = False  # Run 1 batch for testing


@dataclass
class TrainerState:
    """Internal state of the trainer."""

    epoch: int = 0
    global_step: int = 0
    batch_idx: int = 0
    best_metric: float = float("inf")
    best_model_path: Optional[str] = None
    should_stop: bool = False
    current_train_batch: Optional[Any] = None
    current_val_batch: Optional[Any] = None
    logs: Dict[str, Any] = field(default_factory=dict)


class Trainer:
    """
    High-level trainer for PyFlame models.

    Provides a training loop with callbacks, validation, checkpointing,
    and various optimizations.

    Example:
        >>> model = MyModel()
        >>> optimizer = pf.optim.Adam(model.parameters(), lr=1e-4)
        >>> criterion = pf.nn.CrossEntropyLoss()
        >>>
        >>> trainer = Trainer(
        ...     config=TrainerConfig(max_epochs=10),
        ...     callbacks=[EarlyStopping(patience=3)],
        ... )
        >>>
        >>> trainer.fit(
        ...     model=model,
        ...     train_loader=train_loader,
        ...     val_loader=val_loader,
        ...     optimizer=optimizer,
        ...     criterion=criterion,
        ... )
    """

    def __init__(
        self,
        config: Optional[TrainerConfig] = None,
        callbacks: Optional[List[Callback]] = None,
        logger: Optional[Any] = None,
    ):
        self.config = config or TrainerConfig()
        self.callbacks = CallbackList(callbacks or [])
        self.logger = logger
        self.state = TrainerState()

        # Will be set during fit()
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.train_loader = None
        self.val_loader = None
        self.scheduler = None

        # Setup
        self._setup()

    def _setup(self):
        """Initialize trainer components."""
        # Set seed for reproducibility
        if self.config.seed is not None:
            self._set_seed(self.config.seed)

        # Create checkpoint directory
        os.makedirs(self.config.save_dir, exist_ok=True)

    def _set_seed(self, seed: int):
        """Set random seed for reproducibility."""
        import random

        random.seed(seed)

        try:
            import numpy as np

            np.random.seed(seed)
        except ImportError:
            pass

    def fit(
        self,
        model: Any,
        train_loader: Any,
        val_loader: Optional[Any] = None,
        optimizer: Optional[Any] = None,
        criterion: Optional[Any] = None,
        scheduler: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Train the model.

        Args:
            model: Model to train.
            train_loader: DataLoader for training data.
            val_loader: DataLoader for validation data.
            optimizer: Optimizer instance.
            criterion: Loss function.
            scheduler: Learning rate scheduler.

        Returns:
            Dictionary with training results.
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler

        # Reset state
        self.state = TrainerState()

        # Callbacks: on_fit_start
        self.callbacks.on_fit_start(self)

        try:
            self._training_loop()
        except KeyboardInterrupt:
            print("\nTraining interrupted by user.")
        finally:
            # Callbacks: on_fit_end
            self.callbacks.on_fit_end(self)

        return {
            "epochs": self.state.epoch,
            "global_step": self.state.global_step,
            "best_metric": self.state.best_metric,
            "best_model_path": self.state.best_model_path,
        }

    def _training_loop(self):
        """Main training loop."""
        for epoch in range(self.config.max_epochs):
            if self.state.should_stop:
                break

            self.state.epoch = epoch

            # Callbacks: on_epoch_start
            self.callbacks.on_epoch_start(self)

            # Training epoch
            train_metrics = self._train_epoch()

            # Validation
            if self.val_loader is not None:
                val_metrics = self._validate()
                self.state.logs.update(val_metrics)
            else:
                val_metrics = {}

            # Merge metrics
            metrics = {**train_metrics, **val_metrics}
            self.state.logs.update(metrics)

            # Learning rate scheduler step
            if self.scheduler is not None:
                if hasattr(self.scheduler, "step"):
                    if "val_loss" in val_metrics:
                        self.scheduler.step(val_metrics["val_loss"])
                    else:
                        self.scheduler.step()

            # Callbacks: on_epoch_end
            self.callbacks.on_epoch_end(self)

            # Check max steps
            if (
                self.config.max_steps
                and self.state.global_step >= self.config.max_steps
            ):
                break

    def _train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()

        total_loss = 0.0
        num_batches = 0
        epoch_start_time = time.time()

        # Callbacks: on_train_epoch_start
        self.callbacks.on_train_epoch_start(self)

        for batch_idx, batch in enumerate(self.train_loader):
            if self.state.should_stop:
                break

            self.state.batch_idx = batch_idx
            self.state.current_train_batch = batch

            # Callbacks: on_train_batch_start
            self.callbacks.on_train_batch_start(self, batch, batch_idx)

            # Training step
            loss = self._training_step(batch, batch_idx)

            # Accumulate loss
            total_loss += loss
            num_batches += 1

            # Callbacks: on_train_batch_end
            self.callbacks.on_train_batch_end(self, batch, batch_idx)

            # Increment global step
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                self.state.global_step += 1

            # Log every n steps
            if self.state.global_step % self.config.log_every_n_steps == 0:
                self._log_metrics({"train_loss": loss}, step=self.state.global_step)

            # Check max steps
            if (
                self.config.max_steps
                and self.state.global_step >= self.config.max_steps
            ):
                break

            # Fast dev run
            if self.config.fast_dev_run:
                break

        # Callbacks: on_train_epoch_end
        self.callbacks.on_train_epoch_end(self)

        avg_loss = total_loss / max(num_batches, 1)
        epoch_time = time.time() - epoch_start_time

        return {
            "train_loss": avg_loss,
            "train_time": epoch_time,
        }

    def _training_step(self, batch: Any, batch_idx: int) -> float:
        """
        Perform a single training step.

        Override this method for custom training logic.
        """
        # Unpack batch
        if isinstance(batch, (tuple, list)):
            inputs, targets = batch[0], batch[1]
        else:
            inputs, targets = batch, None

        # Forward pass
        outputs = self.model(inputs)

        # Compute loss
        if self.criterion is not None and targets is not None:
            loss = self.criterion(outputs, targets)
        elif hasattr(outputs, "loss"):
            loss = outputs.loss
        else:
            loss = outputs

        # Scale loss for gradient accumulation
        scaled_loss = loss / self.config.gradient_accumulation_steps

        # Backward pass
        if hasattr(scaled_loss, "backward"):
            scaled_loss.backward()

        # Optimizer step (with accumulation)
        if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
            # Gradient clipping
            if self.config.gradient_clip_val is not None:
                self._clip_gradients()

            # Optimizer step
            if self.optimizer is not None:
                self.optimizer.step()
                self.optimizer.zero_grad()

        return float(loss)

    def _clip_gradients(self):
        """Clip gradients according to config."""
        if self.config.gradient_clip_algorithm == "norm":
            # Clip by norm
            if hasattr(self.model, "parameters"):
                params = list(self.model.parameters())
                total_norm = 0.0
                for p in params:
                    if hasattr(p, "grad") and p.grad is not None:
                        param_norm = p.grad.norm()
                        total_norm += param_norm**2
                total_norm = total_norm**0.5

                clip_coef = self.config.gradient_clip_val / (total_norm + 1e-6)
                if clip_coef < 1:
                    for p in params:
                        if hasattr(p, "grad") and p.grad is not None:
                            p.grad *= clip_coef
        else:
            # Clip by value
            if hasattr(self.model, "parameters"):
                for p in self.model.parameters():
                    if hasattr(p, "grad") and p.grad is not None:
                        p.grad.clamp_(
                            -self.config.gradient_clip_val,
                            self.config.gradient_clip_val,
                        )

    def _validate(self) -> Dict[str, float]:
        """Run validation loop."""
        self.model.eval()

        total_loss = 0.0
        num_batches = 0

        # Callbacks: on_validation_start
        self.callbacks.on_validation_start(self)

        for batch_idx, batch in enumerate(self.val_loader):
            self.state.current_val_batch = batch

            # Callbacks: on_validation_batch_start
            self.callbacks.on_validation_batch_start(self, batch, batch_idx)

            # Validation step
            loss = self._validation_step(batch, batch_idx)

            total_loss += loss
            num_batches += 1

            # Callbacks: on_validation_batch_end
            self.callbacks.on_validation_batch_end(self, batch, batch_idx)

            # Limit validation batches
            if (
                self.config.limit_val_batches
                and batch_idx >= self.config.limit_val_batches
            ):
                break

            # Fast dev run
            if self.config.fast_dev_run:
                break

        # Callbacks: on_validation_end
        self.callbacks.on_validation_end(self)

        avg_loss = total_loss / max(num_batches, 1)

        return {"val_loss": avg_loss}

    def _validation_step(self, batch: Any, batch_idx: int) -> float:
        """
        Perform a single validation step.

        Override this method for custom validation logic.
        """
        # Unpack batch
        if isinstance(batch, (tuple, list)):
            inputs, targets = batch[0], batch[1]
        else:
            inputs, targets = batch, None

        # Forward pass (no gradients)
        outputs = self.model(inputs)

        # Compute loss
        if self.criterion is not None and targets is not None:
            loss = self.criterion(outputs, targets)
        elif hasattr(outputs, "loss"):
            loss = outputs.loss
        else:
            loss = outputs

        return float(loss)

    def _log_metrics(self, metrics: Dict[str, Any], step: int):
        """Log metrics."""
        if self.logger is not None:
            self.logger.log_metrics(metrics, step=step)

    def save_checkpoint(self, path: Optional[str] = None) -> str:
        """Save a checkpoint using safe serialization.

        Uses JSON for metadata and NumPy's .npz format for tensor data,
        avoiding pickle to prevent potential security issues when sharing
        checkpoint files.
        """
        if path is None:
            path = os.path.join(
                self.config.save_dir,
                f"checkpoint-epoch{self.state.epoch}-step{self.state.global_step}.npz",
            )

        # Metadata stored as JSON-serializable dict
        metadata = {
            "epoch": int(self.state.epoch),
            "global_step": int(self.state.global_step),
            "best_metric": (
                float(self.state.best_metric)
                if self.state.best_metric != float("inf")
                else None
            ),
            "format_version": 2,  # Version 2 = safe format (no pickle)
        }

        try:
            import json

            import numpy as np

            # Collect all arrays to save
            arrays_to_save = {}

            # Save metadata as JSON string in a special array
            metadata_json = json.dumps(metadata)
            arrays_to_save["__metadata__"] = np.array(
                [ord(c) for c in metadata_json], dtype=np.uint8
            )

            # Save model state
            if hasattr(self.model, "state_dict"):
                model_state = self.model.state_dict()
                for key, value in model_state.items():
                    arr = self._to_numpy(value)
                    if arr is not None:
                        arrays_to_save[f"model.{key}"] = arr

            # Save optimizer state (simplified - scalar values only for safety)
            if self.optimizer is not None and hasattr(self.optimizer, "state_dict"):
                opt_state = self.optimizer.state_dict()
                # Only save param_groups config, not full state (avoids pickle)
                if "param_groups" in opt_state:
                    for i, group in enumerate(opt_state["param_groups"]):
                        for param_key, param_val in group.items():
                            if isinstance(param_val, (int, float)):
                                # Store scalars as 0-d arrays
                                arrays_to_save[f"optim.group{i}.{param_key}"] = (
                                    np.array(param_val)
                                )

            # Save scheduler state (simplified)
            if self.scheduler is not None and hasattr(self.scheduler, "state_dict"):
                sched_state = self.scheduler.state_dict()
                for key, value in sched_state.items():
                    if isinstance(value, (int, float)):
                        arrays_to_save[f"scheduler.{key}"] = np.array(value)

            # Save using numpy's compressed format (safe, no pickle)
            np.savez_compressed(path, **arrays_to_save)

        except Exception as e:
            logger.warning(f"Could not save checkpoint: {e}")

        return path

    def _to_numpy(self, value) -> Optional[Any]:
        """Convert a value to numpy array if possible."""
        try:
            import numpy as np

            if isinstance(value, np.ndarray):
                return value
            if hasattr(value, "numpy"):
                return value.numpy()
            if hasattr(value, "detach"):
                return value.detach().cpu().numpy()
            if isinstance(value, (list, tuple)):
                return np.array(value)
            if isinstance(value, (int, float)):
                return np.array(value)
            return None
        except Exception:
            return None

    def load_checkpoint(self, path: str):
        """Load a checkpoint with security restrictions.

        Supports two formats:
        - Version 2 (recommended): Safe .npz format with JSON metadata
        - Version 1 (legacy): Pickle format with RestrictedUnpickler

        The format is auto-detected based on file contents.
        """
        import json

        import numpy as np

        # Try to load as safe .npz format first
        try:
            data = np.load(path, allow_pickle=False)  # Security: Disable pickle

            # Check for metadata
            if "__metadata__" in data:
                # New safe format (version 2)
                metadata_bytes = data["__metadata__"]
                metadata_json = "".join(chr(b) for b in metadata_bytes)
                metadata = json.loads(metadata_json)

                # Restore state
                self.state.epoch = metadata.get("epoch", 0)
                self.state.global_step = metadata.get("global_step", 0)
                best_metric = metadata.get("best_metric")
                self.state.best_metric = (
                    float(best_metric) if best_metric is not None else float("inf")
                )

                # Restore model state
                if hasattr(self.model, "load_state_dict"):
                    model_state = {}
                    for key in data.files:
                        if key.startswith("model."):
                            param_name = key[6:]  # Remove "model." prefix
                            model_state[param_name] = data[key]

                    if model_state:
                        self.model.load_state_dict(model_state, strict=False)

                # Restore optimizer state (simplified)
                if self.optimizer is not None and hasattr(self.optimizer, "state_dict"):
                    current_state = self.optimizer.state_dict()
                    if "param_groups" in current_state:
                        for i, group in enumerate(current_state["param_groups"]):
                            for param_key in group:
                                npz_key = f"optim.group{i}.{param_key}"
                                if npz_key in data:
                                    group[param_key] = float(data[npz_key])

                # Restore scheduler state
                if self.scheduler is not None and hasattr(self.scheduler, "state_dict"):
                    current_state = self.scheduler.state_dict()
                    for key in current_state:
                        npz_key = f"scheduler.{key}"
                        if npz_key in data:
                            current_state[key] = float(data[npz_key])

                logger.info(
                    f"Loaded checkpoint (safe format v2): epoch={self.state.epoch}, "
                    f"step={self.state.global_step}"
                )
                return

        except Exception as e:
            # Not a valid .npz file, try legacy pickle format
            logger.debug(f"Could not load as .npz format: {e}")

        # Legacy pickle format with security restrictions
        logger.warning(
            "Loading checkpoint in legacy pickle format. "
            "Consider re-saving in safe format using trainer.save_checkpoint()."
        )
        try:
            with open(path, "rb") as f:
                checkpoint = RestrictedUnpickler(f).load()
        except pickle.UnpicklingError as e:
            raise RuntimeError(
                f"Checkpoint file may be corrupted or contain unsafe content: {e}"
            )
        except Exception as e:
            raise RuntimeError(f"Could not load checkpoint: {e}")

        # Restore state
        self.state.epoch = checkpoint.get("epoch", 0)
        self.state.global_step = checkpoint.get("global_step", 0)
        self.state.best_metric = checkpoint.get("best_metric", float("inf"))

        # Restore model
        if "model_state_dict" in checkpoint and hasattr(self.model, "load_state_dict"):
            self.model.load_state_dict(checkpoint["model_state_dict"])

        # Restore optimizer
        if "optimizer_state_dict" in checkpoint and self.optimizer is not None:
            if hasattr(self.optimizer, "load_state_dict"):
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # Restore scheduler
        if "scheduler_state_dict" in checkpoint and self.scheduler is not None:
            if hasattr(self.scheduler, "load_state_dict"):
                self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    def test(
        self,
        model: Optional[Any] = None,
        test_loader: Optional[Any] = None,
        criterion: Optional[Any] = None,
    ) -> Dict[str, float]:
        """
        Run test evaluation.

        Args:
            model: Model to test (uses self.model if None).
            test_loader: DataLoader for test data.
            criterion: Loss function (uses self.criterion if None).

        Returns:
            Dictionary with test metrics.
        """
        model = model or self.model
        criterion = criterion or self.criterion

        model.eval()

        total_loss = 0.0
        num_batches = 0

        for batch in test_loader:
            # Unpack batch
            if isinstance(batch, (tuple, list)):
                inputs, targets = batch[0], batch[1]
            else:
                inputs, targets = batch, None

            # Forward pass
            outputs = model(inputs)

            # Compute loss
            if criterion is not None and targets is not None:
                loss = criterion(outputs, targets)
            elif hasattr(outputs, "loss"):
                loss = outputs.loss
            else:
                loss = 0.0

            total_loss += float(loss)
            num_batches += 1

        return {
            "test_loss": total_loss / max(num_batches, 1),
        }

    def predict(
        self,
        model: Optional[Any] = None,
        dataloader: Optional[Any] = None,
    ) -> List[Any]:
        """
        Generate predictions.

        Args:
            model: Model to use (uses self.model if None).
            dataloader: DataLoader for prediction data.

        Returns:
            List of predictions.
        """
        model = model or self.model
        model.eval()

        predictions = []

        for batch in dataloader:
            # Handle different batch formats
            if isinstance(batch, (tuple, list)):
                inputs = batch[0]
            else:
                inputs = batch

            # Forward pass
            outputs = model(inputs)

            predictions.append(outputs)

        return predictions

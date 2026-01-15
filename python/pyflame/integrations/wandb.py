"""
Weights & Biases integration for PyFlame.

Provides experiment tracking, logging, and visualization.
"""

import os
from typing import Any, Dict, List, Optional


class WandbCallback:
    """Callback for logging training to Weights & Biases.

    Logs metrics, hyperparameters, and optionally models to W&B.

    Example:
        >>> callback = WandbCallback(
        ...     project="my-project",
        ...     name="experiment-1",
        ...     config={"lr": 0.001, "batch_size": 32}
        ... )
        >>> trainer = Trainer(callbacks=[callback])
    """

    def __init__(
        self,
        project: Optional[str] = None,
        entity: Optional[str] = None,
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        notes: Optional[str] = None,
        log_model: bool = False,
        log_gradients: bool = False,
        log_freq: int = 1,
        save_code: bool = True,
    ):
        """Initialize W&B callback.

        Args:
            project: W&B project name
            entity: W&B team/username
            name: Run name
            config: Hyperparameters to log
            tags: Tags for the run
            notes: Notes for the run
            log_model: Log model checkpoints
            log_gradients: Log gradient histograms
            log_freq: Logging frequency (batches)
            save_code: Save code to W&B
        """
        self.project = project
        self.entity = entity
        self.name = name
        self.config = config or {}
        self.tags = tags
        self.notes = notes
        self.log_model = log_model
        self.log_gradients = log_gradients
        self.log_freq = log_freq
        self.save_code = save_code

        self._run = None
        self._step = 0

    def _init_wandb(self):
        """Initialize W&B run if not already done."""
        if self._run is not None:
            return

        try:
            import wandb
        except ImportError:
            raise ImportError(
                "wandb is required for this callback. "
                "Install with: pip install wandb"
            )

        self._run = wandb.init(
            project=self.project,
            entity=self.entity,
            name=self.name,
            config=self.config,
            tags=self.tags,
            notes=self.notes,
            save_code=self.save_code,
            reinit=True,
        )

    def on_train_begin(self, trainer, **kwargs):
        """Called at the start of training.

        Args:
            trainer: Trainer instance
        """
        self._init_wandb()

        import wandb

        # Log trainer config
        if hasattr(trainer, "config"):
            config_dict = (
                trainer.config.__dict__ if hasattr(trainer.config, "__dict__") else {}
            )
            wandb.config.update(config_dict)

        # Log model architecture summary
        if hasattr(trainer, "model"):
            try:
                model_summary = str(trainer.model)
                wandb.config["model_architecture"] = model_summary[:1000]
            except Exception:
                pass

    def on_batch_end(self, batch: int, logs: Optional[Dict[str, Any]] = None, **kwargs):
        """Called at the end of each batch.

        Args:
            batch: Batch index
            logs: Metrics dict
        """
        if self._run is None:
            return

        import wandb

        self._step += 1

        if self._step % self.log_freq != 0:
            return

        if logs:
            # Filter and prepare metrics
            metrics = {}
            for key, value in logs.items():
                if isinstance(value, (int, float)):
                    metrics[f"train/{key}"] = value

            wandb.log(metrics, step=self._step)

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None, **kwargs):
        """Called at the end of each epoch.

        Args:
            epoch: Epoch index
            logs: Metrics dict
        """
        if self._run is None:
            return

        import wandb

        if logs:
            metrics = {}
            for key, value in logs.items():
                if isinstance(value, (int, float)):
                    prefix = "val/" if "val" in key else "train/"
                    clean_key = key.replace("val_", "").replace("train_", "")
                    metrics[f"{prefix}{clean_key}"] = value

            metrics["epoch"] = epoch
            wandb.log(metrics, step=self._step)

    def on_train_end(self, trainer, **kwargs):
        """Called at the end of training.

        Args:
            trainer: Trainer instance
        """
        if self._run is None:
            return

        import wandb

        # Log final model if requested
        if self.log_model and hasattr(trainer, "model"):
            try:
                import pyflame as pf

                artifact = wandb.Artifact(
                    "model",
                    type="model",
                    metadata={"framework": "pyflame"},
                )

                # Save model weights to temp file
                import tempfile

                with tempfile.NamedTemporaryFile(suffix=".pf", delete=False) as f:
                    model_path = f.name

                pf.save(trainer.model.state_dict(), model_path)
                artifact.add_file(model_path)
                wandb.log_artifact(artifact)

                # Cleanup
                os.unlink(model_path)

            except Exception as e:
                print(f"Warning: Could not log model to W&B: {e}")

        # Finish run
        wandb.finish()

    def log(self, data: Dict[str, Any], step: Optional[int] = None):
        """Log custom data to W&B.

        Args:
            data: Data to log
            step: Optional step number
        """
        if self._run is None:
            self._init_wandb()

        import wandb

        wandb.log(data, step=step or self._step)

    def log_image(
        self,
        key: str,
        images: List[Any],
        caption: Optional[List[str]] = None,
    ):
        """Log images to W&B.

        Args:
            key: Metric key
            images: List of images (numpy arrays or PIL Images)
            caption: Optional captions
        """
        if self._run is None:
            self._init_wandb()

        import wandb

        wandb_images = []
        for i, img in enumerate(images):
            cap = caption[i] if caption and i < len(caption) else None
            wandb_images.append(wandb.Image(img, caption=cap))

        wandb.log({key: wandb_images}, step=self._step)

    def log_table(
        self,
        key: str,
        columns: List[str],
        data: List[List[Any]],
    ):
        """Log a table to W&B.

        Args:
            key: Table key
            columns: Column names
            data: Table data
        """
        if self._run is None:
            self._init_wandb()

        import wandb

        table = wandb.Table(columns=columns, data=data)
        wandb.log({key: table}, step=self._step)

    def watch_model(self, model, log_freq: int = 100):
        """Watch model parameters and gradients.

        Args:
            model: Model to watch
            log_freq: Logging frequency
        """
        if self._run is None:
            self._init_wandb()

        # Note: Direct model watching may not be fully supported
        # This is a placeholder for future implementation
        pass


def init_wandb(
    project: str,
    entity: Optional[str] = None,
    name: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    **kwargs,
):
    """Initialize a W&B run.

    Convenience function for standalone W&B initialization.

    Args:
        project: Project name
        entity: Team/username
        name: Run name
        config: Configuration dict
        **kwargs: Additional wandb.init arguments

    Returns:
        W&B run object

    Example:
        >>> run = init_wandb("my-project", config={"lr": 0.001})
        >>> wandb.log({"loss": 0.5})
        >>> run.finish()
    """
    try:
        import wandb
    except ImportError:
        raise ImportError("wandb is required. Install with: pip install wandb")

    return wandb.init(
        project=project,
        entity=entity,
        name=name,
        config=config,
        **kwargs,
    )

"""
MLflow integration for PyFlame.

Provides experiment tracking, model logging, and artifact management.
"""

from typing import Any, Dict, List, Optional, Union
import os


class MLflowCallback:
    """Callback for logging training to MLflow.

    Logs metrics, parameters, and optionally models to MLflow.

    Example:
        >>> import mlflow
        >>> mlflow.set_experiment("my-experiment")
        >>> callback = MLflowCallback(log_models=True)
        >>> trainer = Trainer(callbacks=[callback])
    """

    def __init__(
        self,
        tracking_uri: Optional[str] = None,
        experiment_name: Optional[str] = None,
        run_name: Optional[str] = None,
        log_models: bool = True,
        log_params: bool = True,
        log_metrics_every_n_steps: int = 1,
        nested: bool = False,
        tags: Optional[Dict[str, str]] = None,
    ):
        """Initialize MLflow callback.

        Args:
            tracking_uri: MLflow tracking server URI
            experiment_name: Experiment name
            run_name: Run name
            log_models: Log model artifacts
            log_params: Log hyperparameters
            log_metrics_every_n_steps: Metric logging frequency
            nested: Create nested run
            tags: Run tags
        """
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.log_models = log_models
        self.log_params = log_params
        self.log_metrics_every_n_steps = log_metrics_every_n_steps
        self.nested = nested
        self.tags = tags or {}

        self._run = None
        self._step = 0

    def _init_mlflow(self):
        """Initialize MLflow run if not already done."""
        if self._run is not None:
            return

        try:
            import mlflow
        except ImportError:
            raise ImportError(
                "mlflow is required for this callback. "
                "Install with: pip install mlflow"
            )

        if self.tracking_uri:
            mlflow.set_tracking_uri(self.tracking_uri)

        if self.experiment_name:
            mlflow.set_experiment(self.experiment_name)

        self._run = mlflow.start_run(
            run_name=self.run_name,
            nested=self.nested,
            tags=self.tags,
        )

    def on_train_begin(self, trainer, **kwargs):
        """Called at the start of training.

        Args:
            trainer: Trainer instance
        """
        self._init_mlflow()

        import mlflow

        # Log hyperparameters
        if self.log_params and hasattr(trainer, "config"):
            try:
                config = trainer.config
                if hasattr(config, "__dict__"):
                    params = {}
                    for key, value in config.__dict__.items():
                        if isinstance(value, (int, float, str, bool)):
                            params[key] = value
                        elif value is None:
                            params[key] = "None"
                    mlflow.log_params(params)
            except Exception as e:
                print(f"Warning: Could not log params to MLflow: {e}")

        # Log model info as tag
        if hasattr(trainer, "model"):
            try:
                model_name = type(trainer.model).__name__
                mlflow.set_tag("model_type", model_name)
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

        import mlflow

        self._step += 1

        if self._step % self.log_metrics_every_n_steps != 0:
            return

        if logs:
            metrics = {}
            for key, value in logs.items():
                if isinstance(value, (int, float)):
                    metrics[key] = value

            if metrics:
                mlflow.log_metrics(metrics, step=self._step)

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None, **kwargs):
        """Called at the end of each epoch.

        Args:
            epoch: Epoch index
            logs: Metrics dict
        """
        if self._run is None:
            return

        import mlflow

        if logs:
            metrics = {}
            for key, value in logs.items():
                if isinstance(value, (int, float)):
                    # Prefix with epoch_ to distinguish from batch metrics
                    metrics[f"epoch_{key}"] = value

            if metrics:
                mlflow.log_metrics(metrics, step=epoch)

    def on_train_end(self, trainer, **kwargs):
        """Called at the end of training.

        Args:
            trainer: Trainer instance
        """
        if self._run is None:
            return

        import mlflow

        # Log final model
        if self.log_models and hasattr(trainer, "model"):
            try:
                import pyflame as pf
                import tempfile

                # Save model to temp file
                with tempfile.NamedTemporaryFile(suffix=".pf", delete=False) as f:
                    model_path = f.name

                pf.save(trainer.model.state_dict(), model_path)
                mlflow.log_artifact(model_path, artifact_path="model")

                # Cleanup
                os.unlink(model_path)

            except Exception as e:
                print(f"Warning: Could not log model to MLflow: {e}")

        # End run
        mlflow.end_run()
        self._run = None

    def log_metric(self, key: str, value: float, step: Optional[int] = None):
        """Log a single metric.

        Args:
            key: Metric name
            value: Metric value
            step: Optional step number
        """
        if self._run is None:
            self._init_mlflow()

        import mlflow
        mlflow.log_metric(key, value, step=step or self._step)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log multiple metrics.

        Args:
            metrics: Dictionary of metrics
            step: Optional step number
        """
        if self._run is None:
            self._init_mlflow()

        import mlflow
        mlflow.log_metrics(metrics, step=step or self._step)

    def log_params(self, params: Dict[str, Any]):
        """Log parameters.

        Args:
            params: Dictionary of parameters
        """
        if self._run is None:
            self._init_mlflow()

        import mlflow

        # Convert non-string values
        safe_params = {}
        for key, value in params.items():
            if isinstance(value, (int, float, str, bool)):
                safe_params[key] = value
            else:
                safe_params[key] = str(value)

        mlflow.log_params(safe_params)

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """Log an artifact.

        Args:
            local_path: Local file path
            artifact_path: Artifact destination path
        """
        if self._run is None:
            self._init_mlflow()

        import mlflow
        mlflow.log_artifact(local_path, artifact_path)

    def set_tag(self, key: str, value: str):
        """Set a run tag.

        Args:
            key: Tag name
            value: Tag value
        """
        if self._run is None:
            self._init_mlflow()

        import mlflow
        mlflow.set_tag(key, value)


def init_mlflow(
    experiment_name: str,
    tracking_uri: Optional[str] = None,
    run_name: Optional[str] = None,
    tags: Optional[Dict[str, str]] = None,
    **kwargs,
):
    """Initialize an MLflow run.

    Convenience function for standalone MLflow initialization.

    Args:
        experiment_name: Experiment name
        tracking_uri: MLflow tracking server URI
        run_name: Run name
        tags: Run tags
        **kwargs: Additional mlflow.start_run arguments

    Returns:
        MLflow run object

    Example:
        >>> run = init_mlflow("my-experiment", run_name="training-1")
        >>> mlflow.log_metric("loss", 0.5)
        >>> mlflow.end_run()
    """
    try:
        import mlflow
    except ImportError:
        raise ImportError(
            "mlflow is required. Install with: pip install mlflow"
        )

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    mlflow.set_experiment(experiment_name)

    return mlflow.start_run(run_name=run_name, tags=tags, **kwargs)


class PyFlameMLflowModel:
    """MLflow model flavor for PyFlame models.

    Enables saving and loading PyFlame models with MLflow.

    Example:
        >>> import mlflow
        >>> from pyflame.integrations.mlflow import PyFlameMLflowModel
        >>>
        >>> # Log model
        >>> mlflow.pyfunc.log_model(
        ...     artifact_path="model",
        ...     python_model=PyFlameMLflowModel(model),
        ... )
        >>>
        >>> # Load model
        >>> loaded_model = mlflow.pyfunc.load_model("runs:/<run_id>/model")
    """

    def __init__(self, model):
        """Initialize wrapper.

        Args:
            model: PyFlame model
        """
        self.model = model

    def predict(self, context, model_input):
        """Run prediction.

        Args:
            context: MLflow context
            model_input: Input data (pandas DataFrame or numpy array)

        Returns:
            Model predictions
        """
        import numpy as np

        # Convert input
        if hasattr(model_input, "values"):
            # Pandas DataFrame
            input_array = model_input.values
        else:
            input_array = np.asarray(model_input)

        # Convert to PyFlame tensor
        try:
            import pyflame as pf
            input_tensor = pf.tensor(input_array)
        except Exception:
            input_tensor = input_array

        # Run inference
        self.model.eval()
        output = self.model(input_tensor)

        # Convert output
        if hasattr(output, "numpy"):
            return output.numpy()
        return np.asarray(output)

    def save(self, path: str):
        """Save model to path.

        Args:
            path: Save path
        """
        import pyflame as pf
        pf.save(self.model.state_dict(), os.path.join(path, "model.pf"))

    @classmethod
    def load(cls, path: str, model_class=None, **model_kwargs):
        """Load model from path.

        Args:
            path: Load path
            model_class: Model class for reconstruction
            **model_kwargs: Model constructor arguments

        Returns:
            Loaded model wrapper
        """
        import pyflame as pf

        if model_class is None:
            raise ValueError("model_class required to load PyFlame model")

        model = model_class(**model_kwargs)
        state_dict = pf.load(os.path.join(path, "model.pf"))
        model.load_state_dict(state_dict)

        return cls(model)

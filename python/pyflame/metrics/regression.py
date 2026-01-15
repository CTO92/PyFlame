"""
Regression metrics for PyFlame.

Provides metrics for evaluating regression models.
"""

import math
from typing import Any, Optional

from .base import Metric


class MeanSquaredError(Metric):
    """
    Compute Mean Squared Error.

    MSE = mean((y_pred - y_true)^2)

    Args:
        squared: If True, return MSE; if False, return RMSE.

    Example:
        >>> mse = MeanSquaredError()
        >>> mse.update(predictions, targets)
        >>> result = mse.compute()  # 0.05
    """

    def __init__(self, squared: bool = True):
        super().__init__()
        self.squared = squared

        self.add_state("sum_squared_error", 0.0)
        self.add_state("total", 0)

    def _to_numpy(self, x: Any):
        if hasattr(x, "numpy"):
            return x.numpy()
        elif hasattr(x, "cpu"):
            return x.cpu().numpy()
        return x

    def update(self, preds: Any, target: Any) -> None:
        preds = self._to_numpy(preds)
        target = self._to_numpy(target)

        preds = preds.flatten()
        target = target.flatten()

        squared_errors = (preds - target) ** 2
        self._state["sum_squared_error"] += float(squared_errors.sum())
        self._state["total"] += len(preds)

    def compute(self) -> float:
        if self._state["total"] == 0:
            return 0.0

        mse = self._state["sum_squared_error"] / self._state["total"]

        if self.squared:
            return mse
        return math.sqrt(mse)


class RootMeanSquaredError(MeanSquaredError):
    """
    Compute Root Mean Squared Error.

    RMSE = sqrt(mean((y_pred - y_true)^2))

    Example:
        >>> rmse = RootMeanSquaredError()
        >>> rmse.update(predictions, targets)
        >>> result = rmse.compute()  # 0.22
    """

    def __init__(self):
        super().__init__(squared=False)


class MeanAbsoluteError(Metric):
    """
    Compute Mean Absolute Error.

    MAE = mean(|y_pred - y_true|)

    Example:
        >>> mae = MeanAbsoluteError()
        >>> mae.update(predictions, targets)
        >>> result = mae.compute()  # 0.15
    """

    def __init__(self):
        super().__init__()

        self.add_state("sum_abs_error", 0.0)
        self.add_state("total", 0)

    def _to_numpy(self, x: Any):
        if hasattr(x, "numpy"):
            return x.numpy()
        elif hasattr(x, "cpu"):
            return x.cpu().numpy()
        return x

    def update(self, preds: Any, target: Any) -> None:
        preds = self._to_numpy(preds)
        target = self._to_numpy(target)

        preds = preds.flatten()
        target = target.flatten()

        abs_errors = abs(preds - target)
        self._state["sum_abs_error"] += float(abs_errors.sum())
        self._state["total"] += len(preds)

    def compute(self) -> float:
        if self._state["total"] == 0:
            return 0.0
        return self._state["sum_abs_error"] / self._state["total"]


class R2Score(Metric):
    """
    Compute R^2 (coefficient of determination).

    R^2 = 1 - SS_res / SS_tot

    where:
    - SS_res = sum((y_true - y_pred)^2)
    - SS_tot = sum((y_true - mean(y_true))^2)

    Args:
        adjusted: If True, compute adjusted R^2.
        num_features: Number of features (required for adjusted R^2).

    Example:
        >>> r2 = R2Score()
        >>> r2.update(predictions, targets)
        >>> result = r2.compute()  # 0.92
    """

    def __init__(
        self,
        adjusted: bool = False,
        num_features: Optional[int] = None,
    ):
        super().__init__()
        self.adjusted = adjusted
        self.num_features = num_features

        self.add_state("sum_squared_error", 0.0)
        self.add_state("sum_target", 0.0)
        self.add_state("sum_target_squared", 0.0)
        self.add_state("total", 0)

    def _to_numpy(self, x: Any):
        if hasattr(x, "numpy"):
            return x.numpy()
        elif hasattr(x, "cpu"):
            return x.cpu().numpy()
        return x

    def update(self, preds: Any, target: Any) -> None:
        preds = self._to_numpy(preds)
        target = self._to_numpy(target)

        preds = preds.flatten()
        target = target.flatten()

        self._state["sum_squared_error"] += float(((target - preds) ** 2).sum())
        self._state["sum_target"] += float(target.sum())
        self._state["sum_target_squared"] += float((target**2).sum())
        self._state["total"] += len(target)

    def compute(self) -> float:
        n = self._state["total"]
        if n == 0:
            return 0.0

        ss_res = self._state["sum_squared_error"]

        # SS_tot = sum((y - mean(y))^2) = sum(y^2) - n * mean(y)^2
        mean_target = self._state["sum_target"] / n
        ss_tot = self._state["sum_target_squared"] - n * (mean_target**2)

        if ss_tot == 0:
            return 0.0

        r2 = 1 - (ss_res / ss_tot)

        if self.adjusted and self.num_features is not None:
            # Adjusted R^2 = 1 - (1-R^2) * (n-1) / (n-p-1)
            p = self.num_features
            if n - p - 1 > 0:
                r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

        return r2


class MeanAbsolutePercentageError(Metric):
    """
    Compute Mean Absolute Percentage Error.

    MAPE = mean(|y_true - y_pred| / |y_true|) * 100

    Example:
        >>> mape = MeanAbsolutePercentageError()
        >>> mape.update(predictions, targets)
        >>> result = mape.compute()  # 5.2 (%)
    """

    def __init__(self, epsilon: float = 1e-8):
        super().__init__()
        self.epsilon = epsilon

        self.add_state("sum_ape", 0.0)
        self.add_state("total", 0)

    def _to_numpy(self, x: Any):
        if hasattr(x, "numpy"):
            return x.numpy()
        elif hasattr(x, "cpu"):
            return x.cpu().numpy()
        return x

    def update(self, preds: Any, target: Any) -> None:
        preds = self._to_numpy(preds)
        target = self._to_numpy(target)

        preds = preds.flatten()
        target = target.flatten()

        # Avoid division by zero
        ape = abs(target - preds) / (abs(target) + self.epsilon)
        self._state["sum_ape"] += float(ape.sum())
        self._state["total"] += len(preds)

    def compute(self) -> float:
        if self._state["total"] == 0:
            return 0.0
        return (self._state["sum_ape"] / self._state["total"]) * 100


class ExplainedVariance(Metric):
    """
    Compute Explained Variance Score.

    EV = 1 - Var(y_true - y_pred) / Var(y_true)

    Example:
        >>> ev = ExplainedVariance()
        >>> ev.update(predictions, targets)
        >>> result = ev.compute()  # 0.89
    """

    def __init__(self):
        super().__init__()

        self.add_state("sum_error", 0.0)
        self.add_state("sum_error_squared", 0.0)
        self.add_state("sum_target", 0.0)
        self.add_state("sum_target_squared", 0.0)
        self.add_state("total", 0)

    def _to_numpy(self, x: Any):
        if hasattr(x, "numpy"):
            return x.numpy()
        elif hasattr(x, "cpu"):
            return x.cpu().numpy()
        return x

    def update(self, preds: Any, target: Any) -> None:
        preds = self._to_numpy(preds)
        target = self._to_numpy(target)

        preds = preds.flatten()
        target = target.flatten()

        error = target - preds

        self._state["sum_error"] += float(error.sum())
        self._state["sum_error_squared"] += float((error**2).sum())
        self._state["sum_target"] += float(target.sum())
        self._state["sum_target_squared"] += float((target**2).sum())
        self._state["total"] += len(preds)

    def compute(self) -> float:
        n = self._state["total"]
        if n == 0:
            return 0.0

        # Variance of error = E[error^2] - E[error]^2
        mean_error = self._state["sum_error"] / n
        var_error = self._state["sum_error_squared"] / n - mean_error**2

        # Variance of target
        mean_target = self._state["sum_target"] / n
        var_target = self._state["sum_target_squared"] / n - mean_target**2

        if var_target == 0:
            return 0.0

        return 1 - (var_error / var_target)


class PearsonCorrelation(Metric):
    """
    Compute Pearson Correlation Coefficient.

    r = Cov(X, Y) / (Std(X) * Std(Y))

    Example:
        >>> corr = PearsonCorrelation()
        >>> corr.update(predictions, targets)
        >>> result = corr.compute()  # 0.95
    """

    def __init__(self):
        super().__init__()

        self.add_state("sum_x", 0.0)
        self.add_state("sum_y", 0.0)
        self.add_state("sum_xy", 0.0)
        self.add_state("sum_x_squared", 0.0)
        self.add_state("sum_y_squared", 0.0)
        self.add_state("total", 0)

    def _to_numpy(self, x: Any):
        if hasattr(x, "numpy"):
            return x.numpy()
        elif hasattr(x, "cpu"):
            return x.cpu().numpy()
        return x

    def update(self, preds: Any, target: Any) -> None:
        preds = self._to_numpy(preds)
        target = self._to_numpy(target)

        preds = preds.flatten()
        target = target.flatten()

        self._state["sum_x"] += float(preds.sum())
        self._state["sum_y"] += float(target.sum())
        self._state["sum_xy"] += float((preds * target).sum())
        self._state["sum_x_squared"] += float((preds**2).sum())
        self._state["sum_y_squared"] += float((target**2).sum())
        self._state["total"] += len(preds)

    def compute(self) -> float:
        n = self._state["total"]
        if n == 0:
            return 0.0

        sum_x = self._state["sum_x"]
        sum_y = self._state["sum_y"]
        sum_xy = self._state["sum_xy"]
        sum_x2 = self._state["sum_x_squared"]
        sum_y2 = self._state["sum_y_squared"]

        # Pearson correlation formula
        numerator = n * sum_xy - sum_x * sum_y
        denominator = math.sqrt((n * sum_x2 - sum_x**2) * (n * sum_y2 - sum_y**2))

        if denominator == 0:
            return 0.0

        return numerator / denominator

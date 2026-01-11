"""
Tests for PyFlame Phase 3 Metrics Module.

Tests classification and regression metrics.
"""

import pytest
import sys
import os

# Add Python module to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from pyflame.metrics.base import Metric, MetricCollection
from pyflame.metrics.classification import (
    Accuracy,
    Precision,
    Recall,
    F1Score,
    ConfusionMatrix,
    AUROC,
    AveragePrecision,
)
from pyflame.metrics.regression import (
    MeanSquaredError,
    MeanAbsoluteError,
    RootMeanSquaredError,
    R2Score,
    MeanAbsolutePercentageError,
    PearsonCorrelation,
)


# =============================================================================
# Test Utilities
# =============================================================================

def make_numpy_array(data):
    """Create a numpy-like array for testing."""
    try:
        import numpy as np
        return np.array(data)
    except ImportError:
        pytest.skip("NumPy not available")


# =============================================================================
# Classification Metric Tests
# =============================================================================

class TestAccuracy:
    """Test cases for Accuracy metric."""

    def test_perfect_accuracy(self):
        preds = make_numpy_array([1, 1, 0, 0, 1])
        target = make_numpy_array([1, 1, 0, 0, 1])
        acc = Accuracy()
        acc.update(preds, target)
        assert acc.compute() == 1.0

    def test_zero_accuracy(self):
        preds = make_numpy_array([1, 1, 1, 1, 1])
        target = make_numpy_array([0, 0, 0, 0, 0])
        acc = Accuracy()
        acc.update(preds, target)
        assert acc.compute() == 0.0

    def test_partial_accuracy(self):
        preds = make_numpy_array([1, 0, 1, 0])
        target = make_numpy_array([1, 1, 0, 0])
        acc = Accuracy()
        acc.update(preds, target)
        assert acc.compute() == 0.5

    def test_accumulation(self):
        acc = Accuracy()
        preds1 = make_numpy_array([1, 1])
        target1 = make_numpy_array([1, 1])
        acc.update(preds1, target1)

        preds2 = make_numpy_array([0, 0])
        target2 = make_numpy_array([1, 1])
        acc.update(preds2, target2)

        assert acc.compute() == 0.5

    def test_reset(self):
        acc = Accuracy()
        acc.update(make_numpy_array([1]), make_numpy_array([1]))
        acc.reset()
        assert acc.compute() == 0.0

    def test_threshold(self):
        preds = make_numpy_array([0.9, 0.1, 0.6, 0.4])
        target = make_numpy_array([1, 0, 1, 0])
        acc = Accuracy(threshold=0.5)
        acc.update(preds, target)
        assert acc.compute() == 1.0


class TestPrecision:
    """Test cases for Precision metric."""

    def test_perfect_precision(self):
        preds = make_numpy_array([1, 1, 0, 0])
        target = make_numpy_array([1, 1, 0, 0])
        precision = Precision()
        precision.update(preds, target)
        assert precision.compute() == 1.0

    def test_precision_with_fp(self):
        preds = make_numpy_array([1, 1, 1, 1])
        target = make_numpy_array([1, 1, 0, 0])
        precision = Precision()
        precision.update(preds, target)
        assert precision.compute() == 0.5  # 2 TP / (2 TP + 2 FP)


class TestRecall:
    """Test cases for Recall metric."""

    def test_perfect_recall(self):
        preds = make_numpy_array([1, 1, 0, 0])
        target = make_numpy_array([1, 1, 0, 0])
        recall = Recall()
        recall.update(preds, target)
        assert recall.compute() == 1.0

    def test_recall_with_fn(self):
        preds = make_numpy_array([1, 0, 0, 0])
        target = make_numpy_array([1, 1, 0, 0])
        recall = Recall()
        recall.update(preds, target)
        assert recall.compute() == 0.5  # 1 TP / (1 TP + 1 FN)


class TestF1Score:
    """Test cases for F1Score metric."""

    def test_perfect_f1(self):
        preds = make_numpy_array([1, 1, 0, 0])
        target = make_numpy_array([1, 1, 0, 0])
        f1 = F1Score()
        f1.update(preds, target)
        assert f1.compute() == 1.0

    def test_f1_calculation(self):
        # Precision = 0.5, Recall = 0.5
        # F1 = 2 * (0.5 * 0.5) / (0.5 + 0.5) = 0.5
        preds = make_numpy_array([1, 0, 1, 0])
        target = make_numpy_array([1, 1, 0, 0])
        f1 = F1Score()
        f1.update(preds, target)
        result = f1.compute()
        assert abs(result - 0.5) < 0.01


class TestConfusionMatrix:
    """Test cases for ConfusionMatrix metric."""

    def test_binary_confusion_matrix(self):
        preds = make_numpy_array([0, 0, 1, 1])
        target = make_numpy_array([0, 1, 0, 1])
        cm = ConfusionMatrix(num_classes=2)
        cm.update(preds, target)
        matrix = cm.compute()
        # [[TN, FP], [FN, TP]] = [[1, 1], [1, 1]]
        assert matrix[0][0] == 1  # TN
        assert matrix[0][1] == 1  # FP
        assert matrix[1][0] == 1  # FN
        assert matrix[1][1] == 1  # TP

    def test_multiclass_confusion_matrix(self):
        preds = make_numpy_array([0, 1, 2, 0, 1, 2])
        target = make_numpy_array([0, 1, 2, 1, 0, 2])
        cm = ConfusionMatrix(num_classes=3)
        cm.update(preds, target)
        matrix = cm.compute()
        assert matrix[0][0] == 1  # Correct class 0
        assert matrix[2][2] == 2  # Correct class 2


class TestAUROC:
    """Test cases for AUROC metric."""

    def test_perfect_auroc(self):
        preds = make_numpy_array([0.9, 0.8, 0.1, 0.2])
        target = make_numpy_array([1, 1, 0, 0])
        auroc = AUROC()
        auroc.update(preds, target)
        assert auroc.compute() == 1.0

    def test_random_auroc(self):
        # For random predictions, AUC should be around 0.5
        preds = make_numpy_array([0.5, 0.5, 0.5, 0.5])
        target = make_numpy_array([1, 0, 1, 0])
        auroc = AUROC()
        auroc.update(preds, target)
        result = auroc.compute()
        assert 0.4 <= result <= 0.6


# =============================================================================
# Regression Metric Tests
# =============================================================================

class TestMSE:
    """Test cases for Mean Squared Error."""

    def test_perfect_mse(self):
        preds = make_numpy_array([1.0, 2.0, 3.0])
        target = make_numpy_array([1.0, 2.0, 3.0])
        mse = MeanSquaredError()
        mse.update(preds, target)
        assert mse.compute() == 0.0

    def test_mse_calculation(self):
        preds = make_numpy_array([1.0, 2.0, 3.0])
        target = make_numpy_array([2.0, 2.0, 2.0])
        mse = MeanSquaredError()
        mse.update(preds, target)
        # MSE = ((1-2)^2 + (2-2)^2 + (3-2)^2) / 3 = (1 + 0 + 1) / 3 = 0.667
        assert abs(mse.compute() - 0.667) < 0.01


class TestRMSE:
    """Test cases for Root Mean Squared Error."""

    def test_rmse_calculation(self):
        preds = make_numpy_array([1.0, 2.0, 3.0])
        target = make_numpy_array([2.0, 2.0, 2.0])
        rmse = RootMeanSquaredError()
        rmse.update(preds, target)
        # RMSE = sqrt(0.667) = 0.816
        assert abs(rmse.compute() - 0.816) < 0.01


class TestMAE:
    """Test cases for Mean Absolute Error."""

    def test_perfect_mae(self):
        preds = make_numpy_array([1.0, 2.0, 3.0])
        target = make_numpy_array([1.0, 2.0, 3.0])
        mae = MeanAbsoluteError()
        mae.update(preds, target)
        assert mae.compute() == 0.0

    def test_mae_calculation(self):
        preds = make_numpy_array([1.0, 2.0, 3.0])
        target = make_numpy_array([2.0, 2.0, 2.0])
        mae = MeanAbsoluteError()
        mae.update(preds, target)
        # MAE = (|1-2| + |2-2| + |3-2|) / 3 = (1 + 0 + 1) / 3 = 0.667
        assert abs(mae.compute() - 0.667) < 0.01


class TestR2Score:
    """Test cases for R2 Score."""

    def test_perfect_r2(self):
        preds = make_numpy_array([1.0, 2.0, 3.0])
        target = make_numpy_array([1.0, 2.0, 3.0])
        r2 = R2Score()
        r2.update(preds, target)
        assert abs(r2.compute() - 1.0) < 0.01

    def test_r2_mean_predictor(self):
        # If we predict the mean, R2 should be 0
        preds = make_numpy_array([2.0, 2.0, 2.0])
        target = make_numpy_array([1.0, 2.0, 3.0])
        r2 = R2Score()
        r2.update(preds, target)
        assert abs(r2.compute() - 0.0) < 0.01


class TestPearsonCorrelation:
    """Test cases for Pearson Correlation."""

    def test_perfect_correlation(self):
        preds = make_numpy_array([1.0, 2.0, 3.0, 4.0, 5.0])
        target = make_numpy_array([1.0, 2.0, 3.0, 4.0, 5.0])
        corr = PearsonCorrelation()
        corr.update(preds, target)
        assert abs(corr.compute() - 1.0) < 0.01

    def test_negative_correlation(self):
        preds = make_numpy_array([1.0, 2.0, 3.0, 4.0, 5.0])
        target = make_numpy_array([5.0, 4.0, 3.0, 2.0, 1.0])
        corr = PearsonCorrelation()
        corr.update(preds, target)
        assert abs(corr.compute() - (-1.0)) < 0.01


# =============================================================================
# MetricCollection Tests
# =============================================================================

class TestMetricCollection:
    """Test cases for MetricCollection."""

    def test_collection_update(self):
        metrics = MetricCollection({
            "accuracy": Accuracy(),
            "precision": Precision(),
        })
        preds = make_numpy_array([1, 1, 0, 0])
        target = make_numpy_array([1, 1, 0, 0])
        metrics.update(preds, target)
        results = metrics.compute()
        assert "accuracy" in results
        assert "precision" in results
        assert results["accuracy"] == 1.0

    def test_collection_with_prefix(self):
        metrics = MetricCollection(
            {"accuracy": Accuracy()},
            prefix="train_"
        )
        preds = make_numpy_array([1, 0])
        target = make_numpy_array([1, 0])
        metrics.update(preds, target)
        results = metrics.compute()
        assert "train_accuracy" in results

    def test_collection_reset(self):
        metrics = MetricCollection({"accuracy": Accuracy()})
        preds = make_numpy_array([1])
        target = make_numpy_array([1])
        metrics.update(preds, target)
        metrics.reset()
        results = metrics.compute()
        assert results["accuracy"] == 0.0


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

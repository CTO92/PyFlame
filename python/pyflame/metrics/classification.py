"""
Classification metrics for PyFlame.

Provides metrics for evaluating classification models.
"""

from typing import Any, List, Optional

from .base import Metric, StatScores


class Accuracy(Metric):
    """
    Compute accuracy for classification.

    Args:
        threshold: Threshold for converting probabilities to predictions.
        num_classes: Number of classes (for multi-class).
        top_k: If > 1, compute top-k accuracy.

    Example:
        >>> accuracy = Accuracy()
        >>> accuracy.update(predictions, targets)
        >>> result = accuracy.compute()  # 0.95
    """

    def __init__(
        self,
        threshold: float = 0.5,
        num_classes: Optional[int] = None,
        top_k: int = 1,
    ):
        super().__init__()
        self.threshold = threshold
        self.num_classes = num_classes
        self.top_k = top_k

        self.add_state("correct", 0)
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

        # Handle different prediction formats
        if len(preds.shape) > 1 and preds.shape[-1] > 1:
            # Multi-class: preds are logits/probabilities [N, C]
            if self.top_k == 1:
                preds = preds.argmax(axis=-1)
            else:
                # Top-k accuracy
                import numpy as np

                top_k_preds = np.argsort(preds, axis=-1)[..., -self.top_k :]
                # Check if target is in top-k
                target_expanded = target.reshape(-1, 1)
                correct = (top_k_preds == target_expanded).any(axis=-1).sum()
                self._state["correct"] += int(correct)
                self._state["total"] += len(target)
                return
        elif len(preds.shape) == 1 or preds.shape[-1] == 1:
            # Binary: apply threshold
            import numpy as np

            preds = preds.flatten()
            # Check if dtype is floating point (numpy dtype comparison)
            if np.issubdtype(preds.dtype, np.floating):
                preds = (preds >= self.threshold).astype(int)

        target = target.flatten()
        correct = (preds == target).sum()

        self._state["correct"] += int(correct)
        self._state["total"] += len(target)

    def compute(self) -> float:
        if self._state["total"] == 0:
            return 0.0
        return self._state["correct"] / self._state["total"]


class Precision(StatScores):
    """
    Compute precision for classification.

    Precision = TP / (TP + FP)

    Args:
        num_classes: Number of classes.
        threshold: Threshold for binary classification.
        average: Averaging method ("micro", "macro", "weighted", "none").

    Example:
        >>> precision = Precision()
        >>> precision.update(predictions, targets)
        >>> result = precision.compute()
    """

    def compute(self) -> float:
        tp = self._state["tp"]
        fp = self._state["fp"]

        if tp + fp == 0:
            return 0.0
        return tp / (tp + fp)


class Recall(StatScores):
    """
    Compute recall (sensitivity) for classification.

    Recall = TP / (TP + FN)

    Args:
        num_classes: Number of classes.
        threshold: Threshold for binary classification.
        average: Averaging method ("micro", "macro", "weighted", "none").

    Example:
        >>> recall = Recall()
        >>> recall.update(predictions, targets)
        >>> result = recall.compute()
    """

    def compute(self) -> float:
        tp = self._state["tp"]
        fn = self._state["fn"]

        if tp + fn == 0:
            return 0.0
        return tp / (tp + fn)


class F1Score(StatScores):
    """
    Compute F1 score for classification.

    F1 = 2 * (Precision * Recall) / (Precision + Recall)

    Args:
        num_classes: Number of classes.
        threshold: Threshold for binary classification.
        average: Averaging method ("micro", "macro", "weighted", "none").

    Example:
        >>> f1 = F1Score()
        >>> f1.update(predictions, targets)
        >>> result = f1.compute()
    """

    def compute(self) -> float:
        tp = self._state["tp"]
        fp = self._state["fp"]
        fn = self._state["fn"]

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)


class ConfusionMatrix(Metric):
    """
    Compute confusion matrix for classification.

    Args:
        num_classes: Number of classes.
        normalize: Normalization mode ("true", "pred", "all", None).

    Example:
        >>> cm = ConfusionMatrix(num_classes=3)
        >>> cm.update(predictions, targets)
        >>> matrix = cm.compute()  # [3, 3] array
    """

    def __init__(
        self,
        num_classes: int,
        normalize: Optional[str] = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.normalize = normalize

        # Initialize confusion matrix
        self.add_state("matrix", [[0] * num_classes for _ in range(num_classes)])

    def _to_numpy(self, x: Any):
        if hasattr(x, "numpy"):
            return x.numpy()
        elif hasattr(x, "cpu"):
            return x.cpu().numpy()
        return x

    def update(self, preds: Any, target: Any) -> None:
        preds = self._to_numpy(preds)
        target = self._to_numpy(target)

        # Handle logits
        if len(preds.shape) > 1:
            preds = preds.argmax(axis=-1)

        preds = preds.flatten().astype(int)
        target = target.flatten().astype(int)

        for p, t in zip(preds, target):
            if 0 <= p < self.num_classes and 0 <= t < self.num_classes:
                self._state["matrix"][t][p] += 1

    def compute(self) -> List[List[int]]:
        matrix = [row[:] for row in self._state["matrix"]]

        if self.normalize == "true":
            # Normalize by row (true labels)
            for i in range(self.num_classes):
                row_sum = sum(matrix[i])
                if row_sum > 0:
                    matrix[i] = [x / row_sum for x in matrix[i]]
        elif self.normalize == "pred":
            # Normalize by column (predictions)
            col_sums = [
                sum(matrix[i][j] for i in range(self.num_classes))
                for j in range(self.num_classes)
            ]
            for i in range(self.num_classes):
                for j in range(self.num_classes):
                    if col_sums[j] > 0:
                        matrix[i][j] /= col_sums[j]
        elif self.normalize == "all":
            # Normalize by total
            total = sum(sum(row) for row in matrix)
            if total > 0:
                matrix = [[x / total for x in row] for row in matrix]

        return matrix


class AUROC(Metric):
    """
    Compute Area Under the ROC Curve.

    Args:
        num_classes: Number of classes (for multi-class).
        average: Averaging method for multi-class.

    Example:
        >>> auroc = AUROC()
        >>> auroc.update(probabilities, targets)
        >>> result = auroc.compute()  # 0.85
    """

    def __init__(
        self,
        num_classes: Optional[int] = None,
        average: str = "macro",
    ):
        super().__init__()
        self.num_classes = num_classes
        self.average = average

        self.add_state("preds", [])
        self.add_state("target", [])

    def _to_numpy(self, x: Any):
        if hasattr(x, "numpy"):
            return x.numpy()
        elif hasattr(x, "cpu"):
            return x.cpu().numpy()
        return x

    def update(self, preds: Any, target: Any) -> None:
        preds = self._to_numpy(preds)
        target = self._to_numpy(target)

        # Handle multi-class predictions: extract positive class probability
        if len(preds.shape) > 1 and preds.shape[-1] > 1:
            if preds.shape[-1] == 2:
                # Binary classification with 2 classes: use probability of positive class
                preds = preds[:, 1]
            else:
                raise ValueError(
                    f"AUROC with multi-class ({preds.shape[-1]} classes) requires "
                    "explicit handling. Pass probabilities for the positive class only, "
                    "or use a multi-class AUROC implementation."
                )

        self._state["preds"].extend(preds.flatten().tolist())
        self._state["target"].extend(target.flatten().tolist())

    def compute(self) -> float:
        preds = self._state["preds"]
        target = self._state["target"]

        if len(preds) == 0:
            return 0.0

        # Simple AUC calculation for binary classification
        # Sort by prediction score
        paired = sorted(zip(preds, target), key=lambda x: x[0], reverse=True)

        # Calculate AUC using trapezoidal rule
        n_pos = sum(1 for _, t in paired if t == 1)
        n_neg = len(paired) - n_pos

        if n_pos == 0 or n_neg == 0:
            # AUROC is undefined when only one class is present
            return float("nan")

        auc = 0.0
        tp = 0
        fp = 0
        prev_tp = 0

        for _, t in paired:
            if t == 1:
                tp += 1
            else:
                fp += 1
                # Add trapezoid area
                auc += (tp + prev_tp) / 2.0

            prev_tp = tp

        auc /= n_pos * n_neg
        return auc


class AveragePrecision(Metric):
    """
    Compute Average Precision (area under precision-recall curve).

    Args:
        num_classes: Number of classes (for multi-class).
        average: Averaging method for multi-class.

    Example:
        >>> ap = AveragePrecision()
        >>> ap.update(probabilities, targets)
        >>> result = ap.compute()  # 0.72
    """

    def __init__(
        self,
        num_classes: Optional[int] = None,
        average: str = "macro",
    ):
        super().__init__()
        self.num_classes = num_classes
        self.average = average

        self.add_state("preds", [])
        self.add_state("target", [])

    def _to_numpy(self, x: Any):
        if hasattr(x, "numpy"):
            return x.numpy()
        elif hasattr(x, "cpu"):
            return x.cpu().numpy()
        return x

    def update(self, preds: Any, target: Any) -> None:
        preds = self._to_numpy(preds)
        target = self._to_numpy(target)

        self._state["preds"].extend(preds.flatten().tolist())
        self._state["target"].extend(target.flatten().tolist())

    def compute(self) -> float:
        preds = self._state["preds"]
        target = self._state["target"]

        if len(preds) == 0:
            return 0.0

        # Sort by prediction score (descending)
        paired = sorted(zip(preds, target), key=lambda x: x[0], reverse=True)

        n_pos = sum(1 for _, t in paired if t == 1)

        if n_pos == 0:
            return 0.0

        # Calculate AP
        ap = 0.0
        tp = 0
        n_retrieved = 0

        for _, t in paired:
            n_retrieved += 1
            if t == 1:
                tp += 1
                precision = tp / n_retrieved
                ap += precision

        ap /= n_pos
        return ap


class MulticlassAccuracy(Metric):
    """
    Compute accuracy for multi-class classification.

    Args:
        num_classes: Number of classes.
        average: Averaging method ("micro", "macro", "weighted").
    """

    def __init__(
        self,
        num_classes: int,
        average: str = "micro",
    ):
        super().__init__()
        self.num_classes = num_classes
        self.average = average

        self.add_state("correct_per_class", [0] * num_classes)
        self.add_state("total_per_class", [0] * num_classes)

    def _to_numpy(self, x: Any):
        if hasattr(x, "numpy"):
            return x.numpy()
        elif hasattr(x, "cpu"):
            return x.cpu().numpy()
        return x

    def update(self, preds: Any, target: Any) -> None:
        preds = self._to_numpy(preds)
        target = self._to_numpy(target)

        if len(preds.shape) > 1:
            preds = preds.argmax(axis=-1)

        preds = preds.flatten().astype(int)
        target = target.flatten().astype(int)

        for p, t in zip(preds, target):
            if 0 <= t < self.num_classes:
                self._state["total_per_class"][t] += 1
                if p == t:
                    self._state["correct_per_class"][t] += 1

    def compute(self) -> float:
        if self.average == "micro":
            total_correct = sum(self._state["correct_per_class"])
            total = sum(self._state["total_per_class"])
            return total_correct / total if total > 0 else 0.0

        elif self.average == "macro":
            accs = []
            for c in range(self.num_classes):
                if self._state["total_per_class"][c] > 0:
                    acc = (
                        self._state["correct_per_class"][c]
                        / self._state["total_per_class"][c]
                    )
                    accs.append(acc)
            return sum(accs) / len(accs) if accs else 0.0

        elif self.average == "weighted":
            total = sum(self._state["total_per_class"])
            if total == 0:
                return 0.0
            weighted_sum = 0.0
            for c in range(self.num_classes):
                if self._state["total_per_class"][c] > 0:
                    acc = (
                        self._state["correct_per_class"][c]
                        / self._state["total_per_class"][c]
                    )
                    weight = self._state["total_per_class"][c] / total
                    weighted_sum += acc * weight
            return weighted_sum

        return 0.0

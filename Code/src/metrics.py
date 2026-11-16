"""Metrics that the evaluation script writes for every run."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)


def compute_metrics(labels: np.ndarray, predictions: np.ndarray, class_names: list[str]) -> dict:
    precision, recall, f1, support = precision_recall_fscore_support(
        labels,
        predictions,
        labels=list(range(len(class_names))),
        average=None,
        zero_division=0,
    )
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="macro",
        zero_division=0,
    )
    matrix = confusion_matrix(labels, predictions, labels=list(range(len(class_names))))
    per_class = []
    for index, name in enumerate(class_names):
        per_class.append(
            {
                "class_name": name,
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
        )
    return {
        "accuracy": float(accuracy_score(labels, predictions)) if len(labels) else 0.0,
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)) if len(labels) else 0.0,
        "precision": float(macro_p) if len(labels) else 0.0,
        "recall": float(macro_r) if len(labels) else 0.0,
        "macro_f1": float(macro_f1) if len(labels) else 0.0,
        "per_class": per_class,
        "confusion_matrix": matrix,
    }

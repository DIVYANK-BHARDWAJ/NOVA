"""NOVA Quantitative Machine Learning — Evaluation Metrics & Benchmarks (Person 2).

Provides standard quantitative regression and directional accuracy metrics:
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Directional Accuracy (Hit Rate: sign(pred) == sign(true))
- R-squared (Coefficient of Determination)
- Defensible benchmark comparison against Zero Return and Historical Mean
"""

from __future__ import annotations

import math
from typing import Any, Sequence

from nova.quant.contracts import validate_finite


def mean_absolute_error(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Compute Mean Absolute Error (MAE)."""
    if len(y_true) != len(y_pred):
        raise ValueError(f"Length mismatch: len(y_true)={len(y_true)} != len(y_pred)={len(y_pred)}")
    if not y_true:
        return 0.0
    return sum(abs(t - p) for t, p in zip(y_true, y_pred)) / len(y_true)


def root_mean_squared_error(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Compute Root Mean Squared Error (RMSE)."""
    if len(y_true) != len(y_pred):
        raise ValueError(f"Length mismatch: len(y_true)={len(y_true)} != len(y_pred)={len(y_pred)}")
    if not y_true:
        return 0.0
    mse = sum((t - p) ** 2 for t, p in zip(y_true, y_pred)) / len(y_true)
    return math.sqrt(mse)


def directional_accuracy(y_true: Sequence[float], y_pred: Sequence[float], zero_tol: float = 1e-6) -> float:
    """Compute Directional Accuracy (percentage of signs matching).

    Returns value in [0.0, 1.0]. A value > 0.50 signifies directional edge.
    """
    if len(y_true) != len(y_pred):
        raise ValueError(f"Length mismatch: len(y_true)={len(y_true)} != len(y_pred)={len(y_pred)}")
    if not y_true:
        return 0.50

    matches = 0
    total = len(y_true)
    for t, p in zip(y_true, y_pred):
        # Sign agreement
        if (t > zero_tol and p > zero_tol) or (t < -zero_tol and p < -zero_tol):
            matches += 1
        elif abs(t) <= zero_tol and abs(p) <= zero_tol:
            matches += 1

    return matches / total


def r2_score(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Compute coefficient of determination (R^2)."""
    if len(y_true) != len(y_pred):
        raise ValueError(f"Length mismatch: len(y_true)={len(y_true)} != len(y_pred)={len(y_pred)}")
    if len(y_true) < 2:
        return 0.0

    mean_y = sum(y_true) / len(y_true)
    ss_tot = sum((y - mean_y) ** 2 for y in y_true)
    ss_res = sum((t - p) ** 2 for t, p in zip(y_true, y_pred))

    if ss_tot <= 1e-12:
        return 0.0

    return 1.0 - (ss_res / ss_tot)


def evaluate_regression_metrics(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    dataset_split: str = "test",
) -> dict[str, Any]:
    """Compute full quantitative metrics report for a set of predictions."""
    if not y_true:
        return {
            "split": dataset_split,
            "sample_count": 0,
            "mae": 0.0,
            "rmse": 0.0,
            "directional_accuracy": 0.50,
            "r2": 0.0,
        }

    mae = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    da = directional_accuracy(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return {
        "split": dataset_split,
        "sample_count": len(y_true),
        "mae": round(mae, 6),
        "rmse": round(rmse, 6),
        "directional_accuracy_pct": round(da * 100.0, 2),
        "r2": round(r2, 6),
    }


def compare_to_baselines(
    y_true: Sequence[float],
    y_pred_model: Sequence[float],
) -> dict[str, Any]:
    """Compare model predictions against defensible reference baselines.

    Baselines:
    1. Zero Return (Flat): y_pred = 0.0
    2. Historical Mean Return: y_pred = mean(y_true)
    """
    n = len(y_true)
    if n == 0:
        return {}

    y_zero = [0.0] * n
    mean_val = sum(y_true) / n
    y_mean = [mean_val] * n

    model_metrics = evaluate_regression_metrics(y_true, y_pred_model, dataset_split="model")
    zero_metrics = evaluate_regression_metrics(y_true, y_zero, dataset_split="zero_baseline")
    mean_metrics = evaluate_regression_metrics(y_true, y_mean, dataset_split="mean_baseline")

    # Outperformance check
    rmse_imprv_vs_zero = round((zero_metrics["rmse"] - model_metrics["rmse"]) / max(1e-6, zero_metrics["rmse"]) * 100.0, 2)
    rmse_imprv_vs_mean = round((mean_metrics["rmse"] - model_metrics["rmse"]) / max(1e-6, mean_metrics["rmse"]) * 100.0, 2)

    return {
        "model": model_metrics,
        "zero_baseline": zero_metrics,
        "mean_baseline": mean_metrics,
        "rmse_improvement_vs_zero_pct": rmse_imprv_vs_zero,
        "rmse_improvement_vs_mean_pct": rmse_imprv_vs_mean,
        "directional_advantage_vs_random_pct": round(model_metrics["directional_accuracy_pct"] - 50.0, 2),
    }


# ============================================================================
# Classification Metrics (Binary Directional Prediction)
# ============================================================================

def accuracy_score(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    """Compute standard classification accuracy (proportion of correct labels)."""
    if len(y_true) != len(y_pred):
        raise ValueError(f"Length mismatch: len(y_true)={len(y_true)} != len(y_pred)={len(y_pred)}")
    if len(y_true) == 0:
        return 0.50
    return sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)


def confusion_matrix(y_true: Sequence[int], y_pred: Sequence[int]) -> dict[str, int]:
    """Compute binary confusion matrix counts (TN, FP, FN, TP)."""
    if len(y_true) != len(y_pred):
        raise ValueError(f"Length mismatch: len(y_true)={len(y_true)} != len(y_pred)={len(y_pred)}")
    tn = fp = fn = tp = 0
    for t, p in zip(y_true, y_pred):
        if t == 1 and p == 1:
            tp += 1
        elif t == 0 and p == 1:
            fp += 1
        elif t == 1 and p == 0:
            fn += 1
        else:
            tn += 1
    return {"tn": tn, "fp": fp, "fn": fn, "tp": tp}


def balanced_accuracy_score(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    """Compute balanced accuracy: (Sensitivity + Specificity) / 2."""
    cm = confusion_matrix(y_true, y_pred)
    tp, fn = cm["tp"], cm["fn"]
    tn, fp = cm["tn"], cm["fp"]
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.5
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.5
    return (sensitivity + specificity) / 2.0


def precision_recall_f1(y_true: Sequence[int], y_pred: Sequence[int]) -> dict[str, float]:
    """Compute precision, recall, and F1-score for positive class."""
    cm = confusion_matrix(y_true, y_pred)
    tp, fp, fn = cm["tp"], cm["fp"], cm["fn"]
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1_score": round(f1, 6),
    }


def brier_score(y_true: Sequence[int], y_prob: Sequence[float]) -> float:
    """Compute Brier Score (Mean Squared Error of predicted probabilities)."""
    if len(y_true) != len(y_prob):
        raise ValueError(f"Length mismatch: len(y_true)={len(y_true)} != len(y_prob)={len(y_prob)}")
    if len(y_true) == 0:
        return 0.25
    return sum((p - float(t)) ** 2 for t, p in zip(y_true, y_prob)) / len(y_true)


def log_loss(y_true: Sequence[int], y_prob: Sequence[float], eps: float = 1e-15) -> float:
    """Compute Binary Cross-Entropy / Log Loss."""
    if len(y_true) != len(y_prob):
        raise ValueError(f"Length mismatch: len(y_true)={len(y_true)} != len(y_prob)={len(y_prob)}")
    if len(y_true) == 0:
        return 0.693147  # ln(2)
    loss = 0.0
    for t, p in zip(y_true, y_prob):
        p_clamped = max(eps, min(1.0 - eps, float(p)))
        if t == 1:
            loss -= math.log(p_clamped)
        else:
            loss -= math.log(1.0 - p_clamped)
    return loss / len(y_true)


def roc_auc_score(y_true: Sequence[int], y_prob: Sequence[float]) -> float:
    """Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC)."""
    if len(y_true) != len(y_prob):
        raise ValueError(f"Length mismatch: len(y_true)={len(y_true)} != len(y_prob)={len(y_prob)}")
    n_pos = sum(1 for y in y_true if y == 1)
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.50

    paired = sorted(zip(y_prob, y_true), key=lambda x: x[0], reverse=True)
    distinct_probs = sorted(list({p for p, _ in paired}), reverse=True)
    points = [(0.0, 0.0)]
    tp = 0
    fp = 0
    i = 0
    n = len(paired)
    for p_thresh in distinct_probs:
        while i < n and paired[i][0] >= p_thresh:
            if paired[i][1] == 1:
                tp += 1
            else:
                fp += 1
            i += 1
        points.append((fp / n_neg, tp / n_pos))

    auc = 0.0
    for j in range(1, len(points)):
        x0, y0 = points[j - 1]
        x1, y1 = points[j]
        auc += (x1 - x0) * (y0 + y1) / 2.0
    return auc


def evaluate_classification_metrics(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    threshold: float = 0.50,
    dataset_split: str = "test",
) -> dict[str, Any]:
    """Compute comprehensive classification evaluation dictionary."""
    if len(y_true) == 0:
        return {
            "split": dataset_split,
            "sample_count": 0,
            "accuracy_pct": 50.0,
            "balanced_accuracy_pct": 50.0,
            "roc_auc": 0.50,
            "brier_score": 0.25,
            "log_loss": 0.693147,
            "confusion_matrix": {"tn": 0, "fp": 0, "fn": 0, "tp": 0},
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
        }

    y_pred = [1 if p >= threshold else 0 for p in y_prob]
    acc = accuracy_score(y_true, y_pred)
    b_acc = balanced_accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    bs = brier_score(y_true, y_prob)
    ll = log_loss(y_true, y_prob)
    cm = confusion_matrix(y_true, y_pred)
    prf = precision_recall_f1(y_true, y_pred)

    return {
        "split": dataset_split,
        "sample_count": len(y_true),
        "threshold": threshold,
        "accuracy_pct": round(acc * 100.0, 2),
        "balanced_accuracy_pct": round(b_acc * 100.0, 2),
        "roc_auc": round(auc, 6),
        "brier_score": round(bs, 6),
        "log_loss": round(ll, 6),
        "confusion_matrix": cm,
        "precision": prf["precision"],
        "recall": prf["recall"],
        "f1_score": prf["f1_score"],
    }



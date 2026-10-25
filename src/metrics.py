"""Evaluation metrics, calibration analysis, and thresholding helpers for fraud detection.

Implements:
- pr_auc, roc_auc, brier
- threshold_for_precision (lowest threshold with precision >= target, validation only)
- precision_recall_f1_at_threshold
- recall_at_precision
- threshold_for_recall, fp_at_recall
- reliability plot generation
- mean_std_over_seeds aggregation
"""

from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def pr_auc(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> float:
    """Compute Area Under the Precision-Recall Curve (PR-AUC / Average Precision)."""
    y_t = np.asarray(y_true, dtype=int)
    y_p = np.asarray(y_pred, dtype=float)
    if y_t.sum() == 0:
        return 0.0
    return float(average_precision_score(y_t, y_p))


def roc_auc(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> float:
    """Compute Area Under the Receiver Operating Characteristic Curve (ROC-AUC)."""
    y_t = np.asarray(y_true, dtype=int)
    y_p = np.asarray(y_pred, dtype=float)
    if len(np.unique(y_t)) < 2:
        return 0.5
    return float(roc_auc_score(y_t, y_p))


def brier(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> float:
    """Compute Brier score loss (mean squared probability error)."""
    y_t = np.asarray(y_true, dtype=int)
    y_p = np.asarray(y_pred, dtype=float)
    return float(brier_score_loss(y_t, y_p))


def precision_recall_f1_at_threshold(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    threshold: float,
) -> Tuple[float, float, float]:
    """Compute precision, recall, and F1 score at a fixed probability threshold."""
    y_t = np.asarray(y_true, dtype=int)
    y_p = np.asarray(y_pred, dtype=float)
    y_hat = (y_p >= threshold).astype(int)

    prec = float(precision_score(y_t, y_hat, zero_division=0.0))
    rec = float(recall_score(y_t, y_hat, zero_division=0.0))
    f1 = float(f1_score(y_t, y_hat, zero_division=0.0))
    return prec, rec, f1


def threshold_for_precision(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    target_precision: float = 0.9,
) -> float:
    """Find the lowest threshold that achieves precision >= target_precision.

    Fitted strictly on validation sets. If no threshold achieves target precision,
    returns 1.0.
    """
    y_t = np.asarray(y_true, dtype=int)
    y_p = np.asarray(y_pred, dtype=float)

    if y_t.sum() == 0:
        return 1.0

    precisions, _, thresholds = precision_recall_curve(y_t, y_p)
    # precision has length len(thresholds) + 1; precision[:-1] matches thresholds
    valid_indices = np.where(precisions[:-1] >= target_precision)[0]
    if len(valid_indices) == 0:
        return 1.0

    # thresholds is sorted ascending, so the first valid index is the lowest threshold
    lowest_thresh = float(thresholds[valid_indices[0]])
    return lowest_thresh


def recall_at_precision(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    target_precision: float = 0.9,
) -> float:
    """Compute recall achieved at the lowest threshold meeting target precision."""
    thresh = threshold_for_precision(y_true, y_pred, target_precision)
    _, rec, _ = precision_recall_f1_at_threshold(y_true, y_pred, thresh)
    return rec


def threshold_for_recall(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    target_recall: float = 0.8,
) -> float:
    """Find the highest threshold that achieves recall >= target_recall.

    Maximizes precision / minimizes false positives while meeting recall constraint.
    If no threshold meets target recall, returns 0.0.
    """
    y_t = np.asarray(y_true, dtype=int)
    y_p = np.asarray(y_pred, dtype=float)

    if y_t.sum() == 0:
        return 0.0

    _, recalls, thresholds = precision_recall_curve(y_t, y_p)
    valid_indices = np.where(recalls[:-1] >= target_recall)[0]
    if len(valid_indices) == 0:
        return 0.0

    # recalls decreases as index increases; the last index has the highest threshold
    highest_thresh = float(thresholds[valid_indices[-1]])
    return highest_thresh


def fp_at_recall(
    y_val: np.ndarray | pd.Series,
    p_val: np.ndarray | pd.Series,
    y_test: np.ndarray | pd.Series,
    p_test: np.ndarray | pd.Series,
    target_recall: float = 0.8,
) -> int:
    """Compute false positive count on test at the validation-chosen recall threshold.

    Parameters
    ----------
    y_val, p_val : Validation ground truth and predicted probabilities.
    y_test, p_test : Test ground truth and predicted probabilities.
    target_recall : Desired minimum recall on validation (default 0.8).

    Returns
    -------
    int
        False positive count on test partition.
    """
    thresh = threshold_for_recall(y_val, p_val, target_recall=target_recall)
    y_t = np.asarray(y_test, dtype=int)
    p_t = np.asarray(p_test, dtype=float)

    test_pred = (p_t >= thresh).astype(int)
    fp = int(((test_pred == 1) & (y_t == 0)).sum())
    return fp


def plot_reliability(
    y_true: np.ndarray | pd.Series,
    p_raw: np.ndarray | pd.Series,
    p_cal: np.ndarray | pd.Series,
    n_bins: int = 10,
    ax: plt.Axes | None = None,
    title: str = "Reliability Diagram",
) -> plt.Axes:
    """Plot calibration curves for raw vs. Platt-calibrated probabilities."""
    y_t = np.asarray(y_true, dtype=int)
    p_r = np.asarray(p_raw, dtype=float)
    p_c = np.asarray(p_cal, dtype=float)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    # Perfect calibration reference
    ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")

    prob_true_raw, prob_pred_raw = calibration_curve(y_t, p_r, n_bins=n_bins, strategy="uniform")
    ax.plot(prob_pred_raw, prob_true_raw, "s-", color="#e41a1c", label="Raw XGBoost")

    prob_true_cal, prob_pred_cal = calibration_curve(y_t, p_c, n_bins=n_bins, strategy="uniform")
    ax.plot(prob_pred_cal, prob_true_cal, "o-", color="#377eb8", label="Platt Calibrated")

    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives (Empirical)")
    ax.set_title(title)
    ax.legend(loc="upper left")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])

    return ax


def mean_std_over_seeds(
    results_df: pd.DataFrame,
    group_cols: list[str],
    metric_cols: list[str],
) -> pd.DataFrame:
    """Aggregate per-seed metrics computing mean and sample standard deviation."""
    agg_funcs = {m: ["mean", "std"] for m in metric_cols}
    grouped = results_df.groupby(group_cols, observed=False).agg(agg_funcs)

    # Flatten hierarchical columns
    flattened_cols = []
    for col, stat in grouped.columns:
        flattened_cols.append(f"{col}_{stat}")
    grouped.columns = flattened_cols
    return grouped.reset_index()


if __name__ == "__main__":
    y_test = np.array([0, 0, 1, 1, 0, 1, 0, 0])
    p_test = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.7, 0.4, 0.15])
    print("PR-AUC:", pr_auc(y_test, p_test))
    print("ROC-AUC:", roc_auc(y_test, p_test))
    print("Brier:", brier(y_test, p_test))
    th = threshold_for_precision(y_test, p_test, 0.9)
    print("Thresh for prec >= 0.9:", th)
    print("P/R/F1 at thresh:", precision_recall_f1_at_threshold(y_test, p_test, th))

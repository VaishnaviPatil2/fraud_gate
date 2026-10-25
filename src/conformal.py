"""Conformal prediction, distribution-free risk control, and selective review module.

Implements:
1. Split conformal prediction sets:
   - Nonconformity s = 1 - p(true class)
   - Quantile fitted strictly on VALIDATION partition
   - Prediction sets C(x) = {y : 1 - p(y|x) <= q_hat}
   - Ambiguous sets ({0, 1}) routed to REVIEW
2. Conformal-style risk control for missed fraud:
   - Score cut tau_alpha chosen on validation such that validation FNR <= alpha
   - Realised FNR, review rate, auto recall, and auto precision on test
3. Plain uncertainty heuristic comparison at matched review rate
"""

from typing import Any, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score


def compute_nonconformity(
    y_true: np.ndarray | pd.Series,
    p_cal: np.ndarray | pd.Series,
) -> np.ndarray:
    """Compute nonconformity scores s = 1 - p(true_class).

    For y=1: s = 1 - p
    For y=0: s = p
    """
    y_t = np.asarray(y_true, dtype=int)
    p = np.asarray(p_cal, dtype=float)
    return np.where(y_t == 1, 1.0 - p, p)


def conformal_quantile(
    scores: np.ndarray | pd.Series,
    alpha: float,
) -> float:
    """Compute the finite-sample valid conformal quantile on calibration/validation scores.

    level = min(1.0, ceil((n + 1) * (1 - alpha)) / n)
    """
    s = np.asarray(scores, dtype=float)
    n = len(s)
    if n == 0:
        return 1.0
    q_level = min(1.0, float(np.ceil((n + 1) * (1.0 - alpha)) / n))
    q_hat = float(np.quantile(s, q_level, method="higher"))
    return q_hat


def predict_conformal_sets(
    p_cal: np.ndarray | pd.Series,
    q_hat: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Determine class membership for conformal prediction sets.

    0 in C(x) <=> p <= q_hat
    1 in C(x) <=> p >= 1 - q_hat

    Returns
    -------
    (in_set_0, in_set_1) : tuple of boolean ndarrays
    """
    p = np.asarray(p_cal, dtype=float)
    in_set_0 = p <= q_hat
    in_set_1 = p >= (1.0 - q_hat)
    return in_set_0, in_set_1


def evaluate_conformal_sets(
    y_true: np.ndarray | pd.Series,
    p_cal: np.ndarray | pd.Series,
    q_hat: float,
) -> dict[str, float]:
    """Evaluate conformal prediction sets on test data.

    Ambiguous sets ({0, 1}) and empty sets are routed to REVIEW.
    Singletons ({0} or {1}) are auto-decided.
    """
    y_t = np.asarray(y_true, dtype=int)
    in_0, in_1 = predict_conformal_sets(p_cal, q_hat)

    # Coverage: true class is in prediction set
    covered = np.where(y_t == 1, in_1, in_0)
    empirical_coverage = float(covered.mean())

    set_sizes = in_0.astype(int) + in_1.astype(int)
    share_ambiguous = float((set_sizes == 2).mean())
    share_empty = float((set_sizes == 0).mean())
    share_singleton = float((set_sizes == 1).mean())

    # Review rate: non-singletons routed to review
    review_mask = set_sizes != 1
    review_rate = float(review_mask.mean())

    # Auto-decided part: singletons
    auto_mask = ~review_mask
    if auto_mask.sum() > 0:
        auto_y = y_t[auto_mask]
        auto_pred = in_1[auto_mask].astype(int)
        auto_rec = float(recall_score(auto_y, auto_pred, zero_division=0.0))
        auto_prec = float(precision_score(auto_y, auto_pred, zero_division=0.0))
    else:
        auto_rec, auto_prec = 0.0, 0.0

    return {
        "q_hat": q_hat,
        "empirical_coverage": empirical_coverage,
        "share_ambiguous": share_ambiguous,
        "share_empty": share_empty,
        "share_singleton": share_singleton,
        "review_rate": review_rate,
        "auto_recall": auto_rec,
        "auto_precision": auto_prec,
    }


def risk_control_threshold(
    y_val: np.ndarray | pd.Series,
    p_val_cal: np.ndarray | pd.Series,
    alpha: float,
) -> float:
    """Find score cut tau_alpha on validation such that validation FNR <= alpha.

    Transactions with score < tau_alpha are auto-allowed.
    The threshold is chosen as the empirical alpha-quantile of validation fraud scores.
    """
    y_v = np.asarray(y_val, dtype=int)
    p_v = np.asarray(p_val_cal, dtype=float)

    fraud_scores = p_v[y_v == 1]
    if len(fraud_scores) == 0:
        return 0.0

    tau_alpha = float(np.quantile(fraud_scores, alpha, method="lower"))
    return tau_alpha


def evaluate_risk_control(
    y_test: np.ndarray | pd.Series,
    p_test_cal: np.ndarray | pd.Series,
    tau_alpha: float,
    decision_thresh: float = 0.5,
) -> dict[str, float]:
    """Evaluate conformal-style risk control for missed fraud on test data.

    - Auto-allow: p < tau_alpha (decided as 0)
    - Review: p >= tau_alpha
    - Realised FNR: fraction of test frauds with p < tau_alpha
    """
    y_t = np.asarray(y_test, dtype=int)
    p_t = np.asarray(p_test_cal, dtype=float)

    review_mask = p_t >= tau_alpha
    review_rate = float(review_mask.mean())

    fraud_mask = y_t == 1
    n_fraud = int(fraud_mask.sum())
    if n_fraud > 0:
        missed_fraud = (p_t[fraud_mask] < tau_alpha).sum()
        realised_fnr = float(missed_fraud / n_fraud)
    else:
        realised_fnr = 0.0

    auto_mask = ~review_mask
    if auto_mask.sum() > 0:
        auto_y = y_t[auto_mask]
        auto_pred = (p_t[auto_mask] >= decision_thresh).astype(int)
        auto_rec = float(recall_score(auto_y, auto_pred, zero_division=0.0))
        auto_prec = float(precision_score(auto_y, auto_pred, zero_division=0.0))
        # Purity of auto-allow: fraction of auto-allowed that are truly legitimate
        auto_allow_purity = float((auto_y == 0).mean())
    else:
        auto_rec, auto_prec = 0.0, 0.0
        auto_allow_purity = 0.0

    return {
        "tau_alpha": tau_alpha,
        "realised_fnr": realised_fnr,
        "review_rate": review_rate,
        "auto_allow_purity": auto_allow_purity,
        "auto_recall": auto_rec,
        "auto_precision": auto_prec,
    }


def evaluate_uncertainty_benchmark(
    y_val: np.ndarray | pd.Series,
    p_val_cal: np.ndarray | pd.Series,
    y_test: np.ndarray | pd.Series,
    p_test_cal: np.ndarray | pd.Series,
    target_review_rate: float,
    decision_thresh: float = 0.5,
) -> dict[str, float]:
    """Evaluate the plain uncertainty heuristic from exp1 matched to target review rate.

    Uncertainty u = 1 - max(p, 1-p) = min(p, 1-p).
    Threshold u_cut chosen on validation to match target_review_rate.
    """
    y_v = np.asarray(y_val, dtype=int)
    p_v = np.asarray(p_val_cal, dtype=float)
    y_t = np.asarray(y_test, dtype=int)
    p_t = np.asarray(p_test_cal, dtype=float)

    u_val = 1.0 - np.maximum(p_v, 1.0 - p_v)
    clamped_rate = float(np.clip(target_review_rate, 0.0001, 0.9999))
    u_cut = float(np.quantile(u_val, 1.0 - clamped_rate))

    u_test = 1.0 - np.maximum(p_t, 1.0 - p_t)
    review_mask = u_test >= u_cut
    test_review_rate = float(review_mask.mean())

    # Under uncertainty routing, cases with u < u_cut and p < decision_thresh are auto-allowed
    fraud_mask = y_t == 1
    n_fraud = int(fraud_mask.sum())
    if n_fraud > 0:
        missed_fraud = ((~review_mask & fraud_mask) & (p_t < decision_thresh)).sum()
        realised_fnr = float(missed_fraud / n_fraud)
    else:
        realised_fnr = 0.0

    auto_mask = ~review_mask
    if auto_mask.sum() > 0:
        auto_y = y_t[auto_mask]
        auto_pred = (p_t[auto_mask] >= decision_thresh).astype(int)
        auto_rec = float(recall_score(auto_y, auto_pred, zero_division=0.0))
        auto_prec = float(precision_score(auto_y, auto_pred, zero_division=0.0))
    else:
        auto_rec, auto_prec = 0.0, 0.0

    return {
        "u_cut": u_cut,
        "test_review_rate": test_review_rate,
        "realised_fnr": realised_fnr,
        "auto_recall": auto_rec,
        "auto_precision": auto_prec,
    }

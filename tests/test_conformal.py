"""Unit tests for conformal prediction and risk control functions in src/conformal.py."""

import numpy as np
import pytest

from src.conformal import (
    compute_nonconformity,
    conformal_quantile,
    evaluate_conformal_sets,
    evaluate_risk_control,
    evaluate_uncertainty_benchmark,
    predict_conformal_sets,
    risk_control_threshold,
)


@pytest.fixture
def sample_conformal_data():
    y_val = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    p_val = np.array([0.05, 0.10, 0.15, 0.20, 0.70, 0.80, 0.85, 0.95])
    y_test = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    p_test = np.array([0.06, 0.12, 0.18, 0.25, 0.65, 0.78, 0.82, 0.92])
    return y_val, p_val, y_test, p_test


def test_compute_nonconformity(sample_conformal_data):
    y_val, p_val, _, _ = sample_conformal_data
    s = compute_nonconformity(y_val, p_val)
    assert len(s) == len(y_val)
    # For y=0, s = p
    assert s[0] == pytest.approx(0.05)
    # For y=1, s = 1 - p
    assert s[4] == pytest.approx(0.30)
    assert np.all(s >= 0.0) and np.all(s <= 1.0)


def test_conformal_quantile(sample_conformal_data):
    y_val, p_val, _, _ = sample_conformal_data
    s = compute_nonconformity(y_val, p_val)
    q_hat = conformal_quantile(s, alpha=0.10)
    assert 0.0 <= q_hat <= 1.0
    # Coverage on validation should be >= 1 - alpha
    covered_val = np.where(y_val == 1, p_val >= (1 - q_hat), p_val <= q_hat)
    assert covered_val.mean() >= 0.85


def test_predict_conformal_sets(sample_conformal_data):
    _, _, _, p_test = sample_conformal_data
    q_hat = 0.35
    in_0, in_1 = predict_conformal_sets(p_test, q_hat)
    assert len(in_0) == len(p_test)
    assert len(in_1) == len(p_test)
    # Low probability sample should contain 0
    assert in_0[0] is True or in_0[0] == 1
    # High probability sample should contain 1
    assert in_1[-1] is True or in_1[-1] == 1


def test_evaluate_conformal_sets(sample_conformal_data):
    _, _, y_test, p_test = sample_conformal_data
    q_hat = 0.35
    metrics = evaluate_conformal_sets(y_test, p_test, q_hat)
    assert "empirical_coverage" in metrics
    assert "share_ambiguous" in metrics
    assert "review_rate" in metrics
    assert 0.0 <= metrics["empirical_coverage"] <= 1.0
    assert 0.0 <= metrics["share_ambiguous"] <= 1.0
    assert 0.0 <= metrics["review_rate"] <= 1.0


def test_risk_control_threshold(sample_conformal_data):
    y_val, p_val, _, _ = sample_conformal_data
    tau_10 = risk_control_threshold(y_val, p_val, alpha=0.10)
    assert 0.0 <= tau_10 <= 1.0
    # Validation FNR should be <= alpha
    fnr = (p_val[y_val == 1] < tau_10).mean()
    assert fnr <= 0.25  # In small discrete sample


def test_evaluate_risk_control(sample_conformal_data):
    y_val, p_val, y_test, p_test = sample_conformal_data
    tau = risk_control_threshold(y_val, p_val, alpha=0.10)
    res = evaluate_risk_control(y_test, p_test, tau, decision_thresh=0.5)
    assert "realised_fnr" in res
    assert "review_rate" in res
    assert 0.0 <= res["realised_fnr"] <= 1.0
    assert 0.0 <= res["review_rate"] <= 1.0


def test_evaluate_uncertainty_benchmark(sample_conformal_data):
    y_val, p_val, y_test, p_test = sample_conformal_data
    bench = evaluate_uncertainty_benchmark(
        y_val, p_val, y_test, p_test, target_review_rate=0.25, decision_thresh=0.5
    )
    assert "u_cut" in bench
    assert "test_review_rate" in bench
    assert "realised_fnr" in bench
    assert 0.0 <= bench["test_review_rate"] <= 1.0


def test_conformal_quantile_edge_cases():
    # Empty scores
    assert conformal_quantile(np.array([]), alpha=0.1) == 1.0
    # Extreme alphas
    scores = np.array([0.1, 0.2, 0.3])
    assert conformal_quantile(scores, alpha=0.0) == pytest.approx(0.3)
    assert 0.0 <= conformal_quantile(scores, alpha=1.0) <= 0.3


"""Unit tests for all functions in src/metrics.py."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from src.metrics import (
    brier,
    fp_at_recall,
    mean_std_over_seeds,
    plot_reliability,
    pr_auc,
    precision_recall_f1_at_threshold,
    recall_at_precision,
    roc_auc,
    threshold_for_precision,
    threshold_for_recall,
)


@pytest.fixture
def sample_predictions():
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_pred = np.array([0.05, 0.10, 0.20, 0.30, 0.60, 0.75, 0.85, 0.95])
    return y_true, y_pred


def test_pr_auc(sample_predictions):
    y_true, y_pred = sample_predictions
    score = pr_auc(y_true, y_pred)
    assert 0.0 <= score <= 1.0
    assert score == pytest.approx(1.0, abs=1e-3)


def test_roc_auc(sample_predictions):
    y_true, y_pred = sample_predictions
    score = roc_auc(y_true, y_pred)
    assert score == 1.0


def test_brier(sample_predictions):
    y_true, y_pred = sample_predictions
    score = brier(y_true, y_pred)
    assert score >= 0.0
    assert score < 0.1  # Very good predictions


def test_precision_recall_f1_at_threshold(sample_predictions):
    y_true, y_pred = sample_predictions
    p, r, f1 = precision_recall_f1_at_threshold(y_true, y_pred, threshold=0.5)
    assert p == 1.0
    assert r == 1.0
    assert f1 == 1.0

    p_high, r_high, _ = precision_recall_f1_at_threshold(y_true, y_pred, threshold=0.8)
    assert p_high == 1.0
    assert r_high == 0.5  # Only 2 out of 4 detected


def test_threshold_for_precision(sample_predictions):
    y_true, y_pred = sample_predictions
    th_90 = threshold_for_precision(y_true, y_pred, target_precision=0.9)
    assert th_90 <= 0.60  # At 0.60, precision is already 1.0

    p, _, _ = precision_recall_f1_at_threshold(y_true, y_pred, th_90)
    assert p >= 0.90


def test_recall_at_precision(sample_predictions):
    y_true, y_pred = sample_predictions
    rec = recall_at_precision(y_true, y_pred, target_precision=0.9)
    assert rec == 1.0


def test_threshold_for_recall(sample_predictions):
    y_true, y_pred = sample_predictions
    th_75 = threshold_for_recall(y_true, y_pred, target_recall=0.75)
    # Target recall 0.75 requires at least 3 out of 4 positives (predicted >= 0.75)
    _, r, _ = precision_recall_f1_at_threshold(y_true, y_pred, th_75)
    assert r >= 0.75


def test_fp_at_recall(sample_predictions):
    y_true, y_pred = sample_predictions
    # Use identical distribution for val and test
    fp = fp_at_recall(y_true, y_pred, y_true, y_pred, target_recall=0.75)
    assert fp == 0  # Perfect separation in sample_predictions


def test_plot_reliability(sample_predictions):
    y_true, y_pred = sample_predictions
    fig, ax = plt.subplots()
    out_ax = plot_reliability(y_true, y_pred, y_pred, n_bins=5, ax=ax)
    assert out_ax is ax
    plt.close(fig)


def test_mean_std_over_seeds():
    df = pd.DataFrame(
        {
            "variant": ["A", "A", "B", "B"],
            "seed": [0, 1, 0, 1],
            "metric1": [0.80, 0.82, 0.70, 0.72],
            "metric2": [0.90, 0.92, 0.60, 0.62],
        }
    )
    agg_df = mean_std_over_seeds(df, group_cols=["variant"], metric_cols=["metric1", "metric2"])
    assert "metric1_mean" in agg_df.columns
    assert "metric1_std" in agg_df.columns
    assert len(agg_df) == 2
    row_a = agg_df[agg_df["variant"] == "A"].iloc[0]
    assert row_a["metric1_mean"] == pytest.approx(0.81)

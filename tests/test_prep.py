"""Unit tests for data preparation, feature engineering, and leak prevention."""

import numpy as np
import pandas as pd
import pytest

from src.common import split_temporal
from src.features import (
    EXTENDED_FEATURE_COLS,
    STRICT_FEATURE_COLS,
    assert_no_leak,
    compute_features,
    get_feature_columns,
)


def test_assert_no_leak_passes_valid():
    valid_cols = ["step", "amount", "oldbalanceOrg", "type_TRANSFER"]
    assert_no_leak(valid_cols)


def test_assert_no_leak_raises_on_target():
    with pytest.raises(ValueError, match="Data leakage detected"):
        assert_no_leak(["step", "amount", "isFraud"])


def test_assert_no_leak_raises_on_flagged():
    with pytest.raises(ValueError, match="Data leakage detected"):
        assert_no_leak(["step", "amount", "isFlaggedFraud"])


def test_assert_no_leak_raises_on_newbalance():
    with pytest.raises(ValueError, match="Data leakage detected"):
        assert_no_leak(["step", "amount", "newbalanceOrig"])
    with pytest.raises(ValueError, match="Data leakage detected"):
        assert_no_leak(["step", "amount", "newbalanceDest"])


def test_compute_features():
    df = pd.DataFrame(
        {
            "step": np.array([1, 10, 500, 600], dtype=np.int16),
            "type": ["TRANSFER", "CASH_OUT", "PAYMENT", "DEBIT"],
            "amount": np.array([100.0, 50.0, 20.0, 10.0], dtype=np.float32),
            "oldbalanceOrg": np.array([100.0, 0.0, 50.0, 100.0], dtype=np.float32),
            "oldbalanceDest": np.array([0.0, 100.0, 0.0, 50.0], dtype=np.float32),
        }
    )

    X_strict = compute_features(df, feature_set="strict")
    assert list(X_strict.columns) == list(STRICT_FEATURE_COLS)
    assert X_strict.shape == (4, len(STRICT_FEATURE_COLS))

    X_ext = compute_features(df, feature_set="extended")
    assert list(X_ext.columns) == list(EXTENDED_FEATURE_COLS)
    assert X_ext.shape == (4, len(EXTENDED_FEATURE_COLS))
    assert np.all(np.isfinite(X_ext.to_numpy()))


def test_split_temporal():
    df = pd.DataFrame(
        {
            "step": [1, 100, 520, 521, 600, 631, 632, 700, 743],
            "val": range(9),
        }
    )
    train_df, val_df, test_df = split_temporal(df)
    assert train_df["step"].max() <= 520
    assert val_df["step"].min() >= 521
    assert val_df["step"].max() <= 631
    assert test_df["step"].min() >= 632
    assert test_df["step"].max() <= 743
    assert len(train_df) == 3
    assert len(val_df) == 3
    assert len(test_df) == 3

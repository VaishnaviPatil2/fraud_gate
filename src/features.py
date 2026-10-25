"""Feature engineering module for PaySim tabular transaction data.

Implements:
- STRICT feature set: step, amount, type one-hot.
- EXTENDED feature set: STRICT + oldbalanceOrg, oldbalanceDest,
  amount/(oldbalanceOrg+1), amount>=oldbalanceOrg, oldbalanceDest==0.
- assert_no_leak(cols): Enforces zero leakage of target or post-transaction balances.
"""

from collections.abc import Iterable
from typing import Tuple

import numpy as np
import pandas as pd

# Standard categorical order for PaySim transaction types
TRANSACTION_TYPES = ("CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER")

TYPE_ONE_HOT_COLS = tuple(f"type_{t}" for t in TRANSACTION_TYPES)

STRICT_BASE_COLS = ("step", "amount")
STRICT_FEATURE_COLS = STRICT_BASE_COLS + TYPE_ONE_HOT_COLS

EXTENDED_ENGINEERED_COLS = (
    "oldbalanceOrg",
    "oldbalanceDest",
    "amount_to_oldbalanceOrg_ratio",
    "amount_ge_oldbalanceOrg",
    "oldbalanceDest_is_zero",
)
EXTENDED_FEATURE_COLS = STRICT_FEATURE_COLS + EXTENDED_ENGINEERED_COLS


def assert_no_leak(cols: Iterable[str]) -> None:
    """Validate that forbidden columns do not appear in feature columns.

    Forbidden columns include:
    - Anything matching newbalance* (e.g. newbalanceOrig, newbalanceDest)
    - isFlaggedFraud
    - isFraud (ground truth target)

    Raises
    ------
    ValueError
        If any forbidden column or pattern is detected.
    """
    for col in cols:
        col_lower = str(col).lower()
        if "newbalance" in col_lower:
            raise ValueError(
                f"Data leakage detected! Forbidden column in feature set: '{col}' "
                "(post-transaction balances must never be used as features)."
            )
        if col_lower in ("isflaggedfraud", "isflagged_fraud"):
            raise ValueError(
                f"Data leakage detected! Forbidden column in feature set: '{col}' "
                "(isFlaggedFraud must never be used as a feature)."
            )
        if col_lower in ("isfraud", "is_fraud"):
            raise ValueError(
                f"Data leakage detected! Forbidden column in feature set: '{col}' "
                "(ground-truth label must never be used as a feature)."
            )


# Verify default lists immediately upon import
assert_no_leak(STRICT_FEATURE_COLS)
assert_no_leak(EXTENDED_FEATURE_COLS)


def get_feature_columns(feature_set: str = "extended") -> list[str]:
    """Return validated list of feature column names."""
    feature_set = feature_set.lower()
    if feature_set == "strict":
        cols = list(STRICT_FEATURE_COLS)
    elif feature_set == "extended":
        cols = list(EXTENDED_FEATURE_COLS)
    else:
        raise ValueError(
            f"Unknown feature set: '{feature_set}'. Must be 'strict' or 'extended'."
        )
    assert_no_leak(cols)
    return cols


def compute_features(
    df: pd.DataFrame,
    feature_set: str = "extended",
) -> pd.DataFrame:
    """Compute and extract features from transaction dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with raw transaction columns.
    feature_set : str
        'strict' or 'extended' (default 'extended').

    Returns
    -------
    pd.DataFrame
        DataFrame containing strictly the selected feature columns in float32/int16.
    """
    feature_set = feature_set.lower()
    if feature_set not in ("strict", "extended"):
        raise ValueError(f"Invalid feature_set '{feature_set}'. Choose 'strict' or 'extended'.")

    # 1. Base STRICT features
    X = pd.DataFrame(index=df.index)
    X["step"] = df["step"].astype(np.int16)
    X["amount"] = df["amount"].astype(np.float32)

    # 2. Vectorized One-hot encoding for transaction type
    if isinstance(df["type"].dtype, pd.CategoricalDtype):
        type_series = df["type"]
    else:
        type_series = df["type"].astype(
            pd.CategoricalDtype(categories=list(TRANSACTION_TYPES))
        )

    for t in TRANSACTION_TYPES:
        col_name = f"type_{t}"
        X[col_name] = (type_series == t).astype(np.float32)

    # 3. EXTENDED engineered features
    if feature_set == "extended":
        old_org = df["oldbalanceOrg"].astype(np.float32)
        old_dest = df["oldbalanceDest"].astype(np.float32)
        amt = X["amount"]

        X["oldbalanceOrg"] = old_org
        X["oldbalanceDest"] = old_dest
        # Vectorized ratio avoiding division by zero
        X["amount_to_oldbalanceOrg_ratio"] = (amt / (old_org + np.float32(1.0))).astype(np.float32)
        X["amount_ge_oldbalanceOrg"] = (amt >= old_org).astype(np.float32)
        X["oldbalanceDest_is_zero"] = (old_dest == np.float32(0.0)).astype(np.float32)

    expected_cols = get_feature_columns(feature_set)
    assert_no_leak(X.columns)
    return X[expected_cols]


def extract_features_and_target(
    df: pd.DataFrame,
    feature_set: str = "extended",
) -> Tuple[pd.DataFrame, pd.Series | None]:
    """Extract feature matrix X and target y (if present).

    Calls assert_no_leak on feature matrix columns.
    """
    X = compute_features(df, feature_set=feature_set)
    assert_no_leak(X.columns)

    y = None
    if "isFraud" in df.columns:
        y = df["isFraud"].astype(np.int8)

    return X, y


if __name__ == "__main__":
    test_cols = ["step", "amount", "oldbalanceOrg"]
    assert_no_leak(test_cols)
    print("assert_no_leak verified successfully.")
    print("STRICT features:", get_feature_columns("strict"))
    print("EXTENDED features:", get_feature_columns("extended"))

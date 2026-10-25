"""Unit tests for Experiment 1b diagnostic output structure and metric bounds."""

import pandas as pd
import pytest

from exp.exp1b_diagnostic import run_diagnostic


def test_exp1b_diagnostic_output_structure_and_bounds():
    # Run in sample mode for quick unit testing
    results_df = run_diagnostic(mode="sample")

    expected_cols = [
        "type",
        "is_fraud",
        "label",
        "row_count",
        "share_amount_ge_oldbalanceOrg",
        "share_oldbalanceDest_zero",
        "share_both_conditions",
        "median_amount_ratio",
    ]
    for col in expected_cols:
        assert col in results_df.columns, f"Missing expected column: {col}"

    assert len(results_df) >= 4, "Must contain at least 4 subgroups"

    share_cols = [
        "share_amount_ge_oldbalanceOrg",
        "share_oldbalanceDest_zero",
        "share_both_conditions",
    ]
    for col in share_cols:
        for val in results_df[col]:
            assert 0.0 <= val <= 1.0, f"Share {col}={val} out of bounds [0, 1]"

    for count in results_df["row_count"]:
        assert count >= 0, "Row count must be non-negative"

    for med in results_df["median_amount_ratio"]:
        assert med >= 0.0, "Median amount ratio must be non-negative"

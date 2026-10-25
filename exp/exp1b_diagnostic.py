"""Experiment 1b: Diagnostic Feature Analysis on Training Set (Steps 1-520).

Evaluates the behavioral separation of the EXTENDED balance features:
- row count
- share of rows with amount >= oldbalanceOrg
- share of rows with oldbalanceDest == 0
- share of rows with both conditions true
- median of amount/(oldbalanceOrg+1)
Evaluated strictly on TRAIN partition (steps 1-520), separately for fraud (isFraud=1)
and legitimate (isFraud=0) rows broken down by transaction type (TRANSFER, CASH_OUT).
No model training, no validation, and no test data are used.
"""

import argparse
import sys
import time
from pathlib import Path

# Ensure project root is in sys.path
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import numpy as np
import pandas as pd

from src.common import get_project_root, setup_logger

NEEDED_COLUMNS = [
    "step",
    "type",
    "amount",
    "oldbalanceOrg",
    "oldbalanceDest",
    "isFraud",
]


def load_diagnostic_data(mode: str = "full") -> pd.DataFrame:
    """Load only the required columns from the processed parquet."""
    root = get_project_root()
    parquet_filename = "paysim.parquet" if mode == "full" else "paysim_sample.parquet"
    parquet_path = root / "data" / "processed" / parquet_filename

    if not parquet_path.is_file():
        raise FileNotFoundError(f"Processed file not found: {parquet_path}")

    df = pd.read_parquet(parquet_path, columns=NEEDED_COLUMNS)
    return df


def run_diagnostic(mode: str = "full") -> pd.DataFrame:
    """Run diagnostic feature analysis on TRAIN partition (steps 1-520 only)."""
    start_time = time.time()
    root = get_project_root()
    results_dir = root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger("exp1b_diagnostic", "exp1b_diagnostic.log")
    logger.info("Starting exp1b_diagnostic in mode='%s'...", mode)

    # 1. Load data
    df = load_diagnostic_data(mode=mode)
    logger.info("Loaded %d rows total.", len(df))

    # 2. Restrict strictly to TRAIN partition (steps 1-520 only)
    train_df = df[df["step"] <= 520].copy()
    logger.info(
        "Filtered to TRAIN partition (step <= 520): %d rows (max step=%d).",
        len(train_df),
        int(train_df["step"].max()),
    )

    # 3. Analyze TRANSFER, CASH_OUT, and Combined
    groups = [
        ("TRANSFER", 1, "Fraud"),
        ("TRANSFER", 0, "Legitimate"),
        ("CASH_OUT", 1, "Fraud"),
        ("CASH_OUT", 0, "Legitimate"),
        ("ALL (TRANSFER+CASH_OUT)", 1, "Fraud"),
        ("ALL (TRANSFER+CASH_OUT)", 0, "Legitimate"),
    ]

    records = []
    for tx_type, fraud_val, label in groups:
        if tx_type == "ALL (TRANSFER+CASH_OUT)":
            mask = train_df["type"].isin(["TRANSFER", "CASH_OUT"]) & (train_df["isFraud"] == fraud_val)
        else:
            mask = (train_df["type"] == tx_type) & (train_df["isFraud"] == fraud_val)

        sub = train_df[mask]
        n_rows = len(sub)

        if n_rows > 0:
            cond1 = sub["amount"] >= sub["oldbalanceOrg"]
            cond2 = sub["oldbalanceDest"] == 0.0
            cond_both = cond1 & cond2
            ratio = sub["amount"] / (sub["oldbalanceOrg"] + np.float32(1.0))

            share_amount_ge_oldbalanceOrg = float(cond1.mean())
            share_oldbalanceDest_zero = float(cond2.mean())
            share_both_conditions = float(cond_both.mean())
            median_amount_ratio = float(ratio.median())
        else:
            share_amount_ge_oldbalanceOrg = 0.0
            share_oldbalanceDest_zero = 0.0
            share_both_conditions = 0.0
            median_amount_ratio = 0.0

        records.append(
            {
                "type": tx_type,
                "is_fraud": fraud_val,
                "label": label,
                "row_count": n_rows,
                "share_amount_ge_oldbalanceOrg": share_amount_ge_oldbalanceOrg,
                "share_oldbalanceDest_zero": share_oldbalanceDest_zero,
                "share_both_conditions": share_both_conditions,
                "median_amount_ratio": median_amount_ratio,
            }
        )

    results_df = pd.DataFrame(records)
    csv_path = results_dir / "exp1b_diagnostic.csv"
    results_df.to_csv(csv_path, index=False)
    logger.info("Saved diagnostic results to %s", csv_path)

    elapsed = time.time() - start_time
    logger.info("Diagnostic completed in %.2f seconds.", elapsed)

    # Print results to stdout
    print("\n" + "=" * 90)
    print("        EXPERIMENT 1B: DIAGNOSTIC FEATURE ANALYSIS (TRAIN STEPS 1-520 ONLY)")
    print("=" * 90)
    print(results_df.to_string(index=False))
    print("=" * 90 + "\n")

    return results_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Experiment 1b: Diagnostic Feature Analysis.")
    parser.add_argument(
        "--mode",
        choices=["sample", "full"],
        default="sample",
        help="Execution mode (default: 'sample').",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_diagnostic(mode=args.mode)


if __name__ == "__main__":
    main()

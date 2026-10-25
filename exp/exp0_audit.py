"""Experiment 0: Dataset Audit & Validation against Reference Targets.

Audits:
- Total rows, fraud count, and fraud rate
- Rows and fraud per transaction type
- Duplicate rows count
- Rows and fraud per temporal split partition (train 1-520, val 521-631, test 632-743)
- Validates against dissertation benchmarks (PASS/MISMATCH)
- Saves results/exp0_audit.csv
- Generates figures/exp0_class_distribution.png and figures/exp0_fraud_by_type.png
"""

import argparse
import sys
import time
from pathlib import Path

# Ensure project root is in sys.path
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.common import get_project_root, load_config, load_data, setup_logger, split_temporal

REFERENCE_TARGETS = {
    "total_rows": 6_362_620,
    "fraud_count": 8_213,
    "fraud_rate_pct": 0.129082,
    "cash_out_fraud": 4_116,
    "transfer_fraud": 4_097,
    "train_rows": 6_082_007,
    "val_rows": 191_147,
    "test_rows": 89_466,
}


def run_audit(mode: str = "full") -> pd.DataFrame:
    """Run data audit and generate validation table and plots."""
    start_time = time.time()
    root = get_project_root()
    results_dir = root / "results"
    figures_dir = root / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger("exp0_audit", "exp0_audit.log")
    logger.info("Starting exp0_audit in mode='%s'...", mode)

    # 1. Load data
    df = load_data(mode=mode)
    total_rows = len(df)
    logger.info("Loaded %d rows for audit.", total_rows)

    # 2. Overall counts and rates
    fraud_count = int(df["isFraud"].sum())
    fraud_rate_pct = (fraud_count / total_rows) * 100.0 if total_rows > 0 else 0.0

    # 3. Duplicate rows check
    logger.info("Checking for duplicate rows...")
    duplicate_rows = int(df.duplicated().sum())
    logger.info("Duplicate rows count: %d", duplicate_rows)

    # 4. Rows and fraud per transaction type
    type_summary = (
        df.groupby("type", observed=False)["isFraud"]
        .agg(total_rows="count", fraud_count="sum")
        .reset_index()
    )
    type_summary["fraud_rate_pct"] = (
        type_summary["fraud_count"] / type_summary["total_rows"] * 100.0
    )

    type_counts = {row["type"]: int(row["total_rows"]) for _, row in type_summary.iterrows()}
    type_frauds = {row["type"]: int(row["fraud_count"]) for _, row in type_summary.iterrows()}

    cash_out_fraud = type_frauds.get("CASH_OUT", 0)
    transfer_fraud = type_frauds.get("TRANSFER", 0)

    # 5. Rows and fraud per temporal split
    train_df, val_df, test_df = split_temporal(df)
    train_rows = len(train_df)
    val_rows = len(val_df)
    test_rows = len(test_df)

    train_fraud = int(train_df["isFraud"].sum())
    val_fraud = int(val_df["isFraud"].sum())
    test_fraud = int(test_df["isFraud"].sum())

    train_fraud_rate = (train_fraud / train_rows * 100.0) if train_rows > 0 else 0.0
    val_fraud_rate = (val_fraud / val_rows * 100.0) if val_rows > 0 else 0.0
    test_fraud_rate = (test_fraud / test_rows * 100.0) if test_rows > 0 else 0.0

    # 6. Evaluation against targets
    records = []

    def evaluate_metric(name: str, observed: float, target: float | None, is_rate: bool = False):
        if target is None or mode != "full":
            status = "N/A (sample mode)" if mode != "full" else "INFO"
            diff = 0.0
        else:
            if is_rate:
                diff = observed - target
                status = "PASS" if abs(diff) < 0.0001 else "MISMATCH"
            else:
                diff = observed - target
                status = "PASS" if diff == 0 else "MISMATCH"

        records.append(
            {
                "metric": name,
                "observed": observed,
                "target": target if target is not None else np.nan,
                "diff": diff,
                "status": status,
            }
        )

    evaluate_metric("total_rows", total_rows, REFERENCE_TARGETS["total_rows"])
    evaluate_metric("fraud_count", fraud_count, REFERENCE_TARGETS["fraud_count"])
    evaluate_metric("fraud_rate_pct", fraud_rate_pct, REFERENCE_TARGETS["fraud_rate_pct"], is_rate=True)
    evaluate_metric("cash_out_fraud", cash_out_fraud, REFERENCE_TARGETS["cash_out_fraud"])
    evaluate_metric("transfer_fraud", transfer_fraud, REFERENCE_TARGETS["transfer_fraud"])
    evaluate_metric("train_rows", train_rows, REFERENCE_TARGETS["train_rows"])
    evaluate_metric("val_rows", val_rows, REFERENCE_TARGETS["val_rows"])
    evaluate_metric("test_rows", test_rows, REFERENCE_TARGETS["test_rows"])
    evaluate_metric("duplicate_rows", duplicate_rows, 0)

    # Additional contextual metrics (not reference target gates, but audit details)
    evaluate_metric("train_fraud_count", train_fraud, None)
    evaluate_metric("train_fraud_rate_pct", train_fraud_rate, None)
    evaluate_metric("val_fraud_count", val_fraud, None)
    evaluate_metric("val_fraud_rate_pct", val_fraud_rate, None)
    evaluate_metric("test_fraud_count", test_fraud, None)
    evaluate_metric("test_fraud_rate_pct", test_fraud_rate, None)

    for t in ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]:
        evaluate_metric(f"rows_{t}", type_counts.get(t, 0), None)
        evaluate_metric(f"fraud_{t}", type_frauds.get(t, 0), None)

    elapsed = time.time() - start_time
    evaluate_metric("runtime_seconds", elapsed, None)

    results_df = pd.DataFrame(records)
    csv_path = results_dir / "exp0_audit.csv"
    results_df.to_csv(csv_path, index=False)
    logger.info("Saved audit results to %s", csv_path)

    # 7. Print Console Audit Report
    print("\n" + "=" * 70)
    print("                 EXPERIMENT 0: DATASET AUDIT REPORT")
    print("=" * 70)
    for _, r in results_df.iterrows():
        target_str = f"{r['target']:>12}" if pd.notna(r["target"]) else "         N/A"
        obs_str = f"{r['observed']:>14.4f}" if isinstance(r["observed"], float) else f"{r['observed']:>14}"
        print(f"{r['metric']:<26} | Observed: {obs_str} | Target: {target_str} | Status: {r['status']}")
    print("=" * 70 + "\n")

    # 8. Render and save Figures
    logger.info("Generating plots...")

    # Plot 1: Class Distribution
    fig, ax = plt.subplots(figsize=(6, 5))
    classes = ["Legitimate (0)", "Fraud (1)"]
    counts = [total_rows - fraud_count, fraud_count]
    colors = ["#2b5c8f", "#d95f02"]
    bars = ax.bar(classes, counts, color=colors, width=0.5)
    ax.set_yscale("log")
    ax.set_ylabel("Transaction Count (Log Scale)")
    ax.set_title("PaySim Class Distribution (Log Scale)")
    for bar in bars:
        h = bar.get_height()
        ax.annotate(
            f"{int(h):,}",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
        )
    plt.tight_layout()
    plot1_path = figures_dir / "exp0_class_distribution.png"
    plt.savefig(plot1_path, dpi=200)
    plt.close()
    logger.info("Saved class distribution plot to %s", plot1_path)

    # Plot 2: Fraud Count by Transaction Type
    fig, ax = plt.subplots(figsize=(8, 5))
    types = list(type_frauds.keys())
    f_counts = [type_frauds[t] for t in types]
    bar_colors = ["#7570b3" if c == 0 else "#e7298a" for c in f_counts]
    bars = ax.bar(types, f_counts, color=bar_colors, width=0.5)
    ax.set_ylabel("Fraud Cases")
    ax.set_title("PaySim Fraud Count by Transaction Type")
    for bar in bars:
        h = bar.get_height()
        ax.annotate(
            f"{int(h):,}",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
        )
    plt.tight_layout()
    plot2_path = figures_dir / "exp0_fraud_by_type.png"
    plt.savefig(plot2_path, dpi=200)
    plt.close()
    logger.info("Saved fraud by type plot to %s", plot2_path)

    return results_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit PaySim dataset.")
    parser.add_argument(
        "--mode",
        choices=["sample", "full"],
        default="sample",
        help="Evaluation mode: 'sample' or 'full'.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_audit(mode=args.mode)


if __name__ == "__main__":
    main()

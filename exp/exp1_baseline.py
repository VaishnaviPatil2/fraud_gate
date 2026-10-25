"""Experiment 1: Supervised XGBoost Baselines (STRICT vs. EXTENDED).

Evaluates:
- Feature sets: STRICT and EXTENDED
- Seeds: 0 to 4
- XGBoost: n_estimators=300, max_depth=6, lr=0.08, subsample=0.9, colsample=0.9,
  min_child_weight=1, tree_method="hist", binary:logistic, aucpr, n_jobs=2.
- Two variants: unweighted (scale_pos_weight=1) and weighted (scale_pos_weight=neg/pos).
- Platt scaling fitted strictly on validation scores.
- Thresholds chosen on validation: precision >= 0.9, precision >= 0.5, recall >= 0.8.
- Uncertainty u = 1 - max(p, 1-p) on calibrated p with selective review policy.
- Outputs:
    - results/exp1_results.csv (per-seed metrics + mean/std rows)
    - results/exp1_features.txt (feature importances for seed 0)
    - figures/exp1_pr_curves.png (PR curves: STRICT vs. EXTENDED)
    - figures/exp1_reliability.png (Reliability diagrams: raw vs. calibrated)
    - figures/exp1_confusion_matrix.png (Confusion matrix at precision >= 0.5)
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, precision_recall_curve
from xgboost import XGBClassifier

from src.common import get_project_root, load_config, load_data, setup_logger, split_temporal
from src.features import extract_features_and_target
from src.metrics import (
    brier,
    fp_at_recall,
    mean_std_over_seeds,
    plot_reliability,
    pr_auc,
    precision_recall_f1_at_threshold,
    roc_auc,
    threshold_for_precision,
    threshold_for_recall,
)


def run_experiment(mode: str = "sample") -> pd.DataFrame:
    """Execute Experiment 1 across feature sets, variants, and seeds."""
    total_start_time = time.time()
    root = get_project_root()
    results_dir = root / "results"
    figures_dir = root / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger("exp1_baseline", "exp1_baseline.log")
    logger.info("Starting exp1_baseline in mode='%s'...", mode)

    cfg = load_config()
    seeds = [int(s) for s in cfg.get("seeds", [0, 1, 2, 3, 4])]
    feature_sets = ["strict", "extended"]
    variants = ["unweighted", "weighted"]

    # 1. Load data
    raw_df = load_data(mode=mode)
    logger.info("Loaded %d rows for Experiment 1.", len(raw_df))

    # 2. Split temporally strictly by step (Rule 1)
    train_df, val_df, test_df = split_temporal(raw_df)
    logger.info(
        "Temporal split - Train: %d (fraud=%d), Val: %d (fraud=%d), Test: %d (fraud=%d)",
        len(train_df),
        int(train_df["isFraud"].sum()),
        len(val_df),
        int(val_df["isFraud"].sum()),
        len(test_df),
        int(test_df["isFraud"].sum()),
    )

    results_records = []
    feature_importance_records = {}

    # Store data for plots (seed 0)
    plot_cache = {}

    for f_set in feature_sets:
        # Precompute features for each partition
        X_train, y_train = extract_features_and_target(train_df, feature_set=f_set)
        X_val, y_val = extract_features_and_target(val_df, feature_set=f_set)
        X_test, y_test = extract_features_and_target(test_df, feature_set=f_set)

        n_pos = int(y_train.sum())
        n_neg = int(len(y_train) - n_pos)
        scale_pos = float(n_neg / n_pos) if n_pos > 0 else 1.0

        for variant in variants:
            spw = scale_pos if variant == "weighted" else 1.0

            for seed in seeds:
                run_start = time.time()
                logger.info(
                    "Running config: feature_set=%s, variant=%s, seed=%d (scale_pos_weight=%.2f)...",
                    f_set,
                    variant,
                    seed,
                    spw,
                )

                model = XGBClassifier(
                    n_estimators=300,
                    max_depth=6,
                    learning_rate=0.08,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    min_child_weight=1,
                    tree_method="hist",
                    objective="binary:logistic",
                    eval_metric="aucpr",
                    scale_pos_weight=spw,
                    random_state=seed,
                    n_jobs=2,
                )

                # Fit on Train partition only
                model.fit(X_train, y_train)

                # Predict raw probabilities on Validation
                p_val_raw = model.predict_proba(X_val)[:, 1]

                # Fit Platt scaling strictly on Validation scores (Rule 3)
                calibrator = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
                calibrator.fit(p_val_raw.reshape(-1, 1), y_val)

                p_val_cal = calibrator.predict_proba(p_val_raw.reshape(-1, 1))[:, 1]

                # Thresholds chosen on Validation only (Rule 3)
                th_p90 = threshold_for_precision(y_val, p_val_cal, target_precision=0.9)
                th_p50 = threshold_for_precision(y_val, p_val_cal, target_precision=0.5)
                th_r80 = threshold_for_recall(y_val, p_val_cal, target_recall=0.8)

                # Validation uncertainty cutoff for selective policy: top 5% most uncertain
                u_val = 1.0 - np.maximum(p_val_cal, 1.0 - p_val_cal)
                u_cut = float(np.quantile(u_val, 0.95))

                # Inference on Test partition (Rule 3: Test used only for final measurement)
                p_test_raw = model.predict_proba(X_test)[:, 1]
                p_test_cal = calibrator.predict_proba(p_test_raw.reshape(-1, 1))[:, 1]

                # Test Metrics
                test_pr_auc = pr_auc(y_test, p_test_cal)
                test_roc_auc = roc_auc(y_test, p_test_cal)
                test_brier_raw = brier(y_test, p_test_raw)
                test_brier_cal = brier(y_test, p_test_cal)

                prec_p90, rec_p90, f1_p90 = precision_recall_f1_at_threshold(y_test, p_test_cal, th_p90)
                prec_p50, rec_p50, f1_p50 = precision_recall_f1_at_threshold(y_test, p_test_cal, th_p50)

                test_fp_r80 = fp_at_recall(y_val, p_val_cal, y_test, p_test_cal, target_recall=0.8)

                # Selective review policy on Test
                u_test = 1.0 - np.maximum(p_test_cal, 1.0 - p_test_cal)
                review_mask = (u_test >= u_cut)
                review_rate = float(review_mask.mean())

                auto_mask = ~review_mask
                if auto_mask.sum() > 0:
                    auto_y_true = y_test[auto_mask]
                    auto_p_test = p_test_cal[auto_mask]
                    auto_prec, auto_rec, _ = precision_recall_f1_at_threshold(auto_y_true, auto_p_test, th_p50)
                else:
                    auto_prec, auto_rec = 0.0, 0.0

                run_elapsed = time.time() - run_start

                record = {
                    "feature_set": f_set,
                    "variant": variant,
                    "seed": seed,
                    "pr_auc": test_pr_auc,
                    "roc_auc": test_roc_auc,
                    "brier_raw": test_brier_raw,
                    "brier_cal": test_brier_cal,
                    "val_thresh_p90": th_p90,
                    "test_prec_p90": prec_p90,
                    "test_rec_p90": rec_p90,
                    "test_f1_p90": f1_p90,
                    "val_thresh_p50": th_p50,
                    "test_prec_p50": prec_p50,
                    "test_rec_p50": rec_p50,
                    "test_f1_p50": f1_p50,
                    "val_thresh_r80": th_r80,
                    "test_fp_at_r80": test_fp_r80,
                    "val_u_cut": u_cut,
                    "test_review_rate": review_rate,
                    "test_auto_recall": auto_rec,
                    "test_auto_precision": auto_prec,
                    "runtime_sec": run_elapsed,
                }
                results_records.append(record)

                # Cache seed 0 for plots and feature importance
                if seed == 0:
                    key = f"{f_set}_{variant}"
                    plot_cache[key] = {
                        "y_test": y_test.to_numpy(),
                        "p_test_raw": p_test_raw,
                        "p_test_cal": p_test_cal,
                        "th_p50": th_p50,
                    }
                    importances = model.feature_importances_
                    feature_names = list(X_train.columns)
                    feat_dict = dict(zip(feature_names, importances))
                    feature_importance_records[key] = sorted(
                        feat_dict.items(), key=lambda x: x[1], reverse=True
                    )

    # Convert results to DataFrame
    raw_results_df = pd.DataFrame(results_records)

    # Compute mean and std over seeds
    metric_cols = [
        "pr_auc",
        "roc_auc",
        "brier_raw",
        "brier_cal",
        "val_thresh_p90",
        "test_prec_p90",
        "test_rec_p90",
        "test_f1_p90",
        "val_thresh_p50",
        "test_prec_p50",
        "test_rec_p50",
        "test_f1_p50",
        "val_thresh_r80",
        "test_fp_at_r80",
        "val_u_cut",
        "test_review_rate",
        "test_auto_recall",
        "test_auto_precision",
        "runtime_sec",
    ]

    mean_std_df = mean_std_over_seeds(
        raw_results_df,
        group_cols=["feature_set", "variant"],
        metric_cols=metric_cols,
    )

    # Save results to CSV (individual seed rows followed by aggregated summary)
    csv_path = results_dir / "exp1_results.csv"

    # Create unified CSV containing individual seeds plus labeled mean and std summary rows
    summary_rows = []
    for (f_set, var), grp in raw_results_df.groupby(["feature_set", "variant"], observed=False):
        mean_row = {"feature_set": f_set, "variant": var, "seed": "mean"}
        std_row = {"feature_set": f_set, "variant": var, "seed": "std"}
        for m in metric_cols:
            mean_row[m] = grp[m].mean()
            std_row[m] = grp[m].std()
        summary_rows.append(mean_row)
        summary_rows.append(std_row)

    combined_df = pd.concat([raw_results_df, pd.DataFrame(summary_rows)], ignore_index=True)
    combined_df.to_csv(csv_path, index=False)
    logger.info("Saved complete experiment 1 results to %s", csv_path)

    # Save feature importances to txt
    feat_txt_path = results_dir / "exp1_features.txt"
    with open(feat_txt_path, "w", encoding="utf-8") as f:
        f.write("EXPERIMENT 1: FEATURE IMPORTANCES (Seed 0)\n")
        f.write("=" * 60 + "\n\n")
        for key, feat_list in feature_importance_records.items():
            f.write(f"Configuration: {key}\n")
            f.write("-" * 40 + "\n")
            for rank, (feat_name, imp) in enumerate(feat_list, start=1):
                f.write(f"{rank:2d}. {feat_name:<32} : {imp:.6f}\n")
            f.write("\n")
    logger.info("Saved feature importances to %s", feat_txt_path)

    # 3. Generate Visualizations
    logger.info("Rendering figures...")

    # Plot 1: PR Curves (STRICT vs EXTENDED for Seed 0)
    fig, ax = plt.subplots(figsize=(7, 6))
    for key, pdata in plot_cache.items():
        if "unweighted" in key:
            prec_c, rec_c, _ = precision_recall_curve(pdata["y_test"], pdata["p_test_cal"])
            auc_val = pr_auc(pdata["y_test"], pdata["p_test_cal"])
            label = f"{key.upper()} (AUC = {auc_val:.4f})"
            color = "#1f77b4" if "strict" in key else "#ff7f0e"
            linestyle = "--" if "strict" in key else "-"
            ax.plot(rec_c, prec_c, color=color, linestyle=linestyle, label=label, linewidth=2)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves: STRICT vs. EXTENDED (Seed 0, Unweighted)")
    ax.legend(loc="lower left")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    plt.tight_layout()
    fig1_path = figures_dir / "exp1_pr_curves.png"
    plt.savefig(fig1_path, dpi=200)
    plt.close()
    logger.info("Saved PR curves to %s", fig1_path)

    # Plot 2: Reliability Diagram (Seed 0, EXTENDED unweighted)
    if "extended_unweighted" in plot_cache:
        ext_data = plot_cache["extended_unweighted"]
        fig, ax = plt.subplots(figsize=(6, 6))
        plot_reliability(
            y_true=ext_data["y_test"],
            p_raw=ext_data["p_test_raw"],
            p_cal=ext_data["p_test_cal"],
            n_bins=10,
            ax=ax,
            title="Reliability Diagram: Raw vs. Platt Calibrated (EXTENDED, Seed 0)",
        )
        plt.tight_layout()
        fig2_path = figures_dir / "exp1_reliability.png"
        plt.savefig(fig2_path, dpi=200)
        plt.close()
        logger.info("Saved reliability diagram to %s", fig2_path)

    # Plot 3: Confusion Matrix at precision >= 0.5 threshold (Seed 0, EXTENDED unweighted)
    if "extended_unweighted" in plot_cache:
        ext_data = plot_cache["extended_unweighted"]
        y_test_arr = ext_data["y_test"]
        p_cal_arr = ext_data["p_test_cal"]
        th_p50 = ext_data["th_p50"]
        y_hat = (p_cal_arr >= th_p50).astype(int)

        cm = confusion_matrix(y_test_arr, y_hat)
        fig, ax = plt.subplots(figsize=(5, 5))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Legit", "Fraud"])
        disp.plot(cmap="Blues", ax=ax, values_format="d", colorbar=False)
        ax.set_title(f"Confusion Matrix (EXTENDED Seed 0, thresh={th_p50:.4f})")
        plt.tight_layout()
        fig3_path = figures_dir / "exp1_confusion_matrix.png"
        plt.savefig(fig3_path, dpi=200)
        plt.close()
        logger.info("Saved confusion matrix to %s", fig3_path)

    total_elapsed = time.time() - total_start_time
    logger.info("Experiment 1 completed in %.2f seconds.", total_elapsed)

    return combined_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Experiment 1: XGBoost Baselines.")
    parser.add_argument(
        "--mode",
        choices=["sample", "full"],
        default="sample",
        help="Execution mode (default: 'sample').",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_experiment(mode=args.mode)


if __name__ == "__main__":
    main()

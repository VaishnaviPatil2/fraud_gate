"""Experiment 1c: Conformal-Style Risk-Aware Routing and Prediction Sets.

Evaluates:
- Feature sets: STRICT (MAIN contract) and EXTENDED (SECONDARY "balance-aware" contract)
- Seeds: 0, 1, 2
- Model: STRICT / EXTENDED XGBoost from exp1 (reused specification: n_estimators=300,
  max_depth=6, lr=0.08, subsample=0.9, colsample=0.9, min_child_weight=1, tree_method="hist")
- Two variants: unweighted and weighted
- Platt-calibrated probabilities on validation
- Alpha levels: {0.05, 0.10} from config.yaml
- Part 1: Split conformal prediction sets (s = 1 - p(true class), q_hat on validation)
- Part 2: Conformal-style risk control for missed fraud (score cut where val FNR <= alpha)
- Part 3: Plain uncertainty heuristic comparison at matched review rate
- Output: results/exp1c_conformal.csv
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
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from src.common import get_project_root, load_config, load_data, setup_logger, split_temporal
from src.conformal import (
    compute_nonconformity,
    conformal_quantile,
    evaluate_conformal_sets,
    evaluate_risk_control,
    evaluate_uncertainty_benchmark,
    risk_control_threshold,
)
from src.features import extract_features_and_target
from src.metrics import threshold_for_precision


def train_and_calibrate_xgb(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    variant: str = "unweighted",
    seed: int = 0,
) -> tuple[XGBClassifier, LogisticRegression]:
    """Train XGBoost matching exp1 specification and calibrate via Platt scaling."""
    n_pos = int(y_train.sum())
    n_neg = int(len(y_train) - n_pos)
    spw = float(n_neg / n_pos) if (variant == "weighted" and n_pos > 0) else 1.0

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
    model.fit(X_train, y_train)

    p_val_raw = model.predict_proba(X_val)[:, 1]
    calibrator = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
    calibrator.fit(p_val_raw.reshape(-1, 1), y_val)

    return model, calibrator


def run_conformal_experiment(mode: str = "sample") -> pd.DataFrame:
    """Execute conformal prediction sets and risk-control evaluation."""
    start_time = time.time()
    root = get_project_root()
    results_dir = root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger("exp1c_conformal", "exp1c_conformal.log")
    logger.info("Starting exp1c_conformal in mode='%s'...", mode)

    cfg = load_config()
    alpha_levels = [float(a) for a in cfg.get("conformal", {}).get("alpha_values", [0.05, 0.10])]
    seeds = [0, 1, 2]  # Seeds 0-2 per task requirement
    feature_sets = ["strict", "extended"]
    variants = ["unweighted", "weighted"]

    # 1. Load data and split temporally (Rule 1)
    df = load_data(mode=mode)
    train_df, val_df, test_df = split_temporal(df)
    logger.info(
        "Temporal Split - Train: %d, Val: %d, Test: %d",
        len(train_df),
        len(val_df),
        len(test_df),
    )

    records = []

    for f_set in feature_sets:
        contract_role = "main" if f_set == "strict" else "secondary (balance-aware)"
        X_train, y_train = extract_features_and_target(train_df, feature_set=f_set)
        X_val, y_val = extract_features_and_target(val_df, feature_set=f_set)
        X_test, y_test = extract_features_and_target(test_df, feature_set=f_set)

        for variant in variants:
            for seed in seeds:
                logger.info(
                    "Fitting model: feature_set=%s (%s), variant=%s, seed=%d...",
                    f_set,
                    contract_role,
                    variant,
                    seed,
                )
                model, calibrator = train_and_calibrate_xgb(
                    X_train, y_train, X_val, y_val, variant=variant, seed=seed
                )

                # Validation calibrated probabilities (Rule 3)
                p_val_raw = model.predict_proba(X_val)[:, 1]
                p_val_cal = calibrator.predict_proba(p_val_raw.reshape(-1, 1))[:, 1]

                # Decision threshold from validation (precision >= 0.5)
                th_p50 = threshold_for_precision(y_val, p_val_cal, target_precision=0.5)

                # Nonconformity scores on validation
                s_val = compute_nonconformity(y_val, p_val_cal)

                # Test calibrated probabilities
                p_test_raw = model.predict_proba(X_test)[:, 1]
                p_test_cal = calibrator.predict_proba(p_test_raw.reshape(-1, 1))[:, 1]

                for alpha in alpha_levels:
                    # ---------------------------------------------------------
                    # Part 1: Split Conformal Prediction Sets
                    # ---------------------------------------------------------
                    q_hat = conformal_quantile(s_val, alpha=alpha)
                    conf_res = evaluate_conformal_sets(y_test, p_test_cal, q_hat)

                    # ---------------------------------------------------------
                    # Part 2: Conformal-Style Risk Control for Missed Fraud
                    # ---------------------------------------------------------
                    tau_allow = risk_control_threshold(y_val, p_val_cal, alpha=alpha)
                    # Validation FNR to verify it meets alpha on validation
                    val_fraud = y_val == 1
                    val_fnr = float((p_val_cal[val_fraud] < tau_allow).mean()) if val_fraud.sum() > 0 else 0.0

                    risk_res = evaluate_risk_control(
                        y_test, p_test_cal, tau_allow, decision_thresh=th_p50
                    )

                    # Validation review rate achieved by this cut
                    val_review_rate = float((p_val_cal >= tau_allow).mean())

                    # ---------------------------------------------------------
                    # Part 3: Plain Uncertainty Heuristic Comparison
                    # ---------------------------------------------------------
                    unc_res = evaluate_uncertainty_benchmark(
                        y_val=y_val,
                        p_val_cal=p_val_cal,
                        y_test=y_test,
                        p_test_cal=p_test_cal,
                        target_review_rate=val_review_rate,
                        decision_thresh=th_p50,
                    )

                    record = {
                        "routing_layer": "conformal-style risk-aware routing",
                        "feature_set": f_set,
                        "contract_role": contract_role,
                        "variant": variant,
                        "seed": seed,
                        "alpha": alpha,
                        # Part 1: Prediction Sets
                        "conf_q_hat": conf_res["q_hat"],
                        "conf_coverage": conf_res["empirical_coverage"],
                        "conf_share_ambiguous": conf_res["share_ambiguous"],
                        "conf_share_empty": conf_res["share_empty"],
                        "conf_share_singleton": conf_res["share_singleton"],
                        "conf_review_rate": conf_res["review_rate"],
                        "conf_auto_recall": conf_res["auto_recall"],
                        "conf_auto_precision": conf_res["auto_precision"],
                        # Part 2: Risk Control (Score Cut)
                        "risk_tau_allow": risk_res["tau_alpha"],
                        "risk_val_fnr": val_fnr,
                        "risk_test_fnr": risk_res["realised_fnr"],
                        "risk_test_review_rate": risk_res["review_rate"],
                        "risk_test_auto_allow_rate": 1.0 - risk_res["review_rate"],
                        "risk_test_auto_purity": risk_res["auto_allow_purity"],
                        "risk_test_auto_recall": risk_res["auto_recall"],
                        "risk_test_auto_precision": risk_res["auto_precision"],
                        # Part 3: Plain Uncertainty at Matched Review Rate
                        "unc_u_cut": unc_res["u_cut"],
                        "unc_test_review_rate": unc_res["test_review_rate"],
                        "unc_test_fnr": unc_res["realised_fnr"],
                        "unc_test_auto_recall": unc_res["auto_recall"],
                        "unc_test_auto_precision": unc_res["auto_precision"],
                    }
                    records.append(record)

    results_df = pd.DataFrame(records)
    csv_path = results_dir / "exp1c_conformal.csv"
    results_df.to_csv(csv_path, index=False)
    logger.info("Saved conformal results to %s", csv_path)

    elapsed = time.time() - start_time
    logger.info("Experiment 1c completed in %.2f seconds.", elapsed)

    # Print to console
    print("\n" + "=" * 100)
    print("                  EXPERIMENT 1C: CONFORMAL-STYLE ROUTING LAYER RESULTS")
    print("=" * 100)
    print(results_df.to_string(index=False))
    print("=" * 100 + "\n")

    return results_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Experiment 1c: Conformal-Style Routing.")
    parser.add_argument(
        "--mode",
        choices=["sample", "full"],
        default="sample",
        help="Execution mode (default: 'sample').",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_conformal_experiment(mode=args.mode)


if __name__ == "__main__":
    main()

PROJECT: M.Tech dissertation "Evidence-Gated Risk-Aware Fraud Prevention, Detection and Recovery for Digital Payment Systems". PaySim (primary), IEEE-CIS (secondary).

RULES
1. Temporal split ONLY, by 'step': train 1-520, validation 521-631, test 632-743. All rows of a step stay in one partition. Never a random split.
2. Never use newbalanceOrig, newbalanceDest, isFlaggedFraud or isFraud as features. Account names are used only to build graph/history features.
3. Thresholds, calibration, conformal quantiles, hyperparameters and adaptation settings are fitted on VALIDATION (or earlier training steps) only. Test is used only for final measurement.
4. Seeds come from config.yaml. Save every result to results/*.csv. NEVER invent, estimate, round by hand or hardcode a reported number; every number is produced by code and read from a saved file. Numbers in the old paper (e.g. PR-AUC 0.3277) are reproduction targets only; never tune toward them.
5. Hardware: 8 GB RAM (about 6 GB usable), CPU only, 2 cores. float32, category/integer codes, parquet, usecols, vectorized pandas (no row loops), n_jobs<=2. Every script accepts --mode sample|full (default sample). Run full ONLY when I say "run full".
6. Do not modify files from earlier tasks unless I say so. Reuse via import.
7. After each task: git commit with a clear message; print the full contents of every result CSV and the runtime; list files created. Do not interpret results. State explicitly anything that failed, was skipped or assumed.
8. If a requirement is unclear, ask me instead of guessing.
9. Python 3.12, Windows 11, PowerShell. Use pathlib; no Linux-only commands or shell syntax.
10. Every script wraps its main code in if __name__ == "__main__": (Windows multiprocessing).
11. Print progress and write logs to logs/. On a memory error stop and report; never silently shrink the data.
12. Install nothing beyond requirements.txt unless the task text explicitly allows it; tell me what was installed and the version.
13. Any simulated or assumed quantity (step-up success, recovery success, delays, costs) must have "assumed" or "simulated" in its column name or label.
14. Re-implemented baselines are labelled "our re-implementation on our split, not an exact reproduction".
15. Report weak or negative results exactly as they come out. Never adjust code or data to make a target pass.
16. MAIN feature contract = STRICT (plus graph features once available). EXTENDED (adds oldbalanceOrg, oldbalanceDest and derived ratios) is a SECONDARY "balance-aware" contract, reported separately and labelled as sensitive to a PaySim simulation artifact (fraud drains the origin account; see results/exp1b_diagnostic.csv). Where a task says EXTENDED, use STRICT as the main set unless the task text says "also run EXTENDED"; then report EXTENDED as secondary rows.

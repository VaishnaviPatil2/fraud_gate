"""Master orchestration script for running all dissertation experiments sequentially.

Usage:
    python run_all.py [--mode sample|full]
"""

import argparse
import logging
import sys
from pathlib import Path


def setup_logger(logs_dir: Path) -> logging.Logger:
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = logs_dir / "run_all.log"

    logger = logging.getLogger("run_all")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run all dissertation experiments in sequence."
    )
    parser.add_argument(
        "--mode",
        choices=["sample", "full"],
        default="sample",
        help="Execution mode: sample (fast, default) or full (full dataset).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workspace_root = Path(__file__).resolve().parent
    logs_dir = workspace_root / "logs"
    logger = setup_logger(logs_dir)

    logger.info("Starting run_all orchestration script.")
    logger.info("Mode: %s", args.mode)
    logger.info("Workspace root: %s", workspace_root)

    # Experiment modules will be called here as each is implemented.
    logger.info("All experiment stubs are currently unpopulated.")
    logger.info("Master runner initialized successfully.")


if __name__ == "__main__":
    main()

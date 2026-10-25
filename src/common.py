"""Common utilities, configuration loader, data loader, temporal splitter, and seed management.

Adheres strictly to project rules:
- Temporal splitting only (train 1-520, val 521-631, test 632-743).
- Seed management via config.yaml.
- Safe path resolution using pathlib.
"""

import logging
import random
from pathlib import Path
from typing import Any, Tuple

import numpy as np
import pandas as pd
import yaml


def get_project_root() -> Path:
    """Return the absolute path to the project root directory."""
    return Path(__file__).resolve().parent.parent


def setup_logger(name: str, log_filename: str | None = None) -> logging.Logger:
    """Configure a logger that outputs to both console and a file in logs/."""
    root = get_project_root()
    logs_dir = root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if log_filename:
        fh = logging.FileHandler(logs_dir / log_filename, encoding="utf-8")
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def load_config(config_path: Path | str | None = None) -> dict[str, Any]:
    """Load configuration YAML file."""
    if config_path is None:
        config_path = get_project_root() / "config.yaml"
    else:
        config_path = Path(config_path)

    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def set_seed(seed: int | None = None) -> int:
    """Set random seed across random and numpy.

    If seed is None, the first seed from config.yaml is used.
    """
    if seed is None:
        cfg = load_config()
        seed = int(cfg["seeds"][0])

    random.seed(seed)
    np.random.seed(seed)
    return seed


def load_data(
    mode: str = "sample",
    data_dir: Path | str | None = None,
) -> pd.DataFrame:
    """Load PaySim dataset in either 'sample' or 'full' mode.

    Parameters
    ----------
    mode : str
        'sample' for paysim_sample.parquet, 'full' for paysim.parquet.
    data_dir : Path or str, optional
        Processed data directory. If None, resolves to project_root / data / processed.

    Returns
    -------
    pd.DataFrame
        Loaded dataset with compact dtypes.
    """
    root = get_project_root()
    if data_dir is None:
        processed_dir = root / "data" / "processed"
    else:
        processed_dir = Path(data_dir)

    if mode == "full":
        file_path = processed_dir / "paysim.parquet"
    elif mode == "sample":
        file_path = processed_dir / "paysim_sample.parquet"
    else:
        raise ValueError(f"Invalid mode: '{mode}'. Must be 'sample' or 'full'.")

    if not file_path.is_file():
        raise FileNotFoundError(
            f"Processed data file not found: {file_path}. "
            f"Please run src/data_prep.py first."
        )

    df = pd.read_parquet(file_path)
    return df


def split_temporal(
    df: pd.DataFrame,
    split_cfg: dict[str, int] | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split dataframe strictly temporally by 'step'.

    Partitions:
    - Train: step 1 to 520
    - Val:   step 521 to 631
    - Test:  step 632 to 743

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing a 'step' column.
    split_cfg : dict, optional
        Dict containing train_end, val_end, test_end. If None, loaded from config.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        (train_df, val_df, test_df)
    """
    if "step" not in df.columns:
        raise KeyError("Dataframe must contain 'step' column for temporal splitting.")

    if split_cfg is None:
        cfg = load_config()
        split_cfg = cfg["split"]

    train_end = int(split_cfg.get("train_end", 520))
    val_end = int(split_cfg.get("val_end", 631))
    test_end = int(split_cfg.get("test_end", 743))

    train_df = df[df["step"] <= train_end].copy()
    val_df = df[(df["step"] > train_end) & (df["step"] <= val_end)].copy()
    test_df = df[(df["step"] > val_end) & (df["step"] <= test_end)].copy()

    # Rule 1 integrity assertions
    if len(train_df) > 0 and len(val_df) > 0:
        assert train_df["step"].max() < val_df["step"].min(), (
            "Temporal leak: Train max step >= Val min step"
        )
    if len(val_df) > 0 and len(test_df) > 0:
        assert val_df["step"].max() < test_df["step"].min(), (
            "Temporal leak: Val max step >= Test min step"
        )

    return train_df, val_df, test_df


if __name__ == "__main__":
    cfg = load_config()
    print("Project Root:", get_project_root())
    print("Loaded Config Seeds:", cfg.get("seeds"))
    print("Loaded Split Bounds:", cfg.get("split"))

"""Data preparation, cleaning, compact encoding, and persistence for PaySim dataset.

Transforms raw PaySim CSV into compact Parquet format:
- Chunked reading with usecols and compact dtypes (step int16, amounts float32, type category).
- Account names mapped to persistent int32 account_id (no string objects retained).
- Merchant flag is_merchant_dest (1 if nameDest starts with 'M', 0 otherwise).
- isFlaggedFraud dropped.
- newbalance columns stored exclusively in separate audit parquet.
- Outputs:
    - data/processed/paysim.parquet (full dataset)
    - data/processed/paysim_sample.parquet (all fraud + 10% legit, seed 42, chronological order)
    - data/processed/paysim_balance_audit.parquet (step, newbalanceOrig, newbalanceDest)
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Any

# Ensure project root is in sys.path
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.common import get_project_root, load_config, setup_logger

TRANSACTION_CATEGORIES = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]

CSV_USECOLS = [
    "step",
    "type",
    "amount",
    "nameOrig",
    "oldbalanceOrg",
    "newbalanceOrig",
    "nameDest",
    "oldbalanceDest",
    "newbalanceDest",
    "isFraud",
]

CSV_DTYPES = {
    "step": "int16",
    "type": "category",
    "amount": "float32",
    "nameOrig": "string",
    "oldbalanceOrg": "float32",
    "newbalanceOrig": "float32",
    "nameDest": "string",
    "oldbalanceDest": "float32",
    "newbalanceDest": "float32",
    "isFraud": "int8",
}


def process_paysim(
    raw_csv_path: Path,
    output_dir: Path,
    chunksize: int = 500_000,
    mode: str = "full",
    logger: Any = None,
) -> dict[str, Any]:
    """Process PaySim CSV in memory-efficient chunks."""
    start_time = time.time()
    if logger is None:
        logger = setup_logger("data_prep", "data_prep.log")

    output_dir.mkdir(parents=True, exist_ok=True)
    full_parquet_path = output_dir / "paysim.parquet"
    sample_parquet_path = output_dir / "paysim_sample.parquet"
    audit_parquet_path = output_dir / "paysim_balance_audit.parquet"

    logger.info("Starting PaySim preprocessing in mode='%s'.", mode)
    logger.info("Input CSV: %s", raw_csv_path)
    logger.info("Chunk size: %d", chunksize)

    if not raw_csv_path.is_file():
        raise FileNotFoundError(f"Raw PaySim CSV not found: {raw_csv_path}")

    # Shared mapping for account names across all chunks
    account_map: dict[str, int] = {}
    next_account_id = 0

    main_writer: pq.ParquetWriter | None = None
    audit_writer: pq.ParquetWriter | None = None

    total_rows = 0
    total_fraud = 0

    chunk_reader = pd.read_csv(
        raw_csv_path,
        usecols=CSV_USECOLS,
        dtype=CSV_DTYPES,
        chunksize=chunksize,
    )

    type_dtype = pd.CategoricalDtype(categories=TRANSACTION_CATEGORIES, ordered=False)

    for chunk_idx, chunk in enumerate(chunk_reader, start=1):
        chunk_rows = len(chunk)
        total_rows += chunk_rows
        total_fraud += int(chunk["isFraud"].sum())

        logger.info(
            "Processing chunk %d (%d rows, cumulative %d rows)...",
            chunk_idx,
            chunk_rows,
            total_rows,
        )

        # 1. Separate audit balance columns
        audit_chunk = pd.DataFrame(
            {
                "step": chunk["step"].astype(np.int16),
                "newbalanceOrig": chunk["newbalanceOrig"].astype(np.float32),
                "newbalanceDest": chunk["newbalanceDest"].astype(np.float32),
            }
        )
        audit_table = pa.Table.from_pandas(audit_chunk, preserve_index=False)
        if audit_writer is None:
            audit_writer = pq.ParquetWriter(
                audit_parquet_path,
                audit_table.schema,
                compression="snappy",
            )
        audit_writer.write_table(audit_table)

        # 2. Extract merchant flag (destination starts with 'M')
        is_merchant = chunk["nameDest"].str.startswith("M").astype(np.int8)

        # 3. Update persistent account mapping
        combined_names = pd.concat([chunk["nameOrig"], chunk["nameDest"]], ignore_index=True)
        unique_chunk_names = combined_names.unique()

        for name in unique_chunk_names:
            if name not in account_map:
                account_map[name] = next_account_id
                next_account_id += 1

        orig_ids = chunk["nameOrig"].map(account_map).astype(np.int32)
        dest_ids = chunk["nameDest"].map(account_map).astype(np.int32)

        # 4. Construct compact main chunk dataframe
        processed_chunk = pd.DataFrame(
            {
                "step": chunk["step"].astype(np.int16),
                "type": chunk["type"].astype(type_dtype),
                "amount": chunk["amount"].astype(np.float32),
                "nameOrig": orig_ids,
                "oldbalanceOrg": chunk["oldbalanceOrg"].astype(np.float32),
                "nameDest": dest_ids,
                "oldbalanceDest": chunk["oldbalanceDest"].astype(np.float32),
                "is_merchant_dest": is_merchant,
                "isFraud": chunk["isFraud"].astype(np.int8),
            }
        )

        main_table = pa.Table.from_pandas(processed_chunk, preserve_index=False)
        if main_writer is None:
            main_writer = pq.ParquetWriter(
                full_parquet_path,
                main_table.schema,
                compression="snappy",
            )
        main_writer.write_table(main_table)

        # In sample mode, process only 2 chunks (~1,000,000 rows) as a fast test
        if mode == "sample" and chunk_idx >= 2:
            logger.info("Sample mode: stopping early after %d chunks.", chunk_idx)
            break

    if main_writer is not None:
        main_writer.close()
    if audit_writer is not None:
        audit_writer.close()

    logger.info(
        "Finished streaming PaySim. Total rows written: %d. Unique accounts: %d.",
        total_rows,
        len(account_map),
    )

    # 5. Create paysim_sample.parquet: ALL fraud rows + 10% of legit rows (seed 42, chronological)
    logger.info("Generating sample dataset (all fraud + 10%% legit, seed 42)...")
    full_df = pd.read_parquet(full_parquet_path)

    fraud_mask = (full_df["isFraud"] == 1).to_numpy()
    fraud_indices = np.where(fraud_mask)[0]
    legit_indices = np.where(~fraud_mask)[0]

    rng = np.random.default_rng(42)
    n_sample_legit = int(round(len(legit_indices) * 0.10))
    sampled_legit = rng.choice(legit_indices, size=n_sample_legit, replace=False)

    sample_indices = np.sort(np.concatenate([fraud_indices, sampled_legit]))
    sample_df = full_df.iloc[sample_indices].copy()
    sample_df.to_parquet(sample_parquet_path, index=False, compression="snappy")

    elapsed = time.time() - start_time
    logger.info(
        "Saved %s (rows=%d, fraud=%d, legit=%d).",
        sample_parquet_path,
        len(sample_df),
        len(fraud_indices),
        len(sampled_legit),
    )
    logger.info("Data preparation completed in %.2f seconds.", elapsed)

    stats = {
        "total_rows": total_rows,
        "total_fraud": total_fraud,
        "unique_accounts": len(account_map),
        "sample_rows": len(sample_df),
        "sample_fraud": len(fraud_indices),
        "sample_legit": len(sampled_legit),
        "runtime_seconds": elapsed,
    }
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare PaySim dataset.")
    parser.add_argument(
        "--mode",
        choices=["sample", "full"],
        default="sample",
        help="Processing mode: 'sample' (subset for rapid testing) or 'full' (all 6.36M rows).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = get_project_root()
    cfg = load_config()

    raw_path = root / cfg["paths"]["raw_paysim"]
    processed_dir = root / cfg["paths"]["processed_dir"]

    logger = setup_logger("data_prep", "data_prep.log")
    try:
        stats = process_paysim(
            raw_csv_path=raw_path,
            output_dir=processed_dir,
            mode=args.mode,
            logger=logger,
        )
        logger.info("Summary stats: %s", stats)
    except MemoryError:
        logger.exception("MemoryError encountered during data_prep! Stopping per Rule 11.")
        sys.exit(1)
    except Exception:
        logger.exception("Unexpected error in data_prep.")
        raise


if __name__ == "__main__":
    main()

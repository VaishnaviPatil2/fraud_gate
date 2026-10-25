"""Data loading, cleaning, validation, and strict temporal partitioning.

Applies temporal splits strictly by step (train 1-520, val 521-631, test 632-743),
enforcing non-leaking preprocessing, category encodings, and parquet persistence.
"""

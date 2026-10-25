"""Feature engineering module for tabular transaction data.

Computes temporal, velocity, and statistical aggregation features while strictly
excluding post-transaction balance fields (newbalanceOrig, newbalanceDest) and ground-truth labels.
"""

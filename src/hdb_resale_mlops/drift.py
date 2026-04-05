"""
Data drift detection using PSI (categorical) and KS (numeric).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


DEFAULT_PSI_THRESHOLD = 0.2
DEFAULT_KS_ALPHA = 0.05
_PSI_EPSILON = 1e-4


@dataclass(frozen=True)
class ColumnDriftResult:
    column: str
    drift_type: str  # "psi" or "ks"
    statistic: float
    threshold: float
    p_value: float | None = None
    is_drifted: bool = False


@dataclass(frozen=True)
class DriftReport:
    column_results: list[ColumnDriftResult] = field(default_factory=list)
    overall_drift_detected: bool = False


def _compute_psi(reference: np.ndarray, current: np.ndarray) -> float:
    """Population Stability Index for a single categorical column."""
    categories = np.union1d(np.unique(reference), np.unique(current))
    ref_counts = {cat: 0 for cat in categories}
    cur_counts = {cat: 0 for cat in categories}
    for val in reference:
        ref_counts[val] = ref_counts.get(val, 0) + 1
    for val in current:
        cur_counts[val] = cur_counts.get(val, 0) + 1

    ref_total = len(reference)
    cur_total = len(current)
    if ref_total == 0 or cur_total == 0:
        return 0.0

    psi = 0.0
    for cat in categories:
        ref_pct = max(ref_counts[cat] / ref_total, _PSI_EPSILON)
        cur_pct = max(cur_counts[cat] / cur_total, _PSI_EPSILON)
        psi += (cur_pct - ref_pct) * np.log(cur_pct / ref_pct)

    return float(psi)


def detect_categorical_drift(
    reference_df,
    current_df,
    columns: list[str],
    psi_threshold: float = DEFAULT_PSI_THRESHOLD,
) -> list[ColumnDriftResult]:
    results: list[ColumnDriftResult] = []
    for col in columns:
        if col not in reference_df.columns or col not in current_df.columns:
            continue
        ref_values = reference_df[col].dropna().values
        cur_values = current_df[col].dropna().values
        psi_value = _compute_psi(ref_values, cur_values)
        results.append(
            ColumnDriftResult(
                column=col,
                drift_type="psi",
                statistic=psi_value,
                threshold=psi_threshold,
                is_drifted=psi_value > psi_threshold,
            )
        )
    return results


def detect_numeric_drift(
    reference_df,
    current_df,
    columns: list[str],
    ks_alpha: float = DEFAULT_KS_ALPHA,
) -> list[ColumnDriftResult]:
    from scipy.stats import ks_2samp

    results: list[ColumnDriftResult] = []
    for col in columns:
        if col not in reference_df.columns or col not in current_df.columns:
            continue
        ref_values = reference_df[col].dropna().astype(float).values
        cur_values = current_df[col].dropna().astype(float).values
        if len(ref_values) == 0 or len(cur_values) == 0:
            continue
        stat, p_value = ks_2samp(ref_values, cur_values)
        results.append(
            ColumnDriftResult(
                column=col,
                drift_type="ks",
                statistic=float(stat),
                threshold=ks_alpha,
                p_value=float(p_value),
                is_drifted=p_value < ks_alpha,
            )
        )
    return results


def run_drift_checks(
    train_df,
    test_df,
    categorical_columns: list[str],
    numeric_columns: list[str],
    psi_threshold: float = DEFAULT_PSI_THRESHOLD,
    ks_alpha: float = DEFAULT_KS_ALPHA,
) -> DriftReport:
    cat_results = detect_categorical_drift(
        train_df, test_df, categorical_columns, psi_threshold
    )
    num_results = detect_numeric_drift(train_df, test_df, numeric_columns, ks_alpha)
    all_results = cat_results + num_results
    overall_drifted = any(r.is_drifted for r in all_results)
    return DriftReport(
        column_results=all_results, overall_drift_detected=overall_drifted
    )

"""
Evaluation utilities for overall metrics and segment-level breakdowns.

This module answers two related questions after training:

1. How good is the model overall?
   - RMSE
   - MAE
2. Where is the model strong or weak?
   - per-town metrics
   - per-flat-type metrics

Those segment tables are important because the promotion workflow does not only
look at average quality. It also checks whether the candidate regressed badly
for specific groups even if the overall metrics look acceptable.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable

from hdb_resale_mlops.features import TARGET_COLUMN, split_features_and_target


@dataclass(frozen=True)
class EvaluationResult:
    overall_metrics: dict[str, float]
    scored_frame: Any
    segment_metrics: dict[str, Any]


def regression_metrics(
    y_true: Iterable[float], y_pred: Iterable[float]
) -> dict[str, float]:
    actual = [float(value) for value in y_true]
    predicted = [float(value) for value in y_pred]
    if len(actual) != len(predicted):
        raise ValueError("Actual and predicted collections must be the same length.")
    if not actual:
        raise ValueError("Cannot compute regression metrics for an empty collection.")

    squared_errors = [
        (truth - prediction) ** 2 for truth, prediction in zip(actual, predicted)
    ]
    absolute_errors = [
        abs(truth - prediction) for truth, prediction in zip(actual, predicted)
    ]
    rmse = math.sqrt(sum(squared_errors) / len(squared_errors))
    mae = sum(absolute_errors) / len(absolute_errors)
    return {"rmse": rmse, "mae": mae}


def build_segment_metrics_frame(scored_frame, segment_column: str):
    import pandas as pd

    rows: list[dict[str, float | str | int]] = []
    # Compute the same regression metrics separately for each segment value so
    # policy can reason about regressions hidden by the overall average.
    for segment_value, group in scored_frame.groupby(segment_column):
        metrics = regression_metrics(
            group[TARGET_COLUMN].tolist(), group["prediction"].tolist()
        )
        rows.append(
            {
                "segment": segment_value,
                "count": int(len(group)),
                "rmse": metrics["rmse"],
                "mae": metrics["mae"],
                "mean_actual": float(group[TARGET_COLUMN].mean()),
                "mean_prediction": float(group["prediction"].mean()),
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "segment",
                "count",
                "rmse",
                "mae",
                "mean_actual",
                "mean_prediction",
            ]
        )
    return (
        pd.DataFrame(rows)
        .sort_values(["rmse", "segment"], ascending=[True, True])
        .reset_index(drop=True)
    )


def evaluate_model(
    model, prepared_frame, segment_columns: tuple[str, ...] = ("town", "flat_type")
) -> EvaluationResult:
    # `prepared_frame` is expected to already have gone through
    # `prepare_training_frame()`. This function focuses only on scoring.
    features, target = split_features_and_target(prepared_frame)
    predictions = model.predict(features)
    overall = regression_metrics(target.tolist(), predictions.tolist())

    scored_frame = prepared_frame.copy()
    scored_frame["prediction"] = predictions
    segment_tables = {
        column: build_segment_metrics_frame(scored_frame, column)
        for column in segment_columns
    }
    return EvaluationResult(
        overall_metrics=overall,
        scored_frame=scored_frame,
        segment_metrics=segment_tables,
    )

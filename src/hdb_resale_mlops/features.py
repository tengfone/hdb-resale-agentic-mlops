"""
Feature engineering and sklearn pipeline construction for the tabular model.

This file defines the actual prediction problem:
- which columns are used as inputs
- how raw HDB resale records are converted into model-ready features
- which estimator and hyperparameters are used

The end result is a sklearn `Pipeline` with:
1. a preprocessing step for categorical and numeric columns
2. an `XGBRegressor` that predicts `resale_price`

This is the core predictive model in the repo. The LLM-based explainer lives in
`explainer.py` and is separate from this training pipeline.
"""

from __future__ import annotations

import re
from typing import Any, Iterable

CATEGORICAL_FEATURES = ["town", "flat_type", "flat_model", "storey_range"]
NUMERIC_FEATURES = [
    "floor_area_sqm",
    "flat_age_years",
    "remaining_lease_years",
    "storey_midpoint",
]
FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES
TARGET_COLUMN = "resale_price"
DATE_COLUMN = "sale_month"


def parse_remaining_lease_years(value: Any) -> float | None:
    if value is None:
        return None

    text = str(value).strip().lower()
    if not text:
        return None

    years_match = re.search(r"(\d+)\s+year", text)
    months_match = re.search(r"(\d+)\s+month", text)
    if years_match or months_match:
        years = float(years_match.group(1)) if years_match else 0.0
        months = float(months_match.group(1)) if months_match else 0.0
        return years + months / 12.0

    try:
        return float(text)
    except ValueError:
        return None


def parse_storey_midpoint(value: Any) -> float | None:
    if value is None:
        return None

    text = str(value).strip().upper()
    if not text:
        return None

    bounds = re.findall(r"(\d+)", text)
    if len(bounds) >= 2:
        lower = float(bounds[0])
        upper = float(bounds[1])
        return (lower + upper) / 2.0

    if len(bounds) == 1:
        return float(bounds[0])

    return None


def build_model_hyperparameters(
    random_seed: int = 7, overrides: dict[str, Any] | None = None
) -> dict[str, Any]:
    params: dict[str, Any] = {
        "n_estimators": 400,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "min_child_weight": 1.0,
        "objective": "reg:squarederror",
        "random_state": random_seed,
        "n_jobs": -1,
    }
    if overrides:
        params.update(overrides)
    return params


def prepare_training_frame(raw_frame):
    import pandas as pd

    frame = raw_frame.copy()
    if "month" not in frame.columns:
        raise KeyError("Expected a 'month' column in the raw HDB resale dataset.")

    # Convert the raw government dataset into the normalized columns the model
    # expects. This is the "raw CSV -> training table" boundary.
    frame[DATE_COLUMN] = pd.to_datetime(frame["month"], format="%Y-%m")
    frame["floor_area_sqm"] = pd.to_numeric(frame["floor_area_sqm"], errors="coerce")
    frame["lease_commence_date"] = pd.to_numeric(
        frame["lease_commence_date"], errors="coerce"
    )
    frame[TARGET_COLUMN] = pd.to_numeric(frame[TARGET_COLUMN], errors="coerce")

    # These are engineered features derived from the raw HDB columns rather
    # than copied 1:1 from the source CSV.
    sale_year_fraction = frame[DATE_COLUMN].dt.year + (
        frame[DATE_COLUMN].dt.month.sub(1) / 12.0
    )
    frame["flat_age_years"] = sale_year_fraction - frame["lease_commence_date"]
    if "remaining_lease" in frame.columns:
        frame["remaining_lease_years"] = frame["remaining_lease"].map(
            parse_remaining_lease_years
        )
    else:
        frame["remaining_lease_years"] = None
    frame["remaining_lease_years"] = frame["remaining_lease_years"].fillna(
        99.0 - frame["flat_age_years"]
    )
    frame["storey_midpoint"] = frame["storey_range"].map(parse_storey_midpoint)

    keep_columns = [DATE_COLUMN, TARGET_COLUMN, *FEATURE_COLUMNS]
    prepared = (
        frame[keep_columns]
        .dropna(subset=[TARGET_COLUMN, *FEATURE_COLUMNS])
        .reset_index(drop=True)
    )
    return prepared


def split_features_and_target(prepared_frame):
    features = prepared_frame.loc[:, FEATURE_COLUMNS].copy()
    target = prepared_frame.loc[:, TARGET_COLUMN].copy()
    return features, target


def build_preprocessor():
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder

    encoder = OneHotEncoder(handle_unknown="ignore")
    # Categorical columns are one-hot encoded, while numeric columns pass
    # through unchanged into the XGBoost regressor.
    return ColumnTransformer(
        transformers=[
            ("categorical", encoder, CATEGORICAL_FEATURES),
            ("numeric", "passthrough", NUMERIC_FEATURES),
        ],
        remainder="drop",
    )


def build_training_pipeline(
    random_seed: int = 7, model_overrides: dict[str, Any] | None = None
):
    from sklearn.pipeline import Pipeline
    from xgboost import XGBRegressor

    model = XGBRegressor(
        **build_model_hyperparameters(
            random_seed=random_seed, overrides=model_overrides
        )
    )
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor()),
            ("model", model),
        ]
    )


def feature_schema() -> dict[str, Iterable[str]]:
    return {
        "categorical_features": CATEGORICAL_FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
        "date_column": DATE_COLUMN,
    }

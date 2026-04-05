"""
Train the HDB resale model locally (no SageMaker required).

This module provides the same fit-and-evaluate logic that
``training_entrypoint.py`` runs inside a SageMaker container, but
executed directly in the current Python process.  It is the shared
training core used by the open-source Colab notebook and can also be
called from any local environment.

Think of this file as the common "fit a model" function that both notebook
paths reuse:
- local/Colab calls it directly
- SageMaker script mode calls it through `training_entrypoint.py`
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib

from hdb_resale_mlops.evaluation import EvaluationResult, evaluate_model
from hdb_resale_mlops.features import (
    build_model_hyperparameters,
    build_training_pipeline,
    prepare_training_frame,
    split_features_and_target,
)


def train_locally(
    raw_train,
    raw_validation,
    *,
    random_seed: int = 7,
    model_overrides: dict[str, Any] | None = None,
) -> tuple[Any, EvaluationResult]:
    """Fit a model on *raw_train* and evaluate on *raw_validation*.

    Parameters
    ----------
    raw_train:
        Raw training DataFrame (as returned by ``chronological_split``).
    raw_validation:
        Raw validation DataFrame.
    random_seed:
        Reproducibility seed forwarded to the pipeline.
    model_overrides:
        Optional hyperparameter overrides.

    Returns
    -------
    tuple[pipeline, EvaluationResult]
        The fitted sklearn ``Pipeline`` and validation evaluation result.
    """
    # Convert raw split frames into the exact feature schema expected by the
    # training pipeline before fitting the estimator.
    prepared_train = prepare_training_frame(raw_train)
    prepared_validation = prepare_training_frame(raw_validation)

    X_train, y_train = split_features_and_target(prepared_train)

    pipeline = build_training_pipeline(
        random_seed=random_seed,
        model_overrides=model_overrides,
    )
    pipeline.fit(X_train, y_train)

    validation_evaluation = evaluate_model(pipeline, prepared_validation)
    return pipeline, validation_evaluation


def save_model(model, destination: Path) -> Path:
    """Persist a fitted model to *destination* using joblib."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, destination)
    return destination

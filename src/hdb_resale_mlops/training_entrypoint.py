"""
Training script used inside the SageMaker container.

This is intentionally thin. It reads the train/validation CSVs from the
standard SageMaker channel directories, forwards hyperparameters from CLI
arguments, and then reuses the shared `train_locally()` function.

That means there is not a separate "SageMaker training implementation" here.
Remote training and local training both funnel through the same package logic.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import joblib
import pandas as pd

from hdb_resale_mlops.evaluation import regression_metrics
from hdb_resale_mlops.features import (
    build_model_hyperparameters,
    build_training_pipeline,
    prepare_training_frame,
    split_features_and_target,
)
from hdb_resale_mlops.local_training import train_locally


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the HDB resale candidate model inside SageMaker."
    )
    parser.add_argument("--random-seed", type=int, default=7)
    parser.add_argument(
        "--n-estimators", "--n_estimators", dest="n_estimators", type=int, default=None
    )
    parser.add_argument(
        "--max-depth", "--max_depth", dest="max_depth", type=int, default=None
    )
    parser.add_argument(
        "--learning-rate",
        "--learning_rate",
        dest="learning_rate",
        type=float,
        default=None,
    )
    parser.add_argument("--subsample", type=float, default=None)
    parser.add_argument(
        "--colsample-bytree",
        "--colsample_bytree",
        dest="colsample_bytree",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--reg-lambda", "--reg_lambda", dest="reg_lambda", type=float, default=None
    )
    parser.add_argument(
        "--min-child-weight",
        "--min_child_weight",
        dest="min_child_weight",
        type=float,
        default=None,
    )
    parser.add_argument("--objective", type=str, default=None)
    parser.add_argument(
        "--random-state", "--random_state", dest="random_state", type=int, default=None
    )
    parser.add_argument("--n-jobs", "--n_jobs", dest="n_jobs", type=int, default=None)
    return parser.parse_args()


def _first_csv(channel_dir: str) -> Path:
    root = Path(channel_dir)
    csv_paths = sorted(root.rglob("*.csv"))
    if not csv_paths:
        raise FileNotFoundError(
            f"No CSV files found under SageMaker channel directory: {channel_dir}"
        )
    return csv_paths[0]


def _model_overrides(args: argparse.Namespace) -> dict[str, float | int]:
    overrides = {
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "learning_rate": args.learning_rate,
        "subsample": args.subsample,
        "colsample_bytree": args.colsample_bytree,
        "reg_lambda": args.reg_lambda,
        "min_child_weight": args.min_child_weight,
        "objective": args.objective,
        "random_state": args.random_state,
        "n_jobs": args.n_jobs,
    }
    return {key: value for key, value in overrides.items() if value is not None}


def train() -> dict[str, float]:
    args = parse_args()
    train_path = _first_csv(os.environ["SM_CHANNEL_TRAIN"])
    validation_path = _first_csv(os.environ["SM_CHANNEL_VALIDATION"])

    raw_train = pd.read_csv(train_path)
    raw_validation = pd.read_csv(validation_path)

    overrides = _model_overrides(args)
    # Reuse the same shared training core as the local notebook path.
    pipeline, validation_evaluation = train_locally(
        raw_train,
        raw_validation,
        random_seed=args.random_seed,
        model_overrides=overrides,
    )
    validation_metrics = validation_evaluation.overall_metrics
    resolved_hyperparameters = build_model_hyperparameters(
        random_seed=args.random_seed,
        overrides=overrides,
    )

    model_dir = Path(os.environ["SM_MODEL_DIR"])
    output_dir = Path(os.environ["SM_OUTPUT_DATA_DIR"])
    model_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(pipeline, model_dir / "model.joblib")
    (model_dir / "training_metrics.json").write_text(
        json.dumps(validation_metrics, indent=2), encoding="utf-8"
    )
    (output_dir / "training_metrics.json").write_text(
        json.dumps(validation_metrics, indent=2), encoding="utf-8"
    )
    (model_dir / "hyperparameters.json").write_text(
        json.dumps(resolved_hyperparameters, indent=2),
        encoding="utf-8",
    )
    (output_dir / "hyperparameters.json").write_text(
        json.dumps(resolved_hyperparameters, indent=2),
        encoding="utf-8",
    )
    return validation_metrics


def main() -> None:
    metrics = train()
    print(json.dumps({"validation_metrics": metrics}, indent=2))

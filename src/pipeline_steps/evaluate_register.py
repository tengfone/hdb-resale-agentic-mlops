from __future__ import annotations

import argparse
from pathlib import Path

from hdb_resale_mlops.sagemaker_pipeline import run_evaluate_register_step


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the trained model and register the candidate in MLflow."
    )
    parser.add_argument("--model-artifact-input", type=Path, required=True)
    parser.add_argument("--validation-input", type=Path, required=True)
    parser.add_argument("--test-input", type=Path, required=True)
    parser.add_argument("--metadata-input", type=Path, required=True)
    parser.add_argument("--registration-dir", type=Path, required=True)
    parser.add_argument("--evaluation-dir", type=Path, required=True)
    parser.add_argument("--training-job-name", type=str, default=None)
    parser.add_argument("--model-artifact-s3-uri", type=str, default=None)
    parser.add_argument("--random-seed", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run_evaluate_register_step(
        model_artifact_input=args.model_artifact_input,
        validation_input=args.validation_input,
        test_input=args.test_input,
        metadata_input_dir=args.metadata_input,
        registration_dir=args.registration_dir,
        evaluation_dir=args.evaluation_dir,
        training_job_name=args.training_job_name,
        model_artifact_s3_uri=args.model_artifact_s3_uri,
        random_seed=args.random_seed,
    )
    print(payload)


if __name__ == "__main__":
    main()

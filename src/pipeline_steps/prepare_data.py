from __future__ import annotations

import argparse
from pathlib import Path

from hdb_resale_mlops.sagemaker_pipeline import run_prepare_data_step


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare data.gov.sg data and chronological splits for the SageMaker pipeline."
    )
    parser.add_argument("--dataset-collection-id", type=str, default=None)
    parser.add_argument("--dataset-id", type=str, default=None)
    parser.add_argument("--dataset-name", type=str, default=None)
    parser.add_argument("--random-seed", type=int, default=None)
    parser.add_argument("--validation-months", type=int, default=None)
    parser.add_argument("--test-months", type=int, default=None)
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--validation-dir", type=Path, required=True)
    parser.add_argument("--test-dir", type=Path, required=True)
    parser.add_argument("--metadata-dir", type=Path, required=True)
    parser.add_argument("--force-snapshot-refresh", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = run_prepare_data_step(
        train_dir=args.train_dir,
        validation_dir=args.validation_dir,
        test_dir=args.test_dir,
        metadata_dir=args.metadata_dir,
        dataset_collection_id=args.dataset_collection_id,
        dataset_id=args.dataset_id,
        dataset_name=args.dataset_name,
        random_seed=args.random_seed,
        validation_months=args.validation_months,
        test_months=args.test_months,
        force_snapshot_refresh=args.force_snapshot_refresh,
    )
    print(metadata)


if __name__ == "__main__":
    main()

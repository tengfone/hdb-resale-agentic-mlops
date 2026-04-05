from __future__ import annotations

import argparse
from pathlib import Path

from hdb_resale_mlops.sagemaker_pipeline import run_policy_gate_step


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the deterministic policy gate for the SageMaker pipeline."
    )
    parser.add_argument("--train-input", type=Path, required=True)
    parser.add_argument("--test-input", type=Path, required=True)
    parser.add_argument("--registration-input", type=Path, required=True)
    parser.add_argument("--evaluation-input", type=Path, required=True)
    parser.add_argument("--policy-dir", type=Path, required=True)
    parser.add_argument("--handoff-dir", type=Path, required=True)
    parser.add_argument("--pipeline-execution-id", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run_policy_gate_step(
        train_input=args.train_input,
        test_input=args.test_input,
        registration_input=args.registration_input,
        evaluation_input=args.evaluation_input,
        policy_dir=args.policy_dir,
        handoff_dir=args.handoff_dir,
        pipeline_execution_id=args.pipeline_execution_id,
    )
    print(payload)


if __name__ == "__main__":
    main()
